"""S2V-DQN training for 0/1 Knapsack."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from GNNForKnapSack.instance_loader import load_instance, list_instances
from s2v_env import GraphKnapsackEnv
from s2v_model import S2VQNetwork, save_s2v_checkpoint
from s2v_replay import GraphReplayBuffer, GraphTransition
from rl_training_logger import RLTrainingLogger


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEFAULTS = {
    "gamma":               0.99,
    "lr":                  3e-4,
    "batch_size":          64,
    "buffer_size":         50_000,
    "min_buffer_size":     1_000,
    "target_update_steps": 1_000,
    "eps_start":           1.0,
    "eps_end":             0.05,
    "eps_decay_steps":     30_000,
    "grad_clip":           5.0,
}

LOG_EVERY = 1_000   # log every 1k steps
VAL_EVERY = 5_000   # validate every 5k steps


def mark(msg: str):
    print(f"[S2V-DQN] {msg}", flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epsilon_by_step(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return eps_end
    t = step / max(1, decay_steps)
    return eps_start + t * (eps_end - eps_start)


def pick_action_eps_greedy(
    q_values:   torch.Tensor,
    valid_mask: np.ndarray,
    eps:        float,
    rng:        np.random.Generator,
) -> int:
    """ε-greedy action selection over valid items."""
    valid_idx = np.where(valid_mask > 0.5)[0]
    if len(valid_idx) == 0:
        return -1

    if rng.random() < eps:
        return int(rng.choice(valid_idx))

    q = q_values.detach().cpu().numpy().copy()
    q[valid_mask < 0.5] = -1e9
    return int(np.argmax(q))


def load_instances_for_training(dataset_dir: Path) -> List[dict]:
    """Load all NPZ instances from directory."""
    files = list_instances(dataset_dir)
    instances = []
    for path in files:
        W, V, C = load_instance(path)
        instances.append({"weights": W, "values": V, "capacity": int(C),
                          "name": path.name})
    return instances


@torch.no_grad()
def evaluate_on_set(
    model: S2VQNetwork,
    instances: List[dict],
    device: torch.device,
    k: int = 16,
    limit: int = 50,
) -> dict:
    """Run greedy policy (eps=0) on instances, return avg metrics."""
    model.eval()
    values = []
    feasibles = 0
    for inst in instances[:limit]:
        env = GraphKnapsackEnv(inst["weights"], inst["values"],
                                inst["capacity"], k=k)
        s = env.reset()
        done = False
        while not done:
            mask = env.valid_actions_mask()
            if mask.sum() == 0:
                break
            s_dev = s.to(device)
            q = model(s_dev)
            a = pick_action_eps_greedy(q, mask, 0.0, np.random.default_rng(0))
            if a < 0:
                break
            out = env.step(a)
            s = out.next_state
            done = out.done
        values.append(env.compute_solution_value())
        if env.compute_solution_weight() <= inst["capacity"] + 1e-6:
            feasibles += 1
    model.train()
    return {
        "avg_value":    float(np.mean(values)) if values else 0.0,
        "feasibility":  feasibles / max(len(values), 1),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train S2V-DQN (GNN-based DQN) on 0/1 Knapsack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=Path, required=True,
                        help="Training data directory")
    parser.add_argument("--val_dir",     type=Path, default=None)
    parser.add_argument("--test_dir",    type=Path, default=None)
    parser.add_argument("--out_dir",     type=Path, default=Path("results/S2V_DQN"))
    parser.add_argument("--device",      type=str,  default="cpu")
    parser.add_argument("--hidden_dim",  type=int,  default=128)
    parser.add_argument("--num_layers",  type=int,  default=3)
    parser.add_argument("--k",           type=int,  default=16,
                        help="kNN graph neighborhood size")
    parser.add_argument("--train_steps", type=int,  default=50_000)
    parser.add_argument("--lr",          type=float, default=DEFAULTS["lr"])
    parser.add_argument("--batch_size",  type=int,  default=DEFAULTS["batch_size"])
    parser.add_argument("--seed",        type=int,  default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load data
    mark(f"Loading train: {args.dataset_dir}")
    train_set = load_instances_for_training(args.dataset_dir)

    val_set, test_set = [], []
    if args.val_dir:
        val_set = load_instances_for_training(args.val_dir)
    if args.test_dir:
        test_set = load_instances_for_training(args.test_dir)

    mark(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    # Init networks
    online = S2VQNetwork(
        in_dim=7,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.1,
    ).to(device)

    target = S2VQNetwork(
        in_dim=7,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.1,
    ).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)
    buffer    = GraphReplayBuffer(DEFAULTS["buffer_size"], seed=args.seed)

    n_params = sum(p.numel() for p in online.parameters())
    mark(f"S2VQNetwork: hidden={args.hidden_dim} layers={args.num_layers} "
         f"params={n_params:,}")

    # Training logger
    logger = RLTrainingLogger(args.out_dir / "training_log.csv")
    mark(f"Logger → {args.out_dir / 'training_log.csv'}")

    # Training loop
    step_count = 0
    updates = 0
    best_val = -float("inf")
    last_loss = None
    latest_val_value = None
    start_time = time.perf_counter()

    while step_count < args.train_steps:
        inst = train_set[int(rng.integers(0, len(train_set)))]
        env = GraphKnapsackEnv(inst["weights"], inst["values"], inst["capacity"],
                                k=args.k)
        s = env.reset()
        done = False

        while not done and step_count < args.train_steps:
            mask = env.valid_actions_mask()
            if mask.sum() == 0:
                break

            eps = epsilon_by_step(step_count, DEFAULTS["eps_start"],
                                   DEFAULTS["eps_end"], DEFAULTS["eps_decay_steps"])

            with torch.no_grad():
                s_dev = s.to(device)
                q = online(s_dev)

            action = pick_action_eps_greedy(q, mask, eps, rng)
            if action < 0:
                break

            out = env.step(action)
            s2 = out.next_state
            mask2 = env.valid_actions_mask()

            buffer.push(GraphTransition(
                s=s, a=int(action), r=float(out.reward),
                s2=s2, done=bool(out.done), valid_mask=mask2,
            ))

            s = s2
            done = out.done
            step_count += 1

            # Optimize
            if len(buffer) >= DEFAULTS["min_buffer_size"]:
                s_b, a_b, r_b, s2_b, d_b, mask_list = buffer.sample(args.batch_size)

                s_b  = s_b.to(device)
                s2_b = s2_b.to(device)
                a_b  = a_b.to(device)
                r_b  = r_b.to(device)
                d_b  = d_b.to(device)

                q_all = online(s_b)
                ptr = s_b.ptr
                global_a = ptr[:-1] + a_b
                q_sa = q_all[global_a]

                with torch.no_grad():
                    q2_all = target(s2_b)
                    ptr2 = s2_b.ptr
                    max_q2 = torch.zeros(args.batch_size, device=device)
                    for i in range(args.batch_size):
                        start = int(ptr2[i].item())
                        end   = int(ptr2[i+1].item())
                        q_graph = q2_all[start:end]
                        m = torch.from_numpy(mask_list[i]).float().to(device)
                        if m.sum() > 0:
                            q_masked = q_graph + (m - 1.0) * 1e9
                            max_q2[i] = q_masked.max()
                        else:
                            max_q2[i] = 0.0

                    y = r_b + (1.0 - d_b) * DEFAULTS["gamma"] * max_q2

                loss = F.smooth_l1_loss(q_sa, y)
                last_loss = float(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), DEFAULTS["grad_clip"])
                optimizer.step()
                updates += 1

                if step_count % DEFAULTS["target_update_steps"] == 0:
                    target.load_state_dict(online.state_dict())

            # Validation
            if step_count % VAL_EVERY == 0 and val_set:
                val_metrics = evaluate_on_set(online, val_set, device, k=args.k, limit=50)
                latest_val_value = val_metrics["avg_value"]
                mark(f"  [val] step={step_count} avg_value={latest_val_value:.2f} "
                     f"feas={val_metrics['feasibility']:.3f}")
                if latest_val_value > best_val:
                    best_val = latest_val_value
                    save_s2v_checkpoint(online, args.out_dir / "s2v_dqn_best.pt")
                    mark(f"  [val] new best → s2v_dqn_best.pt")

            # Periodic logging to CSV
            if step_count % LOG_EVERY == 0 or step_count == 1:
                elapsed = time.perf_counter() - start_time
                loss_str = f"{last_loss:.4f}" if last_loss else "N/A"
                mark(f"step={step_count:>6} eps={eps:.3f} buf={len(buffer):>5} "
                     f"loss={loss_str} elapsed={elapsed:.1f}s")

                logger.log(
                    step=step_count,
                    updates=updates,
                    epsilon=eps,
                    loss=last_loss,
                    avg_value_val=latest_val_value,
                    best_value_val=best_val if best_val > -float("inf") else None,
                    buffer_size=len(buffer),
                    elapsed_sec=elapsed,
                )

    train_time = time.perf_counter() - start_time

    # Save final
    save_s2v_checkpoint(online, args.out_dir / "s2v_dqn.pt")

    meta = {
        "train_time_sec":  round(train_time, 2),
        "train_steps":     args.train_steps,
        "updates":         updates,
        "n_train":         len(train_set),
        "n_val":           len(val_set),
        "n_test":          len(test_set),
        "hidden_dim":      args.hidden_dim,
        "num_layers":      args.num_layers,
        "best_val_value":  best_val if best_val > -float("inf") else None,
        "params":          n_params,
    }
    with (args.out_dir / "train_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    mark(f"Training complete: {train_time:.1f}s, {updates} updates")
    mark(f"Best val avg_value: {best_val:.2f}" if best_val > -float("inf") else "No validation performed")
    mark(f"Model → {args.out_dir / 's2v_dqn.pt'}")
    mark(f"Log   → {args.out_dir / 'training_log.csv'}")


if __name__ == "__main__":
    main()