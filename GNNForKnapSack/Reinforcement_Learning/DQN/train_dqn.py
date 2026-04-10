"""Huấn luyện DQN cho bài toán 0/1 Knapsack."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dqn_config import DQNConfig
from dqn_env import KnapsackEnv
from dqn_model import QNetwork
from dqn_replay import ReplayBuffer, Transition
from GNNForKnapSack.instance_loader import load_instance, list_instances
from GNNForKnapSack.rl_training_logger import RLTrainingLogger

CHECKPOINT_EVERY = 10_000
LOG_EVERY        = 1_000    # log every 1k steps
VAL_EVERY        = 5_000    # validate every 5k steps


def mark(msg: str):
    print(f"[DQN-TRAIN] {msg}", flush=True)

def set_seed(seed: int):
    """Thiết lập seed để kết quả chạy lại được (reproducible)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epsilon_by_step(step: int, cfg: DQNConfig) -> float:
    """Tính epsilon giảm dần theo thời gian (annealing)."""
    if step >= cfg.eps_decay_steps:
        return cfg.eps_end
    t = step / cfg.eps_decay_steps
    return cfg.eps_start + t * (cfg.eps_end - cfg.eps_start)


def pick_action(q_values: np.ndarray, valid_mask: np.ndarray,
                eps: float, rng: np.random.Generator) -> int:
    """Chọn action theo epsilon-greedy + respect valid mask."""
    if rng.random() < eps:
        valid_actions = np.where(valid_mask > 0.5)[0]
        return int(rng.choice(valid_actions))
    q = q_values.copy()
    q[valid_mask < 0.5] = -1e9
    return int(np.argmax(q))


def load_instances_for_training(dataset_dir: Path):
    """Load all NPZ instances from directory for DQN training."""
    files = list_instances(dataset_dir)
    instances = []
    for path in files:
        W, V, C = load_instance(path)
        W = np.asarray(W, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        order = np.argsort(-(V / (W + 1e-8)))
        W, V = W[order], V[order]
        instances.append({
            "weights": W,
            "values": V,
            "capacity": int(C),
            "name": path.name,
        })
    return instances

def split_instances(instances: list, seed: int, train_ratio: float, val_ratio: float):
    """Deterministic shuffle + split."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(instances))
    rng.shuffle(idx)

    n = len(instances)
    n_train = max(1, int(round(n * train_ratio)))
    n_val   = max(1, int(round(n * val_ratio)))
    n_train = min(n_train, n - 2) if n >= 3 else n_train

    train = [instances[i] for i in idx[:n_train]]
    val   = [instances[i] for i in idx[n_train:n_train + n_val]]
    test  = [instances[i] for i in idx[n_train + n_val:]]
    return train, val, test


@torch.no_grad()
def evaluate_dqn_on_set(
    model: QNetwork,
    instances: list,
    device: torch.device,
    cfg: DQNConfig,
    limit: int = 50,
) -> dict:
    """Greedy evaluation on a subset — avg value + feasibility."""
    model.eval()
    values = []
    feasibles = 0
    for inst in instances[:limit]:
        env = KnapsackEnv(inst["weights"], inst["values"],
                          inst["capacity"], eps=cfg.eps)
        s = env.reset()
        done = False
        while not done:
            mask = env.valid_actions_mask()
            q = model(torch.from_numpy(s).unsqueeze(0).to(device)).cpu().numpy()[0]
            q[mask < 0.5] = -1e9
            a = int(np.argmax(q))
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
        "n":            len(values),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Huấn luyện DQN cho bài toán 0/1 Knapsack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=Path, required=True,
                        help="Thư mục chứa dữ liệu huấn luyện (.npz)")
    parser.add_argument("--val_dir", type=Path, default=None,
                        help="Separate validation directory")
    parser.add_argument("--test_dir", type=Path, default=None,
                        help="Separate test directory")
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parents[2] / "results" / "DQN")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--train_steps", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = DQNConfig()

    cfg.train_steps = args.train_steps
    cfg.lr = args.lr
    cfg.seed = args.seed
    cfg.eps_decay_steps = int(cfg.train_steps * 0.6)

    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    mark(f"Loading instances from {args.dataset_dir}")
    all_train = load_instances_for_training(args.dataset_dir)

    if args.val_dir and args.test_dir:
        # 3 separate directories
        train_set = all_train
        val_set   = load_instances_for_training(args.val_dir)
        test_set  = load_instances_for_training(args.test_dir)
        mark(f"Dataset: 3 SEPARATE dirs")
        mark(f"  Train: {args.dataset_dir} ({len(train_set)} instances)")
        mark(f"  Val:   {args.val_dir} ({len(val_set)} instances)")
        mark(f"  Test:  {args.test_dir} ({len(test_set)} instances)")
    elif args.test_dir:
        # Train+Val from dataset_dir, Test from test_dir
        all_instances = all_train
        n_tv = len(all_instances)
        n_train = max(1, int(n_tv * 0.9))
        train_set = all_instances[:n_train]
        val_set   = all_instances[n_train:]
        test_set  = load_instances_for_training(args.test_dir)
        mark(f"  Train+Val: {args.dataset_dir} ({n_tv} → train={len(train_set)}, val={len(val_set)})")
        mark(f"  Test: {args.test_dir} ({len(test_set)} instances)")
    else:
        # Single directory, auto-split
        train_set, val_set, test_set = split_instances(
            all_train, cfg.seed, cfg.train_ratio, cfg.val_ratio
        )
        mark(f"Loaded {len(all_train)} → train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    # Init env to get state_dim
    sample_env = KnapsackEnv(train_set[0]["weights"],
                       train_set[0]["values"],
                       train_set[0]["capacity"], eps=cfg.eps)
    state_dim = sample_env.reset().shape[0]
    mark(f"State dimension: {state_dim}")

    # Init networks
    device = torch.device(args.device)

    online_net = QNetwork(state_dim, hidden_dim=args.hidden_dim).to(device)
    target_net = QNetwork(state_dim, hidden_dim=args.hidden_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=cfg.lr)
    replay_buffer = ReplayBuffer(cfg.buffer_size, seed=cfg.seed)

    params = sum(p.numel() for p in online_net.parameters())
    mark(f"QNetwork: state_dim={state_dim} hidden={args.hidden_dim} params={params:,}")
    mark(f"Training: {cfg.train_steps} steps, lr={cfg.lr}")

    # Training logger
    logger = RLTrainingLogger(args.out_dir / "training_log.csv")

    mark(f"Bắt đầu huấn luyện DQN với {cfg.train_steps:,} steps trên device: {device}")

    # Training loop
    step = 0
    updates = 0
    best_val = -float("inf")
    last_loss = None
    latest_val_value = None
    start_time = time.perf_counter()

    while step < cfg.train_steps:
        instance = random.choice(train_set)
        env = KnapsackEnv(instance["weights"], instance["values"],
                          instance["capacity"], eps=cfg.eps)
        state = env.reset()
        done = False

        while not done and step < cfg.train_steps:
            valid_mask = env.valid_actions_mask()
            eps = epsilon_by_step(step, cfg)

            with torch.no_grad():
                q_values = online_net(torch.from_numpy(state).unsqueeze(0).to(device))
                q_values = q_values.cpu().numpy()[0]

            action = pick_action(q_values, valid_mask, eps, rng)

            step_output = env.step(action)
            next_state = step_output.next_state
            reward = step_output.reward
            done = step_output.done
            next_mask = env.valid_actions_mask() if not done else np.array([1, 0], dtype=np.int64)

            replay_buffer.push(Transition(s=state, a=int(action), r=float(reward),
                                   s2=next_state, done=bool(done), mask2=next_mask))
            state = next_state
            step += 1

            # ====================== TRAINING STEP ======================
            if len(replay_buffer) >= cfg.min_buffer_size:
                s_batch, a_batch, r_batch, s2_batch, done_batch, mask2_b = replay_buffer.sample(cfg.batch_size)

                s_t  = torch.from_numpy(s_batch).to(device)
                a_t  = torch.from_numpy(a_batch).to(device)
                r_t  = torch.from_numpy(r_batch).to(device)
                s2_t = torch.from_numpy(s2_batch).to(device)
                done_t  = torch.from_numpy(done_batch).to(device)
                mask2_t = torch.from_numpy(mask2_b).to(device)

                current_q = online_net(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_net(s2_t)
                    next_q = next_q + (mask2_t - 1.0) * 1e9
                    max_next_q = next_q.max(dim=1).values
                    target_q = r_t + (1.0 - done_t) * cfg.gamma * max_next_q

                # Tính loss
                loss = F.smooth_l1_loss(current_q, target_q)
                last_loss = float(loss.item())

                #Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                updates += 1

                if step % cfg.target_update_steps == 0:
                    target_net.load_state_dict(online_net.state_dict())

            # Validation + save best
            if step % VAL_EVERY == 0 and val_set:
                val_metrics = evaluate_dqn_on_set(online_net, val_set, device, cfg, limit=50)
                latest_val_value = val_metrics["avg_value"]
                mark(f"  [val] step={step} avg_value={latest_val_value:.2f} "
                     f"feas={val_metrics['feasibility']:.3f}")
                if latest_val_value > best_val:
                    best_val = latest_val_value
                    torch.save({
                        "state_dim": state_dim,
                        "hidden_dim": args.hidden_dim,
                        "model_state": online_net.state_dict(),
                        "config": cfg.__dict__,
                    }, args.out_dir / "dqn_best.pt")
                    mark(f"  [val] new best → dqn_best.pt")

            # ====================== VALIDATION & LOGGING ======================
            if step % LOG_EVERY == 0 or step == 1:
                elapsed = time.perf_counter() - start_time
                loss_str = f"{last_loss:.4f}" if last_loss is not None else "N/A"
                mark(f"Step={step:6d} | Eps={eps:.3f} "
                     f"Buffer={len(replay_buffer):6d} | Loss={loss_str} "
                     f"Updates {updates:5d} | Elapsed={elapsed/60:.1f} m")

                logger.log(
                    step=step,
                    updates=updates,
                    epsilon=eps,
                    loss=last_loss,
                    avg_value_val=latest_val_value,
                    best_value_val=best_val if best_val > -float("inf") else None,
                    buffer_size=len(replay_buffer),
                    elapsed_sec=elapsed,
                )
    # ====================== KẾT THÚC HUẤN LUYỆN ======================
    total_time = time.perf_counter() - start_time
    mark(f"HOÀN TẤT HUẤN LUYỆN! Thời gian: {total_time / 60:.1f} phút")

    # Lưu model cuối cùng
    final_path = args.out_dir / "dqn.pt"
    torch.save({
        "state_dim": state_dim,
        "hidden_dim": args.hidden_dim,
        "model_state": online_net.state_dict(),
        "config": cfg.__dict__,
        "train_steps": step,
    }, final_path)

    # Save metadata
    meta_path = args.out_dir / "train_meta.json"
    with meta_path.open("w") as f:
        json.dump({
            "train_time_sec": round(total_time, 2),
            "train_steps": cfg.train_steps,
            "updates": updates,
            "seed": cfg.seed,
            "dataset_dir": str(args.dataset_dir),
            "n_train": len(train_set),
            "n_val": len(val_set),
            "n_test": len(test_set),
            "state_dim": state_dim,
            "hidden_dim": args.hidden_dim,
            "best_val_value": best_val if best_val > -float("inf") else None,
        }, f, indent=2)

    mark(f"Model đã được lưu tại: {final_path}")

if __name__ == "__main__":
    main()