"""GNN + REINFORCE training for 0/1 Knapsack."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

_HERE     = Path(__file__).resolve().parent
_GNN_ROOT = _HERE.parent
for _p in [str(_GNN_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.dataset import GeneratedKnapsack01Dataset
from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.model import KnapsackGNN, save_checkpoint, load_checkpoint


# ---------------------------------------------------------------------------
# Sampling and repair
# ---------------------------------------------------------------------------

def sample_and_repair(
    logits:   torch.Tensor,    # [n] pre-sigmoid logits for one instance
    weights:  torch.Tensor,    # [n]
    values:   torch.Tensor,    # [n]
    capacity: float,
    greedy:   bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Sample an action (binary selection), then repair to feasibility."""
    probs = torch.sigmoid(logits)

    if greedy:
        sampled = (probs > 0.5).float()
    else:
        dist = torch.distributions.Bernoulli(probs=probs)
        sampled = dist.sample()

    # Repair to feasibility: drop lowest value/weight ratio items until fit
    # We compute log_prob AFTER repair so that (log_prob, reward) corresponds
    # to the SAME action — this is required for REINFORCE to be correct.
    selected = sampled.detach().cpu().clone()
    w_cpu = weights.detach().cpu()
    v_cpu = values.detach().cpu()
    total_w = float((selected * w_cpu).sum())

    if total_w > capacity + 1e-6:
        sel_idx = torch.where(selected > 0.5)[0]
        if len(sel_idx) > 0:
            ratios = v_cpu[sel_idx] / (w_cpu[sel_idx] + 1e-8)
            order  = sel_idx[torch.argsort(ratios)]  # ascending: worst first
            for i in order:
                if total_w <= capacity + 1e-6:
                    break
                selected[i] = 0
                total_w -= float(w_cpu[i])

    action_final = selected.to(logits.device)

    eps = 1e-8
    log_p_per_node = (
        action_final * torch.log(probs + eps)
        + (1 - action_final) * torch.log(1 - probs + eps)
    )
    log_prob = log_p_per_node.sum()

    value = float((selected * v_cpu).sum())
    return selected, log_prob, value


# ---------------------------------------------------------------------------
# REINFORCE step over a batch
# ---------------------------------------------------------------------------

def reinforce_batch_loss(
    model:     KnapsackGNN,
    baseline:  KnapsackGNN,
    batch,
    device:    torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute REINFORCE loss for one batch."""
    batch = batch.to(device)

    logits_all = model(batch)  # [total_nodes]

    with torch.no_grad():
        base_logits_all = baseline(batch)

    batch_vec = batch.batch
    n_graphs = int(batch_vec.max().item()) + 1

    log_probs: List[torch.Tensor] = []
    rewards:   List[float]         = []
    baselines: List[float]         = []

    for g in range(n_graphs):
        mask  = batch_vec == g
        lg    = logits_all[mask]
        base_lg = base_logits_all[mask]
        w_g   = batch.wts[mask]
        v_g   = batch.vals[mask]
        cap_g = float(batch.cap[g].item() if batch.cap.dim() > 0 else batch.cap.item())

        # Sample from policy
        _, log_p, reward = sample_and_repair(lg, w_g, v_g, cap_g, greedy=False)

        # Baseline: greedy from baseline network
        _, _, base_reward = sample_and_repair(base_lg, w_g, v_g, cap_g, greedy=True)

        log_probs.append(log_p)
        rewards.append(reward)
        baselines.append(base_reward)

    rewards_t   = torch.tensor(rewards,   device=device, dtype=torch.float32)
    baselines_t = torch.tensor(baselines, device=device, dtype=torch.float32)
    advantages  = rewards_t - baselines_t

    # Normalize advantages (reduces variance)
    if advantages.numel() > 1:
        adv_std = advantages.std()
        if adv_std > 1e-6:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

    log_probs_t = torch.stack(log_probs)

    # REINFORCE loss: -E[advantage * log_prob]
    loss = -(advantages * log_probs_t).mean()

    stats = {
        "avg_reward":    float(rewards_t.mean().item()),
        "avg_baseline":  float(baselines_t.mean().item()),
        "raw_advantage": float((rewards_t - baselines_t).mean().item()),
    }
    return loss, stats


# ---------------------------------------------------------------------------
# Validation and baseline update
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_policy(
    model:  KnapsackGNN,
    loader: DataLoader,
    device: torch.device,
    greedy: bool = True,
) -> Dict[str, float]:
    """Run policy on loader, return avg value and avg ratio vs DP."""
    model.eval()
    values = []
    ratios = []
    feasibles = 0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        logits_all = model(batch)
        batch_vec = batch.batch
        n_graphs = int(batch_vec.max().item()) + 1

        for g in range(n_graphs):
            mask  = batch_vec == g
            lg    = logits_all[mask]
            w_g   = batch.wts[mask]
            v_g   = batch.vals[mask]
            cap_g = float(batch.cap[g].item() if batch.cap.dim() > 0 else batch.cap.item())

            action, _, value = sample_and_repair(lg, w_g, v_g, cap_g, greedy=greedy)
            total_w = float((action * w_g.cpu()).sum())

            values.append(value)
            feasibles += int(total_w <= cap_g + 1e-6)
            n += 1

            # Ratio vs DP (if label available)
            dp_sol = batch.y[mask].view(-1).cpu()
            dp_val = float((dp_sol * v_g.cpu()).sum())
            if dp_val > 0:
                ratios.append(value / dp_val)

    model.train()
    return {
        "avg_value":      float(np.mean(values)) if values else 0.0,
        "avg_ratio_vs_dp": float(np.mean(ratios)) if ratios else 0.0,
        "feasibility":    feasibles / max(n, 1),
        "n":              n,
    }


@torch.no_grad()
def paired_t_test_update(
    model:     KnapsackGNN,
    baseline:  KnapsackGNN,
    val_loader: DataLoader,
    device:    torch.device,
    alpha:     float = 0.05,
) -> bool:
    """One-sided paired t-test: is model significantly better than baseline?"""
    was_training = model.training
    model.eval()

    model_vals = []
    base_vals  = []

    for batch in val_loader:
        batch = batch.to(device)
        m_logits = model(batch)
        b_logits = baseline(batch)
        batch_vec = batch.batch
        n_graphs = int(batch_vec.max().item()) + 1

        for g in range(n_graphs):
            mask = batch_vec == g
            w_g  = batch.wts[mask]
            v_g  = batch.vals[mask]
            cap_g = float(batch.cap[g].item() if batch.cap.dim() > 0 else batch.cap.item())

            _, _, m_val = sample_and_repair(m_logits[mask], w_g, v_g, cap_g, greedy=True)
            _, _, b_val = sample_and_repair(b_logits[mask], w_g, v_g, cap_g, greedy=True)
            model_vals.append(m_val)
            base_vals.append(b_val)

    if was_training:
        model.train()

    mv = np.asarray(model_vals)
    bv = np.asarray(base_vals)
    diff = mv - bv
    n = len(diff)
    if n < 2 or diff.std() < 1e-9:
        return float(diff.mean()) > 0

    t_stat = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))
    # One-sided test: need t > t_critical for significance
    # For alpha=0.05 and large n, t_critical ≈ 1.645
    # Use a slightly conservative threshold (1.96 = 2-sided 0.05)
    t_critical = 1.96
    return t_stat > t_critical and float(diff.mean()) > 0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class TrainingLogger:
    HEADER = ["epoch", "train_loss", "avg_reward", "avg_baseline",
              "val_avg_value", "val_ratio", "val_feasibility",
              "baseline_updated", "lr", "time_sec"]

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.rows: List[dict] = []
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict) -> None:
        self.rows.append(row)
        with self.log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADER)
            writer.writeheader()
            writer.writerows(self.rows)


# ---------------------------------------------------------------------------
# CLI and Main
# ---------------------------------------------------------------------------

def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "knapsack_ilp" / "train"

def _default_save_path() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "GNN_REINFORCE" / "gnn_reinforce.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train KnapsackGNN with REINFORCE (policy gradient).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=str, default=_default_data_dir())
    parser.add_argument("--val_dir",       type=str, default=None)
    parser.add_argument("--test_dir",      type=str, default=None)
    parser.add_argument("--k",             type=int, default=16)
    parser.add_argument("--train_ratio",   type=float, default=0.8)
    parser.add_argument("--val_ratio",     type=float, default=0.1)
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "GNN_REINFORCE")

    parser.add_argument("--epochs",        type=int, default=50)
    parser.add_argument("--batch_size",    type=int, default=16)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--hidden_dim",    type=int, default=128)
    parser.add_argument("--num_layers",    type=int, default=3)
    parser.add_argument("--dropout",       type=float, default=0.0)

    parser.add_argument("--conv_type", choices=["gin", "sage", "hybrid"], default="gin")
    parser.add_argument("--no_global_ctx", action="store_true")

    parser.add_argument("--baseline_update_every", type=int, default=1,
                        help="Check for baseline update every N epochs.")
    parser.add_argument("--pretrained",   type=str, default=None,
                        help="Path to a pretrained GNN checkpoint to warmstart.")

    parser.add_argument("--save_path",     type=str, default=_default_save_path())
    parser.add_argument("--seed",          type=int, default=2025)
    parser.add_argument("--grad_clip",     type=float, default=1.0)
    return parser.parse_args()


def load_datasets(args):
    """Load train/val/test datasets with same 3-mode logic as Run_train.py."""
    from torch.utils.data import Subset
    from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.dataset import split_dataset_by_instances

    if args.val_dir and args.test_dir:
        # 3 separate directories — each dataset is complete, no Subset needed
        train_ds = GeneratedKnapsack01Dataset(root_dir=args.dataset_dir, k=args.k)
        val_ds   = GeneratedKnapsack01Dataset(root_dir=args.val_dir,       k=args.k)
        test_ds  = GeneratedKnapsack01Dataset(root_dir=args.test_dir,      k=args.k)

        # Sanity check: warn if any directories are identical (data leakage risk)
        paths = {
            "train": Path(args.dataset_dir).resolve(),
            "val":   Path(args.val_dir).resolve(),
            "test":  Path(args.test_dir).resolve(),
        }
        if len(set(paths.values())) < 3:
            print("WARNING: train/val/test directories are not all distinct!")
            for name, p in paths.items():
                print(f"  {name}: {p}")

        print(f"Dataset: 3 SEPARATE dirs")
        print(f"  Train: {args.dataset_dir} ({len(train_ds)})")
        print(f"  Val:   {args.val_dir} ({len(val_ds)})")
        print(f"  Test:  {args.test_dir} ({len(test_ds)})")
        return train_ds, train_ds, val_ds, test_ds
    elif args.test_dir:
        tv_ds = GeneratedKnapsack01Dataset(root_dir=args.dataset_dir, k=args.k)
        te_ds = GeneratedKnapsack01Dataset(root_dir=args.test_dir,      k=args.k)
        n_tv = len(tv_ds)
        n_train = max(1, int(n_tv * 0.9))
        train = Subset(tv_ds, list(range(n_train)))
        val   = Subset(tv_ds, list(range(n_train, n_tv)))
        test  = Subset(te_ds, list(range(len(te_ds))))
        print(f"Dataset: train+val from {args.dataset_dir}, test from {args.test_dir}")
        return tv_ds, train, val, test
    else:
        ds = GeneratedKnapsack01Dataset(root_dir=args.dataset_dir, k=args.k)
        train, val, test = split_dataset_by_instances(
            ds, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        )
        print(f"Dataset: single dir total={len(ds)} "
              f"train={len(train)} val={len(val)} test={len(test)}")
        return ds, train, val, test


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data
    base_ds, train_set, val_set, test_set = load_datasets(args)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)

    # Model + baseline (frozen copy)
    in_dim = base_ds[0].num_node_features

    model = KnapsackGNN(
        in_dim=in_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, dropout=args.dropout,
        use_global_ctx=not args.no_global_ctx,
        conv_type=args.conv_type,
    ).to(device)

    # Warmstart from pretrained supervised checkpoint if provided
    if args.pretrained:
        print(f"Warmstarting from {args.pretrained}")
        pretrained = load_checkpoint(args.pretrained, device=device, dropout=args.dropout)
        model.load_state_dict(pretrained.state_dict(), strict=False)

    # Baseline = frozen copy of model
    baseline = KnapsackGNN(
        in_dim=in_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, dropout=args.dropout,
        use_global_ctx=not args.no_global_ctx,
        conv_type=args.conv_type,
    ).to(device)
    baseline.load_state_dict(model.state_dict())
    baseline.eval()
    for p in baseline.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: conv={args.conv_type} hidden={args.hidden_dim} "
          f"layers={args.num_layers} params={params:,}")
    print(f"Training: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")

    # Logger
    logger = TrainingLogger(
        Path(__file__).resolve().parents[1] / "results" / "GNN_REINFORCE" / "gnnrl_training_log.csv"
    )

    # Training loop
    best_val_ratio = 0.0
    best_epoch = 0
    train_start = time.perf_counter()

    print(f"\n{'Epoch':>5} | {'Loss':>8} | {'AvgRwd':>8} {'AvgBase':>8} | "
          f"{'Val':>8} {'Ratio':>7} {'Feas':>6} | {'BaseUpdate':>8} | {'time':>6}")
    print("-" * 95)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        # Train
        model.train()
        epoch_loss   = 0.0
        n_batches    = 0
        epoch_reward = 0.0
        epoch_base   = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            loss, stats = reinforce_batch_loss(model, baseline, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss   += loss.item()
            epoch_reward += stats["avg_reward"]
            epoch_base   += stats["avg_baseline"]
            n_batches    += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_rwd  = epoch_reward / max(n_batches, 1)
        avg_bse  = epoch_base / max(n_batches, 1)

        # Validate
        val_metrics = evaluate_policy(model, val_loader, device, greedy=True)

        # Baseline update (paired t-test)
        baseline_updated = False
        if epoch % args.baseline_update_every == 0:
            if paired_t_test_update(model, baseline, val_loader, device):
                baseline.load_state_dict(model.state_dict())
                baseline_updated = True

        elapsed = time.perf_counter() - t0

        updated_str = "YES" if baseline_updated else "no"
        print(f"{epoch:>5} | {avg_loss:>+8.4f} | {avg_rwd:>8.1f} {avg_bse:>8.1f} | "
              f"{val_metrics['avg_value']:>8.1f} {val_metrics['avg_ratio_vs_dp']:>7.4f} "
              f"{val_metrics['feasibility']:>6.3f} | {updated_str:>8} | {elapsed:>5.1f}s")

        # Log
        logger.log({
            "epoch":            epoch,
            "train_loss":       round(avg_loss, 4),
            "avg_reward":       round(avg_rwd, 2),
            "avg_baseline":     round(avg_bse, 2),
            "val_avg_value":    round(val_metrics["avg_value"], 2),
            "val_ratio":        round(val_metrics["avg_ratio_vs_dp"], 4),
            "val_feasibility":  round(val_metrics["feasibility"], 4),
            "baseline_updated": int(baseline_updated),
            "lr":               optimizer.param_groups[0]["lr"],
            "time_sec":         round(elapsed, 1),
        })

        # Save best
        if val_metrics["avg_ratio_vs_dp"] > best_val_ratio:
            best_val_ratio = val_metrics["avg_ratio_vs_dp"]
            best_epoch = epoch
            save_checkpoint(model, args.save_path)

    total_time = time.perf_counter() - train_start

    # Final summary
    print(f"\n{'=' * 95}")
    print(f"TRAINING COMPLETE — {total_time:.1f}s")
    print(f"  Best val ratio: {best_val_ratio:.4f} @ epoch {best_epoch}")

    # Test set
    print(f"\n=== TEST SET EVALUATION ===")
    best_model_path = Path(args.save_path)
    if best_model_path.exists():
        best_model = load_checkpoint(best_model_path, device=device, dropout=0.0)
        test_metrics = evaluate_policy(best_model, test_loader, device, greedy=True)
        print(f"  avg_value:    {test_metrics['avg_value']:.2f}")
        print(f"  avg_ratio:    {test_metrics['avg_ratio_vs_dp']:.4f}")
        print(f"  feasibility:  {test_metrics['feasibility']:.4f}")
        print(f"  n instances:  {test_metrics['n']}")

    print(f"\nModel saved: {args.save_path}")
    print(f"Log saved:   {logger}")


if __name__ == "__main__":
    main()