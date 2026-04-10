"""Training entry point for KnapsackGNN — v2 stability edition."""

from __future__ import annotations

import wandb
import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

_HERE     = Path(__file__).resolve().parent
_GNN_ROOT = _HERE.parent
for _p in [str(_GNN_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.dataset import (
    GeneratedKnapsack01Dataset,
    KnapsackDataset,
    split_dataset_by_instances,
)
from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.model import (
    KnapsackGNN, save_checkpoint, load_checkpoint,
)
from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.Train_eval import (
    train_one_epoch, evaluate_node_accuracy,
)
from GNNForKnapSack.decode_utils import greedy_feasible_decode, greedy_ratio_decode

# ---------------------------------------------------------------------------
# Greedy baseline pre-computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_greedy_baseline(loader: DataLoader, label: str = "val") -> Dict:
    """Pre-compute Greedy solution values for all instances in a loader."""
    greedy_values:   List[float] = []
    dp_values:       List[float] = []
    greedy_ratios:   List[float] = []
    greedy_feasible: int = 0
    total:           int = 0

    for batch in loader:
        batch_vec = batch.batch if hasattr(batch, "batch") else \
                    torch.zeros(batch.num_nodes, dtype=torch.long)
        n_graphs  = int(batch_vec.max().item()) + 1

        for g in range(n_graphs):
            mask  = batch_vec == g
            w_g   = batch.wts[mask].cpu()
            v_g   = batch.vals[mask].cpu()
            cap_g = float(batch.cap[g].item() if batch.cap.dim() > 0
                          else batch.cap.item())
            dp_sol = batch.y[mask].view(-1).cpu()

            dp_val = float((dp_sol * v_g).sum())

            gr_sel = greedy_ratio_decode(v_g, w_g, cap_g)
            gr_val = float((gr_sel * v_g).sum())
            gr_w   = float((gr_sel * w_g).sum())

            greedy_values.append(gr_val)
            dp_values.append(dp_val)
            greedy_feasible += int(gr_w <= cap_g + 1e-6)

            ratio = gr_val / dp_val if dp_val > 0 else 0.0
            greedy_ratios.append(ratio)
            total += 1

    avg_ratio = float(np.mean(greedy_ratios)) if greedy_ratios else 0.0
    std_ratio = float(np.std(greedy_ratios))  if greedy_ratios else 0.0
    optimal_count = sum(1 for r in greedy_ratios if abs(r - 1.0) < 1e-6)

    print(f"  [{label}] Greedy baseline: ratio={avg_ratio:.4f} +/- {std_ratio:.4f} | "
          f"feasible={greedy_feasible}/{total} | "
          f"optimal={optimal_count}/{total}")

    return {
        "greedy_values":   greedy_values,
        "dp_values":       dp_values,
        "greedy_ratios":   greedy_ratios,
        "avg_ratio":       avg_ratio,
        "std_ratio":       std_ratio,
        "feasible_rate":   greedy_feasible / max(total, 1),
        "optimal_count":   optimal_count,
        "n_instances":     total,
    }


# ---------------------------------------------------------------------------
# GNN evaluation with Greedy comparison
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_with_greedy(
    model:           nn.Module,
    loader:          DataLoader,
    device:          torch.device,
    greedy_baseline: Dict,
) -> Dict:
    """Evaluate GNN and compare against pre-computed Greedy baseline."""
    model.eval()

    gnn_values:       List[float] = []
    gnn_ratios:       List[float] = []
    gnn_feasible:     int = 0
    gnn_beats_greedy: int = 0
    gnn_ties_greedy:  int = 0
    instance_idx = 0

    for batch in loader:
        batch = batch.to(device)
        batch_vec = batch.batch if hasattr(batch, "batch") else \
                    torch.zeros(batch.num_nodes, dtype=torch.long, device=device)
        n_graphs  = int(batch_vec.max().item()) + 1

        logits = model(batch)
        probs  = torch.sigmoid(logits)

        for g in range(n_graphs):
            mask  = batch_vec == g
            p_g   = probs[mask].detach().cpu()
            w_g   = batch.wts[mask].detach().cpu()
            v_g   = batch.vals[mask].detach().cpu()
            cap_g = float(batch.cap[g].item() if batch.cap.dim() > 0
                          else batch.cap.item())

            dp_sol = batch.y[mask].view(-1).detach().cpu()
            dp_val = float((dp_sol * v_g).sum())

            gnn_sel = greedy_feasible_decode(p_g, w_g, cap_g)
            gnn_val = float((gnn_sel * v_g).sum())
            gnn_w   = float((gnn_sel * w_g).sum())

            gnn_values.append(gnn_val)
            gnn_feasible += int(gnn_w <= cap_g + 1e-6)

            ratio = gnn_val / dp_val if dp_val > 0 else 0.0
            gnn_ratios.append(ratio)

            if instance_idx < len(greedy_baseline["greedy_values"]):
                gr_val = greedy_baseline["greedy_values"][instance_idx]
                if gnn_val > gr_val + 1e-6:
                    gnn_beats_greedy += 1
                elif abs(gnn_val - gr_val) < 1e-6:
                    gnn_ties_greedy += 1

            instance_idx += 1

    n = max(instance_idx, 1)
    gnn_avg_ratio    = float(np.mean(gnn_ratios)) if gnn_ratios else 0.0
    gnn_std_ratio    = float(np.std(gnn_ratios))  if gnn_ratios else 0.0
    greedy_avg_ratio = greedy_baseline["avg_ratio"]
    gnn_loses_greedy = n - gnn_beats_greedy - gnn_ties_greedy

    return {
        "gnn_avg_ratio":      gnn_avg_ratio,
        "gnn_std_ratio":      gnn_std_ratio,
        "greedy_avg_ratio":   greedy_avg_ratio,
        "gnn_feasible_rate":  gnn_feasible / n,
        "gnn_beats_greedy":   gnn_beats_greedy,
        "gnn_ties_greedy":    gnn_ties_greedy,
        "gnn_loses_greedy":   gnn_loses_greedy,
        "beats_rate":         gnn_beats_greedy / n,
        "n_instances":        n,
        "advantage":          gnn_avg_ratio - greedy_avg_ratio,
    }


# ---------------------------------------------------------------------------
# Training history logger
# ---------------------------------------------------------------------------

class TrainingLogger:
    """Log per-epoch metrics to CSV with Greedy comparison."""

    HEADER = [
        "epoch", "train_loss", "val_acc",
        "gnn_ratio", "gnn_std", "greedy_ratio",
        "gnn_beats_greedy", "gnn_loses_greedy", "ties",
        "advantage", "lr", "time_sec",
    ]

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path
        self.rows: List[dict] = []
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict):
        self.rows.append(row)
        if self.log_path:
            with self.log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.HEADER)
                writer.writeheader()
                writer.writerows(self.rows)


# ---------------------------------------------------------------------------
# LR warmup helper
# ---------------------------------------------------------------------------

def get_warmup_lr(base_lr: float, epoch: int, warmup_epochs: int) -> float:
    """Linear warmup: lr scales from 0.1*base_lr to base_lr over warmup_epochs."""
    if epoch >= warmup_epochs:
        return base_lr
    return base_lr * (0.1 + 0.9 * epoch / max(warmup_epochs, 1))


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

def _default_data_dir() -> str:
    return str(_GNN_ROOT / "data" / "knapsack_ilp" / "train")

def _default_save_path() -> str:
    return str(_GNN_ROOT / "results" / "GNN" / "gnn.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train KnapsackGNN v2 — stability edition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_source", choices=["excel", "generated"], default="generated")
    parser.add_argument("--generated_dir",  type=str, default=_default_data_dir())
    parser.add_argument("--val_dir",        type=str, default=None)
    parser.add_argument("--test_dir",       type=str, default=None)
    parser.add_argument("--k",              type=int, default=16)
    parser.add_argument("--train_ratio",    type=float, default=0.8)
    parser.add_argument("--val_ratio",      type=float, default=0.1)

    # Training config — tuned for n=10-200 dataset
    parser.add_argument("--epochs",     type=int,   default=150)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear LR warmup for first N epochs")

    # Model — larger capacity for n=200
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout",    type=float, default=0.15)

    # Architecture
    parser.add_argument("--conv_type", choices=["gin", "sage", "hybrid"], default="gin")
    parser.add_argument("--no_global_ctx", action="store_true")

    parser.add_argument("--save_path",       type=str, default=_default_save_path())
    parser.add_argument("--early_stop_wait", type=int, default=30,
                        help="Early stopping patience (v2: increased from 15)")
    parser.add_argument("--seed",            type=int, default=2025)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"KnapsackGNN v2 — stability edition")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Dataset ---
    excel_path = str(_GNN_ROOT / "data" / "data.xlsx")

    if args.dataset_source == "excel":
        dataset = KnapsackDataset(excel_path=excel_path)
        train_set, val_set, test_set = split_dataset_by_instances(
            dataset, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        )
        print(f"Dataset: excel  total={len(dataset)} "
              f"train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")
    else:
        from torch.utils.data import Subset

        has_val_dir  = args.val_dir  is not None
        has_test_dir = args.test_dir is not None

        if has_val_dir and has_test_dir:
            # MODE 1: Three separate directories
            train_dataset = GeneratedKnapsack01Dataset.get_lazy(root_dir=args.generated_dir, k=args.k)
            val_dataset   = GeneratedKnapsack01Dataset.get_lazy(root_dir=args.val_dir,       k=args.k)
            test_dataset  = GeneratedKnapsack01Dataset.get_lazy(root_dir=args.test_dir,      k=args.k)

            train_set = train_dataset
            val_set   = val_dataset
            test_set  = test_dataset
            dataset   = train_dataset
            print(f"Dataset: 3 SEPARATE dirs")
            print(f"  Train: {args.generated_dir} ({len(train_set)})")
            print(f"  Val:   {args.val_dir} ({len(val_set)})")
            print(f"  Test:  {args.test_dir} ({len(test_set)})")

        elif has_test_dir:
            # MODE 2: train/val from generated_dir, test from test_dir
            train_val_dataset = GeneratedKnapsack01Dataset.get_lazy(root_dir=args.generated_dir, k=args.k)
            test_dataset      = GeneratedKnapsack01Dataset.get_lazy(root_dir=args.test_dir,      k=args.k)
            n_tv    = len(train_val_dataset)
            n_train = max(1, int(n_tv * 0.9))
            train_set = Subset(train_val_dataset, list(range(n_train)))
            val_set   = Subset(train_val_dataset, list(range(n_train, n_tv)))
            test_set  = test_dataset
            dataset   = train_val_dataset
            print(f"Dataset: train+val from {args.generated_dir} "
                  f"({n_tv} → train={len(train_set)}, val={len(val_set)})")
            print(f"  Test:      {args.test_dir} ({len(test_set)} instances)")

        else:
            # MODE 3: Single directory, auto-split
            dataset = GeneratedKnapsack01Dataset.get_lazy(root_dir=args.generated_dir, k=args.k)
            train_set, val_set, test_set = split_dataset_by_instances(
                dataset, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
            )
            print(f"Dataset: {args.dataset_source}  total={len(dataset)} "
                  f"train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)

    # --- Pre-compute Greedy baselines ---
    print("\nPre-computing Greedy baselines...")
    val_greedy  = precompute_greedy_baseline(val_loader,  "val")
    test_greedy = precompute_greedy_baseline(test_loader, "test")

    greedy_target = val_greedy["avg_ratio"]
    print(f"\n  Target to beat: Greedy val ratio = {greedy_target:.4f}")
    print(f"  Greedy optimal count: {val_greedy['optimal_count']}/{val_greedy['n_instances']}")

    # --- Model ---
    in_dim = dataset[0].num_node_features if len(dataset) > 0 else 7
    model  = KnapsackGNN(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_global_ctx=not args.no_global_ctx,
        conv_type=args.conv_type,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-5,
    )

    # Cosine annealing with warm restarts — more stable than ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=25,   # First restart at epoch 25
        T_mult=2, # Then 50, 100, ...
        eta_min=args.lr * 0.01,  # Min LR = 1% of base
    )

    criterion = nn.BCEWithLogitsLoss()

    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: in={in_dim} hidden={args.hidden_dim} layers={args.num_layers} "
          f"conv={args.conv_type} ctx={not args.no_global_ctx} params={params:,}")
    print(f"Training: epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
          f"warmup={args.warmup_epochs}")

    # --- Logger ---
    logger = TrainingLogger(
        Path(__file__).resolve().parents[1] / "results" / "GNN" / "gnn_training_log.csv"
    )

    # --- Training loop ---
    best_ratio  = 0.0
    best_epoch  = 0
    wait        = 0

    print(f"\n{'Epoch':>5} | {'Loss':>7} {'Acc':>6} | "
          f"{'GNN':>7} {'±':>6} | {'Greedy':>7} | "
          f"{'Beats':>5} {'Loses':>5} {'Ties':>4} | "
          f"{'Adv':>7} | {'LR':>8} | {'time':>5}")
    print("-" * 105)

    train_start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        t0 = perf_counter()

        # LR warmup for first epochs
        if epoch <= args.warmup_epochs:
            lr_epoch = get_warmup_lr(args.lr, epoch, args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_epoch

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
        )

        val_acc     = evaluate_node_accuracy(model, val_loader, device)
        val_metrics = evaluate_with_greedy(model, val_loader, device, val_greedy)

        elapsed = perf_counter() - t0

        gnn_r  = val_metrics["gnn_avg_ratio"]
        gnn_s  = val_metrics["gnn_std_ratio"]
        gr_r   = val_metrics["greedy_avg_ratio"]
        beats  = val_metrics["gnn_beats_greedy"]
        loses  = val_metrics["gnn_loses_greedy"]
        ties_  = val_metrics["gnn_ties_greedy"]
        adv    = val_metrics["advantage"]
        cur_lr = optimizer.param_groups[0]["lr"]

        adv_str = f"{adv:>+7.4f}" if adv != 0 else "  0.000"
        beat_marker = " *" if adv > 0 else ""

        print(f"{epoch:>5} | {train_loss:>7.4f} {val_acc:>6.3f} | "
              f"{gnn_r:>7.4f} {gnn_s:>5.3f} | {gr_r:>7.4f} | "
              f"{beats:>5} {loses:>5} {ties_:>4} | "
              f"{adv_str} | {cur_lr:>.2e} | {elapsed:>5.1f}s{beat_marker}")

        # Log to CSV
        logger.log({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_acc": round(val_acc, 4),
            "gnn_ratio": round(gnn_r, 4), "gnn_std": round(gnn_s, 4),
            "greedy_ratio": round(gr_r, 4),
            "gnn_beats_greedy": beats, "gnn_loses_greedy": loses, "ties": ties_,
            "advantage": round(adv, 4),
            "lr": round(cur_lr, 6),
            "time_sec": round(elapsed, 1),
        })


        # Save best model (gnn_best.pt)
        if gnn_r > best_ratio:
            best_ratio = gnn_r
            best_epoch = epoch
            wait = 0
            best_path = Path(args.save_path).parent / "gnn_best.pt"
            save_checkpoint(model, best_path)
        else:
            wait += 1

        # Step scheduler (only after warmup)
        if epoch > args.warmup_epochs:
            scheduler.step()

        # Early stopping
        if wait >= args.early_stop_wait:
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"Best: ratio={best_ratio:.4f} @ epoch {best_epoch}")
            break

    # Save final model too
    save_checkpoint(model, args.save_path)
    total_time = perf_counter() - train_start

    # --- Summary ---
    print(f"\n{'='*90}")
    print(f"TRAINING COMPLETE — {total_time:.1f}s")
    print(f"{'='*90}")
    print(f"  Best GNN ratio: {best_ratio:.4f} @ epoch {best_epoch}")
    print(f"  Greedy baseline: {val_greedy['avg_ratio']:.4f}")
    print(f"  Advantage: {best_ratio - val_greedy['avg_ratio']:+.4f}")
    if best_ratio > val_greedy["avg_ratio"]:
        print(f"  >>> GNN BEATS GREEDY on validation! <<<")
    else:
        gap = val_greedy["avg_ratio"] - best_ratio
        print(f"  GNN still behind Greedy by {gap:.4f}")

    # --- Test set evaluation ---
    print(f"\n{'='*90}")
    print("TEST SET EVALUATION (best model)")
    print(f"{'='*90}")

    best_model_path = Path(args.save_path).parent / "gnn_best.pt"
    if best_model_path.exists():
        best_model = load_checkpoint(best_model_path, device=device, dropout=0.0)
        test_metrics = evaluate_with_greedy(best_model, test_loader, device, test_greedy)

        print(f"\n  {'Solver':<10} {'Ratio':>8} {'Feasible':>10} {'Beats':>6}")
        print(f"  {'-'*40}")
        print(f"  {'DP':<10} {'1.0000':>8} {'100%':>10} {'—':>6}")
        print(f"  {'Greedy':<10} {test_greedy['avg_ratio']:>8.4f} "
              f"{test_greedy['feasible_rate']*100:>9.1f}% {'—':>6}")
        print(f"  {'GNN':<10} {test_metrics['gnn_avg_ratio']:>8.4f} "
              f"{test_metrics['gnn_feasible_rate']*100:>9.1f}% "
              f"{test_metrics['gnn_beats_greedy']:>6}")

        n_test = test_metrics["n_instances"]
        print(f"\n  GNN vs Greedy on test ({n_test} instances):")
        print(f"    GNN wins:    {test_metrics['gnn_beats_greedy']:>4} "
              f"({test_metrics['beats_rate']*100:.1f}%)")
        print(f"    Greedy wins: {test_metrics['gnn_loses_greedy']:>4} "
              f"({test_metrics['gnn_loses_greedy']/max(n_test,1)*100:.1f}%)")
        print(f"    Ties:        {test_metrics['gnn_ties_greedy']:>4}")
        print(f"    Advantage:   {test_metrics['advantage']:+.4f}")

if __name__ == "__main__":
    main()