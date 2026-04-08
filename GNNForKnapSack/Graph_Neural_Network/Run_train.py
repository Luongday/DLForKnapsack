"""Training entry point for KnapsackGNN with Greedy integration.

Greedy is deeply embedded in the training pipeline:
    1. Pre-computed Greedy baseline on val/test (one-time, before training)
    2. Per-epoch comparison: GNN ratio vs Greedy ratio vs DP
    3. Key metric: "GNN beats Greedy" rate — the real goal
    4. Optional greedy-margin loss: penalize GNN when it's worse than Greedy
    5. Post-training: full comparison table GNN vs Greedy vs DP on test set

Usage:
    # Standard training
    python Run_train.py --conv_type gin --generated_dir data/pisinger/type_03_strongly_correlated

    # With greedy-margin loss (penalize GNN when worse than Greedy)
    python Run_train.py --conv_type gin --lambda_greedy 0.1

    # Full ablation
    python Run_train.py --conv_type gin --lambda_greedy 0.1 --lambda_cap_max 0.05
"""

from __future__ import annotations

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
from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.model import KnapsackGNN, save_checkpoint, load_checkpoint
from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.Train_eval import train_one_epoch, evaluate_node_accuracy
from GNNForKnapSack.decode_utils import greedy_feasible_decode, greedy_ratio_decode


# ---------------------------------------------------------------------------
# Greedy baseline pre-computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_greedy_baseline(loader: DataLoader, label: str = "val") -> Dict:
    """Pre-compute Greedy solution values for all instances in a loader.

    Done ONCE before training — O(n log n) per instance, negligible cost.
    Returns per-instance greedy values and DP values for comparison.

    Returns:
        Dict with greedy_values, dp_values, greedy_ratios, avg stats.
    """
    greedy_values:  List[float] = []
    dp_values:      List[float] = []
    greedy_ratios:  List[float] = []
    greedy_feasible: int = 0
    total: int = 0

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

            # Greedy decode
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
    """Evaluate GNN and compare against pre-computed Greedy baseline.

    Returns:
        Dict with gnn_ratio, greedy_ratio, gnn_beats_greedy count, etc.
    """
    model.eval()

    gnn_values:     List[float] = []
    gnn_ratios:     List[float] = []
    gnn_feasible:   int = 0
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

            # GNN decode
            gnn_sel = greedy_feasible_decode(p_g, w_g, cap_g)
            gnn_val = float((gnn_sel * v_g).sum())
            gnn_w   = float((gnn_sel * w_g).sum())

            gnn_values.append(gnn_val)
            gnn_feasible += int(gnn_w <= cap_g + 1e-6)

            ratio = gnn_val / dp_val if dp_val > 0 else 0.0
            gnn_ratios.append(ratio)

            # Compare with Greedy
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
# Instance-level comparison
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_instance_level(
    model:           nn.Module,
    loader:          DataLoader,
    device:          torch.device,
    greedy_baseline: Optional[Dict] = None,
    num_cases:       int = 15,
) -> None:
    """Print per-instance comparison: DP vs GNN vs Greedy."""
    model.eval()
    seen = 0
    gnn_wins = gnn_loses = ties = 0

    print(f"\n{'case':>5} | {'cap':>7} | {'DP':>8} | {'GNN':>8} {'ratio':>6} | "
          f"{'Greedy':>8} {'ratio':>6} | {'winner':>7}")
    print("-" * 80)

    for batch in loader:
        batch = batch.to(device)
        batch_vec = batch.batch if hasattr(batch, "batch") else \
                    torch.zeros(batch.num_nodes, dtype=torch.long, device=batch.x.device)
        n_graphs = int(batch_vec.max().item()) + 1

        logits = model(batch)
        probs  = torch.sigmoid(logits)

        for g in range(n_graphs):
            if seen >= num_cases:
                break
            mask = batch_vec == g
            if mask.sum() == 0:
                continue

            p_g   = probs[mask].detach().cpu()
            w_g   = batch.wts[mask].detach().cpu()
            v_g   = batch.vals[mask].detach().cpu()
            cap_g = float(batch.cap[g].item() if batch.cap.dim() > 0
                          else batch.cap.item())
            dp_sol = batch.y[mask].view(-1).detach().cpu()
            dp_val = float((dp_sol * v_g).sum())

            # GNN decode
            gnn_sel = greedy_feasible_decode(p_g, w_g, cap_g)
            gnn_val = float((gnn_sel * v_g).sum())
            gnn_r   = gnn_val / dp_val if dp_val > 0 else 0.0

            # Greedy decode
            gr_sel = greedy_ratio_decode(v_g, w_g, cap_g)
            gr_val = float((gr_sel * v_g).sum())
            gr_r   = gr_val / dp_val if dp_val > 0 else 0.0

            # Winner
            if gnn_val > gr_val + 1e-6:
                winner = "GNN"
                gnn_wins += 1
            elif gr_val > gnn_val + 1e-6:
                winner = "Greedy"
                gnn_loses += 1
            else:
                winner = "tie"
                ties += 1

            print(f"{seen+1:>5} | {cap_g:>7.0f} | {dp_val:>8.0f} | "
                  f"{gnn_val:>8.0f} {gnn_r:>6.3f} | "
                  f"{gr_val:>8.0f} {gr_r:>6.3f} | {winner:>7}")
            seen += 1
        if seen >= num_cases:
            break

    total = max(seen, 1)
    print("-" * 80)
    print(f"  GNN wins: {gnn_wins}/{total}  |  Greedy wins: {gnn_loses}/{total}  |  "
          f"Ties: {ties}/{total}")


# ---------------------------------------------------------------------------
# Greedy-margin loss (optional)
# ---------------------------------------------------------------------------

def train_one_epoch_with_greedy(
    model, loader, optimizer, criterion, device,
    lambda_cap: float = 0.0,
    lambda_greedy: float = 0.0,
) -> float:
    """Train one epoch with optional greedy-margin penalty.

    Standard BCE loss + capacity penalty + greedy margin penalty.

    Greedy-margin loss: when GNN solution value < Greedy value,
    add a penalty proportional to the gap. This teaches the model
    "at minimum, be as good as Greedy."

    Args:
        lambda_cap:     Capacity violation penalty (0 = off).
        lambda_greedy:  Greedy-margin penalty (0 = off). Try 0.05-0.2.
    """
    import torch.nn.functional as F

    model.train()
    total_loss  = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch)
        y      = batch.y.view(-1)

        # BCE loss
        bce_loss = criterion(logits, y)
        loss = bce_loss

        probs     = torch.sigmoid(logits)
        batch_vec = batch.batch if hasattr(batch, "batch") else \
                    torch.zeros(probs.size(0), dtype=torch.long, device=device)
        n_graphs  = int(batch_vec.max().item()) + 1

        # Capacity penalty
        if lambda_cap > 0.0:
            cap_violations = []
            for g in range(n_graphs):
                mask  = batch_vec == g
                p_g   = probs[mask]
                w_g   = batch.wts[mask]
                cap_g = batch.cap[g].item() if batch.cap.dim() > 0 else float(batch.cap.item())
                violation = F.relu((p_g * w_g).sum() - cap_g)
                cap_violations.append(violation)
            loss = loss + lambda_cap * torch.stack(cap_violations).mean()

        # Greedy-margin penalty: penalize when soft-GNN value < Greedy value
        if lambda_greedy > 0.0:
            greedy_gaps = []
            for g in range(n_graphs):
                mask  = batch_vec == g
                p_g   = probs[mask]
                w_g   = batch.wts[mask]
                v_g   = batch.vals[mask]
                cap_g = batch.cap[g].item() if batch.cap.dim() > 0 else float(batch.cap.item())

                # Soft GNN value (differentiable)
                gnn_soft_val = (p_g * v_g).sum()

                # Greedy value (non-differentiable, computed as constant)
                with torch.no_grad():
                    gr_sel = greedy_ratio_decode(v_g.cpu(), w_g.cpu(), cap_g)
                    gr_val = (gr_sel.to(device) * v_g).sum()

                # Penalize: max(0, greedy_value - gnn_soft_value)
                gap = F.relu(gr_val - gnn_soft_val)
                greedy_gaps.append(gap)

            if greedy_gaps:
                greedy_penalty = torch.stack(greedy_gaps).mean()
                loss = loss + lambda_greedy * greedy_penalty

        loss.backward()
        optimizer.step()

        total_loss  += loss.item() * y.numel()
        total_nodes += y.numel()

    return total_loss / max(total_nodes, 1)


# ---------------------------------------------------------------------------
# Training history logger
# ---------------------------------------------------------------------------

class TrainingLogger:
    """Log per-epoch metrics to CSV with Greedy comparison."""

    HEADER = [
        "epoch", "train_loss", "val_acc",
        "gnn_ratio", "gnn_std", "greedy_ratio",
        "gnn_beats_greedy", "gnn_loses_greedy", "ties",
        "advantage", "lr", "lambda_cap", "time_sec",
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
        description="Train KnapsackGNN with Greedy integration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_source", choices=["excel", "generated"], default="generated")
    parser.add_argument("--generated_dir",  type=str,   default=_default_data_dir(),
                        help="Training data directory (instance_*.npz)")
    parser.add_argument("--val_dir",        type=str,   default=None,
                        help="Separate validation directory. "
                             "If not set, split from generated_dir.")
    parser.add_argument("--test_dir",       type=str,   default=None,
                        help="Separate test directory. "
                             "If not set, split from generated_dir.")
    parser.add_argument("--k",              type=int,   default=16)
    parser.add_argument("--train_ratio",    type=float, default=0.8,
                        help="Train fraction (only used when val_dir/test_dir not set)")
    parser.add_argument("--val_ratio",      type=float, default=0.1,
                        help="Val fraction (only used when val_dir/test_dir not set)")
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--hidden_dim",     type=int,   default=128)
    parser.add_argument("--num_layers",     type=int,   default=3)
    parser.add_argument("--dropout",        type=float, default=0.1)

    # Loss components
    parser.add_argument("--lambda_cap_max", type=float, default=0.05,
                        help="Max capacity penalty (0 = off).")
    parser.add_argument("--lambda_cap_warmup_epochs", type=int, default=15)
    parser.add_argument("--lambda_greedy", type=float, default=0.0,
                        help="Greedy-margin penalty: penalize GNN when it "
                             "produces worse value than Greedy. Try 0.05-0.2. "
                             "0 = off (standard BCE only).")

    # Architecture
    parser.add_argument("--conv_type", choices=["gin", "sage", "hybrid"],
                        default="gin")
    parser.add_argument("--no_global_ctx", action="store_true")

    parser.add_argument("--save_path",       type=str, default=_default_save_path())
    parser.add_argument("--early_stop_wait", type=int, default=15)
    parser.add_argument("--seed",            type=int, default=2025)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)

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
            # MODE 1: Three separate directories (train / val / test)
            train_dataset = GeneratedKnapsack01Dataset(root_dir=args.generated_dir, k=args.k)
            val_dataset   = GeneratedKnapsack01Dataset(root_dir=args.val_dir,       k=args.k)
            test_dataset  = GeneratedKnapsack01Dataset(root_dir=args.test_dir,      k=args.k)

            train_set = Subset(train_dataset, list(range(len(train_dataset))))
            val_set   = Subset(val_dataset,   list(range(len(val_dataset))))
            test_set  = Subset(test_dataset,  list(range(len(test_dataset))))
            dataset   = train_dataset  # for in_dim detection

            print(f"Dataset: 3 SEPARATE dirs")
            print(f"  Train: {args.generated_dir} ({len(train_set)} instances)")
            print(f"  Val:   {args.val_dir} ({len(val_set)} instances)")
            print(f"  Test:  {args.test_dir} ({len(test_set)} instances)")

        elif has_test_dir:
            # MODE 2: Train+Val from generated_dir, Test from test_dir
            train_val_dataset = GeneratedKnapsack01Dataset(root_dir=args.generated_dir, k=args.k)
            test_dataset      = GeneratedKnapsack01Dataset(root_dir=args.test_dir,      k=args.k)

            # Split train_val: rescale ratios to sum to 1.0
            total_tv = args.train_ratio + args.val_ratio
            inner_train = args.train_ratio / total_tv if total_tv > 0 else 0.9

            n_tv    = len(train_val_dataset)
            n_train = max(1, min(int(n_tv * inner_train), n_tv - 1))

            indices   = list(range(n_tv))
            train_set = Subset(train_val_dataset, indices[:n_train])
            val_set   = Subset(train_val_dataset, indices[n_train:])
            test_set  = Subset(test_dataset, list(range(len(test_dataset))))
            dataset   = train_val_dataset

            print(f"Dataset: SEPARATE test dir")
            print(f"  Train+Val: {args.generated_dir} ({n_tv} → "
                  f"train={len(train_set)}, val={len(val_set)})")
            print(f"  Test:      {args.test_dir} ({len(test_set)} instances)")

        else:
            # MODE 3: Single directory, auto-split
            dataset = GeneratedKnapsack01Dataset(root_dir=args.generated_dir, k=args.k)
            train_set, val_set, test_set = split_dataset_by_instances(
                dataset, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
            )
            print(f"Dataset: {args.dataset_source}  total={len(dataset)} "
                  f"train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)

    # --- Pre-compute Greedy baselines (one-time) ---
    print("\nPre-computing Greedy baselines...")
    val_greedy  = precompute_greedy_baseline(val_loader,  "val")
    test_greedy = precompute_greedy_baseline(test_loader, "test")

    greedy_target = val_greedy["avg_ratio"]
    print(f"\n  Target to beat: Greedy val ratio = {greedy_target:.4f}")
    print(f"  Greedy optimal count: {val_greedy['optimal_count']}/{val_greedy['n_instances']}")

    # --- Model ---
    in_dim = dataset[0].num_node_features if len(dataset) > 0 else 6
    model  = KnapsackGNN(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_global_ctx=not args.no_global_ctx,
        conv_type=args.conv_type,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5,
    )
    criterion = nn.BCEWithLogitsLoss()

    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: in={in_dim} hidden={args.hidden_dim} layers={args.num_layers} "
          f"conv={args.conv_type} ctx={not args.no_global_ctx} params={params:,}")
    if args.lambda_greedy > 0:
        print(f"Greedy-margin loss: lambda_greedy={args.lambda_greedy}")
    if args.lambda_cap_max > 0:
        print(f"Capacity penalty: max={args.lambda_cap_max} warmup={args.lambda_cap_warmup_epochs}ep")

    # --- Logger ---
    logger = TrainingLogger(
        Path(args.save_path).parent.parent / "logs" / "training_log.csv"
    )

    # --- Training loop ---
    best_ratio  = 0.0
    best_epoch  = 0
    best_beats  = 0
    wait        = 0

    print(f"\n{'Epoch':>5} | {'Loss':>7} {'Acc':>6} | "
          f"{'GNN':>7} {'±':>6} | {'Greedy':>7} | "
          f"{'Beats':>5} {'Loses':>5} {'Ties':>4} | "
          f"{'Adv':>7} | {'time':>5}")
    print("-" * 90)

    train_start = perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0 = perf_counter()

        # Capacity penalty curriculum
        if args.lambda_cap_max > 0 and args.lambda_cap_warmup_epochs > 0:
            lambda_cap = min(
                args.lambda_cap_max,
                args.lambda_cap_max * epoch / args.lambda_cap_warmup_epochs,
            )
        else:
            lambda_cap = 0.0

        # Train (with greedy-margin if enabled)
        if args.lambda_greedy > 0:
            train_loss = train_one_epoch_with_greedy(
                model, train_loader, optimizer, criterion, device,
                lambda_cap=lambda_cap,
                lambda_greedy=args.lambda_greedy,
            )
        else:
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                lambda_cap=lambda_cap,
            )

        val_acc = evaluate_node_accuracy(model, val_loader, device)

        # Evaluate GNN vs Greedy on validation
        val_metrics = evaluate_with_greedy(model, val_loader, device, val_greedy)

        elapsed = perf_counter() - t0

        # Print compact epoch line
        gnn_r  = val_metrics["gnn_avg_ratio"]
        gnn_s  = val_metrics["gnn_std_ratio"]
        gr_r   = val_metrics["greedy_avg_ratio"]
        beats  = val_metrics["gnn_beats_greedy"]
        loses  = val_metrics["gnn_loses_greedy"]
        ties_  = val_metrics["gnn_ties_greedy"]
        adv    = val_metrics["advantage"]
        n_inst = val_metrics["n_instances"]

        adv_str = f"{adv:>+7.4f}" if adv != 0 else "  0.000"
        beat_marker = " *" if adv > 0 else ""

        print(f"{epoch:>5} | {train_loss:>7.4f} {val_acc:>6.3f} | "
              f"{gnn_r:>7.4f} {gnn_s:>5.3f} | {gr_r:>7.4f} | "
              f"{beats:>5} {loses:>5} {ties_:>4} | "
              f"{adv_str} | {elapsed:>5.1f}s{beat_marker}")

        # Log to CSV
        logger.log({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_acc": round(val_acc, 4),
            "gnn_ratio": round(gnn_r, 4), "gnn_std": round(gnn_s, 4),
            "greedy_ratio": round(gr_r, 4),
            "gnn_beats_greedy": beats, "gnn_loses_greedy": loses, "ties": ties_,
            "advantage": round(adv, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "lambda_cap": round(lambda_cap, 4),
            "time_sec": round(elapsed, 1),
        })

        # Save best model (by GNN ratio, not by beats)
        if gnn_r > best_ratio:
            best_ratio = gnn_r
            best_epoch = epoch
            best_beats = beats
            wait = 0
            save_checkpoint(model, args.save_path)
        else:
            wait += 1

        # LR scheduler
        scheduler.step(gnn_r)

        # Early stopping
        if wait >= args.early_stop_wait:
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"Best: ratio={best_ratio:.4f} @ epoch {best_epoch}")
            break

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

    best_model_path = Path(args.save_path)
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
        print(f"    GNN wins:  {test_metrics['gnn_beats_greedy']:>4} "
              f"({test_metrics['beats_rate']*100:.1f}%)")
        print(f"    Greedy wins: {test_metrics['gnn_loses_greedy']:>4} "
              f"({test_metrics['gnn_loses_greedy']/max(n_test,1)*100:.1f}%)")
        print(f"    Ties:      {test_metrics['gnn_ties_greedy']:>4}")
        print(f"    Advantage: {test_metrics['advantage']:+.4f}")

        # Instance-level comparison
        print(f"\n--- Per-instance comparison (first 15 test cases) ---")
        evaluate_instance_level(best_model, test_loader, device, test_greedy, num_cases=15)
    else:
        print("  Best model not found — showing current model results")
        evaluate_instance_level(model, test_loader, device, None, num_cases=15)


if __name__ == "__main__":
    main()