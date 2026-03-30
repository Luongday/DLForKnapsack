"""Training entry point for KnapsackGNN.

Run from the GNNForKnapSack/ root directory:
    python Graph_Neural_Network/Run_train.py --generated_dir data/knapsack_ilp/train
    python Graph_Neural_Network/Run_train.py --generated_dir data/knapsack_ilp/train --epochs 30
    python Graph_Neural_Network/Run_train.py --dataset_source excel
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch_geometric.loader import DataLoader

import sys
import os
# Thêm thư mục cha của GNNForKnapSack vào sys.path


# ---------------------------------------------------------------------------
# Path setup — must come before any local imports
# Adds GNNForKnapSack/ and Graph_Neural_Network/ to sys.path so that
# both "Knapsack_GNN.xxx" and "BenchMark_GNN" can be found regardless
# of which directory the user runs the script from.
# ---------------------------------------------------------------------------
_HERE       = Path(__file__).resolve().parent          # Graph_Neural_Network/
_GNN_ROOT   = _HERE.parent                             # GNNForKnapSack/

for _p in [str(_GNN_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Local imports (all use paths relative to GNNForKnapSack/)
# ---------------------------------------------------------------------------
from GNNForKnapSack.Graph_Neural_Network.BenchMark_GNN import run_gnn_benchmark                          # noqa: E402
from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.dataset import (                                    # noqa: E402
    GeneratedKnapsack01Dataset,
    KnapsackDataset,
    split_dataset_by_instances,
)
from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.model import KnapsackGNN, save_checkpoint           # noqa: E402
from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.Train_eval import train_one_epoch, evaluate_node_accuracy  # noqa: E402
from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.Eval_compare import evaluate_gnn_vs_dp_on_excel    # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def greedy_feasible_decode(
    probs: torch.Tensor,
    weights: torch.Tensor,
    capacity: float,
) -> torch.Tensor:
    """Pick items by descending GNN probability while respecting capacity."""
    idx   = torch.argsort(probs, descending=True)
    x_hat = torch.zeros_like(probs)
    total = 0.0
    for i in idx:
        w_i = weights[i].item()
        if total + w_i <= capacity + 1e-6:
            x_hat[i] = 1.0
            total    += w_i
    return x_hat


@torch.no_grad()
def evaluate_instance_level(
    model,
    loader,
    num_cases: int = 10,
    device=None,
) -> None:
    model.eval()
    feasible_count  = 0
    value_ratios    = []
    optimal_matches = 0
    seen            = 0

    for batch in loader:
        if device is not None:
            batch = batch.to(device)
        logits    = model(batch)
        probs     = torch.sigmoid(logits)
        batch_vec = (
            batch.batch
            if hasattr(batch, "batch")
            else torch.zeros(probs.size(0), dtype=torch.long, device=probs.device)
        )
        num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 1

        for g in range(num_graphs):
            if seen >= num_cases:
                break
            mask = batch_vec == g
            if mask.sum() == 0:
                continue

            p_g    = probs[mask].detach().cpu()
            w_g    = batch.wts[mask].detach().cpu()
            v_g    = batch.vals[mask].detach().cpu()
            cap_g  = batch.cap[g].item() if batch.cap.dim() > 0 else float(batch.cap.item())
            dp_sol = batch.y[mask].view(-1).detach().cpu()

            x_hat      = greedy_feasible_decode(p_g, w_g, cap_g)
            gnn_weight = float((x_hat * w_g).sum())
            gnn_value  = float((x_hat * v_g).sum())
            dp_value   = float((dp_sol * v_g).sum())
            dp_weight  = float((dp_sol * w_g).sum())
            feasible   = gnn_weight <= cap_g + 1e-6
            ratio      = gnn_value / dp_value if dp_value > 0 else 0.0

            feasible_count  += 1 if feasible else 0
            value_ratios.append(ratio)
            optimal_matches += 1 if abs(gnn_value - dp_value) < 1e-6 else 0

            print(
                f"case {seen+1:03d} | cap={cap_g:.1f} | "
                f"DP: val={dp_value:.1f} wt={dp_weight:.1f} | "
                f"GNN: val={gnn_value:.1f} wt={gnn_weight:.1f} "
                f"feasible={feasible} ratio={ratio:.3f} k={int(x_hat.sum())}"
            )
            seen += 1
        if seen >= num_cases:
            break

    total = max(seen, 1)
    print(
        f"\nEval summary ({total} cases) | "
        f"feasible={feasible_count/total:.3f} | "
        f"avg_ratio={sum(value_ratios)/total:.3f} | "
        f"optimal_match={optimal_matches/total:.3f}"
    )


# ---------------------------------------------------------------------------
# Default paths (relative to GNNForKnapSack/)
# ---------------------------------------------------------------------------

def _default_data_dir() -> str:
    return str(_GNN_ROOT / "data" / "knapsack_ilp" / "train")


def _default_results_dir() -> str:
    return str(_GNN_ROOT / "results")


def _default_save_path() -> str:
    return str(_GNN_ROOT / "results" / "GNN" / "gnn.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train KnapsackGNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_source", choices=["excel", "generated"], default="generated",
    )
    parser.add_argument(
        "--generated_dir", type=str, default=_default_data_dir(),
        help="Directory containing instance_*.npz files.",
    )
    parser.add_argument("--k",           type=int,   default=16)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio",   type=float, default=0.1)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--hidden_dim",  type=int,   default=64)
    parser.add_argument(
        "--save_path", type=str, default=_default_save_path(),
        help="Path to save best model checkpoint.",
    )
    parser.add_argument(
        "--do_benchmark", action="store_true",
        help="Run inference benchmark on test split after training.",
    )
    parser.add_argument("--benchmark_n",       type=int, default=100)
    parser.add_argument(
        "--benchmark_out_dir", type=str, default=_default_results_dir(),
    )
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────
    excel_path = str(_GNN_ROOT / "data" / "data.xlsx")

    if args.dataset_source == "excel":
        dataset = KnapsackDataset(excel_path=excel_path)
    else:
        dataset = GeneratedKnapsack01Dataset(
            root_dir=args.generated_dir,
            k=args.k,
        )

    train_set, val_set, test_set = split_dataset_by_instances(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    print(
        f"Dataset: {args.dataset_source}  total={len(dataset)} "
        f"train={len(train_set)}  val={len(val_set)}  test={len(test_set)}"
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────────
    in_dim = dataset[0].num_node_features if len(dataset) > 0 else 4
    model  = KnapsackGNN(in_dim=in_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Model: in_dim={in_dim}  hidden={args.hidden_dim}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    # ── Training loop ────────────────────────────────────────────────────
    train_start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        t0         = perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc    = evaluate_node_accuracy(model, val_loader, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Loss={train_loss:.4f}  ValAcc={val_acc:.4f}  "
            f"time={perf_counter()-t0:.1f}s"
        )

    print(f"\nTotal training time: {perf_counter()-train_start:.1f}s")

    # ── Save checkpoint ───────────────────────────────────────────────────
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, save_path)
    print(f"Model saved → {save_path}")

    # ── Post-training evaluation ──────────────────────────────────────────
    if args.dataset_source == "excel":
        evaluate_gnn_vs_dp_on_excel(model, excel_path=excel_path)
    else:
        evaluate_instance_level(model, test_loader, num_cases=10, device=device)

        if args.do_benchmark:
            saved = run_gnn_benchmark(
                model, test_loader, device,
                args.benchmark_out_dir,
                n_instances=args.benchmark_n,
                seed=args.seed,
            )
            if saved:
                print("Benchmark outputs:")
                for k, v in saved.items():
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    main()