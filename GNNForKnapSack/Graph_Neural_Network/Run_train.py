"""Training entry point for KnapsackGNN.

Fixes vs original:
    - Uses save_checkpoint() / load_checkpoint() from model.py so that
      evaluate_gnn.py can correctly load the saved file (was broken: trainer
      saved a raw state_dict but loader expected a dict with 'model_state' key).
    - Removed _split_with_minimum() — uses dataset.split_dataset_by_instances()
      which already exists and handles edge cases.
    - No hardcoded absolute paths; all defaults are relative to repo root.
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from BenchMark_GNN import run_gnn_benchmark
from Knapsack_GNN.dataset import GeneratedKnapsack01Dataset, KnapsackDataset, split_dataset_by_instances
from Knapsack_GNN.model import KnapsackGNN, save_checkpoint
from Knapsack_GNN.Train_eval import train_one_epoch, evaluate_node_accuracy
from Knapsack_GNN.Eval_compare import evaluate_gnn_vs_dp_on_excel


def greedy_feasible_decode(
    probs: torch.Tensor, weights: torch.Tensor, capacity: float
) -> torch.Tensor:
    """Pick items by descending probability while respecting capacity."""
    idx = torch.argsort(probs, descending=True)
    x_hat = torch.zeros_like(probs)
    total_w = 0.0
    for i in idx:
        w_i = weights[i].item()
        if total_w + w_i <= capacity:
            x_hat[i] = 1.0
            total_w += w_i
    return x_hat


@torch.no_grad()
def evaluate_instance_level(model, loader, num_cases: int = 10, device=None):
    model.eval()
    feasible_count = 0
    value_ratios = []
    optimal_matches = 0
    seen = 0

    for batch in loader:
        batch = batch.to(device) if device is not None else batch
        logits = model(batch)
        probs = torch.sigmoid(logits)

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

            p_g = probs[mask].detach().cpu()
            w_g = batch.wts[mask].detach().cpu()
            v_g = batch.vals[mask].detach().cpu()
            cap_g = batch.cap[g].item() if batch.cap.dim() > 0 else float(batch.cap.item())
            dp_sol = batch.y[mask].view(-1).detach().cpu()

            x_hat = greedy_feasible_decode(p_g, w_g, cap_g)

            gnn_weight = float((x_hat * w_g).sum())
            gnn_value = float((x_hat * v_g).sum())
            feasible = gnn_weight <= cap_g + 1e-6
            dp_value = float((dp_sol * v_g).sum())
            dp_weight = float((dp_sol * w_g).sum())
            ratio = gnn_value / dp_value if dp_value > 0 else 0.0

            feasible_count += 1 if feasible else 0
            value_ratios.append(ratio)
            optimal_matches += 1 if abs(gnn_value - dp_value) < 1e-6 else 0

            print(
                f"case {seen+1:03d} | cap={cap_g:.1f} | "
                f"DP: value={dp_value:.1f}, weight={dp_weight:.1f} | "
                f"GNN: value={gnn_value:.1f}, weight={gnn_weight:.1f}, feasible={feasible} | "
                f"ratio={ratio:.3f} | selected_k={int(x_hat.sum())}"
            )
            seen += 1
        if seen >= num_cases:
            break

    total = max(seen, 1)
    print(
        f"Eval summary over {total} cases | "
        f"feasible_rate={feasible_count/total:.3f} | "
        f"avg_value_ratio={sum(value_ratios)/total:.3f} | "
        f"optimal_match_rate={optimal_matches/total:.3f}"
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_generated_dir() -> str:
    return str(_repo_root() / "dataset" / "knapsack01_medium")


def _default_results_dir() -> str:
    return str(_repo_root() / "results")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Knapsack GNN.")
    parser.add_argument(
        "--dataset_source",
        choices=["excel", "generated"],
        default="generated",
    )
    parser.add_argument(
        "--generated_dir",
        type=str,
        default=_default_generated_dir(),
        help="Directory with generated NPZ files.",
    )
    parser.add_argument("--k", type=int, default=16, help="k for kNN graph.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--do_gnn_benchmark",
        action="store_true",
        help="Run GNN inference benchmark on the test split (generated mode only).",
    )
    parser.add_argument("--benchmark_n", type=int, default=100)
    parser.add_argument(
        "--benchmark_out_dir",
        type=str,
        default=_default_results_dir(),
    )
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    excel_path = str(_repo_root() / "dataset" / "data.xlsx")

    # --- Dataset ---
    if args.dataset_source == "excel":
        dataset = KnapsackDataset(excel_path=excel_path)
    else:
        dataset = GeneratedKnapsack01Dataset(root_dir=args.generated_dir, k=args.k)

    # Use the shared split utility — no duplicated logic.
    train_set, val_set, test_set = split_dataset_by_instances(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    print(
        f"Dataset source: {args.dataset_source} | "
        f"total: {len(dataset)} | "
        f"train: {len(train_set)} | val: {len(val_set)} | test: {len(test_set)}"
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    in_dim = dataset[0].num_node_features if len(dataset) > 0 else 4
    model = KnapsackGNN(in_dim=in_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # --- Training loop ---
    train_start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        t0 = perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate_node_accuracy(model, val_loader, device)
        print(
            f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | time: {perf_counter()-t0:.2f}s"
        )
    print(f"Total training time: {perf_counter()-train_start:.2f}s")

    # --- Save checkpoint (consistent dict format) ---
    gnn_save_dir = _repo_root() / "results" / "GNN"
    gnn_save_dir.mkdir(parents=True, exist_ok=True)
    gnn_save_path = gnn_save_dir / "gnn.pt"
    save_checkpoint(model, gnn_save_path)          # <-- was: torch.save(model.state_dict(), ...)
    print(f"[GNN-TRAIN] Saved checkpoint to {gnn_save_path}")

    # --- Post-training evaluation ---
    if args.dataset_source == "excel":
        evaluate_gnn_vs_dp_on_excel(model, excel_path=excel_path)
    else:
        print("Skipping Excel comparison (generated dataset mode).")
        evaluate_instance_level(model, test_loader, num_cases=10, device=device)

        if args.do_gnn_benchmark:
            saved_paths = run_gnn_benchmark(
                model,
                test_loader,
                device,
                args.benchmark_out_dir,
                n_instances=args.benchmark_n,
                seed=args.seed,
            )
            if saved_paths:
                print("Benchmark outputs:")
                for key, path in saved_paths.items():
                    print(f"  {key}: {path}")


if __name__ == "__main__":
    main()