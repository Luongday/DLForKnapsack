"""Per-instance GNN evaluation — aligned with DP/Greedy/GA CSV schema.

Improvements vs original:
    - Uses centralized decode_utils (single source of truth)
    - Uses centralized instance_loader
    - Explicit OOR guard: asserts probs length == n_items before decode
    - model.eval() + dropout=0.0 for deterministic inference
    - build_knapsack_graph_inference for clean inference graph (no solution)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List

import sys
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for _p in [str(_HERE), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.model import load_checkpoint
from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.Graph_builder import build_knapsack_graph_inference
from GNNForKnapSack.decode_utils import greedy_feasible_decode
from GNNForKnapSack.instance_loader import load_instance, list_instances


def mark(msg: str) -> None:
    print(f"[GNN-EVAL] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

def _default_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "GNNForKnapSack" / "data" / "knapsack_ilp" / "test"


def _default_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "GNN" / "gnn.pt"


def _default_out_csv() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "GNN" / "gnn_eval_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained KnapsackGNN on NPZ instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=Path, default=_default_dataset_dir())
    parser.add_argument("--model_path",  type=Path, default=_default_model_path())
    parser.add_argument("--out_csv",     type=Path, default=_default_out_csv())
    parser.add_argument("--n",           type=int,  default=None,
                        help="Limit to first N instances")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model with dropout=0.0 for deterministic inference
    model = load_checkpoint(args.model_path, device=device, dropout=0.0)
    model.eval()

    files = list_instances(args.dataset_dir, limit=args.n)
    mark(f"Found {len(files)} instances | model on {device}")
    mark(f"Model: {model.__class__.__name__} "
         f"in_dim={getattr(model, 'in_dim', '?')} "
         f"hidden={getattr(model, 'hidden_dim', '?')} "
         f"layers={getattr(model, 'num_layers', '?')}")

    results: List[dict] = []

    for idx, path in enumerate(files):
        W, V, C = load_instance(path)
        n = len(W)

        graph = build_knapsack_graph_inference(W.tolist(), V.tolist(), int(C)).to(device)

        # Keep original weights on CPU for decode (not normalized)
        W_orig = torch.tensor(W, dtype=torch.float32, device="cpu")
        V_orig = torch.tensor(V, dtype=torch.float32, device="cpu")

        t0     = time.perf_counter()
        logits = model(graph)
        probs  = torch.sigmoid(logits).detach().cpu()

        # FIX: Explicit OOR guard — only take first n probs
        # This prevents the bug where logits might be longer than n_items
        if probs.shape[0] > n:
            mark(f"[WARN] {path.name}: logits has {probs.shape[0]} nodes but n_items={n}. Slicing to n.")
            probs = probs[:n]
        elif probs.shape[0] < n:
            mark(f"[ERROR] {path.name}: logits has {probs.shape[0]} nodes but n_items={n}. Skipping.")
            continue

        assert probs.shape[0] == n, \
            f"probs length {probs.shape[0]} != n_items {n}"

        # Centralized decode — guaranteed feasible, guaranteed in-range
        x_hat    = greedy_feasible_decode(probs, W_orig, float(C))
        infer_ms = (time.perf_counter() - t0) * 1000.0

        total_weight = float((x_hat * W_orig).sum())
        total_value  = float((x_hat * V_orig).sum())
        feasible     = 1 if total_weight <= float(C) + 1e-6 else 0
        selected_idx = [int(i) for i, val in enumerate(x_hat.tolist()) if val == 1.0]

        # Sanity check: all selected indices must be in [0, n)
        assert all(0 <= i < n for i in selected_idx), \
            f"OOR items in {path.name}: {[i for i in selected_idx if i >= n]}"

        results.append({
            "instance_file":     path.name,
            "n_items":           n,
            "capacity":          float(C),
            "total_weight":      round(total_weight, 4),
            "total_value":       round(total_value,  4),
            "feasible":          feasible,
            "inference_time_ms": round(infer_ms, 4),
            "selected_items":    json.dumps(selected_idx),
        })

        if (idx + 1) % 50 == 0:
            mark(f"[{idx+1}/{len(files)}] val={total_value:.1f} "
                 f"wt={total_weight:.1f}/{C} feasible={bool(feasible)}")

    # Summary
    if results:
        avg_value = float(np.mean([r["total_value"]       for r in results]))
        feas_rate = float(np.mean([r["feasible"]          for r in results]))
        avg_time  = float(np.mean([r["inference_time_ms"] for r in results]))
        mark(f"avg_value={avg_value:.2f} | feasibility={feas_rate:.3f} | avg_time={avg_time:.2f}ms")

    # Write CSV (same schema as DP/Greedy/GA)
    fieldnames = [
        "instance_file", "n_items", "capacity",
        "total_weight", "total_value", "feasible",
        "inference_time_ms", "selected_items",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    mark(f"Wrote {len(results)} rows → {args.out_csv}")


if __name__ == "__main__":
    main()