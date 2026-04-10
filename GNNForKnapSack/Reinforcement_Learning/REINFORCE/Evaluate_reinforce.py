"""REINFORCE (GNN policy gradient) evaluation — CSV schema aligned with all solvers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for _p in [str(_HERE), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from GNNForKnapSack.instance_loader import load_instance, list_instances
from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.Graph_builder import build_knapsack_graph_inference
from GNNForKnapSack.decode_utils import greedy_feasible_decode
from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.model import load_checkpoint

def mark(msg: str) -> None:
    print(f"[REINFORCE-EVAL] {msg}", flush=True)


@torch.no_grad()
def run_reinforce_on_instance(
    model,
    device: torch.device,
    weights: np.ndarray,
    values:  np.ndarray,
    capacity: int,
    k: int = 16,
) -> dict:
    """Run REINFORCE-trained GNN on one instance with greedy decode."""
    n = len(weights)

    # Build inference graph (no labels)
    data = build_knapsack_graph_inference(
        weights=weights.tolist(),
        values=values.tolist(),
        capacity=int(capacity),
        k=min(k, max(n - 1, 1)),
    )
    data = data.to(device)

    t0 = time.perf_counter()
    logits = model(data)

    # Safety: ensure logits match graph size
    if logits.numel() != n:
        raise ValueError(
            f"Model output {logits.numel()} != n_items {n}. "
            f"Check feature dimensions and graph construction."
        )

    probs = torch.sigmoid(logits).cpu()
    w_tensor = torch.tensor(weights, dtype=torch.float32)
    v_tensor = torch.tensor(values,  dtype=torch.float32)

    # Greedy feasible decode
    x_hat = greedy_feasible_decode(probs, w_tensor, float(capacity))
    infer_ms = (time.perf_counter() - t0) * 1000.0

    total_weight = float((x_hat * w_tensor).sum().item())
    total_value  = float((x_hat * v_tensor).sum().item())
    feasible     = 1 if total_weight <= float(capacity) + 1e-6 else 0
    selected_idx = [int(i) for i in range(n) if x_hat[i].item() > 0.5]

    return {
        "total_weight":      total_weight,
        "total_value":       total_value,
        "feasible":          feasible,
        "inference_time_ms": round(infer_ms, 4),
        "selected_items":    selected_idx,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate REINFORCE-trained GNN on Knapsack instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--model_path",  type=Path,
                        default=Path(__file__).resolve().parents[1]/ "results" / "GNN_REINFORCE" / "gnn_reinforce.pt")
    parser.add_argument("--out_csv",     type=Path,
                        default=Path("results/GNN_REINFORCE/reinforce_eval_results.csv"))
    parser.add_argument("--device",      type=str, default="cpu")
    parser.add_argument("--k",           type=int, default=16)
    parser.add_argument("--n",           type=int, default=None,
                        help="Limit to first N instances")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    mark(f"Loading model: {args.model_path}")
    model = load_checkpoint(args.model_path, device=device, dropout=0.0)
    model.eval()

    files = list_instances(args.dataset_dir, limit=args.n)
    mark(f"Found {len(files)} instances in {args.dataset_dir}")

    results = []
    total_start = time.perf_counter()

    for idx, path in enumerate(files):
        W, V, C = load_instance(path)
        sol = run_reinforce_on_instance(model, device, W, V, int(C), k=args.k)

        results.append({
            "instance_file":     path.name,
            "n_items":           len(W),
            "capacity":          float(C),
            "total_weight":      sol["total_weight"],
            "total_value":       sol["total_value"],
            "feasible":          sol["feasible"],
            "inference_time_ms": sol["inference_time_ms"],
            "selected_items":    json.dumps(sol["selected_items"]),
        })

        if (idx + 1) % 50 == 0:
            mark(f"[{idx+1}/{len(files)}] val={sol['total_value']:.1f} "
                 f"feasible={bool(sol['feasible'])}")

    fieldnames = [
        "instance_file", "n_items", "capacity",
        "total_weight", "total_value",
        "feasible", "inference_time_ms", "selected_items",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total_time = time.perf_counter() - total_start
    if results:
        avg_val  = float(np.mean([r["total_value"]       for r in results]))
        avg_time = float(np.mean([r["inference_time_ms"] for r in results]))
        feas     = float(np.mean([r["feasible"]          for r in results]))
        mark(f"Done: {len(results)} in {total_time:.1f}s")
        mark(f"avg_value={avg_val:.2f} | avg_time={avg_time:.3f}ms | feasible={feas:.3f}")
    mark(f"Results → {args.out_csv}")


if __name__ == "__main__":
    main()