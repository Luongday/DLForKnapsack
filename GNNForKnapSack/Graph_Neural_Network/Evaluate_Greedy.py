"""Greedy baseline evaluation — CSV schema aligned with DP/GNN/GA.

Evaluates every instance_*.npz using value/weight ratio greedy,
then writes per-instance results to CSV for merge_results.py.

Usage:
    python Evaluate_Greedy.py
    python Evaluate_Greedy.py --dataset_dir data/knapsack_ilp/test
    python Evaluate_Greedy.py --dataset_dir data/knapsack_ilp/test --out_csv results/Greedy/greedy_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List

import numpy as np

import sys
_HERE = Path(__file__).resolve().parent
for _p in [str(_HERE), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ..instance_loader import load_instance, list_instances
from GNNForKnapSack.decode_utils import decode_to_solution_dict

def mark(msg: str) -> None:
    print(f"[GREEDY-EVAL] {msg}", flush=True)


def greedy_knapsack(
    weights: np.ndarray,
    values:  np.ndarray,
    capacity: int,
) -> np.ndarray:
    """Greedy 0/1 Knapsack: chọn items theo value/weight ratio giảm dần.

    Không dừng ở item đầu tiên không vừa — tiếp tục scan tìm item nhỏ hơn.
    Đảm bảo 100% feasibility.

    Returns:
        Binary int8 array, 1 = item được chọn.
    """
    n = len(weights)
    if n == 0:
        return np.array([], dtype=np.int8)

    ratios  = values.astype(float) / (weights.astype(float) + 1e-8)
    indices = np.argsort(ratios)[::-1]

    solution    = np.zeros(n, dtype=np.int8)
    remaining_w = float(capacity)

    for i in indices:
        w_i = float(weights[i])
        if w_i <= remaining_w + 1e-6:
            solution[i] = 1
            remaining_w -= w_i

    return solution


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

def _default_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "GNNForKnapSack" / "data" / "knapsack_ilp" / "test"


def _default_out_csv() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "Greedy" / "greedy_eval_results.csv"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Greedy baseline on NPZ Knapsack dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=Path, default=_default_dataset_dir())
    parser.add_argument("--out_csv",     type=Path, default=_default_out_csv())
    parser.add_argument("--n",           type=int,  default=None,
                        help="Limit to first N instances (default: all)")
    parser.add_argument("--quiet",       action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    files = list_instances(args.dataset_dir, limit=args.n)
    mark(f"Found {len(files)} instances in {args.dataset_dir}")

    results: List[dict] = []
    total_start = time.perf_counter()

    for idx, path in enumerate(files):
        W, V, C = load_instance(path)
        n = len(W)

        t0 = time.perf_counter()
        solution = greedy_knapsack(W, V, C)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        total_weight = int((W * solution).sum())
        total_value  = int((V * solution).sum())
        feasible     = 1 if total_weight <= C else 0
        selected_idx = [int(i) for i in range(n) if solution[i] == 1]

        results.append({
            "instance_file":     path.name,
            "n_items":           n,
            "capacity":          float(C),
            "total_weight":      float(total_weight),
            "total_value":       float(total_value),
            "feasible":          feasible,
            "inference_time_ms": round(solve_ms, 4),
            "selected_items":    json.dumps(selected_idx),
        })

        if verbose and ((idx + 1) % 50 == 0 or idx == 0):
            elapsed = time.perf_counter() - total_start
            mark(f"[{idx+1}/{len(files)}] elapsed={elapsed:.1f}s  "
                 f"val={total_value}  wt={total_weight}/{C}  time={solve_ms:.3f}ms")

    # Write CSV (same schema as DP and GNN)
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

    # Summary
    total_time = time.perf_counter() - total_start
    if results:
        avg_value = float(np.mean([r["total_value"]       for r in results]))
        avg_time  = float(np.mean([r["inference_time_ms"] for r in results]))
        feas_rate = float(np.mean([r["feasible"]          for r in results]))
        mark(f"Done: {len(results)} instances in {total_time:.1f}s")
        mark(f"Avg value={avg_value:.2f} | Avg time={avg_time:.4f}ms | Feasible={feas_rate:.3f}")
    mark(f"Results → {args.out_csv}")


if __name__ == "__main__":
    main()