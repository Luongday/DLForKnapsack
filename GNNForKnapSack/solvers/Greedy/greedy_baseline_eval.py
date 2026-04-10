"""Greedy baseline evaluation for 0/1 Knapsack."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

import sys
_HERE = Path(__file__).resolve().parent
for _p in [str(_HERE), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from GNNForKnapSack.instance_loader import load_instance, list_instances


def mark(msg: str) -> None:
    print(f"[GREEDY-EVAL] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_knapsack_greedy(
    weights: np.ndarray,
    values:  np.ndarray,
    capacity: int,
) -> List[int]:
    """Greedy 0/1 Knapsack: sort by value/weight ratio descending.

    Continues scanning past items that don't fit.
    Guaranteed 100% feasibility.

    Returns:
        List of selected item indices (sorted).
    """
    n = len(weights)
    if n == 0:
        return []

    ratios  = values.astype(float) / (weights.astype(float) + 1e-8)
    indices = np.argsort(ratios)[::-1]

    selected = []
    remaining = float(capacity)

    for i in indices:
        w_i = float(weights[i])
        if w_i <= remaining + 1e-6:
            selected.append(int(i))
            remaining -= w_i

    selected.sort()
    return selected


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

def _default_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "knapsack_ilp" / "test"

def _default_out_csv() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "Greedy" / "greedy_eval_results.csv"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    dataset_dir: Path,
    out_csv:     Path,
    n_limit:     Optional[int] = None,
    verbose:     bool = True,
) -> None:
    files = list_instances(dataset_dir, limit=n_limit)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mark(f"Dataset: {dataset_dir} ({len(files)} instances)")

    fieldnames = [
        "instance_file", "n_items", "capacity",
        "total_weight", "total_value",
        "feasible", "inference_time_ms", "selected_items",
    ]

    results = []
    total_start = time.perf_counter()

    for idx, path in enumerate(files):
        W, V, C = load_instance(path)

        t0 = time.perf_counter()
        selected_idx = solve_knapsack_greedy(W, V, int(C))
        t_ms = (time.perf_counter() - t0) * 1000.0

        if selected_idx:
            sel_arr = np.array(selected_idx, dtype=np.int32)
            total_weight = int(W[sel_arr].sum())
            total_value  = int(V[sel_arr].sum())
        else:
            total_weight = total_value = 0

        feasible = 1 if total_weight <= C else 0

        results.append({
            "instance_file":     path.name,
            "n_items":           int(W.shape[0]),
            "capacity":          int(C),
            "total_weight":      total_weight,
            "total_value":       total_value,
            "feasible":          feasible,
            "inference_time_ms": round(t_ms, 4),
            "selected_items":    json.dumps(selected_idx),
        })

        if verbose and ((idx + 1) % 10 == 0 or idx == 0):
            elapsed = time.perf_counter() - total_start
            mark(f"[{idx+1}/{len(files)}] elapsed={elapsed:.1f}s val={total_value} time={t_ms:.3f}ms")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total_time = time.perf_counter() - total_start
    if results:
        avg_val  = float(np.mean([r["total_value"]       for r in results]))
        avg_time = float(np.mean([r["inference_time_ms"] for r in results]))
        feas     = float(np.mean([r["feasible"]          for r in results]))
        mark(f"Done: {len(results)} in {total_time:.1f}s")
        mark(f"Avg value={avg_val:.2f} | time={avg_time:.4f}ms | feasible={feas:.3f}")
    mark(f"Results → {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Greedy baseline evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=Path, default=_default_dataset_dir())
    parser.add_argument("--out_csv",     type=Path, default=_default_out_csv())
    parser.add_argument("--n",           type=int,  default=None)
    parser.add_argument("--quiet",       action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        dataset_dir=args.dataset_dir, out_csv=args.out_csv,
        n_limit=args.n, verbose=not args.quiet,
    )