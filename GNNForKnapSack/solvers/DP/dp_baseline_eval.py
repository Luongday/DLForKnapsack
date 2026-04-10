"""DP baseline evaluation for 0/1 Knapsack."""

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

DEFAULT_MAX_CAPACITY = 100_000


def mark(msg: str) -> None:
    print(f"[DP-EVAL] {msg}", flush=True)


def solve_knapsack_dp(
    weights: np.ndarray, values: np.ndarray, capacity: int,
) -> List[int]:
    """0/1 Knapsack DP with numpy bool backtracking array."""
    n  = len(weights)
    dp = np.zeros(capacity + 1, dtype=np.int64)
    choice = np.zeros((n, capacity + 1), dtype=bool)

    for i in range(n):
        w_i = int(weights[i])
        v_i = int(values[i])
        for w in range(capacity, w_i - 1, -1):
            take = dp[w - w_i] + v_i
            if take > dp[w]:
                dp[w] = take
                choice[i, w] = True

    selected: List[int] = []
    w = capacity
    for i in range(n - 1, -1, -1):
        if choice[i, w]:
            selected.append(i)
            w -= int(weights[i])
    selected.sort()
    return selected


def _default_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "knapsack_ilp" / "test"

def _default_out_csv() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "DP" / "dp_results.csv"


def evaluate(
    dataset_dir: Path, out_csv: Path,
    n_limit: Optional[int] = None,
    verbose: bool = True,
    max_capacity: int = DEFAULT_MAX_CAPACITY,
) -> None:
    files = list_instances(dataset_dir, limit=n_limit)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mark(f"Dataset: {dataset_dir} ({len(files)} instances)")
    mark(f"Max capacity: {max_capacity:,}")

    fieldnames = [
        "instance_file", "n_items", "capacity",
        "total_weight", "total_value",
        "feasible", "inference_time_ms", "selected_items",
    ]

    results = []
    skipped = 0
    total_start = time.perf_counter()

    for idx, path in enumerate(files):
        try:
            W, V, C = load_instance(path)
        except (KeyError, ValueError) as e:
            mark(f"[SKIP] {path.name}: {e}")
            skipped += 1
            continue

        if C > max_capacity:
            mark(f"[SKIP] {path.name}: capacity={C} > max={max_capacity}")
            skipped += 1
            continue

        t0 = time.perf_counter()
        selected_idx = solve_knapsack_dp(W, V, C)
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

        if verbose and ((idx + 1) % 50 == 0 or idx == 0):
            elapsed = time.perf_counter() - total_start
            mark(f"[{idx+1}/{len(files)}] elapsed={elapsed:.1f}s val={total_value} time={t_ms:.2f}ms")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total_time = time.perf_counter() - total_start
    if results:
        avg_val  = float(np.mean([r["total_value"]       for r in results]))
        avg_time = float(np.mean([r["inference_time_ms"] for r in results]))
        feas     = float(np.mean([r["feasible"]          for r in results]))
        mark(f"Done: {len(results)} in {total_time:.1f}s (skipped={skipped})")
        mark(f"Avg value={avg_val:.2f} | time={avg_time:.3f}ms | feasible={feas:.3f}")
    mark(f"Results → {out_csv}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="DP baseline evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir",  type=Path, default=_default_dataset_dir())
    parser.add_argument("--out_csv",      type=Path, default=_default_out_csv())
    parser.add_argument("--n",            type=int,  default=None)
    parser.add_argument("--max_capacity", type=int,  default=DEFAULT_MAX_CAPACITY)
    parser.add_argument("--quiet",        action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(
        dataset_dir=args.dataset_dir, out_csv=args.out_csv,
        n_limit=args.n, verbose=not args.quiet,
        max_capacity=args.max_capacity,
    )