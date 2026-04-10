"""Branch-and-Bound baseline evaluation for 0/1 Knapsack.

Implements the classic Horowitz-Sahni algorithm:
    1. Sort items by value/weight ratio (descending)
    2. DFS with LP relaxation upper bound pruning
    3. Timeout + max_nodes safety nets for hard instances

CSV schema aligned with DP/Greedy/GA/GNN for Merge_results.py.

B&B is exact (same result as DP) but:
    + Much faster on "easy" instances (uncorrelated) due to strong pruning
    + Much slower on hard instances (strongly correlated) — worst case O(2^n)
    → Useful as a comparison point showing where exact methods break down.

Usage:
    python bb_baseline_eval.py --dataset_dir data/knapsack_ilp/test
    python bb_baseline_eval.py --dataset_dir data/pisinger/type_03/test --timeout_sec 30
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in [str(_HERE), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from GNNForKnapSack.instance_loader import load_instance, list_instances

DEFAULT_MAX_NODES   = 2_000_000
DEFAULT_TIMEOUT_SEC = 60.0


def mark(msg: str) -> None:
    print(f"[BB-EVAL] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Branch-and-Bound core
# ---------------------------------------------------------------------------

def _upper_bound(
    v_sorted: List[float],
    w_sorted: List[float],
    capacity: float,
    level:    int,
    cur_val:  float,
    cur_w:    float,
) -> float:
    """LP relaxation upper bound: take full items greedily, then fractional of next.

    Since items are sorted by v/w descending, this is the optimal LP relaxation
    starting from `level`.
    """
    ub        = cur_val
    remaining = capacity - cur_w
    n = len(v_sorted)

    j = level
    while j < n and w_sorted[j] <= remaining:
        ub        += v_sorted[j]
        remaining -= w_sorted[j]
        j += 1

    # Fractional part of next item
    if j < n and remaining > 0:
        ub += v_sorted[j] * (remaining / w_sorted[j])

    return ub


def solve_knapsack_bb(
    weights:      np.ndarray,
    values:       np.ndarray,
    capacity:     int,
    max_nodes:    int   = DEFAULT_MAX_NODES,
    timeout_sec:  Optional[float] = DEFAULT_TIMEOUT_SEC,
) -> Tuple[List[int], bool, int]:
    """Branch-and-Bound for 0/1 Knapsack.

    Returns:
        selected:   List of selected item indices (sorted, in ORIGINAL order)
        optimal:    True if search finished, False if timeout/max_nodes hit
        n_nodes:    Number of B&B nodes explored
    """
    n = len(weights)
    if n == 0:
        return [], True, 0

    # Sort indices by v/w ratio descending
    w_arr = weights.astype(float)
    v_arr = values.astype(float)
    ratios = v_arr / np.maximum(w_arr, 1e-9)
    order  = np.argsort(-ratios)  # descending

    w_sorted = [float(w_arr[i]) for i in order]
    v_sorted = [float(v_arr[i]) for i in order]

    # Increase recursion limit for large n
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, n + 500))

    # Mutable state captured by closure
    state = {
        "best_value":    0.0,
        "best_sol":      [0] * n,   # in sorted order
        "current_sol":   [0] * n,
        "nodes":         0,
        "timed_out":     False,
        "start_time":    time.perf_counter(),
    }

    cap_float = float(capacity)

    def branch(level: int, cur_val: float, cur_w: float) -> None:
        # Termination checks
        state["nodes"] += 1
        if state["nodes"] > max_nodes:
            state["timed_out"] = True
            return
        if timeout_sec is not None and \
           (time.perf_counter() - state["start_time"]) > timeout_sec:
            state["timed_out"] = True
            return

        if cur_w > cap_float + 1e-9:
            return

        if level == n:
            if cur_val > state["best_value"]:
                state["best_value"] = cur_val
                state["best_sol"]   = state["current_sol"].copy()
            return

        # Prune: if LP upper bound ≤ best known, skip
        ub = _upper_bound(v_sorted, w_sorted, cap_float, level, cur_val, cur_w)
        if ub <= state["best_value"] + 1e-9:
            return

        # Branch 1: take item (only if feasible)
        w_i = w_sorted[level]
        v_i = v_sorted[level]
        if cur_w + w_i <= cap_float + 1e-9:
            state["current_sol"][level] = 1
            branch(level + 1, cur_val + v_i, cur_w + w_i)
            if state["timed_out"]:
                return
            state["current_sol"][level] = 0

        # Branch 2: skip item
        branch(level + 1, cur_val, cur_w)

    try:
        branch(0, 0.0, 0.0)
    finally:
        sys.setrecursionlimit(old_limit)

    # Map sorted solution back to original indices
    selected = [int(order[i]) for i in range(n) if state["best_sol"][i]]
    selected.sort()

    optimal = not state["timed_out"]
    return selected, optimal, state["nodes"]


# ---------------------------------------------------------------------------
# Eval pipeline
# ---------------------------------------------------------------------------

def _default_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "knapsack_ilp" / "test"

def _default_out_csv() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "BB" / "bb_results.csv"


def evaluate(
    dataset_dir: Path,
    out_csv:     Path,
    n_limit:     Optional[int] = None,
    verbose:     bool          = True,
    max_nodes:   int           = DEFAULT_MAX_NODES,
    timeout_sec: float         = DEFAULT_TIMEOUT_SEC,
) -> None:
    files = list_instances(dataset_dir, limit=n_limit)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mark(f"Dataset: {dataset_dir} ({len(files)} instances)")
    mark(f"Max nodes: {max_nodes:,} | Timeout: {timeout_sec}s")

    fieldnames = [
        "instance_file", "n_items", "capacity",
        "total_weight", "total_value",
        "feasible", "inference_time_ms", "selected_items",
    ]

    results     = []
    skipped     = 0
    timeouts    = 0
    total_nodes = 0
    total_start = time.perf_counter()

    for idx, path in enumerate(files):
        try:
            W, V, C = load_instance(path)
        except (KeyError, ValueError) as e:
            mark(f"[SKIP] {path.name}: {e}")
            skipped += 1
            continue

        t0 = time.perf_counter()
        selected_idx, optimal, n_nodes = solve_knapsack_bb(
            W, V, int(C),
            max_nodes=max_nodes,
            timeout_sec=timeout_sec,
        )
        t_ms = (time.perf_counter() - t0) * 1000.0

        total_nodes += n_nodes
        if not optimal:
            timeouts += 1

        if selected_idx:
            sel_arr      = np.array(selected_idx, dtype=np.int32)
            total_weight = int(W[sel_arr].sum())
            total_value  = int(V[sel_arr].sum())
        else:
            total_weight = 0
            total_value  = 0

        feasible = 1 if total_weight <= int(C) else 0

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
            opt_str = "OPT" if optimal else "TIMEOUT"
            mark(f"[{idx+1}/{len(files)}] elapsed={elapsed:.1f}s "
                 f"val={total_value} time={t_ms:.2f}ms "
                 f"nodes={n_nodes} [{opt_str}]")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total_time = time.perf_counter() - total_start
    if results:
        avg_val   = float(np.mean([r["total_value"]       for r in results]))
        avg_time  = float(np.mean([r["inference_time_ms"] for r in results]))
        feas      = float(np.mean([r["feasible"]          for r in results]))
        avg_nodes = total_nodes / max(len(results), 1)

        mark(f"Done: {len(results)} in {total_time:.1f}s "
             f"(skipped={skipped}, timeouts={timeouts})")
        mark(f"Avg value={avg_val:.2f} | time={avg_time:.3f}ms "
             f"| nodes={avg_nodes:.0f} | feasible={feas:.3f}")

        if timeouts > 0:
            mark(f"WARNING: {timeouts}/{len(results)} instances hit timeout — "
                 f"results may be suboptimal. Consider --timeout_sec larger.")

    mark(f"Results → {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Branch-and-Bound baseline evaluation for 0/1 Knapsack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir",  type=Path,  default=_default_dataset_dir())
    parser.add_argument("--out_csv",      type=Path,  default=_default_out_csv())
    parser.add_argument("--n",            type=int,   default=None,
                        help="Limit to first N instances")
    parser.add_argument("--max_nodes",    type=int,   default=DEFAULT_MAX_NODES,
                        help="Max B&B nodes to explore per instance")
    parser.add_argument("--timeout_sec",  type=float, default=DEFAULT_TIMEOUT_SEC,
                        help="Timeout per instance in seconds")
    parser.add_argument("--quiet",        action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        dataset_dir=args.dataset_dir,
        out_csv=args.out_csv,
        n_limit=args.n,
        verbose=not args.quiet,
        max_nodes=args.max_nodes,
        timeout_sec=args.timeout_sec,
    )