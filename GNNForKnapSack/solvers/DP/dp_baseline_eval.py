"""DP baseline evaluation for 0/1 Knapsack.

Evaluates every instance_*.npz in a dataset directory using the 1-D
rolling DP solver, then writes per-instance results to CSV — schema
aligned with gnn_eval_results.csv and dqn eval_results.csv so that
merge_results.py can join them directly.

Usage:
    python dp_baseline_eval.py --dataset_dir data/knapsack_ilp/train
    python dp_baseline_eval.py --dataset_dir data/knapsack_ilp/test \\
                               --out_csv results/DP/dp_results_test.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mark(msg: str) -> None:
    print(f"[DP-EVAL] {msg}", flush=True)


def _default_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "knapsack_ilp" / "train"


def _default_out_csv() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "DP" / "dp_results.csv"


# ---------------------------------------------------------------------------
# Instance loader
# ---------------------------------------------------------------------------

def load_instance(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load weights, values, capacity from an NPZ file.

    Tries multiple key aliases so the loader works with both
    data_generate_01.py and datagen.py output formats.
    """
    arr = np.load(npz_path, allow_pickle=True)

    def pick(keys):
        for k in keys:
            if k in arr.files:
                return arr[k]
        return None

    W = pick(["weights", "w", "W"])
    V = pick(["values",  "v", "V"])
    C = pick(["capacity", "cap", "C"])

    if W is None or V is None or C is None:
        raise KeyError(
            f"Missing weights/values/capacity in {npz_path.name}, "
            f"found keys={arr.files}"
        )

    W = np.asarray(W).astype(np.int32).reshape(-1)
    V = np.asarray(V).astype(np.int32).reshape(-1)
    C = int(np.asarray(C).reshape(()))

    if W.shape != V.shape:
        raise ValueError(
            f"weights/values shape mismatch in {npz_path.name}: "
            f"{W.shape} vs {V.shape}"
        )
    return W, V, C


# ---------------------------------------------------------------------------
# DP solver  (1-D rolling array — O(n·W) time, O(W) space)
# ---------------------------------------------------------------------------

def solve_knapsack_dp(
    weights: np.ndarray,
    values:  np.ndarray,
    capacity: int,
) -> List[int]:
    """0/1 Knapsack via 1-D rolling DP with backtracking.

    Returns a list of selected item indices (0-based), sorted ascending.
    Always feasible by construction — DP never violates capacity.
    """
    n   = len(weights)
    dp  = [0] * (capacity + 1)
    # choice[i][w] = True means item i was taken when remaining cap = w
    choice = [[False] * (capacity + 1) for _ in range(n)]

    for i in range(n):
        w_i = int(weights[i])
        v_i = int(values[i])
        # Iterate right-to-left to avoid using item i twice
        for w in range(capacity, w_i - 1, -1):
            take = dp[w - w_i] + v_i
            if take > dp[w]:
                dp[w] = take
                choice[i][w] = True

    # Backtrack to recover selected items
    selected: List[int] = []
    w = capacity
    for i in range(n - 1, -1, -1):
        if choice[i][w]:
            selected.append(i)
            w -= int(weights[i])
    selected.sort()
    return selected


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    dataset_dir: Path,
    out_csv:     Path,
    n_limit:     int | None = None,
    verbose:     bool = True,
) -> None:
    files = sorted(dataset_dir.glob("instance_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No instance_*.npz files found in {dataset_dir}"
        )
    if n_limit:
        files = files[:n_limit]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mark(f"Dataset : {dataset_dir}  ({len(files)} instances)")
    mark(f"Output  : {out_csv}")

    fieldnames = [
        "instance_file", "n_items", "capacity",
        "total_weight", "total_value",
        "feasible", "inference_time_ms", "selected_items",
    ]

    results = []
    total_start = time.perf_counter()

    for idx, path in enumerate(files):
        try:
            W, V, C = load_instance(path)
        except (KeyError, ValueError) as e:
            mark(f"[SKIP] {path.name}: {e}")
            continue

        t0 = time.perf_counter()
        selected_idx = solve_knapsack_dp(W, V, C)
        t_ms = (time.perf_counter() - t0) * 1000.0

        total_weight = int(W[selected_idx].sum()) if selected_idx else 0
        total_value  = int(V[selected_idx].sum()) if selected_idx else 0

        # Sanity check — DP should always be feasible
        feasible = 1 if total_weight <= C else 0
        if not feasible:
            mark(f"[WARN] {path.name}: DP solution exceeded capacity — bug!")

        results.append({
            "instance_file":    path.name,
            "n_items":          int(W.shape[0]),
            "capacity":         int(C),
            "total_weight":     total_weight,
            "total_value":      total_value,
            "feasible":         feasible,
            "inference_time_ms": round(t_ms, 4),
            "selected_items":   json.dumps(selected_idx),
        })

        if verbose and ((idx + 1) % 100 == 0 or idx == 0):
            elapsed = time.perf_counter() - total_start
            mark(f"[{idx+1}/{len(files)}] elapsed={elapsed:.1f}s  "
                 f"last: value={total_value}  time={t_ms:.2f}ms")

    # Write CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
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
        mark(f"Avg value={avg_value:.2f} | "
             f"Avg time={avg_time:.3f}ms | "
             f"Feasible={feas_rate:.3f}")
    mark(f"Results written to {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DP baseline on NPZ Knapsack dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir", type=Path,
        default=_default_dataset_dir(),
        help="Directory containing instance_*.npz files",
    )
    parser.add_argument(
        "--out_csv", type=Path,
        default=_default_out_csv(),
        help="Output CSV path for per-instance results",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Limit evaluation to first N instances (default: all)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        dataset_dir=args.dataset_dir,
        out_csv=args.out_csv,
        n_limit=args.n,
        verbose=not args.quiet,
    )