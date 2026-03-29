"""Greedy baseline solver for 0/1 Knapsack.

Ported from Greedy.py (original Neuro-Knapsack project).

Key changes vs original:
    - Solution is binary 0/1 array (not one-hot [n, 2]).
    - Fixed bug: original breaks immediately when capacity exceeded,
      skipping smaller items that could still fit. New version continues
      scanning remaining items — correct greedy behaviour.
    - Added value/weight return for convenience.
    - No dependency on Utils.py (self-contained).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def greedy_knapsack(
    values: Sequence[float],
    weights: Sequence[float],
    capacity: float,
) -> np.ndarray:
    """Greedy 0/1 Knapsack via value/weight ratio.

    Items are sorted by descending ratio. Items that fit within remaining
    capacity are selected. Unlike the original, this does NOT stop at the
    first infeasible item — it continues scanning for smaller items that
    may still fit (correct greedy).

    Args:
        values:   Item values  (length n).
        weights:  Item weights (length n).
        capacity: Knapsack capacity.

    Returns:
        Binary 0/1 numpy array of length n.
        1 = item selected, 0 = item not selected.
    """
    v = np.asarray(values,  dtype=float)
    w = np.asarray(weights, dtype=float)
    n = len(v)

    if n == 0:
        return np.array([], dtype=np.int8)

    # Sort by value/weight ratio descending
    ratios  = v / (w + 1e-8)
    indices = np.argsort(ratios)[::-1]  # descending

    solution    = np.zeros(n, dtype=np.int8)
    remaining_w = float(capacity)

    for i in indices:
        if w[i] <= remaining_w + 1e-6:
            solution[i] = 1
            remaining_w -= w[i]

    return solution


def greedy_knapsack_with_stats(
    values: Sequence[float],
    weights: Sequence[float],
    capacity: float,
) -> Tuple[np.ndarray, float, float]:
    """Greedy solve and return solution + total value + total weight.

    Returns:
        (solution, total_value, total_weight)
    """
    sol = greedy_knapsack(values, weights, capacity)
    v   = np.asarray(values,  dtype=float)
    w   = np.asarray(weights, dtype=float)
    return sol, float(np.dot(v, sol)), float(np.dot(w, sol))


def evaluate_greedy_on_dataset(
    dataset_dir: str,
    n_instances: int | None = None,
) -> dict:
    """Run greedy on all NPZ instances and return aggregate stats.

    Useful as a quick baseline before training GNN.

    Args:
        dataset_dir:  Directory with instance_*.npz files.
        n_instances:  Limit to first N (None = all).

    Returns:
        Dict with avg_value, avg_fill_ratio, feasibility_rate, avg_ratio_vs_dp.
    """
    from pathlib import Path
    import numpy as np

    files = sorted(Path(dataset_dir).glob("instance_*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files in {dataset_dir}")
    if n_instances:
        files = files[:n_instances]

    results = []
    for path in files:
        arr = np.load(path)
        w   = arr["weights"].astype(float)
        v   = arr["values"].astype(float)
        c   = float(arr["capacity"])
        dp_val = float(arr["dp_value"])

        sol, g_val, g_w = greedy_knapsack_with_stats(v, w, c)
        feasible = g_w <= c + 1e-6
        ratio    = g_val / dp_val if dp_val > 0 else 0.0

        results.append({
            "greedy_value":  g_val,
            "greedy_weight": g_w,
            "dp_value":      dp_val,
            "capacity":      c,
            "feasible":      feasible,
            "ratio_vs_dp":   ratio,
            "fill_ratio":    g_w / c if c > 0 else 0.0,
        })

    n = len(results)
    return {
        "n_instances":      n,
        "feasibility_rate": sum(r["feasible"] for r in results) / n,
        "avg_greedy_value": np.mean([r["greedy_value"]  for r in results]),
        "avg_dp_value":     np.mean([r["dp_value"]      for r in results]),
        "avg_ratio_vs_dp":  np.mean([r["ratio_vs_dp"]   for r in results]),
        "avg_fill_ratio":   np.mean([r["fill_ratio"]     for r in results]),
    }