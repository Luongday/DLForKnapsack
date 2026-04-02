"""Greedy baseline solver for 0/1 Knapsack.

Clean standalone solver — used by Evaluate_Greedy.py and as baseline
in EvaluateCallback.

Improvements vs original:
    - Cleaner interface
    - evaluate_greedy_on_dataset uses centralized instance_loader
    - Per-instance ratio then average (correct for CO literature)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def greedy_knapsack(
    values:   Sequence[float],
    weights:  Sequence[float],
    capacity: float,
) -> np.ndarray:
    """Greedy 0/1 Knapsack: sort by v/w ratio descending, pack greedily.

    Continues scanning past items that don't fit — picks smaller items
    that still fit. Guaranteed 100% feasibility.

    Returns:
        Binary int8 array, 1 = selected.
    """
    v = np.asarray(values,  dtype=float)
    w = np.asarray(weights, dtype=float)
    n = len(v)

    if n == 0:
        return np.array([], dtype=np.int8)

    ratios  = v / (w + 1e-8)
    indices = np.argsort(ratios)[::-1]

    solution    = np.zeros(n, dtype=np.int8)
    remaining_w = float(capacity)

    for i in indices:
        if w[i] <= remaining_w + 1e-6:
            solution[i] = 1
            remaining_w -= w[i]

    return solution


def greedy_knapsack_with_stats(
    values: Sequence[float], weights: Sequence[float], capacity: float,
) -> Tuple[np.ndarray, float, float]:
    """Greedy solve + return (solution, total_value, total_weight)."""
    sol = greedy_knapsack(values, weights, capacity)
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    return sol, float(np.dot(v, sol)), float(np.dot(w, sol))