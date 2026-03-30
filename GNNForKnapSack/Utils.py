"""Utility functions for Knapsack GNN project.

Ported from Utils.py (original Neuro-Knapsack project).

Key changes vs original:
    - All functions now work with binary 0/1 arrays (not one-hot encoding).
    - Removed scipy.misc.imsave (deprecated) → matplotlib.
    - Fixed IndentationError in beam_search_decoder (mixed tab/space).
    - Removed load_data (project uses NPZ format via dataset.py).
    - Added value_ratio() — main metric for comparing GNN vs optimal.
    - All functions are pure NumPy, no Keras/TF dependency.
"""

from __future__ import annotations

from math import log
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Feasibility & cost
# ---------------------------------------------------------------------------

def check_capacity(
    weights: Sequence[float],
    solution: np.ndarray,
    capacity: float,
) -> bool:
    """Return True if the binary solution respects capacity.

    Args:
        weights:  Item weights (list or 1-D array).
        solution: Binary 0/1 selection array of length n.
        capacity: Knapsack capacity.
    """
    total = float(np.dot(np.asarray(weights, dtype=float), np.asarray(solution, dtype=float)))
    return total <= capacity + 1e-6


def get_cost(
    values: Sequence[float],
    solution: np.ndarray,
) -> float:
    """Return total value of selected items.

    Args:
        values:   Item values (list or 1-D array).
        solution: Binary 0/1 selection array of length n.
    """
    return float(np.dot(np.asarray(values, dtype=float), np.asarray(solution, dtype=float)))


def get_weight(
    weights: Sequence[float],
    solution: np.ndarray,
) -> float:
    """Return total weight of selected items."""
    return float(np.dot(np.asarray(weights, dtype=float), np.asarray(solution, dtype=float)))


def value_ratio(
    gnn_value: float,
    optimal_value: float,
) -> float:
    """Approximation ratio: how close GNN is to optimal (1.0 = perfect).

    Returns gnn_value / optimal_value, clamped to [0, 1].
    Lower is worse, 1.0 is optimal.
    """
    if optimal_value <= 0:
        return 0.0
    return min(gnn_value / optimal_value, 1.0)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def solution_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    verbose: bool = False,
) -> Tuple[bool, int]:
    """Compare two binary solutions element-wise.

    Args:
        y_true: Ground-truth 0/1 array of length n.
        y_pred: Predicted  0/1 array of length n.
        verbose: Print per-item comparison.

    Returns:
        (exact_match, n_correct_variables)
        exact_match is True only if every element matches.
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    n = len(y_true)

    correct = int((y_true == y_pred).sum())
    if verbose:
        for i in range(n):
            status = "OK" if y_true[i] == y_pred[i] else "WRONG"
            print(f"  [{status}] item {i}: true={y_true[i]}, pred={y_pred[i]}")

    return (correct == n), correct


# ---------------------------------------------------------------------------
# Beam search decoder (for sequence models — optional use)
# ---------------------------------------------------------------------------

def beam_search_decoder(
    data: np.ndarray,
    k: int,
) -> List[Tuple[List[int], float]]:
    """Beam search over per-step probability distributions.

    Args:
        data: [n_steps, n_classes] probability matrix.
        k:    Beam width.
    Returns:
        List of (sequence, score) tuples, best first.
        Score is cumulative negative log-likelihood (lower = better).
    """
    sequences: List[Tuple[List[int], float]] = [([], 0.0)]

    for row in data:
        all_candidates = []
        for seq, score in sequences:
            for j, prob in enumerate(row):
                # Clip to avoid log(0)
                p = max(float(prob), 1e-12)
                all_candidates.append((seq + [j], score + (-log(p))))
        # Keep top-k by score (lower = better)
        sequences = sorted(all_candidates, key=lambda t: t[1])[:k]

    return sequences


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def save_solution_image(filename: str, solutions: np.ndarray) -> None:
    """Save a 2-D binary solution matrix as a BMP image.

    Replaces the original scipy.misc.imsave (deprecated).

    Args:
        filename: Output path without extension.
        solutions: [n_instances, n_items] binary array.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for save_solution_image().")

    path = Path(filename).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(4, solutions.shape[1] / 5), max(2, solutions.shape[0] / 20)))
    ax.imshow(solutions, aspect="auto", cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xlabel("Item index")
    ax.set_ylabel("Instance index")
    ax.set_title("Selection matrix (black = selected)")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Saved solution image to {path}")


def plot_selection_distribution(
    solutions: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Bar chart showing selection probability per item position.

    Args:
        solutions:  [n_instances, n_items] binary array.
        save_path:  If given, save figure to this path (PNG/PDF).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot_selection_distribution().")

    n_instances, n_items = solutions.shape
    ones  = solutions.mean(axis=0)
    zeros = 1.0 - ones
    idx   = np.arange(n_items)

    fig, ax = plt.subplots(figsize=(max(8, n_items // 3), 4))
    ax.bar(idx,        zeros, 0.4, alpha=0.6, color="steelblue", label="Not selected")
    ax.bar(idx + 0.4,  ones,  0.4, alpha=0.6, color="seagreen",  label="Selected")
    ax.set_xlabel("Item index")
    ax.set_ylabel("Fraction of instances")
    ax.set_title("Selection distribution across instances")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved distribution plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)
