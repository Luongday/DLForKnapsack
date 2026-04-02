"""Centralized decoding utilities for Knapsack solvers.

ALL decode logic lives here — no more 6 copies across the codebase.
Every file that needs greedy_feasible_decode imports from this module.

Functions:
    greedy_feasible_decode  — probability-guided greedy (for GNN output)
    greedy_ratio_decode     — value/weight ratio greedy (classic heuristic)
    decode_to_solution_dict — convenience wrapper returning structured result
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


def greedy_feasible_decode(
    probs:    torch.Tensor,
    weights:  torch.Tensor,
    capacity: float,
    tol:      float = 1e-6,
) -> torch.Tensor:
    """Chọn items theo xác suất giảm dần, đảm bảo feasibility.

    Dùng cho output của GNN/RL — model cho probability per item,
    decode chọn items theo thứ tự prob giảm dần mà không vượt capacity.

    Args:
        probs:    1-D tensor [n_items] — probability mỗi item được chọn.
        weights:  1-D tensor [n_items] — ORIGINAL weights (không normalized).
        capacity: Sức chứa túi (original, không normalized).
        tol:      Tolerance cho floating-point comparison.

    Returns:
        x_hat: binary 1-D tensor [n_items], 1 = item được chọn.
    """
    assert probs.dim() == 1 and weights.dim() == 1, \
        f"Expected 1-D tensors, got probs={probs.shape}, weights={weights.shape}"
    assert probs.shape == weights.shape, \
        f"Shape mismatch: probs={probs.shape}, weights={weights.shape}"

    idx   = torch.argsort(probs, descending=True)
    x_hat = torch.zeros_like(probs)
    total = 0.0

    for i in idx:
        w_i = weights[i].item()
        if total + w_i <= capacity + tol:
            x_hat[i] = 1.0
            total    += w_i

    return x_hat


def greedy_ratio_decode(
    values:   torch.Tensor,
    weights:  torch.Tensor,
    capacity: float,
    tol:      float = 1e-6,
) -> torch.Tensor:
    """Classic greedy: chọn items theo value/weight ratio giảm dần.

    Không phụ thuộc vào model prediction — dùng làm baseline.

    Args:
        values:   1-D tensor [n_items].
        weights:  1-D tensor [n_items].
        capacity: Sức chứa túi.
        tol:      Tolerance.

    Returns:
        x_hat: binary 1-D tensor [n_items].
    """
    ratios = values / (weights + 1e-8)
    return greedy_feasible_decode(ratios, weights, capacity, tol)


def decode_to_solution_dict(
    x_hat:    torch.Tensor,
    weights:  torch.Tensor,
    values:   torch.Tensor,
    capacity: float,
    tol:      float = 1e-6,
) -> Dict:
    """Convert binary selection tensor to structured result dict.

    Args:
        x_hat:    Binary 1-D tensor [n_items].
        weights:  1-D tensor [n_items].
        values:   1-D tensor [n_items].
        capacity: Knapsack capacity.

    Returns:
        Dict with total_value, total_weight, feasible, selected_items, n_selected.
    """
    total_weight = float((x_hat * weights).sum().item())
    total_value  = float((x_hat * values).sum().item())
    feasible     = total_weight <= capacity + tol
    selected     = [int(i) for i in range(len(x_hat)) if x_hat[i] > 0.5]

    return {
        "total_value":    total_value,
        "total_weight":   total_weight,
        "feasible":       feasible,
        "selected_items": selected,
        "n_selected":     len(selected),
    }


def compute_ratio(solver_value: float, optimal_value: float) -> Optional[float]:
    """Approximation ratio: solver_value / optimal_value. 1.0 = perfect."""
    if optimal_value is None or optimal_value <= 0:
        return None
    return solver_value / optimal_value


def compute_gap(solver_value: float, optimal_value: float) -> Optional[float]:
    """Optimality gap: (optimal - solver) / optimal. 0.0 = perfect."""
    if optimal_value is None or optimal_value <= 0:
        return None
    return (optimal_value - solver_value) / optimal_value