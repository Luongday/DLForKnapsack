"""Centralized decoding utilities for Knapsack solvers — v3 Clean Edition.

ALL decode logic lives here.
This is the single source of truth for all decoding strategies.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import torch


DecodeStrategy = Literal["greedy_prob", "greedy_ratio", "beam_search"]


def greedy_feasible_decode(
    scores:   torch.Tensor,   # Có thể là probs hoặc ratios
    weights:  torch.Tensor,
    capacity: float,
    tol:      float = 1e-6,
) -> torch.Tensor:
    """Sort by scores (descending) and greedily add items while respecting capacity."""
    assert scores.dim() == 1 and weights.dim() == 1, \
        f"Expected 1D tensors, got scores={scores.shape}, weights={weights.shape}"
    assert scores.shape == weights.shape, \
        f"Shape mismatch: scores={scores.shape}, weights={weights.shape}"

    # Move to CPU for stable sorting and looping (fast enough for n<=1000)
    scores = scores.detach().cpu()
    weights = weights.detach().cpu()

    idx = torch.argsort(scores, descending=True)
    x_hat = torch.zeros_like(scores)
    total_weight = 0.0

    for i in idx:
        w = float(weights[i])
        if total_weight + w <= capacity + tol:
            x_hat[i] = 1.0
            total_weight += w

    return x_hat


def greedy_ratio_decode(
    values:   torch.Tensor,
    weights:  torch.Tensor,
    capacity: float,
    tol:      float = 1e-6,
) -> torch.Tensor:
    """Classic greedy by value/weight ratio."""
    ratios = values / (weights + 1e-8)
    return greedy_feasible_decode(ratios, weights, capacity, tol)


def beam_search_decode(
    probs:       torch.Tensor,
    weights:     torch.Tensor,
    capacity:    float,
    beam_width:  int = 5,
    tol:         float = 1e-6,
) -> torch.Tensor:
    """Beam search decoding — usually better quality than pure greedy."""
    probs = probs.detach().cpu()
    weights = weights.detach().cpu()
    n = len(probs)

    # Each beam: (total_value, total_weight, selected_mask)
    beams = [(0.0, 0.0, torch.zeros(n, dtype=torch.float32))]

    for i in range(n):
        new_beams = []
        p = float(probs[i])
        w = float(weights[i])

        for val, wt, mask in beams:
            # Không chọn item i
            new_beams.append((val, wt, mask.clone()))

            # Chọn item i (nếu feasible)
            if wt + w <= capacity + tol:
                new_mask = mask.clone()
                new_mask[i] = 1.0
                new_beams.append((val + p, wt + w, new_mask))

        # Giữ top beam_width beams
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

    # Chọn beam tốt nhất
    _, _, best_mask = beams[0]
    return best_mask


def decode(
    scores_or_probs: torch.Tensor,
    weights:         torch.Tensor,
    capacity:        float,
    strategy:        DecodeStrategy = "greedy_prob",
    values:          Optional[torch.Tensor] = None,
    beam_width:      int = 5,
    tol:             float = 1e-6,
) -> torch.Tensor:
    """Unified decode interface — recommended way to call decoder."""
    if strategy == "greedy_prob":
        return greedy_feasible_decode(scores_or_probs, weights, capacity, tol)
    elif strategy == "greedy_ratio":
        assert values is not None, "values must be provided for greedy_ratio"
        return greedy_ratio_decode(values, weights, capacity, tol)
    elif strategy == "beam_search":
        return beam_search_decode(scores_or_probs, weights, capacity, beam_width, tol)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def decode_to_solution_dict(
    x_hat:    torch.Tensor,
    weights:  torch.Tensor,
    values:   torch.Tensor,
    capacity: float,
    tol:      float = 1e-6,
) -> Dict:
    """Convert binary solution to structured result."""
    x_hat = x_hat.detach().cpu()
    weights = weights.detach().cpu()
    values = values.detach().cpu()

    total_weight = float((x_hat * weights).sum().item())
    total_value = float((x_hat * values).sum().item())
    feasible = total_weight <= capacity + tol
    selected = [int(i) for i in range(len(x_hat)) if x_hat[i] > 0.5]

    return {
        "total_value":    round(total_value, 4),
        "total_weight":   round(total_weight, 4),
        "feasible":       feasible,
        "selected_items": selected,
        "n_selected":     len(selected),
        "ratio_to_optimal": None,   # sẽ được điền sau khi có DP value
    }


def compute_ratio(solver_value: float, optimal_value: float) -> Optional[float]:
    """Approximation ratio: solver / optimal"""
    if optimal_value is None or optimal_value <= 0:
        return None
    return solver_value / optimal_value


def compute_gap(solver_value: float, optimal_value: float) -> Optional[float]:
    """Optimality gap: (optimal - solver) / optimal"""
    if optimal_value is None or optimal_value <= 0:
        return None
    return (optimal_value - solver_value) / optimal_value