"""Knapsack graph builder — v2 with adaptive k and improved features."""

from __future__ import annotations

import math
from typing import List

import torch
from torch_geometric.data import Data

DEFAULT_K = 16  # Kept for backward compat but adaptive_k is preferred
IN_DIM    = 7   # Increased from 6


def adaptive_k(n: int, base_k: int = 16) -> int:
    """Adaptive kNN size: grows with sqrt(n).

    n=10  → min(9, max(9, 8))  = 9   (fully connected since n-1=9)
    n=50  → max(9, 11)         = 11
    n=100 → max(9, 16)         = 16
    n=200 → max(9, 22)         = 22
    n=500 → max(9, 35)         = 35
    n=1000→ max(9, 50)         = 50
    """
    k_adaptive = max(base_k // 2, int(math.sqrt(n) * 1.6))
    return min(k_adaptive, max(1, n - 1))


def _build_knn_edges(x: torch.Tensor, k: int) -> torch.Tensor:
    """Build kNN edges based on Euclidean distance in feature space."""
    n = x.size(0)
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    k_eff = min(k, n - 1)
    dist  = torch.cdist(x, x, p=2)
    dist.fill_diagonal_(float("inf"))
    knn_idx = dist.topk(k_eff, largest=False).indices  # [n, k_eff]

    row = torch.arange(n).unsqueeze(1).expand(-1, k_eff).reshape(-1)
    col = knn_idx.reshape(-1)
    return torch.stack([row, col], dim=0)  # [2, n*k_eff]


def build_knapsack_graph(
    weights:  List[int],
    values:   List[int],
    capacity: int,
    solution: List[int],
    k:        int = DEFAULT_K,
) -> Data:
    """Build PyG graph for a Knapsack instance with DP-optimal labels.

    Node features (7-dim, normalized):
        0: w_norm     — weight_i / max(weights)
        1: v_norm     — value_i / max(values)
        2: ratio_norm — (v_i/w_i) / max(v/w)
        3: cap_ratio  — weight_i / capacity
        4: cap_util   — capacity / sum(weights) (global)
        5: item_frac  — 1 / n_items (global)
        6: w_vs_mean  — weight_i / mean(weights)  — NEW: outlier signal

    Args:
        k: If set, use fixed k. Otherwise adaptive based on n.
    """
    w   = torch.tensor(weights,  dtype=torch.float32)
    v   = torch.tensor(values,   dtype=torch.float32)
    sol = torch.tensor(solution, dtype=torch.float32)
    n   = len(weights)

    # Per-item normalized features
    ratio      = v / (w + 1e-8)
    w_norm     = w     / (w.max()     + 1e-8)
    v_norm     = v     / (v.max()     + 1e-8)
    ratio_norm = ratio / (ratio.max() + 1e-8)

    cap_ratio = w / (float(capacity) + 1e-8)

    # Global features
    w_sum     = w.sum().item() + 1e-8
    w_mean    = w_sum / max(n, 1)
    cap_util  = torch.full((n,), float(capacity) / w_sum)
    item_frac = torch.full((n,), 1.0 / max(n, 1))

    # NEW: weight relative to mean — helps identify outliers
    w_vs_mean = w / w_mean

    # 7-dim feature vector
    x = torch.stack([
        w_norm, v_norm, ratio_norm,
        cap_ratio, cap_util, item_frac,
        w_vs_mean,
    ], dim=1)

    # Adaptive k based on graph size
    k_used = adaptive_k(n, base_k=k)
    edge_index = _build_knn_edges(x, k=k_used)

    y = sol.unsqueeze(1)  # [n, 1]

    return Data(
        x          = x,
        edge_index = edge_index,
        y          = y,
        wts        = w,
        vals       = v,
        cap        = torch.tensor([capacity], dtype=torch.float32),
    )


def build_knapsack_graph_inference(
    weights:  List[int],
    values:   List[int],
    capacity: int,
    k:        int = DEFAULT_K,
) -> Data:
    """Build graph for inference (no solution labels needed).

    Identical features as build_knapsack_graph but with dummy y=0.
    """
    dummy_solution = [0] * len(weights)
    return build_knapsack_graph(weights, values, capacity, dummy_solution, k=k)