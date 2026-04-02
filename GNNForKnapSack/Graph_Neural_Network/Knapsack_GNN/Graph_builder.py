"""Knapsack graph builder — improved sparse kNN edition.

Improvements vs original:
    - Feature vector expanded from 4 to 6 dimensions:
      + Added global capacity utilization: capacity / sum(weights)
      + Added normalized item fraction: 1/n_items
      These give the GNN awareness of the global problem structure,
      not just per-item statistics.
    - cap_ratio feature fixed: now w_i / capacity (consistent with docstring)
    - Documentation aligned with actual implementation
    - build_knapsack_graph_inference() added for eval (no solution needed)
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch_geometric.data import Data

DEFAULT_K = 16


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

    Node features (6-dim):
        0: w_norm     — weight_i / max(weights)
        1: v_norm     — value_i / max(values)
        2: ratio_norm — (v_i/w_i) / max(v/w)
        3: cap_ratio  — weight_i / capacity  (how much of capacity this item uses)
        4: cap_util   — capacity / sum(weights) (global: how tight is the knapsack)
        5: item_frac  — 1 / n_items (global: inverse problem size)

    Features 4-5 are global (same for all nodes in one graph) — they give the
    GNN context about the overall problem structure.

    Args:
        weights:  Item weights (list of int).
        values:   Item values (list of int).
        capacity: Knapsack capacity.
        solution: Binary 0/1 DP-optimal selection.
        k:        kNN neighbourhood size.

    Returns:
        PyG Data with x=[n,6], edge_index, y=[n,1], wts, vals, cap.
    """
    w   = torch.tensor(weights,  dtype=torch.float32)
    v   = torch.tensor(values,   dtype=torch.float32)
    sol = torch.tensor(solution, dtype=torch.float32)
    n   = len(weights)

    # Per-item features (normalized)
    ratio = v / (w + 1e-8)

    w_norm     = w     / (w.max()     + 1e-8)
    v_norm     = v     / (v.max()     + 1e-8)
    ratio_norm = ratio / (ratio.max() + 1e-8)

    # Per-item: fraction of capacity this item occupies
    cap_ratio = w / (float(capacity) + 1e-8)

    # Global features (broadcast to all nodes)
    cap_util  = torch.full((n,), float(capacity) / (w.sum().item() + 1e-8))
    item_frac = torch.full((n,), 1.0 / max(n, 1))

    # 6-dim feature vector
    x = torch.stack([w_norm, v_norm, ratio_norm, cap_ratio, cap_util, item_frac], dim=1)

    edge_index = _build_knn_edges(x, k=k)
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