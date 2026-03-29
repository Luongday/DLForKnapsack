"""Knapsack graph builder — sparse kNN edition.

Replaces the original O(n²) fully-connected graph with a k-NN sparse graph
so that instances with n=200 items stay tractable at training time.

Public API is backward-compatible: build_knapsack_graph() still accepts the
same arguments and returns a PyG Data object.

Node features (4-dim, matching dataset._build_sparse_graph):
    [weight_norm, value_norm, ratio_norm, cap_norm]

Edges:
    Directed k-NN based on Euclidean distance in feature space.
    Default k=16 gives ~16×n edges instead of n² — a 12× reduction at n=200.

Labels:
    Node-level 0/1 selection from DP.
"""

from typing import List

import torch
from torch_geometric.data import Data

# Default neighbourhood size.  Can be overridden per call.
DEFAULT_K = 16


def _build_knn_edges(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return a directed k-NN edge_index over node feature matrix x.

    Args:
        x: [n, d] float tensor.
        k: number of nearest neighbours per node.

    Returns:
        edge_index: [2, n*k_eff] long tensor (k_eff = min(k, n-1)).
    """
    n = x.size(0)
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    k_eff = min(k, n - 1)
    dist = torch.cdist(x, x, p=2)          # [n, n]
    dist.fill_diagonal_(float("inf"))       # exclude self
    knn_idx = dist.topk(k_eff, largest=False).indices  # [n, k_eff]

    row = torch.arange(n).unsqueeze(1).expand(-1, k_eff).reshape(-1)
    col = knn_idx.reshape(-1)
    return torch.stack([row, col], dim=0)   # [2, n*k_eff]


def build_knapsack_graph(
    weights: List[int],
    values: List[int],
    capacity: int,
    solution: List[int],
    k: int = DEFAULT_K,
) -> Data:
    """Convert one knapsack instance into a PyG graph (Data).

    Node features (4-dim):
        [weight_norm, value_norm, ratio_norm, cap_norm]

    Edges:
        Directed k-NN (default k=16).  O(n·k) instead of O(n²).

    Labels:
        Node-level 0/1 selection from DP.

    Args:
        weights:  Item weights (list of ints).
        values:   Item values  (list of ints).
        capacity: Knapsack capacity.
        solution: DP-optimal 0/1 selection list.
        k:        k for k-NN graph construction.

    Returns:
        PyG Data object ready for GNN training / inference.
    """
    w = torch.tensor(weights, dtype=torch.float32)
    v = torch.tensor(values, dtype=torch.float32)
    sol = torch.tensor(solution, dtype=torch.float32)  # [n]

    ratio = v / (w + 1e-8)
    w_norm = w / (w.max() + 1e-8)
    v_norm = v / (v.max() + 1e-8)
    ratio_norm = ratio / (ratio.max() + 1e-8)
    cap_norm = torch.full_like(w_norm, float(capacity) / (w.sum() + 1e-8))

    # 4-dim features — consistent with dataset._build_sparse_graph
    x = torch.stack([w_norm, v_norm, ratio_norm, cap_norm], dim=1)  # [n, 4]

    edge_index = _build_knn_edges(x, k=k)

    y = sol.unsqueeze(1)  # [n, 1]

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        wts=w,
        vals=v,
        cap=torch.tensor([capacity], dtype=torch.float32),
    )