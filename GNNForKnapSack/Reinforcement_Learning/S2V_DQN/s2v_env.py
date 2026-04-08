"""S2V-DQN graph environment for 0/1 Knapsack.

Different from dqn_env.py (MLP-based):
    - State is a PyTorch Geometric Data graph, not a flat vector
    - Each node has features encoding both static info (w, v, ratio)
      and dynamic state (whether selected, whether current item, remaining capacity)
    - Action space: select any unselected item (with feasibility mask)

This matches the S2V-DQN pattern from Dai et al. 2017:
    "Learning Combinatorial Optimization Algorithms over Graphs"
    where state = current partial solution embedded into graph structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.Graph_builder import _build_knn_edges  # reuse existing kNN builder


# Node feature dimensions (static + dynamic)
S2V_NODE_DIM = 7
# Features per node:
#   0: weight (normalized by max_w)
#   1: value (normalized by max_v)
#   2: value/weight ratio (normalized by max_ratio)
#   3: selected flag (0 or 1)
#   4: feasible flag (1 if can fit in remaining capacity)
#   5: remaining capacity (normalized by total capacity, broadcast to all nodes)
#   6: fraction of items selected (broadcast)


@dataclass
class GraphStepOutput:
    next_state: Data           # graph state
    reward: float
    done: bool
    info: Dict[str, Any]


class GraphKnapsackEnv:
    """Graph-based knapsack environment for S2V-DQN.

    State at each step is a PyG Data graph where:
        - Nodes = items (n nodes per instance)
        - Edges = kNN by value/weight ratio (built once)
        - Node features include both static (w, v) and dynamic (selected, fits)

    Action space: index of item to select next (0 to n-1).
    Invalid actions (already selected or doesn't fit) are masked at agent level.
    """

    def __init__(
        self,
        weights: np.ndarray,
        values:  np.ndarray,
        capacity: int,
        k:       int = 16,
        eps:     float = 1e-8,
    ):
        self.w     = np.asarray(weights).astype(np.float32)
        self.v     = np.asarray(values).astype(np.float32)
        self.W     = float(capacity)
        self.k     = k
        self.eps   = eps
        self.n     = int(self.w.shape[0])

        # Normalization constants (computed once)
        self.w_max = float(self.w.max()) if self.n > 0 else 1.0
        self.v_max = float(self.v.max()) if self.n > 0 else 1.0
        self.r_max = self.v_max / (self.w_max + self.eps)

        # Build edges once (graph structure is static)
        if self.n > 1:
            # Use static features (w_norm, v_norm, ratio) for kNN distance
            w_static = self.w / (self.w_max + self.eps)
            v_static = self.v / (self.v_max + self.eps)
            r_static = (self.v / (self.w + self.eps)) / (self.r_max + self.eps)
            static_x = torch.from_numpy(
                np.stack([w_static, v_static, r_static], axis=1).astype(np.float32)
            )
            self.edge_index = _build_knn_edges(static_x, k=min(self.k, self.n - 1))
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)

        self.reset()

    def reset(self) -> Data:
        self.cap          = float(self.W)
        self.total_value  = 0.0
        self.total_weight = 0.0
        self.selection    = np.zeros(self.n, dtype=np.int64)
        self.steps_taken  = 0
        return self._build_graph_state()

    def _build_graph_state(self) -> Data:
        """Construct PyG Data graph for current state."""
        # Static features
        w_norm = self.w / (self.w_max + self.eps)
        v_norm = self.v / (self.v_max + self.eps)
        ratios = (self.v / (self.w + self.eps)) / (self.r_max + self.eps)

        # Dynamic features
        selected_flag = self.selection.astype(np.float32)
        feasible_flag = (self.w <= self.cap + self.eps).astype(np.float32)
        # Already-selected items shouldn't be feasible to re-select
        feasible_flag = feasible_flag * (1.0 - selected_flag)

        cap_remaining_norm = float(self.cap / (self.W + self.eps))
        items_selected_frac = float(self.selection.sum() / self.n) if self.n > 0 else 0.0

        cap_broadcast  = np.full(self.n, cap_remaining_norm, dtype=np.float32)
        frac_broadcast = np.full(self.n, items_selected_frac, dtype=np.float32)

        # Stack: [n, 7]
        x = np.stack([
            w_norm, v_norm, ratios,
            selected_flag, feasible_flag,
            cap_broadcast, frac_broadcast,
        ], axis=1).astype(np.float32)

        return Data(
            x=torch.from_numpy(x),
            edge_index=self.edge_index,
            wts=torch.from_numpy(self.w),
            vals=torch.from_numpy(self.v),
            cap=torch.tensor([self.W], dtype=torch.float32),
        )

    def valid_actions_mask(self) -> np.ndarray:
        """Boolean mask of length n: True if item can be selected.

        An action is valid iff:
            - item not already selected
            - item weight <= remaining capacity
        """
        not_selected = (self.selection == 0)
        fits = (self.w <= self.cap + self.eps)
        return (not_selected & fits).astype(np.int64)

    def step(self, action: int) -> GraphStepOutput:
        """Select item `action`. Returns reward = item value if valid, else 0."""
        mask = self.valid_actions_mask()

        if mask.sum() == 0:
            # No valid actions left → terminal
            return GraphStepOutput(
                self._build_graph_state(), 0.0, True,
                {"terminal": True, "reason": "no_valid_actions"}
            )

        # Enforce valid action
        if action < 0 or action >= self.n or mask[action] == 0:
            # Invalid action — treat as terminal (agent should respect mask)
            return GraphStepOutput(
                self._build_graph_state(), 0.0, True,
                {"terminal": True, "reason": "invalid_action"}
            )

        # Apply action: select item
        w_i = float(self.w[action])
        v_i = float(self.v[action])
        self.cap          -= w_i
        self.total_weight += w_i
        self.total_value  += v_i
        self.selection[action] = 1
        self.steps_taken  += 1

        reward = v_i

        # Check if done: no more valid actions
        new_mask = self.valid_actions_mask()
        done = (new_mask.sum() == 0)

        return GraphStepOutput(
            self._build_graph_state(), reward, done,
            {
                "selected": action,
                "cap_remaining": self.cap,
                "total_value": self.total_value,
                "total_weight": self.total_weight,
                "n_selected": int(self.selection.sum()),
            }
        )

    def compute_solution_value(self) -> float:
        return float(self.total_value)

    def compute_solution_weight(self) -> float:
        return float(self.total_weight)

    def get_selection(self) -> np.ndarray:
        return self.selection.copy()