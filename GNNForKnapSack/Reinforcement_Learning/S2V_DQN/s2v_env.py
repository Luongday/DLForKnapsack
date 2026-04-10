"""S2V-DQN graph environment for 0/1 Knapsack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from torch_geometric.data import Data

from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.Graph_builder import _build_knn_edges


# Node feature dimensions (static + dynamic) — GIỮ NGUYÊN
S2V_NODE_DIM = 7


@dataclass
class GraphStepOutput:
    next_state: Data
    reward: float
    done: bool
    info: Dict[str, Any]


class GraphKnapsackEnv:
    """Graph-based knapsack environment for S2V-DQN."""

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

        # Normalization constants (compute 1 lần)
        self.w_max = float(self.w.max()) if self.n > 0 else 1.0
        self.v_max = float(self.v.max()) if self.n > 0 else 1.0
        self.r_max = self.v_max / (self.w_max + self.eps)

        # ============== [FIX 3] Cache static features ==============
        # Static node features [n, 3]: (w_norm, v_norm, ratio_norm)
        # Những giá trị này phụ thuộc instance, không đổi trong episode.
        w_norm_np = self.w / (self.w_max + self.eps)
        v_norm_np = self.v / (self.v_max + self.eps)
        ratios_np = (self.v / (self.w + self.eps)) / (self.r_max + self.eps)
        self._static_x = torch.from_numpy(
            np.stack([w_norm_np, v_norm_np, ratios_np], axis=1).astype(np.float32)
        )  # [n, 3]

        # Cache các tensor phụ trợ để reset/step không phải tạo lại
        self._wts_tensor  = torch.from_numpy(self.w.copy())
        self._vals_tensor = torch.from_numpy(self.v.copy())
        self._cap_tensor  = torch.tensor([self.W], dtype=torch.float32)
        # =============================================================

        # Build edges 1 lần (graph structure tĩnh)
        if self.n > 1:
            self.edge_index = _build_knn_edges(
                self._static_x, k=min(self.k, self.n - 1)
            )
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
        """Construct PyG Data graph for current state.

        [FIX 3] Chỉ build 4 dynamic columns rồi concat với static tensor
        đã cache. Thứ tự cột giữ NGUYÊN như bản gốc:
            [w_norm, v_norm, ratio, selected, feasible, cap, frac]
        """
        # Dynamic features
        selected_flag_np = self.selection.astype(np.float32)
        feasible_np      = (self.w <= self.cap + self.eps).astype(np.float32)
        # Already-selected items shouldn't be feasible to re-select
        feasible_flag_np = feasible_np * (1.0 - selected_flag_np)

        cap_norm = float(self.cap / (self.W + self.eps))
        frac     = float(self.selection.sum() / self.n) if self.n > 0 else 0.0

        # Stack 4 dynamic columns thành [n, 4]
        dynamic_np = np.stack([
            selected_flag_np,
            feasible_flag_np,
            np.full(self.n, cap_norm, dtype=np.float32),
            np.full(self.n, frac,     dtype=np.float32),
        ], axis=1)
        dynamic_x = torch.from_numpy(dynamic_np)  # [n, 4]

        # Concat static (cached) + dynamic → [n, 7]
        x = torch.cat([self._static_x, dynamic_x], dim=1)

        return Data(
            x=x,
            edge_index=self.edge_index,
            wts=self._wts_tensor,
            vals=self._vals_tensor,
            cap=self._cap_tensor,
        )

    def valid_actions_mask(self) -> np.ndarray:
        """Boolean mask of length n: True if item can be selected.

        An action is valid iff:
            - item not already selected
            - item weight <= remaining capacity
        """
        not_selected = (self.selection == 0)
        fits         = (self.w <= self.cap + self.eps)
        return (not_selected & fits).astype(np.int64)

    def step(self, action: int) -> GraphStepOutput:
        """Select item `action`. Returns reward = item value if valid, else 0."""
        mask = self.valid_actions_mask()

        if mask.sum() == 0:
            return GraphStepOutput(
                self._build_graph_state(), 0.0, True,
                {"terminal": True, "reason": "no_valid_actions"}
            )

        if action < 0 or action >= self.n or mask[action] == 0:
            return GraphStepOutput(
                self._build_graph_state(), 0.0, True,
                {"terminal": True, "reason": "invalid_action"}
            )

        # Apply action
        w_i = float(self.w[action])
        v_i = float(self.v[action])
        self.cap          -= w_i
        self.total_weight += w_i
        self.total_value  += v_i
        self.selection[action] = 1
        self.steps_taken  += 1

        reward = v_i

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