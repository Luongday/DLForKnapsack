"""Môi trường (Environment) cho bài toán 0/1 Knapsack theo kiểu Sequential Decision."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np

@dataclass
class StepOutput:
    next_state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]

class KnapsackEnv:
    """Sequential knapsack environment for a single instance.

    At step i, you decide action a in {0,1} for item i.
    Feasibility is enforced by masking: if w[i] > cap, action=1 is invalid.
    """
    def __init__(self, weights: np.ndarray, values: np.ndarray, capacity: int, eps: float = 1e-8):
        self.w = np.asarray(weights).astype(np.float32)
        self.v = np.asarray(values).astype(np.float32)
        self.W = float(capacity)
        self.eps = eps
        self.n = int(self.w.shape[0])
        self.reset()

    def reset(self):
        self.i = 0
        self.cap = float(self.W)
        self.total_value = 0.0
        self.total_weight = 0.0
        self.selection = np.zeros(self.n, dtype=np.int64)
        return self._state()

    def valid_actions_mask(self) -> np.ndarray:
        """Trả về mask hợp lệ: [1, 1] nếu có thể lấy, [1, 0] nếu không thể lấy"""
        if self.i >= self.n:
            return np.array([1, 0], dtype=np.int64)
        can_take = (self.w[self.i] <= self.cap + self.eps)
        return np.array([1, 1 if can_take else 0], dtype=np.int64)

    def step(self, action: int) -> StepOutput:
        if self.i >= self.n:
            # already done
            return StepOutput(self._state(), 0.0, True, {"terminal": True})

        mask = self.valid_actions_mask()
        # enforce feasibility: if invalid, force a=0
        if action == 1 and mask[1] == 0:
            action = 0

        reward = 0.0
        if action == 1:
            self.cap -= float(self.w[self.i])
            self.total_weight += float(self.w[self.i])
            self.total_value += float(self.v[self.i])
            self.selection[self.i] = 1
            reward = float(self.v[self.i])

        self.i += 1
        done = (self.i >= self.n)
        return StepOutput(self._state(), reward, done, {
            "i": self.i,
            "cap": self.cap,
            "total_value": self.total_value,
            "total_weight": self.total_weight,
        })

    def _state(self) -> np.ndarray:
        """Xây dựng vector trạng thái (14 chiều) - Đây là input cho Q-Network"""
        if self.i >= self.n:
            return np.zeros(14, dtype=np.float32)

        w_max = self.w.max()
        v_max = self.v.max()

        # Current item
        wi, vi = self.w[self.i], self.v[self.i]
        ratio = vi / (wi + self.eps)
        cur = np.array([wi / w_max, vi / v_max, ratio], dtype=np.float32)

        # Global features
        cap_norm = self.cap / (self.W + self.eps)
        progress = self.i / self.n

        # Remaining items statistics
        w_rem = self.w[self.i:]
        v_rem = self.v[self.i:]
        r_rem = v_rem / (w_rem + self.eps)

        def get_stats(x):
            return np.array([x.mean(), x.std(ddof=0), x.max()], dtype=np.float32)

        rem_stats = np.concatenate([
            get_stats(w_rem / w_max),
            get_stats(v_rem / v_max),
            get_stats(r_rem)
        ])

        return np.concatenate([cur, [cap_norm, progress], rem_stats]).astype(np.float32)

    def compute_solution_value(self) -> float:
        return float(self.total_value)

    def compute_solution_weight(self) -> float:
        return float(self.total_weight)