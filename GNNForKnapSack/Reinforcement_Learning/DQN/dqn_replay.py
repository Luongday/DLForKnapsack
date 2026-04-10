"""Replay Buffer cho DQN - Lưu trữ các transition (trải nghiệm) để train."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import random

@dataclass
class Transition:
    """Một transition (bước trải nghiệm) trong môi trường."""
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool
    mask2: np.ndarray  # valid action mask at next state

class ReplayBuffer:
    """Replay Buffer sử dụng cơ chế Circular Buffer (vòng tròn)."""
    def __init__(self, capacity: int = 200_000, seed: int = 42):
        self.capacity = int(capacity)
        self.rng = random.Random(seed)
        self.data: List[Transition] = []
        self.pos = 0

    def __len__(self):
        return len(self.data)

    def push(self, t: Transition):
        """Thêm một transition mới vào buffer."""
        if len(self.data) < self.capacity:
            self.data.append(t)
        else:
            self.data[self.pos] = t
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        """Lấy ngẫu nhiên batch_size transition từ buffer."""
        batch = self.rng.sample(self.data, k=batch_size)
        # Chuyển sang numpy array để train nhanh
        s = np.stack([b.s for b in batch], axis=0)
        a = np.asarray([b.a for b in batch], dtype=np.int64)
        r = np.asarray([b.r for b in batch], dtype=np.float32)
        s2 = np.stack([b.s2 for b in batch], axis=0)
        done = np.asarray([b.done for b in batch], dtype=np.float32)
        mask2 = np.stack([b.mask2 for b in batch], axis=0).astype(np.float32)
        return s, a, r, s2, done, mask2
