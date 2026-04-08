"""Replay buffer for S2V-DQN.

Stores graph transitions (PyG Data objects) instead of flat vectors.
Uses PyG Batch.from_data_list to batch graphs efficiently.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, Batch


@dataclass
class GraphTransition:
    """One transition for graph DQN.

    Attributes:
        s:           Current graph state
        a:           Action taken (node index)
        r:           Reward received
        s2:          Next graph state
        done:        Episode terminated
        valid_mask:  Valid action mask at next state [n_items_in_s2]
    """
    s:          Data
    a:          int
    r:          float
    s2:         Data
    done:       bool
    valid_mask: np.ndarray  # [n] boolean mask of valid actions at s2


class GraphReplayBuffer:
    """FIFO replay buffer for graph transitions."""

    def __init__(self, capacity: int, seed: int = 42):
        self.capacity = int(capacity)
        self.rng = random.Random(seed)
        self.data: List[GraphTransition] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.data)

    def push(self, t: GraphTransition) -> None:
        if len(self.data) < self.capacity:
            self.data.append(t)
        else:
            self.data[self.pos] = t
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[Batch, torch.Tensor, torch.Tensor,
                                                 Batch, torch.Tensor, List[np.ndarray]]:
        """Sample a batch of transitions.

        Returns:
            s_batch:     Batched current states (PyG Batch)
            a_batch:     Action indices [batch_size] (LOCAL indices within each graph)
            r_batch:     Rewards [batch_size]
            s2_batch:    Batched next states (PyG Batch)
            done_batch:  Done flags [batch_size]
            mask_list:   List of valid_mask arrays (one per next state)
        """
        batch = self.rng.sample(self.data, k=batch_size)

        s_list  = [t.s  for t in batch]
        s2_list = [t.s2 for t in batch]

        s_batch  = Batch.from_data_list(s_list)
        s2_batch = Batch.from_data_list(s2_list)

        a = torch.tensor([t.a for t in batch], dtype=torch.long)
        r = torch.tensor([t.r for t in batch], dtype=torch.float32)
        d = torch.tensor([float(t.done) for t in batch], dtype=torch.float32)

        mask_list = [t.valid_mask for t in batch]

        return s_batch, a, r, s2_batch, d, mask_list