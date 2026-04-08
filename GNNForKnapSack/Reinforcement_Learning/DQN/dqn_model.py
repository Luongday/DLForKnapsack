"""DQN Q-Network for 0/1 Knapsack.
Đây là mạng Q-Network đơn giản dùng MLP (không phải GNN).
Mục đích: Dự đoán Q-value cho 2 hành động: Skip(0) hoặc Take(1).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np

class QNetwork(nn.Module):
    """Q-Network for sequential knapsack decisions.

    Input : Vector trạng thái (mặc định 14 chiều)
    Output: Q-values cho 2 hành động [Q(skip), Q(take)]
    """
    def __init__(self, state_dim: int = 14, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # Q for actions: 0=skip, 1=take
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """Chọn hành động theo chiến lược Epsilon-Greedy
                - epsilon cao → khám phá ngẫu nhiên
                - epsilon thấp → khai thác (chọn action có Q-value cao nhất)
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, 2)
        with torch.no_grad():
            q_values = self.forward(state.unsqueeze(0))
            return q_values.argmax(dim=1).item()
