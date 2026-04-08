"""S2V-DQN Q-Network for 0/1 Knapsack."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool


def _make_gin_conv(in_dim: int, out_dim: int) -> GINConv:
    """Build one GINConv layer with internal MLP."""
    mlp = nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )
    return GINConv(nn=mlp, train_eps=True)


class S2VQNetwork(nn.Module):
    """S2V-DQN Q-network for Knapsack.

    Forward returns Q-values per node (one Q-value per item).
    Higher Q = more valuable to select that item next.

    Args:
        in_dim:     Input node feature dimension (7 from s2v_env.py)
        hidden_dim: GNN hidden dimension
        num_layers: Number of GIN layers
        dropout:    Dropout rate

    Forward input:  PyG Data or Batch with x [N, in_dim], edge_index, batch
    Forward output: Tensor [N] of Q-values, one per node
    """

    def __init__(
        self,
        in_dim:     int = 7,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.in_dim     = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p  = dropout

        # GNN encoder
        self.convs = nn.ModuleList()
        self.convs.append(_make_gin_conv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(_make_gin_conv(hidden_dim, hidden_dim))

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(p=dropout)

        # Q-head: combines node embedding + graph embedding → scalar Q
        # Input: [node_emb, graph_emb] → 2 * hidden_dim
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        """Compute Q-values per node.

        Args:
            data: PyG Data or Batch.
                  Must have x [N, in_dim], edge_index, batch (or single graph).

        Returns:
            q_values: Tensor [N] — Q-value for selecting each node.
        """
        x, edge_index = data.x, data.edge_index

        if x.size(1) != self.in_dim:
            raise ValueError(
                f"Expected {self.in_dim} features, got {x.size(1)}"
            )

        # Batch vector
        if hasattr(data, "batch") and data.batch is not None:
            batch_vec = data.batch
        else:
            batch_vec = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # GNN encoder
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = self.dropout(x)

        # Graph-level embedding (mean pool over nodes per graph)
        graph_emb = global_mean_pool(x, batch_vec)  # [n_graphs, hidden]

        # Broadcast graph emb back to each node
        graph_emb_per_node = graph_emb[batch_vec]  # [N, hidden]

        # Concatenate node embedding with its graph embedding
        combined = torch.cat([x, graph_emb_per_node], dim=1)  # [N, 2*hidden]

        # Q-value per node
        q_values = self.q_head(combined).squeeze(-1)  # [N]

        return q_values


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_s2v_checkpoint(model: S2VQNetwork, path) -> None:
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "in_dim":      model.in_dim,
        "hidden_dim":  model.hidden_dim,
        "num_layers":  model.num_layers,
        "dropout":     model.dropout_p,
        "arch":        "s2v_dqn_v1",
    }, path)


def load_s2v_checkpoint(path, device: torch.device) -> S2VQNetwork:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = S2VQNetwork(
        in_dim=ckpt.get("in_dim", 7),
        hidden_dim=ckpt.get("hidden_dim", 128),
        num_layers=ckpt.get("num_layers", 3),
        dropout=ckpt.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model