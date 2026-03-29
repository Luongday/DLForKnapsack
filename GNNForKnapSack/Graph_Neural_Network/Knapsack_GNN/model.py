"""KnapsackGNN model definition with safe checkpoint utilities."""

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Feature dimension produced by dataset._build_sparse_graph:
#   [weight_norm, value_norm, ratio_norm, cap_norm]
DEFAULT_IN_DIM = 4


class KnapsackGNN(nn.Module):
    """A simple 2-layer GCN for node-level selection prediction.

    Args:
        in_dim:     Number of input node features. Must match the dataset.
                    Defaults to DEFAULT_IN_DIM (4) — the dimension produced by
                    GeneratedKnapsack01Dataset / _build_sparse_graph.
        hidden_dim: Width of both GCN hidden layers.
    """

    def __init__(self, in_dim: int = DEFAULT_IN_DIM, hidden_dim: int = 64):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        # Guard against feature-dim mismatch at runtime (helpful error > silent NaN).
        if x.size(1) != self.in_dim:
            raise ValueError(
                f"KnapsackGNN expects {self.in_dim} input features per node, "
                f"but received {x.size(1)}. "
                "Check that the dataset and model are built with the same in_dim."
            )

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        logits = self.lin(x).squeeze(-1)  # [num_nodes]
        return logits


# ---------------------------------------------------------------------------
# Checkpoint helpers (shared by run_train.py and evaluate_gnn.py)
# ---------------------------------------------------------------------------

CKPT_KEY_STATE = "model_state"
CKPT_KEY_IN_DIM = "in_dim"
CKPT_KEY_HIDDEN = "hidden_dim"


def save_checkpoint(model: KnapsackGNN, path) -> None:
    """Save model weights and hyper-params under consistent keys.

    Always saves a dict so loaders can distinguish weights-only files.
    """
    torch.save(
        {
            CKPT_KEY_STATE: model.state_dict(),
            CKPT_KEY_IN_DIM: model.in_dim,
            CKPT_KEY_HIDDEN: model.hidden_dim,
        },
        path,
    )


def load_checkpoint(path, device: torch.device) -> KnapsackGNN:
    """Load a checkpoint saved by save_checkpoint.

    Also handles legacy files saved as a raw state_dict (no wrapper dict),
    in which case DEFAULT_IN_DIM and hidden_dim=64 are assumed.
    """
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and CKPT_KEY_STATE in ckpt:
        in_dim = ckpt.get(CKPT_KEY_IN_DIM, DEFAULT_IN_DIM)
        hidden_dim = ckpt.get(CKPT_KEY_HIDDEN, 64)
        state_dict = ckpt[CKPT_KEY_STATE]
    else:
        # Legacy: file is a raw state_dict
        in_dim = DEFAULT_IN_DIM
        hidden_dim = 64
        state_dict = ckpt

    model = KnapsackGNN(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model