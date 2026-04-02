"""KnapsackGNN model — improved architecture.

Improvements vs original:
    - in_dim default updated: 4 → 6 (matches new Graph_builder features).
    - Added GlobalAttentionReadout: a lightweight global attention pooling
      that gives each node awareness of the *entire graph's* state.
      This is critical for constraint satisfaction — items need to "know"
      what other items have been selected globally.
    - Added global context injection: after GNN layers, concatenate
      graph-level readout to each node before final classification.
      Node sees [local_features | global_context] before deciding.
    - Kept backward compatibility: load_checkpoint handles old 4-dim models.
    - SAGEConv + BatchNorm + Residual preserved from previous version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from click.types import convert_type
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv,GINConv,global_mean_pool

DEFAULT_IN_DIM     = 6
DEFAULT_HIDDEN_DIM = 128
DEFAULT_NUM_LAYERS = 3
DEFAULT_DROPOUT    = 0.1
DEFAULT_CONV_TYPE = "gin"


def _make_gin_conv(in_dim: int, out_dim: int) -> GINConv:
    mlp = nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )
    return GINConv(nn=mlp, train_eps=True)


def _make_sage_conv(in_dim: int, out_dim: int) -> SAGEConv:
    return SAGEConv(in_dim, out_dim)


def _make_conv(conv_type: str, in_dim: int, out_dim: int):
    if conv_type == "gin":
        return _make_gin_conv(in_dim, out_dim)
    elif conv_type == "sage":
        return _make_sage_conv(in_dim, out_dim)
    else:
        raise ValueError(f"Unknown conv_type: {conv_type}")

class GlobalContextInjection(nn.Module):
    """Attention-weighted global pooling broadcast to each node."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        # Attention weights per node
        attn_scores = self.attn_gate(x)                    # [N, 1]
        attn_weights = torch.zeros_like(attn_scores)

        # Softmax per graph
        for g in range(int(batch.max().item()) + 1):
            mask = batch == g
            if mask.sum() > 0:
                attn_weights[mask] = torch.softmax(attn_scores[mask], dim=0)

        # Weighted sum per graph
        weighted = x * attn_weights                         # [N, H]
        graph_emb = global_mean_pool(weighted, batch)       # [G, H] (mean over weighted)

        # Scale correction: mean_pool on weighted = sum/count, we want sum
        # So multiply by count per graph
        counts = torch.zeros(graph_emb.size(0), 1, device=x.device)
        for g in range(graph_emb.size(0)):
            counts[g] = (batch == g).sum()
        graph_emb = graph_emb * counts

        # Project and broadcast back to nodes
        context = self.context_proj(graph_emb)              # [G, H]
        node_context = context[batch]                       # [N, H]

        return node_context


class KnapsackGNN(nn.Module):
    def __init__(
        self,
        in_dim:        int   = DEFAULT_IN_DIM,
        hidden_dim:    int   = DEFAULT_HIDDEN_DIM,
        num_layers:    int   = DEFAULT_NUM_LAYERS,
        dropout:       float = DEFAULT_DROPOUT,
        use_residual:  bool  = True,
        use_global_ctx: bool = True,
        conv_type: str = DEFAULT_CONV_TYPE,
    ):
        super().__init__()
        self.in_dim        = in_dim
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.dropout_p     = dropout
        self.use_residual  = use_residual
        self.use_global_ctx = use_global_ctx
        self.conv_type = conv_type

        # Message passing layers
        self.convs = nn.ModuleList()
        if conv_type == "hybrid":
            self.convs.append(_make_gin_conv(in_dim, hidden_dim))
            for i in range(1, num_layers - 1):
                self.convs.append(_make_gin_conv(hidden_dim, hidden_dim))
            if num_layers > 1:
                self.convs.append(_make_sage_conv(hidden_dim, hidden_dim))
        else:
            self.convs.append(_make_conv(conv_type, in_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(_make_conv(conv_type, hidden_dim, hidden_dim))

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(p=dropout)

        # Residual projection for first layer (in_dim → hidden_dim)
        self.res_proj = nn.Linear(in_dim, hidden_dim, bias=False) if use_residual else None

        # Global context injection
        if use_global_ctx:
            self.global_ctx = GlobalContextInjection(hidden_dim)
            self.fusion     = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )
        else:
            self.global_ctx = None
            self.fusion     = None

        # Output head
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        if x.size(1) != self.in_dim:
            raise ValueError(
                f"KnapsackGNN expects {self.in_dim} input features, "
                f"got {x.size(1)}. Check Graph_builder in_dim."
            )

        # Infer batch vector
        if hasattr(data, 'batch') and data.batch is not None:
            batch_vec = data.batch
        else:
            batch_vec = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Message passing with residual connections
        for i, conv in enumerate(self.convs):
            x_in = x
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = torch.relu(x)
            x = self.dropout(x)

            if self.use_residual:
                if i == 0 and self.res_proj is not None:
                    x = x + self.res_proj(x_in)
                elif i > 0:
                    x = x + x_in

        # Global context injection
        if self.use_global_ctx and self.global_ctx is not None:
            ctx = self.global_ctx(x, batch_vec)   # [N, H]
            x   = self.fusion(torch.cat([x, ctx], dim=1))  # [N, 2H] → [N, H]

        logits = self.lin(x).squeeze(-1)  # [num_nodes]
        return logits


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

CKPT_KEY_STATE      = "model_state"
CKPT_KEY_IN_DIM     = "in_dim"
CKPT_KEY_HIDDEN     = "hidden_dim"
CKPT_KEY_LAYERS     = "num_layers"
CKPT_KEY_DROPOUT    = "dropout"
CKPT_KEY_RESIDUAL   = "use_residual"
CKPT_KEY_GLOBAL_CTX = "use_global_ctx"
CKPT_KEY_CONV_TYPE  = "conv_type"
CKPT_KEY_ARCH       = "arch"


def save_checkpoint(model: KnapsackGNN, path: Path | str) -> None:
    """Save model weights and hyperparams."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            CKPT_KEY_STATE:      model.state_dict(),
            CKPT_KEY_IN_DIM:     model.in_dim,
            CKPT_KEY_HIDDEN:     model.hidden_dim,
            CKPT_KEY_LAYERS:     model.num_layers,
            CKPT_KEY_DROPOUT:    model.dropout_p,
            CKPT_KEY_RESIDUAL:   model.use_residual,
            CKPT_KEY_GLOBAL_CTX: model.use_global_ctx,
            CKPT_KEY_CONV_TYPE:  model.conv_type,
            CKPT_KEY_ARCH:       "gin_v3",
        },path)

def load_checkpoint(
    path:    Path | str,
    device:  torch.device,
    dropout: Optional[float] = None,
) -> KnapsackGNN:
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and CKPT_KEY_STATE in ckpt:
        in_dim       = ckpt.get(CKPT_KEY_IN_DIM,     DEFAULT_IN_DIM)
        hidden_dim   = ckpt.get(CKPT_KEY_HIDDEN,     DEFAULT_HIDDEN_DIM)
        num_layers   = ckpt.get(CKPT_KEY_LAYERS,     DEFAULT_NUM_LAYERS)
        dp           = ckpt.get(CKPT_KEY_DROPOUT,    DEFAULT_DROPOUT)
        use_residual = ckpt.get(CKPT_KEY_RESIDUAL,   True)
        use_global   = ckpt.get(CKPT_KEY_GLOBAL_CTX, True)
        conv_type    = ckpt.get(CKPT_KEY_CONV_TYPE, "sage")
        arch         = ckpt.get(CKPT_KEY_ARCH,       "unknown")
        state_dict   = ckpt[CKPT_KEY_STATE]

        if arch in ("gcn", "sage", "sage_v2"):
            print(f"[load_checkpoint] Old {arch} checkpoint -> rebuilding as {conv_type}.")
    else:
        state_dict   = ckpt
        in_dim       = DEFAULT_IN_DIM
        hidden_dim   = DEFAULT_HIDDEN_DIM
        num_layers   = DEFAULT_NUM_LAYERS
        dp           = DEFAULT_DROPOUT
        use_residual = True
        use_global   = True
        conv_type = "sage"
        print("[load_checkpoint] Legacy raw state_dict. Using defaults.")

    if dropout is not None:
        dp = dropout

    model = KnapsackGNN(
        in_dim=in_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, dropout=dp,
        use_residual=use_residual,
        use_global_ctx=use_global,
        conv_type=conv_type,
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_checkpoint] Missing keys: {missing}")
    if unexpected:
        print(f"[load_checkpoint] Unexpected keys: {unexpected}")

    model.eval()
    return model