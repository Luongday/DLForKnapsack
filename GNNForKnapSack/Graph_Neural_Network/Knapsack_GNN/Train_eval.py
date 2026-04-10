"""Training and evaluation utilities for KnapsackGNN — v2."""

from __future__ import annotations


import torch
from torch_geometric.loader import DataLoader

try:
    from decode_utils import greedy_feasible_decode
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from GNNForKnapSack.decode_utils import greedy_feasible_decode


def train_one_epoch(
    model,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  torch.nn.Module,
    device:     torch.device,
) -> float:
    """Train one epoch with optional VECTORIZED capacity penalty.

    Capacity penalty computed in a single pass via scatter_add — no Python loop.
    """
    model.train()
    total_loss  = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch)
        y      = batch.y.view(-1)

        bce_loss = criterion(logits, y)
        loss     = bce_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item() * y.numel()
        total_nodes += y.numel()

    return total_loss / max(total_nodes, 1)


@torch.no_grad()
def evaluate_node_accuracy(
    model,
    loader:    DataLoader,
    device:    torch.device,
    threshold: float = 0.5,
) -> float:
    """Node-level classification accuracy (secondary metric)."""
    model.eval()
    correct = 0
    total   = 0

    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch)
        preds  = (torch.sigmoid(logits) >= threshold).float()
        y      = batch.y.view(-1)
        correct += (preds == y).sum().item()
        total   += y.numel()

    return correct / max(total, 1)


@torch.no_grad()
def evaluate_approx_ratio(
    model,
    loader: DataLoader,
    device: torch.device,
    tol:    float = 1e-6,
) -> dict:
    """Evaluate solution quality: GNN value / DP optimal value."""
    model.eval()
    ratios, gnn_values, dp_values = [], [], []
    feasible_count = 0
    total_count    = 0

    for batch in loader:
        batch     = batch.to(device)
        batch_vec = batch.batch if hasattr(batch, "batch") else \
                    torch.zeros(batch.num_nodes, dtype=torch.long, device=device)
        n_graphs  = int(batch_vec.max().item()) + 1

        logits = model(batch)
        probs  = torch.sigmoid(logits)

        for g in range(n_graphs):
            mask  = batch_vec == g
            p_g   = probs[mask].detach().cpu()
            w_g   = batch.wts[mask].detach().cpu()
            v_g   = batch.vals[mask].detach().cpu()
            cap_g = batch.cap[g].item() if batch.cap.dim() > 0 else float(batch.cap.item())
            dp_val = float((batch.y[mask].view(-1).detach().cpu() * v_g).sum())

            x_hat   = greedy_feasible_decode(p_g, w_g, cap_g, tol)
            gnn_val = float((x_hat * v_g).sum())
            gnn_w   = float((x_hat * w_g).sum())

            feasible = gnn_w <= cap_g + tol
            ratio    = gnn_val / dp_val if dp_val > 0 else 0.0

            gnn_values.append(gnn_val)
            dp_values.append(dp_val)
            ratios.append(ratio)
            feasible_count += int(feasible)
            total_count    += 1

    return {
        "avg_ratio":        float(torch.tensor(ratios).mean()) if ratios else 0.0,
        "feasibility_rate": feasible_count / max(total_count, 1),
        "avg_gnn_value":    float(torch.tensor(gnn_values).mean()) if gnn_values else 0.0,
        "avg_dp_value":     float(torch.tensor(dp_values).mean())  if dp_values  else 0.0,
    }