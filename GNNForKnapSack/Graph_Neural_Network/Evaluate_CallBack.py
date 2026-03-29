"""Post-epoch evaluation callback for KnapsackGNN training.

Ported from EvaluateCallBack.py (original Keras-based Neuro-Knapsack).

Key changes vs original:
    - Rewritten for PyTorch — no Keras dependency.
    - Removed CSVDataFrame (custom dependency) → standard csv module.
    - Early stopping tracked by approx_ratio (GNN value / DP value),
      same logic as original but cleaned up.
    - Model saved via save_checkpoint() from model.py (consistent format).
    - Greedy baseline evaluated alongside GNN every epoch for comparison.
    - Compatible with run_train.py training loop.

Usage in run_train.py:
    from evaluate_callback import EvaluateCallback
    callback = EvaluateCallback(
        model, test_loader, device,
        save_path="results/GNN/gnn.pt",
        max_wait=5,
    )
    for epoch in range(epochs):
        train_one_epoch(...)
        stop = callback.on_epoch_end(epoch)
        if stop:
            break
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from Knapsack_GNN.model import KnapsackGNN, save_checkpoint
from GNNForKnapSack.solvers.Greedy.Greedy import greedy_knapsack


def _greedy_feasible_decode(
    probs: torch.Tensor,
    weights: torch.Tensor,
    capacity: float,
) -> torch.Tensor:
    """Pick items by descending GNN probability while respecting capacity."""
    idx   = torch.argsort(probs, descending=True)
    x_hat = torch.zeros_like(probs)
    total = 0.0
    for i in idx:
        w_i = weights[i].item()
        if total + w_i <= capacity + 1e-6:
            x_hat[i] = 1.0
            total    += w_i
    return x_hat


class EvaluateCallback:
    """Evaluate GNN and Greedy baselines after each training epoch.

    Tracks approximation ratio (GNN value / DP optimal value).
    Saves model when ratio improves. Stops training when no improvement
    for max_wait epochs.

    Args:
        model:      KnapsackGNN instance being trained.
        loader:     DataLoader for the validation/test split.
        device:     torch.device.
        save_path:  Path to save best model checkpoint.
        max_wait:   Epochs without improvement before early stopping.
        log_path:   Optional CSV path for per-epoch metrics.
    """

    HEADER = [
        "epoch", "gnn_approx_ratio", "greedy_approx_ratio",
        "gnn_feasibility", "greedy_feasibility",
        "avg_gnn_value", "avg_dp_value", "epoch_time_sec",
    ]

    def __init__(
        self,
        model:     KnapsackGNN,
        loader:    DataLoader,
        device:    torch.device,
        save_path: str = "results/GNN/gnn_best.pt",
        max_wait:  int = 5,
        log_path:  Optional[str] = "logs/eval_callback.csv",
    ):
        self.model     = model
        self.loader    = loader
        self.device    = device
        self.save_path = Path(save_path)
        self.max_wait  = max_wait
        self.log_path  = Path(log_path) if log_path else None

        self._best_ratio: float = 0.0   # higher is better (GNN/DP ratio)
        self._wait:       int   = 0
        self._rows:       List[dict] = []

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def on_epoch_end(self, epoch: int) -> bool:
        """Run evaluation. Returns True if training should stop.

        Args:
            epoch: Current epoch number (0-based).

        Returns:
            True  → stop training (early stopping triggered).
            False → continue.
        """
        t0 = time.perf_counter()
        self.model.eval()

        gnn_values, dp_values         = [], []
        gnn_weights, capacities       = [], []
        greedy_values, greedy_weights = [], []

        for batch in self.loader:
            batch    = batch.to(self.device)
            batch_v  = batch.batch if hasattr(batch, "batch") else \
                       torch.zeros(batch.num_nodes, dtype=torch.long, device=self.device)
            n_graphs = int(batch_v.max().item()) + 1

            logits = self.model(batch)
            probs  = torch.sigmoid(logits)

            for g in range(n_graphs):
                mask   = batch_v == g
                p_g    = probs[mask].detach().cpu()
                w_g    = batch.wts[mask].detach().cpu()
                v_g    = batch.vals[mask].detach().cpu()
                cap_g  = float(batch.cap[g].item() if batch.cap.dim() > 0 else batch.cap.item())
                dp_val = float((batch.y[mask].view(-1).detach().cpu() * v_g).sum())

                # GNN decode
                gnn_sel = _greedy_feasible_decode(p_g, w_g, cap_g)
                gnn_val = float((gnn_sel * v_g).sum())
                gnn_w   = float((gnn_sel * w_g).sum())

                # Greedy decode
                gr_sel = torch.tensor(
                    greedy_knapsack(v_g.numpy(), w_g.numpy(), cap_g),
                    dtype=torch.float32,
                )
                gr_val = float((gr_sel * v_g).sum())
                gr_w   = float((gr_sel * w_g).sum())

                gnn_values.append(gnn_val)
                dp_values.append(dp_val)
                gnn_weights.append(gnn_w)
                capacities.append(cap_g)
                greedy_values.append(gr_val)
                greedy_weights.append(gr_w)

        # Aggregate
        avg_gnn  = np.mean(gnn_values)
        avg_dp   = np.mean(dp_values)
        avg_gr   = np.mean(greedy_values)

        gnn_ratio    = avg_gnn / avg_dp if avg_dp > 0 else 0.0
        greedy_ratio = avg_gr  / avg_dp if avg_dp > 0 else 0.0

        gnn_feas    = np.mean([w <= c + 1e-6 for w, c in zip(gnn_weights, capacities)])
        greedy_feas = np.mean([w <= c + 1e-6 for w, c in zip(greedy_weights, capacities)])

        elapsed = time.perf_counter() - t0

        print(
            f"  Eval epoch {epoch+1:03d} | "
            f"GNN ratio={gnn_ratio:.4f} | Greedy ratio={greedy_ratio:.4f} | "
            f"GNN feas={gnn_feas:.3f} | Greedy feas={greedy_feas:.3f} | "
            f"time={elapsed:.2f}s"
        )

        # --- Save best model ---
        if gnn_ratio > self._best_ratio:
            self._best_ratio = gnn_ratio
            self._wait       = 0
            save_checkpoint(self.model, self.save_path)
            print(f"  [callback] Best ratio improved → {gnn_ratio:.4f}. Saved to {self.save_path}")
        else:
            self._wait += 1
            print(f"  [callback] No improvement ({self._wait}/{self.max_wait})")

        # --- Log to CSV ---
        row = {
            "epoch":               epoch + 1,
            "gnn_approx_ratio":    round(gnn_ratio,    4),
            "greedy_approx_ratio": round(greedy_ratio, 4),
            "gnn_feasibility":     round(float(gnn_feas),    3),
            "greedy_feasibility":  round(float(greedy_feas), 3),
            "avg_gnn_value":       round(float(avg_gnn), 2),
            "avg_dp_value":        round(float(avg_dp),  2),
            "epoch_time_sec":      round(elapsed, 2),
        }
        self._rows.append(row)
        if self.log_path:
            self._flush_csv()

        # --- Early stopping ---
        if self._wait >= self.max_wait:
            print(f"  [callback] Early stopping after {self.max_wait} epochs without improvement.")
            return True

        return False

    def _flush_csv(self) -> None:
        """Write all rows to CSV (overwrite each time for safety)."""
        with self.log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADER)
            writer.writeheader()
            writer.writerows(self._rows)

    def summary(self) -> dict:
        """Return best metrics achieved during training."""
        return {
            "best_gnn_approx_ratio": self._best_ratio,
            "total_epochs_run":      len(self._rows),
            "early_stopped":         self._wait >= self.max_wait,
        }