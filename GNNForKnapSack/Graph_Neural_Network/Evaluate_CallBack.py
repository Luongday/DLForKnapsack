"""Post-epoch evaluation callback for KnapsackGNN training.

Improvements vs original:
    - Uses centralized decode_utils (no duplicate greedy decode)
    - Per-instance ratio computation (correct) before averaging
    - Tracks best_epoch for summary
    - Compares with Greedy baseline each epoch
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

try:
    from model import KnapsackGNN, save_checkpoint
    from decode_utils import greedy_feasible_decode, greedy_ratio_decode
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.model import KnapsackGNN, save_checkpoint
    from GNNForKnapSack.decode_utils import greedy_feasible_decode, greedy_ratio_decode


class EvaluateCallback:
    """Evaluate GNN and Greedy baseline after each training epoch.

    Tracks approximation ratio (GNN value / DP optimal value).
    Saves model when ratio improves. Early stops if no improvement.

    Prerequisite: batch.y must be binary DP-optimal vector (0/1 per node).

    Args:
        model:      KnapsackGNN being trained.
        loader:     DataLoader for validation split.
        device:     torch.device.
        save_path:  Path to save best model checkpoint.
        max_wait:   Epochs without improvement before early stop.
        log_path:   CSV path for per-epoch metrics (None = no log).
    """

    HEADER = [
        "epoch",
        "gnn_approx_ratio_mean", "gnn_approx_ratio_std",
        "greedy_approx_ratio_mean",
        "gnn_feasibility", "greedy_feasibility",
        "avg_gnn_value", "avg_dp_value",
        "epoch_time_sec",
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

        self._best_ratio: float = 0.0
        self._best_epoch: int   = 0
        self._wait:       int   = 0
        self._rows:       List[dict] = []

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def on_epoch_end(self, epoch: int) -> bool:
        """Run evaluation. Returns True if training should stop (early stopping)."""
        t0 = time.perf_counter()
        self.model.eval()

        gnn_ratios:      List[float] = []
        greedy_ratios:   List[float] = []
        gnn_feas_list:   List[bool]  = []
        greedy_feas_list:List[bool]  = []
        gnn_values:      List[float] = []
        dp_values:       List[float] = []

        for batch in self.loader:
            batch    = batch.to(self.device)
            batch_v  = batch.batch if hasattr(batch, "batch") else \
                       torch.zeros(batch.num_nodes, dtype=torch.long, device=self.device)
            n_graphs = int(batch_v.max().item()) + 1

            logits = self.model(batch)
            probs  = torch.sigmoid(logits)

            for g in range(n_graphs):
                mask  = batch_v == g
                p_g   = probs[mask].detach().cpu()
                w_g   = batch.wts[mask].detach().cpu()
                v_g   = batch.vals[mask].detach().cpu()
                cap_g = float(batch.cap[g].item() if batch.cap.dim() > 0
                              else batch.cap.item())

                dp_sol = batch.y[mask].view(-1).detach().cpu()
                dp_val = float((dp_sol * v_g).sum())

                # GNN decode (centralized)
                gnn_sel = greedy_feasible_decode(p_g, w_g, cap_g)
                gnn_val = float((gnn_sel * v_g).sum())
                gnn_w   = float((gnn_sel * w_g).sum())

                # Greedy baseline decode (centralized)
                gr_sel  = greedy_ratio_decode(v_g, w_g, cap_g)
                gr_val  = float((gr_sel * v_g).sum())
                gr_w    = float((gr_sel * w_g).sum())

                # Per-instance ratio
                gnn_ratio    = gnn_val / dp_val if dp_val > 0 else 0.0
                greedy_ratio = gr_val  / dp_val if dp_val > 0 else 0.0

                gnn_ratios.append(gnn_ratio)
                greedy_ratios.append(greedy_ratio)
                gnn_feas_list.append(gnn_w <= cap_g + 1e-6)
                greedy_feas_list.append(gr_w <= cap_g + 1e-6)
                gnn_values.append(gnn_val)
                dp_values.append(dp_val)

        # Aggregate
        gnn_ratio_mean    = float(np.mean(gnn_ratios))    if gnn_ratios    else 0.0
        gnn_ratio_std     = float(np.std(gnn_ratios))     if gnn_ratios    else 0.0
        greedy_ratio_mean = float(np.mean(greedy_ratios)) if greedy_ratios else 0.0
        gnn_feas          = float(np.mean(gnn_feas_list))
        greedy_feas       = float(np.mean(greedy_feas_list))
        avg_gnn           = float(np.mean(gnn_values))
        avg_dp            = float(np.mean(dp_values))

        elapsed = time.perf_counter() - t0

        print(
            f"  Eval epoch {epoch+1:03d} | "
            f"GNN ratio={gnn_ratio_mean:.4f}±{gnn_ratio_std:.3f} | "
            f"Greedy ratio={greedy_ratio_mean:.4f} | "
            f"GNN feas={gnn_feas:.3f} | "
            f"time={elapsed:.2f}s"
        )

        if gnn_ratio_mean > self._best_ratio:
            self._best_ratio = gnn_ratio_mean
            self._best_epoch = epoch + 1
            self._wait       = 0
            save_checkpoint(self.model, self.save_path)
            print(f"  [callback] Best ratio → {gnn_ratio_mean:.4f} "
                  f"(epoch {epoch+1}). Saved to {self.save_path}")
        else:
            self._wait += 1
            print(f"  [callback] No improvement "
                  f"({self._wait}/{self.max_wait}, best={self._best_ratio:.4f} "
                  f"@ epoch {self._best_epoch})")

        row = {
            "epoch":                    epoch + 1,
            "gnn_approx_ratio_mean":    round(gnn_ratio_mean,    4),
            "gnn_approx_ratio_std":     round(gnn_ratio_std,     4),
            "greedy_approx_ratio_mean": round(greedy_ratio_mean, 4),
            "gnn_feasibility":          round(gnn_feas,          3),
            "greedy_feasibility":       round(greedy_feas,       3),
            "avg_gnn_value":            round(avg_gnn,           2),
            "avg_dp_value":             round(avg_dp,            2),
            "epoch_time_sec":           round(elapsed,           2),
        }
        self._rows.append(row)
        if self.log_path:
            self._flush_csv()

        if self._wait >= self.max_wait:
            print(
                f"  [callback] Early stopping. "
                f"Best: ratio={self._best_ratio:.4f} @ epoch {self._best_epoch}."
            )
            return True

        return False

    def _flush_csv(self) -> None:
        with self.log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADER)
            writer.writeheader()
            writer.writerows(self._rows)

    def summary(self) -> dict:
        return {
            "best_gnn_approx_ratio": self._best_ratio,
            "best_epoch":            self._best_epoch,
            "total_epochs_run":      len(self._rows),
            "early_stopped":         self._wait >= self.max_wait,
        }