"""Shared training logger for DQN and S2V-DQN."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional


class RLTrainingLogger:
    """Append-only CSV logger for RL training progress."""

    HEADER = [
        "step", "updates", "epsilon", "loss",
        "avg_value_val", "best_value_val",
        "buffer_size", "elapsed_sec",
    ]

    def __init__(self, log_path: Optional[Path]):
        self.log_path = log_path
        self.rows: List[dict] = []
        self._header_written = False

        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Reset file at start of training
            if log_path.exists():
                log_path.unlink()

    def log(
        self,
        step:           int,
        updates:        int,
        epsilon:        float,
        loss:           Optional[float]   = None,
        avg_value_val:  Optional[float]   = None,
        best_value_val: Optional[float]   = None,
        buffer_size:    Optional[int]     = None,
        elapsed_sec:    Optional[float]   = None,
    ) -> None:
        row = {
            "step":           step,
            "updates":        updates,
            "epsilon":        round(epsilon, 4),
            "loss":           round(loss, 6) if loss is not None else "",
            "avg_value_val":  round(avg_value_val, 2) if avg_value_val is not None else "",
            "best_value_val": round(best_value_val, 2) if best_value_val is not None else "",
            "buffer_size":    buffer_size if buffer_size is not None else "",
            "elapsed_sec":    round(elapsed_sec, 1) if elapsed_sec is not None else "",
        }
        self.rows.append(row)

        if self.log_path is None:
            return

        # Rewrite entire file each time (safe for crash recovery)
        with self.log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADER)
            writer.writeheader()
            writer.writerows(self.rows)
