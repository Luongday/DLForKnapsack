"""Cyclic Learning Rate scheduler for PyTorch.

Ported from clr_callback.py (original Keras-based Neuro-Knapsack).
Original paper: https://arxiv.org/abs/1506.01186 (Smith, 2017)

Key changes vs original:
    - Rewritten as a PyTorch LRScheduler (no Keras dependency).
    - Three modes preserved: triangular, triangular2, exp_range.
    - Supports custom scale_fn same as original.
    - Compatible with standard PyTorch training loop.

Usage in run_train.py:
    from clr_scheduler import CyclicLR

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CyclicLR(
        optimizer,
        base_lr=1e-4,
        max_lr=1e-2,
        step_size=len(train_loader) * 2,   # 2 epochs per half-cycle
        mode="triangular2",
    )
    for epoch in range(epochs):
        for batch in train_loader:
            loss.backward()
            optimizer.step()
            scheduler.step()   # call after each batch
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CyclicLR(LRScheduler):
    """Cyclical Learning Rate scheduler.

    Cycles the learning rate between base_lr and max_lr with period
    2 * step_size iterations.

    Args:
        optimizer:   PyTorch optimizer.
        base_lr:     Lower boundary of LR cycle.
        max_lr:      Upper boundary of LR cycle.
        step_size:   Number of iterations per half-cycle.
                     Authors recommend 2–8 × (iterations per epoch).
        mode:        One of 'triangular', 'triangular2', 'exp_range'.
        gamma:       Used in 'exp_range' mode: scale = gamma^(iteration).
        scale_fn:    Custom scaling function (overrides mode).
                     Must be a callable: f(x) → [0, 1] for all x ≥ 0.
        scale_mode:  Whether scale_fn uses 'cycle' or 'iterations' as input.
        last_epoch:  Index of the last iteration (-1 for fresh start).
    """

    def __init__(
        self,
        optimizer:  Optimizer,
        base_lr:    float = 1e-4,
        max_lr:     float = 1e-2,
        step_size:  int   = 2000,
        mode:       str   = "triangular",
        gamma:      float = 1.0,
        scale_fn:   Optional[Callable[[float], float]] = None,
        scale_mode: str   = "cycle",
        last_epoch: int   = -1,
    ):
        self.base_lr   = base_lr
        self.max_lr    = max_lr
        self.step_size = step_size
        self.mode      = mode
        self.gamma     = gamma

        if scale_fn is None:
            if mode == "triangular":
                self.scale_fn   = lambda x: 1.0
                self.scale_mode = "cycle"
            elif mode == "triangular2":
                self.scale_fn   = lambda x: 1.0 / (2.0 ** (x - 1))
                self.scale_mode = "cycle"
            elif mode == "exp_range":
                self.scale_fn   = lambda x: gamma ** x
                self.scale_mode = "iterations"
            else:
                raise ValueError(
                    f"mode must be 'triangular', 'triangular2', or 'exp_range', got '{mode}'"
                )
        else:
            self.scale_fn   = scale_fn
            self.scale_mode = scale_mode

        super().__init__(optimizer, last_epoch=last_epoch)

    def _clr(self, iteration: float) -> float:
        """Compute current LR for a given iteration count."""
        cycle = math.floor(1 + iteration / (2 * self.step_size))
        x     = abs(iteration / self.step_size - 2 * cycle + 1)

        if self.scale_mode == "cycle":
            scale = self.scale_fn(cycle)
        else:
            scale = self.scale_fn(iteration)

        return self.base_lr + (self.max_lr - self.base_lr) * max(0.0, 1 - x) * scale

    def get_lr(self) -> list[float]:
        """Return LR for each param group (called by PyTorch internals)."""
        lr = self._clr(self.last_epoch)
        return [lr for _ in self.optimizer.param_groups]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_clr_scheduler(
    optimizer: Optimizer,
    train_loader_len: int,
    base_lr:     float = 1e-4,
    max_lr:      float = 1e-2,
    epochs_per_cycle: int = 4,
    mode:        str  = "triangular2",
) -> CyclicLR:
    """Create a CyclicLR with step_size derived from loader length.

    step_size = epochs_per_cycle / 2 × len(train_loader)
    (Authors recommend 2–8 epochs per half-cycle.)

    Args:
        optimizer:         PyTorch optimizer.
        train_loader_len:  len(train_loader) — iterations per epoch.
        base_lr:           Lower LR boundary.
        max_lr:            Upper LR boundary.
        epochs_per_cycle:  Full cycle length in epochs (default 4).
        mode:              'triangular', 'triangular2', or 'exp_range'.

    Returns:
        CyclicLR scheduler ready to use.

    Example:
        scheduler = make_clr_scheduler(optimizer, len(train_loader))
        # In training loop, call scheduler.step() after each batch.
    """
    step_size = (epochs_per_cycle // 2) * train_loader_len
    return CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size=step_size,
        mode=mode,
    )