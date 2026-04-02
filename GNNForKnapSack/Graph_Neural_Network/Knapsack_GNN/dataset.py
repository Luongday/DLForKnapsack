"""PyG dataset classes for 0/1 Knapsack GNN training.

Two dataset classes:
    KnapsackDataset            — loads from Excel (small problems, demo)
    GeneratedKnapsack01Dataset — loads from NPZ files (main training)

Both produce PyG Data with node feature schema (6-dim):
    x          [n, 6]  (w_norm, v_norm, ratio_norm, cap_ratio, cap_util, item_frac)
    edge_index [2, E]  kNN sparse graph
    y          [n, 1]  binary 0/1 DP-optimal selection
    wts        [n]     raw weights (float32)
    vals       [n]     raw values  (float32)
    cap        [1]     raw capacity (float32)

Improvements vs original:
    - Feature dimension 4 → 6 (matches improved Graph_builder)
    - build_sparse_graph delegates to Graph_builder (single source of truth)
    - split_dataset_by_instances guarantees non-empty val/test
    - _LazyKnapsackDataset for large datasets that don't fit in RAM
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data, InMemoryDataset

# Centralized imports — single source of truth
try:
    from .Dp import solve_knapsack_dp
    from .Graph_builder import build_knapsack_graph
    from .io_excel import load_excel_knapsack_instances
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Dp import solve_knapsack_dp
    from Graph_builder import build_knapsack_graph
    from io_excel import load_excel_knapsack_instances


# ---------------------------------------------------------------------------
# Shared graph builder (public — used by eval scripts too)
# ---------------------------------------------------------------------------

def build_sparse_graph(
    weights:  Sequence[int],
    values:   Sequence[int],
    capacity: int,
    solution: Sequence[int],
    k:        int  = 16,
    use_knn:  bool = True,
) -> Data:
    """Convert one Knapsack instance into a sparse PyG Data object.

    Delegates to Graph_builder.build_knapsack_graph (single source of truth).

    Returns:
        PyG Data with x=[n,6], edge_index, y, wts, vals, cap.
    """
    if use_knn:
        return build_knapsack_graph(weights, values, capacity, solution, k=k)

    # Ring fallback for very small graphs
    w   = torch.tensor(weights,  dtype=torch.float32)
    v   = torch.tensor(values,   dtype=torch.float32)
    sol = torch.tensor(solution, dtype=torch.float32)
    n   = len(weights)

    ratio    = v / (w + 1e-8)
    w_norm   = w / (w.max() + 1e-8)
    v_norm   = v / (v.max() + 1e-8)
    r_norm   = ratio / (ratio.max() + 1e-8)
    cap_ratio = w / (float(capacity) + 1e-8)
    cap_util  = torch.full((n,), float(capacity) / (w.sum().item() + 1e-8))
    item_frac = torch.full((n,), 1.0 / max(n, 1))

    x = torch.stack([w_norm, v_norm, r_norm, cap_ratio, cap_util, item_frac], dim=1)

    k_eff = min(k, max(1, n - 1))
    src   = torch.arange(n).repeat_interleave(k_eff)
    offs  = torch.arange(1, k_eff + 1) % n
    dst   = (torch.arange(n).unsqueeze(1) + offs) % n
    edge_index = torch.stack([src, dst.reshape(-1)], dim=0)

    return Data(
        x=x, edge_index=edge_index,
        y=sol.unsqueeze(1),
        wts=w, vals=v,
        cap=torch.tensor([capacity], dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Excel-based dataset
# ---------------------------------------------------------------------------

class KnapsackDataset(InMemoryDataset):
    """In-memory dataset from Excel instances — for demo/small problems."""

    def __init__(self, excel_path: str, k: int = 16, transform=None, pre_transform=None):
        self._excel_path = excel_path
        self._k = k
        super().__init__(root=".", transform=transform, pre_transform=pre_transform)
        self.data, self.slices = self._generate()

    def _generate(self):
        data_list = []
        for inst in load_excel_knapsack_instances(self._excel_path):
            w, v, c = inst["weights"], inst["values"], inst["capacity"]
            sol = solve_knapsack_dp(w, v, c)
            data_list.append(build_sparse_graph(w, v, c, sol, k=self._k))
        return self.collate(data_list)


# ---------------------------------------------------------------------------
# Lazy dataset
# ---------------------------------------------------------------------------

class _LazyKnapsackDataset(Dataset):
    """Loads each NPZ on demand — low RAM usage for large datasets."""

    def __init__(self, files: List[Path], k: int, use_knn: bool):
        self._files   = files
        self._k       = k
        self._use_knn = use_knn

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> Data:
        with np.load(self._files[idx]) as arrs:
            weights  = arrs["weights"].tolist()
            values   = arrs["values"].tolist()
            capacity = int(arrs["capacity"].item())
            solution = arrs["solution"].tolist()
        return build_sparse_graph(
            weights, values, capacity, solution,
            k=self._k, use_knn=self._use_knn,
        )


# ---------------------------------------------------------------------------
# NPZ-based dataset (main training)
# ---------------------------------------------------------------------------

class GeneratedKnapsack01Dataset(InMemoryDataset):
    """Dataset for generated NPZ knapsack instances.

    Compatible with data_generate_01.py and Generate_Data.py output.
    """

    def __init__(
        self,
        root_dir:  Union[str, Path],
        k:         int  = 16,
        use_knn:   bool = True,
        use_cache: bool = False,
        transform=None,
        pre_transform=None,
    ):
        self.root_dir  = Path(root_dir)
        self.k         = k
        self.use_knn   = use_knn
        self.use_cache = use_cache
        self._cache_path = self.root_dir / "processed_dataset.pt"
        super().__init__(root=".", transform=transform, pre_transform=pre_transform)
        self.data, self.slices = self._load_or_generate()

    def _npz_files(self) -> List[Path]:
        files = sorted(self.root_dir.glob("instance_*.npz"))
        if not files:
            raise FileNotFoundError(f"No instance_*.npz in {self.root_dir}")
        return files

    def _load_or_generate(self):
        if self._cache_path.exists() and not self.use_cache:
            try:
                self._cache_path.unlink()
            except OSError:
                pass

        data_list: List[Data] = []
        for path in self._npz_files():
            with np.load(path) as arrs:
                weights  = arrs["weights"].tolist()
                values   = arrs["values"].tolist()
                capacity = int(arrs["capacity"].item())
                solution = arrs["solution"].tolist()
            data_list.append(
                build_sparse_graph(weights, values, capacity, solution,
                                   k=self.k, use_knn=self.use_knn)
            )
        return self.collate(data_list)

    @classmethod
    def get_lazy(cls, root_dir: Union[str, Path], k: int = 16, use_knn: bool = True):
        """Return lazy Dataset that loads NPZ on demand."""
        root_dir = Path(root_dir)
        files = sorted(root_dir.glob("instance_*.npz"))
        if not files:
            raise FileNotFoundError(f"No instance_*.npz in {root_dir}")
        return _LazyKnapsackDataset(files, k=k, use_knn=use_knn)


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_dataset_by_instances(
    dataset:     Union[InMemoryDataset, Dataset],
    train_ratio: float         = 0.8,
    val_ratio:   float         = 0.1,
    shuffle:     bool          = False,
    seed:        Optional[int] = None,
) -> Tuple[Subset, Subset, Subset]:
    """Split dataset into train / val / test subsets.

    Guarantees at least 1 sample in val and test when dataset >= 3.
    """
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0 <= val_ratio < 1):
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    n       = len(dataset)
    indices = list(range(n))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    if n >= 3:
        n_val   = max(1, n_val)
        n_test  = max(1, n_test)
        n_train = n - n_val - n_test
        while n_train < 1 and (n_val > 1 or n_test > 1):
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            n_train = n - n_val - n_test

    train_idx = indices[:n_train]
    val_idx   = indices[n_train: n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)