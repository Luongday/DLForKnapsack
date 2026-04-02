"""Centralized NPZ instance loader for all solvers.

Every evaluation script imports load_instance() from here instead of
defining its own version with slightly different key aliases.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_instance(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load weights, values, capacity from NPZ file.

    Supports key aliases from both data_generate_01.py and Generate_Data.py.

    Returns:
        (weights, values, capacity) — weights/values as int32 arrays.
    """
    arr = np.load(npz_path, allow_pickle=True)

    def pick(keys):
        for k in keys:
            if k in arr.files:
                return arr[k]
        return None

    W = pick(["weights", "w", "W"])
    V = pick(["values",  "v", "V"])
    C = pick(["capacity", "cap", "C"])

    if W is None or V is None or C is None:
        raise KeyError(
            f"Missing weights/values/capacity in {npz_path.name}, "
            f"found keys={arr.files}"
        )

    W = np.asarray(W).astype(np.int32).reshape(-1)
    V = np.asarray(V).astype(np.int32).reshape(-1)
    C = int(np.asarray(C).reshape(()))

    if W.shape != V.shape:
        raise ValueError(
            f"weights/values shape mismatch in {npz_path.name}: "
            f"{W.shape} vs {V.shape}"
        )
    return W, V, C


def load_instance_with_solution(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, int]:
    """Load instance + DP/ILP optimal solution if available.

    Returns:
        (weights, values, capacity, solution, dp_value)
        solution is int8 array, dp_value is int.
        If solution/dp_value not in file, returns zeros.
    """
    arr = np.load(npz_path, allow_pickle=True)

    def pick(keys):
        for k in keys:
            if k in arr.files:
                return arr[k]
        return None

    W, V, C = load_instance(npz_path)

    sol_raw = pick(["solution", "selected", "y"])
    opt_raw = pick(["dp_value", "optimal_value", "opt"])

    if sol_raw is not None:
        solution = np.asarray(sol_raw).reshape(-1).astype(np.int8)
    else:
        solution = np.zeros(len(W), dtype=np.int8)

    if opt_raw is not None:
        dp_value = int(np.asarray(opt_raw).reshape(()))
    else:
        dp_value = int((W * solution).sum()) if solution.any() else 0

    return W, V, C, solution, dp_value


def list_instances(dataset_dir: Path, limit: int = None) -> List[Path]:
    """List all instance_*.npz files in a directory, sorted."""
    files = sorted(dataset_dir.glob("instance_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No instance_*.npz files found in {dataset_dir}"
        )
    if limit:
        files = files[:limit]
    return files