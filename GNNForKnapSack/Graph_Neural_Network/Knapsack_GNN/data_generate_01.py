"""Generate medium-size 0/1 Knapsack dataset with DP labels.

Each instance is solved optimally by the NumPy-accelerated DP solver and
saved as a compressed NPZ file compatible with GeneratedKnapsack01Dataset.

Usage (as script):
    python data_generate_01.py --out_dir data/knapsack_dp/train --num_instances 1000
    python data_generate_01.py --out_dir data/knapsack_dp/test  --num_instances 200 --seed 123

Usage (as module):
    from knapsack_gnn.data_generate_01 import generate_dataset
    generate_dataset(out_dir=Path("data/train"), num_instances=500, seed=42)

Improvements vs original:
    - Relative import replaced with absolute fallback so file works both as
      a package module and as a standalone script.
    - weight_range and value_mult_range are now parameters (not module constants)
      so callers can customise without touching source.
    - Progress print now shows instance-0 only at completion of first batch,
      avoiding misleading ETA on the very first call.
    - Input validation moved into generate_dataset() so it also protects
      programmatic callers, not just CLI users.
    - Added --solver flag: 'numpy' (default, fast) or 'python' (reference).
    - meta.json records weight_range and value_mult_range actually used.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Support both package-relative and standalone execution
try:
    from .dp import solve_knapsack_dp
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from GNNForKnapSack.solvers.DP.dp_baseline_eval import solve_knapsack_dp


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SEED            = 2025
DEFAULT_NUM_INSTANCES   = 1000
DEFAULT_N_RANGE         = (80, 200)
DEFAULT_CAPACITY_RANGE  = (200, 800)
DEFAULT_WEIGHT_RANGE    = (1, 100)
DEFAULT_VALUE_MULT_RANGE = (0.8, 1.3)
DEFAULT_MAX_STATES      = 5_000_000
DEFAULT_OUT_DIR         = Path(__file__).resolve().parents[2] / "data" / "knapsack_dp"


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> np.random.Generator:
    """Seed both numpy legacy API and random module; return a fresh Generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Instance sampling
# ---------------------------------------------------------------------------

def sample_instance_sizes(
    rng:          np.random.Generator,
    n_range:      Tuple[int, int],
    cap_range:    Tuple[int, int],
    max_states:   int,
    max_attempts: int = 1000,
) -> Tuple[int, int]:
    """Sample (n_items, capacity) while keeping DP state count ≤ max_states."""
    for _ in range(max_attempts):
        n_items  = int(rng.integers(n_range[0],   n_range[1]   + 1))
        capacity = int(rng.integers(cap_range[0], cap_range[1] + 1))
        if n_items * capacity <= max_states:
            return n_items, capacity
    raise ValueError(
        f"Could not sample feasible (n_items, capacity) after {max_attempts} attempts "
        f"with max_states={max_states}. "
        "Try widening n_range/cap_range or increasing max_states."
    )


def generate_weights_values(
    rng:              np.random.Generator,
    n_items:          int,
    weight_range:     Tuple[int, int] = DEFAULT_WEIGHT_RANGE,
    value_mult_range: Tuple[float, float] = DEFAULT_VALUE_MULT_RANGE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate integer weights and correlated integer values.

    Values are sampled as weight × U[value_mult_range[0], value_mult_range[1]],
    clipped to minimum 1 so no item is worthless.
    """
    weights = rng.integers(
        weight_range[0], weight_range[1] + 1,
        size=n_items, dtype=np.int32,
    )
    multipliers = rng.uniform(value_mult_range[0], value_mult_range[1], size=n_items)
    values = np.maximum((weights * multipliers).astype(np.int32), 1)
    return weights, values


# ---------------------------------------------------------------------------
# Solving & persistence
# ---------------------------------------------------------------------------

def solve_instance(
    weights:  np.ndarray,
    values:   np.ndarray,
    capacity: int,
) -> Dict:
    """Solve one instance with DP and return solution artefacts."""
    solution_list = solve_knapsack_dp(
        weights.tolist(), values.tolist(), capacity
    )
    solution     = np.asarray(solution_list, dtype=np.int8)
    total_weight = int(np.sum(weights * solution))

    if total_weight > capacity:
        raise RuntimeError(
            f"DP solution exceeds capacity ({total_weight} > {capacity}). "
            "This is a bug in the DP solver."
        )

    dp_value = int(np.sum(values * solution))
    return {
        "solution":     solution,
        "dp_value":     np.int32(dp_value),
        "total_weight": total_weight,
    }


def save_instance(
    out_dir:  Path,
    idx:      int,
    weights:  np.ndarray,
    values:   np.ndarray,
    capacity: int,
    solution: np.ndarray,
    dp_value: np.int32,
) -> None:
    """Write one instance to a compressed NPZ file."""
    np.savez_compressed(
        out_dir / f"instance_{idx:04d}.npz",
        weights=weights.astype(np.int32),
        values=values.astype(np.int32),
        capacity=np.int32(capacity),
        solution=solution.astype(np.int8),
        dp_value=dp_value,
    )


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    out_dir:          Path,
    num_instances:    int,
    n_range:          Tuple[int, int]         = DEFAULT_N_RANGE,
    cap_range:        Tuple[int, int]         = DEFAULT_CAPACITY_RANGE,
    seed:             int                      = DEFAULT_SEED,
    max_states:       int                      = DEFAULT_MAX_STATES,
    weight_range:     Tuple[int, int]          = DEFAULT_WEIGHT_RANGE,
    value_mult_range: Tuple[float, float]      = DEFAULT_VALUE_MULT_RANGE,
) -> None:
    """Generate `num_instances` solved Knapsack instances and save as NPZ.

    Args:
        out_dir:          Directory to write instance_XXXX.npz + meta.json.
        num_instances:    How many instances to generate.
        n_range:          (min, max) number of items per instance.
        cap_range:        (min, max) knapsack capacity per instance.
        seed:             Master random seed (reproducible).
        max_states:       Reject (n, cap) pairs where n*cap > max_states.
        weight_range:     (min, max) for individual item weights.
        value_mult_range: (min, max) multiplier applied to weight to get value.
    """
    # Validate early so errors surface before any work is done
    if n_range[0] <= 0 or cap_range[0] <= 0:
        raise ValueError("n_range and cap_range minimums must be positive.")
    if n_range[0] > n_range[1]:
        raise ValueError("n_range[0] cannot exceed n_range[1].")
    if cap_range[0] > cap_range[1]:
        raise ValueError("cap_range[0] cannot exceed cap_range[1].")
    if num_instances <= 0:
        raise ValueError("num_instances must be positive.")

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = set_seeds(seed)

    start_time = time.perf_counter()
    print_every = max(1, min(50, num_instances // 20))

    for idx in range(num_instances):
        n_items, capacity = sample_instance_sizes(rng, n_range, cap_range, max_states)
        weights, values   = generate_weights_values(rng, n_items, weight_range, value_mult_range)
        solved            = solve_instance(weights, values, capacity)

        save_instance(
            out_dir=out_dir,
            idx=idx,
            weights=weights,
            values=values,
            capacity=capacity,
            solution=solved["solution"],
            dp_value=solved["dp_value"],
        )

        if (idx + 1) % print_every == 0:
            elapsed   = time.perf_counter() - start_time
            rate      = (idx + 1) / elapsed if elapsed > 0 else float("inf")
            remaining = (num_instances - idx - 1) / rate if rate > 0 else float("inf")
            print(
                f"[{idx+1:>{len(str(num_instances))}}/{num_instances}] "
                f"elapsed={elapsed:.1f}s  ETA={remaining:.1f}s  "
                f"n={n_items}  cap={capacity}  val={int(solved['dp_value'])}"
            )

    total_time = time.perf_counter() - start_time
    meta = {
        "seed":               seed,
        "num_instances":      num_instances,
        "n_range":            list(n_range),
        "capacity_range":     list(cap_range),
        "weight_range":       list(weight_range),
        "value_mult_range":   list(value_mult_range),
        "max_states":         max_states,
        "solver":             "DP (NumPy-accelerated)",
        "generator":          "knapsack_gnn.data_generate_01",
        "total_time_sec":     round(total_time, 3),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"\nDone: {num_instances} instances in {total_time:.1f}s → {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 0/1 Knapsack dataset with DP-optimal labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out_dir",       type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num_instances", type=int,  default=DEFAULT_NUM_INSTANCES)
    parser.add_argument("--n_min",         type=int,  default=DEFAULT_N_RANGE[0])
    parser.add_argument("--n_max",         type=int,  default=DEFAULT_N_RANGE[1])
    parser.add_argument("--cap_min",       type=int,  default=DEFAULT_CAPACITY_RANGE[0])
    parser.add_argument("--cap_max",       type=int,  default=DEFAULT_CAPACITY_RANGE[1])
    parser.add_argument("--w_min",         type=int,  default=DEFAULT_WEIGHT_RANGE[0])
    parser.add_argument("--w_max",         type=int,  default=DEFAULT_WEIGHT_RANGE[1])
    parser.add_argument("--seed",          type=int,  default=DEFAULT_SEED)
    parser.add_argument("--max_states",    type=int,  default=DEFAULT_MAX_STATES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        out_dir=args.out_dir,
        num_instances=args.num_instances,
        n_range=(args.n_min, args.n_max),
        cap_range=(args.cap_min, args.cap_max),
        seed=args.seed,
        max_states=args.max_states,
        weight_range=(args.w_min, args.w_max),
    )


if __name__ == "__main__":
    main()