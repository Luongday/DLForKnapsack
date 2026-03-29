from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pulp

WEIGHT_LOW = 1
WEIGHT_HIGH = 1000
VALUE_MULT_LOW = 0.8
VALUE_MULT_HIGH = 1.3

def _build_instance(
    n_items: int,
    ins_index: int,
    n_instances: int,
    rng: np.random.Generator,
)-> Tuple[np.ndarray, np.ndarray, int]:
    """Generate weights, values, and capacity for one Knapsack instance.

        Capacity strategy (preserved from original):
            c = (ins_index + 1) / (n_instances + 1) * sum(weights)
        This creates a spectrum of difficulty — early instances are tight
        (low capacity ratio), later instances are loose (high capacity ratio).

        Values are correlated with weights via a random multiplier in
        [VALUE_MULT_LOW, VALUE_MULT_HIGH], giving realistic value/weight ratios
        instead of the original hardcoded f[i] = weight + 100.

        Args:
            n_items:     Number of items in this instance.
            ins_index:   Index of this instance within the current batch (0-based).
            n_instances: Total number of instances — used to set capacity tier.
            rng:         NumPy random Generator for reproducibility.

        Returns:
            weights, values (both int32 arrays of length n_items), capacity (int).
    """
    weights = rng.integers(WEIGHT_LOW, WEIGHT_HIGH + 1, size=n_items, dtype=np.int32)

    multipliers = rng.uniform(VALUE_MULT_LOW, VALUE_MULT_HIGH, size=n_items)
    values = np.maximum((multipliers * weights).astype(np.int32), 1)

    capacity = int(((ins_index + 1) / float(n_instances + 1)) * weights.sum())

    capacity = max(capacity, int(weights.max()))

    return weights, values, capacity

def solve_knapsack_ilp(
    weights: np.ndarray,
    values: np.ndarray,
    capacity: int,
    time_limit: int = 30,
) -> Optional[Tuple[np.ndarray, int]]:
    """Solve a 0/1 Knapsack instance via ILP using PuLP + CBC.

        Args:
            weights:    Item weights (int32 array).
            values:     Item values  (int32 array).
            capacity:   Knapsack capacity.
            time_limit: CBC solver time limit in seconds.

        Returns:
            (solution, optimal_value) if Optimal status reached, else None.
            solution is a 0/1 int8 array of length n.
    """
    n = len(weights)
    prob = pulp.LpProblem("Knapsack", pulp.LpMaximize)

    x = [pulp.LpVariable(f"x{i}", cat=pulp.LpBinary) for i in range(n)]

    # Maximise total value
    prob += pulp.lpSum(int(values[i]) * x[i] for i in range(n))

    # Total weight <= capacity
    prob += pulp.lpSum(int(weights[i]) * x[i] for i in range(n)) <= int(capacity)

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    solution = np.array(
        [int(round(pulp.value(x[i]))) for i in range(n)], dtype=np.int8
    )
    opt_value = int(pulp.value(prob.objective))

    return solution, opt_value

def generate_dataset(
    out_dir: Path,
    n_samples: int,
    min_items: int,
    max_items: int,
    n_instances: int,
    seed: int,
    max_attempts: int = 20,
    time_limit: int = 30,
    verbose: bool = True,
) -> None:
    """Generate a dataset of solved Knapsack instances saved as NPZ files.

        Output format (per instance_XXXX.npz) is identical to data_generate_01.py:
            weights   int32[n]  item weights
            values    int32[n]  item values
            capacity  int32     knapsack capacity
            solution  int8[n]   0/1 ILP-optimal selection
            dp_value  int32     optimal objective value

        The field name dp_value is kept for schema compatibility even though the
        solver here is ILP (PuLP/CBC) rather than DP — both produce optimal solutions
        for the 0/1 Knapsack problem.

        Args:
            out_dir:      Directory to write NPZ files and meta.json.
            n_samples:    Number of instances to generate.
            min_items:    Minimum number of items per instance.
            max_items:    Maximum number of items per instance.
            n_instances:  Number of capacity tiers (controls difficulty spread).
            seed:         Random seed.
            max_attempts: Max retries if ILP fails to reach Optimal status.
            time_limit:   CBC solver time limit per instance (seconds).
            verbose:      Print progress to stdout.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    failed = 0
    start = time.perf_counter()

    for i in range(n_samples):
        n_items = int(rng.integers(min_items, max_items + 1))
        ins_index = i % n_instances

        solved = None
        for attempt in range(max_attempts):
            weights, values, capacity = _build_instance(n_items, ins_index, n_instances, rng)
            result = solve_knapsack_ilp(weights, values, capacity, time_limit=time_limit)
            if result is not None:
                solved = result
                break
            if verbose:
                print(f" [warn] instance {i} attempt {attempt + 1} not Optimal, retrying...")

        if solved is None:
            failed += 1
            if verbose:
                print(f" [skip] instance {i} failed after {max_attempts} attempts.")
            continue

        solution, opt_value = solved

        total_weight = int((weights * solution).sum())
        if total_weight > capacity:
            failed += 1
            if verbose:
                print(f" [skip] instance {i} ILP solution exceeds capacity (bug)")
            continue

        out_path = out_dir / f"instance_{i:04d}.npz"
        np.savez_compressed(
            out_path,
            weights=weights,
            values=values,
            capacity=np.int32(capacity),
            solution=solution,
            dp_value=np.int32(opt_value),
        )

        if verbose and ((i + 1) % 50 == 0 or i == 0):
            elapsed = time.perf_counter() - start
            rate = (i + 1) / elapsed if elapsed > 0 else float("inf")
            eta = (n_samples - i - 1) / rate if rate > 0 else float("inf")
            print(
                f"[{i + 1}/{n_samples}] elapsed={elapsed:.1f}s | ETA={eta:.1f}s | "
                f"n={n_items} cap={capacity} val={opt_value}"
            )
    total_time = time.perf_counter() - start

    meta = {
        "seed": seed,
        "num_instances": n_samples,
        "failed_instances": failed,
        "successful_instances": n_samples - failed,
        "min_items": min_items,
        "max_items": max_items,
        "n_instances_tiers": n_instances,
        "weight_range": [WEIGHT_LOW, WEIGHT_HIGH],
        "value_mult_range": [VALUE_MULT_LOW, VALUE_MULT_HIGH],
        "solver": "PuLP/CBC (ILP)",
        "total_time_sec": round(total_time, 3),
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=42)

    print(f"\nDone: {n_samples - failed}/{n_samples} instances saved to {out_dir}")
    print(f"Total time: {total_time:.1f}s | Failed: {failed}")


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 0/1 knapsack dataset using PuLP ILP solver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "samples", type=int,
        help="Number of instances to generate",
    )
    parser.add_argument(
        "min", type=int,
        help="Minimum number of items per instances",
    )
    parser.add_argument(
        "max", type=int,
        help="Maximum number of items per instances",
    )
    parser.add_argument(
        "-p", "--path", type=Path,
        default=Path("data"),
        help="Output directory for NPZ files and meta.json",
    )
    parser.add_argument(
        "-s", "--seed", type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "-i", "--instances", type=int,
        default=100,
        help="Number of capacity tiers (controls difficulty spread",
    )
    parser.add_argument(
        "--max_attempts", type=int,
        default=20,
        help="Max retries per instances if ILP fails",
    )
    parser.add_argument(
        "--time_limit", type=int,
        default=30,
        help="CBC solver time limit per instances (seconds)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.min <= 0:
        raise ValueError("Min must be positive")
    if args.min > args.max:
        raise ValueError("Min can't exceed max")
    if args.samples <= 0:
        raise ValueError("Samples must be positive")
    generate_dataset(
        out_dir=args.path,
        n_samples=args.samples,
        min_items=args.min,
        max_items=args.max,
        n_instances=args.instances,
        seed=args.seed,
        max_attempts=args.max_attempts,
        time_limit=args.time_limit,
        verbose=args.quiet,
    )

if __name__ == "__main__":
    main()
