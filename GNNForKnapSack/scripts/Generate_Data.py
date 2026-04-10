"""Generate Knapsack dataset using PuLP ILP solver."""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pulp

WEIGHT_LOW = 10
WEIGHT_HIGH = 1200
VALUE_MULT_LOW = 0.8
VALUE_MULT_HIGH = 1.3


def _build_instance(
    n_items: int, ins_index: int, n_instances: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int]:
    weights = rng.integers(WEIGHT_LOW, WEIGHT_HIGH + 1, size=n_items, dtype=np.int32)
    multipliers = rng.uniform(VALUE_MULT_LOW, VALUE_MULT_HIGH, size=n_items)
    values = np.maximum((multipliers * weights).astype(np.int32), 1)
    capacity = int(((ins_index + 1) / float(n_instances + 1)) * weights.sum())
    capacity = max(capacity, int(weights.max()))
    return weights, values, capacity


def solve_knapsack_ilp(
    weights: np.ndarray, values: np.ndarray, capacity: int,
    time_limit: int = 30,
) -> Optional[Tuple[np.ndarray, int]]:
    n = len(weights)
    prob = pulp.LpProblem("Knapsack", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat=pulp.LpBinary) for i in range(n)]
    prob += pulp.lpSum(int(values[i]) * x[i] for i in range(n))
    prob += pulp.lpSum(int(weights[i]) * x[i] for i in range(n)) <= int(capacity)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    prob.solve(solver)
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    solution = np.array([int(round(pulp.value(x[i]))) for i in range(n)], dtype=np.int8)
    return solution, int(pulp.value(prob.objective))


def generate_dataset(
    out_dir: Path, n_samples: int, min_items: int, max_items: int,
    n_instances: int, seed: int, max_attempts: int = 20,
    time_limit: int = 30, verbose: bool = True,
) -> None:
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
                print(f" [warn] instance {i} attempt {attempt+1} not Optimal, retrying...")

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
                print(f" [skip] instance {i} ILP exceeds capacity (bug)")
            continue

        np.savez_compressed(
            out_dir / f"instance_{i:04d}.npz",
            weights=weights, values=values,
            capacity=np.int32(capacity),
            solution=solution, dp_value=np.int32(opt_value),
        )

        if verbose and ((i + 1) % 50 == 0 or i == 0):
            elapsed = time.perf_counter() - start
            rate = (i + 1) / elapsed if elapsed > 0 else float("inf")
            eta = (n_samples - i - 1) / rate if rate > 0 else float("inf")
            print(f"[{i+1}/{n_samples}] elapsed={elapsed:.1f}s | ETA={eta:.1f}s | "
                  f"n={n_items} cap={capacity} val={opt_value}")

    total_time = time.perf_counter() - start
    meta = {
        "seed": seed, "num_instances": n_samples,
        "failed_instances": failed,
        "successful_instances": n_samples - failed,
        "min_items": min_items, "max_items": max_items,
        "n_instances_tiers": n_instances,
        "weight_range": [WEIGHT_LOW, WEIGHT_HIGH],
        "value_mult_range": [VALUE_MULT_LOW, VALUE_MULT_HIGH],
        "solver": "PuLP/CBC (ILP)",
        "total_time_sec": round(total_time, 3),
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone: {n_samples - failed}/{n_samples} instances → {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 0/1 knapsack dataset using PuLP ILP solver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("samples", type=int)
    parser.add_argument("min", type=int)
    parser.add_argument("max", type=int)
    parser.add_argument("-p", "--path", type=Path, default=Path("data/knapsack_ilp"))
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-i", "--instances", type=int, default=100)
    parser.add_argument("--max_attempts", type=int, default=20)
    parser.add_argument("--time_limit", type=int, default=30)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min <= 0:
        raise ValueError("Min must be positive")
    if args.min > args.max:
        raise ValueError("Min can't exceed max")
    generate_dataset(
        out_dir=args.path, n_samples=args.samples,
        min_items=args.min, max_items=args.max,
        n_instances=args.instances, seed=args.seed,
        max_attempts=args.max_attempts, time_limit=args.time_limit,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()