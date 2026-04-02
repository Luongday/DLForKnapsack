"""Genetic Algorithm evaluation — CSV schema aligned with DP/GNN/Greedy.

Evaluates every instance_*.npz using GA solver, writes per-instance
results to CSV for merge_results.py comparison pipeline.

Usage:
    python Evaluate_GA.py
    python Evaluate_GA.py --dataset_dir data/knapsack_ilp/test --population 100 --generations 300
    python Evaluate_GA.py --dataset_dir data/knapsack_ilp/test --out_csv results/GA/ga_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List

import numpy as np

import sys
_HERE = Path(__file__).resolve().parent
for _p in [str(_HERE), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ..instance_loader import load_instance, list_instances
from ..solvers.GA.GA import KnapsackGA


def mark(msg: str) -> None:
    print(f"[GA-EVAL] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

def _default_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "GNNForKnapSack" / "data" / "knapsack_ilp" / "test"


def _default_out_csv() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "GA" / "ga_eval_results.csv"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GA solver on NPZ Knapsack dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir",  type=Path, default=_default_dataset_dir())
    parser.add_argument("--out_csv",      type=Path, default=_default_out_csv())
    parser.add_argument("--n",            type=int,  default=None,
                        help="Limit to first N instances (default: all)")

    # GA hyperparameters
    parser.add_argument("--population",   type=int,   default=100,
                        help="Population size")
    parser.add_argument("--generations",  type=int,   default=500,
                        help="Max generations")
    parser.add_argument("--mutation_rate",type=float,  default=0.05,
                        help="Per-gene mutation probability")
    parser.add_argument("--elite_ratio",  type=float,  default=0.2,
                        help="Fraction of population kept as elite")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--quiet",        action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    files = list_instances(args.dataset_dir, limit=args.n)
    mark(f"Found {len(files)} instances in {args.dataset_dir}")
    mark(f"GA config: pop={args.population} gen={args.generations} "
         f"mut={args.mutation_rate} elite={args.elite_ratio}")

    results: List[dict] = []
    total_start = time.perf_counter()

    for idx, path in enumerate(files):
        W, V, C = load_instance(path)
        n = len(W)

        t0 = time.perf_counter()

        ga = KnapsackGA(
            weights=W.astype(float),
            values=V.astype(float),
            capacity=float(C),
            population_size=args.population,
            mutation_rate=args.mutation_rate,
            max_generations=args.generations,
            elite_ratio=args.elite_ratio,
            seed=args.seed + idx,  # per-instance seed for reproducibility
        )
        solution, best_value, gens_run = ga.solve(verbose=False)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        total_weight = int((W * solution).sum())
        total_value  = int((V * solution).sum())
        feasible     = 1 if total_weight <= C else 0
        selected_idx = [int(i) for i in range(n) if solution[i] == 1]

        results.append({
            "instance_file":     path.name,
            "n_items":           n,
            "capacity":          float(C),
            "total_weight":      float(total_weight),
            "total_value":       float(total_value),
            "feasible":          feasible,
            "inference_time_ms": round(solve_ms, 4),
            "selected_items":    json.dumps(selected_idx),
            "generations_run":   gens_run,
        })

        if verbose and ((idx + 1) % 10 == 0 or idx == 0):
            elapsed = time.perf_counter() - total_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta  = (len(files) - idx - 1) / rate if rate > 0 else 0
            mark(f"[{idx+1}/{len(files)}] elapsed={elapsed:.1f}s ETA={eta:.1f}s  "
                 f"val={total_value}  wt={total_weight}/{C}  "
                 f"gens={gens_run}  time={solve_ms:.1f}ms")

    # Write CSV (same core schema as DP/GNN/Greedy + extra GA column)
    fieldnames = [
        "instance_file", "n_items", "capacity",
        "total_weight", "total_value",
        "feasible", "inference_time_ms", "selected_items",
        "generations_run",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    total_time = time.perf_counter() - total_start
    if results:
        avg_value = float(np.mean([r["total_value"]       for r in results]))
        avg_time  = float(np.mean([r["inference_time_ms"] for r in results]))
        feas_rate = float(np.mean([r["feasible"]          for r in results]))
        avg_gens  = float(np.mean([r["generations_run"]   for r in results]))
        mark(f"Done: {len(results)} instances in {total_time:.1f}s")
        mark(f"Avg value={avg_value:.2f} | Avg time={avg_time:.1f}ms | "
             f"Feasible={feas_rate:.3f} | Avg gens={avg_gens:.0f}")
    mark(f"Results → {args.out_csv}")


if __name__ == "__main__":
    main()