"""GA baseline evaluation for 0/1 Knapsack.

Same structure as dp_baseline_eval.py. Same CSV schema for Merge_results.py.

Usage:
    python ga_baseline_eval.py --dataset_dir data/knapsack_ilp/test
    python ga_baseline_eval.py --dataset_dir data/knapsack_ilp/test --population 100 --generations 500
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

import sys
_HERE = Path(__file__).resolve().parent
for _p in [str(_HERE), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from GNNForKnapSack.instance_loader import load_instance, list_instances


def mark(msg: str) -> None:
    print(f"[GA-EVAL] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_knapsack_ga(
    weights: np.ndarray,
    values:  np.ndarray,
    capacity: int,
    population_size: int = 100,
    max_generations: int = 500,
    mutation_rate:   float = 0.05,
    elite_ratio:     float = 0.2,
    seed:            int = 42,
) -> List[int]:
    """Genetic Algorithm for 0/1 Knapsack with repair operator.

    Guaranteed 100% feasibility via greedy repair.

    Returns:
        List of selected item indices (sorted).
    """
    random.seed(seed)
    np.random.seed(seed)

    w = weights.astype(np.float64)
    v = values.astype(np.float64)
    cap = float(capacity)
    n = len(w)

    def repair(ind):
        ind = ind.copy()
        total_w = float((ind * w).sum())
        if total_w <= cap:
            return ind
        sel_idx = np.where(ind == 1)[0]
        ratios = v[sel_idx] / (w[sel_idx] + 1e-8)
        for i in sel_idx[np.argsort(ratios)]:
            if total_w <= cap:
                break
            ind[i] = 0
            total_w -= float(w[i])
        return ind

    # Init population
    pop = []
    for _ in range(population_size):
        ind = np.zeros(n, dtype=np.int8)
        order = np.random.permutation(n)
        rem = cap
        for i in order:
            if w[i] <= rem:
                ind[i] = 1
                rem -= w[i]
        pop.append(ind)

    # Evolve
    def batch_fitness(population):
        return np.stack(population) @ v

    fitness = batch_fitness(pop)
    best_idx = int(np.argmax(fitness))
    best_ind = pop[best_idx].copy()
    best_val = float(fitness[best_idx])
    no_improve = 0

    for gen in range(max_generations):
        n_elite = max(2, int(population_size * elite_ratio))
        elite_idx = np.argsort(fitness)[-n_elite:][::-1]
        elite = [pop[i] for i in elite_idx]

        children = []
        np.random.shuffle(elite)
        for i in range(0, len(elite) - 1, 2):
            point = random.randint(1, n - 1) if n > 1 else 0
            c1 = np.concatenate([elite[i][:point], elite[i+1][point:]])
            c2 = np.concatenate([elite[i+1][:point], elite[i][point:]])
            children.extend([c1, c2])
        if len(elite) % 2 == 1:
            point = random.randint(1, n - 1) if n > 1 else 0
            c1 = np.concatenate([elite[-1][:point], elite[0][point:]])
            children.append(c1)

        for j in range(len(children)):
            mask = np.random.random(n) < mutation_rate
            children[j][mask] ^= 1
            children[j] = repair(children[j])

        pop = (elite + children)[:population_size]
        fitness = batch_fitness(pop)
        cur_best = int(np.argmax(fitness))
        cur_val = float(fitness[cur_best])

        if cur_val > best_val + 1e-6:
            best_val = cur_val
            best_ind = pop[cur_best].copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 100:
            break

    selected = [int(i) for i in range(n) if best_ind[i] == 1]
    selected.sort()
    return selected


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

def _default_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "knapsack_ilp" / "test"

def _default_out_csv() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "GA" / "ga_eval_results.csv"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    dataset_dir:     Path,
    out_csv:         Path,
    n_limit:         Optional[int] = None,
    verbose:         bool = True,
    population_size: int = 100,
    max_generations: int = 500,
    mutation_rate:   float = 0.05,
    seed:            int = 42,
) -> None:
    files = list_instances(dataset_dir, limit=n_limit)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mark(f"Dataset: {dataset_dir} ({len(files)} instances)")
    mark(f"GA config: pop={population_size} gen={max_generations} "
         f"mut={mutation_rate} seed={seed}")

    fieldnames = [
        "instance_file", "n_items", "capacity",
        "total_weight", "total_value",
        "feasible", "inference_time_ms", "selected_items",
    ]

    results = []
    total_start = time.perf_counter()

    for idx, path in enumerate(files):
        W, V, C = load_instance(path)

        t0 = time.perf_counter()
        selected_idx = solve_knapsack_ga(
            W, V, int(C),
            population_size=population_size,
            max_generations=max_generations,
            mutation_rate=mutation_rate,
            seed=seed + idx,
        )
        t_ms = (time.perf_counter() - t0) * 1000.0

        if selected_idx:
            sel_arr = np.array(selected_idx, dtype=np.int32)
            total_weight = int(W[sel_arr].sum())
            total_value  = int(V[sel_arr].sum())
        else:
            total_weight = total_value = 0

        feasible = 1 if total_weight <= C else 0

        results.append({
            "instance_file":     path.name,
            "n_items":           int(W.shape[0]),
            "capacity":          int(C),
            "total_weight":      total_weight,
            "total_value":       total_value,
            "feasible":          feasible,
            "inference_time_ms": round(t_ms, 4),
            "selected_items":    json.dumps(selected_idx),
        })

        if verbose and ((idx + 1) % 10 == 0 or idx == 0):
            elapsed = time.perf_counter() - total_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta  = (len(files) - idx - 1) / rate if rate > 0 else 0
            mark(f"[{idx+1}/{len(files)}] elapsed={elapsed:.1f}s ETA={eta:.1f}s "
                 f"val={total_value} time={t_ms:.1f}ms")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total_time = time.perf_counter() - total_start
    if results:
        avg_val  = float(np.mean([r["total_value"]       for r in results]))
        avg_time = float(np.mean([r["inference_time_ms"] for r in results]))
        feas     = float(np.mean([r["feasible"]          for r in results]))
        mark(f"Done: {len(results)} in {total_time:.1f}s")
        mark(f"Avg value={avg_val:.2f} | time={avg_time:.1f}ms | feasible={feas:.3f}")
    mark(f"Results → {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="GA baseline evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir",    type=Path,  default=_default_dataset_dir())
    parser.add_argument("--out_csv",        type=Path,  default=_default_out_csv())
    parser.add_argument("--n",              type=int,   default=None)
    parser.add_argument("--population",     type=int,   default=100)
    parser.add_argument("--generations",    type=int,   default=500)
    parser.add_argument("--mutation_rate",  type=float, default=0.05)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--quiet",          action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        dataset_dir=args.dataset_dir, out_csv=args.out_csv,
        n_limit=args.n, verbose=not args.quiet,
        population_size=args.population,
        max_generations=args.generations,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
    )