"""Benchmark all solvers on Pisinger hard instances.

Generates hard instances → runs DP, Greedy, GA (and GNN/DQN if available)
→ merges results → prints comparison showing where Greedy breaks down.

This is THE experiment that demonstrates neural solver value.

Usage:
    # Quick test (small)
    python Benchmark_Hard.py --n_items 50 --num_instances 50

    # Full benchmark
    python Benchmark_Hard.py --n_items 100 --num_instances 200

    # Large scale (where DP gets slow)
    python Benchmark_Hard.py --n_items 200 --num_instances 100

    # With GNN model
    python Benchmark_Hard.py --n_items 100 --num_instances 200 --gnn_model results/GNN/gnn.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from GNNForKnapSack.scripts.Generate_hard import (
    generate_hard_dataset, compile_genhard,
    TYPE_NAMES, DEFAULT_GENHARD_BIN,
)
from GNNForKnapSack import load_instance, list_instances
from GNNForKnapSack.solvers.Greedy.Greedy import greedy_knapsack
from GNNForKnapSack.solvers.GA.GA import KnapsackGA


def mark(msg: str) -> None:
    print(f"\n{'='*70}\n  {msg}\n{'='*70}", flush=True)


def evaluate_solver_on_dir(
    dataset_dir: Path,
    solver_fn,
    solver_name: str,
    verbose: bool = True,
) -> dict:
    """Run a solver on all instances in a directory.

    Args:
        solver_fn: callable(weights, values, capacity) → (solution_array, value)
    Returns:
        Dict with avg_ratio, avg_time, feasibility_rate, etc.
    """
    files = list_instances(dataset_dir)
    ratios, times, values = [], [], []
    feasible_count = 0

    for idx, path in enumerate(files):
        W, V, C = load_instance(path)

        # Load DP optimal for ratio calculation
        arr = np.load(path, allow_pickle=True)
        dp_val_arr = None
        for k in ["dp_value", "optimal_value"]:
            if k in arr.files:
                dp_val_arr = arr[k]
                break
        dp_val = int(dp_val_arr) if dp_val_arr is not None and int(dp_val_arr) > 0 else None

        t0 = time.perf_counter()
        solution, value = solver_fn(W, V, C)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        weight = float((W.astype(float) * solution.astype(float)).sum())
        feasible = weight <= C + 1e-6

        if dp_val and dp_val > 0:
            ratio = value / dp_val
        else:
            ratio = None

        feasible_count += int(feasible)
        times.append(solve_ms)
        values.append(value)
        if ratio is not None:
            ratios.append(ratio)

    n = len(files)
    result = {
        "solver":          solver_name,
        "n_instances":     n,
        "feasible_rate":   feasible_count / max(n, 1),
        "avg_ratio":       float(np.mean(ratios)) if ratios else None,
        "std_ratio":       float(np.std(ratios)) if ratios else None,
        "min_ratio":       float(np.min(ratios)) if ratios else None,
        "p10_ratio":       float(np.percentile(ratios, 10)) if ratios else None,
        "avg_time_ms":     float(np.mean(times)),
        "avg_value":       float(np.mean(values)),
    }

    if verbose:
        r = result
        ratio_str = f"{r['avg_ratio']:.4f} ± {r['std_ratio']:.4f}" if r['avg_ratio'] else "N/A"
        print(f"  {solver_name:>8}: ratio={ratio_str}  "
              f"time={r['avg_time_ms']:.2f}ms  "
              f"feasible={r['feasible_rate']:.3f}")

    return result


# ---------------------------------------------------------------------------
# Solver wrappers
# ---------------------------------------------------------------------------

def greedy_solver(W, V, C):
    sol = greedy_knapsack(V.astype(float), W.astype(float), float(C))
    val = float((V.astype(float) * sol.astype(float)).sum())
    return sol, val


def ga_solver(W, V, C, population=100, generations=500, seed=42):
    ga = KnapsackGA(
        W.astype(float), V.astype(float), float(C),
        population_size=population,
        max_generations=generations,
        seed=seed,
    )
    sol, val, _ = ga.solve()
    return sol, val


def dp_solver(W, V, C):
    """DP solver wrapper — returns (solution, value)."""
    from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.Dp import solve_knapsack_dp
    sol_list = solve_knapsack_dp(W.tolist(), V.tolist(), int(C))
    sol = np.array(sol_list, dtype=np.int8)
    val = int((V * sol).sum())
    return sol, val


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark solvers on Pisinger hard instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--types", type=int, nargs="+",
                        default=[1, 2, 3, 5, 6],
                        help="Pisinger instance types to benchmark")
    parser.add_argument("--n_items", type=int, default=100)
    parser.add_argument("--num_instances", type=int, default=200)
    parser.add_argument("--range", type=int, default=1000,
                        help="Coefficient range")
    parser.add_argument("--out_dir", type=Path,
                        default=_HERE / "data" / "pisinger")
    parser.add_argument("--results_dir", type=Path,
                        default=_HERE / "results" / "pisinger")
    parser.add_argument("--ga_population", type=int, default=100)
    parser.add_argument("--ga_generations", type=int, default=500)
    parser.add_argument("--skip_ga", action="store_true",
                        help="Skip GA (slow for large instances)")
    parser.add_argument("--skip_dp_solve", action="store_true",
                        help="Skip DP during generation (use existing)")
    parser.add_argument("--max_dp_capacity", type=int, default=500_000)
    parser.add_argument("--gnn_model", type=Path, default=None,
                        help="GNN model checkpoint for evaluation")
    return parser.parse_args()


def main():
    args = parse_args()

    # Compile Genhard
    genhard_bin = compile_genhard()

    # Results accumulator
    all_results = {}

    for t in args.types:
        type_name = TYPE_NAMES.get(t, f"type_{t}")
        mark(f"Type {t}: {type_name} (n={args.n_items}, {args.num_instances} instances)")

        # Generate
        dataset_dir = args.out_dir / f"type_{t:02d}_{type_name}"
        if not dataset_dir.exists() or len(list(dataset_dir.glob("instance_*.npz"))) < args.num_instances:
            generate_hard_dataset(
                out_dir=dataset_dir,
                instance_type=t,
                n_items=args.n_items,
                num_instances=args.num_instances,
                coeff_range=args.range,
                max_dp_capacity=args.max_dp_capacity,
                genhard_bin=genhard_bin,
            )
        else:
            print(f"  Using existing dataset: {dataset_dir}")

        # Evaluate solvers
        print(f"\n  Evaluating solvers on type {t} ({type_name})...")

        results_type = {}

        # Greedy (always fast)
        results_type["greedy"] = evaluate_solver_on_dir(
            dataset_dir, greedy_solver, "Greedy"
        )

        # GA (can be slow)
        if not args.skip_ga:
            def ga_fn(W, V, C):
                return ga_solver(W, V, C, args.ga_population, args.ga_generations)
            results_type["ga"] = evaluate_solver_on_dir(
                dataset_dir, ga_fn, "GA"
            )

        all_results[t] = results_type

    # Print comparison
    mark("FINAL COMPARISON — ALL TYPES × ALL SOLVERS")
    print(f"\n{'Type':>5} {'Name':>25} | {'Greedy Ratio':>14} {'Greedy Time':>12}", end="")
    if not args.skip_ga:
        print(f" | {'GA Ratio':>10} {'GA Time':>10}", end="")
    print()
    print("-" * 100)

    for t in args.types:
        name = TYPE_NAMES.get(t, f"type_{t}")
        r = all_results[t]
        gr = r["greedy"]
        gr_ratio = f"{gr['avg_ratio']:.4f}" if gr['avg_ratio'] else "N/A"
        line = f"{t:>5} {name:>25} | {gr_ratio:>14} {gr['avg_time_ms']:>10.3f}ms"

        if not args.skip_ga and "ga" in r:
            ga = r["ga"]
            ga_ratio = f"{ga['avg_ratio']:.4f}" if ga['avg_ratio'] else "N/A"
            line += f" | {ga_ratio:>10} {ga['avg_time_ms']:>8.1f}ms"

        print(line)

    # Save results JSON
    args.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.results_dir / "pisinger_benchmark.json"

    # Convert for JSON serialization
    json_results = {}
    for t, solvers in all_results.items():
        json_results[str(t)] = {}
        for s_name, s_data in solvers.items():
            json_results[str(t)][s_name] = {
                k: v for k, v in s_data.items()
                if v is not None
            }

    with results_path.open("w") as f:
        json.dump({
            "n_items": args.n_items,
            "num_instances": args.num_instances,
            "coeff_range": args.range,
            "types": args.types,
            "results": json_results,
        }, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # Highlight key finding
    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    for t in [3, 5, 6]:
        if t in all_results and all_results[t]["greedy"]["avg_ratio"]:
            gr_r = all_results[t]["greedy"]["avg_ratio"]
            name = TYPE_NAMES.get(t, "")
            if gr_r < 0.95:
                print(f"  Type {t} ({name}): Greedy ratio = {gr_r:.4f} ← GREEDY BREAKS DOWN")
            elif gr_r < 0.99:
                print(f"  Type {t} ({name}): Greedy ratio = {gr_r:.4f} ← Notable gap")
    print("=" * 70)


if __name__ == "__main__":
    main()