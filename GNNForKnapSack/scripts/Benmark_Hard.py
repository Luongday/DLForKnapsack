"""Benchmark classical solvers (DP, Greedy, GA) on Pisinger hard instances.

Generates hard instances via Genhard, runs solvers, prints comparison
showing where Greedy breaks down.

Note: This script focuses on CLASSICAL solvers. For neural solvers
(GNN, DQN, S2V-DQN, REINFORCE), use cross_scale_eval.py after
generating the Pisinger datasets here.

Usage:
    # Quick test
    python Benchmark_Hard.py --n_items 50 --num_instances 50

    # Full benchmark (all types)
    python Benchmark_Hard.py --n_items 100 --num_instances 200

    # Skip slow solvers
    python Benchmark_Hard.py --skip_ga --skip_dp
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

from Generate_Hard import (
    generate_hard_dataset, compile_genhard,
    TYPE_NAMES,
)
from GNNForKnapSack.instance_loader import load_instance, list_instances
from GNNForKnapSack.solvers.Greedy.greedy_baseline_eval import solve_knapsack_greedy
from GNNForKnapSack.solvers.GA.ga_baseline_eval import solve_knapsack_ga
from GNNForKnapSack.Graph_Neural_Network.Knapsack_GNN.Dp import solve_knapsack_dp


def mark(msg: str) -> None:
    print(f"\n{'='*70}\n  {msg}\n{'='*70}", flush=True)


# ---------------------------------------------------------------------------
# Solver wrappers — uniform signature: (W, V, C) → (solution_array, value)
# ---------------------------------------------------------------------------

def greedy_solver(W, V, C):
    """Greedy by value/weight ratio."""
    selected = solve_knapsack_greedy(W, V, int(C))
    sol = np.zeros(len(W), dtype=np.int8)
    for i in selected:
        sol[i] = 1
    value = float((V.astype(float) * sol.astype(float)).sum())
    return sol, value


def ga_solver(W, V, C, population=100, generations=500, seed=42):
    """Genetic algorithm solver."""
    selected = solve_knapsack_ga(
        W, V, int(C),
        population_size=population,
        max_generations=generations,
        seed=seed,
    )
    sol = np.zeros(len(W), dtype=np.int8)
    for i in selected:
        sol[i] = 1
    value = float((V.astype(float) * sol.astype(float)).sum())
    return sol, value


def dp_solver(W, V, C):
    """Exact DP solver (slow for large n × capacity).

    Handles two possible return formats from solve_knapsack_dp:
      - 0/1 list of length n      (Dp.py: solve_knapsack_dp_np)
      - list of selected indices  (dp_baseline_eval.solve_knapsack_dp)
    """
    n = len(W)
    raw = solve_knapsack_dp(W.tolist(), V.tolist(), int(C))

    if raw is None:
        # Capacity too large → DP skipped
        sol = np.zeros(n, dtype=np.int8)
        return sol, 0.0

    sol = np.zeros(n, dtype=np.int8)

    # Detect format: 0/1 list of length n, else list of indices
    if len(raw) == n and all(x in (0, 1) for x in raw):
        sol = np.asarray(raw, dtype=np.int8)
    else:
        # list of selected indices
        for i in raw:
            idx = int(i)
            if 0 <= idx < n:
                sol[idx] = 1

    value = float((V.astype(float) * sol.astype(float)).sum())
    return sol, value


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_solver_on_dir(
    dataset_dir: Path,
    solver_fn,
    solver_name: str,
    verbose: bool = True,
) -> dict:
    """Run a solver on all instances in a directory."""
    files = list_instances(dataset_dir)
    ratios, times, values = [], [], []
    feasible_count = 0

    for path in files:
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
        feasible = weight <= float(C) + 1e-6

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
        "solver":        solver_name,
        "n_instances":   n,
        "feasible_rate": feasible_count / max(n, 1),
        "avg_ratio":     float(np.mean(ratios))      if ratios else None,
        "std_ratio":     float(np.std(ratios))       if ratios else None,
        "min_ratio":     float(np.min(ratios))       if ratios else None,
        "p10_ratio":     float(np.percentile(ratios, 10)) if ratios else None,
        "avg_time_ms":   float(np.mean(times))       if times  else 0.0,
        "avg_value":     float(np.mean(values))      if values else 0.0,
    }

    if verbose:
        r = result
        ratio_str = f"{r['avg_ratio']:.4f} ± {r['std_ratio']:.4f}" if r['avg_ratio'] else "N/A"
        print(f"  {solver_name:>8}: ratio={ratio_str}  "
              f"time={r['avg_time_ms']:.2f}ms  "
              f"feasible={r['feasible_rate']:.3f}")

    return result


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark classical solvers on Pisinger hard instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--types", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5, 6],
                        help="Pisinger instance types to benchmark "
                             "(1=uncorrelated, 2=weakly, 3=strongly, "
                             "4=inverse strongly, 5=almost strongly, 6=subset sum)")
    parser.add_argument("--n_items", type=int, default=100)
    parser.add_argument("--num_instances", type=int, default=200)
    parser.add_argument("--range", type=int, default=1000,
                        help="Coefficient range")
    parser.add_argument("--out_dir", type=Path,
                        default=_HERE / "data" / "pisinger")
    parser.add_argument("--results_dir", type=Path,
                        default=_HERE / "results" / "pisinger")
    parser.add_argument("--ga_population",  type=int, default=100)
    parser.add_argument("--ga_generations", type=int, default=500)
    parser.add_argument("--skip_dp", action="store_true",
                        help="Skip DP evaluation (DP values still loaded from NPZ)")
    parser.add_argument("--skip_ga", action="store_true",
                        help="Skip GA (slow for large instances)")
    parser.add_argument("--max_dp_capacity", type=int, default=500_000)
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
        existing = list(dataset_dir.glob("instance_*.npz")) if dataset_dir.exists() else []
        if len(existing) < args.num_instances:
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
            print(f"  Using existing dataset: {dataset_dir} ({len(existing)} instances)")

        # Evaluate solvers
        print(f"\n  Evaluating solvers on type {t} ({type_name})...")

        results_type = {}

        # Greedy (always fast)
        results_type["greedy"] = evaluate_solver_on_dir(
            dataset_dir, greedy_solver, "Greedy"
        )

        # DP (exact, can be slow for large capacity)
        if not args.skip_dp:
            results_type["dp"] = evaluate_solver_on_dir(
                dataset_dir, dp_solver, "DP"
            )

        # GA (can be slow)
        if not args.skip_ga:
            def ga_fn(W, V, C):
                return ga_solver(W, V, C, args.ga_population, args.ga_generations)
            results_type["ga"] = evaluate_solver_on_dir(
                dataset_dir, ga_fn, "GA"
            )

        all_results[t] = results_type

    # Print comparison table
    mark("FINAL COMPARISON — ALL TYPES × ALL SOLVERS")

    header = f"{'Type':>5} {'Name':>25} | {'Greedy Ratio':>14} {'Greedy Time':>12}"
    if not args.skip_dp:
        header += f" | {'DP Ratio':>10} {'DP Time':>10}"
    if not args.skip_ga:
        header += f" | {'GA Ratio':>10} {'GA Time':>10}"
    print()
    print(header)
    print("-" * len(header))

    for t in args.types:
        name = TYPE_NAMES.get(t, f"type_{t}")
        r = all_results[t]
        gr = r["greedy"]
        gr_ratio = f"{gr['avg_ratio']:.4f}" if gr['avg_ratio'] else "N/A"
        line = f"{t:>5} {name:>25} | {gr_ratio:>14} {gr['avg_time_ms']:>10.3f}ms"

        if not args.skip_dp and "dp" in r:
            dp = r["dp"]
            dp_ratio = f"{dp['avg_ratio']:.4f}" if dp['avg_ratio'] else "N/A"
            line += f" | {dp_ratio:>10} {dp['avg_time_ms']:>8.1f}ms"

        if not args.skip_ga and "ga" in r:
            ga = r["ga"]
            ga_ratio = f"{ga['avg_ratio']:.4f}" if ga['avg_ratio'] else "N/A"
            line += f" | {ga_ratio:>10} {ga['avg_time_ms']:>8.1f}ms"

        print(line)

    # Save results JSON
    args.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.results_dir / "pisinger_benchmark.json"

    # Convert for JSON serialization (drop None values)
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
            "n_items":       args.n_items,
            "num_instances": args.num_instances,
            "coeff_range":   args.range,
            "types":         args.types,
            "results":       json_results,
        }, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # Highlight key finding
    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    for t in [3, 4, 5, 6]:
        if t in all_results and all_results[t]["greedy"]["avg_ratio"]:
            gr_r = all_results[t]["greedy"]["avg_ratio"]
            name = TYPE_NAMES.get(t, "")
            if gr_r < 0.95:
                print(f"  Type {t} ({name}): Greedy ratio = {gr_r:.4f} ← GREEDY BREAKS DOWN")
            elif gr_r < 0.99:
                print(f"  Type {t} ({name}): Greedy ratio = {gr_r:.4f} ← Notable gap")
            else:
                print(f"  Type {t} ({name}): Greedy ratio = {gr_r:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()