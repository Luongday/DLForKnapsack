"""Run all solver evaluations and merge results.

One-command pipeline:
    1. DP baseline evaluation
    2. GNN evaluation
    3. Greedy evaluation
    4. GA evaluation
    5. (DQN — when ready)
    6. Merge all results into comparison table

Usage:
    python run_all_evaluations.py --dataset_dir data/knapsack_ilp/test
    python run_all_evaluations.py --dataset_dir data/knapsack_ilp/test --skip gnn
    python run_all_evaluations.py --dataset_dir data/knapsack_ilp/test --only greedy ga
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def mark(msg: str) -> None:
    print(f"\n{'='*70}\n  {msg}\n{'='*70}", flush=True)


def run_cmd(label: str, cmd: list, cwd: Path = None) -> bool:
    """Run a command, return True if successful."""
    mark(f"Running: {label}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    elapsed = time.perf_counter() - t0
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (code={result.returncode})"
    print(f"  {status} in {elapsed:.1f}s")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all solver evaluations and merge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=Path, required=True,
                        help="Directory with instance_*.npz files")
    parser.add_argument("--results_dir", type=Path, default=Path("results"),
                        help="Root results directory")
    parser.add_argument("--model_path",  type=Path, default=None,
                        help="GNN model checkpoint (default: results/GNN/gnn.pt)")
    parser.add_argument("--n",           type=int, default=None,
                        help="Limit to first N instances")

    # GA hyperparameters
    parser.add_argument("--ga_population",   type=int,   default=100)
    parser.add_argument("--ga_generations",  type=int,   default=500)
    parser.add_argument("--ga_mutation_rate", type=float, default=0.05)

    # Solver selection
    parser.add_argument("--skip",  nargs="*", default=[],
                        choices=["dp", "gnn", "greedy", "ga", "dqn"],
                        help="Skip these solvers")
    parser.add_argument("--only",  nargs="*", default=None,
                        choices=["dp", "gnn", "greedy", "ga", "dqn"],
                        help="Run ONLY these solvers (overrides --skip)")

    # DQN (future)
    parser.add_argument("--dqn_csv", type=Path, default=None,
                        help="Pre-computed DQN results CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    py   = sys.executable

    dataset_dir = args.dataset_dir.resolve()
    results_dir = args.results_dir.resolve()

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # Determine which solvers to run
    all_solvers = ["dp", "greedy", "ga", "gnn"]
    if args.only:
        solvers = [s for s in all_solvers if s in args.only]
    else:
        solvers = [s for s in all_solvers if s not in args.skip]

    print(f"Dataset:  {dataset_dir}")
    print(f"Results:  {results_dir}")
    print(f"Solvers:  {', '.join(s.upper() for s in solvers)}")

    n_flag = ["--n", str(args.n)] if args.n else []

    csv_paths = {}
    results_ok = {}

    # 1. DP
    if "dp" in solvers:
        csv_path = results_dir / "DP" / "dp_results.csv"
        csv_paths["dp"] = csv_path
        results_ok["dp"] = run_cmd("DP Baseline", [
            # py, str(here / "dp_baseline_eval.py"),
            py, "-m", "GNNForKnapSack.solvers.DP.dp_baseline_eval",
            "--dataset_dir", str(dataset_dir),
            "--out_csv", str(csv_path),
        ] + n_flag)

    # 2. Greedy
    if "greedy" in solvers:
        csv_path = results_dir / "Greedy" / "greedy_eval_results.csv"
        csv_paths["greedy"] = csv_path
        results_ok["greedy"] = run_cmd("Greedy Baseline", [
            # py, str(here / "Evaluate_Greedy.py"),
            py, "-m", "GNNForKnapSack.Graph_Neural_Network.Evaluate_Greedy",
            "--dataset_dir", str(dataset_dir),
            "--out_csv", str(csv_path),
        ] + n_flag)

    # 3. GA
    if "ga" in solvers:
        csv_path = results_dir / "GA" / "ga_eval_results.csv"
        csv_paths["ga"] = csv_path
        results_ok["ga"] = run_cmd("Genetic Algorithm", [
            # py, str(here / "Evaluate_GA.py"),
            py, "-m", "GNNForKnapSack.Graph_Neural_Network.Evaluate_GA",
            "--dataset_dir", str(dataset_dir),
            "--out_csv", str(csv_path),
            "--population", str(args.ga_population),
            "--generations", str(args.ga_generations),
            "--mutation_rate", str(args.ga_mutation_rate),
        ] + n_flag)

    # 4. GNN
    if "gnn" in solvers:
        results_dir = Path(__file__).resolve().parents[1] / "results"
        model_path = results_dir / "GNN" / "gnn.pt"
        if not model_path.exists():
            print(f"  [SKIP] GNN model not found: {model_path}")
            results_ok["gnn"] = False
        else:
            csv_path = results_dir / "GNN" / "gnn_eval_results.csv"
            csv_paths["gnn"] = csv_path
            results_ok["gnn"] = run_cmd("GNN Evaluation", [
                # py, str(here / "Evaluate_GNN.py"),
                py, "-m", "GNNForKnapSack.Graph_Neural_Network.Evaluate_GNN",
                "--dataset_dir", str(dataset_dir),
                "--model_path", str(model_path),
                "--out_csv", str(csv_path),
            ] + n_flag)

    # 5. DQN (future)
    if args.dqn_csv and args.dqn_csv.exists():
        csv_paths["dqn"] = args.dqn_csv
        results_ok["dqn"] = True

    # 6. Merge
    merge_args = [
        # py, str(here / "Merge_results.py"),
        py, "-m", "GNNForKnapSack.tools.Merge_results",
        "--out_dir", str(results_dir / "compare"),
        "--skip_missing",
    ]
    for solver_name, csv_flag in [
        ("dp", "--dp_csv"), ("gnn", "--gnn_csv"),
        ("greedy", "--greedy_csv"), ("ga", "--ga_csv"),
        ("dqn", "--dqn_csv"),
    ]:
        if solver_name in csv_paths and results_ok.get(solver_name, False):
            merge_args.extend([csv_flag, str(csv_paths[solver_name])])

    run_cmd("Merge All Results", merge_args)

    # Summary
    mark("PIPELINE COMPLETE")
    for name, ok in results_ok.items():
        status = "OK" if ok else "FAILED/SKIPPED"
        print(f"  {name.upper():>8}: {status}")
    print(f"\n  Merged results → {results_dir / 'compare'}")


if __name__ == "__main__":
    main()