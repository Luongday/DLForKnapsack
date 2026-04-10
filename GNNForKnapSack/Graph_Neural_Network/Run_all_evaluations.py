"""Run all solver evaluations and merge results."""

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
    parser.add_argument("--results_dir", type=Path, default=Path(__file__).resolve().parents[1] / "results" "compare",
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
                        choices=["dp", "gnn", "greedy", "ga", "bb", "dqn", "s2v", "reinforce"],
                        help="Skip these solvers")
    parser.add_argument("--only",  nargs="*", default=None,
                        choices=["dp", "gnn", "greedy", "ga", "bb", "dqn", "s2v", "reinforce"],
                        help="Run ONLY these solvers (overrides --skip)")

    # B&B
    parser.add_argument("--bb_timeout", type=float, default=60.0,
                        help="B&B timeout per instance (seconds)")
    parser.add_argument("--bb_max_nodes", type=int, default=2_000_000,
                        help="B&B max nodes per instance")

    # DQN
    parser.add_argument("--dqn_csv", type=Path, default=None,
                        help="Pre-computed DQN results CSV")
    parser.add_argument("--dqn_model", type=Path, default=None,
                        help="DQN model checkpoint (default: results/DQN/dqn.pt)")

    # S2V-DQN
    parser.add_argument("--s2v_csv", type=Path, default=None,
                        help="Pre-computed S2V-DQN results CSV")
    parser.add_argument("--s2v_model", type=Path, default=None,
                        help="S2V-DQN model checkpoint (default: results/S2V_DQN/s2v_dqn_best.pt)")

    # REINFORCE
    parser.add_argument("--reinforce_csv", type=Path, default=None,
                        help="Pre-computed REINFORCE results CSV")
    parser.add_argument("--reinforce_model", type=Path, default=None,
                        help="REINFORCE checkpoint (default: results/GNN_REINFORCE/gnn_reinforce.pt)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parents[1] / "tools"
    py   = sys.executable

    dataset_dir = args.dataset_dir.resolve()
    results_dir = args.results_dir.resolve()

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # Determine which solvers to run
    all_solvers = ["dp", "greedy", "ga", "bb", "gnn", "dqn", "s2v", "reinforce"]
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
            py, str(Path(__file__).resolve().parents[1] / "solvers" / "DP" / "dp_baseline_eval.py"),
            "--dataset_dir", str(dataset_dir),
            "--out_csv", str(csv_path),
        ] + n_flag)

    # 2. Greedy
    if "greedy" in solvers:
        csv_path = results_dir / "Greedy" / "greedy_eval_results.csv"
        csv_paths["greedy"] = csv_path
        results_ok["greedy"] = run_cmd("Greedy Baseline", [
            py, str(Path(__file__).resolve().parents[1] / "solvers" / "Greedy" / "greedy_baseline_eval.py"),
            "--dataset_dir", str(dataset_dir),
            "--out_csv", str(csv_path),
        ] + n_flag)

    # 3. GA
    if "ga" in solvers:
        csv_path = results_dir / "GA" / "ga_eval_results.csv"
        csv_paths["ga"] = csv_path
        results_ok["ga"] = run_cmd("Genetic Algorithm", [
            py, str(Path(__file__).resolve().parents[1] / "solvers" / "GA" / "ga_baseline_eval.py"),
            "--dataset_dir", str(dataset_dir),
            "--out_csv", str(csv_path),
            "--population", str(args.ga_population),
            "--generations", str(args.ga_generations),
            "--mutation_rate", str(args.ga_mutation_rate),
        ] + n_flag)

    # 3b. B&B (Branch-and-Bound, exact)
    if "bb" in solvers:
        csv_path = results_dir / "BB" / "bb_results.csv"
        csv_paths["bb"] = csv_path
        results_ok["bb"] = run_cmd("Branch-and-Bound", [
            py, str(Path(__file__).resolve().parents[1] / "solvers" / "B&B" / "bb_baseline_eval.py"),
            "--dataset_dir", str(dataset_dir),
            "--out_csv", str(csv_path),
            "--timeout_sec", str(args.bb_timeout),
            "--max_nodes", str(args.bb_max_nodes),
        ] + n_flag)

    # 4. GNN
    if "gnn" in solvers:
        model_path = args.model_path or (Path(__file__).resolve().parents[2] / "GNNForKnapSack" / "results" / "GNN" / "gnn_best.pt")
        if not model_path.exists():
            print(f"  [SKIP] GNN model not found: {model_path}")
            results_ok["gnn"] = False
        else:
            csv_path = results_dir / "GNN" / "gnn_eval_results.csv"
            csv_paths["gnn"] = csv_path
            results_ok["gnn"] = run_cmd("GNN Evaluation", [
                py, str(Path(__file__).resolve().parent / "Evaluate_GNN.py"),
                "--dataset_dir", str(dataset_dir),
                "--model_path", str(model_path),
                "--out_csv", str(csv_path),
            ] + n_flag)

    # 5. DQN
    if "dqn" in solvers:
        dqn_model = args.dqn_model or (Path(__file__).resolve().parents[2] / "GNNForKnapSack" / "results" / "DQN" / "dqn_best.pt")
        if args.dqn_csv and args.dqn_csv.exists():
            # Use pre-computed CSV
            csv_paths["dqn"] = args.dqn_csv
            results_ok["dqn"] = True
        elif dqn_model.exists():
            csv_path = results_dir / "DQN" / "dqn_eval_results.csv"
            csv_paths["dqn"] = csv_path
            results_ok["dqn"] = run_cmd("DQN Evaluation", [
                py, str(Path(__file__).resolve().parents[1] / "Reinforcement_Learning" / "DQN" / "Evaluate_DQN.py"),
                "--dataset_dir", str(dataset_dir),
                "--model_path", str(dqn_model),
                "--out_csv", str(csv_path),
            ] + n_flag)
        else:
            print(f"  [SKIP] DQN model not found: {dqn_model}")
            results_ok["dqn"] = False

    # 6. S2V-DQN
    if "s2v" in solvers:
        s2v_model = args.s2v_model or (Path(__file__).resolve().parents[2] / "GNNForKnapSack" / "results" / "S2V_DQN" / "s2v_dqn_best.pt")
        if args.s2v_csv and args.s2v_csv.exists():
            csv_paths["s2v"] = args.s2v_csv
            results_ok["s2v"] = True
        elif s2v_model.exists():
            csv_path = results_dir / "S2V_DQN" / "s2v_dqn_eval_results.csv"
            csv_paths["s2v"] = csv_path
            results_ok["s2v"] = run_cmd("S2V-DQN Evaluation", [
                py, str(Path(__file__).resolve().parents[1] / "Reinforcement_Learning" / "S2V_DQN" / "Evaluate_S2V_DQN.py"),
                "--dataset_dir", str(dataset_dir),
                "--model_path", str(s2v_model),
                "--out_csv", str(csv_path),
            ] + n_flag)
        else:
            print(f"  [SKIP] S2V-DQN model not found: {s2v_model}")
            results_ok["s2v"] = False

    # 7. REINFORCE
    if "reinforce" in solvers:
        rf_model = args.reinforce_model or (Path(__file__).resolve().parents[2] / "GNNForKnapSack" / "results" / "GNN_REINFORCE" / "gnn_reinforce_best.pt")
        if args.reinforce_csv and args.reinforce_csv.exists():
            csv_paths["reinforce"] = args.reinforce_csv
            results_ok["reinforce"] = True
        elif rf_model.exists():
            csv_path = results_dir / "GNN_REINFORCE" / "reinforce_eval_results.csv"
            csv_paths["reinforce"] = csv_path
            results_ok["reinforce"] = run_cmd("REINFORCE Evaluation", [
                py, str(Path(__file__).resolve().parents[1] / "Reinforcement_Learning" / "REINFORCE" / "Evaluate_REINFORCE.py"),
                "--dataset_dir", str(dataset_dir),
                "--model_path", str(rf_model),
                "--out_csv", str(csv_path),
            ] + n_flag)
        else:
            print(f"  [SKIP] REINFORCE model not found: {rf_model}")
            results_ok["reinforce"] = False

    # 6. Merge
    merge_args = [
        py, str(here / "Merge_results.py"),
        "--out_dir", str(results_dir / "compare"),
        "--skip_missing",
    ]
    for solver_name, csv_flag in [
        ("dp", "--dp_csv"), ("gnn", "--gnn_csv"),
        ("greedy", "--greedy_csv"), ("ga", "--ga_csv"),
        ("bb", "--bb_csv"),
        ("dqn", "--dqn_csv"),
        ("s2v", "--s2v_csv"),
        ("reinforce", "--reinforce_csv"),
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