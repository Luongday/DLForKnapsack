"""Cross-scale / cross-distribution evaluation for neural Knapsack solvers."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def mark(msg: str) -> None:
    print(f"[CROSS-SCALE] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Solver runners — wrap existing eval scripts
# ---------------------------------------------------------------------------

def run_eval_script(
    script: str,
    dataset_dir: Path,
    out_csv: Path,
    extra_args: Optional[List[str]] = None,
) -> bool:
    """Execute one solver's eval script and check output."""
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / script),
        "--dataset_dir", str(dataset_dir),
        "--out_csv",     str(out_csv),
    ]
    if extra_args:
        cmd.extend(extra_args)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mark(f"  Running {script} on {dataset_dir.name}...")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        mark(f"  [FAIL] {script}: {result.stderr[-300:]}")
        return False

    mark(f"  [OK] {script} in {elapsed:.1f}s")
    return True


# ---------------------------------------------------------------------------
# CSV loading + metric computation
# ---------------------------------------------------------------------------

def load_eval_csv(csv_path: Path) -> List[Dict]:
    """Load CSV rows into list of dicts."""
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_metrics_vs_dp(
    solver_rows: List[Dict],
    dp_rows:     List[Dict],
) -> Dict:
    """Compute avg ratio vs DP, avg time, feasibility rate."""
    dp_index = {r["instance_file"]: float(r["total_value"]) for r in dp_rows}

    ratios = []
    times = []
    feasibles = 0
    matched = 0

    for r in solver_rows:
        name = r.get("instance_file", "")
        dp_val = dp_index.get(name)
        if dp_val is None or dp_val <= 0:
            continue

        try:
            sv = float(r.get("total_value", 0))
            tm = float(r.get("inference_time_ms", 0))
            fs = int(r.get("feasible", 0))
        except (ValueError, TypeError):
            continue

        ratios.append(sv / dp_val)
        times.append(tm)
        feasibles += fs
        matched += 1

    if matched == 0:
        return {"avg_ratio": None, "avg_time_ms": None,
                "feasibility": None, "n": 0}

    return {
        "avg_ratio":   sum(ratios) / matched,
        "avg_time_ms": sum(times) / matched,
        "feasibility": feasibles / matched,
        "n":           matched,
        "min_ratio":   min(ratios),
        "max_ratio":   max(ratios),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_extra_models(specs: List[str]) -> List[Tuple[str, Path]]:
    """Parse 'type:path' specs into list of tuples."""
    result = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Bad spec '{spec}'. Expected 'type:path'")
        model_type, path = spec.split(":", 1)
        result.append((model_type.lower(), Path(path)))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-scale and cross-distribution evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_type", type=str, default=None,
                        choices=["gnn", "dqn", "s2v", "reinforce"],
                        help="Primary neural model type")
    parser.add_argument("--model_path", type=Path, default=None,
                        help="Primary neural model checkpoint")
    parser.add_argument("--extra_models", nargs="*", default=[],
                        help="Extra models as type:path (e.g., 'dqn:results/DQN/dqn.pt')")

    parser.add_argument("--test_sets", nargs="+", required=True, type=Path,
                        help="List of test set directories")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each test set (same length as test_sets)")

    parser.add_argument("--include_baselines", action="store_true",
                        help="Also run DP/Greedy/GA/BB baselines on every test set")
    parser.add_argument("--skip_dp", action="store_true",
                        help="Skip DP (use when DP too slow for large n)")
    parser.add_argument("--skip_bb", action="store_true",
                        help="Skip Branch-and-Bound (slow on strongly correlated instances)")
    parser.add_argument("--bb_timeout", type=float, default=60.0,
                        help="B&B timeout per instance (seconds)")

    parser.add_argument("--out_dir", type=Path, default=Path("results/cross_scale"),
                        help="Output directory for per-set CSVs and summary")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N instances per test set (for quick runs)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate
    test_sets = [Path(p) for p in args.test_sets]
    for d in test_sets:
        if not d.exists():
            raise SystemExit(f"Test set not found: {d}")

    if args.labels:
        if len(args.labels) != len(test_sets):
            raise SystemExit("--labels must match --test_sets length")
        labels = args.labels
    else:
        labels = [d.name for d in test_sets]

    # Build model list (primary + extras)
    models: List[Tuple[str, Path]] = []
    if args.model_type and args.model_path:
        models.append((args.model_type, args.model_path))
    for spec in parse_extra_models(args.extra_models):
        models.append(spec)

    if not models and not args.include_baselines:
        raise SystemExit("Nothing to evaluate. Specify --model_type/path or --include_baselines")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mark(f"Output dir: {args.out_dir}")
    mark(f"Test sets ({len(test_sets)}): {', '.join(labels)}")
    mark(f"Models ({len(models)}): {[(t, p.name) for t, p in models]}")

    # Eval script mapping
    EVAL_SCRIPTS = {
        "dp":        ("dp_baseline_eval.py",      []),
        "greedy":    ("greedy_baseline_eval.py",  []),
        "ga":        ("ga_baseline_eval.py",      []),
        "bb":        ("bb_baseline_eval.py",      []),
        "gnn":       ("Evaluate_GNN.py",          ["--model_path"]),
        "dqn":       ("Evaluate_DQN.py",          ["--model_path"]),
        "s2v":       ("Evaluate_S2V_DQN.py",      ["--model_path"]),
        "reinforce": ("Evaluate_REINFORCE.py",    ["--model_path"]),
    }

    n_flag = ["--n", str(args.n)] if args.n else []

    # Results: test_set_label → solver → metrics
    all_results: Dict[str, Dict[str, Dict]] = {}

    for label, test_dir in zip(labels, test_sets):
        mark(f"\n=== Test set: {label} ({test_dir}) ===")
        set_out = args.out_dir / label
        set_out.mkdir(parents=True, exist_ok=True)

        all_results[label] = {}
        dp_rows: List[Dict] = []

        # Baselines
        if args.include_baselines:
            # DP (ground truth)
            if not args.skip_dp:
                dp_csv = set_out / "dp.csv"
                ok = run_eval_script("dp_baseline_eval.py", test_dir, dp_csv, n_flag)
                if ok:
                    dp_rows = load_eval_csv(dp_csv)

            # Greedy
            gr_csv = set_out / "greedy.csv"
            if run_eval_script("greedy_baseline_eval.py", test_dir, gr_csv, n_flag):
                gr_rows = load_eval_csv(gr_csv)
                all_results[label]["greedy"] = compute_metrics_vs_dp(gr_rows, dp_rows)

            # GA
            ga_csv = set_out / "ga.csv"
            if run_eval_script("ga_baseline_eval.py", test_dir, ga_csv, n_flag):
                ga_rows = load_eval_csv(ga_csv)
                all_results[label]["ga"] = compute_metrics_vs_dp(ga_rows, dp_rows)

            # B&B (exact, can be slow on hard instances)
            if not args.skip_bb:
                bb_csv = set_out / "bb.csv"
                bb_flags = n_flag + ["--timeout_sec", str(args.bb_timeout)]
                if run_eval_script("bb_baseline_eval.py", test_dir, bb_csv, bb_flags):
                    bb_rows = load_eval_csv(bb_csv)
                    all_results[label]["bb"] = compute_metrics_vs_dp(bb_rows, dp_rows)

            # DP self-ratio = 1.0
            if dp_rows:
                all_results[label]["dp"] = {
                    "avg_ratio": 1.0, "avg_time_ms": None,
                    "feasibility": 1.0, "n": len(dp_rows),
                    "min_ratio": 1.0, "max_ratio": 1.0,
                }
                # Compute avg DP time
                times = [float(r["inference_time_ms"]) for r in dp_rows]
                all_results[label]["dp"]["avg_time_ms"] = sum(times) / len(times)

        # Neural models
        for model_type, model_path in models:
            if model_type not in EVAL_SCRIPTS:
                mark(f"  Unknown model type: {model_type}, skip")
                continue
            script, model_flag = EVAL_SCRIPTS[model_type]
            model_csv = set_out / f"{model_type}.csv"
            extra = model_flag + [str(model_path)] + n_flag
            if run_eval_script(script, test_dir, model_csv, extra):
                rows = load_eval_csv(model_csv)
                all_results[label][model_type] = compute_metrics_vs_dp(rows, dp_rows)

    # ---- Write summary ----
    summary_csv = args.out_dir / "cross_scale_results.csv"
    summary_json = args.out_dir / "cross_scale_summary.json"

    # Collect all solver names
    all_solvers = set()
    for lbl_results in all_results.values():
        all_solvers.update(lbl_results.keys())
    solver_order = [s for s in ["dp", "greedy", "ga", "bb", "gnn", "dqn", "s2v", "reinforce"] if s in all_solvers]

    # CSV: one row per (test_set, solver)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "test_set", "solver", "n_instances",
            "avg_ratio", "min_ratio", "max_ratio",
            "avg_time_ms", "feasibility",
        ])
        for label in labels:
            for solver in solver_order:
                m = all_results.get(label, {}).get(solver)
                if m is None:
                    continue
                writer.writerow([
                    label, solver.upper(), m["n"],
                    round(m["avg_ratio"], 4) if m["avg_ratio"] is not None else "",
                    round(m.get("min_ratio", 0), 4) if m.get("min_ratio") is not None else "",
                    round(m.get("max_ratio", 0), 4) if m.get("max_ratio") is not None else "",
                    round(m["avg_time_ms"], 4) if m["avg_time_ms"] is not None else "",
                    round(m["feasibility"], 4) if m["feasibility"] is not None else "",
                ])

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump({
            "labels": labels,
            "test_sets": [str(t) for t in test_sets],
            "models": [{"type": t, "path": str(p)} for t, p in models],
            "results": all_results,
        }, f, indent=2)

    # ---- Print final table ----
    print("\n" + "=" * 100)
    print("CROSS-SCALE / CROSS-DISTRIBUTION EVALUATION")
    print("=" * 100)
    header = f"{'Test Set':<25} " + " ".join(f"{s.upper():>10}" for s in solver_order)
    print(header)
    print("-" * 100)
    for label in labels:
        row = f"{label:<25} "
        for solver in solver_order:
            m = all_results.get(label, {}).get(solver)
            if m is None or m.get("avg_ratio") is None:
                row += f"{'—':>10} "
            else:
                row += f"{m['avg_ratio']:>10.4f} "
        print(row)
    print("=" * 100)
    mark(f"\nResults → {summary_csv}")
    mark(f"Details → {summary_json}")


if __name__ == "__main__":
    main()