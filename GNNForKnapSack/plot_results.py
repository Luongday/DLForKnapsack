"""Visualization for Knapsack solver comparison. """

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed. Install with: pip install matplotlib")


def mark(msg: str):
    print(f"[PLOT] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Solver comparison bar chart
# ---------------------------------------------------------------------------

def plot_solver_comparison(summary_path: Path, out_dir: Path) -> None:
    """Bar chart comparing ratio, time, feasibility across solvers."""
    with summary_path.open() as f:
        summary = json.load(f)

    solvers = []
    ratios = []
    times = []
    labels_detail = []

    solver_order = ["dp", "ga", "greedy", "gnn", "dqn", "s2v", "reinforce"]
    solver_colors = {
        "dp": "#5F5E5A", "ga": "#1D9E75", "greedy": "#7F77DD",
        "gnn": "#D85A30", "dqn": "#378ADD", "s2v": "#E8A33D",
        "reinforce": "#A23B72",
    }

    for name in solver_order:
        if name not in summary:
            continue
        d = summary[name]
        r = d.get("avg_ratio_vs_dp_feasible")
        t = d.get("avg_time_ms")
        if r is None:
            continue
        solvers.append(name.upper())
        ratios.append(r)
        times.append(t if t else 0)
        labels_detail.append(f"{r:.4f}")

    if not solvers:
        mark("No solver data found in summary.json")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ratio chart
    ax = axes[0]
    colors = [solver_colors.get(s.lower(), "#888") for s in solvers]
    bars = ax.bar(solvers, ratios, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Approximation Ratio (vs DP)")
    ax.set_title("Solution Quality")
    ax.set_ylim(min(ratios) - 0.02, 1.005)
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, label="Optimal")
    for bar, label in zip(bars, labels_detail):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                label, ha="center", va="bottom", fontsize=9)

    # Time chart (log scale)
    ax = axes[1]
    bars = ax.bar(solvers, times, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Avg Time (ms)")
    ax.set_title("Solve Speed")
    ax.set_yscale("log")
    for bar, t in zip(bars, times):
        label = f"{t:.2f}" if t < 10 else f"{t:.0f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                label, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = out_dir / "solver_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    mark(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 2. Per-instance ratio distribution
# ---------------------------------------------------------------------------

def plot_ratio_distribution(merged_csv: Path, out_dir: Path) -> None:
    """Histogram of ratio distribution per solver."""
    import csv

    with merged_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return

    solver_cols = {
        "GNN": "gnn_ratio", "Greedy": "greedy_ratio",
        "GA": "ga_ratio", "DQN": "dqn_ratio", "S2V-DQN": "s2v_ratio",
        "REINFORCE": "reinforce_ratio",
    }
    solver_colors_map = {
        "GNN": "#D85A30", "Greedy": "#7F77DD",
        "GA": "#1D9E75", "DQN": "#378ADD", "S2V-DQN": "#E8A33D",
        "REINFORCE": "#A23B72",
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    for name, col in solver_cols.items():
        vals = []
        for r in rows:
            v = r.get(col)
            if v and v not in ("", "None", "nan"):
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        if vals:
            ax.hist(vals, bins=30, alpha=0.5, label=f"{name} (n={len(vals)})",
                    color=solver_colors_map.get(name, "#888"), edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Approximation Ratio (vs DP)")
    ax.set_ylabel("Count")
    ax.set_title("Ratio Distribution per Solver")
    ax.legend()
    ax.axvline(x=1.0, color="black", linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = out_dir / "ratio_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    mark(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 3. Ratio by problem size
# ---------------------------------------------------------------------------

def plot_ratio_by_size(merged_csv: Path, out_dir: Path) -> None:
    """Line chart: ratio vs n_items for each solver."""
    import csv

    with merged_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    solver_info = [
        ("GNN", "gnn_ratio", "#D85A30"),
        ("Greedy", "greedy_ratio", "#7F77DD"),
        ("GA", "ga_ratio", "#1D9E75"),
        ("DQN", "dqn_ratio", "#378ADD"),
        ("S2V-DQN", "s2v_ratio", "#E8A33D"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    bins = [(10, 25), (25, 40), (40, 60), (60, 80), (80, 101)]
    bin_labels = ["10-25", "25-40", "40-60", "60-80", "80-100"]

    for name, col, color in solver_info:
        means = []
        valid = False
        for lo, hi in bins:
            vals = []
            for r in rows:
                try:
                    n = int(r.get("n_items", 0))
                    v = r.get(col)
                    if lo <= n < hi and v and v not in ("", "None", "nan"):
                        vals.append(float(v))
                except (ValueError, TypeError):
                    pass
            means.append(np.mean(vals) if vals else None)
            if vals:
                valid = True

        if valid:
            x_vals = list(range(len(bins)))
            y_vals = [m for m in means if m is not None]
            x_valid = [x for x, m in zip(x_vals, means) if m is not None]
            ax.plot(x_valid, y_vals, "o-", label=name, color=color, linewidth=2, markersize=6)

    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Problem Size (n_items)")
    ax.set_ylabel("Avg Approximation Ratio")
    ax.set_title("Solution Quality by Problem Size")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / "ratio_by_size.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    mark(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 4. Training curve
# ---------------------------------------------------------------------------

def plot_training_curve(log_csv: Path, out_dir: Path) -> None:
    """Plot training curve — auto-detects GNN (epoch) or RL (step) format.

    GNN log schema: epoch, train_loss, gnn_ratio, greedy_ratio, ...
    RL log schema:  step, updates, epsilon, loss, avg_value_val, best_value_val, ...
    """
    import csv

    with log_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        mark("Training log is empty")
        return

    header = rows[0].keys()
    is_rl = ("step" in header and "avg_value_val" in header)
    is_gnn = ("epoch" in header and "gnn_ratio" in header)

    if is_rl:
        _plot_rl_training_curve(rows, log_csv, out_dir)
    elif is_gnn:
        _plot_gnn_training_curve(rows, log_csv, out_dir)
    else:
        mark(f"Unknown log format in {log_csv.name} (header: {list(header)[:5]}...)")


def _plot_gnn_training_curve(rows: list, log_csv: Path, out_dir: Path) -> None:
    """Plot GNN training log: epoch × (ratio, loss)."""
    epochs = [int(r["epoch"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ratio over epochs
    ax = axes[0]
    gnn_ratios = [float(r["gnn_ratio"]) for r in rows]
    ax.plot(epochs, gnn_ratios, "-", color="#D85A30", linewidth=2, label="GNN ratio")

    if "greedy_ratio" in rows[0] and rows[0]["greedy_ratio"]:
        gr_ratio = float(rows[0]["greedy_ratio"])
        ax.axhline(y=gr_ratio, color="#7F77DD", linestyle="--", linewidth=1.5,
                   label=f"Greedy baseline ({gr_ratio:.4f})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Approximation Ratio")
    ax.set_title("GNN Training: Quality")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Loss
    ax = axes[1]
    if "train_loss" in rows[0]:
        losses = [float(r["train_loss"]) for r in rows if r["train_loss"]]
        ax.plot(epochs[:len(losses)], losses, "-", color="#D85A30", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("GNN Training: Loss")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    stem = log_csv.stem
    path = out_dir / f"{stem}_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    mark(f"Saved: {path}")


def _plot_rl_training_curve(rows: list, log_csv: Path, out_dir: Path) -> None:
    """Plot RL training log (DQN / S2V-DQN): step × (val_value, loss, epsilon)."""
    steps = [int(r["step"]) for r in rows]

    # Extract series (skip empty cells)
    def col(name: str) -> list:
        return [float(r[name]) if r.get(name) not in ("", None) else None for r in rows]

    avg_vals = col("avg_value_val")
    losses   = col("loss")
    eps      = col("epsilon")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Avg validation value over steps
    ax = axes[0]
    valid_pairs = [(s, v) for s, v in zip(steps, avg_vals) if v is not None]
    if valid_pairs:
        xs, ys = zip(*valid_pairs)
        ax.plot(xs, ys, "-o", color="#378ADD", linewidth=2, markersize=4, label="Val avg value")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Avg Value on Val Set")
    ax.set_title("RL Training: Validation Quality")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    # 2. Loss over steps
    ax = axes[1]
    valid_pairs = [(s, v) for s, v in zip(steps, losses) if v is not None]
    if valid_pairs:
        xs, ys = zip(*valid_pairs)
        ax.plot(xs, ys, "-", color="#D85A30", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("RL Training: Loss")
    ax.grid(axis="y", alpha=0.3)

    # 3. Epsilon schedule
    ax = axes[2]
    valid_pairs = [(s, v) for s, v in zip(steps, eps) if v is not None]
    if valid_pairs:
        xs, ys = zip(*valid_pairs)
        ax.plot(xs, ys, "-", color="#1D9E75", linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Epsilon")
    ax.set_title("Epsilon-Greedy Schedule")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    stem = log_csv.stem
    path = out_dir / f"{stem}_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    mark(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 5. GNN vs Greedy head-to-head
# ---------------------------------------------------------------------------

def plot_head_to_head(merged_csv: Path, out_dir: Path) -> None:
    """Scatter plot: GNN ratio vs Greedy ratio per instance."""
    import csv

    with merged_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    gnn_vals, gr_vals = [], []
    for r in rows:
        gnn_r = r.get("gnn_ratio")
        gr_r = r.get("greedy_ratio")
        if gnn_r and gr_r and gnn_r not in ("", "None") and gr_r not in ("", "None"):
            try:
                gnn_vals.append(float(gnn_r))
                gr_vals.append(float(gr_r))
            except ValueError:
                pass

    if not gnn_vals:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(gr_vals, gnn_vals, alpha=0.4, s=20, color="#D85A30", edgecolors="white", linewidth=0.3)
    lims = [min(min(gnn_vals), min(gr_vals)) - 0.02, 1.01]
    ax.plot(lims, lims, "--", color="black", alpha=0.3, label="Equal performance")
    ax.set_xlabel("Greedy Ratio")
    ax.set_ylabel("GNN Ratio")
    ax.set_title("GNN vs Greedy (per instance)")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Count wins
    gnn_wins = sum(1 for g, r in zip(gnn_vals, gr_vals) if g > r + 1e-6)
    gr_wins = sum(1 for g, r in zip(gnn_vals, gr_vals) if r > g + 1e-6)
    ties = len(gnn_vals) - gnn_wins - gr_wins
    ax.text(0.05, 0.95, f"GNN wins: {gnn_wins}\nGreedy wins: {gr_wins}\nTies: {ties}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    path = out_dir / "gnn_vs_greedy.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    mark(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 6. Cross-scale / cross-distribution plot
# ---------------------------------------------------------------------------

def plot_cross_scale(cross_scale_csv: Path, out_dir: Path) -> None:
    """Grouped bar chart: ratio by test set × solver.

    Visualizes output of cross_scale_eval.py showing scalability
    and generalization across different test distributions.
    """
    import csv

    with cross_scale_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        mark("Cross-scale CSV is empty")
        return

    # Organize: test_set -> solver -> ratio
    data: Dict[str, Dict[str, float]] = {}
    solvers_seen: List[str] = []

    for r in rows:
        ts = r["test_set"]
        sv = r["solver"]
        try:
            ratio = float(r["avg_ratio"]) if r["avg_ratio"] else None
        except ValueError:
            ratio = None
        if ratio is None:
            continue
        data.setdefault(ts, {})[sv] = ratio
        if sv not in solvers_seen:
            solvers_seen.append(sv)

    if not data:
        return

    # Preserve test set order from CSV
    test_sets = list(data.keys())

    # Canonical solver colors (match other plots)
    color_map = {
        "DP": "#5F5E5A", "GA": "#1D9E75", "GREEDY": "#7F77DD",
        "GNN": "#D85A30", "DQN": "#378ADD", "S2V": "#E8A33D",
        "REINFORCE": "#A23B72",
    }

    fig, ax = plt.subplots(figsize=(max(10, len(test_sets) * 2), 6))
    n_solvers = len(solvers_seen)
    bar_width = 0.8 / max(n_solvers, 1)
    x_base = np.arange(len(test_sets))

    for i, solver in enumerate(solvers_seen):
        heights = [data[ts].get(solver, 0) for ts in test_sets]
        offsets = x_base + (i - n_solvers / 2 + 0.5) * bar_width
        color = color_map.get(solver.upper(), "#888")
        ax.bar(offsets, heights, bar_width, label=solver,
               color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x_base)
    ax.set_xticklabels(test_sets, rotation=25, ha="right")
    ax.set_ylabel("Approximation Ratio (vs DP)")
    ax.set_title("Cross-scale / Cross-distribution Evaluation")
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.3)

    # y-range: zoom in if all values > 0.9
    all_vals = [v for d in data.values() for v in d.values()]
    if all_vals and min(all_vals) > 0.9:
        ax.set_ylim(min(all_vals) - 0.02, 1.01)

    ax.legend(loc="lower left", ncol=min(n_solvers, 4))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / "cross_scale.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    mark(f"Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot solver comparison and training results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", type=Path, default=None,
                        help="Directory with merged_results.csv and summary.json")
    parser.add_argument("--training_log", type=Path, default=None,
                        help="Training log CSV (Run_train.py, train_dqn.py, train_s2v_dqn.py)")
    parser.add_argument("--training_logs", type=Path, nargs="*", default=None,
                        help="Multiple training logs to plot")
    parser.add_argument("--cross_scale_csv", type=Path, default=None,
                        help="Cross-scale summary CSV (from cross_scale_eval.py)")
    parser.add_argument("--out_dir", type=Path, default=Path("plots"),
                        help="Output directory for charts")
    return parser.parse_args()


def main():
    if not HAS_MPL:
        print("ERROR: matplotlib required. pip install matplotlib")
        sys.exit(1)

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.results_dir:
        summary_path = args.results_dir / "summary.json"
        merged_path  = args.results_dir / "merged_results.csv"

        if summary_path.exists():
            plot_solver_comparison(summary_path, args.out_dir)

        if merged_path.exists():
            plot_ratio_distribution(merged_path, args.out_dir)
            plot_ratio_by_size(merged_path, args.out_dir)
            plot_head_to_head(merged_path, args.out_dir)

    if args.training_log and args.training_log.exists():
        plot_training_curve(args.training_log, args.out_dir)

    if args.training_logs:
        for log in args.training_logs:
            if Path(log).exists():
                plot_training_curve(Path(log), args.out_dir)

    if args.cross_scale_csv and args.cross_scale_csv.exists():
        plot_cross_scale(args.cross_scale_csv, args.out_dir)

    if not any([args.results_dir, args.training_log, args.training_logs,
                args.cross_scale_csv]):
        print("No input specified.")
        print("Examples:")
        print("  python plot_results.py --results_dir results/compare")
        print("  python plot_results.py --training_log logs/training_log.csv")
        print("  python plot_results.py --cross_scale_csv results/cross_scale/cross_scale_results.csv")

    mark(f"All plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()