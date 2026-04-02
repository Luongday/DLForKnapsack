"""Merge DP, GNN, Greedy, GA, and DQN per-instance results into comparison table.

Reads CSV from each solver, joins on instance_file, computes comparison metrics:
    - merged_results.csv        : per-instance side-by-side table
    - summary.json              : aggregate statistics per solver
    - infeasibility_report.json : infeasibility analysis

NEW vs original:
    - Greedy and GA solvers fully integrated
    - Cleaner solver registration (add new solvers in one place)
    - DQN/RL slot ready (uncomment when available)

Usage:
    python Merge_results.py
    python Merge_results.py --dp_csv results/DP/dp_results.csv \\
                            --gnn_csv results/GNN/gnn_eval_results.csv \\
                            --greedy_csv results/Greedy/greedy_eval_results.csv \\
                            --ga_csv results/GA/ga_eval_results.csv
    # When DQN is ready:
    # python Merge_results.py --dqn_csv results/DQN/dqn_eval_results.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(format="[MERGE] %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def _repo_root() -> Path:
    candidate = Path(__file__).resolve().parents[2]
    if candidate.exists() and (candidate / "results").exists():
        return candidate
    return Path.cwd()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(row: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        v = row.get(k)
        if v not in (None, "", "nan", "None", "null", "NaN"):
            try:
                return float(v)
            except (ValueError, TypeError):
                continue
    return None


def _safe_int(row: Dict[str, Any], *keys: str) -> Optional[int]:
    v = _safe_float(row, *keys)
    return int(v) if v is not None else None


def _check_feasible(weight: Optional[float], capacity: Optional[float]) -> int:
    if weight is None or capacity is None:
        return 0
    return 1 if weight <= capacity + 1e-6 else 0


def _gap(optimal: Optional[float], other: Optional[float], feasible: int) -> Optional[float]:
    if not feasible or optimal is None or other is None or optimal == 0:
        return None
    return round((optimal - other) / optimal, 6)


def _ratio(optimal: Optional[float], other: Optional[float], feasible: int) -> Optional[float]:
    if not feasible or optimal is None or other is None or optimal == 0:
        return None
    return round(other / optimal, 6)


def _parse_items(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    try:
        r = ast.literal_eval(s)
        return r if isinstance(r, list) else None
    except (ValueError, SyntaxError, TypeError):
        return None


def _items_out_of_range(selected_str: Optional[str], n_items: Optional[int]) -> Tuple[int, int]:
    if n_items is None:
        return 0, 0
    items = _parse_items(selected_str)
    if items is None:
        return 0, 0
    bad = sum(1 for i in items if i >= n_items or i < 0)
    return (1 if bad > 0 else 0), bad


def _infeasibility_reason(
    weight: Optional[float], capacity: Optional[float],
    selected_str: Optional[str], n_items: Optional[int],
) -> str:
    oor_flag, _ = _items_out_of_range(selected_str, n_items)
    if oor_flag:
        return "out_of_range"
    if weight is None or capacity is None:
        return "missing_data"
    if weight > capacity:
        return "weight_exceeded"
    return "ok"


# ---------------------------------------------------------------------------
# CSV Loader
# ---------------------------------------------------------------------------

def load_csv(path: Path, label: str) -> Optional[Dict[str, Dict[str, Any]]]:
    if not path.exists():
        log.warning("CSV not found: %s → skipped", path)
        return None

    rows: Dict[str, Dict[str, Any]] = {}
    duplicates = 0

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row.get("instance_file")
            if not key:
                continue
            if key in rows:
                duplicates += 1
            rows[key] = row

    if duplicates:
        log.warning("%s CSV has %d duplicates (last row kept)", label, duplicates)
    log.info("Loaded %d rows from %s", len(rows), label)
    return rows


# ---------------------------------------------------------------------------
# Stats accumulator
# ---------------------------------------------------------------------------

class _Stats:
    def __init__(self) -> None:
        self.count = 0
        self.feasible_count = 0
        self.oor_instances_count = 0
        self.total_oor_items = 0
        self.infeasibility_reasons: List[str] = []
        self.times:           List[float] = []
        self.values_all:      List[float] = []
        self.values_feasible: List[float] = []
        self.gaps:            List[float] = []
        self.ratios:          List[float] = []

    def update(self, feasible: int, time_ms=None, value=None,
               gap=None, ratio=None, oor_flag=0, oor_count=0,
               infeasibility_reason="ok"):
        self.count += 1
        self.feasible_count += feasible
        self.oor_instances_count += oor_flag
        self.total_oor_items += oor_count
        if not feasible:
            self.infeasibility_reasons.append(infeasibility_reason)
        if time_ms is not None: self.times.append(time_ms)
        if value is not None:
            self.values_all.append(value)
            if feasible:
                self.values_feasible.append(value)
        if gap   is not None: self.gaps.append(gap)
        if ratio is not None: self.ratios.append(ratio)

    def to_dict(self) -> Dict[str, Any]:
        def safe_mean(lst): return round(statistics.mean(lst), 4) if lst else None
        def safe_std(lst):  return round(statistics.stdev(lst), 4) if len(lst) > 1 else 0.0

        reason_counts = {}
        for r in self.infeasibility_reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1

        return {
            "count":                    self.count,
            "feasible":                 self.feasible_count,
            "infeasible":               self.count - self.feasible_count,
            "feasible_rate":            round(self.feasible_count / max(self.count, 1), 4),
            "oor_instances":            self.oor_instances_count,
            "oor_items_total":          self.total_oor_items,
            "infeasibility_breakdown":  reason_counts,
            "avg_time_ms":              safe_mean(self.times),
            "avg_value_all":            safe_mean(self.values_all),
            "avg_value_feasible_only":  safe_mean(self.values_feasible),
            "avg_gap_vs_dp_feasible":   safe_mean(self.gaps),
            "avg_ratio_vs_dp_feasible": safe_mean(self.ratios),
            "std_ratio_vs_dp":          safe_std(self.ratios),
        }


# ---------------------------------------------------------------------------
# Process one solver
# ---------------------------------------------------------------------------

def _process_solver(
    row: Optional[Dict], n_items: Optional[int], capacity: Optional[float],
    dp_value: Optional[float],
) -> Dict[str, Any]:
    """Extract metrics for one solver on one instance."""
    if row is None:
        return {
            "value": None, "weight": None, "time_ms": None,
            "selected": None, "feasible_reported": None,
            "feasible_actual": 0, "oor_flag": 0, "oor_count": 0,
            "infeas_reason": "missing_data", "gap": None, "ratio": None,
        }

    value     = _safe_float(row, "total_value", "value")
    weight    = _safe_float(row, "total_weight", "weight")
    time_ms   = _safe_float(row, "inference_time_ms", "time_ms")
    selected  = row.get("selected_items")
    feas_rep  = _safe_int(row, "feasible")

    feas_actual  = _check_feasible(weight, capacity)
    oor_flag, oor_count = _items_out_of_range(selected, n_items)
    infeas_reason = _infeasibility_reason(weight, capacity, selected, n_items)

    # If OOR items exist, mark as infeasible regardless of weight check
    if oor_flag:
        feas_actual = 0

    gap   = _gap(dp_value, value, feas_actual)
    ratio = _ratio(dp_value, value, feas_actual)

    return {
        "value": value, "weight": weight, "time_ms": time_ms,
        "selected": selected, "feasible_reported": feas_rep,
        "feasible_actual": feas_actual, "oor_flag": oor_flag,
        "oor_count": oor_count, "infeas_reason": infeas_reason,
        "gap": gap, "ratio": ratio,
    }


# ---------------------------------------------------------------------------
# Core merge
# ---------------------------------------------------------------------------

def merge(
    dp_rows:      Optional[Dict] = None,
    gnn_rows:     Optional[Dict] = None,
    greedy_rows:  Optional[Dict] = None,
    ga_rows:      Optional[Dict] = None,
    dqn_rows:     Optional[Dict] = None,
    out_dir:      Path = Path("results/compare"),
    verbose:      bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all solver data
    solver_data = {
        "dp":      dp_rows,
        "gnn":     gnn_rows,
        "greedy":  greedy_rows,
        "ga":      ga_rows,
        "dqn":     dqn_rows,
    }
    active_solvers = {k: v for k, v in solver_data.items() if v is not None}

    # Find common instances (must have DP as ground truth)
    all_keys = set()
    for rows in active_solvers.values():
        all_keys |= rows.keys()

    # Common = present in DP and at least one other solver
    common_keys = sorted(
        k for k in all_keys
        if dp_rows is not None and k in dp_rows
    )
    log.info(f"Total unique instances: {len(all_keys)} | With DP ground truth: {len(common_keys)}")

    # Stats per solver
    stats = {name: _Stats() for name in active_solvers}

    records = []

    for key in common_keys:
        dp_row = dp_rows.get(key) if dp_rows else None
        n_items  = _safe_int(dp_row or {}, "n_items")
        capacity = _safe_float(dp_row or {}, "capacity")

        # DP ground truth
        dp_m = _process_solver(dp_row, n_items, capacity, None)
        dp_value = dp_m["value"]

        # Update DP stats (ratio to self = 1.0)
        if "dp" in stats and dp_row:
            stats["dp"].update(
                feasible=dp_m["feasible_actual"],
                time_ms=dp_m["time_ms"], value=dp_m["value"],
                gap=0.0 if dp_m["feasible_actual"] else None,
                ratio=1.0 if dp_m["feasible_actual"] else None,
            )

        # Build record
        record = {
            "instance_file": key,
            "n_items":       n_items,
            "capacity":      capacity,
            "dp_value":      dp_value,
            "dp_weight":     dp_m["weight"],
            "dp_feasible":   dp_m["feasible_actual"],
            "dp_time_ms":    dp_m["time_ms"],
            "dp_selected_items": dp_m["selected"],
        }

        # Process each non-DP solver
        for solver_name in ["gnn", "greedy", "ga", "dqn"]:
            solver_rows = solver_data.get(solver_name)
            if solver_rows is None:
                # Add None columns for missing solvers
                for col in ["value", "weight", "feasible_actual", "time_ms",
                            "gap", "ratio", "selected_items"]:
                    record[f"{solver_name}_{col}"] = None
                if solver_name == "gnn":
                    record["gnn_feasible_reported"] = None
                    record["gnn_items_out_of_range"] = None
                    record["gnn_oor_item_count"] = None
                    record["gnn_infeasibility_reason"] = None
                continue

            row = solver_rows.get(key)
            m   = _process_solver(row, n_items, capacity, dp_value)

            record[f"{solver_name}_value"]          = m["value"]
            record[f"{solver_name}_weight"]         = m["weight"]
            record[f"{solver_name}_feasible_actual"] = m["feasible_actual"]
            record[f"{solver_name}_time_ms"]        = m["time_ms"]
            record[f"{solver_name}_gap"]            = m["gap"]
            record[f"{solver_name}_ratio"]          = m["ratio"]
            record[f"{solver_name}_selected_items"] = m["selected"]

            # Extra GNN columns
            if solver_name == "gnn":
                record["gnn_feasible_reported"]    = m["feasible_reported"]
                record["gnn_items_out_of_range"]   = m["oor_flag"]
                record["gnn_oor_item_count"]       = m["oor_count"]
                record["gnn_infeasibility_reason"] = m["infeas_reason"]

            # Update stats
            if solver_name in stats and row is not None:
                stats[solver_name].update(
                    feasible=m["feasible_actual"],
                    time_ms=m["time_ms"], value=m["value"],
                    gap=m["gap"], ratio=m["ratio"],
                    oor_flag=m["oor_flag"], oor_count=m["oor_count"],
                    infeasibility_reason=m["infeas_reason"],
                )

        records.append(record)

    # Write merged CSV
    merged_csv = out_dir / "merged_results.csv"
    if records:
        with merged_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        log.info("Merged CSV → %s (%d rows)", merged_csv, len(records))

    # Summary JSON
    summary: Dict[str, Any] = {"n_common_instances": len(common_keys)}
    for name, s in stats.items():
        summary[name] = s.to_dict()

    summary_json = out_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary → %s", summary_json)

    # Infeasibility report
    infeas_report = {}
    for name, s in stats.items():
        d = s.to_dict()
        infeas_report[name] = {
            "infeasible_count":    d["infeasible"],
            "infeasible_rate":     round(1 - (d["feasible_rate"] or 0), 6),
            "oor_instances":       d["oor_instances"],
            "oor_items_total":     d["oor_items_total"],
            "breakdown_by_reason": d["infeasibility_breakdown"],
        }

    infeas_path = out_dir / "infeasibility_report.json"
    with infeas_path.open("w", encoding="utf-8") as f:
        json.dump(infeas_report, f, indent=2)
    log.info("Infeasibility report → %s", infeas_path)

    # Print comparison table
    _print_comparison_table(stats)


def _print_comparison_table(stats: Dict[str, _Stats]) -> None:
    print("\n" + "=" * 110)
    print("SUMMARY COMPARISON — ALL SOLVERS")
    print("=" * 110)
    print(f"{'Solver':>8} | {'Feasible':>12} | {'Rate':>7} | "
          f"{'Avg Value (feas)':>16} | {'Avg Time (ms)':>14} | "
          f"{'Avg Ratio vs DP':>16} | {'Std':>8}")
    print("-" * 110)

    for name, s in stats.items():
        d = s.to_dict()
        ratio_str = f"{d['avg_ratio_vs_dp_feasible']:.4f}" if d.get("avg_ratio_vs_dp_feasible") is not None else "N/A"
        std_str   = f"{d['std_ratio_vs_dp']:.4f}" if d.get("std_ratio_vs_dp") is not None else "N/A"
        val_str   = f"{d['avg_value_feasible_only']:.2f}" if d.get("avg_value_feasible_only") is not None else "N/A"
        time_str  = f"{d['avg_time_ms']:.4f}" if d.get("avg_time_ms") is not None else "N/A"
        print(
            f"{name.upper():>8} | "
            f"{d['feasible']}/{d['count']:>4} "
            f"({d['feasible_rate']:.1%}) | "
            f"{val_str:>16} | "
            f"{time_str:>14} | "
            f"{ratio_str:>16} | "
            f"{std_str:>8}"
        )
    print("=" * 110)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Merge DP / GNN / Greedy / GA / DQN results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dp_csv",      type=Path,
                        default=root / "results" / "DP" / "dp_results.csv")
    parser.add_argument("--gnn_csv",     type=Path,
                        default=root / "results" / "GNN" / "gnn_eval_results.csv")
    parser.add_argument("--greedy_csv",  type=Path,
                        default=root / "results" / "Greedy" / "greedy_eval_results.csv")
    parser.add_argument("--ga_csv",      type=Path,
                        default=root / "results" / "GA" / "ga_eval_results.csv")
    parser.add_argument("--dqn_csv",     type=Path, default=None,
                        help="[Optional] DQN eval CSV. Leave empty if not ready.")
    parser.add_argument("--out_dir",     type=Path,
                        default=root / "results" / "compare")
    parser.add_argument("--skip_missing", action="store_true",
                        help="Continue even if some CSVs are missing")
    parser.add_argument("--verbose",     action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    dp_rows      = load_csv(args.dp_csv,     "DP")
    gnn_rows     = load_csv(args.gnn_csv,    "GNN")
    greedy_rows  = load_csv(args.greedy_csv, "Greedy")
    ga_rows      = load_csv(args.ga_csv,     "GA")
    dqn_rows     = load_csv(args.dqn_csv, "DQN") if args.dqn_csv else None

    if dp_rows is None and not args.skip_missing:
        raise SystemExit("DP results required as ground truth. Run dp_baseline_eval.py first.")

    merge(
        dp_rows=dp_rows,
        gnn_rows=gnn_rows,
        greedy_rows=greedy_rows,
        ga_rows=ga_rows,
        dqn_rows=dqn_rows,
        out_dir=args.out_dir,
        verbose=args.verbose,
    )