"""Merge DP, GNN, and DQN per-instance results into a comparison table.

Reads three CSV files (one per solver), joins on instance_file, computes
comparison metrics, and writes:
    - merged_results.csv  : per-instance side-by-side table
    - summary.json        : aggregate statistics + DQN training metadata

Incorporates augment_results_with_meta.py (previously a separate script).

Usage:
    python merge_results.py
    python merge_results.py --dp_csv results/DP/dp_results.csv \\
                            --gnn_csv results/GNN/gnn_eval_results.csv \\
                            --dqn_csv results/DQN/eval_results.csv \\
                            --out_dir results/compare
    python merge_results.py --skip_missing   # run even if some CSVs absent
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mark(msg: str) -> None:
    print(f"[MERGE] {msg}", flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_float(row: Dict[str, Any], *keys: str) -> Optional[float]:
    """Try multiple key names; return first parseable float or None."""
    for k in keys:
        v = row.get(k)
        if v not in (None, "", "nan", "None"):
            try:
                return float(v)
            except (ValueError, TypeError):
                continue
    return None


def _safe_int(row: Dict[str, Any], *keys: str) -> Optional[int]:
    v = _safe_float(row, *keys)
    return int(v) if v is not None else None


def _gap(optimal: Optional[float], other: Optional[float]) -> Optional[float]:
    """Optimality gap: (optimal - other) / optimal.  0 = perfect, 1 = worst."""
    if optimal is None or other is None or optimal == 0:
        return None
    return round((optimal - other) / optimal, 6)


def _ratio(optimal: Optional[float], other: Optional[float]) -> Optional[float]:
    """Approximation ratio: other / optimal.  1.0 = perfect."""
    if optimal is None or other is None or optimal == 0:
        return None
    return round(other / optimal, 6)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_csv(path: Path, label: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Load a result CSV into a dict keyed by instance_file.

    Returns None (with a warning) if the file does not exist.
    """
    if not path.exists():
        mark(f"[WARN] {label} CSV not found: {path}  → skipped")
        return None
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row.get("instance_file")
            if key:
                rows[key] = row
    mark(f"Loaded {len(rows):>5} rows  [{label}]  {path.name}")
    return rows


# ---------------------------------------------------------------------------
# Stats accumulator
# ---------------------------------------------------------------------------

class _Stats:
    def __init__(self):
        self.count    = 0
        self.feasible = 0
        self.times:  List[float] = []
        self.values: List[float] = []
        self.gaps:   List[float] = []
        self.ratios: List[float] = []

    def update(
        self,
        feasible:  int,
        time_ms:   Optional[float],
        value:     Optional[float],
        gap:       Optional[float],
        ratio:     Optional[float],
    ):
        self.count    += 1
        self.feasible += feasible
        if time_ms is not None:  self.times.append(time_ms)
        if value   is not None:  self.values.append(value)
        if gap     is not None:  self.gaps.append(gap)
        if ratio   is not None:  self.ratios.append(ratio)

    def _avg(self, lst: List[float]) -> Optional[float]:
        return round(sum(lst) / len(lst), 6) if lst else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count":            self.count,
            "feasible":         self.feasible,
            "feasible_rate":    round(self.feasible / self.count, 4) if self.count else None,
            "avg_value":        self._avg(self.values),
            "avg_time_ms":      self._avg(self.times),
            "avg_gap_vs_dp":    self._avg(self.gaps),
            "avg_ratio_vs_dp":  self._avg(self.ratios),
        }


# ---------------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------------

def merge(
    dp_rows:  Optional[Dict[str, Dict[str, Any]]],
    gnn_rows: Optional[Dict[str, Dict[str, Any]]],
    dqn_rows: Optional[Dict[str, Dict[str, Any]]],
    dqn_meta_path: Optional[Path],
    out_dir:  Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find common instance keys across all available solvers
    available = [r for r in [dp_rows, gnn_rows, dqn_rows] if r is not None]
    if not available:
        raise RuntimeError("No result files loaded — nothing to merge.")

    common_keys = sorted(set.intersection(*[set(r.keys()) for r in available]))
    if not common_keys:
        raise RuntimeError(
            "No overlapping instance_file keys found across the loaded CSVs."
        )
    mark(f"Common instances: {len(common_keys)}")

    # Accumulators
    s_dp  = _Stats()
    s_gnn = _Stats()
    s_dqn = _Stats()

    records: List[Dict[str, Any]] = []

    for key in common_keys:
        dp  = dp_rows.get(key)  if dp_rows  else None
        gnn = gnn_rows.get(key) if gnn_rows else None
        dqn = dqn_rows.get(key) if dqn_rows else None

        # Shared metadata (prefer DP as ground truth)
        src = dp or gnn or dqn or {}
        n_items  = _safe_int(src, "n_items")
        capacity = _safe_float(src, "capacity")

        # DP (optimal reference)
        dp_value    = _safe_float(dp,  "total_value", "dp_value", "value") if dp else None
        dp_weight   = _safe_float(dp,  "total_weight", "dp_weight")        if dp else None
        dp_feasible = _safe_int(dp,    "feasible") if dp else None
        dp_time_ms  = _safe_float(dp,  "inference_time_ms", "time_ms")      if dp else None
        dp_sel      = dp.get("selected_items")                               if dp else None

        # GNN
        gnn_value    = _safe_float(gnn, "total_value", "gnn_value", "value") if gnn else None
        gnn_weight   = _safe_float(gnn, "total_weight", "gnn_weight")         if gnn else None
        gnn_feasible = _safe_int(gnn,   "feasible") if gnn else None
        gnn_time_ms  = _safe_float(gnn, "inference_time_ms", "time_ms")       if gnn else None
        gnn_sel      = gnn.get("selected_items")                               if gnn else None

        # DQN
        dqn_value    = _safe_float(dqn, "total_value_selected", "total_value", "value") if dqn else None
        dqn_weight   = _safe_float(dqn, "total_weight_selected", "total_weight")         if dqn else None
        dqn_feasible = _safe_int(dqn,   "feasible") if dqn else None
        dqn_time_ms  = _safe_float(dqn, "inference_time_ms", "time_ms")                  if dqn else None
        dqn_sel      = dqn.get("selected_items")                                          if dqn else None

        # Metrics vs DP optimal
        gap_gnn   = _gap(dp_value, gnn_value)
        gap_dqn   = _gap(dp_value, dqn_value)
        ratio_gnn = _ratio(dp_value, gnn_value)
        ratio_dqn = _ratio(dp_value, dqn_value)

        # Update stats
        if dp is not None:
            s_dp.update(dp_feasible or 1, dp_time_ms, dp_value, None, None)
        if gnn is not None:
            s_gnn.update(gnn_feasible or 0, gnn_time_ms, gnn_value, gap_gnn, ratio_gnn)
        if dqn is not None:
            s_dqn.update(dqn_feasible or 0, dqn_time_ms, dqn_value, gap_dqn, ratio_dqn)

        records.append({
            "instance_file":    key,
            "n_items":          n_items,
            "capacity":         capacity,
            # DP
            "dp_value":         dp_value,
            "dp_weight":        dp_weight,
            "dp_feasible":      dp_feasible,
            "dp_time_ms":       dp_time_ms,
            # GNN
            "gnn_value":        gnn_value,
            "gnn_weight":       gnn_weight,
            "gnn_feasible":     gnn_feasible,
            "gnn_time_ms":      gnn_time_ms,
            "gap_gnn":          gap_gnn,
            "ratio_gnn":        ratio_gnn,
            # DQN
            "dqn_value":        dqn_value,
            "dqn_weight":       dqn_weight,
            "dqn_feasible":     dqn_feasible,
            "dqn_time_ms":      dqn_time_ms,
            "gap_dqn":          gap_dqn,
            "ratio_dqn":        ratio_dqn,
            # Selected items
            "dp_selected_items":  dp_sel,
            "gnn_selected_items": gnn_sel,
            "dqn_selected_items": dqn_sel,
        })

    # Write merged CSV
    merged_csv = out_dir / "merged_results.csv"
    fieldnames = list(records[0].keys()) if records else []
    with merged_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    mark(f"Merged CSV  → {merged_csv}  ({len(records)} rows)")

    # Build summary
    summary: Dict[str, Any] = {
        "n_common_instances": len(common_keys),
        "dp":  s_dp.to_dict()  if dp_rows  else None,
        "gnn": s_gnn.to_dict() if gnn_rows else None,
        "dqn": s_dqn.to_dict() if dqn_rows else None,
    }

    # Incorporate DQN training metadata (was augment_results_with_meta.py)
    if dqn_meta_path and dqn_meta_path.exists():
        with dqn_meta_path.open(encoding="utf-8") as f:
            dqn_meta = json.load(f)
        summary["dqn_training"] = {
            "algorithm":               "DQN",
            "training_steps":          dqn_meta.get("total_steps", 50000),
            "original_planned_steps":  200000,
            "note": "Training steps reduced due to hardware constraints",
        }
        mark(f"DQN training metadata injected from {dqn_meta_path.name}")
    else:
        mark("[INFO] DQN training meta not found — skipped (non-critical)")

    summary_json = out_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    mark(f"Summary JSON → {summary_json}")

    # Print quick comparison table
    print("\n  Solver   | feasible |  avg_value |  avg_time_ms | avg_ratio_vs_dp")
    print("  " + "-" * 62)
    for label, s in [("DP", s_dp), ("GNN", s_gnn), ("DQN", s_dqn)]:
        d = s.to_dict()
        feas  = f"{d['feasible_rate']:.3f}"   if d['feasible_rate']   is not None else "  n/a "
        val   = f"{d['avg_value']:.2f}"        if d['avg_value']       is not None else "    n/a"
        t     = f"{d['avg_time_ms']:.3f}"      if d['avg_time_ms']     is not None else "      n/a"
        ratio = f"{d['avg_ratio_vs_dp']:.4f}"  if d['avg_ratio_vs_dp'] is not None else "     n/a"
        print(f"  {label:<8} | {feas:>8} | {val:>10} | {t:>12} | {ratio:>15}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Merge DP / GNN / DQN results into a comparison table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dp_csv",  type=Path,
        default=root / "results" / "DP"  / "dp_results.csv",
    )
    parser.add_argument(
        "--gnn_csv", type=Path,
        default=root / "results" / "GNN" / "gnn_eval_results.csv",
    )
    parser.add_argument(
        "--dqn_csv", type=Path,
        default=root / "results" / "DQN" / "eval_results.csv",
    )
    parser.add_argument(
        "--dqn_meta", type=Path,
        default=root / "results" / "DQN" / "train_meta.json",
        help="DQN training metadata JSON (optional)",
    )
    parser.add_argument(
        "--out_dir", type=Path,
        default=root / "results" / "compare",
        help="Output directory for merged_results.csv and summary.json",
    )
    parser.add_argument(
        "--skip_missing", action="store_true",
        help="Continue even if some result CSVs are absent",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dp_rows  = load_csv(args.dp_csv,  "DP")
    gnn_rows = load_csv(args.gnn_csv, "GNN")
    dqn_rows = load_csv(args.dqn_csv, "DQN")

    missing = [l for l, r in [("DP", dp_rows), ("GNN", gnn_rows), ("DQN", dqn_rows)] if r is None]
    if missing and not args.skip_missing:
        raise SystemExit(
            f"Missing result files: {missing}\n"
            "Run with --skip_missing to proceed without them."
        )

    merge(
        dp_rows=dp_rows,
        gnn_rows=gnn_rows,
        dqn_rows=dqn_rows,
        dqn_meta_path=args.dqn_meta,
        out_dir=args.out_dir,
    )