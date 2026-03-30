"""Quick data checker for generated Knapsack NPZ datasets.

Usage:
    python check_data.py data/knapsack_ilp/train
    python check_data.py data/knapsack_ilp/train --n 20
    python check_data.py data/knapsack_ilp/train --show 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def check_instance(path: Path) -> Dict:
    """Load one NPZ and run all validation checks. Returns a result dict."""
    arr = np.load(path, allow_pickle=True)

    def pick(keys):
        for k in keys:
            if k in arr.files:
                return arr[k]
        return None

    w_arr   = pick(["weights", "w", "W"])
    v_arr   = pick(["values",  "v", "V"])
    c_raw   = pick(["capacity", "cap", "C"])
    sol_arr = pick(["solution", "selected", "y"])
    opt_raw = pick(["dp_value", "optimal_value", "opt"])

    if w_arr is None or v_arr is None or c_raw is None or sol_arr is None:
        missing = [k for k, v in
                   [("weights", w_arr), ("values", v_arr),
                    ("capacity", c_raw), ("solution", sol_arr)]
                   if v is None]
        return {
            "file": path.name, "n_items": 0, "capacity": 0,
            "n_selected": 0, "fill_ratio": 0.0, "opt_value": None,
            "feasible": False,
            "errors": [f"MISSING KEYS: {missing}"],
        }

    # .item() safely converts 0-d numpy arrays to Python scalars
    w   = np.asarray(w_arr).reshape(-1).astype(int)
    v   = np.asarray(v_arr).reshape(-1).astype(int)
    c   = int(np.asarray(c_raw).reshape(()).item())
    sol = np.asarray(sol_arr).reshape(-1).astype(int)
    opt = int(np.asarray(opt_raw).reshape(()).item()) if opt_raw is not None else None

    total_w    = int((w * sol).sum())
    total_v    = int((v * sol).sum())
    n_selected = int(sol.sum())

    errors: List[str] = []

    if total_w > c:
        errors.append(f"CAPACITY VIOLATED: weight={total_w} > cap={c}")
    if opt is not None and total_v != opt:
        errors.append(f"VALUE MISMATCH: computed={total_v} != saved={opt}")
    if not set(sol.tolist()).issubset({0, 1}):
        errors.append("SOLUTION NOT BINARY")
    if (w <= 0).any():
        errors.append("NON-POSITIVE WEIGHT found")
    if (v <= 0).any():
        errors.append("NON-POSITIVE VALUE found")
    if c <= 0:
        errors.append("NON-POSITIVE CAPACITY")
    if n_selected == 0:
        errors.append("WARN: no items selected (very tight capacity?)")
    if opt is None:
        errors.append("WARN: dp_value key missing — value check skipped")

    return {
        "file":       path.name,
        "n_items":    len(w),
        "capacity":   c,
        "n_selected": n_selected,
        "fill_ratio": total_w / c if c > 0 else 0.0,
        "opt_value":  opt if opt is not None else total_v,
        "feasible":   total_w <= c,
        "errors":     errors,
    }


def print_summary(results: List[Dict]) -> None:
    n = len(results)
    if n == 0:
        print("No instances found.")
        return

    n_items  = [r["n_items"]    for r in results]
    caps     = [r["capacity"]   for r in results]
    fills    = [r["fill_ratio"] for r in results]
    values   = [r["opt_value"]  for r in results if r["opt_value"] is not None]
    selected = [r["n_selected"] for r in results]
    feasible = sum(1 for r in results if r["feasible"])
    problems = [r for r in results if r["errors"]]

    print("=" * 58)
    print(f"  Instances checked : {n}")
    print(f"  Feasible          : {feasible}/{n}  ({feasible/n*100:.1f}%)")
    print("-" * 58)
    print(f"  {'Metric':<20} {'min':>8} {'mean':>8} {'max':>8}")
    print(f"  {'n_items':<20} {min(n_items):>8} {np.mean(n_items):>8.1f} {max(n_items):>8}")
    print(f"  {'capacity':<20} {min(caps):>8} {np.mean(caps):>8.1f} {max(caps):>8}")
    print(f"  {'fill_ratio':<20} {min(fills):>8.3f} {np.mean(fills):>8.3f} {max(fills):>8.3f}")
    if values:
        print(f"  {'opt_value':<20} {min(values):>8} {np.mean(values):>8.1f} {max(values):>8}")
    print(f"  {'n_selected':<20} {min(selected):>8} {np.mean(selected):>8.1f} {max(selected):>8}")
    print("=" * 58)

    if problems:
        print(f"\n  PROBLEMS FOUND in {len(problems)} instance(s):")
        for r in problems[:10]:
            print(f"  [{r['file']}]")
            for e in r["errors"]:
                tag = "WARN" if e.startswith("WARN") else "ERR "
                print(f"    [{tag}] {e}")
        if len(problems) > 10:
            print(f"  ... and {len(problems) - 10} more")
    else:
        print("\n  All checks passed.")


def load_meta(directory: Path) -> None:
    meta_path = directory / "meta.json"
    if not meta_path.exists():
        print("  (no meta.json found)\n")
        return
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    print("  meta.json:")
    for k, val in meta.items():
        print(f"    {k}: {val}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a generated Knapsack NPZ dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "directory", type=Path,
        help="Directory containing instance_*.npz files",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Check only the first N instances (default: all)",
    )
    parser.add_argument(
        "--show", type=int, default=5,
        help="Print detail rows for first N instances (default: 5)",
    )
    args = parser.parse_args()

    directory: Path = args.directory

    if not directory.exists():
        print(f"[ERROR] Directory not found: {directory.resolve()}")
        raise SystemExit(1)

    files = sorted(directory.glob("instance_*.npz"))
    if not files:
        print(f"[ERROR] No instance_*.npz files found in {directory.resolve()}")
        raise SystemExit(1)

    if args.n:
        files = files[: args.n]

    print(f"\nDataset : {directory.resolve()}")
    load_meta(directory)
    print(f"Checking {len(files)} instance(s)...\n")

    results: List[Dict] = []
    for path in files:
        try:
            results.append(check_instance(path))
        except Exception as e:
            print(f"  [ERROR] {path.name}: {e}")

    if not results:
        print("No results to summarise.")
        return

    if args.show > 0:
        print(f"  {'File':<24} {'n':>4} {'cap':>6} {'sel':>4} {'fill':>6} {'val':>8}  status")
        print("  " + "-" * 62)
        for r in results[: args.show]:
            val_str = f"{r['opt_value']:>8}" if r["opt_value"] is not None else "     n/a"
            status  = "OK" if not r["errors"] else r["errors"][0]
            print(
                f"  {r['file']:<24} {r['n_items']:>4} {r['capacity']:>6} "
                f"{r['n_selected']:>4} {r['fill_ratio']:>6.3f} {val_str}  {status}"
            )
        if len(results) > args.show:
            print(f"  ... ({len(results) - args.show} more)")
        print()

    print_summary(results)


if __name__ == "__main__":
    main()