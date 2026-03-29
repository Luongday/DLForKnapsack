"""Quick data checker for generated Knapsack NPZ datasets.

Usage:
    python check_data.py data/knapsack_ilp/train
    python check_data.py data/knapsack_ilp/test --n 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def check_instance(path: Path) -> dict:
    """Load one NPZ and run all validation checks. Returns a result dict."""
    arr = np.load(path)
    w = arr["weights"]
    v = arr["values"]
    c = int(arr["capacity"])
    sol = arr["solution"]
    opt = int(arr["dp_value"])

    total_w = int((w * sol).sum())
    total_v = int((v * sol).sum())
    n_selected = int(sol.sum())

    errors = []
    if total_w > c:
        errors.append(f"CAPACITY VIOLATED: weight={total_w} > cap={c}")
    if total_v != opt:
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
        errors.append("WARN: no items selected (tight capacity?)")

    return {
        "file": path.name,
        "n_items": len(w),
        "capacity": c,
        "n_selected": n_selected,
        "fill_ratio": total_w / c if c > 0 else 0.0,
        "opt_value": opt,
        "feasible": total_w <= c,
        "errors": errors,
    }


def print_summary(results: list[dict]) -> None:
    n = len(results)
    if n == 0:
        print("No instances found.")
        return

    n_items   = [r["n_items"]    for r in results]
    caps      = [r["capacity"]   for r in results]
    fills     = [r["fill_ratio"] for r in results]
    values    = [r["opt_value"]  for r in results]
    selected  = [r["n_selected"] for r in results]
    feasible  = sum(1 for r in results if r["feasible"])
    errors    = [r for r in results if r["errors"]]

    print("=" * 56)
    print(f"  Instances checked : {n}")
    print(f"  Feasible          : {feasible}/{n}  ({feasible/n*100:.1f}%)")
    print("-" * 56)
    print(f"  {'Metric':<18} {'min':>8} {'mean':>8} {'max':>8}")
    print(f"  {'n_items':<18} {min(n_items):>8} {np.mean(n_items):>8.1f} {max(n_items):>8}")
    print(f"  {'capacity':<18} {min(caps):>8} {np.mean(caps):>8.1f} {max(caps):>8}")
    print(f"  {'fill_ratio':<18} {min(fills):>8.3f} {np.mean(fills):>8.3f} {max(fills):>8.3f}")
    print(f"  {'opt_value':<18} {min(values):>8} {np.mean(values):>8.1f} {max(values):>8}")
    print(f"  {'n_selected':<18} {min(selected):>8} {np.mean(selected):>8.1f} {max(selected):>8}")
    print("=" * 56)

    if errors:
        print(f"\n  PROBLEMS FOUND in {len(errors)} instance(s):")
        for r in errors[:10]:
            print(f"  [{r['file']}]")
            for e in r["errors"]:
                print(f"    - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors)-10} more")
    else:
        print("\n  All checks passed.")


def load_meta(directory: Path) -> None:
    meta_path = directory / "meta.json"
    if not meta_path.exists():
        print("  (no meta.json found)\n")
        return
    with meta_path.open() as f:
        meta = json.load(f)
    print("\n  meta.json:")
    for k, v in meta.items():
        print(f"    {k}: {v}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a generated Knapsack NPZ dataset."
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
        help="Print detail for first N individual instances (default: 5)",
    )
    args = parser.parse_args()

    files = sorted(args.directory.glob("instance_*.npz"))
    if not files:
        print(f"No instance_*.npz files found in {args.directory}")
        return

    if args.n:
        files = files[: args.n]

    print(f"\nDataset: {args.directory}")
    load_meta(args.directory)
    print(f"Checking {len(files)} instances...\n")

    results = []
    for path in files:
        try:
            r = check_instance(path)
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {path.name}: {e}")

    # Print individual detail for first --show instances
    if args.show > 0:
        print(f"  {'File':<22} {'n':>4} {'cap':>6} {'sel':>4} {'fill':>6} {'val':>7}  status")
        print("  " + "-" * 58)
        for r in results[: args.show]:
            status = "OK" if not r["errors"] else " / ".join(r["errors"])
            print(
                f"  {r['file']:<22} {r['n_items']:>4} {r['capacity']:>6} "
                f"{r['n_selected']:>4} {r['fill_ratio']:>6.3f} {r['opt_value']:>7}  {status}"
            )
        if len(results) > args.show:
            print(f"  ... ({len(results) - args.show} more)")
        print()

    print_summary(results)


if __name__ == "__main__":
    main()