"""Pisinger Hard Knapsack Instance Generator.

Wraps Genhard.c (Pisinger 2005) to generate benchmark instances of varying
difficulty types, solves them with DP, and saves as NPZ files compatible
with the full evaluation pipeline.

Instance types (from Pisinger paper):
    1  = Uncorrelated              (easy for greedy)
    2  = Weakly correlated         (moderate)
    3  = Strongly correlated       (HARD for greedy)
    4  = Inverse strongly corr.    (hard)
    5  = Almost strongly corr.     (hard)
    6  = Subset-sum                (VERY HARD)
    7  = Even-odd subset-sum       (very hard)
    8  = Even-odd knapsack         (hard)
    9  = Uncorr, similar weights   (moderate)
    11 = Uncorr span(2,10)
    12 = Weak corr span(2,10)
    13 = Strong corr span(2,10)
    14 = mstr(3R/10,2R/10,6)
    15 = pceil(3)
    16 = circle(2/3)

Usage:
    # Generate one type
    python Generate_Hard.py --type 3 --n_items 100 --num_instances 200

    # Generate multiple types for comparison
    python Generate_Hard.py --types 1 2 3 5 6 --n_items 100 --num_instances 100

    # Large scale
    python Generate_Hard.py --type 3 --n_items 500 --num_instances 200 --range 10000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TYPE_NAMES = {
    1:  "uncorrelated",
    2:  "weakly_correlated",
    3:  "strongly_correlated",
    4:  "inverse_strongly_corr",
    5:  "almost_strongly_corr",
    6:  "subset_sum",
    7:  "even_odd_subset_sum",
    8:  "even_odd_knapsack",
    9:  "uncorr_similar_weights",
    11: "uncorr_span",
    12: "weak_corr_span",
    13: "strong_corr_span",
    14: "mstr",
    15: "pceil",
    16: "circle",
}

# Default Genhard binary path (relative to this script)
_HERE = Path(__file__).resolve().parent
DEFAULT_GENHARD_BIN = _HERE / "genhard"


# ---------------------------------------------------------------------------
# Genhard C compilation
# ---------------------------------------------------------------------------

def compile_genhard(
    source_path: Path = None,
    output_path: Path = None,
) -> Path:
    """Compile Genhard.c if binary doesn't exist.

    Automatically fixes known issues:
    - void main → int main
    - Missing error() function
    """
    if output_path is None:
        output_path = DEFAULT_GENHARD_BIN
    if source_path is None:
        # Try multiple locations
        candidates = [
            _HERE / "Genhard.c",
        ]
        source_path = next((p for p in candidates if p.exists()), None)
        if source_path is None:
            raise FileNotFoundError(
                f"Genhard.c not found. Searched: {[str(c) for c in candidates]}"
            )

    if output_path.exists():
        return output_path

    print(f"[GENHARD] Compiling {source_path} → {output_path}")

    # Read and fix source
    src = source_path.read_text()

    # Fix void main → int main
    if "void main" in src:
        src = src.replace("void main", "int main")

    # Add error() function if missing
    if "void error(" not in src:
        src = src.replace(
            '#include <math.h>',
            '#include <math.h>\n\n'
            'void error(const char *msg) {\n'
            '  fprintf(stderr, "ERROR: %s\\n", msg);\n'
            '  exit(1);\n'
            '}\n'
        )

    # Write fixed source
    fixed_path = output_path.parent / "Genhard.c"
    fixed_path.write_text(src)

    # Compile
    result = subprocess.run(
        ["gcc", "-O2", "-o", str(output_path), str(fixed_path), "-lm"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed:\n{result.stderr}")

    print(f"[GENHARD] Compiled successfully: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Parse Genhard output
# ---------------------------------------------------------------------------

def parse_genhard_output(filepath: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    with open(filepath) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    n = int(lines[0])
    values  = np.zeros(n, dtype=np.int32)
    weights = np.zeros(n, dtype=np.int32)

    for i in range(n):
        parts = lines[1 + i].split()
        # Format: index profit weight
        idx = int(parts[0])
        values[idx]  = int(parts[1])
        weights[idx] = int(parts[2])

    capacity = int(lines[1 + n])

    return weights, values, capacity


# ---------------------------------------------------------------------------
# Call Genhard binary
# ---------------------------------------------------------------------------

def call_genhard(
    genhard_bin: Path,
    n_items: int,
    coeff_range: int,
    instance_type: int,
    instance_id: int,
    series_size: int,
    work_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, int]:
    result = subprocess.run(
        [
            str(genhard_bin),
            str(n_items),
            str(coeff_range),
            str(instance_type),
            str(instance_id),
            str(series_size),
        ],
        capture_output=True, text=True,
        cwd=str(work_dir),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Genhard failed for type={instance_type} i={instance_id}: "
            f"{result.stderr}"
        )

    test_in = work_dir / "test.in"
    if not test_in.exists():
        raise FileNotFoundError(
            f"Genhard did not create test.in in {work_dir}"
        )

    return parse_genhard_output(test_in)


# ---------------------------------------------------------------------------
# DP solver (reuse from project)
# ---------------------------------------------------------------------------

def solve_knapsack_dp(
    weights: np.ndarray,
    values:  np.ndarray,
    capacity: int,
    max_capacity: int = 500_000,
) -> Optional[Tuple[np.ndarray, int]]:
    """Solve 0/1 Knapsack with DP. Returns (solution, optimal_value) or None if too large."""
    n = len(weights)
    if capacity > max_capacity:
        print(f"  [WARN] capacity={capacity} > max={max_capacity}, skipping DP solve")
        return None

    dp = np.zeros(capacity + 1, dtype=np.int64)
    choice = np.zeros((n, capacity + 1), dtype=np.bool_)

    for i in range(n):
        w_i = int(weights[i])
        v_i = int(values[i])
        for w in range(capacity, w_i - 1, -1):
            take = dp[w - w_i] + v_i
            if take > dp[w]:
                dp[w] = take
                choice[i, w] = True

    # Backtrack
    solution = np.zeros(n, dtype=np.int8)
    w = capacity
    for i in range(n - 1, -1, -1):
        if choice[i, w]:
            solution[i] = 1
            w -= int(weights[i])

    opt_value = int(dp[capacity])
    return solution, opt_value


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_hard_dataset(
    out_dir:        Path,
    instance_type:  int,
    n_items:        int,
    num_instances:  int,
    coeff_range:    int   = 1000,
    series_size:    int   = 1000,
    max_dp_capacity: int  = 500_000,
    genhard_bin:    Path   = None,
    verbose:        bool   = True,
) -> Dict:
    """Generate a dataset of Pisinger hard instances.

    Each instance is:
    1. Generated by Genhard.c
    2. Solved optimally by DP
    3. Saved as NPZ (same schema as other generators)

    Args:
        out_dir:         Output directory for NPZ files + meta.json.
        instance_type:   Pisinger type (1-16).
        n_items:         Number of items per instance.
        num_instances:   How many instances to generate.
        coeff_range:     Range of coefficients (r parameter).
        series_size:     Series size for capacity calculation (S parameter).
        max_dp_capacity: Skip DP if capacity exceeds this.
        genhard_bin:     Path to compiled genhard binary.

    Returns:
        Dict with generation statistics.
    """
    if genhard_bin is None:
        genhard_bin = compile_genhard()
    if not genhard_bin.exists():
        genhard_bin = compile_genhard()

    out_dir.mkdir(parents=True, exist_ok=True)
    type_name = TYPE_NAMES.get(instance_type, f"type_{instance_type}")

    if verbose:
        print(f"\n[GENHARD] Generating {num_instances} instances")
        print(f"  Type: {instance_type} ({type_name})")
        print(f"  n_items: {n_items}, range: {coeff_range}, series: {series_size}")
        print(f"  Output: {out_dir}")

    stats = {
        "generated": 0,
        "dp_solved": 0,
        "dp_skipped": 0,
        "dp_values": [],
        "capacities": [],
    }

    start_time = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for idx in range(num_instances):
            # Instance ID varies from 1 to series_size to get different capacities
            # Spread across the range for diverse difficulty
            instance_id = max(1, int((idx + 1) / num_instances * series_size))

            try:
                W, V, C = call_genhard(
                    genhard_bin, n_items, coeff_range,
                    instance_type, instance_id, series_size,
                    tmpdir,
                )
            except (RuntimeError, FileNotFoundError) as e:
                if verbose:
                    print(f"  [SKIP] instance {idx}: {e}")
                continue

            # Solve with DP
            dp_result = solve_knapsack_dp(W, V, C, max_dp_capacity)

            if dp_result is not None:
                solution, opt_value = dp_result
                stats["dp_solved"] += 1
            else:
                # Save without solution (can be solved later or by other methods)
                solution = np.zeros(len(W), dtype=np.int8)
                opt_value = 0
                stats["dp_skipped"] += 1

            # Save NPZ (compatible with pipeline)
            np.savez_compressed(
                out_dir / f"instance_{idx:04d}.npz",
                weights=W,
                values=V,
                capacity=np.int32(C),
                solution=solution,
                dp_value=np.int32(opt_value),
            )

            stats["generated"] += 1
            stats["dp_values"].append(opt_value)
            stats["capacities"].append(C)

            if verbose and ((idx + 1) % 50 == 0 or idx == 0):
                elapsed = time.perf_counter() - start_time
                rate = (idx + 1) / elapsed
                eta = (num_instances - idx - 1) / rate if rate > 0 else 0
                print(f"  [{idx+1}/{num_instances}] elapsed={elapsed:.1f}s "
                      f"ETA={eta:.1f}s  cap={C}  opt={opt_value}")

    total_time = time.perf_counter() - start_time

    # Write meta.json
    meta = {
        "generator":       "Pisinger/Genhard.c",
        "paper":           "Where are the hard knapsack problems (2005)",
        "instance_type":   instance_type,
        "type_name":       type_name,
        "n_items":         n_items,
        "coeff_range":     coeff_range,
        "series_size":     series_size,
        "num_instances":   stats["generated"],
        "dp_solved":       stats["dp_solved"],
        "dp_skipped":      stats["dp_skipped"],
        "avg_capacity":    float(np.mean(stats["capacities"])) if stats["capacities"] else 0,
        "avg_dp_value":    float(np.mean(stats["dp_values"])) if stats["dp_values"] else 0,
        "total_time_sec":  round(total_time, 3),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if verbose:
        print(f"\n  Done: {stats['generated']} instances in {total_time:.1f}s")
        print(f"  DP solved: {stats['dp_solved']}, skipped: {stats['dp_skipped']}")
        if stats["capacities"]:
            print(f"  Capacity range: {min(stats['capacities'])} - {max(stats['capacities'])}")

    return stats


def generate_multi_type_dataset(
    base_dir:       Path,
    types:          List[int],
    n_items:        int,
    num_per_type:   int,
    coeff_range:    int  = 1000,
    **kwargs,
) -> Dict[int, Dict]:
    """Generate datasets for multiple Pisinger types.

    Creates subdirectories: base_dir/type_01_uncorrelated/, etc.

    Returns:
        Dict mapping type → generation stats.
    """
    all_stats = {}
    for t in types:
        type_name = TYPE_NAMES.get(t, f"type_{t}")
        out_dir = base_dir / f"type_{t:02d}_{type_name}"
        stats = generate_hard_dataset(
            out_dir=out_dir,
            instance_type=t,
            n_items=n_items,
            num_instances=num_per_type,
            coeff_range=coeff_range,
            **kwargs,
        )
        all_stats[t] = stats

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    for t, s in all_stats.items():
        name = TYPE_NAMES.get(t, f"type_{t}")
        print(f"  Type {t:>2} ({name:>25}): {s['generated']:>4} instances, "
              f"DP solved: {s['dp_solved']}")

    return all_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Pisinger hard Knapsack instances using Genhard.c",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--type", type=int, default=None,
                        help="Single instance type (1-16)")
    parser.add_argument("--types", type=int, nargs="+", default=None,
                        help="Multiple instance types to generate")
    parser.add_argument("--n_items", type=int, default=100,
                        help="Number of items per instance")
    parser.add_argument("--num_instances", type=int, default=200,
                        help="Instances per type")
    parser.add_argument("--range", type=int, default=1000,
                        help="Coefficient range (r parameter)")
    parser.add_argument("--series", type=int, default=1000,
                        help="Series size (S parameter)")
    parser.add_argument("--max_dp_capacity", type=int, default=500_000,
                        help="Skip DP solve if capacity exceeds this")
    parser.add_argument("--out_dir", type=Path,
                        default=_HERE / "data" / "pisinger",
                        help="Output base directory")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    # Determine types to generate
    if args.types:
        types = args.types
    elif args.type:
        types = [args.type]
    else:
        # Default: key difficulty types for comparison
        types = [1, 2, 3, 5, 6]
        print("[GENHARD] No type specified, generating key types: "
              f"{types}")

    if len(types) == 1:
        t = types[0]
        type_name = TYPE_NAMES.get(t, f"type_{t}")
        out_dir = args.out_dir / f"type_{t:02d}_{type_name}"
        generate_hard_dataset(
            out_dir=out_dir,
            instance_type=t,
            n_items=args.n_items,
            num_instances=args.num_instances,
            coeff_range=args.range,
            series_size=args.series,
            max_dp_capacity=args.max_dp_capacity,
            verbose=verbose,
        )
    else:
        generate_multi_type_dataset(
            base_dir=args.out_dir,
            types=types,
            n_items=args.n_items,
            num_per_type=args.num_instances,
            coeff_range=args.range,
            series_size=args.series,
            max_dp_capacity=args.max_dp_capacity,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()