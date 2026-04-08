"""DQN evaluation — CSV schema aligned with DP/GNN/Greedy/GA pipeline.

Evaluates trained DQN on per-instance NPZ files, outputs CSV compatible
with Merge_results.py for full 5-solver comparison.

Usage:
    python Evaluate_DQN.py --dataset_dir data/knapsack_ilp/test --model_path results/DQN/dqn.pt

    # Then merge all:
    python Merge_results.py --dqn_csv results/DQN/dqn_eval_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dqn_env import KnapsackEnv
from dqn_model import QNetwork
from GNNForKnapSack.instance_loader import load_instance, list_instances


def mark(msg: str):
    print(f"[DQN-EVAL] {msg}", flush=True)


def greedy_action(q_values: np.ndarray, valid_mask: np.ndarray) -> int:
    """Select action with highest Q-value among valid actions."""
    q = q_values.copy()
    q[valid_mask < 0.5] = -1e9
    return int(np.argmax(q))


def run_dqn_on_instance(
    model: QNetwork,
    device: torch.device,
    weights: np.ndarray,
    values: np.ndarray,
    capacity: int,
) -> dict:
    """Run DQN inference on one instance. Returns solution dict."""
    env = KnapsackEnv(weights, values, capacity, eps=1e-8)
    s = env.reset()
    done = False

    t0 = time.perf_counter()
    while not done:
        valid_mask = env.valid_actions_mask()
        with torch.no_grad():
            q = model(torch.from_numpy(s).unsqueeze(0).to(device)).cpu().numpy()[0]
        a = greedy_action(q, valid_mask)
        out = env.step(a)
        s = out.next_state
        done = out.done
    infer_ms = (time.perf_counter() - t0) * 1000.0

    total_weight = env.compute_solution_weight()
    total_value  = env.compute_solution_value()
    feasible     = 1 if total_weight <= float(capacity) + 1e-6 else 0
    selected_idx = [int(i) for i, v in enumerate(env.selection.tolist()) if v == 1]

    return {
        "total_weight":      float(total_weight),
        "total_value":       float(total_value),
        "feasible":          feasible,
        "inference_time_ms": round(infer_ms, 4),
        "selected_items":    selected_idx,
    }


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

def _default_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "GNNForKnapSack" / "data" / "knapsack_ilp" / "test"

def _default_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "DQN" / "dqn.pt"

def _default_out_csv() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "DQN" / "dqn_eval_results.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained DQN on NPZ Knapsack instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=Path, default=_default_dataset_dir())
    parser.add_argument("--model_path",  type=Path, default=_default_model_path())
    parser.add_argument("--out_csv",     type=Path, default=_default_out_csv())
    parser.add_argument("--device",      type=str,  default="cpu")
    parser.add_argument("--n",           type=int,  default=None,
                        help="Limit to first N instances")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    mark(f"Loading model: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    state_dim  = int(ckpt["state_dim"])
    hidden_dim = int(ckpt.get("hidden_dim", 128))

    device = torch.device(args.device)
    model = QNetwork(state_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mark(f"Model loaded: state_dim={state_dim} hidden={hidden_dim}")

    # Load instances
    files = list_instances(args.dataset_dir, limit=args.n)
    mark(f"Found {len(files)} instances in {args.dataset_dir}")

    # Evaluate
    results: List[dict] = []
    total_start = time.perf_counter()

    for idx, path in enumerate(files):
        W, V, C = load_instance(path)

        sol = run_dqn_on_instance(model, device, W, V, int(C))

        # SAME CSV SCHEMA as DP/GNN/Greedy/GA
        results.append({
            "instance_file":     path.name,
            "n_items":           len(W),
            "capacity":          float(C),
            "total_weight":      sol["total_weight"],
            "total_value":       sol["total_value"],
            "feasible":          sol["feasible"],
            "inference_time_ms": sol["inference_time_ms"],
            "selected_items":    json.dumps(sol["selected_items"]),
        })

        if (idx + 1) % 50 == 0:
            mark(f"[{idx+1}/{len(files)}] val={sol['total_value']:.1f} "
                 f"wt={sol['total_weight']:.1f}/{C} "
                 f"feasible={bool(sol['feasible'])}")

    # Write CSV (IDENTICAL schema to DP/GNN/Greedy/GA)
    fieldnames = [
        "instance_file", "n_items", "capacity",
        "total_weight", "total_value",
        "feasible", "inference_time_ms", "selected_items",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    total_time = time.perf_counter() - total_start
    if results:
        avg_value = float(np.mean([r["total_value"]       for r in results]))
        avg_time  = float(np.mean([r["inference_time_ms"] for r in results]))
        feas_rate = float(np.mean([r["feasible"]          for r in results]))
        mark(f"Done: {len(results)} instances in {total_time:.1f}s")
        mark(f"Avg value={avg_value:.2f} | Avg time={avg_time:.3f}ms | "
             f"Feasible={feas_rate:.3f}")
    mark(f"Results → {args.out_csv}")


if __name__ == "__main__":
    main()