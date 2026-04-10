"""S2V-DQN evaluation — CSV schema aligned with all other solvers."""

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

from GNNForKnapSack.instance_loader import load_instance, list_instances
from s2v_env import GraphKnapsackEnv
from s2v_model import load_s2v_checkpoint


def mark(msg: str):
    print(f"[S2V-EVAL] {msg}", flush=True)


@torch.no_grad()
def run_s2v_on_instance(
    model,
    device: torch.device,
    weights: np.ndarray,
    values:  np.ndarray,
    capacity: int,
    k: int = 16,
) -> dict:
    """Run S2V-DQN on one instance with greedy policy."""
    env = GraphKnapsackEnv(weights, values, capacity, k=k)
    s = env.reset()
    done = False

    t0 = time.perf_counter()
    while not done:
        mask = env.valid_actions_mask()
        if mask.sum() == 0:
            break
        s_dev = s.to(device)
        q = model(s_dev).cpu().numpy()
        q[mask < 0.5] = -1e9
        action = int(np.argmax(q))
        out = env.step(action)
        s = out.next_state
        done = out.done
    infer_ms = (time.perf_counter() - t0) * 1000.0

    total_weight = env.compute_solution_weight()
    total_value  = env.compute_solution_value()
    feasible     = 1 if total_weight <= float(capacity) + 1e-6 else 0
    selected_idx = [int(i) for i, v in enumerate(env.get_selection().tolist()) if v == 1]

    return {
        "total_weight":      float(total_weight),
        "total_value":       float(total_value),
        "feasible":          feasible,
        "inference_time_ms": round(infer_ms, 4),
        "selected_items":    selected_idx,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate S2V-DQN on Knapsack instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--model_path",  type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "S2V_DQN" / "s2v_dqn_best.pt")
    parser.add_argument("--out_csv",     type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "S2V_DQN" / "s2v_dqn_eval_results.csv")
    parser.add_argument("--device",      type=str, default="cpu")
    parser.add_argument("--k",           type=int, default=16)
    parser.add_argument("--n",           type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    mark(f"Loading model: {args.model_path}")
    model = load_s2v_checkpoint(args.model_path, device)

    files = list_instances(args.dataset_dir, limit=args.n)
    mark(f"Found {len(files)} instances in {args.dataset_dir}")

    results = []
    total_start = time.perf_counter()

    for idx, path in enumerate(files):
        W, V, C = load_instance(path)
        sol = run_s2v_on_instance(model, device, W, V, int(C), k=args.k)

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
                 f"feasible={bool(sol['feasible'])}")

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

    total_time = time.perf_counter() - total_start
    if results:
        avg_val  = float(np.mean([r["total_value"]       for r in results]))
        avg_time = float(np.mean([r["inference_time_ms"] for r in results]))
        feas     = float(np.mean([r["feasible"]          for r in results]))
        mark(f"Done: {len(results)} in {total_time:.1f}s")
        mark(f"avg_value={avg_val:.2f} | avg_time={avg_time:.3f}ms | feasible={feas:.3f}")
    mark(f"Results → {args.out_csv}")


if __name__ == "__main__":
    main()