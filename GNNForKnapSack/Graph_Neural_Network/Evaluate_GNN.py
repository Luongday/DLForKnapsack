"""Per-instance GNN evaluation aligned with DP/DQN CSV schema.

Fixes vs original:
    - Removed hardcoded Windows absolute paths; now uses CLI args with sane defaults.
    - Uses load_checkpoint() from model.py so in_dim/hidden_dim are inferred
      automatically — no more manual in_dim probe.
    - Consistent with save_checkpoint() called by run_train.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from Knapsack_GNN.model import load_checkpoint
from Knapsack_GNN.Graph_builder import build_knapsack_graph


def mark(msg: str):
    print(f"[GNN-EVAL] {msg}", flush=True)


def greedy_feasible_decode(
    probs: torch.Tensor, weights: torch.Tensor, capacity: float
) -> torch.Tensor:
    """Pick items by descending probability while respecting capacity."""
    idx = torch.argsort(probs, descending=True)
    x_hat = torch.zeros_like(probs)
    total_w = 0.0
    for i in idx:
        w_i = weights[i].item()
        if total_w + w_i <= capacity:
            x_hat[i] = 1.0
            total_w += w_i
    return x_hat


def build_graph(weights: np.ndarray, values: np.ndarray, capacity: int) -> Data:
    """Replicate training-time graph (solution-free — as in real inference)."""
    dummy_solution = [0] * len(weights)
    return build_knapsack_graph(
        weights.tolist(), values.tolist(), capacity, dummy_solution
    )


def load_instance(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    arr = np.load(npz_path, allow_pickle=True)

    def pick(keys):
        for k in keys:
            if k in arr.files:
                return arr[k]
        return None

    W = pick(["weights", "w", "W"])
    V = pick(["values", "v", "V"])
    C = pick(["capacity", "cap", "C"])
    if W is None or V is None or C is None:
        raise KeyError(
            f"Missing weights/values/capacity in {npz_path}, found keys={arr.files}"
        )
    W = np.asarray(W).reshape(-1)
    V = np.asarray(V).reshape(-1)
    C = int(np.asarray(C).reshape(()))
    if W.shape != V.shape:
        raise ValueError(
            f"weights and values shape mismatch in {npz_path}: {W.shape} vs {V.shape}"
        )
    return W.astype(np.float32), V.astype(np.float32), C


def _default_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "dataset" / "knapsack01_medium"


def _default_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "GNN" / "gnn.pt"


def _default_out_csv() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "GNN" / "gnn_eval_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained KnapsackGNN on NPZ instances."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=_default_dataset_dir(),
        help="Directory containing instance_*.npz files",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=_default_model_path(),
        help="Path to checkpoint saved by run_train.py",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=_default_out_csv(),
        help="Output CSV path for per-instance results",
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    files = sorted(args.dataset_dir.glob("instance_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No instance_*.npz files found in {args.dataset_dir}"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    mark(f"Dataset dir:  {args.dataset_dir}")
    mark(f"Model path:   {args.model_path}")
    mark(f"Output CSV:   {args.out_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load_checkpoint infers in_dim / hidden_dim from the saved checkpoint dict.
    model = load_checkpoint(args.model_path, device=device)

    results: List[dict] = []
    for idx, path in enumerate(files):
        W, V, C = load_instance(path)
        graph = build_graph(W, V, C).to(device)

        t0 = time.perf_counter()
        logits = model(graph)
        probs = torch.sigmoid(logits).detach().cpu()
        x_hat = greedy_feasible_decode(probs, graph.wts.cpu(), float(C))
        infer_ms = (time.perf_counter() - t0) * 1000.0

        total_weight = float((x_hat * graph.wts.cpu()).sum())
        total_value = float((x_hat * graph.vals.cpu()).sum())
        feasible = 1 if total_weight <= float(C) + 1e-6 else 0
        selected_indices = [int(i) for i, val in enumerate(x_hat.tolist()) if val == 1.0]

        results.append(
            {
                "instance_file": path.name,
                "n_items": int(len(W)),
                "capacity": float(C),
                "total_weight": total_weight,
                "total_value": total_value,
                "feasible": feasible,
                "inference_time_ms": infer_ms,
                "selected_items": json.dumps(selected_indices),
            }
        )
        if (idx + 1) % 100 == 0:
            mark(f"Evaluated {idx+1}/{len(files)} instances")

    avg_value = float(np.mean([r["total_value"] for r in results]))
    feasibility_rate = float(np.mean([r["feasible"] for r in results]))
    avg_time_ms = float(np.mean([r["inference_time_ms"] for r in results]))
    mark(
        f"avg_value={avg_value:.2f} "
        f"feasibility_rate={feasibility_rate:.3f} "
        f"avg_inference_time_ms={avg_time_ms:.2f}"
    )

    fieldnames = [
        "instance_file", "n_items", "capacity",
        "total_weight", "total_value", "feasible",
        "inference_time_ms", "selected_items",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    mark(f"Wrote per-instance results to {args.out_csv}")


if __name__ == "__main__":
    main()