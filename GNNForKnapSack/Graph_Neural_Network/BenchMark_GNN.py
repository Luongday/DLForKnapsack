"""GNN inference benchmark — timing and quality metrics.

Improvements vs original:
    - Uses centralized decode_utils (no duplicate greedy decode)
    - Cleaner output format
"""

import csv
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch

try:
    from decode_utils import greedy_feasible_decode
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from GNNForKnapSack.decode_utils import greedy_feasible_decode


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _ensure_attrs(batch):
    for attr in ("wts", "vals", "cap"):
        if not hasattr(batch, attr):
            raise AttributeError(f"Batch missing '{attr}' for benchmarking.")


@torch.no_grad()
def run_gnn_benchmark(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    out_dir: str,
    n_instances: int = 100,
    seed: int = 2025,
):
    """Run GNN inference benchmark on first n_instances from loader."""
    _set_seed(seed)
    model.eval()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    records: List[Dict[str, Any]] = []
    total_start = time.perf_counter()
    seen = 0
    checked_attrs = False

    for batch in loader:
        if not checked_attrs:
            _ensure_attrs(batch)
            checked_attrs = True
        batch = batch.to(device)
        batch_vec = batch.batch if hasattr(batch, "batch") else \
            torch.zeros(batch.num_nodes, dtype=torch.long, device=device)
        num_graphs = int(batch_vec.max().item()) + 1

        fwd_start = time.perf_counter()
        logits = model(batch)
        probs = torch.sigmoid(logits)
        fwd_time = time.perf_counter() - fwd_start

        for g in range(num_graphs):
            if seen >= n_instances:
                break
            mask = batch_vec == g
            if mask.sum() == 0:
                continue

            dec_start = time.perf_counter()
            p_g = probs[mask].detach().cpu()
            w_g = batch.wts[mask].detach().cpu()
            v_g = batch.vals[mask].detach().cpu()
            cap_g = batch.cap[g].item() if batch.cap.dim() > 0 else float(batch.cap.item())

            # Centralized decode
            x_hat = greedy_feasible_decode(p_g, w_g, cap_g)
            dec_time = time.perf_counter() - dec_start

            gnn_weight = float((x_hat * w_g).sum().item())
            gnn_value  = float((x_hat * v_g).sum().item())
            feasible   = gnn_weight <= cap_g + 1e-6
            fill_ratio = gnn_weight / cap_g if cap_g > 0 else 0.0
            selected_k = int(x_hat.sum().item())
            per_graph_time = (fwd_time / num_graphs) + dec_time

            records.append({
                "case_id": seen + 1,
                "capacity": float(cap_g),
                "gnn_value": gnn_value,
                "gnn_weight": gnn_weight,
                "feasible": feasible,
                "fill_ratio": fill_ratio,
                "selected_k": selected_k,
                "solve_time_sec": per_graph_time,
            })
            seen += 1
        if seen >= n_instances:
            break

    total_time = time.perf_counter() - total_start
    if seen == 0:
        raise RuntimeError("No instances processed.")

    feasible_rate = sum(1 for r in records if r["feasible"]) / seen
    gnn_values  = np.array([r["gnn_value"] for r in records])
    fill_ratios = np.array([r["fill_ratio"] for r in records])
    solve_times = np.array([r["solve_time_sec"] for r in records])

    summary = {
        "run_id": run_id, "timestamp": timestamp,
        "device": str(device), "n_instances": seen, "seed": seed,
        "metrics": {
            "feasible_rate": feasible_rate,
            "avg_gnn_value": float(np.mean(gnn_values)),
            "std_gnn_value": float(np.std(gnn_values)),
            "avg_fill_ratio": float(np.mean(fill_ratios)),
            "avg_solve_time_sec": float(np.mean(solve_times)),
            "p95_solve_time_sec": float(np.percentile(solve_times, 95)),
            "total_time_sec": total_time,
        },
    }

    # Save
    details_path = out_path / "gnn_benchmark_details_latest.csv"
    summary_path = out_path / "gnn_benchmark_summary_latest.json"

    fieldnames = ["run_id", "timestamp", "case_id", "capacity",
                  "gnn_value", "gnn_weight", "feasible", "fill_ratio",
                  "selected_k", "solve_time_sec"]
    with details_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({"run_id": run_id, "timestamp": timestamp, **row})

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    m = summary["metrics"]
    print(f"GNN benchmark | N={seen} | feasible={m['feasible_rate']:.3f} | "
          f"avg_value={m['avg_gnn_value']:.2f} | "
          f"avg_time={m['avg_solve_time_sec']:.4f}s")

    return {"details": str(details_path), "summary": str(summary_path)}