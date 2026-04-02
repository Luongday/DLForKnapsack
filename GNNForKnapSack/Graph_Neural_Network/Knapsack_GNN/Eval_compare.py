"""GNN vs DP comparison on Excel instances.

FIX vs original: No data leakage — graph built WITHOUT solution labels.
Uses centralized decode_utils.
"""

from typing import Tuple

import torch

try:
    from Dp import solve_knapsack_dp
    from Graph_builder import build_knapsack_graph_inference
    from decode_utils import greedy_feasible_decode
    from io_excel import load_excel_knapsack_instances
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from Dp import solve_knapsack_dp
    from Graph_builder import build_knapsack_graph_inference
    from GNNForKnapSack.decode_utils import greedy_feasible_decode
    from io_excel import load_excel_knapsack_instances


@torch.no_grad()
def evaluate_gnn_vs_dp_on_excel(
    model, excel_path: str,
) -> Tuple[int, int, int]:
    """Compare GNN with DP on Excel instances (no data leakage)."""
    device = next(model.parameters()).device
    model.eval()

    instances = load_excel_knapsack_instances(excel_path)
    matches = violations = 0

    for idx, inst in enumerate(instances, 1):
        weights, values, capacity = inst["weights"], inst["values"], inst["capacity"]

        # DP reference
        dp_solution = solve_knapsack_dp(weights, values, capacity)
        dp_value  = sum(v * s for v, s in zip(values, dp_solution))
        dp_weight = sum(w * s for w, s in zip(weights, dp_solution))

        # GNN inference (solution-free graph)
        graph  = build_knapsack_graph_inference(weights, values, capacity).to(device)
        logits = model(graph)
        probs  = torch.sigmoid(logits).cpu()
        w_t    = graph.wts.cpu()

        gnn_sel    = greedy_feasible_decode(probs, w_t, float(capacity))
        gnn_value  = float((gnn_sel * torch.tensor(values, dtype=torch.float32)).sum())
        gnn_weight = float((gnn_sel * w_t).sum())
        feasible   = gnn_weight <= capacity + 1e-6
        ratio      = gnn_value / dp_value if dp_value > 0 else 0.0

        print(f"Instance {idx}: DP val={dp_value} wt={dp_weight} | "
              f"GNN val={gnn_value:.1f} wt={gnn_weight:.1f} "
              f"feasible={feasible} ratio={ratio:.3f}")

        if abs(gnn_value - dp_value) < 1e-6:
            matches += 1
        if not feasible:
            violations += 1

    n = len(instances)
    print(f"\nSummary: matches={matches}/{n} | violations={violations}/{n}")
    return n, matches, violations