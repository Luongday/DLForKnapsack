"""GNN vs DP comparison on Excel instances.

Bug fixed vs original:
    The original passed dp_solution as a graph feature during GNN inference,
    creating data leakage — the model effectively received the answer as input.
    This version builds the graph from weights/values/capacity only (no solution),
    matching the inference-time setup where the solution is unknown.

Additional improvement:
    Uses greedy_feasible_decode instead of a raw threshold so the GNN output is
    always capacity-feasible, giving a fair comparison against DP.
"""

from typing import Tuple

import torch

from .Dp import solve_knapsack_dp
from .Graph_builder import build_knapsack_graph
from .io_excel import load_excel_knapsack_instances


def _greedy_feasible_decode(
    probs: torch.Tensor,
    weights: torch.Tensor,
    capacity: float,
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


def _build_inference_graph(
    weights, values, capacity, device: torch.device
):
    """Build a graph that contains NO solution labels — as in real inference."""
    dummy_solution = [0] * len(weights)   # placeholder; not used by the model
    graph = build_knapsack_graph(weights, values, capacity, dummy_solution)
    return graph.to(device)


@torch.no_grad()
def evaluate_gnn_vs_dp_on_excel(
    model,
    excel_path: str,
) -> Tuple[int, int, int]:
    """Compare GNN predictions with DP on the Excel instances.

    The GNN graph is built WITHOUT the DP solution to avoid data leakage.
    Feasibility is enforced via greedy decode.

    Returns:
        (num_instances, matches_value, violations_capacity)
    """
    device = next(model.parameters()).device
    model.eval()

    instances = load_excel_knapsack_instances(excel_path)

    matches = 0
    violations = 0

    for idx, inst in enumerate(instances, 1):
        weights = inst["weights"]
        values = inst["values"]
        capacity = inst["capacity"]

        # --- DP reference solution ---
        dp_solution = solve_knapsack_dp(weights, values, capacity)
        dp_value = sum(v * s for v, s in zip(values, dp_solution))
        dp_weight = sum(w * s for w, s in zip(weights, dp_solution))

        # --- GNN inference (solution-free graph) ---
        graph = _build_inference_graph(weights, values, capacity, device)
        logits = model(graph)
        probs = torch.sigmoid(logits).cpu()
        w_tensor = graph.wts.cpu()

        gnn_sel = _greedy_feasible_decode(probs, w_tensor, float(capacity))

        gnn_value = float((gnn_sel * torch.tensor(values, dtype=torch.float32)).sum())
        gnn_weight = float((gnn_sel * w_tensor).sum())
        is_feasible = gnn_weight <= capacity + 1e-6

        value_ratio = gnn_value / dp_value if dp_value > 0 else 0.0

        print(f"Instance {idx}:")
        print(f"  DP:  value={dp_value}, weight={dp_weight}")
        print(f"  GNN: value={gnn_value:.1f}, weight={gnn_weight:.1f}, feasible={is_feasible}, ratio={value_ratio:.3f}")

        if abs(gnn_value - dp_value) < 1e-6:
            matches += 1
        if not is_feasible:
            violations += 1

    n = len(instances)
    print("\nSummary:")
    print(f"  Instances matching DP value:     {matches}/{n}")
    print(f"  Instances violating capacity:    {violations}/{n}")

    return n, matches, violations