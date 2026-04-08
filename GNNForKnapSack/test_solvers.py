"""Unit tests for Knapsack solvers and utilities.

Run: python -m pytest test_solvers.py -v
  or: python test_solvers.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Known test instances
# ---------------------------------------------------------------------------

# Simple 3-item instance: optimal = items 1+2 (value=220, weight=50)
SIMPLE_W = [10, 20, 30]
SIMPLE_V = [60, 100, 120]
SIMPLE_C = 50
SIMPLE_OPT = 220  # items 1+2: 100+120=220, weight=20+30=50

# Trivial: all items fit
ALL_FIT_W = [5, 5, 5]
ALL_FIT_V = [10, 20, 30]
ALL_FIT_C = 100
ALL_FIT_OPT = 60

# Tight: only 1 item fits
TIGHT_W = [100, 200, 300]
TIGHT_V = [50, 100, 150]
TIGHT_C = 100
TIGHT_OPT = 50  # only item 0


# ---------------------------------------------------------------------------
# DP Solver
# ---------------------------------------------------------------------------

class TestDP:
    def test_simple(self):
        from Dp import solve_knapsack_dp
        sol = solve_knapsack_dp(SIMPLE_W, SIMPLE_V, SIMPLE_C)
        value = sum(v * s for v, s in zip(SIMPLE_V, sol))
        assert value == SIMPLE_OPT, f"Expected {SIMPLE_OPT}, got {value}"

    def test_all_fit(self):
        from Dp import solve_knapsack_dp
        sol = solve_knapsack_dp(ALL_FIT_W, ALL_FIT_V, ALL_FIT_C)
        value = sum(v * s for v, s in zip(ALL_FIT_V, sol))
        assert value == ALL_FIT_OPT

    def test_tight(self):
        from Dp import solve_knapsack_dp
        sol = solve_knapsack_dp(TIGHT_W, TIGHT_V, TIGHT_C)
        value = sum(v * s for v, s in zip(TIGHT_V, sol))
        assert value == TIGHT_OPT

    def test_empty(self):
        from Dp import solve_knapsack_dp
        sol = solve_knapsack_dp([], [], 100)
        assert sol == []

    def test_feasibility(self):
        from Dp import solve_knapsack_dp
        sol = solve_knapsack_dp(SIMPLE_W, SIMPLE_V, SIMPLE_C)
        weight = sum(w * s for w, s in zip(SIMPLE_W, sol))
        assert weight <= SIMPLE_C, f"Weight {weight} exceeds capacity {SIMPLE_C}"

    def test_both_implementations_agree(self):
        from Dp import solve_knapsack_dp_python, solve_knapsack_dp_np
        sol1 = solve_knapsack_dp_python(SIMPLE_W, SIMPLE_V, SIMPLE_C)
        sol2 = solve_knapsack_dp_np(SIMPLE_W, SIMPLE_V, SIMPLE_C)
        v1 = sum(v * s for v, s in zip(SIMPLE_V, sol1))
        v2 = sum(v * s for v, s in zip(SIMPLE_V, sol2))
        assert v1 == v2, f"Python={v1} vs NumPy={v2}"


# ---------------------------------------------------------------------------
# Greedy Solver
# ---------------------------------------------------------------------------

class TestGreedy:
    def test_feasibility(self):
        from greedy_baseline_eval import solve_knapsack_greedy
        W = np.array(SIMPLE_W, dtype=np.int32)
        V = np.array(SIMPLE_V, dtype=np.int32)
        selected = solve_knapsack_greedy(W, V, SIMPLE_C)
        weight = sum(W[i] for i in selected)
        assert weight <= SIMPLE_C

    def test_all_fit(self):
        from greedy_baseline_eval import solve_knapsack_greedy
        W = np.array(ALL_FIT_W, dtype=np.int32)
        V = np.array(ALL_FIT_V, dtype=np.int32)
        selected = solve_knapsack_greedy(W, V, ALL_FIT_C)
        value = sum(V[i] for i in selected)
        assert value == ALL_FIT_OPT

    def test_empty(self):
        from greedy_baseline_eval import solve_knapsack_greedy
        selected = solve_knapsack_greedy(np.array([], dtype=np.int32),
                                          np.array([], dtype=np.int32), 100)
        assert selected == []

    def test_returns_sorted(self):
        from greedy_baseline_eval import solve_knapsack_greedy
        W = np.array([10, 20, 30, 5, 15], dtype=np.int32)
        V = np.array([60, 100, 120, 40, 80], dtype=np.int32)
        selected = solve_knapsack_greedy(W, V, 50)
        assert selected == sorted(selected)


# ---------------------------------------------------------------------------
# GA Solver
# ---------------------------------------------------------------------------

class TestGA:
    def test_feasibility(self):
        from ga_baseline_eval import solve_knapsack_ga
        W = np.array(SIMPLE_W, dtype=np.int32)
        V = np.array(SIMPLE_V, dtype=np.int32)
        selected = solve_knapsack_ga(W, V, SIMPLE_C, population_size=30,
                                      max_generations=100, seed=42)
        weight = sum(W[i] for i in selected)
        assert weight <= SIMPLE_C

    def test_finds_optimal_simple(self):
        from ga_baseline_eval import solve_knapsack_ga
        W = np.array(SIMPLE_W, dtype=np.int32)
        V = np.array(SIMPLE_V, dtype=np.int32)
        selected = solve_knapsack_ga(W, V, SIMPLE_C, population_size=50,
                                      max_generations=200, seed=42)
        value = sum(V[i] for i in selected)
        assert value == SIMPLE_OPT, f"Expected {SIMPLE_OPT}, got {value}"

    def test_reproducible(self):
        from ga_baseline_eval import solve_knapsack_ga
        W = np.array(SIMPLE_W, dtype=np.int32)
        V = np.array(SIMPLE_V, dtype=np.int32)
        s1 = solve_knapsack_ga(W, V, SIMPLE_C, seed=42)
        s2 = solve_knapsack_ga(W, V, SIMPLE_C, seed=42)
        assert s1 == s2


# ---------------------------------------------------------------------------
# Decode Utils
# ---------------------------------------------------------------------------

class TestDecodeUtils:
    def test_greedy_feasible_decode(self):
        import torch
        from decode_utils import greedy_feasible_decode
        probs = torch.tensor([0.9, 0.1, 0.7, 0.5, 0.8])
        weights = torch.tensor([10., 20., 15., 25., 12.])
        capacity = 40.0
        x = greedy_feasible_decode(probs, weights, capacity)
        total_w = (x * weights).sum().item()
        assert total_w <= capacity + 1e-6, f"Weight {total_w} > capacity {capacity}"

    def test_greedy_ratio_decode(self):
        import torch
        from decode_utils import greedy_ratio_decode
        values = torch.tensor([60., 100., 120.])
        weights = torch.tensor([10., 20., 30.])
        x = greedy_ratio_decode(values, weights, 50.0)
        total_w = (x * weights).sum().item()
        assert total_w <= 50.0 + 1e-6

    def test_compute_ratio(self):
        from decode_utils import compute_ratio
        assert compute_ratio(90, 100) == 0.9
        assert compute_ratio(100, 100) == 1.0
        assert compute_ratio(50, 0) is None

    def test_compute_gap(self):
        from decode_utils import compute_gap
        assert abs(compute_gap(90, 100) - 0.1) < 1e-6
        assert compute_gap(100, 100) == 0.0


# ---------------------------------------------------------------------------
# Instance Loader
# ---------------------------------------------------------------------------

class TestInstanceLoader:
    def test_load_save_roundtrip(self):
        from instance_loader import load_instance
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "instance_0000.npz"
            W = np.array([10, 20, 30], dtype=np.int32)
            V = np.array([60, 100, 120], dtype=np.int32)
            C = np.int32(50)
            np.savez_compressed(path, weights=W, values=V, capacity=C)

            W2, V2, C2 = load_instance(path)
            np.testing.assert_array_equal(W, W2)
            np.testing.assert_array_equal(V, V2)
            assert C2 == 50

    def test_list_instances(self):
        from instance_loader import list_instances
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                np.savez(Path(tmpdir) / f"instance_{i:04d}.npz",
                         weights=np.array([1]), values=np.array([1]),
                         capacity=np.int32(10))
            files = list_instances(Path(tmpdir))
            assert len(files) == 5

    def test_list_instances_limit(self):
        from instance_loader import list_instances
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                np.savez(Path(tmpdir) / f"instance_{i:04d}.npz",
                         weights=np.array([1]), values=np.array([1]),
                         capacity=np.int32(10))
            files = list_instances(Path(tmpdir), limit=3)
            assert len(files) == 3


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

class TestGraphBuilder:
    def test_output_shape(self):
        from Graph_builder import build_knapsack_graph
        data = build_knapsack_graph(
            weights=[10, 20, 30],
            values=[60, 100, 120],
            capacity=50,
            solution=[0, 1, 1],
        )
        assert data.x.shape == (3, 6), f"Expected (3,6), got {data.x.shape}"
        assert data.y.shape == (3, 1)
        assert data.wts.shape == (3,)
        assert data.vals.shape == (3,)
        assert data.cap.item() == 50

    def test_inference_graph(self):
        from Graph_builder import build_knapsack_graph_inference
        data = build_knapsack_graph_inference(
            weights=[10, 20, 30],
            values=[60, 100, 120],
            capacity=50,
        )
        assert data.x.shape == (3, 6)
        # y should be all zeros (dummy)
        assert data.y.sum().item() == 0

    def test_feature_values(self):
        from Graph_builder import build_knapsack_graph
        data = build_knapsack_graph(
            weights=[10, 20, 30],
            values=[60, 100, 120],
            capacity=50,
            solution=[0, 1, 1],
        )
        # All features should be normalized (0-1 range approximately)
        assert data.x[:, :4].min() >= 0
        assert data.x[:, :4].max() <= 2.0  # cap_ratio can be > 1


# ---------------------------------------------------------------------------
# DQN Environment
# ---------------------------------------------------------------------------

class TestDQNEnv:
    def test_feasibility_enforced(self):
        from dqn_env import KnapsackEnv
        env = KnapsackEnv(
            weights=np.array([100, 200, 300]),
            values=np.array([50, 100, 150]),
            capacity=100,
        )
        s = env.reset()
        # Try to take all items (only first should fit)
        for i in range(3):
            out = env.step(1)  # always try to take
        assert env.compute_solution_weight() <= 100 + 1e-6

    def test_state_dim(self):
        from dqn_env import KnapsackEnv
        env = KnapsackEnv(
            weights=np.array([10, 20, 30]),
            values=np.array([60, 100, 120]),
            capacity=50,
        )
        s = env.reset()
        assert s.ndim == 1
        assert s.shape[0] == 14  # 3 + 2 + 9

    def test_episode_completes(self):
        from dqn_env import KnapsackEnv
        env = KnapsackEnv(
            weights=np.array([10, 20, 30]),
            values=np.array([60, 100, 120]),
            capacity=50,
        )
        s = env.reset()
        done = False
        steps = 0
        while not done:
            out = env.step(1)
            done = out.done
            steps += 1
        assert steps == 3  # one decision per item


# ---------------------------------------------------------------------------
# Cross-solver consistency
# ---------------------------------------------------------------------------

class TestCrossSolver:
    """Verify all solvers agree on simple instances."""

    def test_all_solvers_feasible(self):
        """Every solver must produce feasible solutions."""
        from Dp import solve_knapsack_dp
        from greedy_baseline_eval import solve_knapsack_greedy
        from ga_baseline_eval import solve_knapsack_ga

        W = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        V = np.array([60, 100, 120, 140, 160], dtype=np.int32)
        C = 80

        # DP
        dp_sol = solve_knapsack_dp(W.tolist(), V.tolist(), C)
        dp_w = sum(W[i] for i in range(len(W)) if dp_sol[i])
        assert dp_w <= C

        # Greedy
        gr_sel = solve_knapsack_greedy(W, V, C)
        gr_w = sum(W[i] for i in gr_sel)
        assert gr_w <= C

        # GA
        ga_sel = solve_knapsack_ga(W, V, C, seed=42)
        ga_w = sum(W[i] for i in ga_sel)
        assert ga_w <= C

    def test_no_solver_beats_dp(self):
        """DP is exact — no solver should exceed its value."""
        from Dp import solve_knapsack_dp
        from greedy_baseline_eval import solve_knapsack_greedy
        from ga_baseline_eval import solve_knapsack_ga

        W = np.array(SIMPLE_W, dtype=np.int32)
        V = np.array(SIMPLE_V, dtype=np.int32)

        dp_sol = solve_knapsack_dp(W.tolist(), V.tolist(), SIMPLE_C)
        dp_val = sum(V[i] * dp_sol[i] for i in range(len(V)))

        gr_sel = solve_knapsack_greedy(W, V, SIMPLE_C)
        gr_val = sum(V[i] for i in gr_sel)
        assert gr_val <= dp_val + 1e-6

        ga_sel = solve_knapsack_ga(W, V, SIMPLE_C, seed=42)
        ga_val = sum(V[i] for i in ga_sel)
        assert ga_val <= dp_val + 1e-6


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])