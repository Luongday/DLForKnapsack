"""Genetic Algorithm baseline solver for 0/1 Knapsack.

Ported and heavily refactored from GA.py (original Neuro-Knapsack project).

Key changes vs original:
    - Removed hardcoded demo at module level (no side effects on import).
    - Replaced recursion with iterative loop — no more RecursionError on
      hard instances or long runs.
    - Population initialised for any n_items (original hardcoded 5).
    - Added max_generations guard — was infinite before.
    - Added mutation_rate parameter (was hardcoded 0.5).
    - Returns solution as binary 0/1 numpy array (project standard).
    - Added evaluate_ga_on_dataset() for batch baseline evaluation.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import numpy as np


class KnapsackGA:
    """Genetic Algorithm for 0/1 Knapsack.

    Args:
        weights:         Item weights.
        values:          Item values (profits).
        capacity:        Knapsack capacity.
        population_size: Number of individuals per generation.
        mutation_rate:   Probability of flipping each gene on mutation.
        max_generations: Hard stop after this many generations.
        seed:            Random seed for reproducibility.
    """

    def __init__(
        self,
        weights: Sequence[float],
        values:  Sequence[float],
        capacity: float,
        population_size: int  = 50,
        mutation_rate:   float = 0.1,
        max_generations: int  = 500,
        seed: int | None = None,
    ):
        self.weights         = list(weights)
        self.values          = list(values)
        self.capacity        = float(capacity)
        self.n               = len(weights)
        self.population_size = population_size
        self.mutation_rate   = mutation_rate
        self.max_generations = max_generations

        if seed is not None:
            random.seed(seed)

        self.population: List[List[int]] = self._init_population()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_population(self) -> List[List[int]]:
        """Random binary population of size population_size."""
        return [
            [random.randint(0, 1) for _ in range(self.n)]
            for _ in range(self.population_size)
        ]

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def _fitness(self, individual: List[int]) -> float:
        """Total value if feasible, else -1 (infeasible penalty)."""
        total_w = sum(self.weights[i] for i, g in enumerate(individual) if g == 1)
        if total_w > self.capacity:
            return -1.0
        return sum(self.values[i] for i, g in enumerate(individual) if g == 1)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select(self) -> List[List[int]]:
        """Keep top half by fitness."""
        scored = sorted(
            self.population,
            key=self._fitness,
            reverse=True,
        )
        return scored[: self.population_size // 2]

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def _crossover(
        self, p1: List[int], p2: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Single-point crossover."""
        if self.n <= 1:
            return p1[:], p2[:]
        point = random.randint(1, self.n - 1)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        return c1, c2

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate(self, individual: List[int]) -> List[int]:
        """Flip each gene with probability mutation_rate."""
        return [
            1 - g if random.random() < self.mutation_rate else g
            for g in individual
        ]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def solve(self, verbose: bool = False) -> Tuple[np.ndarray, float, int]:
        """Run the GA and return the best solution found.

        Args:
            verbose: Print best fitness each generation.

        Returns:
            (solution, best_value, generations_run)
            solution is a binary 0/1 numpy array of length n.
        """
        best_individual = max(self.population, key=self._fitness)
        best_value      = self._fitness(best_individual)
        generations     = 0

        for gen in range(self.max_generations):
            generations = gen + 1
            elite   = self._select()
            children: List[List[int]] = []

            # Crossover pairs
            random.shuffle(elite)
            for i in range(0, len(elite) - 1, 2):
                c1, c2 = self._crossover(elite[i], elite[i + 1])
                children.extend([c1, c2])
            # If odd elite, pair last with first
            if len(elite) % 2 == 1:
                c1, c2 = self._crossover(elite[-1], elite[0])
                children.extend([c1, c2])

            # Mutate children
            children = [self._mutate(c) for c in children]

            # New population = elite + children (trim to population_size)
            self.population = (elite + children)[: self.population_size]

            # Track best
            current_best = max(self.population, key=self._fitness)
            current_val  = self._fitness(current_best)
            if current_val > best_value:
                best_value      = current_val
                best_individual = current_best[:]

            if verbose:
                print(f"Gen {gen+1:04d} | best_value={best_value:.1f}")

        solution = np.array(best_individual, dtype=np.int8)
        return solution, max(best_value, 0.0), generations


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def solve_knapsack_ga(
    weights: Sequence[float],
    values:  Sequence[float],
    capacity: float,
    population_size: int   = 50,
    mutation_rate:   float = 0.1,
    max_generations: int   = 500,
    seed: int | None = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, float]:
    """One-shot GA solve. Returns (solution, value)."""
    ga = KnapsackGA(
        weights, values, capacity,
        population_size=population_size,
        mutation_rate=mutation_rate,
        max_generations=max_generations,
        seed=seed,
    )
    solution, value, _ = ga.solve(verbose=verbose)
    return solution, value


def evaluate_ga_on_dataset(
    dataset_dir: str,
    n_instances: int | None = None,
    population_size: int   = 50,
    max_generations: int   = 200,
    seed: int = 0,
) -> dict:
    """Run GA on all NPZ instances and return aggregate stats.

    Note: GA is slow — use small n_instances for quick checks.
    """
    from pathlib import Path

    files = sorted(Path(dataset_dir).glob("instance_*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files in {dataset_dir}")
    if n_instances:
        files = files[:n_instances]

    ratios, values = [], []
    for idx, path in enumerate(files):
        arr  = np.load(path)
        w    = arr["weights"].astype(float)
        v    = arr["values"].astype(float)
        c    = float(arr["capacity"])
        dp_v = float(arr["dp_value"])

        sol, ga_val = solve_knapsack_ga(
            w, v, c,
            population_size=population_size,
            max_generations=max_generations,
            seed=seed + idx,
        )
        ratio = ga_val / dp_v if dp_v > 0 else 0.0
        ratios.append(ratio)
        values.append(ga_val)
        print(f"[{idx+1}/{len(files)}] ga={ga_val:.1f} dp={dp_v:.1f} ratio={ratio:.3f}")

    return {
        "n_instances":     len(files),
        "avg_ga_value":    float(np.mean(values)),
        "avg_ratio_vs_dp": float(np.mean(ratios)),
        "min_ratio":       float(np.min(ratios)),
    }