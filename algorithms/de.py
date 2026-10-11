"""Baseline Differential Evolution (DE/rand/1/bin)."""

from __future__ import annotations

import numpy as np

from .common import OptimizationResult, de_generation, init_population


def run_de(func, d, lb, ub, pop_size=30, max_evals=10000,
           F=0.5, CR=0.9, seed=None) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    pop = init_population(pop_size, d, lb, ub, rng)
    fitness = func(pop)
    evals = pop_size

    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_solution = pop[best_idx].copy()
    convergence = [best_fitness]

    while evals < max_evals:
        pop, fitness, n_new = de_generation(func, pop, fitness, lb, ub, F, CR,
                                              max_evals - evals, rng)
        evals += n_new

        idx = np.argmin(fitness)
        if fitness[idx] < best_fitness:
            best_fitness = fitness[idx]
            best_solution = pop[idx].copy()
        convergence.append(best_fitness)

    return OptimizationResult(best_fitness, best_solution, convergence, evals)
