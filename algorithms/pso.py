"""Baseline Particle Swarm Optimization."""

from __future__ import annotations

import numpy as np

from .common import OptimizationResult, init_population, pso_step


def run_pso(func, d, lb, ub, pop_size=30, max_evals=10000,
            w=0.7, c1=1.5, c2=1.5, seed=None) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    span = ub - lb
    v_max = 0.2 * span

    pos = init_population(pop_size, d, lb, ub, rng)
    vel = rng.uniform(-1.0, 1.0, size=(pop_size, d)) * 0.1 * span
    fitness = func(pos)
    evals = pop_size

    pbest_pos = pos.copy()
    pbest_fit = fitness.copy()
    best_idx = np.argmin(fitness)
    gbest_pos = pos[best_idx].copy()
    gbest_fit = fitness[best_idx]
    convergence = [gbest_fit]

    while evals < max_evals:
        pos, vel, pbest_pos, pbest_fit, gbest_pos, gbest_fit, n_new = pso_step(
            func, pos, vel, pbest_pos, pbest_fit, gbest_pos, gbest_fit,
            lb, ub, w, c1, c2, v_max, max_evals - evals, rng)
        evals += n_new
        convergence.append(gbest_fit)

    return OptimizationResult(gbest_fit, gbest_pos, convergence, evals)
