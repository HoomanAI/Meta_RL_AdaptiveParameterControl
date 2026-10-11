"""Baseline Quantum-inspired Genetic Algorithm (QiGA) for continuous optimization.

Each individual is encoded as a vector of qubit angles phi in [0, pi/2].
A real-valued solution is decoded as

    x_i = lb_i + sin(phi_i)^2 * (ub_i - lb_i)

Each generation applies:
  1. a quantum rotation gate that rotates every qubit angle toward the
     current global-best individual's angle by a fixed step `theta`,
  2. a quantum-NOT mutation (phi -> pi/2 - phi) applied with probability
     `mutation_rate`,
  3. a uniform crossover of qubit angles between randomly paired
     individuals, applied with probability `crossover_rate`.
"""

from __future__ import annotations

import numpy as np

from .common import OptimizationResult


def _decode(phi, lb, ub):
    return lb + np.sin(phi) ** 2 * (ub - lb)


def run_qga(func, d, lb, ub, pop_size=40, max_evals=10000,
             theta=0.01 * np.pi, mutation_rate=0.05, crossover_rate=0.7,
             seed=None) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    phi = rng.uniform(0.0, np.pi / 2, size=(pop_size, d))
    pop = _decode(phi, lb, ub)
    fitness = func(pop)
    evals = pop_size

    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_solution = pop[best_idx].copy()
    best_phi = phi[best_idx].copy()
    convergence = [best_fitness]

    while evals < max_evals:
        direction = np.sign(best_phi[None, :] - phi)
        phi = phi + theta * direction
        phi = np.clip(phi, 0.0, np.pi / 2)

        mut_mask = rng.random(phi.shape) < mutation_rate
        phi = np.where(mut_mask, np.pi / 2 - phi, phi)

        perm = rng.permutation(pop_size)
        for i in range(0, pop_size - 1, 2):
            a, b = perm[i], perm[i + 1]
            if rng.random() < crossover_rate:
                mask = rng.random(d) < 0.5
                tmp = phi[a, mask].copy()
                phi[a, mask] = phi[b, mask]
                phi[b, mask] = tmp

        pop = _decode(phi, lb, ub)
        remaining = max_evals - evals
        if remaining < pop_size:
            pop = pop[:remaining]
            phi = phi[:remaining]
        fitness = func(pop)
        evals += pop.shape[0]

        idx = np.argmin(fitness)
        if fitness[idx] < best_fitness:
            best_fitness = fitness[idx]
            best_solution = pop[idx].copy()
            best_phi = phi[idx].copy()
        convergence.append(best_fitness)

    return OptimizationResult(best_fitness, best_solution, convergence, evals)
