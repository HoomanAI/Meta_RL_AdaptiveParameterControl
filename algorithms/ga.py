"""Baseline real-coded Genetic Algorithm (population-based)."""

from __future__ import annotations

import numpy as np

from .common import (OptimizationResult, clip_to_bounds, gaussian_mutation,
                      init_population, sbx_crossover, tournament_selection)


def run_ga(func, d, lb, ub, pop_size=40, max_evals=10000,
           crossover_rate=0.9, mutation_rate=0.1, mutation_sigma=0.1,
           eta=15.0, tournament_k=3, seed=None) -> OptimizationResult:
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
        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_selection(pop, fitness, tournament_k, rng)
            p2 = tournament_selection(pop, fitness, tournament_k, rng)
            if rng.random() < crossover_rate:
                c1, c2 = sbx_crossover(p1, p2, eta, lb, ub, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)
        offspring = np.array(offspring)
        offspring = gaussian_mutation(offspring, mutation_rate, mutation_sigma, lb, ub, rng)

        remaining = max_evals - evals
        if remaining < pop_size:
            offspring = offspring[:remaining]
        off_fitness = func(offspring)
        evals += len(offspring)

        # elitism: combine, keep best pop_size
        comb_pop = np.vstack([pop, offspring])
        comb_fit = np.concatenate([fitness, off_fitness])
        order = np.argsort(comb_fit)[:pop_size]
        pop = comb_pop[order]
        fitness = comb_fit[order]

        if fitness[0] < best_fitness:
            best_fitness = fitness[0]
            best_solution = pop[0].copy()
        convergence.append(best_fitness)

    return OptimizationResult(best_fitness, best_solution, convergence, evals)
