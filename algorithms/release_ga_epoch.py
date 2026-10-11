"""RELEASE-GA-E: epoch-based Q-learning strategy selection for a real-coded GA.

Variant of RELEASE-GA where the agent does not re-decide every generation.
Instead it commits to a strategy (crossover/mutation/step-size) for an
"epoch" of N consecutive generations, then observes the *aggregate* effect
(best-fitness improvement + mean-fitness improvement over the epoch) before
choosing the next action. The epoch length grows over the course of the run:

    N(evals) = n_min + (n_max - n_min) * (evals / max_evals) ** p

i.e. short epochs early (fast strategy switching while the population is
changing quickly) and longer epochs later (more reliable, less noisy reward
signal once improvements become small/rare).
"""

from __future__ import annotations

import numpy as np

from .common import (OptimizationResult, ga_generation, init_population,
                      population_diversity)
from .rl_agent import QLearningAgent, StateEncoder, relative_improvement_reward, spectrum

CROSSOVER_RATE = spectrum(0.95, 0.55, 9)
MUTATION_RATE = spectrum(0.02, 0.40, 9)
MUTATION_SIGMA = spectrum(0.02, 0.25, 9)


def run_release_ga_epoch(func, d, lb, ub, pop_size=40, max_evals=10000,
                          eta=15.0, tournament_k=3, seed=None,
                          agent_kwargs=None, n_min=1, n_max=10, p=1.0,
                          best_weight=0.7) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    agent = QLearningAgent(rng=rng, **(agent_kwargs or {}))
    encoder = StateEncoder()

    pop = init_population(pop_size, d, lb, ub, rng)
    fitness = func(pop)
    evals = pop_size

    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_solution = pop[best_idx].copy()
    convergence = [best_fitness]

    diversity = population_diversity(pop, lb, ub)
    state = encoder.encode(best_fitness, diversity, evals / max_evals)
    action_history = []

    while evals < max_evals:
        action = agent.select_action(state)
        crossover_rate = CROSSOVER_RATE[action]
        mutation_rate = MUTATION_RATE[action]
        mutation_sigma = MUTATION_SIGMA[action]
        action_history.append(action)

        n_gens = n_min + (n_max - n_min) * (evals / max_evals) ** p
        n_gens = max(1, int(round(n_gens)))

        epoch_start_best = best_fitness
        epoch_start_mean = float(np.mean(fitness))

        for _ in range(n_gens):
            if evals >= max_evals:
                break
            remaining = max_evals - evals
            pop, fitness, n_new = ga_generation(
                func, pop, fitness, lb, ub, crossover_rate, mutation_rate,
                mutation_sigma, eta, tournament_k, remaining, rng)
            evals += n_new
            if fitness[0] < best_fitness:
                best_fitness = fitness[0]
                best_solution = pop[0].copy()
            convergence.append(best_fitness)

        epoch_end_mean = float(np.mean(fitness))
        diversity = population_diversity(pop, lb, ub)
        next_state = encoder.encode(best_fitness, diversity, evals / max_evals)

        r_best = relative_improvement_reward(epoch_start_best, best_fitness)
        r_mean = relative_improvement_reward(epoch_start_mean, epoch_end_mean)
        reward = best_weight * r_best + (1 - best_weight) * r_mean

        agent.update(state, action, reward, next_state)
        state = next_state

    return OptimizationResult(best_fitness, best_solution, convergence, evals,
                               extra={"Q": agent.Q, "action_history": action_history})
