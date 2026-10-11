"""RELEASE-QGA-E: epoch-based Q-learning strategy selection for the
Quantum-inspired Genetic Algorithm.

Same epoch scheme as RELEASE-GA-E (see release_ga_epoch.py): the agent
commits to a (theta, mutation, crossover) strategy for a growing epoch of
N generations and is rewarded on the aggregate best- and mean-fitness
improvement over that epoch.
"""

from __future__ import annotations

import numpy as np

from .common import OptimizationResult, population_diversity, qga_generation
from .rl_agent import QLearningAgent, StateEncoder, relative_improvement_reward, spectrum

THETA = spectrum(0.005 * np.pi, 0.05 * np.pi, 9)
MUTATION_RATE = spectrum(0.01, 0.30, 9)
CROSSOVER_RATE = spectrum(0.90, 0.30, 9)


def run_release_qga_epoch(func, d, lb, ub, pop_size=40, max_evals=10000,
                           seed=None, agent_kwargs=None,
                           n_min=1, n_max=10, p=1.0,
                           best_weight=0.7) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    agent = QLearningAgent(rng=rng, **(agent_kwargs or {}))
    encoder = StateEncoder()

    phi = rng.uniform(0.0, np.pi / 2, size=(pop_size, d))
    pop = lb + np.sin(phi) ** 2 * (ub - lb)
    fitness = func(pop)
    evals = pop_size

    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_solution = pop[best_idx].copy()
    best_phi = phi[best_idx].copy()
    convergence = [best_fitness]

    diversity = population_diversity(pop, lb, ub)
    state = encoder.encode(best_fitness, diversity, evals / max_evals)
    action_history = []

    while evals < max_evals:
        action = agent.select_action(state)
        theta = THETA[action]
        mutation_rate = MUTATION_RATE[action]
        crossover_rate = CROSSOVER_RATE[action]
        action_history.append(action)

        n_gens = n_min + (n_max - n_min) * (evals / max_evals) ** p
        n_gens = max(1, int(round(n_gens)))

        epoch_start_best = best_fitness
        epoch_start_mean = float(np.mean(fitness))

        for _ in range(n_gens):
            if evals >= max_evals:
                break
            remaining = max_evals - evals
            pop, phi, fitness, n_new = qga_generation(
                func, phi, lb, ub, theta, mutation_rate, crossover_rate,
                best_phi, remaining, rng)
            evals += n_new

            idx = np.argmin(fitness)
            if fitness[idx] < best_fitness:
                best_fitness = fitness[idx]
                best_solution = pop[idx].copy()
                best_phi = phi[idx].copy()
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
