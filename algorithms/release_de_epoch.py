"""RELEASE-DE-E: epoch-based Q-learning strategy selection for
Differential Evolution (DE/rand/1/bin).

Same growing-epoch scheme as RELEASE-GA-E (see release_ga_epoch.py): the
agent commits to a (F, CR) strategy for an epoch of N generations
(N growing over the run), then is rewarded on the aggregate best- and
mean-fitness improvement over that epoch.

Action -> (F, CR):
  A1 (intensification): small scale factor F, low crossover rate CR
  A9 (diversification): large scale factor F, high crossover rate CR
"""

from __future__ import annotations

import numpy as np

from .common import OptimizationResult, de_generation, init_population, population_diversity
from .rl_agent import QLearningAgent, StateEncoder, relative_improvement_reward, spectrum

F_SCALE = spectrum(0.3, 0.9, 9)
CR_RATE = spectrum(0.3, 0.9, 9)


def run_release_de_epoch(func, d, lb, ub, pop_size=30, max_evals=10000,
                          seed=None, agent_kwargs=None,
                          n_min=1, n_max=10, p=1.0,
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
        F = F_SCALE[action]
        CR = CR_RATE[action]
        action_history.append(action)

        n_gens = n_min + (n_max - n_min) * (evals / max_evals) ** p
        n_gens = max(1, int(round(n_gens)))

        epoch_start_best = best_fitness
        epoch_start_mean = float(np.mean(fitness))

        for _ in range(n_gens):
            if evals >= max_evals:
                break
            remaining = max_evals - evals
            pop, fitness, n_new = de_generation(func, pop, fitness, lb, ub, F, CR,
                                                  remaining, rng)
            evals += n_new

            idx = np.argmin(fitness)
            if fitness[idx] < best_fitness:
                best_fitness = fitness[idx]
                best_solution = pop[idx].copy()
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
