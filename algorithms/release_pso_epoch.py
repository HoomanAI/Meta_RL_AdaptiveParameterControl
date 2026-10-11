"""RELEASE-PSO-E: epoch-based Q-learning strategy selection for
Particle Swarm Optimization.

Same growing-epoch scheme as RELEASE-GA-E (see release_ga_epoch.py): the
agent commits to a (w, c1, c2) strategy for an epoch of N generations
(N growing over the run), then is rewarded on the aggregate best- and
mean-fitness improvement over that epoch.

Action -> (w, c1, c2):
  A1 (intensification): low inertia, high cognitive/social pull (fast
      convergence toward pbest/gbest)
  A9 (diversification): high inertia, low cognitive/social pull (more
      free exploration)
"""

from __future__ import annotations

import numpy as np

from .common import OptimizationResult, init_population, population_diversity, pso_step
from .rl_agent import QLearningAgent, StateEncoder, relative_improvement_reward, spectrum

INERTIA = spectrum(0.4, 0.95, 9)
COGNITIVE = spectrum(2.0, 0.5, 9)
SOCIAL = spectrum(2.0, 0.5, 9)


def run_release_pso_epoch(func, d, lb, ub, pop_size=30, max_evals=10000,
                           seed=None, agent_kwargs=None,
                           n_min=1, n_max=10, p=1.0,
                           best_weight=0.7) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    span = ub - lb
    v_max = 0.2 * span

    agent = QLearningAgent(rng=rng, **(agent_kwargs or {}))
    encoder = StateEncoder()

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

    diversity = population_diversity(pos, lb, ub)
    state = encoder.encode(gbest_fit, diversity, evals / max_evals)
    action_history = []

    while evals < max_evals:
        action = agent.select_action(state)
        w = INERTIA[action]
        c1 = COGNITIVE[action]
        c2 = SOCIAL[action]
        action_history.append(action)

        n_gens = n_min + (n_max - n_min) * (evals / max_evals) ** p
        n_gens = max(1, int(round(n_gens)))

        epoch_start_best = gbest_fit
        epoch_start_mean = float(np.mean(pbest_fit))

        for _ in range(n_gens):
            if evals >= max_evals:
                break
            remaining = max_evals - evals
            pos, vel, pbest_pos, pbest_fit, gbest_pos, gbest_fit, n_new = pso_step(
                func, pos, vel, pbest_pos, pbest_fit, gbest_pos, gbest_fit,
                lb, ub, w, c1, c2, v_max, remaining, rng)
            evals += n_new
            convergence.append(gbest_fit)

        epoch_end_mean = float(np.mean(pbest_fit))
        diversity = population_diversity(pos, lb, ub)
        next_state = encoder.encode(gbest_fit, diversity, evals / max_evals)

        r_best = relative_improvement_reward(epoch_start_best, gbest_fit)
        r_mean = relative_improvement_reward(epoch_start_mean, epoch_end_mean)
        reward = best_weight * r_best + (1 - best_weight) * r_mean

        agent.update(state, action, reward, next_state)
        state = next_state

    return OptimizationResult(gbest_fit, gbest_pos, convergence, evals,
                               extra={"Q": agent.Q, "action_history": action_history})
