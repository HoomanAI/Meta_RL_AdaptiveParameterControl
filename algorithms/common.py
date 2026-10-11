"""Shared utilities for the GA / SA / QiGA optimizers and their RELEASE variants."""

from __future__ import annotations

import numpy as np


def clip_to_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.clip(x, lb, ub)


def init_population(pop_size: int, d: int, lb: np.ndarray, ub: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(lb, ub, size=(pop_size, d))


def tournament_selection(pop: np.ndarray, fitness: np.ndarray, k: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Return one selected individual via k-way tournament (minimization)."""
    idx = rng.integers(0, pop.shape[0], size=k)
    best = idx[np.argmin(fitness[idx])]
    return pop[best].copy()


def sbx_crossover(p1: np.ndarray, p2: np.ndarray, eta: float,
                   lb: np.ndarray, ub: np.ndarray,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u = rng.random(p1.shape)
    beta = np.where(
        u <= 0.5,
        (2 * u) ** (1.0 / (eta + 1)),
        (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1)),
    )
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return clip_to_bounds(c1, lb, ub), clip_to_bounds(c2, lb, ub)


def gaussian_mutation(x: np.ndarray, rate: float, sigma_frac: float,
                       lb: np.ndarray, ub: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(x.shape) < rate
    sigma = sigma_frac * (ub - lb)
    noise = rng.normal(0.0, 1.0, size=x.shape) * sigma
    out = np.where(mask, x + noise, x)
    return clip_to_bounds(out, lb, ub)


def ga_generation(func, pop, fitness, lb, ub, crossover_rate, mutation_rate,
                   mutation_sigma, eta, tournament_k, max_new_evals,
                   rng: np.random.Generator):
    """Run one GA generation, returning (new_pop, new_fitness, n_evals)."""
    pop_size = pop.shape[0]
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

    if max_new_evals < pop_size:
        offspring = offspring[:max_new_evals]
    off_fitness = func(offspring)

    comb_pop = np.vstack([pop, offspring])
    comb_fit = np.concatenate([fitness, off_fitness])
    order = np.argsort(comb_fit)[:pop_size]
    return comb_pop[order], comb_fit[order], offspring.shape[0]


def qga_generation(func, phi, lb, ub, theta, mutation_rate, crossover_rate,
                    best_phi, max_new_evals, rng: np.random.Generator):
    """Run one QGA generation, returning (new_pop, new_phi, new_fitness, n_evals)."""
    pop_size = phi.shape[0]
    direction = np.sign(best_phi[None, :] - phi)
    phi = phi + theta * direction
    phi = np.clip(phi, 0.0, np.pi / 2)

    mut_mask = rng.random(phi.shape) < mutation_rate
    phi = np.where(mut_mask, np.pi / 2 - phi, phi)

    perm = rng.permutation(pop_size)
    for i in range(0, pop_size - 1, 2):
        a, b = perm[i], perm[i + 1]
        if rng.random() < crossover_rate:
            mask = rng.random(phi.shape[1]) < 0.5
            tmp = phi[a, mask].copy()
            phi[a, mask] = phi[b, mask]
            phi[b, mask] = tmp

    pop = lb + np.sin(phi) ** 2 * (ub - lb)
    if max_new_evals < pop_size:
        pop = pop[:max_new_evals]
        phi = phi[:max_new_evals]
    fitness = func(pop)
    return pop, phi, fitness, pop.shape[0]


def de_generation(func, pop, fitness, lb, ub, F, CR, max_new_evals,
                   rng: np.random.Generator):
    """One DE/rand/1/bin generation. Returns (new_pop, new_fitness, n_evals)."""
    pop_size, d = pop.shape
    r1 = np.empty(pop_size, dtype=int)
    r2 = np.empty(pop_size, dtype=int)
    r3 = np.empty(pop_size, dtype=int)
    for i in range(pop_size):
        choices = rng.choice(pop_size - 1, size=3, replace=False)
        choices = [c if c < i else c + 1 for c in choices]
        r1[i], r2[i], r3[i] = choices

    mutant = pop[r1] + F * (pop[r2] - pop[r3])
    mutant = clip_to_bounds(mutant, lb, ub)

    cross_mask = rng.random((pop_size, d)) < CR
    j_rand = rng.integers(0, d, size=pop_size)
    cross_mask[np.arange(pop_size), j_rand] = True
    trial = np.where(cross_mask, mutant, pop)

    if max_new_evals < pop_size:
        trial = trial[:max_new_evals]
    trial_fitness = func(trial)
    n_eval = trial.shape[0]

    new_pop = pop.copy()
    new_fitness = fitness.copy()
    better = trial_fitness < fitness[:n_eval]
    new_pop[:n_eval][better] = trial[:n_eval][better]
    new_fitness[:n_eval][better] = trial_fitness[better]
    return new_pop, new_fitness, n_eval


def pso_step(func, pos, vel, pbest_pos, pbest_fit, gbest_pos, gbest_fit,
              lb, ub, w, c1, c2, v_max, max_new_evals, rng: np.random.Generator):
    """One PSO update step. Returns
    (pos, vel, pbest_pos, pbest_fit, gbest_pos, gbest_fit, n_evals)."""
    pop_size, d = pos.shape
    r1 = rng.random((pop_size, d))
    r2 = rng.random((pop_size, d))
    vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos[None, :] - pos)
    vel = np.clip(vel, -v_max, v_max)
    pos = clip_to_bounds(pos + vel, lb, ub)

    if max_new_evals < pop_size:
        pos_eval = pos[:max_new_evals]
    else:
        pos_eval = pos
    n_eval = pos_eval.shape[0]
    fitness = func(pos_eval)

    improved = fitness < pbest_fit[:n_eval]
    pbest_pos = pbest_pos.copy()
    pbest_fit = pbest_fit.copy()
    pbest_pos[:n_eval][improved] = pos_eval[improved]
    pbest_fit[:n_eval][improved] = fitness[improved]

    idx = np.argmin(pbest_fit)
    if pbest_fit[idx] < gbest_fit:
        gbest_fit = pbest_fit[idx]
        gbest_pos = pbest_pos[idx].copy()

    return pos, vel, pbest_pos, pbest_fit, gbest_pos, gbest_fit, n_eval


def population_diversity(pop: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> float:
    """Mean per-dimension std of the population, normalized by the range."""
    rng_span = (ub - lb)
    rng_span = np.where(rng_span == 0, 1.0, rng_span)
    std = np.std(pop, axis=0) / rng_span
    return float(np.mean(std))


class OptimizationResult:
    """Container for a single optimization run."""

    __slots__ = ("best_fitness", "best_solution", "convergence",
                  "evaluations", "extra")

    def __init__(self, best_fitness, best_solution, convergence, evaluations, extra=None):
        self.best_fitness = best_fitness
        self.best_solution = best_solution
        self.convergence = np.asarray(convergence, dtype=float)
        self.evaluations = evaluations
        self.extra = extra or {}
