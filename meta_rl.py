"""Meta-RL (offline, cross-problem) exploration/exploitation control.

Idea 1 + Idea 3 from the design discussion:
  * The controller is trained OFFLINE across many episodes on a distribution
    of TRAINING functions (random shifted/rotated base functions and random
    hybrids, built from the CEC2017-style base-function library), then
    FROZEN and evaluated on held-out CEC2022-style test functions.
  * Each host algorithm exposes ONE exploration/exploitation knob that starts
    at the algorithm's tuned default; actions are multiplicative nudges
    (x0.7, x1, x1/0.7) instead of jumps between fixed parameter settings.
  * State features are scale-invariant (budget phase, stagnation, diversity
    relative to the initial population, knob position).
  * Training reward = log10 reduction of the error to the known optimum of
    the training function per decision. Rewards telescope, so an episode's
    return equals log10(initial error / final error): the controller is
    trained directly on final solution quality, independent of the objective
    scale/offset (the failure mode diagnosed in RELEASE-E).

Host knobs (all other parameters stay at the baseline defaults):
  DE  : scale factor F          (CR fixed at 0.9)
  QGA : rotation step theta     (mutation 0.05, crossover 0.7)
  GA  : Gaussian mutation sigma (crossover 0.9, mutation rate 0.1)
  PSO : inertia weight w        (c1 = c2 = 1.5)
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(1, os.path.dirname(_HERE))  # Simulation/ (algorithms/, benchmarks/)
from algorithms.common import (de_generation, ga_generation, init_population,  # noqa: E402
                               population_diversity, pso_step, qga_generation)
import base_functions as bf  # noqa: E402

MULT = np.array([0.7, 1.0, 1.0 / 0.7])
N_ACTIONS = 3
KEEP = 1  # index of the x1 action
DECIDE_EVERY = 5  # generations per decision

# ---------------------------------------------------------------------------
# Host algorithms with one exposed knob
# ---------------------------------------------------------------------------


class Host:
    knob_init: float
    knob_lo: float
    knob_hi: float

    def __init__(self, func, d, lb, ub, pop_size, rng):
        self.func, self.d, self.lb, self.ub, self.rng = func, d, lb, ub, rng
        self.pop_size = pop_size
        self.evals = 0

    def best(self):
        return self.best_f


class DEHost(Host):
    knob_init, knob_lo, knob_hi = 0.5, 0.1, 1.2
    CR = 0.9

    def __init__(self, *a):
        super().__init__(*a)
        self.pop = init_population(self.pop_size, self.d, self.lb, self.ub, self.rng)
        self.fit = self.func(self.pop)
        self.evals = self.pop_size
        self.best_f = float(self.fit.min())

    def step(self, knob, max_evals):
        self.pop, self.fit, n = de_generation(self.func, self.pop, self.fit, self.lb, self.ub,
                                              knob, self.CR, max_evals - self.evals, self.rng)
        self.evals += n
        self.best_f = min(self.best_f, float(self.fit.min()))

    def diversity(self):
        return population_diversity(self.pop, self.lb, self.ub)


class GAHost(Host):
    knob_init, knob_lo, knob_hi = 0.1, 1e-4, 0.5

    def __init__(self, *a):
        super().__init__(*a)
        self.pop = init_population(self.pop_size, self.d, self.lb, self.ub, self.rng)
        self.fit = self.func(self.pop)
        self.evals = self.pop_size
        self.best_f = float(self.fit.min())

    def step(self, knob, max_evals):
        self.pop, self.fit, n = ga_generation(self.func, self.pop, self.fit, self.lb, self.ub,
                                              0.9, 0.1, knob, 15.0, 3,
                                              max_evals - self.evals, self.rng)
        self.evals += n
        self.best_f = min(self.best_f, float(self.fit.min()))

    def diversity(self):
        return population_diversity(self.pop, self.lb, self.ub)


class QGAHost(Host):
    knob_init, knob_lo, knob_hi = 0.01 * np.pi, 0.001 * np.pi, 0.1 * np.pi

    def __init__(self, *a):
        super().__init__(*a)
        self.phi = self.rng.uniform(0.0, np.pi / 2, size=(self.pop_size, self.d))
        self.pop = self.lb + np.sin(self.phi) ** 2 * (self.ub - self.lb)
        self.fit = self.func(self.pop)
        self.evals = self.pop_size
        i = int(np.argmin(self.fit))
        self.best_f = float(self.fit[i])
        self.best_phi = self.phi[i].copy()

    def step(self, knob, max_evals):
        pop, phi, fit, n = qga_generation(self.func, self.phi, self.lb, self.ub, knob, 0.05, 0.7,
                                          self.best_phi, max_evals - self.evals, self.rng)
        self.evals += n
        i = int(np.argmin(fit))
        if fit[i] < self.best_f:
            self.best_f = float(fit[i])
            self.best_phi = phi[i].copy()
        if n == self.pop_size:
            self.pop, self.phi, self.fit = pop, phi, fit

    def diversity(self):
        return population_diversity(self.pop, self.lb, self.ub)


class PSOHost(Host):
    knob_init, knob_lo, knob_hi = 0.7, 0.2, 1.05

    def __init__(self, *a):
        super().__init__(*a)
        span = self.ub - self.lb
        self.v_max = 0.2 * span
        self.pos = init_population(self.pop_size, self.d, self.lb, self.ub, self.rng)
        self.vel = self.rng.uniform(-1.0, 1.0, size=(self.pop_size, self.d)) * 0.1 * span
        fit = self.func(self.pos)
        self.evals = self.pop_size
        self.pbest_pos, self.pbest_fit = self.pos.copy(), fit.copy()
        i = int(np.argmin(fit))
        self.gbest_pos, self.gbest_fit = self.pos[i].copy(), float(fit[i])
        self.best_f = self.gbest_fit

    def step(self, knob, max_evals):
        (self.pos, self.vel, self.pbest_pos, self.pbest_fit, self.gbest_pos, self.gbest_fit,
         n) = pso_step(self.func, self.pos, self.vel, self.pbest_pos, self.pbest_fit,
                       self.gbest_pos, self.gbest_fit, self.lb, self.ub, knob, 1.5, 1.5,
                       self.v_max, max_evals - self.evals, self.rng)
        self.evals += n
        self.best_f = float(self.gbest_fit)

    def diversity(self):
        return population_diversity(self.pos, self.lb, self.ub)


HOSTS = {"DE": DEHost, "QGA": QGAHost, "GA": GAHost, "PSO": PSOHost}

# ---------------------------------------------------------------------------
# State encoding (scale-invariant)
# ---------------------------------------------------------------------------

N_PHASE, N_STAG, N_DIV, N_KNOB = 4, 3, 4, 3
N_STATES = N_PHASE * N_STAG * N_DIV * N_KNOB  # 144


def encode(frac, stag, div_ratio, knob_pos):
    p = min(int(frac * N_PHASE), N_PHASE - 1)
    s = 0 if stag == 0 else (1 if stag <= 2 else 2)
    lr = np.log10(max(div_ratio, 1e-300))
    dv = 0 if lr > -0.5 else (1 if lr > -1.5 else (2 if lr > -3.0 else 3))
    k = min(int(knob_pos * N_KNOB), N_KNOB - 1)
    return ((p * N_STAG + s) * N_DIV + dv) * N_KNOB + k


# ---------------------------------------------------------------------------
# One controlled run
# ---------------------------------------------------------------------------


def run_controlled(alg, func, d, lb, ub, max_evals, seed, policy, pop_size=30,
                   f_star=None, record=False):
    """policy(state, rng) -> action index. Returns (best_f, transitions, knob_trace)."""
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    H = HOSTS[alg]
    host = H(func, d, lb, ub, pop_size, rng)
    knob = H.knob_init
    log_lo, log_hi = np.log(H.knob_lo), np.log(H.knob_hi)
    div0 = max(host.diversity(), 1e-300)
    stag = 0
    trans = []
    trace = []

    def err(f):
        return max(f - f_star, 1e-8)

    def state():
        kp = (np.log(knob) - log_lo) / (log_hi - log_lo)
        return encode(host.evals / max_evals, stag, host.diversity() / div0, kp)

    s = state()
    while host.evals < max_evals:
        a = policy(s, rng)
        knob = float(np.clip(knob * MULT[a], H.knob_lo, H.knob_hi))
        trace.append(knob)
        prev = host.best()
        for _ in range(DECIDE_EVERY):
            if host.evals >= max_evals:
                break
            host.step(knob, max_evals)
        stag = 0 if host.best() < prev else stag + 1
        s2 = state()
        done = host.evals >= max_evals
        if record:
            r = float(np.log10(err(prev) / err(host.best())))
            trans.append((s, a, r, s2, done))
        s = s2
    return host.best(), trans, trace


# ---------------------------------------------------------------------------
# Training-function distribution (disjoint instances from the CEC2022 test set)
# ---------------------------------------------------------------------------

TRAIN_BASES = [bf.elliptic, bf.bent_cigar, bf.discus, bf.zakharov, bf.rosenbrock,
               bf.rastrigin, bf.schaffer_f6, bf.ackley, bf.griewank, bf.schwefel]


def sample_train_function(d, rng):
    """Random shifted+rotated base function (60%) or random 3-part hybrid (40%).
    Returns (func, f_star). Seeds are drawn from a range disjoint from the
    fixed seeds used by the CEC2022 test suite."""
    seed = int(rng.integers(10_000, 10_000_000))
    if rng.random() < 0.6:
        base = TRAIN_BASES[rng.integers(len(TRAIN_BASES))]
        rot = bf.rotation_matrix(d, seed)
        shift = bf.shift_vector(d, seed + 1)

        def func(x, base=base, rot=rot, shift=shift):
            return base(bf.transform(x, shift, rot))
        f_star = float(func(shift[None, :])[0])
    else:
        idx = rng.choice(len(TRAIN_BASES), size=3, replace=False)
        funcs = [TRAIN_BASES[i] for i in idx]
        ratios = list(rng.dirichlet([2, 2, 2]))

        def func(x, funcs=funcs, ratios=ratios, seed=seed):
            return bf.hybrid(x, seed, funcs, ratios)
        shift = bf.shift_vector(d, seed=3000 + seed)
        f_star = float(func(shift[None, :])[0])
    return func, f_star


# ---------------------------------------------------------------------------
# Offline tabular Q-learning across episodes
# ---------------------------------------------------------------------------


def greedy_policy(Q):
    def pol(s, rng):
        q = Q[s]
        best = np.flatnonzero(q == q.max())
        return KEEP if KEEP in best else int(best[0])
    return pol


def random_policy(s, rng):
    return int(rng.integers(N_ACTIONS))


def keep_policy(s, rng):
    return KEEP


def train(alg, n_episodes, dims=(10, 20), evals_per_dim=2000, alpha=0.1, gamma=0.99,
          eps_start=0.3, eps_end=0.05, seed=0, log_every=50, out=None):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    visits = np.zeros((N_STATES, N_ACTIONS), dtype=int)
    lb_hist = []
    for ep in range(n_episodes):
        eps = eps_start + (eps_end - eps_start) * ep / max(n_episodes - 1, 1)
        d = int(dims[rng.integers(len(dims))])
        func, f_star = sample_train_function(d, rng)
        lb, ub = np.full(d, -100.0), np.full(d, 100.0)

        def pol(s, r):
            if r.random() < eps:
                return int(r.integers(N_ACTIONS))
            q = Q[s]
            best = np.flatnonzero(q == q.max())
            return KEEP if KEEP in best else int(r.choice(best))

        _, trans, _ = run_controlled(alg, func, d, lb, ub, evals_per_dim * d,
                                     int(rng.integers(1 << 31)), pol, f_star=f_star, record=True)
        # backward pass over the episode = faster credit propagation
        for (s, a, r, s2, done) in reversed(trans):
            target = r if done else r + gamma * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])
            visits[s, a] += 1
        ret = sum(t[2] for t in trans)
        lb_hist.append(ret)
        if out and ((ep + 1) % log_every == 0 or ep == n_episodes - 1):
            np.savez(out, Q=Q, visits=visits, returns=np.array(lb_hist))
            print(f"{alg} ep {ep + 1}/{n_episodes} eps={eps:.3f} "
                  f"mean return(last {log_every})={np.mean(lb_hist[-log_every:]):.3f}",
                  flush=True)
    return Q, visits, np.array(lb_hist)


if __name__ == "__main__":
    alg = sys.argv[1]
    n = int(sys.argv[2])
    os.makedirs("policies", exist_ok=True)
    train(alg, n, seed={"DE": 1, "QGA": 2, "GA": 3, "PSO": 4}[alg],
          out=f"policies/{alg}_Q.npz")
