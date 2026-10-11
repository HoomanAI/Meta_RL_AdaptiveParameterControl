"""Meta-RL v2: two knobs per host, offline-tuned starting point, richer
training distribution, checkpoint selection on a separate validation suite.

Protocol (no test leakage):
  1. Starting knob values are tuned on TRAINING functions (grid search).
  2. The Q-learning controller is meta-trained on TRAINING functions
     (base / hybrid / composition, random seeds), starting from those values.
  3. Checkpoints are compared on a VALIDATION suite (CEC2017-style F1-F30,
     fixed seeds), never on CEC2022.
  4. CEC2022 F1-F12 is used once, for the final test.

Knobs (all other parameters at baseline defaults):
  DE : F, CR          QGA: theta, mutation rate
  GA : sigma, p_mut   PSO: w, c (= c1 = c2)
Actions: keep | knob1 x0.7 | knob1 x1/0.7 | knob2 x0.7 | knob2 x1/0.7
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(1, os.path.dirname(_HERE))
from algorithms.common import (de_generation, ga_generation, init_population,  # noqa: E402
                               population_diversity, pso_step, qga_generation)
import base_functions as bf  # noqa: E402

N_ACTIONS = 5
KEEP = 0
DECIDE_EVERY = 5
STEP = 0.7

# name: (defaults, lows, highs)
KNOBS = {
    "DE": ((0.5, 0.9), (0.1, 0.05), (1.2, 1.0)),
    "QGA": ((0.01 * np.pi, 0.05), (0.001 * np.pi, 0.002), (0.1 * np.pi, 0.3)),
    "GA": ((0.1, 0.1), (1e-4, 0.01), (0.5, 1.0)),
    "PSO": ((0.7, 1.5), (0.2, 0.3), (1.05, 2.5)),
    # PSOb: knob2 = cognitive coefficient c1, social c2 = 3 - c1 (balance, constant sum)
    "PSOb": ((0.7, 1.5), (0.2, 0.3), (1.05, 2.7)),
}


class Host:
    def __init__(self, func, d, lb, ub, pop_size, rng):
        self.func, self.d, self.lb, self.ub, self.rng = func, d, lb, ub, rng
        self.pop_size = pop_size


class DEHost(Host):
    def __init__(self, *a):
        super().__init__(*a)
        self.pop = init_population(self.pop_size, self.d, self.lb, self.ub, self.rng)
        self.fit = self.func(self.pop)
        self.evals = self.pop_size
        self.best_f = float(self.fit.min())

    def step(self, k, me):
        self.pop, self.fit, n = de_generation(self.func, self.pop, self.fit, self.lb, self.ub,
                                              k[0], k[1], me - self.evals, self.rng)
        self.evals += n
        self.best_f = min(self.best_f, float(self.fit.min()))

    def diversity(self):
        return population_diversity(self.pop, self.lb, self.ub)


class GAHost(DEHost):
    def step(self, k, me):
        self.pop, self.fit, n = ga_generation(self.func, self.pop, self.fit, self.lb, self.ub,
                                              0.9, k[1], k[0], 15.0, 3, me - self.evals, self.rng)
        self.evals += n
        self.best_f = min(self.best_f, float(self.fit.min()))


class QGAHost(Host):
    def __init__(self, *a):
        super().__init__(*a)
        self.phi = self.rng.uniform(0.0, np.pi / 2, size=(self.pop_size, self.d))
        self.pop = self.lb + np.sin(self.phi) ** 2 * (self.ub - self.lb)
        self.fit = self.func(self.pop)
        self.evals = self.pop_size
        i = int(np.argmin(self.fit))
        self.best_f, self.best_phi = float(self.fit[i]), self.phi[i].copy()

    def step(self, k, me):
        pop, phi, fit, n = qga_generation(self.func, self.phi, self.lb, self.ub, k[0], k[1], 0.7,
                                          self.best_phi, me - self.evals, self.rng)
        self.evals += n
        i = int(np.argmin(fit))
        if fit[i] < self.best_f:
            self.best_f, self.best_phi = float(fit[i]), phi[i].copy()
        if n == self.pop_size:
            self.pop, self.phi, self.fit = pop, phi, fit

    def diversity(self):
        return population_diversity(self.pop, self.lb, self.ub)


class PSOHost(Host):
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

    def step(self, k, me):
        (self.pos, self.vel, self.pbest_pos, self.pbest_fit, self.gbest_pos, self.gbest_fit,
         n) = pso_step(self.func, self.pos, self.vel, self.pbest_pos, self.pbest_fit,
                       self.gbest_pos, self.gbest_fit, self.lb, self.ub, k[0], k[1], k[1],
                       self.v_max, me - self.evals, self.rng)
        self.evals += n
        self.best_f = float(self.gbest_fit)

    def diversity(self):
        return population_diversity(self.pos, self.lb, self.ub)


class PSOBalanceHost(PSOHost):
    def step(self, k, me):
        (self.pos, self.vel, self.pbest_pos, self.pbest_fit, self.gbest_pos, self.gbest_fit,
         n) = pso_step(self.func, self.pos, self.vel, self.pbest_pos, self.pbest_fit,
                       self.gbest_pos, self.gbest_fit, self.lb, self.ub, k[0], k[1], 3.0 - k[1],
                       self.v_max, me - self.evals, self.rng)
        self.evals += n
        self.best_f = float(self.gbest_fit)


HOSTS = {"DE": DEHost, "QGA": QGAHost, "GA": GAHost, "PSO": PSOHost, "PSOb": PSOBalanceHost}

N_PHASE, N_STAG, N_DIV, N_KP = 4, 3, 4, 3
# Ablation switches (defaults = full method)
MASK = set()      # subset of {"phase", "stag", "div", "knob"}
REWARD = "log"    # "log" (telescoping log-error) or "rel" (relative improvement, RELEASE-style)
N_STATES = N_PHASE * N_STAG * N_DIV * N_KP * N_KP  # 432


def encode(frac, stag, div_ratio, kp1, kp2):
    p = min(int(frac * N_PHASE), N_PHASE - 1)
    s = 0 if stag == 0 else (1 if stag <= 2 else 2)
    lr = np.log10(max(div_ratio, 1e-300))
    dv = 0 if lr > -0.5 else (1 if lr > -1.5 else (2 if lr > -3.0 else 3))
    k1 = min(int(kp1 * N_KP), N_KP - 1)
    k2 = min(int(kp2 * N_KP), N_KP - 1)
    return (((p * N_STAG + s) * N_DIV + dv) * N_KP + k1) * N_KP + k2


def run_controlled(alg, func, d, lb, ub, max_evals, seed, policy, init=None,
                   f_star=None, record=False, pop_size=30, ref_hist=None, ref_start=None,
                   return_hist=False):
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    defaults, lo, hi = KNOBS[alg]
    lo, hi = np.array(lo), np.array(hi)
    k = np.array(init if init is not None else defaults, float)
    host = HOSTS[alg](func, d, lb, ub, pop_size, rng)
    div0 = max(host.diversity(), 1e-300)
    start_best = host.best_f
    stag = 0
    trans, trace, best_hist = [], [], []
    llo, lhi = np.log(lo), np.log(hi)

    def state():
        kp = (np.log(k) - llo) / (lhi - llo)
        return encode(0.0 if "phase" in MASK else host.evals / max_evals,
                      0 if "stag" in MASK else stag,
                      1.0 if "div" in MASK else host.diversity() / div0,
                      0.0 if "knob" in MASK else kp[0], 0.0 if "knob" in MASK else kp[1])

    def err(f):
        return max(f - f_star, 1e-8)

    s = state()
    while host.evals < max_evals:
        a = policy(s, rng)
        if a > 0:
            j = (a - 1) // 2
            k[j] = np.clip(k[j] * (STEP if (a - 1) % 2 == 0 else 1 / STEP), lo[j], hi[j])
        trace.append(k.copy())
        prev = host.best_f
        for _ in range(DECIDE_EVERY):
            if host.evals >= max_evals:
                break
            host.step(k, max_evals)
        stag = 0 if host.best_f < prev else stag + 1
        best_hist.append(host.best_f)
        s2 = state()
        if record:
            if REWARD == "rel":
                r = float((prev - host.best_f) / (abs(prev) + 1e-12)) if host.best_f < prev else -0.001
            else:
                r = float(np.log10(err(prev) / err(host.best_f)))
            if ref_hist is not None:  # control variate: subtract the default run's gain
                t = len(best_hist) - 1
                rp = ref_hist[t - 1] if t > 0 else ref_start
                rn = ref_hist[min(t, len(ref_hist) - 1)]
                r -= float(np.log10(err(rp) / err(rn)))
            trans.append((s, a, r, s2, host.evals >= max_evals))
        s = s2
    if return_hist:
        return host.best_f, trans, np.array(trace), best_hist, start_best
    return host.best_f, trans, np.array(trace)


# ------------------------- training distribution ---------------------------
TRAIN_BASES = [bf.elliptic, bf.bent_cigar, bf.discus, bf.zakharov, bf.rosenbrock,
               bf.rastrigin, bf.schaffer_f6, bf.ackley, bf.griewank, bf.schwefel]


def sample_train_function(d, rng):
    seed = int(rng.integers(10_000, 10_000_000))
    u = rng.random()
    if u < 0.4:
        base = TRAIN_BASES[rng.integers(len(TRAIN_BASES))]
        rot, shift = bf.rotation_matrix(d, seed), bf.shift_vector(d, seed + 1)

        def func(x, base=base, rot=rot, shift=shift):
            return base(bf.transform(x, shift, rot))
        xs = shift
    elif u < 0.7:
        idx = rng.choice(len(TRAIN_BASES), 3, replace=False)
        funcs = [TRAIN_BASES[i] for i in idx]
        ratios = list(rng.dirichlet([2, 2, 2]))

        def func(x, funcs=funcs, ratios=ratios, seed=seed):
            return bf.hybrid(x, seed, funcs, ratios)
        xs = bf.shift_vector(d, seed=3000 + seed)
    else:
        n = int(rng.integers(3, 6))
        idx = rng.choice(len(TRAIN_BASES), n, replace=False)
        funcs = [TRAIN_BASES[i] for i in idx]
        sig = [10 * (i + 1) for i in range(n)]
        lam = [float(10 ** rng.uniform(-6, 0)) for _ in range(n)]
        bias = [100 * i for i in range(n)]
        fs = seed % 50_000  # keep composition seeds bounded

        def func(x, funcs=funcs, sig=sig, lam=lam, bias=bias, fs=fs):
            return bf.composition(x, 90_000 + fs, funcs, sig, lam, bias)
        xs = bf.shift_vector(d, seed=5000 + (90_000 + fs) * 100, scale=80.0)
    return func, float(func(xs[None, :])[0])


def eps_greedy(Q, eps):
    def pol(s, r):
        if r.random() < eps:
            return int(r.integers(N_ACTIONS))
        q = Q[s]
        b = np.flatnonzero(q == q.max())
        return KEEP if KEEP in b else int(r.choice(b))
    return pol


def greedy_policy(Q):
    def pol(s, r):
        q = Q[s]
        b = np.flatnonzero(q == q.max())
        return KEEP if KEEP in b else int(b[0])
    return pol


def keep_policy(s, r):
    return KEEP


def random_policy(s, r):
    return int(r.integers(N_ACTIONS))


def train(alg, n_episodes, init, seed, out, alpha=0.1, gamma=0.99, eps0=0.3, eps1=0.05,
          ckpt_every=500, cv=False, resume_from=0):
    rng = np.random.default_rng(seed + 7919 * resume_from)
    Q = np.zeros((N_STATES, N_ACTIONS))
    if resume_from:
        Q = np.load(f"{out}_ep{resume_from}.npy")
    rets = []
    for ep in range(resume_from, n_episodes):
        eps = eps0 + (eps1 - eps0) * ep / max(n_episodes - 1, 1)
        d = int([10, 20][rng.integers(2)])
        func, fs = sample_train_function(d, rng)
        lb, ub = np.full(d, -100.0), np.full(d, 100.0)
        sd = int(rng.integers(1 << 31))
        ref_hist = ref_start = None
        if cv:
            _, _, _, ref_hist, ref_start = run_controlled(alg, func, d, lb, ub, 2000 * d, sd, keep_policy,
                                                          init=init, return_hist=True)
        _, tr, _ = run_controlled(alg, func, d, lb, ub, 2000 * d, sd,
                                  eps_greedy(Q, eps), init=init, f_star=fs, record=True,
                                  ref_hist=ref_hist, ref_start=ref_start)
        for (s, a, r, s2, done) in reversed(tr):
            Q[s, a] += alpha * ((r if done else r + gamma * Q[s2].max()) - Q[s, a])
        rets.append(sum(t[2] for t in tr))
        if (ep + 1) % ckpt_every == 0:
            np.save(f"{out}_ep{ep + 1}.npy", Q)
            print(f"{alg} ep {ep + 1} mean return {np.mean(rets[-ckpt_every:]):.3f}", flush=True)
            np.save(f"{out}_returns_from{resume_from}.npy", np.array(rets))
    np.save(f"{out}_returns.npy", np.array(rets))


if __name__ == "__main__":
    alg, n = sys.argv[1], int(sys.argv[2])
    cv = len(sys.argv) > 3 and sys.argv[3] == "cv"
    init = json.load(open("v2/static_init.json"))[alg]
    seed = {"DE": 11, "QGA": 12, "GA": 13, "PSO": 14, "PSOb": 15}[alg] + (100 if cv else 0)
    resume = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    train(alg, n, init, seed=seed, out=f"v2/{alg}{'_cv' if cv else ''}", cv=cv,
          ckpt_every=250 if cv else 500, resume_from=resume)
