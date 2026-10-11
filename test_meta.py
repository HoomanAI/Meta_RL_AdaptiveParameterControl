"""Held-out evaluation on CEC2022-style F1-F12.

Variants per host algorithm:
  Default      : baseline algorithm with its default parameters
  RELEASE-E    : existing per-run epoch-based Q-learning (Simulation/algorithms)
  Random-nudge : same knob + x0.7/x1/x1.43 action set as Meta-RL, actions uniform random
  Meta-RL      : frozen greedy policy meta-trained offline on training functions
  jDE          : (DE only) self-adaptive DE reference (Brest et al., 2006)
"""

from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meta_rl import run_controlled, random_policy, greedy_policy  # noqa: E402
from benchmarks import cec2022  # noqa: E402
from algorithms.common import init_population, clip_to_bounds  # noqa: E402
from algorithms.de import run_de  # noqa: E402
from algorithms.qga import run_qga  # noqa: E402
from algorithms.ga import run_ga  # noqa: E402
from algorithms.pso import run_pso  # noqa: E402
from algorithms.release_de_epoch import run_release_de_epoch  # noqa: E402
from algorithms.release_qga_epoch import run_release_qga_epoch  # noqa: E402
from algorithms.release_ga_epoch import run_release_ga_epoch  # noqa: E402
from algorithms.release_pso_epoch import run_release_pso_epoch  # noqa: E402

BASE = {"DE": run_de, "QGA": run_qga, "GA": run_ga, "PSO": run_pso}
REL = {"DE": run_release_de_epoch, "QGA": run_release_qga_epoch,
       "GA": run_release_ga_epoch, "PSO": run_release_pso_epoch}


def run_jde(func, d, lb, ub, pop_size=30, max_evals=10000, seed=None):
    rng = np.random.default_rng(seed)
    pop = init_population(pop_size, d, lb, ub, rng)
    fit = func(pop)
    evals = pop_size
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    while evals < max_evals:
        Fn = np.where(rng.random(pop_size) < 0.1, 0.1 + 0.9 * rng.random(pop_size), F)
        CRn = np.where(rng.random(pop_size) < 0.1, rng.random(pop_size), CR)
        idx = np.array([rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                        for i in range(pop_size)])
        mutant = clip_to_bounds(pop[idx[:, 0]] + Fn[:, None] * (pop[idx[:, 1]] - pop[idx[:, 2]]),
                                lb, ub)
        mask = rng.random((pop_size, d)) < CRn[:, None]
        mask[np.arange(pop_size), rng.integers(0, d, pop_size)] = True
        trial = np.where(mask, mutant, pop)
        n = min(pop_size, max_evals - evals)
        tf = func(trial[:n])
        evals += n
        better = np.zeros(pop_size, bool)
        better[:n] = tf < fit[:n]
        pop[better] = trial[better]
        fit[better] = tf[better[:n]]
        F[better] = Fn[better]
        CR[better] = CRn[better]
    return float(fit.min())


def job(args):
    alg, fid, d, run, variants = args
    func = cec2022.get_function(fid)
    lb, ub = cec2022.bounds(d)
    me = 2000 * d
    seed = 1000 * fid + 10 * run + d
    out = {}
    Q = np.load(f"policies/{alg}_Q.npz")["Q"]
    for v in variants:
        if v == "Default":
            out[v] = float(BASE[alg](func, d, lb, ub, pop_size=30, max_evals=me, seed=seed).best_fitness)
        elif v == "RELEASE-E":
            out[v] = float(REL[alg](func, d, lb, ub, pop_size=30, max_evals=me, seed=seed).best_fitness)
        elif v == "Random-nudge":
            out[v] = float(run_controlled(alg, func, d, lb, ub, me, seed, random_policy)[0])
        elif v == "Meta-RL":
            out[v] = float(run_controlled(alg, func, d, lb, ub, me, seed, greedy_policy(Q))[0])
        elif v == "jDE":
            out[v] = run_jde(func, d, lb, ub, 30, me, seed)
    return (alg, fid, d, run, out)


if __name__ == "__main__":
    algs = sys.argv[1].split(",")
    dims = [int(x) for x in sys.argv[2].split(",")]
    runs = int(sys.argv[3])
    tag = sys.argv[4]
    jobs = []
    for alg in algs:
        variants = ["Default", "RELEASE-E", "Random-nudge", "Meta-RL"] + (["jDE"] if alg == "DE" else [])
        for d in dims:
            for fid in range(1, 13):
                for r in range(runs):
                    jobs.append((alg, fid, d, r, variants))
    results = []
    with Pool(2) as p:
        for i, res in enumerate(p.imap_unordered(job, jobs, chunksize=2)):
            results.append(res)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
                json.dump(results, open(f"results_{tag}.json", "w"))
    json.dump(results, open(f"results_{tag}.json", "w"))
    print("done", flush=True)
