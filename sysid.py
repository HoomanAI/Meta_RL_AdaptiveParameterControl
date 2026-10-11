"""System-identification transfer test: output-error estimation of a
Wiener-Hammerstein system (G1 -> static polynomial -> G2), D = 7.

    v = G1(q) u,     w = v + a2 v^2 + a3 v^3,     y = K * G2(q) w + e
    Gi(q) = (1 - 2 ri cos(wi) + ri^2) / (1 - 2 ri cos(wi) q^-1 + ri^2 q^-2)  (unit DC gain)

theta = (r1, w1, a2, a3, r2, w2, K). The cost is the normalised simulation
(output) error, which is non-convex in the pole parameters. The meta-trained
QGA policy is applied FROZEN (it never saw this problem class).
"""
import json
import sys
from multiprocessing import Pool

import numpy as np
from numba import njit

from algorithms.qga import run_qga
from algorithms.de import run_de
from algorithms.pso import run_pso
from algorithms.release_qga_epoch import run_release_qga_epoch
import meta_rl2 as v2

N = 1000
D = 7
LO = np.array([0.30, 0.02 * np.pi, -1.0, -1.0, 0.30, 0.02 * np.pi, np.log(0.2)])
HI = np.array([0.98, 0.98 * np.pi, 1.0, 1.0, 0.98, 0.98 * np.pi, np.log(5.0)])
SYSTEMS = {
    "S1": np.array([0.90, 0.25 * np.pi, 0.30, -0.15, 0.80, 0.60 * np.pi, 1.5]),
    "S2": np.array([0.95, 0.10 * np.pi, -0.20, 0.10, 0.70, 0.40 * np.pi, 0.8]),
}


def decode(x):
    """x in [-100,100]^7 -> physical parameters."""
    t = LO + (np.asarray(x) + 100.0) / 200.0 * (HI - LO)
    t[..., 6] = np.exp(t[..., 6])
    return t


@njit(cache=True)
def simulate(P, u):
    m = P.shape[0]
    n = u.shape[0]
    Y = np.zeros((m, n))
    for i in range(m):
        r1, w1, a2, a3, r2, w2, K = P[i, 0], P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6]
        c1, d1 = 2 * r1 * np.cos(w1), r1 * r1
        c2, d2 = 2 * r2 * np.cos(w2), r2 * r2
        g1, g2 = 1 - c1 + d1, 1 - c2 + d2
        v1 = v2 = z1 = z2 = 0.0
        for t in range(n):
            v = g1 * u[t] + c1 * v1 - d1 * v2
            v2, v1 = v1, v
            w = v + a2 * v * v + a3 * v * v * v
            z = g2 * w + c2 * z1 - d2 * z2
            z2, z1 = z1, z
            Y[i, t] = K * z
    return Y


def make_data(name, seed=2027, snr_db=30.0):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(N)
    y0 = simulate(SYSTEMS[name][None, :], u)[0]
    noise = rng.standard_normal(N) * np.sqrt(np.var(y0) / 10 ** (snr_db / 10))
    return u, y0 + noise, y0


def make_cost(name):
    u, y, _ = make_data(name)
    vy = np.var(y)

    def cost(X):
        X = np.atleast_2d(X)
        Yh = simulate(decode(X), u)
        e = np.mean((Yh - y[None, :]) ** 2, axis=1) / vy
        return np.where(np.isfinite(e), np.minimum(e, 1e6), 1e6)
    return cost


INIT = json.load(open("v2/static_init.json"))
QSEL = np.load("v2/QGA_ep2500.npy")


def job(args):
    name, run, alg = args
    f = make_cost(name)
    lb, ub = np.full(D, -100.0), np.full(D, 100.0)
    me = 2000 * D
    seed = 9000 + 37 * run + (0 if name == "S1" else 5000)
    if alg == "QGA":
        r = run_qga(f, D, lb, ub, pop_size=30, max_evals=me, seed=seed)
        b, x = r.best_fitness, r.best_solution
    elif alg == "RELEASE-QGA-E":
        r = run_release_qga_epoch(f, D, lb, ub, pop_size=30, max_evals=me, seed=seed)
        b, x = r.best_fitness, r.best_solution
    elif alg == "DE":
        r = run_de(f, D, lb, ub, pop_size=30, max_evals=me, seed=seed)
        b, x = r.best_fitness, r.best_solution
    elif alg == "PSO":
        r = run_pso(f, D, lb, ub, pop_size=30, max_evals=me, seed=seed)
        b, x = r.best_fitness, r.best_solution
    else:
        pol = v2.keep_policy if alg == "QGA-static" else v2.greedy_policy(QSEL)
        b = v2.run_controlled("QGA", f, D, lb, ub, me, seed, pol, init=INIT["QGA"])[0]
        x = None
    return name, run, alg, float(b)


ALGS = ["QGA", "RELEASE-QGA-E", "QGA-static", "QGA-MetaRL", "DE", "PSO"]

if __name__ == "__main__":
    runs = int(sys.argv[1])
    simulate(SYSTEMS["S1"][None, :], np.zeros(10))  # compile
    jobs = [(s, r, a) for s in SYSTEMS for r in range(runs) for a in ALGS]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=3)
    json.dump(res, open("sysid_results.json", "w"))
    print("done")
