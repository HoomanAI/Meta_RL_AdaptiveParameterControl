"""Extended identification study: noise robustness (SNR 20/30/40 dB), parameter
estimation error, and adaptive competitors (jDE, SHADE, LDIW-PSO)."""
import json
from multiprocessing import Pool

import numpy as np

import meta_rl2 as v2
from sysid import SYSTEMS, D, LO, HI, decode, simulate, N
from algorithms.qga import run_qga
from algorithms.de import run_de
from algorithms.pso import run_pso
from algorithms.release_qga_epoch import run_release_qga_epoch
from test_meta import run_jde
from baselines_extra import run_shade, run_ldiw_pso

INIT = json.load(open("v2/static_init.json"))["QGA"]
QSEL = np.load("v2/QGA_ep2500.npy")
ALGS = ["QGA", "RELEASE-QGA-E", "QGA-static", "QGA-MetaRL", "DE", "jDE", "SHADE", "PSO", "LDIW-PSO"]
SNRS = [20, 30, 40]


def make(name, snr):
    rng = np.random.default_rng(2027)
    u = rng.standard_normal(N)
    y0 = simulate(SYSTEMS[name][None, :], u)[0]
    y = y0 + rng.standard_normal(N) * np.sqrt(np.var(y0) / 10 ** (snr / 10))
    vy = np.var(y)
    tr = {"best": np.inf, "x": None}

    def cost(X):
        X = np.atleast_2d(X)
        e = np.mean((simulate(decode(X), u) - y[None, :]) ** 2, axis=1) / vy
        e = np.where(np.isfinite(e), np.minimum(e, 1e6), 1e6)
        i = int(np.argmin(e))
        if e[i] < tr["best"]:
            tr["best"], tr["x"] = float(e[i]), X[i].copy()
        return e
    floor = float(np.mean((y - y0) ** 2) / vy)
    return cost, tr, floor


def job(a):
    name, snr, run, alg = a
    f, tr, floor = make(name, snr)
    lb, ub = np.full(D, -100.0), np.full(D, 100.0)
    me, seed = 2000 * D, 9000 + 37 * run + (0 if name == "S1" else 5000) + snr
    if alg == "QGA":
        run_qga(f, D, lb, ub, pop_size=30, max_evals=me, seed=seed)
    elif alg == "RELEASE-QGA-E":
        run_release_qga_epoch(f, D, lb, ub, pop_size=30, max_evals=me, seed=seed)
    elif alg == "DE":
        run_de(f, D, lb, ub, pop_size=30, max_evals=me, seed=seed)
    elif alg == "PSO":
        run_pso(f, D, lb, ub, pop_size=30, max_evals=me, seed=seed)
    elif alg == "jDE":
        run_jde(f, D, lb, ub, 30, me, seed)
    elif alg == "SHADE":
        run_shade(f, D, lb, ub, 30, me, seed)
    elif alg == "LDIW-PSO":
        run_ldiw_pso(f, D, lb, ub, 30, me, seed)
    else:
        pol = v2.keep_policy if alg == "QGA-static" else v2.greedy_policy(QSEL)
        v2.run_controlled("QGA", f, D, lb, ub, me, seed, pol, init=INIT)
    th = decode(tr["x"]); true = SYSTEMS[name]
    # parameter error, each parameter normalised by its search range (K on log scale)
    t1 = th.copy(); t0 = true.copy(); t1[6] = np.log(t1[6]); t0[6] = np.log(t0[6])
    perr = float(np.sqrt(np.mean(((t1 - t0) / (HI - LO)) ** 2)))
    return name, snr, run, alg, tr["best"], perr, floor


if __name__ == "__main__":
    simulate(SYSTEMS["S1"][None, :], np.zeros(5))
    jobs = [(s, snr, r, a) for s in SYSTEMS for snr in SNRS for r in range(25) for a in ALGS]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=5)
    json.dump(res, open("sysid2_results.json", "w"))
    print("done")
