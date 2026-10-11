"""Convergence curves (best-so-far error vs evaluations), QGA family, D=20."""
import json
from multiprocessing import Pool

import numpy as np

import meta_rl2 as v2
from benchmarks import cec2022
from algorithms.qga import run_qga
from algorithms.release_qga_epoch import run_release_qga_epoch

INIT = json.load(open("v2/static_init.json"))["QGA"]
Q = np.load(json.load(open("v2/final_choice.json"))["QGA"]["Q"])
BIAS = {1: 300, 2: 400, 3: 600, 4: 800, 5: 900, 6: 1800, 7: 2000, 8: 2200, 9: 2300, 10: 2400, 11: 2600, 12: 2700}
D, ME, G = 20, 40000, np.linspace(0, 40000, 60)
FIDS = [1, 4, 6, 10]


def curve(ev, best):
    return np.interp(G, ev, np.minimum.accumulate(best)).tolist()


def job(a):
    fid, run, v = a
    f = cec2022.get_function(fid); lb, ub = cec2022.bounds(D); seed = 1000 * fid + 10 * run + D
    if v in ("QGA", "RELEASE-E"):
        r = (run_qga if v == "QGA" else run_release_qga_epoch)(f, D, lb, ub, pop_size=30, max_evals=ME, seed=seed)
        c = np.array(r.convergence); ev = 30 * np.arange(len(c))
    else:
        pol = v2.keep_policy if v == "Static" else v2.greedy_policy(Q)
        _, _, _, h, s0 = v2.run_controlled("QGA", f, D, lb, ub, ME, seed, pol, init=INIT, return_hist=True)
        c = np.array([s0] + list(h)); ev = np.minimum(150 * np.arange(len(c)), ME)
    return fid, run, v, curve(ev, c - BIAS[fid])


if __name__ == "__main__":
    jobs = [(f, r, v) for f in FIDS for r in range(15) for v in ["QGA", "RELEASE-E", "Static", "M2"]]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=4)
    json.dump(dict(grid=G.tolist(), res=res), open("conv.json", "w"))
    print("done")
