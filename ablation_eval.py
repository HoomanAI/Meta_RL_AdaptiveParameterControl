"""Evaluate QGA ablations, the online (learn-within-run) variant and the
training-budget curve on the CEC2022 test functions (D=10,20; 15 runs)."""
import glob
import json
import sys
from multiprocessing import Pool

import numpy as np

import meta_rl2 as v2
from benchmarks import cec2022

INIT = json.load(open("v2/static_init.json"))["QGA"]
DEFAULT = v2.KNOBS["QGA"][0]
ABL = ["full", "no-phase", "no-stag", "no-div", "no-knob", "rel-reward", "default-start"]
CKPT = [500, 1000, 1500, 2000, 2500, 3000]


class Online:
    """Same state/action design, Q-table learned from scratch inside the run
    (relative-improvement reward: the optimum is unknown at run time)."""

    def __init__(self, tracker, n_dec):
        self.Q = np.zeros((v2.N_STATES, v2.N_ACTIONS))
        self.t, self.prev, self.n, self.tr = 0, None, n_dec, tracker

    def __call__(self, s, rng):
        best = self.tr["best"]
        if self.prev is not None:
            ps, pa, pb = self.prev
            r = (pb - best) / (abs(pb) + 1e-12) if best < pb else -0.001
            self.Q[ps, pa] += 0.1 * (r + 0.99 * self.Q[s].max() - self.Q[ps, pa])
        eps = 0.3 + (0.05 - 0.3) * min(self.t / self.n, 1)
        self.t += 1
        if rng.random() < eps:
            a = int(rng.integers(v2.N_ACTIONS))
        else:
            q = self.Q[s]
            b = np.flatnonzero(q == q.max())
            a = v2.KEEP if v2.KEEP in b else int(rng.choice(b))
        self.prev = (s, a, best)
        return a


def job(args):
    fid, d, run, variant = args
    base = cec2022.get_function(fid)
    lb, ub = cec2022.bounds(d)
    seed = 1000 * fid + 10 * run + d
    me = 2000 * d
    v2.MASK, v2.REWARD = set(), "log"
    if variant == "online":
        tr = {"best": np.inf}

        def f(x):
            y = base(x)
            tr["best"] = min(tr["best"], float(np.min(y)))
            return y
        pol = Online(tr, me / (30 * v2.DECIDE_EVERY))
        return fid, d, run, variant, float(v2.run_controlled("QGA", f, d, lb, ub, me, seed, pol, init=INIT)[0])
    if variant.startswith("ep"):
        Q = np.load(f"v2/QGA_{variant}.npy")
        init = INIT
    else:
        Q = np.load(f"v2/abl_{variant}_ep1500.npy")
        if variant.startswith("no-"):
            v2.MASK = {variant[3:]}
        init = DEFAULT if variant == "default-start" else INIT
    return fid, d, run, variant, float(v2.run_controlled("QGA", base, d, lb, ub, me, seed,
                                                         v2.greedy_policy(Q), init=init)[0])


if __name__ == "__main__":
    which = sys.argv[1]
    variants = ABL + ["online"] if which == "abl" else [f"ep{c}" for c in CKPT]
    jobs = [(f, d, r, v) for v in variants for d in (10, 20) for f in range(1, 13) for r in range(15)]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=6)
    json.dump(res, open(f"abl_{which}.json", "w"))
    print("done")
