"""Extended comparison: fills in D=30 for all variants (zero-shot: controllers were
trained on D in {10,20} only) and adds the SHADE / LDIW-PSO competitors.
Records wall-clock time per run for the overhead analysis."""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from benchmarks import cec2022
import meta_rl as v1
import meta_rl2 as v2
from algorithms.de import run_de
from algorithms.qga import run_qga
from algorithms.ga import run_ga
from algorithms.pso import run_pso
from algorithms.release_de_epoch import run_release_de_epoch
from algorithms.release_qga_epoch import run_release_qga_epoch
from algorithms.release_ga_epoch import run_release_ga_epoch
from algorithms.release_pso_epoch import run_release_pso_epoch
from test_meta import run_jde
from baselines_extra import run_shade, run_ldiw_pso

BASE = {"DE": run_de, "QGA": run_qga, "GA": run_ga, "PSO": run_pso}
REL = {"DE": run_release_de_epoch, "QGA": run_release_qga_epoch, "GA": run_release_ga_epoch,
       "PSO": run_release_pso_epoch}
INIT = json.load(open("v2/static_init.json"))
CH = json.load(open("v2/final_choice.json"))


def run_variant(alg, v, f, d, lb, ub, me, seed):
    if v == "Default":
        return BASE[alg](f, d, lb, ub, pop_size=30, max_evals=me, seed=seed).best_fitness
    if v == "RELEASE-E":
        return REL[alg](f, d, lb, ub, pop_size=30, max_evals=me, seed=seed).best_fitness
    if v == "Random-nudge":
        return v1.run_controlled(alg, f, d, lb, ub, me, seed, v1.random_policy)[0]
    if v == "Meta-RL":  # M1
        return v1.run_controlled(alg, f, d, lb, ub, me, seed,
                                 v1.greedy_policy(np.load(f"policies/{alg}_Q.npz")["Q"]))[0]
    if v == "Static-tuned":
        return v2.run_controlled(alg, f, d, lb, ub, me, seed, v2.keep_policy, init=INIT[alg])[0]
    if v == "Meta-RL-final":  # M2
        c = CH[alg]
        if c["family"] == "v1":
            return v1.run_controlled(alg, f, d, lb, ub, me, seed,
                                     v1.greedy_policy(np.load(c["Q"])["Q"]))[0]
        return v2.run_controlled(alg, f, d, lb, ub, me, seed, v2.greedy_policy(np.load(c["Q"])),
                                 init=INIT[alg])[0]
    if v == "jDE":
        return run_jde(f, d, lb, ub, 30, me, seed)
    if v == "SHADE":
        return run_shade(f, d, lb, ub, 30, me, seed)
    if v == "LDIW-PSO":
        return run_ldiw_pso(f, d, lb, ub, 30, me, seed)
    raise ValueError(v)


def job(args):
    alg, fid, d, run, variants = args
    f = cec2022.get_function(fid)
    lb, ub = cec2022.bounds(d)
    seed = 1000 * fid + 10 * run + d
    out, tm = {}, {}
    for v in variants:
        t = time.perf_counter()
        out[v] = float(run_variant(alg, v, f, d, lb, ub, 2000 * d, seed))
        tm[v] = time.perf_counter() - t
    return alg, fid, d, run, out, tm


ALL = ["Default", "RELEASE-E", "Random-nudge", "Meta-RL", "Static-tuned", "Meta-RL-final"]
EXTRA = {"DE": ["jDE", "SHADE"], "PSO": ["LDIW-PSO"], "QGA": [], "GA": []}

if __name__ == "__main__":
    part = sys.argv[1]
    jobs = []
    for alg in sys.argv[2].split(","):
        if part == "d30":
            jobs += [(alg, f, 30, r, ALL + EXTRA[alg]) for f in range(1, 13) for r in range(15)]
        else:  # new competitors at D=10,20
            if EXTRA[alg]:
                jobs += [(alg, f, d, r, [e for e in EXTRA[alg] if e != "jDE"])
                         for d in (10, 20) for f in range(1, 13) for r in range(15)]
    res = []
    with Pool(2) as p:
        for i, r in enumerate(p.imap_unordered(job, jobs, chunksize=2)):
            res.append(r)
            if (i + 1) % 30 == 0:
                json.dump(res, open(f"exp2_{part}_{sys.argv[2].replace(',', '_')}.json", "w"))
                print(i + 1, "/", len(jobs), flush=True)
    json.dump(res, open(f"exp2_{part}_{sys.argv[2].replace(',', '_')}.json", "w"))
    print("done", flush=True)
