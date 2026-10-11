"""Matched random baseline for M2 (same k0, same two parameters, same five actions,
uniform random actions). Resumable: one output file per (alg, d)."""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from benchmarks import cec2022
import meta_rl2 as v2
INIT = json.load(open("v2/static_init.json"))
def job(args):
    alg, fid, d, run = args
    f = cec2022.get_function(fid); lb, ub = cec2022.bounds(d)
    seed = 1000 * fid + 10 * run + d
    return alg, fid, d, run, {"Random-M2": float(v2.run_controlled(alg, f, d, lb, ub, 2000 * d, seed, v2.random_policy, init=INIT[alg])[0])}
if __name__ == "__main__":
    os.makedirs("randm2", exist_ok=True)
    for alg in sys.argv[1].split(","):
        for d in (10, 20, 30):
            fn = f"randm2/{alg}_{d}.json"
            if os.path.exists(fn): continue
            t = time.time()
            with Pool(2) as p:
                res = p.map(job, [(alg, f, d, r) for f in range(1, 13) for r in range(15)], chunksize=3)
            json.dump(res, open(fn, "w")); print(fn, round(time.time() - t), flush=True)
