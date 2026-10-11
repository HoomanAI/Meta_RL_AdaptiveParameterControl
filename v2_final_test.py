"""One-shot final test on CEC2022-style F1-F12 (D=10,20; 15 runs) of the
validation-selected controller per host. Default / RELEASE-E / v1 Meta-RL are
reused from results_main.json (same seeds, deterministic)."""
import json, sys
from multiprocessing import Pool
import numpy as np
from benchmarks import cec2022
import meta_rl as v1
import meta_rl2 as v2
CH = json.load(open("v2/final_choice.json"))
INIT = json.load(open("v2/static_init.json"))
def job(args):
    alg, fid, d, run = args
    f = cec2022.get_function(fid); lb, ub = cec2022.bounds(d)
    seed = 1000 * fid + 10 * run + d; me = 2000 * d
    c = CH[alg]
    out = {"Static-tuned": float(v2.run_controlled(alg, f, d, lb, ub, me, seed, v2.keep_policy, init=INIT[alg])[0])}
    if c["family"] == "v2":
        Q = np.load(c["Q"])
        out["Meta-RL-final"] = float(v2.run_controlled(alg, f, d, lb, ub, me, seed, v2.greedy_policy(Q), init=INIT[alg])[0])
    else:  # v1 single-knob policy: identical to Meta-RL in results_main.json
        Q = np.load(c["Q"])["Q"]
        out["Meta-RL-final"] = float(v1.run_controlled(alg, f, d, lb, ub, me, seed, v1.greedy_policy(Q))[0])
    return alg, fid, d, run, out
if __name__ == "__main__":
    algs = sys.argv[1].split(",")
    jobs = [(a, f, d, r) for a in algs for d in (10, 20) for f in range(1, 13) for r in range(15)]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=4)
    json.dump(res, open(f"v2/final_{'_'.join(algs)}.json", "w"))
    print("done", flush=True)
