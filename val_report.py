"""CEC2017-based validation results (D=10) for the selected controllers:
Default, Tuned static and the selected Meta-RL policy, 10 runs per function.
Runs 0-2 use the exact seeds of checkpoint selection."""
import json, sys, time
from multiprocessing import Pool
import numpy as np
import cec2017_val as c17
import meta_rl as v1
import meta_rl2 as v2
CH = json.load(open("v2/final_choice.json"))
INIT = json.load(open("v2/static_init.json"))
D = 10
def job(args):
    alg, fid, s = args
    f = c17.get_function(fid); lb, ub = c17.bounds(D)
    seed = 424_242 + 100 * fid + s; me = 2000 * D
    c = CH[alg]
    out = {"Default": float(v2.run_controlled(alg, f, D, lb, ub, me, seed, v2.keep_policy)[0]),
           "Static-tuned": float(v2.run_controlled(alg, f, D, lb, ub, me, seed, v2.keep_policy, init=INIT[alg])[0])}
    if c["family"] == "v2":
        Q = np.load(c["Q"])
        out["Meta-RL"] = float(v2.run_controlled(alg, f, D, lb, ub, me, seed, v2.greedy_policy(Q), init=INIT[alg])[0])
    else:
        Q = np.load(c["Q"])["Q"]
        out["Meta-RL"] = float(v1.run_controlled(alg, f, D, lb, ub, me, seed, v1.greedy_policy(Q))[0])
    return alg, fid, s, out
if __name__ == "__main__":
    algs = sys.argv[1].split(","); runs = int(sys.argv[2])
    jobs = [(a, f, s) for a in algs for f in range(1, 31) for s in range(runs)]
    t = time.time()
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=2)
    json.dump(res, open(f"val_{'_'.join(algs)}_{runs}.json", "w"))
    print("done", time.time() - t, flush=True)
