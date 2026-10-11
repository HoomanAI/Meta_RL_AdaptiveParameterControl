import glob, json, re, sys
from multiprocessing import Pool
import numpy as np
import cec2017_val as c17
import meta_rl as v1
import meta_rl2 as v2
SEEDS, D = 3, 10
def cands(a):
    out = [("Default", None), ("v1", f"policies/{a}_Q.npz")]
    for tag in ([a] + (["PSOb"] if a == "PSO" else [])):
        ck = glob.glob(f"v2/{tag}_ep*.npy")
        out.append((f"{tag}-avg", ck))
    return out
def job(args):
    a, fid, s = args
    f = c17.get_function(fid); lb, ub = c17.bounds(D); seed = 424_242 + 100 * fid + s
    init = json.load(open("v2/static_init.json"))
    out = {}
    for name, p in cands(a):
        if name == "Default":
            out[name] = v1.run_controlled(a, f, D, lb, ub, 2000 * D, seed, v1.keep_policy)[0]
        elif name == "v1":
            out[name] = v1.run_controlled(a, f, D, lb, ub, 2000 * D, seed, v1.greedy_policy(np.load(p)["Q"]))[0]
        else:
            tag = name[:-4]
            Q = np.mean([np.load(x) for x in p], axis=0)
            out[name] = v2.run_controlled(tag, f, D, lb, ub, 2000 * D, seed, v2.greedy_policy(Q), init=init[tag])[0]
    return a, fid, out
if __name__ == "__main__":
    algs = sys.argv[1].split(",")
    with Pool(2) as p:
        res = p.map(job, [(a, f, s) for a in algs for f in range(1, 31) for s in range(SEEDS)], chunksize=2)
    for a in algs:
        R = [r for r in res if r[0] == a]
        names = list(R[0][2].keys())
        m = {n: np.array([np.mean([r[2][n] for r in R if r[1] == f]) for f in range(1, 31)]) for n in names}
        print(a, {n: int(np.sum(m[n] < m["Default"])) for n in names if n != "Default"}, flush=True)
