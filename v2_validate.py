"""Step 3: choose the checkpoint on the CEC2017-style VALIDATION suite (never CEC2022)."""
import glob
import json
import re
import sys
from multiprocessing import Pool

import numpy as np

import cec2017_val as c17
from meta_rl2 import run_controlled, greedy_policy, keep_policy

SEEDS = 3
CV = len(sys.argv) > 2 and sys.argv[2] == "cv"
D = 10


def job(args):
    alg, fid, s, variants = args
    f = c17.get_function(fid)
    lb, ub = c17.bounds(D)
    init = json.load(open("v2/static_init.json"))[alg]
    seed = 424_242 + 100 * fid + s
    out = {}
    for name, path in variants:
        if name == "Default":
            out[name] = run_controlled(alg, f, D, lb, ub, 2000 * D, seed, keep_policy)[0]
        elif name == "Static":
            out[name] = run_controlled(alg, f, D, lb, ub, 2000 * D, seed, keep_policy, init=init)[0]
        else:
            Q = np.load(path)
            out[name] = run_controlled(alg, f, D, lb, ub, 2000 * D, seed, greedy_policy(Q), init=init)[0]
    return alg, fid, s, out


if __name__ == "__main__":
    algs = sys.argv[1].split(",")
    jobs = []
    for a in algs:
        tag = a + ("_cv" if CV else "")
        ck = sorted(glob.glob(f"v2/{tag}_ep*.npy"), key=lambda p: int(re.findall(r"ep(\d+)", p)[0]))
        variants = [("Default", None), ("Static", None)] + [(p.split("_")[-1][:-4], p) for p in ck]
        jobs += [(a, fid, s, variants) for fid in range(1, 31) for s in range(SEEDS)]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=2)
    sel = {}
    try:
        sel = json.load(open("v2/selected_cv.json" if CV else "v2/selected.json"))
    except Exception:
        pass
    for a in algs:
        R = [r for r in res if r[0] == a]
        names = list(R[0][3].keys())
        means = {n: np.array([np.mean([r[3][n] for r in R if r[1] == fid]) for fid in range(1, 31)]) for n in names}
        print(f"--- {a} (validation, CEC2017-style F1-F30, D={D}, {SEEDS} seeds)")
        best, best_score = None, -1
        for n in names:
            w = int(np.sum(means[n] < means["Default"]))
            print(f"  {n:8s} beats Default on {w}/30", flush=True)
            if n.startswith("ep") and w > best_score:
                best, best_score = n, w
        sel[a] = dict(checkpoint=f"v2/{a}{'_cv' if CV else ''}_{best}.npy", val_wins=best_score,
                      static_val_wins=int(np.sum(means['Static'] < means['Default'])))
        print("  selected", sel[a])
    json.dump(sel, open("v2/selected_cv.json" if CV else "v2/selected.json", "w"), indent=1)
