"""Step 1: tune the 2 starting knob values per host on TRAINING functions (6x6 log grid)."""
import json, sys
from multiprocessing import Pool
import numpy as np
from meta_rl2 import KNOBS, run_controlled, keep_policy, sample_train_function
G = 6
CENTERED = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else set()
MULTS = [0.35, 0.6, 1.0, 1.67, 2.86]
def grid(alg):
    if alg in CENTERED:
        dft, lo, hi = KNOBS[alg]
        return [np.unique(np.clip(dft[j] * np.array(MULTS), lo[j], hi[j])) for j in range(2)]
    _, lo, hi = KNOBS[alg]
    return [np.exp(np.linspace(np.log(lo[j]), np.log(hi[j]), G)) for j in range(2)]
def job(args):
    alg, i = args
    rng = np.random.default_rng(555_000 + i)
    d = int([10, 20][rng.integers(2)])
    f, fs = sample_train_function(d, rng)
    lb, ub = np.full(d, -100.), np.full(d, 100.)
    seed = int(rng.integers(1 << 31))
    g1, g2 = grid(alg)
    M = np.full((G + 1, G), np.inf)
    for a in range(len(g1)):
        for b in range(len(g2)):
            M[a, b] = np.log10(max(run_controlled(alg, f, d, lb, ub, 2000 * d, seed, keep_policy, init=(g1[a], g2[b]))[0] - fs, 1e-8))
    M[G, 0] = np.log10(max(run_controlled(alg, f, d, lb, ub, 2000 * d, seed, keep_policy)[0] - fs, 1e-8))
    return alg, M.tolist()
if __name__ == "__main__":
    algs = sys.argv[1].split(","); n = int(sys.argv[2])
    with Pool(2) as p:
        res = p.map(job, [(a, i) for a in algs for i in range(n)], chunksize=1)
    out = {}
    try: out = json.load(open("v2/static_init.json"))
    except Exception: pass
    for a in algs:
        M = np.array([r[1] for r in res if r[0] == a])
        mean = M[:, :G, :].mean(0); mean[~np.isfinite(mean)] = np.inf
        i, j = np.unravel_index(np.argmin(mean), mean.shape)
        g1, g2 = grid(a)
        out[a] = [float(g1[i]), float(g2[j])]
        # rank-based check too: fraction of functions where tuned beats default
        frac = float(np.mean(M[:, i, j] < M[:, G, 0]))
        print(a, "tuned", out[a], "mean log10 err tuned", round(mean[i, j], 3), "default", round(M[:, G, 0].mean(), 3), "tuned beats default on", frac, flush=True)
    json.dump(out, open("v2/static_init.json", "w"), indent=1)
