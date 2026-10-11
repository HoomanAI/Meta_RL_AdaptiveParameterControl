"""Revised statistics (response to mathematical/statistical review).

* Paired design: every method starts from the same initial population for a given
  seed (verified), so per-case tests are Wilcoxon signed-rank tests paired by run.
* Statistical family = all planned comparisons of M2 against all alternatives for
  one base optimizer (n_alternatives x 36 cases); Holm correction within the family.
* Effect size: Vargha-Delaney A12 = P(M2 < other) + 0.5 P(tie)  (minimisation).
* Friedman omnibus test + Nemenyi critical difference (Demsar, 2006).
"""
import json, glob
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, rankdata

V = defaultdict(lambda: defaultdict(dict))   # (alg,d,f) -> method -> {run: value}
for alg, fid, d, run, out in json.load(open('results_main.json')):
    for m, x in out.items():
        V[(alg, d, fid)][m][run] = x
for fn in ['v2/final_QGA_DE_PSO.json', 'v2/final_GA.json']:
    for alg, fid, d, run, out in json.load(open(fn)):
        for m, x in out.items():
            V[(alg, d, fid)][m][run] = x
for fn in glob.glob('randm2/*.json'):
    for alg, fid, d, run, out in json.load(open(fn)):
        for m, x in out.items():
            V[(alg, d, fid)][m][run] = x
for fn in glob.glob('exp2_*.json'):
    for alg, fid, d, run, out, tm in json.load(open(fn)):
        for m, x in out.items():
            V[(alg, d, fid)][m][run] = x

def arr(alg, d, f, m):
    dct = V[(alg, d, f)][m]
    return np.array([dct[r] for r in range(15)])

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); m = len(ps); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
    return adj

def a12(x, y):  # prob x better (smaller) than y
    x = np.asarray(x)[:, None]; y = np.asarray(y)[None, :]
    return float(np.mean(x < y) + 0.5 * np.mean(x == y))

def wil(x, y):
    d = np.asarray(x) - np.asarray(y)
    if np.all(np.abs(d) <= 1e-12 * (1 + np.abs(np.asarray(y)))):
        return 1.0
    try:
        return float(wilcoxon(x, y).pvalue)
    except ValueError:
        return 1.0

Q_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}
NAME = {'Default': 'Default', 'Static-tuned': 'Tuned static', 'Random-nudge': 'Random-M1', 'Random-M2': 'Random-M2',
        'RELEASE-E': 'RELEASE-E', 'Meta-RL': 'Meta-RL M1', 'jDE': 'jDE', 'SHADE': 'SHADE',
        'LDIW-PSO': 'LDIW-PSO', 'Meta-RL-final': 'MRL'}
HOSTS = ['QGA', 'DE', 'GA', 'PSO']
cases = [(d, f) for d in (10, 20, 30) for f in range(1, 13)]
out = {}
for h in HOSTS:
    meths = [m for m in NAME if all(len(V[(h, d, f)].get(m, {})) == 15 for d, f in cases)]
    if h in ('DE', 'PSO'): meths.remove('Static-tuned')
    if h == 'GA': meths.remove('Meta-RL')
    M = np.array([[np.median(arr(h, d, f, m)) for m in meths] for d, f in cases])  # medians for ranking
    Mm = np.array([[np.mean(arr(h, d, f, m)) for m in meths] for d, f in cases])
    R = np.array([rankdata(r) for r in Mm]).mean(0)
    k, N = len(meths), len(cases)
    cd = Q_NEMENYI[k] * np.sqrt(k * (k + 1) / (6 * N))
    fp = friedmanchisquare(*Mm.T).pvalue
    comps = [m for m in meths if m != 'Meta-RL-final']
    P, A, B = [], [], []
    for m in comps:
        for d, f in cases:
            x, y = arr(h, d, f, 'Meta-RL-final'), arr(h, d, f, m)
            P.append(wil(x, y)); A.append(a12(x, y)); B.append(np.mean(x) < np.mean(y))
    adj = holm(P)
    res = {'methods': [NAME[m] for m in meths], 'rank': dict(zip([NAME[m] for m in meths], R.round(2).tolist())),
           'friedman_p': fp, 'cd': round(cd, 2), 'k': k, 'wtl': {}, 'a12': {}, 'vs_default_sig': {}}
    for j, m in enumerate(comps):
        sl = slice(j * N, (j + 1) * N)
        w = int(sum((adj[sl] < .05) & (np.array(A[sl]) > .5)))
        l = int(sum((adj[sl] < .05) & (np.array(A[sl]) < .5)))
        res['wtl'][NAME[m]] = [w, N - w - l, l]
        res['a12'][NAME[m]] = round(float(np.mean(A[sl])), 2)
        if m == 'Default':
            res['vs_default_sig'] = {f"{d}_{f}": bool(adj[sl][i] < .05) for i, (d, f) in enumerate(cases)}
            res['d30_vs_default'] = [int(sum((adj[sl][i] < .05) and A[sl][i] > .5 for i, (d, f) in enumerate(cases) if d == 30)),
                                     int(sum((adj[sl][i] < .05) and A[sl][i] < .5 for i, (d, f) in enumerate(cases) if d == 30))]
    out[h] = res
    print(f"== {h}: Friedman p={fp:.1e}, k={k}, CD={cd:.2f}")
    print("   ranks:", res['rank'])
    for m in comps:
        print(f"   MRL vs {NAME[m]:13s} W/T/L {res['wtl'][NAME[m]]}  mean A12 {res['a12'][NAME[m]]}")
    print("   D30 vs default sig W/L:", res['d30_vs_default'])
json.dump(out, open('stats_test.json', 'w'), indent=1)

# median error reduction  median_c (1 - mean_err_MRL,c / mean_err_ref,c)  over the 36 cases
from benchmarks.cec2022 import _BIAS
def merr(h, d, f, m):
    return np.maximum(arr(h, d, f, m) - _BIAS[f], 1e-8).mean()
print("== median error reduction of MRL (and of Random-M2 / Tuned static vs Default)")
for h, m1, m2 in [("QGA", "Meta-RL-final", "Default"), ("QGA", "Static-tuned", "Default"), ("QGA", "Random-M2", "Default"),
                  ("QGA", "Meta-RL-final", "Random-M2"), ("DE", "Meta-RL-final", "Random-M2"), ("PSO", "Meta-RL-final", "Random-M2")]:
    r = [1 - merr(h, d, f, m1) / merr(h, d, f, m2) for d, f in cases]
    n = sum(merr(h, d, f, m1) < merr(h, d, f, m2) for d, f in cases)
    print(f"   {h}: {NAME[m1]} vs {NAME[m2]}: median reduction {100 * np.median(r):.1f}%, lower mean in {n}/36 cases")
