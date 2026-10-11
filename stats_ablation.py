"""Table 4: QGA ablation and training-budget results (D=10,20; 15 runs) and the post-hoc
comparisons of the full framework against each ablation."""
import json
from collections import defaultdict
import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from stats_utils import holm, a12, wil
from benchmarks.cec2022 import _BIAS

DEF = defaultdict(dict)
for alg, fid, d, run, out in json.load(open("results_main.json")):
    if alg == "QGA":
        DEF[(d, fid)][run] = out["Default"]
A = defaultdict(lambda: defaultdict(dict))
for fn in ("abl_abl.json", "abl_ckpt.json"):
    for fid, d, run, v, x in json.load(open(fn)):
        A[v][(d, fid)][run] = x
cases = [(d, f) for d in (10, 20) for f in range(1, 13)]
vec = lambda dct, c: np.array([dct[c][r] for r in range(15)])
out = {}
for group in (["full", "no-phase", "no-stag", "no-div", "no-knob", "rel-reward", "default-start", "online"],
              ["ep500", "ep1000", "ep1500", "ep2000", "ep2500", "ep3000"]):
    P, AA = [], []
    for v in group:
        for c in cases:
            x, y = vec(A[v], c), vec(DEF, c)
            P.append(wil(x, y)); AA.append(a12(x, y))
    adj = holm(P)  # family: all variants x 24 cases
    M = np.array([[vec(A[v], c).mean() for v in group] for c in cases])
    R = np.array([rankdata(r) for r in M]).mean(0)
    print(f"--- Friedman p = {friedmanchisquare(*M.T).pvalue:.2g}")
    for j, v in enumerate(group):
        sl = slice(24 * j, 24 * (j + 1)); a = np.array(AA[sl]); ad = adj[sl]
        better = int(sum(M[i, j] < vec(DEF, c).mean() for i, c in enumerate(cases)))
        w, l = int(sum((ad < .05) & (a > .5))), int(sum((ad < .05) & (a < .5)))
        out[v] = dict(rank=round(float(R[j]), 2), better=better, sig_w=w, sig_l=l, a12=round(float(a.mean()), 2))
        print(f"{v:14s} rank {R[j]:.2f}  better {better}/24  sig W/L {w}/{l}  A12 {a.mean():.2f}")
# post-hoc: full vs each ablation, Wilcoxon on the 24 case-wise log10 ratios of mean errors, Holm over 7
abl = ["no-phase", "no-stag", "no-div", "no-knob", "rel-reward", "default-start", "online"]
err = lambda v, c: np.maximum(vec(A[v], c) - _BIAS[c[1]], 1e-8).mean()  # mean error J - J*
full = np.array([err("full", c) for c in cases])
ph, ps = [], []
for v in abl:
    lr = np.log10(np.array([err(v, c) for c in cases]) / full)
    ps.append(float(wilcoxon(lr).pvalue))
    ph.append(dict(v=v, med=float(np.median(lr)), worse=int((lr > 0).sum()), better=int((lr < 0).sum())))
print("--- post-hoc (full vs ablation)")
for r, p, pa in zip(ph, ps, holm(ps)):
    r["p"], r["p_holm"] = p, float(pa)
    print(f"{r['v']:14s} worse on {r['worse']}/24  p_Holm {pa:.3f}")
json.dump({"table": out, "posthoc": ph}, open("ablation_stats.json", "w"), indent=1)
