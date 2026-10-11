import json, numpy as np
from collections import defaultdict
from scipy.stats import wilcoxon
from stats_utils import holm, a12
R = json.load(open('val_QGA_DE_PSO_GA_10.json'))
V = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for alg, fid, s, out in R:
    for m, x in out.items():
        V[alg][fid][m][s] = x - 100.0 * fid
def arr(a, f, m): return np.array([V[a][f][m][s] for s in range(10)])
def wil(x, y):
    d = x - y
    if np.all(np.abs(d) <= 1e-12 * (1 + np.abs(y))): return 1.0
    try: return float(wilcoxon(x, y).pvalue)
    except ValueError: return 1.0
res = {}
for a in ['QGA', 'DE', 'PSO', 'GA']:
    comps = ['Default'] + (['Static-tuned'] if a in ('QGA', 'GA') else [])
    P, A, B, Rr = [], [], [], []
    for c in comps:
        for f in range(1, 31):
            x, y = arr(a, f, 'Meta-RL'), arr(a, f, c)
            P.append(wil(x, y)); A.append(a12(x, y)); B.append(x.mean() < y.mean())
            Rr.append(np.log10(max(x.mean(), 1e-8) / max(y.mean(), 1e-8)))
    adj = holm(P); res[a] = {}
    for j, c in enumerate(comps):
        sl = slice(30 * j, 30 * (j + 1)); ad = adj[sl]; aa = np.array(A[sl])
        w = int(sum((ad < .05) & (aa > .5))); l = int(sum((ad < .05) & (aa < .5)))
        res[a][c] = dict(better=int(sum(B[sl])), w=w, t=30 - w - l, l=l, a12=round(float(aa.mean()), 2),
                         med_ratio=round(float(1 - 10 ** np.median(Rr[sl])), 3))
        print(a, c, res[a][c])
json.dump(res, open('val_stats.json', 'w'), indent=1)
