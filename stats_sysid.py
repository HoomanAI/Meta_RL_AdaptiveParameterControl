"""Tables 3 and 5: Wiener-Hammerstein identification statistics from sysid2_results.json.
Table 3: median J/J0 and median parameter error per method, system and SNR.
Table 5: MRL (QGA-MetaRL) vs default and tuned static QGA - paired Wilcoxon (Holm over
2x6 comparisons), A12, median paired log10 ratio with pointwise 95% bootstrap interval."""
import json
from collections import defaultdict
import numpy as np
from stats_utils import holm, a12, wil

R = json.load(open("sysid2_results.json"))
J = defaultdict(dict); P = defaultdict(dict); J0 = {}
for s, snr, run, alg, j, perr, floor in R:
    J[(s, snr, alg)][run] = j; P[(s, snr, alg)][run] = perr; J0[(s, snr)] = floor
ALGS = ["QGA", "RELEASE-QGA-E", "QGA-static", "QGA-MetaRL", "DE", "jDE", "SHADE", "PSO", "LDIW-PSO"]
SET = [(s, snr) for s in ("S1", "S2") for snr in (20, 30, 40)]
arr = lambda k: np.array([J[k][r] for r in range(25)])

print("Table 3: median J/J0 | median parameter error (%)")
for a in ALGS:
    cells = [f"{np.median(arr((s, snr, a)) / J0[(s, snr)]):.3g} | {100 * np.median([P[(s, snr, a)][r] for r in range(25)]):.2g}"
             for s, snr in SET]
    print(f"{a:14s}", "  ".join(cells))

rng = np.random.default_rng(0)
rows, ps = [], []
for ref in ("QGA", "QGA-static"):
    for s, snr in SET:
        x, y = arr((s, snr, "QGA-MetaRL")), arr((s, snr, ref))
        lr = np.log10(x / y)
        boot = np.median(lr[rng.integers(0, 25, (5000, 25))], axis=1)
        rows.append(dict(ref=ref, s=s, snr=snr, a12=a12(x, y), med=float(np.median(lr)),
                         ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]))
        ps.append(wil(x, y))
for r, p in zip(rows, holm(ps)):
    r["p_holm"] = float(p)
    print(f"{r['ref']:10s} {r['s']} {r['snr']}  A12 {r['a12']:.2f}  median {r['med']:+.3f}  "
          f"CI [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]  p_Holm {p:.3g}")
json.dump(rows, open("sysid_stats.json", "w"), indent=1)
