"""Quick reproducibility check (about one minute): re-runs a few stored test runs with
the released code and policies and compares them with the stored results."""
import json
import numpy as np
from exp2 import run_variant
from benchmarks import cec2022

stored = {}
for fn in ("results_main.json", "v2/final_QGA_DE_PSO.json", "v2/final_GA.json"):
    for alg, fid, d, run, out in json.load(open(fn)):
        for m, x in out.items():
            stored[(alg, fid, d, run, m)] = x
checks = [("QGA", 1, 10, 0, "Default"), ("QGA", 6, 10, 1, "Meta-RL-final"), ("QGA", 6, 10, 1, "Static-tuned"),
          ("DE", 5, 10, 0, "Meta-RL-final"), ("PSO", 2, 10, 0, "Random-nudge"), ("GA", 3, 10, 2, "Meta-RL-final"),
          ("QGA", 4, 20, 0, "RELEASE-E")]
ok = True
for alg, fid, d, run, m in checks:
    f = cec2022.get_function(fid); lb, ub = cec2022.bounds(d)
    x = float(run_variant(alg, m, f, d, lb, ub, 2000 * d, 1000 * fid + 10 * run + d))
    ref = stored[(alg, fid, d, run, m)]
    same = np.isclose(x, ref, rtol=1e-9, atol=1e-9)
    ok &= bool(same)
    print(f"{alg:4s} F{fid:<2d} D={d} run {run} {m:14s} {x:.6g} vs stored {ref:.6g}  {'OK' if same else 'DIFF'}")
print("ALL MATCH" if ok else "MISMATCH (check NumPy version / platform)")
