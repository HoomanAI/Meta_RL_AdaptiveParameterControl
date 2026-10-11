"""Fig. 4 data: action shares of the frozen QGA policy (MRL) on the CEC2022-style test
functions (D=10,20,30; runs 0-2) by budget quarter and stagnation / diversity bin."""
import json
from multiprocessing import Pool
import numpy as np
import meta_rl2 as v2
from benchmarks import cec2022

INIT = json.load(open("v2/static_init.json"))["QGA"]
Q = np.load(json.load(open("v2/final_choice.json"))["QGA"]["Q"])
G = v2.greedy_policy(Q)


def decode(s):
    s, k2 = divmod(s, v2.N_KP); s, k1 = divmod(s, v2.N_KP)
    s, dv = divmod(s, v2.N_DIV); p, st = divmod(s, v2.N_STAG)
    return p, st, dv


def job(a):
    fid, d, run = a
    f = cec2022.get_function(fid); lb, ub = cec2022.bounds(d)
    log = []

    def pol(s, rng):
        act = G(s, rng); log.append((*decode(s), act)); return act
    v2.run_controlled("QGA", f, d, lb, ub, 2000 * d, 1000 * fid + 10 * run + d, pol, init=INIT)
    return log


if __name__ == "__main__":
    jobs = [(f, d, r) for d in (10, 20, 30) for f in range(1, 13) for r in range(3)]
    C = np.zeros((4, 3, 5)); Dv = np.zeros((4, 4, 5))
    with Pool(2) as p:
        for log in p.map(job, jobs):
            for ph, st, dv, a in log:
                C[ph, st, a] += 1; Dv[ph, dv, a] += 1
    json.dump({"C": C.tolist(), "Dv": Dv.tolist()}, open("policy_counts.json", "w"))
    print("decisions:", int(C.sum()))
