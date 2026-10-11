"""Chunked ablation training for a time-limited shell: resumes from the latest
checkpoint (every 50 episodes) and stops before the time budget."""
import glob
import json
import re
import sys
import time

import meta_rl2 as v2

V = {"full": dict(), "no-phase": dict(mask={"phase"}), "no-stag": dict(mask={"stag"}),
     "no-div": dict(mask={"div"}), "no-knob": dict(mask={"knob"}), "rel-reward": dict(reward="rel"),
     "default-start": dict(start="default")}

if __name__ == "__main__":
    name, total, budget = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    cfg = V[name]
    v2.MASK = cfg.get("mask", set())
    v2.REWARD = cfg.get("reward", "log")
    init = v2.KNOBS["QGA"][0] if cfg.get("start") == "default" else json.load(open("v2/static_init.json"))["QGA"]
    out = f"v2/abl_{name}"
    done = max([int(re.findall(r"_ep(\d+)\.npy", p)[0]) for p in glob.glob(out + "_ep*.npy")] + [0])
    t0 = time.time()
    while done < total and time.time() - t0 < budget:
        nxt = min(done + 50, total)
        v2.train("QGA", nxt, init, seed=12, out=out, ckpt_every=50, resume_from=done)
        done = nxt
    print(name, done, flush=True)
