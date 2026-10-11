"""Figures 2-5 of the paper, written to figures/ (PDF and PNG).
Requires stats_test.py (stats_test.json), conv.py (conv.json), policy_actions.py
(policy_counts.json) and sysid2.py (sysid2_results.json) to have been run."""
import json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from benchmarks.cec2022 import _BIAS

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9})
BLUE, ORANGE, GOLD, GREY = "#2a78d6", "#eb6834", "#f0a30a", "#8c8c8c"


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"figures/{name}.pdf"); fig.savefig(f"figures/{name}.png", dpi=200); plt.close(fig)


# ---------- Fig. 2: per-case log10(mean error Default / mean error MRL) ----------
V = defaultdict(lambda: defaultdict(dict))
def load(fn, tm=False):
    for row in json.load(open(fn)):
        alg, fid, d, run, out = row[:5]
        for m, x in out.items():
            V[(alg, d, fid)][m][run] = x
load("results_main.json"); load("v2/final_QGA_DE_PSO.json"); load("v2/final_GA.json")
for fn in os.listdir("."):
    if fn.startswith("exp2_") and fn.endswith(".json"):
        load(fn)
st = json.load(open("stats_test.json"))
err = lambda a, d, f, m: np.maximum(np.array([V[(a, d, f)][m][r] for r in range(15)]) - _BIAS[f], 1e-8).mean()
rows, sig, labels = [], [], []
for a in ["QGA", "DE", "GA", "PSO"]:
    for d in (10, 20, 30):
        rows.append([np.log10(err(a, d, f, "Default") / err(a, d, f, "Meta-RL-final")) for f in range(1, 13)])
        sig.append([st[a]["vs_default_sig"][f"{d}_{f}"] for f in range(1, 13)])
        labels.append(f"{a}  D={d}")
M = np.clip(np.array(rows), -2, 2)
cmap = LinearSegmentedColormap.from_list("rb", ["#b2182b", "#e8a0a0", "#efede8", "#a6c4e8", "#1f5aa8"])
fig, ax = plt.subplots(figsize=(5.0, 3.4))
im = ax.imshow(M, cmap=cmap, vmin=-2, vmax=2, aspect="auto")
for i in range(M.shape[0]):
    for j in range(12):
        if sig[i][j]:
            ax.plot(j, i, "o", ms=2.5, color="white" if abs(M[i, j]) > 1.2 else "black")
for y in (2.5, 5.5, 8.5):
    ax.axhline(y, color="white", lw=2)
ax.set_xticks(range(12)); ax.set_xticklabels([f"F{k}" for k in range(1, 13)])
ax.set_yticks(range(12)); ax.set_yticklabels(labels); ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
cb.set_label(r"$\log_{10}$(error of default / error of MRL)", fontsize=8)
save(fig, "fig_heat")

# ---------- Fig. 3: convergence, QGA family, D=20 ----------
C = json.load(open("conv.json")); G = np.array(C["grid"]) / 1e3
cur = defaultdict(list)
for fid, run, v, c in C["res"]:
    cur[(fid, v)].append(c)
sty = {"QGA": ("Default QGA", GREY, ":"), "RELEASE-E": ("RELEASE-E", GOLD, "-."),
       "Static": ("Tuned static", ORANGE, "--"), "M2": ("Meta-RL (M2)", BLUE, "-")}
fig, axs = plt.subplots(1, 2, figsize=(3.6, 2.0))
for ax, (fid, title) in zip(axs, [(1, "F1 (Zakharov)"), (6, "F6 (Hybrid 1)")]):
    for v, (lab, col, ls) in sty.items():
        Y = np.maximum(np.array(cur[(fid, v)]), 1e-8)
        q1, med, q3 = np.percentile(Y, [25, 50, 75], axis=0)
        ax.plot(G, med, ls, color=col, lw=1.4, label=lab); ax.fill_between(G, q1, q3, color=col, alpha=0.15, lw=0)
    ax.set_yscale("log"); ax.set_title(title, fontsize=9); ax.set_xlabel(r"Evaluations ($\times10^3$)")
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel(r"$J-J^\star$")
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=6.5, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.84)); fig.savefig("figures/fig3.pdf"); fig.savefig("figures/fig3.png", dpi=200); plt.close(fig)

# ---------- Fig. 4: action shares of the frozen QGA policy ----------
P = json.load(open("policy_counts.json")); Cn = np.array(P["C"])
acts = ["keep", r"$\Delta\theta\times0.7$", r"$\Delta\theta/0.7$", r"$p_m\times0.7$", r"$p_m/0.7$"]
cols = ["#cfcfcf", BLUE, "#9cc3ef", ORANGE, "#f5b89c"]
fig, axs = plt.subplots(1, 3, figsize=(3.6, 2.2), sharey=True)
for k, (ax, title) in enumerate(zip(axs, ["improving", "stalled 1-2", r"stalled $\geq$3"])):
    sh = Cn[:, k, :] / np.maximum(Cn[:, k, :].sum(1, keepdims=True), 1)
    bot = np.zeros(4)
    for a in range(5):
        ax.bar(np.arange(1, 5), sh[:, a], bottom=bot, color=cols[a], width=0.7, label=acts[a]); bot += sh[:, a]
    ax.set_title(title, fontsize=8); ax.set_xticks(range(1, 5)); ax.set_xlabel("budget quarter", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("share of actions")
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=5, fontsize=5.5, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig("figures/fig_actions.pdf"); fig.savefig("figures/fig_actions.png", dpi=200); plt.close(fig)

# ---------- Fig. 5: J/J0 distribution at 30 dB ----------
S = defaultdict(list); J0 = {}
for s, snr, run, alg, j, perr, floor in json.load(open("sysid2_results.json")):
    if snr == 30:
        S[(s, alg)].append(j); J0[s] = floor
algs = ["QGA", "RELEASE-QGA-E", "QGA-static", "QGA-MetaRL", "DE", "jDE", "SHADE", "PSO", "LDIW-PSO"]
names = ["QGA", "RELEASE-E", "Static", "Meta-RL", "DE", "jDE", "SHADE", "PSO", "LDIW"]
fig, axs = plt.subplots(1, 2, figsize=(3.4, 2.6), sharey=True)
for ax, s in zip(axs, ["S1", "S2"]):
    bp = ax.boxplot([np.array(S[(s, a)]) / J0[s] for a in algs], patch_artist=True, widths=0.6,
                    medianprops=dict(color="black"), flierprops=dict(ms=2, markeredgecolor=GREY))
    for b, a in zip(bp["boxes"], algs):
        b.set_facecolor(ORANGE if a == "QGA-static" else BLUE if a == "QGA-MetaRL" else "#d9d9d9")
    ax.axhline(1, ls="--", color=GREY, lw=0.8); ax.set_yscale("log"); ax.set_title(f"{s}, 30 dB", fontsize=9)
    ax.set_xticks(range(1, 10)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel(r"$J/J_0$")
save(fig, "fig_sysid")
print("figures written to figures/")
