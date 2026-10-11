# Meta-Reinforcement Learning for Adaptive Parameter Control in Evolutionary Algorithms

Code, trained policies and run-level results for the paper

> H. Razavi, M. Hajiaghaei-Keshteli, F. Gholian-Jouybari, H. Bonakdari,
> *Meta-Reinforcement Learning for Adaptive Parameter Control in Evolutionary Algorithms*,
> submitted to the 21st IFAC Symposium on System Identification (SYSID 2027).

A tabular Q-learning controller is trained **offline** across many randomly generated
optimization problems and then **frozen**. It adjusts one or two parameters of a base
optimizer (GA, PSO, DE or a quantum-inspired GA) every five generations, using a
scale-independent state, multiplicative parameter updates and a log-error reward. The
frozen policies are evaluated on CEC2017-style (validation) and CEC2022-style (test)
functions and on Wiener–Hammerstein output-error identification.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Python >= 3.10 (tested with 3.13)
pip install -r requirements.txt
bash run_all.sh check      # ~1 min: re-runs stored runs and checks they match bit for bit
bash run_all.sh analysis   # ~2 min: all tables and figures from the stored results
```

`analysis` writes the table values to `logs/` and the figures to `figures/`.
`bash run_all.sh full` reruns the complete study from scratch (tuning, meta-training,
selection, testing, ablations, identification). It takes many CPU-hours and
**overwrites** the shipped policies and results; work on a copy if you want to keep them.

## Where each result comes from

| Paper item | Script | Input → output |
|---|---|---|
| Tables 1–2, 28 % median reduction, Random-M2 comparison, D = 30 counts | `stats_test.py` | `results_main.json`, `v2/final_*.json`, `exp2_*.json`, `randm2/*.json` → `stats_test.json` |
| Validation results (Sec. 6.2) | `val_stats.py` | `val_QGA_DE_PSO_GA_10.json` → `val_stats.json` |
| Table 4 and post-hoc tests | `stats_ablation.py` | `abl_abl.json`, `abl_ckpt.json` → `ablation_stats.json` |
| Tables 3 and 5 | `stats_sysid.py` | `sysid2_results.json` → `sysid_stats.json` |
| Fig. 1 | — | flowchart, part of the LaTeX source |
| Figs. 2–5 | `make_figures.py` | the files above, `conv.json`, `policy_counts.json` → `figures/` |

## Experiment scripts (in the order of `run_all.sh full`)

| Step | Script | Produces |
|---|---|---|
| Tune static start **k**₀ on training functions | `v2_tune.py` | `v2/static_init.json` |
| Meta-train M1 (one parameter, from defaults) | `meta_rl.py` | `policies/*_Q.npz` |
| Meta-train M2 (two parameters, from **k**₀; plain and control-variate runs) | `meta_rl2.py` | `v2/<ALG>_ep*.npy`, `v2/<ALG>_cv_ep*.npy` |
| Select a checkpoint on the CEC2017-style validation suite | `v2_validate.py`, `v2_validate_extra.py` | `v2/selected*.json`, `v2/final_choice.json` |
| Test: Default, RELEASE-E, Random-M1, M1, jDE (D = 10, 20) | `test_meta.py` | `results_main.json` |
| Test: Tuned static and selected controller MRL (D = 10, 20) | `v2_final_test.py` | `v2/final_*.json` |
| Test: unseen D = 30; SHADE, LDIW-PSO | `exp2.py` | `exp2_*.json` |
| Matched random baseline Random-M2 | `random_m2.py` | `randm2/*.json` |
| Validation results of the selected controllers (10 runs) | `val_report.py` | `val_QGA_DE_PSO_GA_10.json` |
| QGA ablations and training-budget curve | `ablation_dev.py`, `ablation_eval.py` | `v2/abl_*_ep1500.npy`, `abl_abl.json`, `abl_ckpt.json` |
| Convergence curves (QGA, D = 20) | `conv.py` | `conv.json` |
| Action shares of the frozen QGA policy | `policy_actions.py` | `policy_counts.json` |
| Wiener–Hammerstein identification (S1, S2; 20/30/40 dB; 25 runs) | `sysid2.py` (uses `sysid.py`) | `sysid2_results.json` |

The selected controllers (MRL) are listed in `v2/final_choice.json`:
M2 checkpoints for QGA (`v2/QGA_ep2500.npy`), DE (`v2/DE_ep500.npy`) and PSO
(`v2/PSO_ep1500.npy`), and the M1 policy for GA (`policies/GA_Q.npz`). Training and
selection logs are in `v2/*.log`.

## Code layout

| File / folder | Content |
|---|---|
| `algorithms/` | base optimizers (GA, PSO, DE, QGA) and the per-run Q-learning baseline RELEASE-E |
| `benchmarks/cec2022.py`, `cec2017_val.py`, `base_functions.py` | CEC2022-style test and CEC2017-style validation suites, shared base functions |
| `meta_rl.py` | M1 controller: state encoding, environment, offline Q-learning |
| `meta_rl2.py` | M2 controller, training-function generator, parameter bounds (`KNOBS`) |
| `baselines_extra.py`, `test_meta.py` (`run_jde`) | SHADE, LDIW-PSO and jDE baselines |
| `stats_utils.py` | Holm correction, Vargha–Delaney A12, paired Wilcoxon test |
| `smoke_test.py` | determinism check against the stored results |

## Reproducibility notes

* **Seeds.** Test run *r* on function *f* in dimension *D* uses seed `1000 f + 10 r + D`
  for every method, so all methods share the initial population and runs are paired.
  Validation runs use `424242 + 100 f + s`; identification runs use
  `9000 + 37 r + 5000·[S2] + SNR`.
* **Benchmarks.** The CEC2017- and CEC2022-style suites follow the published function
  definitions, search range and biases, but use self-generated, fixed-seed shift vectors,
  rotation matrices and shuffle indices (the official data files are not used). Errors are
  therefore not directly comparable with official competition results.
* **Budget.** Population 30 and 2000·D function evaluations for every optimizer.
* **Determinism.** Results are bit-identical for a given NumPy version (generated with
  Python 3.13, NumPy 2.5, SciPy 1.18). Other versions can change random streams slightly;
  the statistics then agree up to sampling noise.
* **Parallelism.** Scripts use `multiprocessing.Pool(2)`; each run is single-threaded,
  so the number of workers does not affect results.
* **Naming in result files.** `Random-nudge` = Random-M1, `Meta-RL` = M1,
  `Meta-RL-final` = selected controller MRL, `Static-tuned` = tuned static **k**₀.

## License

MIT, see `LICENSE`. If you use this code, please cite the paper (`CITATION.cff`).
