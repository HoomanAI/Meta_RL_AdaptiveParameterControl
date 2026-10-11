#!/usr/bin/env bash
# Reproduction pipeline.
#   bash run_all.sh check     ~1 min   re-run a few stored runs and compare (determinism check)
#   bash run_all.sh analysis  ~2 min   all tables and figures from the stored run-level results
#   bash run_all.sh full      days     retrain and re-run every experiment (overwrites stored files)
set -euo pipefail
cd "$(dirname "$0")"
stage="${1:-analysis}"

analysis () {
  python stats_test.py      | tee logs/tables_1_2.txt      # Tables 1-2 (+ D=30 counts)
  python val_stats.py       | tee logs/validation.txt      # validation summary (Sec. 6.2)
  python stats_ablation.py  | tee logs/table_4.txt         # Table 4 + post-hoc tests
  python stats_sysid.py     | tee logs/tables_3_5.txt      # Tables 3 and 5
  python make_figures.py                                   # Figs. 2-5 -> figures/
}

mkdir -p logs
case "$stage" in
  check)    python smoke_test.py ;;
  analysis) analysis ;;
  full)
    # 1. static start k0 on training functions (QGA, GA: log grid; DE, PSO: grid around defaults)
    python v2_tune.py QGA,PSO,DE,GA 40
    python v2_tune.py DE,PSO 40 DE,PSO
    # 2. meta-training: M1 (one parameter) and M2 (two parameters, plain and control-variate runs)
    for a in QGA DE PSO; do python meta_rl.py $a 3000; done; python meta_rl.py GA 2000
    for a in QGA DE PSO; do python meta_rl2.py $a 3000; done
    python meta_rl2.py PSOb 2500; python meta_rl2.py GA 2000
    python meta_rl2.py QGA 1500 cv; for a in DE PSO GA; do python meta_rl2.py $a 2000 cv; done
    # 3. checkpoint selection on the CEC2017-style validation suite (D=10, 3 runs)
    python v2_validate.py QGA,DE,PSO,GA,PSOb
    python v2_validate.py QGA,DE,PSO,GA cv
    python v2_validate_extra.py QGA,DE,PSO,GA          # M1 policies and averaged Q-tables
    echo ">> write the chosen candidate per optimizer to v2/final_choice.json (most validation wins)"
    # 4. one-shot test on the CEC2022-style suite (D=10,20 and unseen D=30; 15 runs)
    python test_meta.py QGA,DE,GA,PSO 10,20 15 main      # Default, RELEASE-E, Random-M1, M1, jDE
    python v2_final_test.py QGA,DE,PSO; python v2_final_test.py GA   # Tuned static, MRL
    python exp2.py d30 QGA,DE,PSO,GA; python exp2.py newcomp DE,PSO   # D=30; SHADE, LDIW-PSO
    python random_m2.py QGA,DE,PSO                      # matched random baseline
    # 5. validation results of the selected controllers (10 runs)
    python val_report.py QGA,DE,PSO,GA 10
    # 6. ablations (1500 episodes each) and training-budget curve
    for v in full no-phase no-stag no-div no-knob rel-reward default-start; do python ablation_dev.py $v 1500 1e9; done
    python ablation_eval.py abl; python ablation_eval.py ckpt
    # 7. convergence curves, learned behaviour, identification
    python conv.py; python policy_actions.py; python sysid2.py
    analysis ;;
  *) echo "usage: bash run_all.sh [check|analysis|full]"; exit 1 ;;
esac
