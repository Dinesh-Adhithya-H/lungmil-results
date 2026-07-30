#!/bin/bash
# Submit interpretability for best model per task, all 5 splits.
#
# Best models per task (from benchmark):
#   ACR cls   → set_mil_mt_no_sab  (BACC 0.623)
#   ACR surv  → longitudinal_mk_no_alibi  (CI 0.679)  [already done]
#   CLAD      → set_mil_mt         (CI 0.563)
#   Death     → longitudinal_mk_no_alibi  (CI 0.771)  [already done]
#
# This script submits only the SetMIL jobs (ACR cls + CLAD) for splits missing results.

set -euo pipefail
REPO=/ictstr01/home/aih/dinesh.haridoss/chicago_mil
INTERP_ROOT=${REPO}/interpretability/set_mil_mt_interp

# ── ACR cls: set_mil_mt_no_sab ────────────────────────────────────────────────
for SPLIT in 0 1 2 3 4; do
    OUT_DIR="${INTERP_ROOT}/set_mil_mt_no_sab_split${SPLIT}_fold0_cls"
    if ls "${OUT_DIR}"/Lpop_K_*.png 2>/dev/null | grep -q .; then
        echo "[SKIP] set_mil_mt_no_sab split${SPLIT} cls — already done"
        continue
    fi
    echo "[SUBMIT] set_mil_mt_no_sab split${SPLIT} acr_cls"
    sbatch --job-name=interp_nosab_cls \
        --partition=gpu_p --qos=gpu_normal --gres=gpu:1 \
        --cpus-per-task=8 --mem=60G --time=04:00:00 \
        --output="${REPO}/results/mm_abmil_v8/slurm_logs/%j_interp_nosab_cls.out" \
        --error="${REPO}/results/mm_abmil_v8/slurm_logs/%j_interp_nosab_cls.err" \
        --wrap="cd ${REPO} && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
            export PYTHONUNBUFFERED=1 && \
            conda run -n chicago python interpretability/interpret_set_mil_mt.py \
                --split ${SPLIT} --fold 0 --variant cls \
                --model-variant set_mil_mt_no_sab \
                --wandb-project chicago-mil-interpretability"
done

# ── CLAD: set_mil_mt ──────────────────────────────────────────────────────────
for SPLIT in 0 1 2 3 4; do
    OUT_DIR="${INTERP_ROOT}/set_mil_mt_split${SPLIT}_fold0_clad_surv"
    if ls "${OUT_DIR}"/Lpop_K_*.png 2>/dev/null | grep -q .; then
        echo "[SKIP] set_mil_mt split${SPLIT} clad_surv — already done"
        continue
    fi
    echo "[SUBMIT] set_mil_mt split${SPLIT} clad_surv"
    sbatch --job-name=interp_smt_clad \
        --partition=gpu_p --qos=gpu_normal --gres=gpu:1 \
        --cpus-per-task=8 --mem=60G --time=04:00:00 \
        --output="${REPO}/results/mm_abmil_v8/slurm_logs/%j_interp_smt_clad.out" \
        --error="${REPO}/results/mm_abmil_v8/slurm_logs/%j_interp_smt_clad.err" \
        --wrap="cd ${REPO} && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
            export PYTHONUNBUFFERED=1 && \
            conda run -n chicago python interpretability/interpret_set_mil_mt.py \
                --split ${SPLIT} --fold 0 --variant clad_surv \
                --wandb-project chicago-mil-interpretability"
done
