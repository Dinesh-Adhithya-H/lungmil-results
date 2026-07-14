#!/bin/bash
# submit_lung_benchmark.sh
# ---------------------------------------------------------------------------
# MCAT (path mode) + MOTCAT (path mode) on lung transplant data.
# HE patch features (1024-dim), 3 survival tasks × 5 splits = 30 jobs per model.
#
# Prerequisites: run prep_lung_benchmark.sh first to create CSV/splits/pt_files.
#
# Array layout (60 jobs):
#   Model MCAT path  (0-14):
#     0-4:   lung_acr    folds 0-4
#     5-9:   lung_clad   folds 0-4
#     10-14: lung_death  folds 0-4
#   Model MOTCAT path (15-29):
#     15-19: lung_acr    folds 0-4
#     20-24: lung_clad   folds 0-4
#     25-29: lung_death  folds 0-4
#
# Submit: sbatch benchmarks/submit_lung_benchmark.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=lung_bench
#SBATCH --array=0-29
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%A_%a_lung_bench.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%A_%a_lung_bench.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -euo pipefail
export PYTHONUNBUFFERED=1
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

REPO="/ictstr01/home/aih/dinesh.haridoss/chicago_mil"
MIL_DIR="/lustre/groups/aih/dinesh.haridoss/mil"
MCAT_DIR="${MIL_DIR}/MCAT"
MOTCAT_DIR="${MIL_DIR}/MOTCAT"
DATA_DIR="${MIL_DIR}/lung_mcat_data"
RESULTS_BASE="${REPO}/results_lung_competitors"
mkdir -p "${RESULTS_BASE}"

TASKS=(lung_acr lung_clad lung_death)

IDX=$SLURM_ARRAY_TASK_ID

# Determine model and task+fold from array index
# 0-14: MCAT  (3 tasks × 5 folds)
# 15-29: MOTCAT (3 tasks × 5 folds)
if [ $IDX -lt 15 ]; then
    MODEL="mcat"
    LOCAL_IDX=$IDX
else
    MODEL="motcat"
    LOCAL_IDX=$((IDX - 15))
fi

TASK_IDX=$((LOCAL_IDX / 5))
FOLD=$((LOCAL_IDX % 5))
TASK=${TASKS[$TASK_IDX]}

echo "========================================"
echo "Model: ${MODEL}  Task: ${TASK}  Fold: ${FOLD}"
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started: $(date)"
echo "========================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
python -m pip install -q scikit-survival tensorboardX pot 2>/dev/null || true

RESULTS_DIR="${RESULTS_BASE}/${MODEL}_${TASK}_fold${FOLD}"
mkdir -p "${RESULTS_DIR}"

if [ "${MODEL}" = "mcat" ]; then
    cd "${MCAT_DIR}"
    python main.py \
        --direct_data_dir "${DATA_DIR}" \
        --direct_csv_path "${DATA_DIR}/${TASK}_all_clean.csv" \
        --split_dir "${TASK}" \
        --which_splits "5foldcv" \
        --model_type "mcat" \
        --mode "path" \
        --wsi_input_dim 1024 \
        --k_start ${FOLD} \
        --k_end $((FOLD + 1)) \
        --results_dir "${RESULTS_DIR}" \
        --max_epochs 50 \
        --val_patience 10 \
        --lr 2e-4 \
        --bag_loss nll_surv \
        --reg 1e-5 \
        --drop_out \
        --weighted_sample \
        --seed 1 \
        --overwrite

else  # motcat
    cd "${MOTCAT_DIR}"
    python main.py \
        --data_root_dir "${DATA_DIR}" \
        --split_dir "${TASK}" \
        --which_splits "5foldcv" \
        --model_type "motcat" \
        --mode "path" \
        --path_input_dim 1024 \
        --k_start ${FOLD} \
        --k_end $((FOLD + 1)) \
        --results_dir "${RESULTS_DIR}" \
        --max_epochs 50 \
        --lr 2e-4 \
        --bag_loss nll_surv \
        --reg 1e-5 \
        --drop_out \
        --weighted_sample \
        --ot_impl pot-uot-l2 \
        --ot_reg 0.1 \
        --ot_tau 0.5 \
        --seed 1 \
        --overwrite
fi

echo "========================================"
echo "${MODEL} ${TASK} fold${FOLD} done  $(date)"
echo "========================================"
