#!/usr/bin/env bash
# submit_slot_collapse_check.sh — Diagnose PMA/slot attention collapse on real data
#
# Generates feature_space_journey.png, routing_by_class.png, slot_alignment.png
#
# Usage (new PMA models — mario_kempes_mt, set_mil_mt, longitudinal_mk_mt):
#   sbatch --export=VARIANT=mario_kempes_mt,SPLIT=0,FOLD=0 \
#          interpretability/submit_slot_collapse_check.sh
#
# Usage (old slot model):
#   sbatch --export=P2_TAG=alt_shared_comp,FOLD=1 \
#          interpretability/submit_slot_collapse_check.sh
#
#SBATCH --job-name=slot_collapse
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=100G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=02:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_slot_collapse.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_slot_collapse.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -uo pipefail
export PYTHONUNBUFFERED=1

VARIANT="${VARIANT:-}"          # e.g. mario_kempes_mt — if set, uses new model path
P2_TAG="${P2_TAG:-alt_shared_comp}"   # for old slot models only
FOLD="${FOLD:-0}"
SPLIT="${SPLIT:-0}"
SPLIT_SET="${SPLIT_SET:-test}"
N_PATIENTS="${N_PATIENTS:-100}"
SLOT_K="${SLOT_K:-16}"
TASK="${TASK:-acr_cls}"
CLUSTER_CSV="${CLUSTER_CSV:-}"  # optional path to stem,cluster CSV
WANDB_PROJECT="${WANDB_PROJECT:-chicago-mil}"

HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
LOG_DIR="${HOME_MIL}/results_mm_abmil_v8/slurm_logs"
mkdir -p "${LOG_DIR}"

echo "======================================="
if [ -n "${VARIANT}" ]; then
    echo " slot_collapse_check (new PMA model)"
    echo " variant=${VARIANT}  split=${SPLIT}  fold=${FOLD}"
else
    echo " slot_collapse_check (old slot model)"
    echo " p2-tag=${P2_TAG}  split=${SPLIT}  fold=${FOLD}"
fi
echo " job=${SLURM_JOB_ID}"
echo "======================================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "Python: $(which python3)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

EXTRA_ARGS=""
if [ -n "${VARIANT}" ]; then
    EXTRA_ARGS="--variant ${VARIANT} --slot-k ${SLOT_K}"
else
    EXTRA_ARGS="--p2-tag ${P2_TAG}"
fi
if [ -n "${CLUSTER_CSV}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --cluster-csv ${CLUSTER_CSV}"
fi

python3 -u "${HOME_MIL}/interpretability/slot_collapse_check.py" \
    --split        "${SPLIT}"       \
    --fold         "${FOLD}"        \
    --split-set    "${SPLIT_SET}"   \
    --n-patients   "${N_PATIENTS}"  \
    --mods         HE BAL CT Clinical \
    --task         "${TASK}"        \
    --max-patches  200              \
    --results-dir  "${HOME_MIL}/results/mm_abmil_v8" \
    --wandb-project "${WANDB_PROJECT}" \
    ${EXTRA_ARGS}

echo "Done."
