#!/usr/bin/env bash
# v8_slot_interpret.sh — SharedSlotMIL interpretability + Sankey plots
#
# Runs interpret_shared_slot_mil.py on the test split using the trained model.
# Defaults to slot_mega_alt_shared (per-fold model, fold 0).
# Pass MODEL_TAG=shared_combined to use the combined model instead.
#
# Usage:
#   sbatch scripts/v8_slot_interpret.sh
#   sbatch --export=MODEL_TAG=shared_combined scripts/v8_slot_interpret.sh
#
#SBATCH --job-name=slot_interp
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=120G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=04:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_slot_interp.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_slot_interp.err
#SBATCH --mail-type=NONE

set -uo pipefail
export PYTHONUNBUFFERED=1

MODEL_TAG="${MODEL_TAG:-alt_shared}"    # alt_shared | shared_combined
FOLD=0
SPLIT=1

HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
OUT_DIR="${HOME_MIL}/results/mm_abmil_v8"
LOG_DIR="${HOME_MIL}/results_mm_abmil_v8/slurm_logs"

MODEL_DIR="${OUT_DIR}/phase2/split${SPLIT}_fold${FOLD}/slot_mega_${MODEL_TAG}"
INTERP_OUT="${OUT_DIR}/interpretability/shared_slot_${MODEL_TAG}/split${SPLIT}_fold${FOLD}"

mkdir -p "${LOG_DIR}"

echo "==============================="
echo " slot_interpret  fold=${FOLD}  tag=${MODEL_TAG}"
echo " model_dir: ${MODEL_DIR}"
echo " out_dir:   ${INTERP_OUT}"
echo "==============================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "Python: $(which python3)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

python3 -u "${HOME_MIL}/interpretability/interpret_shared_slot_mil.py" \
    --model-dir    "${MODEL_DIR}"   \
    --samples-dir  "${SAMPLES}"     \
    --splits-csv   "${SPLITS_CSV}"  \
    --split        "${SPLIT}"       \
    --fold         "${FOLD}"        \
    --split-set    test             \
    --out-dir      "${INTERP_OUT}"  \
    --slot-k       128              \
    --top-slots    32

echo "Done."
