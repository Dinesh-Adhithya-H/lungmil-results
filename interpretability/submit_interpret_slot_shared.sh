#!/usr/bin/env bash
# submit_interpret_slot_shared.sh — SharedSlotMIL interpretability
#
# Runs extraction + all visualizations (alpha bars, slot×cluster heatmaps,
# cluster→task routing, Sankey alluvial) for a trained SharedSlotMIL.
#
# Usage:
#   sbatch interpretability/submit_interpret_slot_shared.sh        # combined model
#   sbatch --export=P2_TAG=shared interpretability/submit_interpret_slot_shared.sh
#
#SBATCH --job-name=interp_slot
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=120G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=04:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_interp_slot.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_interp_slot.err
#SBATCH --mail-type=NONE

set -uo pipefail
export PYTHONUNBUFFERED=1

HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
RESULTS_DIR="${HOME_MIL}/results/mm_abmil_v8"
LOG_DIR="${HOME_MIL}/results_mm_abmil_v8/slurm_logs"

SPLIT="${SPLIT:-1}"
FOLD="${FOLD:-0}"
P2_TAG="${P2_TAG:-alt_shared}"
SLOT_K="${SLOT_K:-128}"
SPLIT_SET="${SPLIT_SET:-test}"
TOP_K_SLOTS="${TOP_K_SLOTS:-20}"
TOP_K_CLUST="${TOP_K_CLUST:-30}"

OUT_DIR="${HOME_MIL}/interpretability/slot_shared_s${SPLIT}f${FOLD}_${P2_TAG}"

mkdir -p "${LOG_DIR}"

echo "==============================="
echo " interp_slot  split=${SPLIT}  fold=${FOLD}  tag=${P2_TAG}"
echo " split-set: ${SPLIT_SET}"
echo " out: ${OUT_DIR}"
echo "==============================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "Python: $(which python3)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

python3 -u "${HOME_MIL}/interpretability/interpret_slot_shared.py" \
    --split          "${SPLIT}"       \
    --fold           "${FOLD}"        \
    --p2-tag         "${P2_TAG}"      \
    --slot-k         "${SLOT_K}"      \
    --split-set      "${SPLIT_SET}"   \
    --out-dir        "${OUT_DIR}"     \
    --samples-dir    "${SAMPLES}"     \
    --splits-csv     "${SPLITS_CSV}"  \
    --results-dir    "${RESULTS_DIR}" \
    --top-k-slots    "${TOP_K_SLOTS}" \
    --top-k-clust    "${TOP_K_CLUST}" \
    --top-k-alpha    32

echo "Done."
