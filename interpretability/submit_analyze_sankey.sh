#!/usr/bin/env bash
# submit_analyze_sankey.sh — per-modality Sankeys + diagnostics for SharedSlotMIL
#
# Default: combined model (alt_shared_combined, trained on train+val with global HP).
# Override with:  sbatch --export=P2_TAG=alt_shared interpretability/submit_analyze_sankey.sh
#
#SBATCH --job-name=sankey_clean
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=120G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=04:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_sankey_clean.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_sankey_clean.err
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
P2_TAG="${P2_TAG:-alt_shared_combined}"
SLOT_K="${SLOT_K:-128}"
SPLIT_SET="${SPLIT_SET:-test}"
TOP_K_SLOTS="${TOP_K_SLOTS:-12}"
TOP_K_CLUST="${TOP_K_CLUST:-15}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"

OUT_DIR="${HOME_MIL}/interpretability/slot_shared_s${SPLIT}f${FOLD}_${P2_TAG}"

mkdir -p "${LOG_DIR}"

echo "==============================="
echo " sankey_clean  tag=${P2_TAG}"
echo " split-set: ${SPLIT_SET}"
echo " NPZ dir:   ${OUT_DIR}/npy"
echo " out dir:   ${OUT_DIR}"
echo "==============================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "Python: $(which python3)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

EXTRA_ARGS=""
[[ "${SKIP_EXTRACT}" == "1" ]] && EXTRA_ARGS="--skip-extract"

python3 -u "${HOME_MIL}/interpretability/analyze_sankey_clean.py" \
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
    --top-k-clusters "${TOP_K_CLUST}" \
    ${EXTRA_ARGS}

echo "Done."
