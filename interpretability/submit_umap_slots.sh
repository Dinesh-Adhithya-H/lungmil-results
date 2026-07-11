#!/usr/bin/env bash
#SBATCH --job-name=umap_slots
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=80G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=02:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_umap_slots.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_umap_slots.err
#SBATCH --mail-type=NONE

set -uo pipefail
export PYTHONUNBUFFERED=1

HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
RESULTS_DIR="${HOME_MIL}/results/mm_abmil_v8"

SPLIT="${SPLIT:-1}"
FOLD="${FOLD:-0}"
P2_TAG="${P2_TAG:-alt_shared_combined}"
SLOT_K="${SLOT_K:-128}"
SPLIT_SET="${SPLIT_SET:-test}"
N_PAT="${N_PAT:-80}"
METHOD="${METHOD:-umap}"

OUT_DIR="${HOME_MIL}/interpretability/slot_shared_s${SPLIT}f${FOLD}_${P2_TAG}/umap"

mkdir -p "${HOME_MIL}/results_mm_abmil_v8/slurm_logs"

echo "==============================="
echo " umap_slots  tag=${P2_TAG}"
echo " method: ${METHOD}  n_patients: ${N_PAT}"
echo " out: ${OUT_DIR}"
echo "==============================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "Python: $(which python3)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

python3 -u "${HOME_MIL}/interpretability/umap_slots.py" \
    --split          "${SPLIT}"      \
    --fold           "${FOLD}"       \
    --p2-tag         "${P2_TAG}"     \
    --slot-k         "${SLOT_K}"     \
    --split-set      "${SPLIT_SET}"  \
    --n-patients     "${N_PAT}"      \
    --method         "${METHOD}"     \
    --out-dir        "${OUT_DIR}"    \
    --samples-dir    "${SAMPLES}"    \
    --splits-csv     "${SPLITS_CSV}" \
    --results-dir    "${RESULTS_DIR}"

echo "Done."
