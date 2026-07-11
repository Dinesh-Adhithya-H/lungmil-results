#!/usr/bin/env bash
# run_baselines.sh — run all classical + unimodal + multimodal baselines
# across all 4 folds and save a summary JSON.
#
#SBATCH --job-name=baselines
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=220G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_baselines.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_baselines.err
#SBATCH --mail-type=NONE

set -uo pipefail
export PYTHONUNBUFFERED=1

HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
OUT_DIR="${HOME_MIL}/results/mm_abmil_v8"
PROPS_CSV="${OUT_DIR}/cluster_proportions.csv"
SPLIT="${SPLIT:-1}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "Python: $(which python3)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

python3 "${HOME_MIL}/scripts/run_baselines.py" \
    --samples-dir  "${SAMPLES}"    \
    --splits-csv   "${SPLITS_CSV}" \
    --split        "${SPLIT}"      \
    --out-dir      "${OUT_DIR}"    \
    --props-csv    "${PROPS_CSV}"
