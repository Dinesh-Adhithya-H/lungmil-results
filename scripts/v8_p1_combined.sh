#!/usr/bin/env bash
# v8_p1_combined.sh — Combined train+val for unimodal Phase 1 baselines
#
# Requires: all 4 per-fold HP sweeps must exist for each task×modality.
# Selects best HP globally (mean val_metric across 4 folds), trains on
# fold 0 train+val with CV-estimated epoch count, evaluates on test.
#
# Saves to {p1_dir}/final_combined/ to avoid overwriting per-fold models.
#
# Usage:
#   sbatch scripts/v8_p1_combined.sh
#
#SBATCH --job-name=p1_combined
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=180G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_p1_combined.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_p1_combined.err
#SBATCH --mail-type=NONE

set -uo pipefail
export PYTHONUNBUFFERED=1

HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
OUT_DIR="${HOME_MIL}/results/mm_abmil_v8"
LOG_DIR="${HOME_MIL}/results_mm_abmil_v8/slurm_logs"
SCRIPT_PATH="$(realpath "$0")"
SPLIT=1
FOLD=0

mkdir -p "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "Python: $(which python3)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

RESUBMIT=0
PY_PID=""

walltime_handler() {
    echo "[SIGUSR1] Wall-time approaching. Stopping and resubmitting..."
    RESUBMIT=1
    [[ -n "${PY_PID}" ]] && kill -SIGTERM "${PY_PID}" 2>/dev/null || true
}
trap walltime_handler SIGUSR1

run_task() {
    local task="$1"
    echo ""
    echo "==============================="
    echo " Phase 1 combined  task=${task}  fold=${FOLD}"
    echo "==============================="

    python3 -u "${HOME_MIL}/train_mm_abmil_v8.py" \
        --samples-dir      "${SAMPLES}"    \
        --splits-csv       "${SPLITS_CSV}" \
        --split            "${SPLIT}"      \
        --fold             "${FOLD}"       \
        --out-dir          "${OUT_DIR}"    \
        --phase            p1              \
        --p1-tasks         "${task}"       \
        --p1-global-hp                     \
        --p1-combined-tag  combined        \
        --max-he-patches   99999           \
        &
    PY_PID=$!
    wait "${PY_PID}"
    local exit_code=$?
    PY_PID=""

    if [[ "${RESUBMIT}" -eq 1 ]]; then
        echo "Wall-time hit during task=${task}. Resubmitting..."
        sbatch "${SCRIPT_PATH}"
        exit 0
    fi

    [[ "${exit_code}" -ne 0 ]] && echo "ERROR: task=${task} exited with ${exit_code}"
}

# Run all 4 tasks (each trains all 4 modalities)
for task in acr acr_surv clad death; do
    run_task "${task}"
    [[ "${RESUBMIT}" -eq 1 ]] && exit 0
done

echo "Done."
