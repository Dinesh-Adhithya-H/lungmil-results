#!/usr/bin/env bash
# v8_baseline_combined.sh — Combined train+val for early/late/middle fusion baselines
#
# Requires: all 4 per-fold HP sweeps must exist for each variant×task.
# Selects best HP globally (mean val_bacc across 4 folds), trains on
# fold 0 train+val, evaluates on test.
#
# One job per variant×task (12 total). Set VARIANT and TASK via --export.
# Default runs all 12 sequentially in one job.
#
# Usage:
#   # All baselines in one job (sequential):
#   sbatch scripts/v8_baseline_combined.sh
#
#   # Single variant×task:
#   sbatch --export=VARIANT=middle,TASK=cls scripts/v8_baseline_combined.sh
#
#SBATCH --job-name=base_combined
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=180G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_base_combined.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_base_combined.err
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

run_one() {
    local variant="$1" task="$2"
    local save_dir="${OUT_DIR}/phase2/split${SPLIT}_fold${FOLD}/${variant}_${task}"
    local status_file="${save_dir}/status_${variant}_final.json"

    # Skip if already done
    if python3 -c "
import json,sys,os
f=os.environ.get('STATUS_FILE','')
if not f or not os.path.exists(f): sys.exit(1)
d=json.load(open(f)); sys.exit(0 if d.get('completed') else 1)
" STATUS_FILE="${status_file}" 2>/dev/null; then
        echo "  [SKIP] ${variant}_${task} already done"
        return 0
    fi

    echo ""
    echo "==============================="
    echo " ${variant}_${task}  fold=${FOLD}"
    echo "==============================="

    python3 -u "${HOME_MIL}/train_mm_abmil_v8.py" \
        --samples-dir    "${SAMPLES}"    \
        --splits-csv     "${SPLITS_CSV}" \
        --split          "${SPLIT}"      \
        --fold           "${FOLD}"       \
        --out-dir        "${OUT_DIR}"    \
        --phase          p2              \
        --p2-variant     "${variant}"    \
        --task           "${task}"       \
        --global-hp                      \
        --combined-train                 \
        --modal-dropout  0.3             \
        --max-he-patches 99999           \
        &
    PY_PID=$!
    wait "${PY_PID}"
    local exit_code=$?
    PY_PID=""

    if [[ "${RESUBMIT}" -eq 1 ]]; then
        echo "Wall-time hit during ${variant}_${task}. Resubmitting..."
        sbatch "${SCRIPT_PATH}"
        exit 0
    fi

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "ERROR: ${variant}_${task} exited with ${exit_code}"
    fi
}

# Run all 12 variant×task combos unless overridden by --export
if [[ -n "${VARIANT:-}" && -n "${TASK:-}" ]]; then
    run_one "${VARIANT}" "${TASK}"
else
    for variant in early late middle; do
        for task in cls acr_surv clad_surv death_surv; do
            run_one "${variant}" "${task}"
            [[ "${RESUBMIT}" -eq 1 ]] && exit 0
        done
    done
fi

echo "Done."
