#!/usr/bin/env bash
# v8_baseline_extend.sh — Per-fold training (folds 1-3 only) for combos that
# hit the 600-epoch ceiling, now with C-index early stopping (P2_EPOCHS=1000).
#
# Affected combos: middle_clad_surv, middle_death_surv, late_clad_surv
#
# Fold 0 is RESERVED for combined (train+val) training — do NOT submit fold=0.
# After all three folds complete, run v8_baseline_combined.sh.
#
# Usage:
#   sbatch --export=FOLD=1 scripts/v8_baseline_extend.sh
#   sbatch --export=FOLD=2 scripts/v8_baseline_extend.sh
#   sbatch --export=FOLD=3 scripts/v8_baseline_extend.sh
#
#SBATCH --job-name=base_extend
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=180G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_base_extend_f%x.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_base_extend_f%x.err
#SBATCH --mail-type=NONE

set -uo pipefail
export PYTHONUNBUFFERED=1

FOLD="${FOLD:-1}"
if [[ "${FOLD}" -eq 0 ]]; then
    echo "ERROR: fold=0 is reserved for combined (train+val) training."
    echo "  Submit folds 1,2,3 with this script, then run v8_baseline_combined.sh."
    exit 1
fi
HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
OUT_DIR="${HOME_MIL}/results/mm_abmil_v8"
LOG_DIR="${HOME_MIL}/results_mm_abmil_v8/slurm_logs"
SCRIPT_PATH="$(realpath "$0")"
SPLIT=1

mkdir -p "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "Python: $(which python3)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "FOLD=${FOLD}"

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

    if python3 -c "
import json,sys,os
f=os.environ.get('STATUS_FILE','')
if not f or not os.path.exists(f): sys.exit(1)
d=json.load(open(f)); sys.exit(0 if d.get('completed') else 1)
" STATUS_FILE="${status_file}" 2>/dev/null; then
        echo "  [SKIP] ${variant}_${task} fold=${FOLD} already done"
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
        --p2-hp-sweep                    \
        --modal-dropout  0.3             \
        --max-he-patches 99999           \
        &
    PY_PID=$!
    wait "${PY_PID}"
    local exit_code=$?
    PY_PID=""

    if [[ "${RESUBMIT}" -eq 1 ]]; then
        echo "Wall-time hit during ${variant}_${task}. Resubmitting..."
        sbatch --export=FOLD="${FOLD}" "${SCRIPT_PATH}"
        exit 0
    fi

    [[ "${exit_code}" -ne 0 ]] && echo "ERROR: ${variant}_${task} exited with ${exit_code}"
}

for combo in "middle clad_surv" "middle death_surv" "late clad_surv"; do
    run_one ${combo}
    [[ "${RESUBMIT}" -eq 1 ]] && exit 0
done

echo "Done fold=${FOLD}."
