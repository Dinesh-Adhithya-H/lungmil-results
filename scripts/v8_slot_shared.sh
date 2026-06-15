#!/usr/bin/env bash
# v8_slot_shared.sh — SharedSlotMIL training (K=128 shared slots, per-modality MHA)
#
# Runs HP sweep on train set (val for selection), then final training on train
# with val early-stopping, evaluates once on test.
#
# Self-resubmits on wall-time so the HP sweep + final training can survive
# across multiple job allocations.
#
# Usage:
#   sbatch --export=FOLD=0 v8_slot_shared.sh
#   sbatch --export=FOLD=1 v8_slot_shared.sh
#   sbatch --export=FOLD=2 v8_slot_shared.sh
#   sbatch --export=FOLD=3 v8_slot_shared.sh
#
#SBATCH --job-name=slot_shared
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=220G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_slot_shared_f%x.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_slot_shared_f%x.err
#SBATCH --mail-type=NONE

set -uo pipefail
export PYTHONUNBUFFERED=1

FOLD="${FOLD:-0}"
HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
OUT_DIR="${HOME_MIL}/results/mm_abmil_v8"
SPLIT=1

SAVE_DIR="${OUT_DIR}/phase2/split${SPLIT}_fold${FOLD}/slot_mega_shared"
STATUS_FINAL="${SAVE_DIR}/status_slot_final.json"
HP_FILE="${SAVE_DIR}/hp_sweep/hp_sweep_p2.json"
LOG_DIR="${HOME_MIL}/results_mm_abmil_v8/slurm_logs"
SCRIPT_PATH="$(realpath "$0")"

mkdir -p "${LOG_DIR}"

echo "==============================="
echo " slot_shared  fold=${FOLD}  job=${SLURM_JOB_ID}"
echo " save_dir: ${SAVE_DIR}"
echo "==============================="
echo "HP sweep done: $([ -f "${HP_FILE}" ] && echo YES || echo no)"

is_done() {
    python3 - <<'PYEOF'
import json, sys, os
f = os.environ.get("STATUS_FILE", "")
if not f or not os.path.exists(f): sys.exit(1)
try:
    d = json.load(open(f))
    sys.exit(0 if d.get("completed") else 1)
except Exception: sys.exit(1)
PYEOF
}

if STATUS_FILE="${STATUS_FINAL}" is_done; then
    echo "Final training already complete. Exiting."
    exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "Python: $(which python3)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

RESUBMIT=0
PY_PID=""

walltime_handler() {
    echo ""
    echo "[SIGUSR1] Wall-time approaching. Stopping Python and resubmitting..."
    RESUBMIT=1
    if [[ -n "${PY_PID}" ]]; then
        kill -SIGTERM "${PY_PID}" 2>/dev/null
        local deadline=$((SECONDS + 90))
        while kill -0 "${PY_PID}" 2>/dev/null && [[ $SECONDS -lt $deadline ]]; do
            sleep 2
        done
        kill -SIGKILL "${PY_PID}" 2>/dev/null || true
    fi
}
trap walltime_handler SIGUSR1

python3 -u "${HOME_MIL}/train_mm_abmil_v8.py" \
    --samples-dir    "${SAMPLES}"    \
    --splits-csv     "${SPLITS_CSV}" \
    --split          "${SPLIT}"      \
    --fold           "${FOLD}"       \
    --out-dir        "${OUT_DIR}"    \
    --phase          p2              \
    --p2-variant     slot            \
    --slot-k         128             \
    --task           mega            \
    --p2-hp-sweep                    \
    --p2-tag         shared          \
    --modal-dropout  0.3             \
    --max-he-patches 99999           \
    &
PY_PID=$!
echo "Python PID: ${PY_PID}"

wait "${PY_PID}"
PY_EXIT=$?
echo "Python exited with code: ${PY_EXIT}"

if [[ "${RESUBMIT}" -eq 1 ]]; then
    echo "Resubmitting fold=${FOLD} (wall-time hit)..."
    NEW_JID=$(sbatch --export=FOLD="${FOLD}" "${SCRIPT_PATH}" | awk '{print $NF}')
    echo "Submitted new job: ${NEW_JID}"
    exit 0
fi

if STATUS_FILE="${STATUS_FINAL}" is_done; then
    echo "Training complete. status_slot_final.json confirmed."
    exit 0
fi

if [[ "${PY_EXIT}" -ne 0 ]]; then
    echo "Python exited with ${PY_EXIT} (not SIGTERM). Resubmitting to recover..."
    NEW_JID=$(sbatch --export=FOLD="${FOLD}" "${SCRIPT_PATH}" | awk '{print $NF}')
    echo "Submitted recovery job: ${NEW_JID}"
    exit 0
fi

echo "Done — no resubmit needed."
