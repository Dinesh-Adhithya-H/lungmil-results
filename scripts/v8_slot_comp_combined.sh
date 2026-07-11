#!/usr/bin/env bash
# v8_slot_comp_combined.sh — SharedSlotMIL (CompetitiveSlotAttn) final combined training
#
# Prerequisites: all 4 per-fold HP sweeps must be complete
#   results/mm_abmil_v8/phase2/split1_fold{0..3}/slot_mega_alt_shared_comp/hp_sweep/hp_sweep_p2.json
#
# Picks global best HP across folds, trains on train+val combined, evaluates on test.
# Self-resubmits on wall-time.
#
# Usage:
#   sbatch scripts/v8_slot_comp_combined.sh
#
#SBATCH --job-name=slot_comp_comb
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=220G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_slot_comp_combined.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_slot_comp_combined.err
#SBATCH --mail-type=NONE

set -uo pipefail
export PYTHONUNBUFFERED=1

HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
OUT_DIR="${HOME_MIL}/results/mm_abmil_v8"
SPLIT=1
FOLD=0

SAVE_DIR="${OUT_DIR}/phase2/split${SPLIT}_fold${FOLD}/slot_mega_alt_shared_comp_combined"
STATUS_FINAL="${SAVE_DIR}/status_slot_final.json"
LOG_DIR="${HOME_MIL}/results_mm_abmil_v8/slurm_logs"
SCRIPT_PATH="$(realpath "$0")"

# Check all 4 per-fold HP sweeps are done before proceeding
for f in 0 1 2 3; do
    HP="${OUT_DIR}/phase2/split${SPLIT}_fold${f}/slot_mega_alt_shared_comp/hp_sweep/hp_sweep_p2.json"
    if [[ ! -f "${HP}" ]]; then
        echo "HP sweep not found for fold ${f}: ${HP}"
        echo "Run v8_slot_comp.sh for all 4 folds first."
        exit 1
    fi
done
echo "All 4 per-fold HP sweeps present."

mkdir -p "${LOG_DIR}"

echo "==============================="
echo " slot_comp_combined  fold=${FOLD}  job=${SLURM_JOB_ID}"
echo " save_dir: ${SAVE_DIR}"
echo "==============================="

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
    echo "Combined training already complete. Exiting."
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
    --samples-dir    "${SAMPLES}"              \
    --splits-csv     "${SPLITS_CSV}"           \
    --split          "${SPLIT}"                \
    --fold           "${FOLD}"                 \
    --out-dir        "${OUT_DIR}"              \
    --phase          p2                        \
    --p2-variant     slot                      \
    --slot-k         128                       \
    --task           mega                      \
    --global-hp                                \
    --combined-train                           \
    --alternating                              \
    --p2-tag         alt_shared_comp_combined  \
    --hp-source-tag  alt_shared_comp           \
    --modal-dropout  0.3                       \
    --max-he-patches 99999                     \
    &
PY_PID=$!
echo "Python PID: ${PY_PID}"

wait "${PY_PID}"
PY_EXIT=$?
echo "Python exited with code: ${PY_EXIT}"

if [[ "${RESUBMIT}" -eq 1 ]]; then
    echo "Resubmitting combined job (wall-time hit)..."
    NEW_JID=$(sbatch "${SCRIPT_PATH}" | awk '{print $NF}')
    echo "Submitted new job: ${NEW_JID}"
    exit 0
fi

if STATUS_FILE="${STATUS_FINAL}" is_done; then
    echo "Training complete. status_slot_final.json confirmed."
    exit 0
fi

if [[ "${PY_EXIT}" -ne 0 ]]; then
    echo "Python exited with ${PY_EXIT}. Resubmitting to recover..."
    NEW_JID=$(sbatch "${SCRIPT_PATH}" | awk '{print $NF}')
    echo "Submitted recovery job: ${NEW_JID}"
    exit 0
fi

echo "Done — no resubmit needed."
