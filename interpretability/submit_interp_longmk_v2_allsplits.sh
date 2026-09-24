#!/usr/bin/env bash
# Submit LongMK-v2 interpretability for all 5 splits × 4 tasks.
# Each job runs interpret_longitudinal_mk.py for one split+task.
# Usage: bash interpretability/submit_interp_longmk_v2_allsplits.sh

set -uo pipefail

VARIANT="longitudinal_mk_no_alibi_v2"
TASKS=(acr_cls acr_surv clad_surv death_surv)
INTERP_ROOT=/home/aih/dinesh.haridoss/chicago_mil/interpretability/longitudinal_mk_interp
LOG_DIR=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs

mkdir -p "${LOG_DIR}"

for SPLIT in 0 1 2 3 4; do
    for TASK in "${TASKS[@]}"; do
        OUT_DIR="${INTERP_ROOT}/${VARIANT}_split${SPLIT}_fold0_${TASK}"
        DONE_FILE="${OUT_DIR}/seed_attribution_data_${TASK}.json"
        # also check without _surv suffix
        DONE_FILE2="${OUT_DIR}/seed_attribution_data_${TASK%_surv}.json"

        if [ -f "${DONE_FILE}" ] || [ -f "${DONE_FILE2}" ]; then
            echo "[SKIP] split=${SPLIT} task=${TASK} — already done"
            continue
        fi

        echo "[SUBMIT] split=${SPLIT}  task=${TASK}"
        sbatch --job-name="v2_interp_s${SPLIT}_${TASK}" \
               --partition=gpu_p \
               --qos=gpu_normal \
               --gres=gpu:1 \
               --cpus-per-task=8 \
               --mem=80G \
               --time=04:00:00 \
               --output="${LOG_DIR}/%j_v2_interp_s${SPLIT}_${TASK}.out" \
               --error="${LOG_DIR}/%j_v2_interp_s${SPLIT}_${TASK}.err" \
               --mail-type=FAIL \
               --mail-user=dinesh.haridoss@helmholtz-munich.de \
               --wrap="
set -uo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate chicago
cd /home/aih/dinesh.haridoss/chicago_mil
python interpretability/interpret_longitudinal_mk.py \
    --variant ${VARIANT} \
    --split ${SPLIT} \
    --fold 0 \
    --task ${TASK} \
    --n-patients 60 \
    --min-biopsies 2 \
    --wandb-project chicago-mil-interpretability
echo Done split=${SPLIT} task=${TASK}
"
    done
done

echo "All jobs submitted."
