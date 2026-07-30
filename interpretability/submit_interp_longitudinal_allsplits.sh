#!/bin/bash
# Submit longitudinal interpretability for all 5 splits × 4 tasks (fold 0).
# Skips if output dir already has Lpop panels.
# Usage: bash submit_interp_longitudinal_allsplits.sh

SCRIPT=$(dirname "$0")/submit_interp_longitudinal.sh
TASKS=(acr_cls acr_surv clad_surv death_surv)

for SPLIT in 0 1 2 3 4; do
    for TASK in "${TASKS[@]}"; do
        OUT_DIR="/ictstr01/home/aih/dinesh.haridoss/chicago_mil/interpretability/longitudinal_mk_interp/split${SPLIT}_fold0_${TASK}"

        TASK_KEY="${TASK%_surv}"
        if [ -f "${OUT_DIR}/seed_attribution_data_${TASK_KEY}.json" ] || [ -f "${OUT_DIR}/seed_attribution_data_${TASK}.json" ]; then
            echo "[SKIP] split${SPLIT} task=${TASK} — attribution data already exists"
            continue
        fi

        echo "[SUBMIT] split=${SPLIT} task=${TASK}"
        sbatch "${SCRIPT}" \
            --split "${SPLIT}" \
            --fold 0 \
            --task "${TASK}" \
            --n-patients 60 \
            --min-biopsies 2
    done
done
