#!/bin/bash
# Submit longitudinal no-alibi interpretability for all 5 splits × 2 variants × 4 tasks.
# Variants: longitudinal_mk_no_alibi, longitudinal_mk_mt_no_alibi
# Tasks: acr_cls acr_surv clad_surv death_surv
# Skips if output dir already has Lpop panels.
# Usage: bash submit_interp_longitudinal_no_alibi_allsplits.sh [--variant longitudinal_mk_no_alibi]

SCRIPT=$(dirname "$0")/submit_interp_longitudinal_no_alibi.sh
TASKS=(acr_cls acr_surv clad_surv death_surv)
VARIANTS=(longitudinal_mk_no_alibi longitudinal_mk_mt_no_alibi)

# Allow overriding variants from command line
if [[ $# -gt 0 && "$1" == "--variant" ]]; then
    VARIANTS=("$2")
fi

INTERP_ROOT=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/interpretability/longitudinal_mk_interp

for VARIANT in "${VARIANTS[@]}"; do
    for SPLIT in 0 1 2 3 4; do
        for TASK in "${TASKS[@]}"; do
            OUT_DIR="${INTERP_ROOT}/${VARIANT}_split${SPLIT}_fold0_${TASK}"

            if [ -f "${OUT_DIR}/seed_attribution_data_${TASK%_surv}.json" ] || [ -f "${OUT_DIR}/seed_attribution_data_${TASK}.json" ]; then
                echo "[SKIP] ${VARIANT} split${SPLIT} task=${TASK} — attribution data already exists"
                continue
            fi

            echo "[SUBMIT] variant=${VARIANT} split=${SPLIT} task=${TASK}"
            sbatch "${SCRIPT}" \
                --variant "${VARIANT}" \
                --split "${SPLIT}" \
                --fold 0 \
                --task "${TASK}" \
                --n-patients 60 \
                --min-biopsies 2
        done
    done
done
