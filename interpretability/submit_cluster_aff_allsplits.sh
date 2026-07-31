#!/bin/bash
# Submit interpretability re-runs to generate cluster_aff_data_{task}.json.
# Best longitudinal models per task:
#   Death    → longitudinal_mk_no_alibi  (--task death_surv)
#   ACR surv → longitudinal_mk_no_alibi  (--task acr_surv)
# Skips if cluster_aff_data already exists.

SCRIPT=$(dirname "$0")/submit_interp_longitudinal_no_alibi.sh
INTERP_ROOT=/home/aih/dinesh.haridoss/chicago_mil/interpretability/longitudinal_mk_interp
VARIANT=longitudinal_mk_no_alibi

for TASK_ARG in death_surv acr_surv; do
    if [ "$TASK_ARG" = "death_surv" ]; then JSON_KEY="death"; DIR_SUFFIX="death_surv"; fi
    if [ "$TASK_ARG" = "acr_surv"  ]; then JSON_KEY="acr_surv"; DIR_SUFFIX="acr_surv"; fi

    for SPLIT in 0 1 2 3 4; do
        JPATH="${INTERP_ROOT}/${VARIANT}_split${SPLIT}_fold0_${DIR_SUFFIX}/cluster_aff_data_${JSON_KEY}.json"
        if [ -f "${JPATH}" ]; then
            echo "[SKIP] split${SPLIT} ${TASK_ARG}"
            continue
        fi
        echo "[SUBMIT] ${VARIANT} split${SPLIT} --task ${TASK_ARG}"
        sbatch "${SCRIPT}" \
            --variant "${VARIANT}" \
            --split "${SPLIT}" \
            --fold 0 \
            --task "${TASK_ARG}" \
            --n-patients 60 \
            --min-biopsies 2
    done
done

echo "[done]"
