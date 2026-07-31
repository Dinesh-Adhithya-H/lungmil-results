#!/bin/bash
# Submit interpretability re-runs to generate cluster_aff_data_{task}.json.
# Skips if cluster_aff_data already exists for ALL tasks of a split.
# Usage: bash submit_cluster_aff_allsplits.sh

SCRIPT=$(dirname "$0")/submit_interp_longitudinal_no_alibi.sh
TASK_KEYS=(acr_cls acr_surv clad death)
TASK_DIRS=(acr_cls acr_surv clad_surv death_surv)
VARIANTS=(longitudinal_mk_no_alibi longitudinal_mk_mt_no_alibi)

INTERP_ROOT=/home/aih/dinesh.haridoss/chicago_mil/interpretability/longitudinal_mk_interp

for VARIANT in "${VARIANTS[@]}"; do
    for SPLIT in 0 1 2 3 4; do
        # Check if ALL 4 cluster_aff JSONs already exist for this split
        all_done=true
        for i in 0 1 2 3; do
            TKEY="${TASK_KEYS[$i]}"
            TDIR="${TASK_DIRS[$i]}"
            JPATH="${INTERP_ROOT}/${VARIANT}_split${SPLIT}_fold0_${TDIR}/cluster_aff_data_${TKEY}.json"
            if [ ! -f "${JPATH}" ]; then
                all_done=false
                break
            fi
        done

        if [ "$all_done" = true ]; then
            echo "[SKIP] ${VARIANT} split${SPLIT} — cluster_aff_data already complete"
            continue
        fi

        echo "[SUBMIT] ${VARIANT} split${SPLIT} (--task mega, generates all 4 tasks)"
        sbatch "${SCRIPT}" \
            --variant "${VARIANT}" \
            --split "${SPLIT}" \
            --fold 0 \
            --task mega \
            --n-patients 60 \
            --min-biopsies 2
    done
done

echo "[done] submission complete"
