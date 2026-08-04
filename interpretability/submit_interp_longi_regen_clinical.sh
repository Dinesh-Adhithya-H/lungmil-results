#!/bin/bash
# Force re-run longitudinal no_alibi interpretability for all splits×tasks
# to regenerate cluster_aff_data_{task}.json WITH Clinical feature affinity.
# Overwrites existing outputs — run only after editing interpret_longitudinal_mk.py
# to add the Clinical PMA affinity extraction.
#
# Usage:
#   bash interpretability/submit_interp_longi_regen_clinical.sh
#   bash interpretability/submit_interp_longi_regen_clinical.sh --variant longitudinal_mk_no_alibi

set -euo pipefail

SCRIPT=$(dirname "$0")/submit_interp_longitudinal_no_alibi.sh
TASKS=(acr_cls acr_surv clad_surv death_surv)
VARIANTS=(longitudinal_mk_no_alibi longitudinal_mk_mt_no_alibi)

if [[ $# -gt 0 && "$1" == "--variant" ]]; then
    VARIANTS=("$2")
fi

for VARIANT in "${VARIANTS[@]}"; do
    for SPLIT in 0 1 2 3 4; do
        for TASK in "${TASKS[@]}"; do
            echo "[SUBMIT] ${VARIANT} split${SPLIT} task=${TASK}"
            sbatch "${SCRIPT}" \
                --variant "${VARIANT}" \
                --split   "${SPLIT}" \
                --fold    0 \
                --task    "${TASK}" \
                --n-patients 60 \
                --min-biopsies 2
        done
    done
done
echo "Submitted all jobs. Monitor with: squeue -u \$USER"
