#!/usr/bin/env bash
# Submit splits-creation jobs for all TCGA cancer types (CPU, reads .pt files).
# Creates: /home/aih/dinesh.haridoss/chicago_mil/data/tcga_splits/{cancer}.csv
set -euo pipefail

REPO="/home/aih/dinesh.haridoss/chicago_mil"
LOGS="${REPO}/results_mm_abmil_v8/slurm_logs"
mkdir -p "${LOGS}"

CANCERS=(gbmlgg blca brca kirc luad)

for CANCER in "${CANCERS[@]}"; do
    JOB_ID=$(sbatch --parsable \
        --job-name="tcga_splits_${CANCER}" \
        --partition=cpu_p \
        --qos=cpu_normal \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G \
        --time=02:00:00 \
        --output="${LOGS}/%j_tcga_splits_${CANCER}.out" \
        --error="${LOGS}/%j_tcga_splits_${CANCER}.err" \
        --mail-type=END,FAIL \
        --mail-user=dinesh.haridoss@helmholtz-munich.de \
        --wrap="
set -euo pipefail
export PYTHONUNBUFFERED=1
source \"\$(conda info --base)/etc/profile.d/conda.sh\"
conda activate chicago
echo \"=== tcga_splits ${CANCER} job=\${SLURM_JOB_ID} \$(date) ===\"
python3 -u '${REPO}/data_prep/make_tcga_multitask_splits.py' --cancer ${CANCER}
echo \"=== DONE \$(date) ===\"
")
    echo "  ${CANCER}: job ${JOB_ID}  →  ${LOGS}/${JOB_ID}_tcga_splits_${CANCER}.out"
done
