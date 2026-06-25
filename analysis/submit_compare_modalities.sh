#!/usr/bin/env bash
# Submit unimodal vs multimodal comparison tables + plots (CPU-only).
set -euo pipefail

REPO="/home/aih/dinesh.haridoss/chicago_mil"
SLURM_LOGS="${REPO}/results_mm_abmil_v8/slurm_logs"
mkdir -p "${SLURM_LOGS}"

JOB_ID=$(sbatch --parsable \
    --job-name="compare_modalities" \
    --partition=cpu_p \
    --qos=cpu_normal \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G \
    --time=00:30:00 \
    --output="${SLURM_LOGS}/%j_compare_modalities.out" \
    --error="${SLURM_LOGS}/%j_compare_modalities.err" \
    --mail-type=END,FAIL \
    --mail-user=dinesh.haridoss@helmholtz-munich.de \
    --wrap="
set -euo pipefail
export PYTHONUNBUFFERED=1
source \"\$(conda info --base)/etc/profile.d/conda.sh\"
conda activate chicago
echo \"=== compare_modalities job=\${SLURM_JOB_ID} \$(date) ===\"
python3 -u '${REPO}/analysis/compare_modalities.py'
echo \"--- generating plots ---\"
python3 -u '${REPO}/analysis/plot_modality_comparison.py'
echo \"=== DONE \$(date) ===\"
")

echo "Submitted → job ${JOB_ID}"
echo "Log: ${SLURM_LOGS}/${JOB_ID}_compare_modalities.out"
echo "Figures: ${REPO}/results/predictions/figures/"
