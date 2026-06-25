#!/usr/bin/env bash
# Submit per-split prediction extraction (GPU) + aggregation (CPU).
# Each of the 5 GPU jobs runs P1+P2 inference on one split's test set.
# The CPU aggregation job runs after all 5 finish (SLURM afterok dependency).

set -euo pipefail

REPO="/home/aih/dinesh.haridoss/chicago_mil"
OUT_DIR="${REPO}/results/mm_abmil_v8"
SAMPLES_DIR="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
SLURM_LOGS="${REPO}/results_mm_abmil_v8/slurm_logs"

mkdir -p "${SLURM_LOGS}"

# ── Submit one GPU job per split ──────────────────────────────────────────────
JOB_IDS=()

for SPLIT in 0 1 2 3 4; do
    JOB_ID=$(sbatch --parsable \
        --job-name="extract_pred_s${SPLIT}" \
        --partition=gpu_p \
        --qos=gpu_normal \
        --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=160G \
        --gres=gpu:1 --constraint="a100_80gb|h100_80gb" \
        --time=04:00:00 \
        --output="${SLURM_LOGS}/%j_extract_pred_s${SPLIT}.out" \
        --error="${SLURM_LOGS}/%j_extract_pred_s${SPLIT}.err" \
        --mail-type=FAIL \
        --mail-user=dinesh.haridoss@helmholtz-munich.de \
        --wrap="
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source \"\$(conda info --base)/etc/profile.d/conda.sh\"
conda activate chicago
echo \"=== extract_predictions split=${SPLIT} job=\${SLURM_JOB_ID} host=\$(hostname) ===\"
python3 -u '${REPO}/analysis/extract_predictions.py' \
    --split        ${SPLIT} \
    --samples-dir  '${SAMPLES_DIR}' \
    --splits-csv   '${SPLITS_CSV}' \
    --results-dir  '${OUT_DIR}' \
    --out-dir      '${REPO}/results/predictions/raw' \
    --workers      8
echo \"=== DONE split=${SPLIT} \$(date) ===\"
")
    JOB_IDS+=("${JOB_ID}")
    echo "Submitted split${SPLIT} → job ${JOB_ID}"
done

# ── Submit CPU aggregation job, depends on all 5 GPU jobs ────────────────────
DEPEND=$(IFS=:; echo "afterok:${JOB_IDS[*]}")

AGG_JOB=$(sbatch --parsable \
    --job-name="aggregate_predictions" \
    --partition=cpu_p \
    --qos=cpu_normal \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G \
    --time=01:00:00 \
    --dependency="${DEPEND}" \
    --output="${SLURM_LOGS}/%j_aggregate_predictions.out" \
    --error="${SLURM_LOGS}/%j_aggregate_predictions.err" \
    --mail-type=END,FAIL \
    --mail-user=dinesh.haridoss@helmholtz-munich.de \
    --wrap="
set -euo pipefail
export PYTHONUNBUFFERED=1
source \"\$(conda info --base)/etc/profile.d/conda.sh\"
conda activate chicago
echo \"=== aggregate_predictions job=\${SLURM_JOB_ID} \$(date) ===\"
python3 -u '${REPO}/analysis/aggregate_predictions.py'
echo \"=== DONE \$(date) ===\"
")
echo "Submitted aggregation → job ${AGG_JOB} (depends on ${JOB_IDS[*]})"
