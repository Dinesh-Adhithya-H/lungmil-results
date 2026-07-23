#!/usr/bin/env bash
#SBATCH --job-name=patient_summaries
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.err

# Generate per-patient L0_summary plots for SetMIL-MT across all test splits.
# Loads cached results_raw.npy (no GPU needed), computes cohort percentile ranks,
# flags statistical discordances, and saves flagged_patients.csv.
#
# Usage:
#   sbatch submit_patient_summaries.sh              # SetMIL-MT (default)
#   sbatch submit_patient_summaries.sh --longitudinal  # LongMIL-MT (TODO: wire up)

set -euo pipefail
REPO=/ictstr01/home/aih/dinesh.haridoss/chicago_mil
cd "${REPO}"
export PYTHONUNBUFFERED=1

echo "=== Patient summaries: $(date) ==="
echo "  Loading all-splits results_raw.npy (no GPU needed)"

BASE="${REPO}/interpretability/set_mil_mt_interp"

# --json-only loads results_raw.npy from --out-dir without touching the model or GPU.
# --patient-summaries generates one L0_summary PNG per record + flagged_patients.csv.
# We use the pre-merged npy (all 4 single-task npys merged by stem into one file).
conda run -n chicago python interpretability/interpret_set_mil_mt.py \
    --json-only \
    --patient-summaries \
    --out-dir "${BASE}/all_splits_merged" \
    --wandb-project chicago-mil-interpretability \
    "$@"

echo "=== Done: $(date) ==="
