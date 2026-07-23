#!/usr/bin/env bash
#SBATCH --job-name=interp_merged_cdg
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.err

# Merges 4 single-task all_splits results_raw.npy files by stem, then runs
# panels C, D, G on the combined dataset and uploads to wandb.

set -euo pipefail
REPO=/ictstr01/home/aih/dinesh.haridoss/chicago_mil
cd "${REPO}"
export PYTHONUNBUFFERED=1

BASE="${REPO}/interpretability/set_mil_mt_interp"

echo "=== Merge single-task npys + panels C,D,G: $(date) ==="
conda run -n chicago python interpretability/interpret_set_mil_mt.py \
    --merge-task-dirs \
        "${BASE}/all_splits_cls" \
        "${BASE}/all_splits_acr_surv" \
        "${BASE}/all_splits_clad_surv" \
        "${BASE}/all_splits_death_surv" \
    --out-dir "${BASE}/all_splits_merged" \
    --panels C,D,G \
    --wandb-project chicago-mil-interpretability
echo "=== Done: $(date) ==="
