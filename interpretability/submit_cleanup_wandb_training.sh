#!/usr/bin/env bash
#SBATCH --job-name=cleanup_wandb_train
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.err

set -euo pipefail
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
export PYTHONUNBUFFERED=1
conda run -n chicago python interpretability/cleanup_wandb_training.py "$@"
echo "=== Done: $(date) ==="
