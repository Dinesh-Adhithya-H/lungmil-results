#!/bin/bash
#SBATCH --job-name=wandb_upload
#SBATCH --output=results_mm_abmil_v8/slurm_logs/%j_wandb_upload.out
#SBATCH --error=results_mm_abmil_v8/slurm_logs/%j_wandb_upload.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
conda run -n chicago python scripts/log_results_to_wandb.py \
    --project chicago-mil \
    "$@"
