#!/bin/bash
#SBATCH --job-name=extract_biopsy_reps
#SBATCH --output=logs/extract_biopsy_reps_%A_%a.out
#SBATCH --error=logs/extract_biopsy_reps_%A_%a.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-4

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

SPLIT=${SLURM_ARRAY_TASK_ID}

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago_mil

echo "=== extract_biopsy_reps split=${SPLIT} ==="
python interpretability/extract_biopsy_reps.py \
    --split  ${SPLIT} \
    --fold   0 \
    --tasks  acr_cls acr_surv clad_surv death_surv \
    --device cuda

echo "=== Done split=${SPLIT} ==="
