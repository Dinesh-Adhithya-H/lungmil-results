#!/bin/bash
#SBATCH --job-name=extract_biopsy_reps
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/extract_biopsy_reps_%A_%a.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/extract_biopsy_reps_%A_%a.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-4

source ~/.bashrc
conda activate chicago
cd /home/aih/dinesh.haridoss/chicago_mil

SPLIT=${SLURM_ARRAY_TASK_ID}

echo "=== extract_biopsy_reps split=${SPLIT} ==="
python interpretability/extract_biopsy_reps.py \
    --split  ${SPLIT} \
    --fold   0 \
    --tasks  acr_surv \
    --device cuda

echo "=== Done split=${SPLIT} ==="
