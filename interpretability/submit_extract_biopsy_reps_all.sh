#!/bin/bash
#SBATCH --job-name=extract_bioreps_all
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/extract_bioreps_all_%A_%a.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/extract_bioreps_all_%A_%a.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

source ~/.bashrc
conda activate chicago
cd /home/aih/dinesh.haridoss/chicago_mil
export PYTHONPATH="/home/aih/dinesh.haridoss/chicago_mil/src:${PYTHONPATH:-}"

SPLIT=${SLURM_ARRAY_TASK_ID}

echo "=== extract_biopsy_reps_all split=${SPLIT} $(date) ==="
python -u interpretability/extract_biopsy_reps_all.py \
    --split  ${SPLIT} \
    --fold   0 \
    --tasks  acr_cls acr_surv clad_surv death_surv \
    --device cuda

echo "=== Done split=${SPLIT} $(date) ==="
