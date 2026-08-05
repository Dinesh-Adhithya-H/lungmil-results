#!/bin/bash
#SBATCH --job-name=longmk_ablation
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/longmk_ablation_%A_%a.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/longmk_ablation_%A_%a.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G
#SBATCH --time=1:00:00
#SBATCH --array=0-4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil

echo "=== LongMK unimodal ablation split=${SLURM_ARRAY_TASK_ID} ==="
python interpretability/run_longmk_unimodal_ablation.py \
    --split ${SLURM_ARRAY_TASK_ID} \
    --fold 0 \
    --gpu 0
echo "=== DONE ==="
