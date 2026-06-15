#!/bin/bash
#SBATCH --job-name=reeval_slot
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=1,2
#SBATCH --output=/home/aih/dinesh.haridoss/logs/reeval_slot_%A_%a.out

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil
python analysis/reeval_slot_mega.py --fold $SLURM_ARRAY_TASK_ID 2>&1
