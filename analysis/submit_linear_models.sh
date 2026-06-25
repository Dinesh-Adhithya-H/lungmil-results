#!/bin/bash
#SBATCH --job-name=linear_models
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/analysis/logs/linear_models_%j.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/analysis/logs/linear_models_%j.err

mkdir -p /ictstr01/home/aih/dinesh.haridoss/chicago_mil/analysis/logs

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago

cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil

echo "=== Start: $(date) ==="
python analysis/train_linear_models.py
echo "=== Done: $(date) ==="
