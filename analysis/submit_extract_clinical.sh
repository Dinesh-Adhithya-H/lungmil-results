#!/bin/bash
#SBATCH --job-name=extract_clinical
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/analysis/logs/extract_clinical_%j.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/analysis/logs/extract_clinical_%j.err

mkdir -p /ictstr01/home/aih/dinesh.haridoss/chicago_mil/analysis/logs

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago

cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil

echo "=== Start: $(date) ==="
python analysis/extract_clinical_features.py
echo "=== Done: $(date) ==="
