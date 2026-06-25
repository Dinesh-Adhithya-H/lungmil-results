#!/bin/bash
#SBATCH --job-name=nature_figs_all
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/analysis/logs/nature_figs_all_%j.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/analysis/logs/nature_figs_all_%j.err

mkdir -p /ictstr01/home/aih/dinesh.haridoss/chicago_mil/analysis/logs

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago

cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil

echo "=== Start: $(date) ==="
echo "Job ID: $SLURM_JOB_ID"

python analysis/nature_figs_all.py

echo "=== Done: $(date) ==="
