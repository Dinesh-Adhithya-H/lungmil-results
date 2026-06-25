#!/bin/bash
#SBATCH --job-name=hero_fig
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/analysis/logs/hero_fig_%j.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/analysis/logs/hero_fig_%j.err

mkdir -p /ictstr01/home/aih/dinesh.haridoss/chicago_mil/analysis/logs

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago

cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil

echo "=== Start: $(date) ==="
python analysis/nature_fig0_hero.py
echo "=== Done: $(date) ==="
