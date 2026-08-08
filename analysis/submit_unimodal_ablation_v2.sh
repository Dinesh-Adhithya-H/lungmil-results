#!/bin/bash
#SBATCH --job-name=unimodal_ablation_v2
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/unimodal_ablation_v2_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/unimodal_ablation_v2_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil

echo "=== unimodal_ablation_v2 ==="
python analysis/plot_unimodal_ablation_v2.py
echo "=== Done ==="
