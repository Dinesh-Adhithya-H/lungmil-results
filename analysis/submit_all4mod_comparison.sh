#!/bin/bash
#SBATCH --job-name=all4mod_cmp
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/all4mod_cmp_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/all4mod_cmp_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
python analysis/plot_all4mod_comparison.py
