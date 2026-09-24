#!/bin/bash
#SBATCH --job-name=tte_dist
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/tte_dist_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/tte_dist_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
python analysis/plot_tte_distributions.py
