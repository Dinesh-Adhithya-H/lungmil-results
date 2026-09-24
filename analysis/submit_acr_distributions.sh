#!/bin/bash
#SBATCH --job-name=acr_distributions
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/acr_distributions_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/acr_distributions_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil

python analysis/plot_acr_distributions.py
