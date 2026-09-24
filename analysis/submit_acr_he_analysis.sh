#!/bin/bash
#SBATCH --job-name=acr_he_analysis
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/acr_he_analysis_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/acr_he_analysis_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
cd /home/aih/dinesh.haridoss/chicago_mil
python analysis/print_acr_he_analysis.py
