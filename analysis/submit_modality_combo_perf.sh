#!/bin/bash
#SBATCH --job-name=mod_combo_perf
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/mod_combo_perf_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/mod_combo_perf_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
python analysis/plot_modality_combo_perf.py
