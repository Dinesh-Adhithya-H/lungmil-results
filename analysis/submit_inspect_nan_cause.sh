#!/bin/bash
#SBATCH --job-name=inspect_nan_cause
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/inspect_nan_cause_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/inspect_nan_cause_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
python analysis/inspect_acr_cls_nan_cause.py
