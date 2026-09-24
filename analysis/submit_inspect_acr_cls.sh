#!/bin/bash
#SBATCH --job-name=inspect_acr_nans
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/inspect_acr_nans_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/inspect_acr_nans_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
python analysis/inspect_acr_cls_nans.py
