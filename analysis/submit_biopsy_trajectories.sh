#!/bin/bash
#SBATCH --job-name=biopsy_traj
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/biopsy_traj_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/biopsy_traj_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
python analysis/plot_biopsy_trajectories.py
