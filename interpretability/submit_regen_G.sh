#!/bin/bash
#SBATCH --job-name=regen_G_panel
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_regen_G.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_regen_G.err

source ~/.bashrc
conda activate chicago
cd /home/aih/dinesh.haridoss/chicago_mil
python interpretability/regen_G_panel.py
