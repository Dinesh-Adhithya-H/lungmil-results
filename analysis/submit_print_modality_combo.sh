#!/bin/bash
#SBATCH --job-name=print_combo
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/print_combo_%j.out
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

source ~/.bashrc
conda activate chicago
cd /home/aih/dinesh.haridoss/chicago_mil
python analysis/print_modality_combo_results.py
