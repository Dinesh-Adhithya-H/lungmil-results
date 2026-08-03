#!/bin/bash
#SBATCH --job-name=multimod_seed_attr
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/multimod_seed_attr_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/multimod_seed_attr_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --time=0:20:00

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago_mil

cd /home/aih/dinesh.haridoss/chicago_mil
python3 analysis/plot_multimod_seed_attribution.py
