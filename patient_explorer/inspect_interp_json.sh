#!/usr/bin/env bash
#SBATCH --job-name=inspect_interp
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/home/aih/dinesh.haridoss/logs/inspect_interp_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/logs/inspect_interp_%j.err

set -euo pipefail
source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
python patient_explorer/inspect_interp.py
