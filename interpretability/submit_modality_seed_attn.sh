#!/usr/bin/env bash
#SBATCH --job-name=mod_seed_attn
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/mod_seed_attn_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/mod_seed_attn_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -uo pipefail
export PYTHONUNBUFFERED=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil
python3 interpretability/plot_modality_seed_attn.py
