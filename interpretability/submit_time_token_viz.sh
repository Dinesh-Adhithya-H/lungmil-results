#!/usr/bin/env bash
#SBATCH --job-name=time_token_viz
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/time_token_viz_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/time_token_viz_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -uo pipefail
export PYTHONUNBUFFERED=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil
python3 interpretability/plot_time_token_viz.py
