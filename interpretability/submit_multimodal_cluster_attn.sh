#!/usr/bin/env bash
#SBATCH --job-name=multimod_cluster
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/multimod_cluster_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/multimod_cluster_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -uo pipefail
export PYTHONUNBUFFERED=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil
python3 interpretability/plot_multimodal_cluster_attn.py
