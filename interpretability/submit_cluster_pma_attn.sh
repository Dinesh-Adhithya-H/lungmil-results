#!/usr/bin/env bash
#SBATCH --job-name=cluster_pma_attn
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --time=0:15:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/cluster_pma_attn_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/cluster_pma_attn_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -uo pipefail
export PYTHONUNBUFFERED=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil
python3 interpretability/plot_cluster_pma_attn.py
