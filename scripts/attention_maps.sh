#!/usr/bin/env bash
#SBATCH --job-name=attention_maps
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs/%j_attention_maps.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs/%j_attention_maps.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -uo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "===== MIL Attention Maps ====="
echo "Host: $(hostname)  Date: $(date)"

mkdir -p /home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs

python3 -u /home/aih/dinesh.haridoss/chicago_mil/scripts/attention_maps.py

echo "===== DONE: $(date) ====="
