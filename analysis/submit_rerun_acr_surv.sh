#!/bin/bash
#SBATCH --job-name=acr_surv_rerun
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/acr_surv_rerun_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/acr_surv_rerun_%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -uo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
cd /home/aih/dinesh.haridoss/chicago_mil

echo "=== ACR survival per-biopsy inference ==="
python analysis/rerun_acr_surv_inference.py
echo "=== Replotting benchmark ==="
python analysis/plot_benchmark.py
echo "=== Done ==="
