#!/usr/bin/env bash
#SBATCH --job-name=slotattn_abl
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=80G
#SBATCH --gres=gpu:1 --constraint="a100_40gb|a100_80gb|h100_80gb"
#SBATCH --time=02:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_slotattn_abl.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_slotattn_abl.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/aih/dinesh.haridoss/chicago_mil
export PYTHONPATH="/home/aih/dinesh.haridoss/chicago_mil/src:${PYTHONPATH:-}"

echo "=== slotattn_abl job=${SLURM_JOB_ID} $(date) ==="
python -u scripts/compute_slotattn_ablation.py --splits 0 1 2 3 4
echo "=== DONE $(date) ==="
