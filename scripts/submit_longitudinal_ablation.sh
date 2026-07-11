#!/usr/bin/env bash
# Submit unimodal ablation inference for longitudinal_mk fold-0 models.
# Processes all 5 splits in one GPU job.
#SBATCH --job-name=lmk_ablation
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=80G
#SBATCH --gres=gpu:1 --constraint="a100_40gb|a100_80gb|h100_80gb"
#SBATCH --time=02:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_lmk_ablation.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_lmk_ablation.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/aih/dinesh.haridoss/chicago_mil
export PYTHONPATH="/home/aih/dinesh.haridoss/chicago_mil/src:${PYTHONPATH:-}"

echo "=== lmk_ablation job=${SLURM_JOB_ID} $(date) ==="
python -u scripts/compute_longitudinal_ablation.py --splits 0 1 2 3 4
echo "=== DONE $(date) ==="
