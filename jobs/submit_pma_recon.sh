#!/bin/bash
#SBATCH --job-name=pma_recon
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_pma_recon.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_pma_recon.err

source ~/.bashrc
conda activate chicago
set -eo pipefail
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p results/mm_abmil_v8/slurm_logs

echo "=== PMA Reconstruction Experiment ==="
echo "Host: $(hostname)   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Split: ${SPLIT:-0}  Fold: ${FOLD:-0}"
echo "======================================"

python scripts/pma_recon_experiment.py \
    --split "${SPLIT:-0}" \
    --fold  "${FOLD:-0}" \
    --n-train "${N_TRAIN:-200}" \
    --n-test  "${N_TEST:-100}"

echo "Done."
