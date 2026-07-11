#!/bin/bash
#SBATCH --job-name=pma_gmm
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_pma_gmm.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_pma_gmm.err

source ~/.bashrc
conda activate chicago
set -eo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/pma_gmm_experiment.py \
    --split 0 \
    --fold  0 \
    --n-train 200 \
    --n-test  100
