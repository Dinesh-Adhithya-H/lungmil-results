#!/bin/bash
#SBATCH --job-name=rep_umaps
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_rep_umaps.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_rep_umaps.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# Run UMAP extraction for all 5 splits sequentially (each ~30 min)
# To run a single split: sbatch --export=SPLIT=2 scripts/submit_rep_umaps.sh

source ~/.bashrc
conda activate chicago

cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPLIT="${SPLIT:-0}"
FOLD=0

echo "========================================"
echo "  rep_umaps  split=${SPLIT}  fold=${FOLD}"
echo "========================================"

python3 scripts/extract_rep_umaps.py \
    --split "${SPLIT}" \
    --fold  "${FOLD}"  \
    --split-set test   \
    --patch-sample 64

echo "Done — split=${SPLIT}"
