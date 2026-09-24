#!/bin/bash
#SBATCH --job-name=unimod_abl_v2
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_unimod_abl_v2_%a.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_unimod_abl_v2_%a.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -uo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
cd /home/aih/dinesh.haridoss/chicago_mil

SPLIT=${SLURM_ARRAY_TASK_ID}

echo "=== unimodal ablation v2 split=${SPLIT} ==="
python interpretability/eval_unimodal_ablation_v2.py \
    --split  ${SPLIT} \
    --fold   0 \
    --tasks  acr_cls acr_surv clad_surv death_surv \
    --device cuda

echo "=== Done split=${SPLIT} ==="
