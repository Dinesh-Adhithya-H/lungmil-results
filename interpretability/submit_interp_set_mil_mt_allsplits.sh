#!/usr/bin/env bash
#SBATCH --job-name=interp_smt_allsplits
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.err

# Usage: sbatch submit_interp_set_mil_mt_allsplits.sh --variant cls
# Variants: cls  acr_surv  clad_surv  death_surv

set -euo pipefail
REPO=/ictstr01/home/aih/dinesh.haridoss/chicago_mil
cd "${REPO}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

conda run -n chicago python interpretability/interpret_set_mil_mt.py \
    --all-splits \
    --wandb-project chicago-mil-interpretability \
    "$@"
