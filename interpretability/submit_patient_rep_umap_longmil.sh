#!/usr/bin/env bash
#SBATCH --job-name=pat_rep_umap
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=220G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=6:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/pat_rep_umap_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/pat_rep_umap_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -uo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil

python3 interpretability/patient_rep_umap_longmil.py
