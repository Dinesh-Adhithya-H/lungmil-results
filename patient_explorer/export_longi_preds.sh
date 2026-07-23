#!/usr/bin/env bash
#SBATCH --job-name=longi_preds
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/logs/longi_preds_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/logs/longi_preds_%j.err

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago

cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python patient_explorer/export_longi_preds.py
