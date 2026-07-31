#!/bin/bash
#SBATCH --job-name=unimodal_ablation
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_unimodal_ablation.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_unimodal_ablation.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

source ~/.bashrc
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil
python interpretability/unimodal_ablation_summary.py
