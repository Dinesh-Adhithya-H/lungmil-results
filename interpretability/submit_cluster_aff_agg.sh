#!/bin/bash
#SBATCH --job-name=cluster_aff_agg
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_cluster_aff_agg.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_cluster_aff_agg.err

source ~/.bashrc
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil
python interpretability/gen_cluster_aff_agg.py
