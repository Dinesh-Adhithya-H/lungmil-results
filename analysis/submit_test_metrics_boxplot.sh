#!/usr/bin/env bash
#SBATCH --job-name=test_metrics_boxplot
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_test_metrics_boxplot.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_test_metrics_boxplot.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

source ~/.bashrc
conda activate chicago

export PYTHONPATH="/ictstr01/home/aih/dinesh.haridoss/chicago_mil/src:${PYTHONPATH:-}"
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil

python analysis/plot_test_metrics_boxplot.py
