#!/bin/bash
#SBATCH --job-name=lglobal_avg
#SBATCH --output=analysis/logs/lglobal_avg_%j.out
#SBATCH --error=analysis/logs/lglobal_avg_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00

source ~/.bashrc
conda activate chicago_mil
cd /home/aih/dinesh.haridoss/chicago_mil

echo "=== L_global 5-split average heatmap ==="
python analysis/plot_L_global_avg.py
echo "=== DONE ==="
