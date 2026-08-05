#!/bin/bash
#SBATCH --job-name=bench_pvalues
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/bench_pvalues_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/bench_pvalues_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --time=0:10:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil

echo "=== LongMK significance heatmap ==="
python3 analysis/plot_benchmark_pvalues.py
echo "=== DONE ==="
