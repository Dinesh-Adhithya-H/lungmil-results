#!/bin/bash
#SBATCH --job-name=bench_table_v2
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/bench_table_v2_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/bench_table_v2_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
cd /home/aih/dinesh.haridoss/chicago_mil

echo "=== Benchmark table v2 ==="
python analysis/plot_benchmark_table_v2.py
echo "=== DONE ==="
