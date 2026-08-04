#!/bin/bash
#SBATCH --job-name=bench_table_v2
#SBATCH --output=analysis/logs/bench_table_v2_%j.out
#SBATCH --error=analysis/logs/bench_table_v2_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

source ~/.bashrc
conda activate chicago_mil
cd /home/aih/dinesh.haridoss/chicago_mil

echo "=== Benchmark table v2 ==="
python analysis/plot_benchmark_table_v2.py
echo "=== DONE ==="
