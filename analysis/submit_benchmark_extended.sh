#!/bin/bash
#SBATCH --job-name=benchmark_ext
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/aih/dinesh.haridoss/logs/benchmark_ext_%j.out

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil
python analysis/compute_benchmark_extended.py 2>&1
