#!/usr/bin/env bash
#SBATCH --job-name=build_benchmark
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.out

set -euo pipefail
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
conda run -n chicago python patient_explorer/build_benchmark_csv.py
