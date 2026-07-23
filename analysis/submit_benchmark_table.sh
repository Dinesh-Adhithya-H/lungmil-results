#!/usr/bin/env bash
#SBATCH --job-name=benchmark_table
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/ictstr01/home/aih/dinesh.haridoss/chicago_mil
cd "${REPO}"
export PYTHONUNBUFFERED=1

conda run -n chicago python analysis/plot_benchmark_table.py "$@"
echo "=== Done: $(date) ==="
