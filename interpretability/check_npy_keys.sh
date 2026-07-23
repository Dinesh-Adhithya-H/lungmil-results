#!/usr/bin/env bash
#SBATCH --job-name=check_npy_keys
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=192G
#SBATCH --time=00:20:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.err

set -euo pipefail
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
export PYTHONUNBUFFERED=1

conda run -n chicago python interpretability/check_npy_keys.py
echo "=== Done: $(date) ==="
