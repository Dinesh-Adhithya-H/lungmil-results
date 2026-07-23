#!/bin/bash
#SBATCH --job-name=check_gate_keys
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=192G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/check_gate_keys_%j.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/check_gate_keys_%j.err

set -e
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
conda run -n chicago python interpretability/check_gate_vals.py
echo "=== Done ==="
