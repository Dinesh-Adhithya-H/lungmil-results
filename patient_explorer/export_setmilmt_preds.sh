#!/usr/bin/env bash
#SBATCH --job-name=export_setmilmt
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.err

set -euo pipefail
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
export PYTHONUNBUFFERED=1

conda run -n chicago python patient_explorer/export_setmilmt_preds.py
