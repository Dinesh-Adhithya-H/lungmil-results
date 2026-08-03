#!/usr/bin/env bash
#SBATCH --job-name=km_curves
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=350G
#SBATCH --time=01:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_km_curves.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_km_curves.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil
export PYTHONPATH="/home/aih/dinesh.haridoss/chicago_mil/src:${PYTHONPATH:-}"

echo "=== km_curves job=${SLURM_JOB_ID} $(date) ==="
python -u analysis/plot_km_from_model.py
echo "=== DONE $(date) ==="
