#!/usr/bin/env bash
#SBATCH --job-name=pat_evo_vid
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/pat_evo_vid_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/pat_evo_vid_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil
export PYTHONPATH="/home/aih/dinesh.haridoss/chicago_mil/src:${PYTHONPATH:-}"

echo "=== pat_evo_vid job=${SLURM_JOB_ID} $(date) ==="
python -u analysis/make_patient_evolution_videos.py
echo "=== DONE $(date) ==="
