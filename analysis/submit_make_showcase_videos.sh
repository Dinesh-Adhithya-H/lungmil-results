#!/usr/bin/env bash
#SBATCH --job-name=showcase_vid
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/showcase_vid_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/showcase_vid_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil
export PYTHONPATH="/home/aih/dinesh.haridoss/chicago_mil/src:/home/aih/dinesh.haridoss/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

echo "=== showcase_vid job=${SLURM_JOB_ID} $(date) ==="
python -u analysis/make_showcase_videos.py
echo "=== DONE $(date) ==="
