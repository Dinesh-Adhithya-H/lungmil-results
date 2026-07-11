#!/usr/bin/env bash
#SBATCH --job-name=pseudobulk_deg
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs/%j_pseudobulk_deg.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs/%j_pseudobulk_deg.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -uo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "===== Pseudobulk DEG + GSEA ====="
echo "Host: $(hostname)  Date: $(date)"

pip install pydeseq2 gseapy --quiet 2>/dev/null || echo "package install attempted"

mkdir -p /home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs

python3 -u /home/aih/dinesh.haridoss/chicago_mil/scripts/pseudobulk_deg.py

echo "===== DONE: $(date) ====="
