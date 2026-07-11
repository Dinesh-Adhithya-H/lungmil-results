#!/usr/bin/env bash
#SBATCH --job-name=mofa_analysis
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs/%j_mofa_analysis.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs/%j_mofa_analysis.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -uo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "===== MOFA+ Analysis ====="
echo "Host: $(hostname)  Date: $(date)"

pip install mofapy2 --quiet 2>/dev/null || echo "mofapy2 install skipped"

mkdir -p /home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs

python3 -u /home/aih/dinesh.haridoss/chicago_mil/scripts/mofa_analysis.py

echo "===== DONE: $(date) ====="
