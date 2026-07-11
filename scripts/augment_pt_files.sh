#!/usr/bin/env bash
#SBATCH --job-name=augment_pt
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs/%j_augment_pt.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs/%j_augment_pt.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=256G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

# NOTE: 256G RAM because we load full BAL h5ad (1.4M cells × 2000 genes)
# and HE h5ad (14.8M patches × 1024 dims) into memory simultaneously.

set -uo pipefail
export PYTHONUNBUFFERED=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "====================================="
echo " Augment .pt files with UMAP + pseudobulk"
echo "====================================="
echo "Host  : $(hostname)"
echo "Date  : $(date)"
echo "Mem   : ${SLURM_MEM_PER_NODE} MB"
echo ""

python3 -u /home/aih/dinesh.haridoss/chicago_mil/scripts/augment_pt_files.py

echo ""
echo "===== DONE: $(date) ====="
