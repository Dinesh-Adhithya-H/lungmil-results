#!/bin/bash
#SBATCH --job-name=biopsy_rep_umap_km
#SBATCH --output=logs/biopsy_rep_umap_km_%j.out
#SBATCH --error=logs/biopsy_rep_umap_km_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago_mil

echo "=== biopsy_rep_umap_km ==="
python analysis/plot_biopsy_rep_umap_km.py \
    --tasks acr_cls acr_surv clad_surv death_surv \
    --n-neighbors 15

echo "=== Done ==="
