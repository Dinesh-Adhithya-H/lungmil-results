#!/bin/bash
#SBATCH --job-name=biopsy_rep_umap_km
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/biopsy_rep_umap_km_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/biopsy_rep_umap_km_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

source ~/.bashrc
conda activate chicago_mil
cd /home/aih/dinesh.haridoss/chicago_mil

echo "=== biopsy_rep_umap_km ==="
python analysis/plot_biopsy_rep_umap_km.py \
    --tasks acr_cls acr_surv clad_surv death_surv \
    --n-neighbors 15

echo "=== Done ==="
