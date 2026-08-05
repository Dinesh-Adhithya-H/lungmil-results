#!/bin/bash
#SBATCH --job-name=death_biopsy_umap
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/death_biopsy_umap_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/death_biopsy_umap_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

source ~/.bashrc
conda activate chicago_mil
cd /home/aih/dinesh.haridoss/chicago_mil

echo "=== death_surv biopsy UMAP ==="
python analysis/plot_biopsy_rep_umap_km.py \
    --tasks death_surv \
    --n-neighbors 15

echo "=== Copying to death/ directory ==="
mkdir -p figures/interpretability/death
cp figures/interpretability/death_surv/biopsy_rep_umap.png figures/interpretability/death/biopsy_rep_umap.png
cp figures/interpretability/death_surv/biopsy_rep_umap.pdf figures/interpretability/death/biopsy_rep_umap.pdf

echo "=== Done ==="
