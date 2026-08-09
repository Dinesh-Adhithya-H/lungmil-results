#!/bin/bash
#SBATCH --job-name=biopsy_umap_export
#SBATCH --output=logs/biopsy_umap_export_%j.out
#SBATCH --error=logs/biopsy_umap_export_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

set -euo pipefail

REPO=/ictstr01/home/aih/dinesh.haridoss/chicago_mil
conda activate chicago_mil_torch26

cd "$REPO"
mkdir -p logs

echo "=== Exporting biopsy UMAP embeddings ==="
python patient_explorer/export_biopsy_umap.py

echo "=== Done ==="
