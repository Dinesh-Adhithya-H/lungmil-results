#!/usr/bin/env bash
#SBATCH --job-name=precompute_v2
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs/%j_precompute_v2.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs/%j_precompute_v2.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=512G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

# Rerun precompute to produce enriched .pt files containing:
#   inputs.BAL_umap     (N_bal, 2)  — per-cell UMAP from X_umap in BAL h5ad
#   inputs.HE_umap      (N_he,  2)  — per-patch UMAP from X_umap in HE  h5ad
#   cluster_labels.{mod}            — per-cell leiden labels (resolution_v2 / subcluster_renamed / leiden)
#   bag_counts_raw.{mod}            — raw proportions (non-CLR, non-negative, sums to 1)
#   bag_instance_cluster_ids.{mod}  — per-instance cluster IDs
#   bag_cluster_names.{mod}         — cluster name list
# Gene names saved once to info.json (bal_gene_names key)

set -uo pipefail
export PYTHONUNBUFFERED=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

CACHE_DIR=/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2

mkdir -p "$CACHE_DIR/samples"
mkdir -p /home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs

echo "======================================"
echo " Precompute v2 — enriched .pt files"
echo "======================================"
echo "Host     : $(hostname)"
echo "Date     : $(date)"
echo "Cache    : $CACHE_DIR"
echo "Mem      : ${SLURM_MEM_PER_NODE} MB"
echo ""

python3 -u /home/aih/dinesh.haridoss/chicago_mil/data_prep/precompute_dataset.py \
    --cache_dir "$CACHE_DIR" \
    --workers 1

echo ""
echo "===== DONE: $(date) ====="
