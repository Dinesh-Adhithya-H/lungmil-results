#!/usr/bin/env bash
#SBATCH --job-name=check_adata
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/he_umap/check_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/he_umap/check_%j.err
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
import anndata as ad
import numpy as np

adata = ad.read_h5ad('/lustre/groups/aih/dinesh.haridoss/datasets/adata_v3.h5ad', backed='r')

obs = adata.obs[['slide_name_clean','record_id','biopsy_date','acr_status','acr_binary',
                  'chunk_left','chunk_top','tile_left','tile_top']].copy()
adata.file.close()

print("n_slides:", obs['slide_name_clean'].nunique())
print("n_patients:", obs['record_id'].nunique())
print("acr_status unique:", obs['acr_status'].unique().tolist())
print("acr_binary unique:", obs['acr_binary'].unique().tolist())

# Patches per slide
pps = obs.groupby('slide_name_clean').size()
print(f"\nPatches per slide: min={pps.min()} median={pps.median():.0f} max={pps.max()}")

# Coordinate ranges
print(f"\nchunk_left:  {obs['chunk_left'].min():.0f} – {obs['chunk_left'].max():.0f}")
print(f"chunk_top:   {obs['chunk_top'].min():.0f} – {obs['chunk_top'].max():.0f}")
print(f"tile_left:   {obs['tile_left'].min():.0f} – {obs['tile_left'].max():.0f}")
print(f"tile_top:    {obs['tile_top'].min():.0f} – {obs['tile_top'].max():.0f}")

# Sample slide info
print("\nSample slides:")
print(obs[['slide_name_clean','record_id','biopsy_date','acr_status']].drop_duplicates('slide_name_clean').head(10).to_string())
PYEOF
