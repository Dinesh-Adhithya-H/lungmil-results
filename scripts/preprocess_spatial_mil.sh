#!/usr/bin/env bash
#SBATCH --job-name=spatial_preproc
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/preproc_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/preproc_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

OUTDIR="/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil"
mkdir -p "$OUTDIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
"""
Preprocessing for 2-level Spatial ABMIL.

Per slide:
  1. DBSCAN on (chunk_left, chunk_top) → spatial neighbourhoods
  2. For each neighbourhood: save ALL patch features (N_i x 1024) from UNI backbone
     so that Level-1 ABMIL can be trained over individual patches within each region

Saved per slide:
  {
    'slide':    str,
    'patient':  str,
    'label':    int,          # ACR binary
    'acr_grade': str,
    'clusters': [             # list of K neighbourhoods
      {
        'feats':     Tensor(N_i, 1024),   # UNI patch features
        'coords':    Tensor(N_i, 2),      # (x, y) of each patch
        'centroid':  Tensor(2,),
        'n_patches': int,
      }, ...
    ],
    'noise_frac': float,      # fraction of patches in empty glass
    'n_patches_total': int,
  }
"""
import anndata as ad
import numpy as np
import pandas as pd
import torch
import h5py
from sklearn.cluster import DBSCAN
from pathlib import Path

H5AD      = "/lustre/groups/aih/dinesh.haridoss/datasets/adata_v3.h5ad"
CSV       = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
OUTDIR    = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil")
FEAT_DIR  = OUTDIR / "slide_cluster_feats"
FEAT_DIR.mkdir(exist_ok=True)

# DBSCAN: eps=1000px — connects tissue regions, splits on empty glass between sections
# At tile stride 112px, eps=1000 ≈ 9 tile-widths → joins tissue blobs, gives 5-20 clusters/slide
DBSCAN_EPS       = 1000
DBSCAN_MIN_PTS   = 10
MIN_CLUSTER_SIZE = 20   # discard tiny fragments

# ── Labels from CSV ────────────────────────────────────────────────────────────
df_csv = pd.read_csv(CSV)
he_df  = df_csv[df_csv['has_HE'] == True].copy()
print(f"HE slides in CSV : {len(he_df)}  ({he_df['patient_id'].nunique()} patients)", flush=True)
print(f"ACR binary — 0: {(he_df['label']==0).sum()}  1: {(he_df['label']==1).sum()}", flush=True)

# ── Load h5ad obs (coords + metadata only — fast) ─────────────────────────────
print("\nLoading h5ad obs...", flush=True)
adata = ad.read_h5ad(H5AD, backed='r')
obs = adata.obs[['chunk_left','chunk_top','slide_name_clean',
                  'record_id','biopsy_date','acr_binary']].copy()
obs['_row'] = np.arange(len(obs))
obs['chunk_left'] = obs['chunk_left'].astype(np.float32)
obs['chunk_top']  = obs['chunk_top'].astype(np.float32)

# Build slide → label from h5ad acr_binary (consistent with CSV)
slide_meta = (obs[['slide_name_clean','record_id','biopsy_date','acr_binary']]
              .drop_duplicates('slide_name_clean'))
slide2label   = dict(zip(slide_meta['slide_name_clean'], slide_meta['acr_binary'].astype(int)))
slide2patient = dict(zip(slide_meta['slide_name_clean'], slide_meta['record_id']))

# Also pull acr_grade from CSV via matching record_id + biopsy order
slide2grade = {}
for pid, grp in he_df.groupby('patient_id'):
    grp_sorted = grp.sort_values('file').reset_index(drop=True)
    sl_sorted  = (slide_meta[slide_meta['record_id'] == pid]
                  .sort_values('biopsy_date').reset_index(drop=True))
    for i, row in sl_sorted.iterrows():
        if i < len(grp_sorted):
            slide2grade[row['slide_name_clean']] = grp_sorted.loc[i, 'acr_grade']

slides = obs['slide_name_clean'].unique()
print(f"  {len(obs):,} patches | {len(slides)} slides", flush=True)

# ── Open h5py directly for feature reads ──────────────────────────────────────
print("Opening h5py...", flush=True)
h5f    = h5py.File(H5AD, 'r')
X_data = h5f['X']   # (14816249, 1024) float32 — UNI features

# ── Per-slide processing ───────────────────────────────────────────────────────
summary_rows = []
n_slides = len(slides)

for si, slide in enumerate(slides):
    sl_obs   = obs[obs['slide_name_clean'] == slide]
    row_idx  = sl_obs['_row'].values
    coords   = sl_obs[['chunk_left','chunk_top']].values.astype(np.float32)
    n_total  = len(sl_obs)

    # DBSCAN
    db      = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_PTS, n_jobs=8)
    clabels = db.fit_predict(coords)   # -1 = noise

    # Load features for this slide (sort rows for sequential h5py reads = fast)
    sort_ord    = np.argsort(row_idx)
    sorted_rows = row_idx[sort_ord]
    feats_sort  = X_data[sorted_rows.tolist()]      # (N, 1024)
    unsort      = np.argsort(sort_ord)
    feats       = feats_sort[unsort]                # restore original order

    # Build cluster list — full patch features per neighbourhood
    cluster_list = []
    for cid in sorted(set(clabels[clabels >= 0])):
        mask = clabels == cid
        if mask.sum() < MIN_CLUSTER_SIZE:
            continue
        c_feats  = torch.from_numpy(feats[mask].astype(np.float32))   # (N_i, 1024)
        c_coords = torch.from_numpy(coords[mask])                      # (N_i, 2)
        cluster_list.append({
            'feats':    c_feats,
            'coords':   c_coords,
            'centroid': c_coords.mean(dim=0),
            'n_patches': int(mask.sum()),
        })

    if len(cluster_list) == 0:
        if (si+1) % 50 == 0:
            print(f"  [{si+1}/{n_slides}] {slide}: 0 clusters — skipping", flush=True)
        continue

    noise_frac = float((clabels == -1).mean())
    safe       = slide.replace('/', '_')

    torch.save({
        'slide':            slide,
        'patient':          slide2patient[slide],
        'label':            slide2label[slide],
        'acr_grade':        slide2grade.get(slide, 'unknown'),
        'clusters':         cluster_list,
        'noise_frac':       noise_frac,
        'n_patches_total':  n_total,
    }, FEAT_DIR / f"{safe}.pt")

    summary_rows.append({
        'slide':      slide,
        'patient':    slide2patient[slide],
        'label':      slide2label[slide],
        'acr_grade':  slide2grade.get(slide, 'unknown'),
        'n_clusters': len(cluster_list),
        'n_patches':  n_total,
        'noise_frac': noise_frac,
    })

    if (si+1) % 50 == 0:
        avg_k = np.mean([c['n_patches'] for c in cluster_list])
        print(f"  [{si+1}/{n_slides}] {slide}: {len(cluster_list)} clusters "
              f"(avg {avg_k:.0f} patches), noise={100*noise_frac:.0f}%", flush=True)

h5f.close()

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTDIR / "slide_cluster_summary.csv", index=False)

print(f"\nSaved {len(summary)} slides to {FEAT_DIR}", flush=True)
print("\nMean stats by ACR label:")
print(summary.groupby('label')[['n_clusters','n_patches','noise_frac']].mean().round(2).to_string())
print("\nPreprocessing complete.", flush=True)
PYEOF
