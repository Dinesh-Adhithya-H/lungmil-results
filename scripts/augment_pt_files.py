"""
augment_pt_files.py — Enrich existing .pt files with:

  1. BAL_umap     (N_bal, 2)   — per-cell 2D UMAP from BAL h5ad obsm['X_umap']
  2. HE_umap      (N_he,  2)   — per-patch 2D UMAP from HE  h5ad obsm['X_umap']
  3. BAL_pseudobulk (n_genes,) — summed raw gene counts across all BAL cells (for DEG)
  4. BAL_gene_names [list]     — gene names for pseudobulk vector
  5. cluster_names  {mod: {id: readable_name}} — human-readable cell/tissue type names
  6. BAL_cell_types (N_bal,)   — per-cell type string from resolution_v2

Strategy: for each .pt file match cells back to h5ad by patient_id + date proximity
(same 45-day window used in precompute_dataset.py). Extract obsm rows for those cells.

Run via sbatch — do NOT run on login node.
"""

import argparse, gc, warnings
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import anndata as ad

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
LUSTRE      = Path("/lustre/groups/aih/dinesh.haridoss")
DATA_DIR    = LUSTRE / "mil/dataset_cache_latest_fixed_large/samples"
OUT_DIR     = DATA_DIR  # augment in-place (save back to same dir)

BAL_H5AD    = str(LUSTRE / "datasets/08clad-4-annotated-v2-FIXED_date_from_id_dropped_recipient.h5ad")
HE_H5AD     = str(LUSTRE / "datasets/adata_v3.h5ad")

WINDOW_DAYS = 45

BAL_ID_COL   = "record_id";  BAL_DATE_COL  = "date_from_id"
HE_ID_COL    = "record_id";  HE_DATE_COL   = "biopsy_date"

BAL_CLUSTER_COL = "resolution_v2"
HE_CLUSTER_COL  = "subcluster_renamed"


# ── helpers ───────────────────────────────────────────────────────────────────

def safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def parse_date(s):
    """Parse ISO date string → pd.Timestamp or None."""
    try:
        return pd.to_datetime(str(s))
    except Exception:
        return None


def filter_obs(obs_df, id_col, date_col, patient_id, anchor_dt, window_days=45):
    """Return boolean mask of obs rows matching patient_id within window of anchor_dt."""
    mask_id = obs_df[id_col].astype(str).str.strip() == str(patient_id).strip()
    if anchor_dt is None:
        return mask_id
    col = obs_df[date_col]
    if hasattr(col, "cat"):
        col = col.astype(str)
    obs_dates = pd.to_datetime(col, errors="coerce")
    delta     = (obs_dates - anchor_dt).abs()
    mask_dt   = delta <= timedelta(days=window_days)
    return mask_id & mask_dt


# ── load h5ad obs (lazy — no feature matrix yet) ─────────────────────────────

print("Loading BAL obs ...", flush=True)
bal_adata = ad.read_h5ad(BAL_H5AD)
print(f"  BAL: {bal_adata.shape}  obsm={list(bal_adata.obsm.keys())}  layers={list(bal_adata.layers.keys())}", flush=True)

bal_obs       = bal_adata.obs[[BAL_ID_COL, BAL_DATE_COL, BAL_CLUSTER_COL]].copy()
bal_obs[BAL_DATE_COL] = pd.to_datetime(bal_obs[BAL_DATE_COL].astype(str), errors="coerce")

bal_umap      = np.array(bal_adata.obsm["X_umap"])           # (N_all, 2)
bal_raw       = bal_adata.layers["counts"]                    # (N_all, n_genes) sparse or dense
bal_gene_names = list(bal_adata.var_names)

# Build cluster-name lookup: resolution_v2 → label string
bal_cluster_names = {str(v): str(v) for v in bal_obs[BAL_CLUSTER_COL].unique()}

print("Loading HE obs ...", flush=True)
he_adata = ad.read_h5ad(HE_H5AD)
print(f"  HE:  {he_adata.shape}  obsm={list(he_adata.obsm.keys())}", flush=True)

he_obs       = he_adata.obs[[HE_ID_COL, HE_DATE_COL, HE_CLUSTER_COL]].copy()
he_obs[HE_DATE_COL] = pd.to_datetime(he_obs[HE_DATE_COL].astype(str), errors="coerce")

he_umap      = np.array(he_adata.obsm["X_umap"])             # (N_all, 2)
he_cluster_names = {str(v): str(v) for v in he_obs[HE_CLUSTER_COL].dropna().unique()}

print("h5ads loaded.", flush=True)

# ── process each .pt file ─────────────────────────────────────────────────────

pt_files = sorted(p for p in DATA_DIR.iterdir() if p.suffix == ".pt" and "_" not in p.stem)
print(f"\nProcessing {len(pt_files)} .pt files ...", flush=True)

n_aug_bal = 0; n_aug_he = 0; n_errors = 0

for i, pt_path in enumerate(pt_files):
    s = safe_load(pt_path)
    if s is None:
        n_errors += 1
        continue

    patient_id  = str(s.get("identifier", ""))
    anchor_time = parse_date(s.get("anchor_time"))
    mod_times   = s.get("modality_times") or {}
    inputs      = s.get("inputs", {})
    augmented   = False

    # ── BAL UMAP + pseudobulk ──────────────────────────────────────────────
    bal_cells = inputs.get("BAL_cells")
    if bal_cells is not None and isinstance(bal_cells, torch.Tensor) and bal_cells.numel() > 0:
        bal_mod_dt = parse_date(mod_times.get("BAL_cells")) or anchor_time
        mask = filter_obs(bal_obs, BAL_ID_COL, BAL_DATE_COL, patient_id, bal_mod_dt, WINDOW_DAYS)
        idx  = np.where(mask.values)[0]

        if len(idx) > 0:
            # UMAP coords — match count to n_patches in bag
            n_patches = bal_cells.shape[0]
            # use the first n_patches matched cells (same order as dataset loader)
            use_idx = idx[:n_patches] if len(idx) >= n_patches else idx
            umap_coords = bal_umap[use_idx]                          # (n, 2)
            if len(umap_coords) < n_patches:
                pad = np.full((n_patches - len(umap_coords), 2), np.nan)
                umap_coords = np.vstack([umap_coords, pad])

            inputs["BAL_umap"] = torch.tensor(umap_coords, dtype=torch.float32)

            # cell type strings
            cell_types = bal_obs[BAL_CLUSTER_COL].iloc[use_idx].astype(str).tolist()
            if len(cell_types) < n_patches:
                cell_types += ["unknown"] * (n_patches - len(cell_types))
            if "cluster_labels" not in s or s["cluster_labels"] is None:
                s["cluster_labels"] = {}
            s["cluster_labels"]["BAL_cells_named"] = cell_types[:n_patches]

            # pseudobulk — sum raw counts over ALL matching cells (not just n_patches)
            raw_sub = bal_raw[idx]
            if hasattr(raw_sub, "toarray"):
                raw_sub = raw_sub.toarray()
            pseudobulk = np.array(raw_sub).sum(axis=0).astype(np.float32)  # (n_genes,)
            inputs["BAL_pseudobulk"] = torch.tensor(pseudobulk, dtype=torch.float32)

            augmented = True
            n_aug_bal += 1

    # ── HE UMAP ───────────────────────────────────────────────────────────
    he_cells = inputs.get("HE_cells")
    if he_cells is not None and isinstance(he_cells, torch.Tensor) and he_cells.numel() > 0:
        he_mod_dt = parse_date(mod_times.get("HE_cells")) or anchor_time
        mask = filter_obs(he_obs, HE_ID_COL, HE_DATE_COL, patient_id, he_mod_dt, WINDOW_DAYS)
        idx  = np.where(mask.values)[0]

        if len(idx) > 0:
            n_patches = he_cells.shape[0]
            use_idx   = idx[:n_patches] if len(idx) >= n_patches else idx
            umap_coords = he_umap[use_idx]
            if len(umap_coords) < n_patches:
                pad = np.full((n_patches - len(umap_coords), 2), np.nan)
                umap_coords = np.vstack([umap_coords, pad])

            inputs["HE_umap"] = torch.tensor(umap_coords, dtype=torch.float32)

            # tissue type strings
            tissue_types = he_obs[HE_CLUSTER_COL].iloc[use_idx].astype(str).tolist()
            if len(tissue_types) < n_patches:
                tissue_types += ["unknown"] * (n_patches - len(tissue_types))
            if "cluster_labels" not in s or s["cluster_labels"] is None:
                s["cluster_labels"] = {}
            s["cluster_labels"]["HE_cells_named"] = tissue_types[:n_patches]

            augmented = True
            n_aug_he += 1

    # ── cluster name lookup dict ───────────────────────────────────────────
    if "cluster_names" not in s:
        s["cluster_names"] = {}
    s["cluster_names"]["BAL"] = bal_cluster_names
    s["cluster_names"]["HE"]  = he_cluster_names

    # ── gene names (stored once per file — same for all) ──────────────────
    if "BAL_gene_names" not in s:
        s["BAL_gene_names"] = bal_gene_names

    s["inputs"] = inputs

    # save back
    try:
        torch.save(s, pt_path)
    except Exception as e:
        print(f"  [ERROR] saving {pt_path.name}: {e}", flush=True)
        n_errors += 1
        continue

    if (i + 1) % 200 == 0:
        print(f"  [{i+1}/{len(pt_files)}]  BAL augmented: {n_aug_bal}  HE augmented: {n_aug_he}  errors: {n_errors}", flush=True)

    del s
    gc.collect()

print(f"\nDone.")
print(f"  BAL umap+pseudobulk added: {n_aug_bal}")
print(f"  HE  umap added:            {n_aug_he}")
print(f"  Errors:                    {n_errors}")
print(f"\nNew .pt fields added:")
print(f"  inputs.BAL_umap         (N_bal, 2)    — per-cell UMAP from X_umap")
print(f"  inputs.HE_umap          (N_he,  2)    — per-patch UMAP from X_umap")
print(f"  inputs.BAL_pseudobulk   (2000,)       — summed raw counts for DEG")
print(f"  cluster_labels.BAL_cells_named [list] — resolution_v2 cell type per cell")
print(f"  cluster_labels.HE_cells_named  [list] — subcluster_renamed tissue type per patch")
print(f"  cluster_names.BAL/HE           dict   — cluster id → readable name")
print(f"  BAL_gene_names                 list   — gene name index for pseudobulk")
