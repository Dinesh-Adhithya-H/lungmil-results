"""
Inspect h5ad files for UMAP coordinates and cluster annotations.
Prints obsm keys, obs columns, and a sample of UMAP coords per modality.
"""
import anndata as ad
import numpy as np
from pathlib import Path

LUSTRE = Path("/lustre/groups/aih/dinesh.haridoss")
PRECOMPUTED = LUSTRE / "datasets/precomputed"

files = {
    "BAL (scRNA)":    str(LUSTRE / "datasets/08clad-4-annotated-v2-FIXED_date_from_id_dropped_recipient.h5ad"),
    "HE (histology)": str(LUSTRE / "datasets/adata_v3.h5ad"),
    "CT (imaging)":   str(LUSTRE / "datasets/combined_ct_embeddings_processed.h5ad"),
    "BAL centroids":  str(PRECOMPUTED / "BAL_centroids.h5ad"),
    "HE centroids":   str(PRECOMPUTED / "HE_centroids.h5ad"),
    "CT centroids":   str(PRECOMPUTED / "CT_centroids.h5ad"),
}

for name, path in files.items():
    p = Path(path)
    if not p.exists():
        print(f"\n[MISSING] {name}: {path}")
        continue
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {path}")
    try:
        adata = ad.read_h5ad(path, backed="r")
        print(f"  shape: {adata.shape}")
        print(f"  obsm keys: {list(adata.obsm.keys())}")
        print(f"  obs columns: {list(adata.obs.columns[:20])}")
        # Print UMAP coords if available
        for key in ["X_umap", "X_umap2", "umap", "X_scVI_umap"]:
            if key in adata.obsm:
                coords = np.array(adata.obsm[key])
                print(f"  {key}: shape={coords.shape}, range=[{coords.min():.2f}, {coords.max():.2f}]")
                print(f"    first 3 rows: {coords[:3]}")
        # Print clustering columns
        clust_cols = [c for c in adata.obs.columns
                      if any(k in c.lower() for k in ["leiden", "cluster", "subcluster", "resolution", "cell_type"])]
        print(f"  clustering cols: {clust_cols[:10]}")
        if "record_id" in adata.obs.columns:
            print(f"  patients: {sorted(adata.obs['record_id'].unique())[:10]} ...")
            print(f"  n_patients: {adata.obs['record_id'].nunique()}")
        adata.file.close()
    except Exception as e:
        print(f"  ERROR: {e}")

print("\nDone.")
