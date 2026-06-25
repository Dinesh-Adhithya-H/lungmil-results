#!/usr/bin/env python3
"""
Extract cluster name → tissue type mappings from original AnnData files.

HE:  subcluster_renamed → tissue_type  (e.g. "0_1" → "Alveolar")
CT:  numeric cluster_id (no tissue annotations, stays numeric)
BAL: cell type names already in mil_v2 vocab (CCR7+ DC1, etc.) — no mapping needed

Saves JSON to results/cluster_name_maps/{MOD}_cluster_map.json
"""
import json
from pathlib import Path

OUT = Path("/home/aih/dinesh.haridoss/chicago_mil/results/cluster_name_maps")
OUT.mkdir(parents=True, exist_ok=True)

# ── HE: subcluster_renamed → tissue_type ─────────────────────────────────────
print("Loading HE AnnData obs (backed, read-only)...")
import anndata as ad

he_path = "/lustre/groups/aih/dinesh.haridoss/datasets/adata_v3.h5ad"
adata_he = ad.read_h5ad(he_path, backed="r")
print(f"  HE shape: {adata_he.shape}")
print(f"  obs cols: {list(adata_he.obs.columns)}")

he_obs = adata_he.obs[["subcluster_renamed", "tissue_type"]].copy()
adata_he.file.close()

# Most common tissue_type per subcluster
he_map = {}
for sub, grp in he_obs.groupby("subcluster_renamed"):
    vc = grp["tissue_type"].value_counts()
    he_map[str(sub)] = str(vc.index[0]) if len(vc) > 0 else str(sub)

he_out = OUT / "HE_cluster_map.json"
json.dump(he_map, open(he_out, "w"), indent=2)
print(f"\nHE cluster map ({len(he_map)} entries) → {he_out}")
for k, v in sorted(he_map.items())[:10]:
    print(f"  {k!r:12s} → {v!r}")

# ── CT: no tissue annotations; create pass-through map (id → "CT-{id}") ─────
print("\nLoading CT AnnData obs...")
ct_path = "/lustre/groups/aih/dinesh.haridoss/datasets/combined_ct_embeddings_processed.h5ad"
adata_ct = ad.read_h5ad(ct_path, backed="r")
print(f"  CT shape: {adata_ct.shape}")
print(f"  CT obs cols: {list(adata_ct.obs.columns)[:15]}")

# Check if there's any cluster annotation
ct_map = {}
for col in adata_ct.obs.columns:
    if any(k in col.lower() for k in ["cluster", "tissue", "type", "annot", "leiden"]):
        vals = sorted(adata_ct.obs[col].dropna().unique())
        print(f"  CT obs[{col}] ({len(vals)} unique): {vals[:5]}")
adata_ct.file.close()

# CT has no tissue annotation — use numeric IDs as-is (no map needed)
ct_out = OUT / "CT_cluster_map.json"
json.dump(ct_map, open(ct_out, "w"), indent=2)
print(f"CT cluster map (empty — no tissue annotations) → {ct_out}")

print(f"\nDone → {OUT}")
print(f"\nscp 'dinesh.haridoss@hpc-submit01.scidom.de:{OUT}/*.json' ~/Desktop/mil_plots/")
