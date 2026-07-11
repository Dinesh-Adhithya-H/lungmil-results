#!/usr/bin/env bash
#SBATCH --job-name=he_umap_annotate
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/he_umap/annotate_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/he_umap/annotate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

OUTDIR="/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/he_umap"
mkdir -p "$OUTDIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
import anndata as ad
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

H5AD   = "/lustre/groups/aih/dinesh.haridoss/datasets/adata_v3.h5ad"
OUTDIR = "/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/he_umap"

# Annotation map: leiden_0.155 cluster → tissue type
ANNOTATION = {
    "0":  "Alveolar with hemorrhage and inflammation",
    "1":  "Alveolar with empty spaces",
    "2":  "Alveolar with hemorrhage and inflammation",
    "3":  "Unknown",
    "4":  "Alveolar with hemorrhage and inflammation",
    "5":  "Alveolar",
    "6":  "Bronchial",
    "7":  "Alveolar with hemorrhage and inflammation",
    "8":  "Bronchial",
    "9":  "Lymphocytoplasmic inflammation",
    "10": "Cartilage",
}

# Colors matching the original figure
TYPE_COLORS = {
    "Alveolar":                                  "#E53030",   # bright red
    "Bronchial":                                 "#1A72CC",   # bright blue
    "Cartilage":                                 "#F5C800",   # bright yellow
    "Alveolar with empty spaces":                "#00BB44",   # bright green
    "Alveolar with hemorrhage and inflammation": "#FF6F00",   # bright orange
    "Lymphocytoplasmic inflammation":            "#CC00CC",   # bright magenta
    "Unknown":                                   "#AAAAAA",   # grey
}

print("Loading h5ad (backed)...", flush=True)
adata = ad.read_h5ad(H5AD, backed='r')
print(f"  Shape: {adata.shape}", flush=True)

print("Reading X_umap and obs...", flush=True)
umap   = np.array(adata.obsm['X_umap'])
leiden = adata.obs['leiden_0.155'].astype(str).values.copy()
adata.file.close()

# Map to annotation
print("Mapping annotations...", flush=True)
tissue_type = np.array([ANNOTATION.get(c, "Unknown") for c in leiden])

# ── Plot ──────────────────────────────────────────────────────────────────────
unique_types = [
    "Alveolar",
    "Alveolar with empty spaces",
    "Alveolar with hemorrhage and inflammation",
    "Bronchial",
    "Cartilage",
    "Lymphocytoplasmic inflammation",
    "Unknown",
]

rng = np.random.default_rng(42)
NSAMPLE = 1_200_000

# Stratified subsample
chosen = []
for t in unique_types:
    where = np.where(tissue_type == t)[0]
    if len(where) == 0:
        continue
    k = min(max(1, int(NSAMPLE * len(where) / len(tissue_type) * 1.2)), len(where))
    chosen.append(rng.choice(where, k, replace=False))
chosen = np.concatenate(chosen)
rng.shuffle(chosen)

xy  = umap[chosen]
lbl = tissue_type[chosen]

c_arr = np.array([TYPE_COLORS[l] for l in lbl])

fig, ax = plt.subplots(figsize=(12, 10))
ax.scatter(xy[:, 0], xy[:, 1],
           c=c_arr, s=0.4, alpha=0.45, linewidths=0, rasterized=True)

ax.set_title("H&E Patches — Tissue Type Annotation", fontsize=15, fontweight='bold', pad=14)
ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)

# Centroid labels on plot
for t in unique_types:
    mask = lbl == t
    if mask.sum() == 0:
        continue
    cx, cy = xy[mask, 0].mean(), xy[mask, 1].mean()
    ax.text(cx, cy, t, fontsize=7.5, fontweight='bold',
            color=TYPE_COLORS[t], ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))

# Legend
handles = [Line2D([0],[0], marker='o', color='w',
                  markerfacecolor=TYPE_COLORS[t], markersize=10, label=t)
           for t in unique_types if t != "Unknown"]
ax.legend(handles=handles, title='Tissue Type', fontsize=9,
          title_fontsize=10, loc='lower left', frameon=True,
          framealpha=0.85, edgecolor='#cccccc')

plt.tight_layout()
out = f"{OUTDIR}/umap_tissue_type_annotated.png"
fig.savefig(out, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}", flush=True)

# ── Also save annotation to h5ad ──────────────────────────────────────────────
print("\nAdding tissue_type column to h5ad...", flush=True)
adata2 = ad.read_h5ad(H5AD)   # full load for writing
adata2.obs['tissue_type'] = [ANNOTATION.get(c, "Unknown")
                              for c in adata2.obs['leiden_0.155'].astype(str)]
adata2.obs['tissue_type'] = adata2.obs['tissue_type'].astype('category')

# Store matching colors in uns
adata2.uns['tissue_type_colors'] = [TYPE_COLORS[t] for t in adata2.obs['tissue_type'].cat.categories]

adata2.write_h5ad(H5AD)
print(f"Saved tissue_type to {H5AD}", flush=True)
print("Done.", flush=True)
PYEOF
