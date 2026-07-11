#!/usr/bin/env bash
#SBATCH --job-name=umap_spatial
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/umap_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/umap_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
"""
Two UMAPs from 2-level spatial ABMIL representations:

  1. Neighbourhood UMAP  — one point per spatial cluster (neighbourhood rep, 256-d)
     Coloured by:  (a) neighbourhood attention weight
                   (b) ACR label of parent slide
                   (c) tissue type (dominant type in cluster by patch count)

  2. Patch UMAP  — one point per patch (patch rep after proj, 256-d), subsampled
     Coloured by:  (a) combined attention = nbhd_attn × patch_attn  (red=danger, green=safe)
                   (b) tissue type  (danger-aware palette)
                   (c) ACR label of parent slide

Tissue colour palette — safe (cool) → dangerous (warm):
  Alveolar                             → #4CAF50  (green)
  Alveolar with empty spaces           → #2196F3  (blue)
  Bronchial                            → #9C27B0  (purple)
  Cartilage                            → #00BCD4  (cyan)
  Alveolar with hemorrhage/inflammation→ #FF9800  (orange)  ← borderline
  Lymphocytoplasmic inflammation       → #F44336  (red)     ← danger
  Unknown                              → #9E9E9E  (grey)
"""
import torch, torch.nn as nn
import numpy as np, pandas as pd
import anndata as ad
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold

FEAT_DIR = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/slide_cluster_feats")
OUTDIR   = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil")
UMAP_DIR = OUTDIR / "umap_plots"
UMAP_DIR.mkdir(exist_ok=True)
H5AD     = "/lustre/groups/aih/dinesh.haridoss/datasets/adata_v3.h5ad"

IN_DIM, HIDDEN, DROPOUT = 1024, 256, 0.25
MAX_PATCHES  = 512
N_SPLITS     = 5
PATCH_SAMPLE = 50_000   # patches to include in patch UMAP

# ── Tissue colour palette — safe (cool) to dangerous (warm) ───────────────────
TISSUE_COLORS = {
    "Alveolar":                                  "#4CAF50",   # green   — safe
    "Alveolar with empty spaces":                "#2196F3",   # blue    — safe
    "Bronchial":                                 "#9C27B0",   # purple  — neutral
    "Cartilage":                                 "#00BCD4",   # cyan    — safe
    "Alveolar with hemorrhage and inflammation": "#FF9800",   # orange  — warning
    "Lymphocytoplasmic inflammation":            "#F44336",   # red     — danger
    "Unknown":                                   "#9E9E9E",   # grey
}

# ── Model (must match training) ────────────────────────────────────────────────
class GatedAttentionPool(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(in_dim, hidden), nn.Sigmoid())
        self.w = nn.Linear(hidden, 1, bias=False)
    def forward(self, h):
        a = self.w(self.V(h) * self.U(h))
        a = torch.softmax(a, dim=1)
        return (a * h).sum(dim=1), a.squeeze(-1)

class SpatialABMIL2Level(nn.Module):
    def __init__(self, in_dim=1024, hidden=256, dropout=0.25):
        super().__init__()
        self.patch_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.patch_attn = GatedAttentionPool(hidden, hidden // 2)
        self.nbhd_proj  = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.nbhd_attn  = GatedAttentionPool(hidden, hidden // 2)
        self.head       = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward_with_reps(self, clusters):
        """Returns logit + all intermediate representations."""
        nbhd_reps, patch_reps_list, patch_attns = [], [], []
        for c in clusters:
            h = self.patch_proj(c.unsqueeze(0))          # (1, N_i, 256)
            rep, pa = self.patch_attn(h)                 # (1, 256), (1, N_i)
            nbhd_reps.append(rep)
            patch_reps_list.append(h.squeeze(0))         # (N_i, 256) — after proj
            patch_attns.append(pa.squeeze(0))            # (N_i,)
        H   = self.nbhd_proj(torch.stack(nbhd_reps, dim=1))   # (1, K, 256)
        slide_rep, na = self.nbhd_attn(H)
        logit = self.head(slide_rep).squeeze(-1)
        nbhd_rep_matrix = torch.stack(nbhd_reps, dim=0).squeeze(1)  # (K, 256)
        return logit, na.squeeze(0), patch_attns, nbhd_rep_matrix, patch_reps_list

class SpatialMILDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        d = torch.load(self.paths[i], map_location='cpu', weights_only=False)
        clusters = []
        for c in d['clusters']:
            f = c['feats']
            if f.shape[0] > MAX_PATCHES:
                f = f[torch.randperm(f.shape[0])[:MAX_PATCHES]]
            clusters.append(f)
        return (clusters,
                torch.tensor(d['label'], dtype=torch.float32).unsqueeze(0),
                d['slide'], d['patient'],
                [c['coords'].numpy() for c in d['clusters']],
                [c['n_patches'] for c in d['clusters']])

def collate_fn(b): return b[0]

# ── Load all slides ────────────────────────────────────────────────────────────
all_paths = sorted(FEAT_DIR.glob("*.pt"))
labels, patients = [], []
for p in all_paths:
    d = torch.load(p, map_location='cpu', weights_only=False)
    labels.append(d['label']); patients.append(d['patient'])
labels, patients = np.array(labels), np.array(patients)
print(f"Slides: {len(all_paths)}  ACR+={labels.sum()}", flush=True)

cv = StratifiedGroupKFold(n_splits=N_SPLITS)
fold_splits = list(cv.split(all_paths, labels, patients))

# ── Collect representations from folds 0 and 1 ────────────────────────────────
# Neighbourhood level
nbhd_reps_all   = []   # (256,) each
nbhd_attn_all   = []   # scalar
nbhd_label_all  = []   # ACR label of parent slide
nbhd_prob_all   = []   # model prob
nbhd_slide_all  = []   # slide name
nbhd_coords_all = []   # centroid (x, y)

# Patch level (subsampled)
patch_reps_all     = []  # (256,)
patch_comb_attn_all= []  # nbhd_attn * patch_attn
patch_label_all    = []  # ACR label
patch_coords_all   = []  # (x, y)
patch_slide_all    = []

for fold in [0, 1]:
    ckpt = OUTDIR / f"spatial_abmil_fold{fold}.pt"
    if not ckpt.exists():
        print(f"Fold {fold}: no checkpoint", flush=True); continue

    model = SpatialABMIL2Level(IN_DIM, HIDDEN, DROPOUT)
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    model.eval()

    _, te_idx = fold_splits[fold]
    te_ds = SpatialMILDataset([all_paths[i] for i in te_idx])
    te_dl = DataLoader(te_ds, batch_size=1, shuffle=False,
                       collate_fn=collate_fn, num_workers=4)

    rng = np.random.default_rng(42)

    with torch.no_grad():
        for clusters, label, slide_name, patient, coords, n_patches in te_dl:
            logit, na, patch_attns, nbhd_rep_mat, patch_rep_list = \
                model.forward_with_reps(clusters)
            prob = torch.sigmoid(logit).item()
            lbl  = int(label.item())
            na_np = na.numpy()

            # Neighbourhood level — all K clusters
            for k in range(len(na_np)):
                cx = float(coords[k][:, 0].mean())
                cy = float(coords[k][:, 1].mean())
                nbhd_reps_all.append(nbhd_rep_mat[k].numpy())
                nbhd_attn_all.append(float(na_np[k]))
                nbhd_label_all.append(lbl)
                nbhd_prob_all.append(prob)
                nbhd_slide_all.append(slide_name)
                nbhd_coords_all.append((cx, cy))

            # Patch level — random subsample per cluster
            n_clusters = len(clusters)
            for k in range(n_clusters):
                pr  = patch_rep_list[k].numpy()    # (N_k, 256)
                pa  = patch_attns[k].numpy()        # (N_k,)
                cr  = coords[k]                     # (N_k, 2)
                comb = pa * float(na_np[k])

                n_k = pr.shape[0]
                # Sample proportional to cluster size, max 200 patches per cluster
                n_take = min(n_k, 200)
                idx = rng.choice(n_k, n_take, replace=False)
                for i in idx:
                    patch_reps_all.append(pr[i])
                    patch_comb_attn_all.append(float(comb[i]))
                    patch_label_all.append(lbl)
                    patch_coords_all.append((float(cr[i, 0]), float(cr[i, 1])))
                    patch_slide_all.append(slide_name)

print(f"Neighbourhoods: {len(nbhd_reps_all)}", flush=True)
print(f"Patches (pre-subsample): {len(patch_reps_all)}", flush=True)

nbhd_reps_arr = np.stack(nbhd_reps_all)    # (N_nbhd, 256)
patch_reps_arr = np.stack(patch_reps_all)  # (N_patch, 256)

# Subsample patches for UMAP if too many
if len(patch_reps_arr) > PATCH_SAMPLE:
    rng2 = np.random.default_rng(0)
    idx  = rng2.choice(len(patch_reps_arr), PATCH_SAMPLE, replace=False)
    patch_reps_arr       = patch_reps_arr[idx]
    patch_comb_attn_all  = [patch_comb_attn_all[i]  for i in idx]
    patch_label_all      = [patch_label_all[i]       for i in idx]
    patch_coords_all     = [patch_coords_all[i]      for i in idx]
    patch_slide_all      = [patch_slide_all[i]       for i in idx]

print(f"Patches for UMAP: {len(patch_reps_arr)}", flush=True)

# ── Load tissue type for patches ───────────────────────────────────────────────
print("Loading tissue type from h5ad...", flush=True)
adata = ad.read_h5ad(H5AD, backed='r')
obs = adata.obs[['slide_name_clean', 'chunk_left', 'chunk_top', 'tissue_type']].copy()
obs['chunk_left'] = obs['chunk_left'].astype(np.float32)
obs['chunk_top']  = obs['chunk_top'].astype(np.float32)

needed_slides = set(patch_slide_all + nbhd_slide_all)
obs_sub = obs[obs['slide_name_clean'].isin(needed_slides)]
slide_tissue_map = {}
for slide, grp in obs_sub.groupby('slide_name_clean'):
    slide_tissue_map[slide] = {
        (round(float(r.chunk_left)), round(float(r.chunk_top))): r.tissue_type
        for _, r in grp.iterrows()
    }
print(f"  Built tissue maps for {len(slide_tissue_map)} slides", flush=True)

def get_tissue(slide, x, y):
    return slide_tissue_map.get(slide, {}).get((round(float(x)), round(float(y))), 'Unknown')

# Tissue for patches
patch_tissue = [get_tissue(s, x, y)
                for s, (x, y) in zip(patch_slide_all, patch_coords_all)]

# Dominant tissue per neighbourhood (majority vote over patch coords)
nbhd_tissue = []
for slide_nm, cx, cy in zip(nbhd_slide_all, *zip(*nbhd_coords_all)):
    nbhd_tissue.append(get_tissue(slide_nm, cx, cy))

# ── Run UMAP ───────────────────────────────────────────────────────────────────
reducer_kw = dict(n_neighbors=30, min_dist=0.1, n_components=2,
                  metric='euclidean', random_state=42, n_jobs=16)

print("Running neighbourhood UMAP...", flush=True)
nbhd_emb = umap.UMAP(**reducer_kw).fit_transform(nbhd_reps_arr)
print("Running patch UMAP...", flush=True)
patch_emb = umap.UMAP(**reducer_kw).fit_transform(patch_reps_arr)
print("UMAP done.", flush=True)

np.save(UMAP_DIR / "nbhd_umap.npy",  nbhd_emb)
np.save(UMAP_DIR / "patch_umap.npy", patch_emb)

# ── Helper ─────────────────────────────────────────────────────────────────────
def despine(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)

def scatter_colored(ax, emb, colors, title, s=3, alpha=0.5, cmap=None,
                    vmin=None, vmax=None, add_cbar=False, fig=None):
    sc = ax.scatter(emb[:,0], emb[:,1], c=colors, s=s, alpha=alpha,
                    linewidths=0, cmap=cmap, vmin=vmin, vmax=vmax,
                    rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)
    despine(ax)
    if add_cbar and fig is not None:
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=7)
    return sc

tissue_legend = [mpatches.Patch(color=c, label=t)
                 for t, c in TISSUE_COLORS.items()]

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Neighbourhood UMAP (3 colourings)
# ════════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))

# (a) Neighbourhood attention weight
attn_arr = np.array(nbhd_attn_all)
scatter_colored(axes1[0], nbhd_emb, attn_arr,
                'Neighbourhood attention weight',
                cmap='RdYlGn_r', vmin=attn_arr.min(), vmax=attn_arr.max(),
                add_cbar=True, fig=fig1)

# (b) ACR label
lbl_arr = np.array(nbhd_label_all)
for lv, col, nm in [(0,'#2196F3','ACR−'), (1,'#F44336','ACR+')]:
    m = lbl_arr == lv
    axes1[1].scatter(nbhd_emb[m,0], nbhd_emb[m,1],
                     c=col, s=3, alpha=0.5, linewidths=0, label=nm,
                     rasterized=True)
axes1[1].set_title('ACR label', fontsize=10)
axes1[1].set_xlabel("UMAP 1", fontsize=8); axes1[1].set_ylabel("UMAP 2", fontsize=8)
axes1[1].legend(fontsize=8, markerscale=3, frameon=False)
despine(axes1[1])

# (c) Dominant tissue type
tc = [TISSUE_COLORS.get(t, '#9E9E9E') for t in nbhd_tissue]
scatter_colored(axes1[2], nbhd_emb, tc,
                'Dominant tissue type\n(safe=green/blue → danger=red)')
fig1.legend(handles=tissue_legend, loc='lower center', ncol=4,
            fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))

plt.suptitle('Neighbourhood UMAP — 256-d neighbourhood representations',
             fontsize=13, y=1.01)
plt.tight_layout()
fig1.savefig(UMAP_DIR / "neighbourhood_umap.png", dpi=150, bbox_inches='tight')
plt.close(fig1)
print("Saved neighbourhood_umap.png", flush=True)

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Patch UMAP (3 colourings)
# ════════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

# (a) Combined attention = nbhd_attn × patch_attn  (log scale)
comb_arr = np.array(patch_comb_attn_all)
comb_log = np.log1p(comb_arr)
vmin_c, vmax_c = np.percentile(comb_log, 2), np.percentile(comb_log, 98)
# Sort so high-attn points render on top
order = np.argsort(comb_log)
sc = axes2[0].scatter(patch_emb[order, 0], patch_emb[order, 1],
                      c=comb_log[order], s=2, alpha=0.6, linewidths=0,
                      cmap='RdYlGn_r', vmin=vmin_c, vmax=vmax_c,
                      rasterized=True)
axes2[0].set_title('Combined attention\n(nbhd_attn × patch_attn, log)', fontsize=10)
axes2[0].set_xlabel("UMAP 1", fontsize=8); axes2[0].set_ylabel("UMAP 2", fontsize=8)
axes2[0].tick_params(labelsize=7); despine(axes2[0])
cb = fig2.colorbar(sc, ax=axes2[0], fraction=0.04, pad=0.02)
cb.set_label('log(1 + attn)', fontsize=8); cb.ax.tick_params(labelsize=7)

# (b) Tissue type — danger-aware palette
ptc = [TISSUE_COLORS.get(t, '#9E9E9E') for t in patch_tissue]
# Render dangerous tissue last so it shows on top
danger_order = sorted(range(len(patch_tissue)),
                      key=lambda i: list(TISSUE_COLORS.keys()).index(
                          patch_tissue[i]) if patch_tissue[i] in TISSUE_COLORS else 0)
axes2[1].scatter(patch_emb[danger_order, 0], patch_emb[danger_order, 1],
                 c=[ptc[i] for i in danger_order], s=2, alpha=0.6,
                 linewidths=0, rasterized=True)
axes2[1].set_title('Tissue type\n(safe=green/blue → danger=red)', fontsize=10)
axes2[1].set_xlabel("UMAP 1", fontsize=8); axes2[1].set_ylabel("UMAP 2", fontsize=8)
axes2[1].tick_params(labelsize=7); despine(axes2[1])

# (c) ACR label
plbl = np.array(patch_label_all)
for lv, col, nm in [(0,'#2196F3','ACR−'), (1,'#F44336','ACR+')]:
    m = plbl == lv
    axes2[2].scatter(patch_emb[m,0], patch_emb[m,1],
                     c=col, s=2, alpha=0.4, linewidths=0, label=nm,
                     rasterized=True)
axes2[2].set_title('ACR label', fontsize=10)
axes2[2].set_xlabel("UMAP 1", fontsize=8); axes2[2].set_ylabel("UMAP 2", fontsize=8)
axes2[2].legend(fontsize=8, markerscale=3, frameon=False)
axes2[2].tick_params(labelsize=7); despine(axes2[2])

fig2.legend(handles=tissue_legend, loc='lower center', ncol=4,
            fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
plt.suptitle('Patch UMAP — 256-d patch representations (after proj layer)',
             fontsize=13, y=1.01)
plt.tight_layout()
fig2.savefig(UMAP_DIR / "patch_umap.png", dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Saved patch_umap.png", flush=True)

print(f"\nAll UMAP plots saved to {UMAP_DIR}", flush=True)
PYEOF
