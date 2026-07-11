#!/usr/bin/env bash
#SBATCH --job-name=spatial_attn_vis
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/attn_vis_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/attn_vis_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
"""
Load saved fold 0 and fold 1 spatial ABMIL models, run inference on their
test slides, and visualise neighbourhood attention scores.

Outputs (per fold):
  - attn_summary_fold{N}.csv         : per-slide cluster attention weights
  - top_slides_fold{N}.png           : spatial scatter of top-attended clusters
                                       for highest-confidence ACR+ and ACR- slides
  - cluster_attn_by_acr_fold{N}.png  : neighbourhood attention distribution ACR+ vs ACR-
  - cluster_size_vs_attn_fold{N}.png : does cluster size predict attention?

Combined across both folds:
  - oof_attn_combined.csv
  - attn_distribution_combined.png
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

FEAT_DIR = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/slide_cluster_feats")
OUTDIR   = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil")
ATTN_DIR = OUTDIR / "attn_plots"
ATTN_DIR.mkdir(exist_ok=True)

IN_DIM  = 1024
HIDDEN  = 256
DROPOUT = 0.25
MAX_PATCHES_PER_CLUSTER = 512
N_SPLITS = 5

# ── Model (must match training script) ────────────────────────────────────────
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

    def forward(self, clusters):
        nbhd_reps, patch_attns = [], []
        for c in clusters:
            h = self.patch_proj(c.unsqueeze(0))
            rep, pa = self.patch_attn(h)
            nbhd_reps.append(rep)
            patch_attns.append(pa.squeeze(0))
        H = torch.stack(nbhd_reps, dim=1)
        H = self.nbhd_proj(H)
        slide_rep, na = self.nbhd_attn(H)
        logit = self.head(slide_rep).squeeze(-1)
        return logit, na.squeeze(0), patch_attns


# ── Dataset ────────────────────────────────────────────────────────────────────
class SpatialMILDataset(Dataset):
    def __init__(self, paths, max_patches=MAX_PATCHES_PER_CLUSTER):
        self.paths = paths
        self.max_p = max_patches

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        d = torch.load(self.paths[i], map_location='cpu', weights_only=False)
        clusters = []
        for c in d['clusters']:
            feats = c['feats']
            if feats.shape[0] > self.max_p:
                idx = torch.randperm(feats.shape[0])[:self.max_p]
                feats = feats[idx]
            clusters.append(feats)
        label   = torch.tensor(d['label'], dtype=torch.float32)
        coords  = [c['coords'].numpy() for c in d['clusters']]
        n_patches = [c['n_patches'] for c in d['clusters']]
        return clusters, label.unsqueeze(0), d['slide'], d['patient'], coords, n_patches

def collate_fn(batch):
    clusters, label, slide, patient, coords, n_patches = batch[0]
    return clusters, label, slide, patient, coords, n_patches


# ── Load all paths and metadata ────────────────────────────────────────────────
all_paths = sorted(FEAT_DIR.glob("*.pt"))
labels, patients, slides = [], [], []
for p in all_paths:
    d = torch.load(p, map_location='cpu', weights_only=False)
    labels.append(d['label'])
    patients.append(d['patient'])
    slides.append(d['slide'])
labels   = np.array(labels)
patients = np.array(patients)
slides   = np.array(slides)

print(f"Slides: {len(all_paths)}  ACR+={labels.sum()}  ACR-={(labels==0).sum()}", flush=True)

# ── Rebuild same CV splits ─────────────────────────────────────────────────────
cv = StratifiedGroupKFold(n_splits=N_SPLITS)
fold_splits = list(cv.split(all_paths, labels, patients))

all_rows = []  # for combined CSV

for fold in [0, 1]:
    ckpt = OUTDIR / f"spatial_abmil_fold{fold}.pt"
    if not ckpt.exists():
        print(f"Fold {fold}: no checkpoint found, skipping", flush=True)
        continue

    tr_idx, te_idx = fold_splits[fold]
    te_paths = [all_paths[i] for i in te_idx]

    model = SpatialABMIL2Level(IN_DIM, HIDDEN, DROPOUT)
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    model.eval()
    print(f"\nFold {fold}: {len(te_paths)} test slides", flush=True)

    # ── Inference ─────────────────────────────────────────────────────────────
    te_ds = SpatialMILDataset(te_paths)
    te_dl = DataLoader(te_ds, batch_size=1, shuffle=False,
                       collate_fn=collate_fn, num_workers=4)

    slide_records = []
    with torch.no_grad():
        for clusters, label, slide_name, patient, coords, n_patches in te_dl:
            logit, nbhd_attn, patch_attns = model(clusters)
            prob = torch.sigmoid(logit).item()
            na   = nbhd_attn.numpy()   # (K,) neighbourhood attention weights
            lbl  = int(label.item())

            # Per-cluster record
            for k, (attn_w, coord_arr, n_p) in enumerate(zip(na, coords, n_patches)):
                cx, cy = coord_arr[:, 0].mean(), coord_arr[:, 1].mean()
                slide_records.append({
                    'fold':       fold,
                    'slide':      slide_name,
                    'patient':    patient,
                    'label':      lbl,
                    'prob':       prob,
                    'cluster':    k,
                    'n_clusters': len(na),
                    'attn_w':     float(attn_w),
                    'n_patches':  n_p if isinstance(n_p, int) else int(n_p),
                    'centroid_x': float(cx),
                    'centroid_y': float(cy),
                })

    df = pd.DataFrame(slide_records)
    all_rows.append(df)
    df.to_csv(ATTN_DIR / f"attn_summary_fold{fold}.csv", index=False)

    # OOF AUC for this fold
    slide_df = df.groupby('slide').first().reset_index()
    auc = roc_auc_score(slide_df['label'], slide_df['prob'])
    print(f"Fold {fold} test AUC (re-computed): {auc:.3f}", flush=True)

    # ── Plot 1: Neighbourhood attention distribution ACR+ vs ACR- ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for lv, col, name in [(0, '#2D8A2D', 'ACR−'), (1, '#CC3333', 'ACR+')]:
        vals = df[df['label'] == lv]['attn_w'].values
        ax.hist(vals, bins=40, density=True, alpha=0.55, color=col, label=f'{name} (n={len(vals)})')
    ax.set_xlabel('Neighbourhood attention weight', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Fold {fold} — Neighbourhood attention: ACR+ vs ACR−', fontsize=11)
    ax.legend(fontsize=10)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    # ── Plot 2: Cluster size vs attention weight ───────────────────────────────
    ax = axes[1]
    for lv, col, name in [(0, '#2D8A2D', 'ACR−'), (1, '#CC3333', 'ACR+')]:
        sub = df[df['label'] == lv]
        ax.scatter(sub['n_patches'], sub['attn_w'], alpha=0.25, s=8, c=col, label=name)
    ax.set_xlabel('Cluster size (# patches)', fontsize=11)
    ax.set_ylabel('Attention weight', fontsize=11)
    ax.set_title(f'Fold {fold} — Cluster size vs attention weight', fontsize=11)
    ax.legend(fontsize=10)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    plt.tight_layout()
    fig.savefig(ATTN_DIR / f"cluster_attn_fold{fold}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved cluster_attn_fold{fold}.png", flush=True)

    # ── Plot 3: Top slides spatial scatter coloured by attention ──────────────
    # Pick 3 highest-confidence ACR+ and 3 ACR- test slides
    slide_df_sorted = slide_df.sort_values('prob', ascending=False)
    top_acr_pos = slide_df_sorted[slide_df_sorted['label'] == 1].head(3)['slide'].tolist()
    top_acr_neg = slide_df_sorted[slide_df_sorted['label'] == 0].tail(3)['slide'].tolist()
    show_slides  = [(s, 1) for s in top_acr_pos] + [(s, 0) for s in top_acr_neg]

    n_show = len(show_slides)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    cmap = plt.get_cmap('RdYlGn_r')  # red=high attn, green=low attn

    for ax_i, (slide_name, lbl) in enumerate(show_slides):
        ax = axes[ax_i]
        sl_df = df[df['slide'] == slide_name]
        if len(sl_df) == 0:
            ax.set_visible(False); continue

        prob_val = sl_df['prob'].iloc[0]
        attn_arr = sl_df['attn_w'].values
        # Normalize per-slide for colour mapping
        attn_norm = (attn_arr - attn_arr.min()) / (attn_arr.max() - attn_arr.min() + 1e-8)

        for _, row in sl_df.iterrows():
            # Load actual patch coords for this cluster
            pt_file = FEAT_DIR / f"{slide_name.replace('/', '_')}.pt"
            d = torch.load(pt_file, map_location='cpu', weights_only=False)
            k = int(row['cluster'])
            if k >= len(d['clusters']): continue
            coords_k = d['clusters'][k]['coords'].numpy()
            c_norm = (row['attn_w'] - attn_arr.min()) / (attn_arr.max() - attn_arr.min() + 1e-8)
            color = cmap(float(c_norm))
            ax.scatter(coords_k[:, 0], -coords_k[:, 1], s=1.0, c=[color]*len(coords_k),
                       alpha=0.7, linewidths=0)

        acr_str = 'ACR+' if lbl == 1 else 'ACR−'
        ax.set_title(f'{acr_str}  prob={prob_val:.2f}\n{slide_name[:40]}', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[-1], fraction=0.08, pad=0.04)
    cbar.set_label('Neighbourhood attention\n(normalised per slide)', fontsize=9)

    plt.suptitle(f'Fold {fold} — Spatial attention maps  (red=high, green=low)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(ATTN_DIR / f"top_slides_fold{fold}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved top_slides_fold{fold}.png", flush=True)

# ── Combined across folds ──────────────────────────────────────────────────────
if all_rows:
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(ATTN_DIR / "oof_attn_combined.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution
    ax = axes[0]
    for lv, col, name in [(0, '#2D8A2D', 'ACR−'), (1, '#CC3333', 'ACR+')]:
        vals = combined[combined['label'] == lv]['attn_w'].values
        ax.hist(vals, bins=50, density=True, alpha=0.55, color=col, label=f'{name} (n={len(vals)})')
    ax.set_xlabel('Neighbourhood attention weight', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Combined folds 0+1 — Neighbourhood attention distribution', fontsize=11)
    ax.legend(fontsize=10)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    # Top-k cluster fraction: what fraction of slides have >= 1 cluster with attn > threshold
    ax = axes[1]
    thresholds = np.linspace(0, 1/combined.groupby(['fold','slide'])['n_clusters'].first().mean(), 50)
    for lv, col, name in [(0, '#2D8A2D', 'ACR−'), (1, '#CC3333', 'ACR+')]:
        sub = combined[combined['label'] == lv]
        fracs = []
        for thr in thresholds:
            has_high = sub.groupby(['fold','slide'])['attn_w'].max() >= thr
            fracs.append(has_high.mean())
        ax.plot(thresholds, fracs, color=col, lw=2, label=name)
    ax.set_xlabel('Attention threshold', fontsize=11)
    ax.set_ylabel('Fraction of slides with ≥1 cluster above threshold', fontsize=11)
    ax.set_title('High-attention cluster prevalence by ACR label', fontsize=11)
    ax.legend(fontsize=10)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    plt.tight_layout()
    fig.savefig(ATTN_DIR / "attn_distribution_combined.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nAll outputs saved to {ATTN_DIR}", flush=True)

    # ── Print top attended cluster stats ──────────────────────────────────────
    print("\n=== Top cluster (max attn) stats by ACR label ===")
    top_cluster = combined.groupby(['fold', 'slide']).apply(
        lambda g: g.loc[g['attn_w'].idxmax()]).reset_index(drop=True)
    print(top_cluster.groupby('label')[['attn_w', 'n_patches', 'n_clusters']].describe().round(3).to_string())
PYEOF
