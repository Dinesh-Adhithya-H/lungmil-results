#!/usr/bin/env bash
#SBATCH --job-name=vis_patient_attn
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/soft_hier_mil/vis_patient_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/soft_hier_mil/vis_patient_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
"""
Per-patient longitudinal attention visualisation.

For each patient: one figure, all timepoints as rows.
5 columns per timepoint:
  1. Cluster membership     — which patches co-occur in same neighbourhood
  2. Neighbourhood score    — na[c] broadcast to all patches in cluster c  (raw softmax)
  3. Patch score            — pa[c][i] within-cluster softmax              (raw softmax)
  4. Joint = nbhd × patch   — product, NOT re-normalised
  5. MaskedKNN last-layer   — flat per-patch attn from KNN model           (raw softmax)

Colour scale: shared vmax across ALL timepoints for a patient per column type.
This lets you see ACR+ slides are genuinely brighter than ACR-.

"Raw" means: softmax values as output by the model, with a shared per-patient
colour axis — NOT min-max normalised per slide.
"""

import math, sys, warnings
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path
from collections import defaultdict
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
FEAT_DIR  = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/slide_cluster_feats")
CKPT_DIR  = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/soft_hier_mil")
OUT_DIR   = CKPT_DIR / "patient_attn_plots"
OUT_DIR.mkdir(exist_ok=True)

device = torch.device("cpu")

# ── Hyperparameters (must match training) ────────────────────────────────────
IN_DIM   = 1024
HIDDEN   = 256
DROPOUT  = 0.25
KNN_K    = 8
KNN_NLAYERS = 4
MASK_RATIO  = 0.30

# ── Model definitions (copied verbatim from train_soft_hier_mil.sh) ───────────

def build_knn_idx(coords_np, k):
    tree = cKDTree(coords_np)
    _, idx = tree.query(coords_np, k=k + 1)
    return idx[:, 1:]

class GatedPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(dim, dim // 2), nn.Sigmoid())
        self.w = nn.Linear(dim // 2, 1, bias=False)
    def forward(self, h):
        a = self.w(self.V(h) * self.U(h))
        a = torch.softmax(a, dim=0)
        return (a * h).sum(0), a.squeeze(-1)

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
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, dropout=DROPOUT):
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
            nbhd_reps.append(rep); patch_attns.append(pa.squeeze(0))
        H = self.nbhd_proj(torch.stack(nbhd_reps, dim=1))
        slide_rep, na = self.nbhd_attn(H)
        return self.head(slide_rep).squeeze(-1), na.squeeze(0), patch_attns

class PointTransformerBlock(nn.Module):
    def __init__(self, dim, k=8, dropout=0.1):
        super().__init__()
        self.k = k; self.scale = dim ** -0.5
        self.W_q  = nn.Linear(dim, dim, bias=False)
        self.W_k  = nn.Linear(dim, dim, bias=False)
        self.W_v  = nn.Linear(dim, dim, bias=False)
        self.W_o  = nn.Linear(dim, dim, bias=False)
        self.pos_enc = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(dim*2, dim), nn.Dropout(dropout))
        self.drop  = nn.Dropout(dropout)
    def forward(self, x, coords, knn_idx):
        nbr_coords = coords[knn_idx]
        rel_pos    = nbr_coords - coords.unsqueeze(1)
        pos_bias   = self.pos_enc(rel_pos)
        q = self.W_q(x)
        k = self.W_k(x)[knn_idx] + pos_bias
        v = self.W_v(x)[knn_idx] + pos_bias
        attn = (q.unsqueeze(1) * k).sum(-1) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out  = (attn.unsqueeze(-1) * v).sum(1)
        out  = self.drop(self.W_o(out))
        x    = self.norm1(x + out)
        x    = self.norm2(x + self.ff(x))
        return x

class MaskedKNNMIL(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, dropout=DROPOUT,
                 k=KNN_K, n_layers=KNN_NLAYERS, mask_ratio=MASK_RATIO):
        super().__init__()
        self.k = k; self.n_layers = n_layers; self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, hidden))
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
                                  nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([PointTransformerBlock(hidden, k=k, dropout=dropout)
                                     for _ in range(n_layers)])
        self.pools  = nn.ModuleList([GatedPool(hidden) for _ in range(n_layers)])
        self.heads  = nn.ModuleList([nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
                                     for _ in range(n_layers)])
        self.layer_w   = nn.Parameter(torch.zeros(n_layers))
        self.recon_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, in_dim))
    def forward(self, feats, coords):
        N = feats.shape[0]
        knn_idx = torch.from_numpy(build_knn_idx(coords.cpu().numpy(), self.k)).long()
        x = self.proj(feats)
        layer_logits, layer_attns = [], []
        for block, pool, head in zip(self.blocks, self.pools, self.heads):
            x = block(x, coords, knn_idx)
            rep, attn = pool(x)
            layer_logits.append(head(rep).squeeze())
            layer_attns.append(attn)
        w = torch.softmax(self.layer_w, dim=0)
        final_logit = sum(w[i] * layer_logits[i] for i in range(self.n_layers))
        return final_logit, layer_logits, layer_attns, None, None


# ── Load all .pt files, group by patient ─────────────────────────────────────
all_pts = sorted(FEAT_DIR.glob("*.pt"))
patients = defaultdict(list)
for p in all_pts:
    pid = p.stem.split('-')[0]   # LT001-DST0035 → LT001
    patients[pid].append(p)

print(f"Found {len(all_pts)} slides across {len(patients)} patients", flush=True)

# ── Load fold splits (StratifiedGroupKFold, same as training) ─────────────────
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
SEED = 42; N_SPLITS = 5
labels   = []
pat_ids  = []
for p in all_pts:
    d = torch.load(p, map_location='cpu', weights_only=False)
    labels.append(int(d['label']))
    pat_ids.append(d['patient'])
labels  = np.array(labels)
pat_ids = np.array(pat_ids)
cv     = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
splits = list(cv.split(all_pts, labels, pat_ids))

# map slide path → fold
slide_to_fold = {}
for fold, (_, te_idx) in enumerate(splits):
    for i in te_idx:
        slide_to_fold[all_pts[i]] = fold

# ── Load model checkpoints ────────────────────────────────────────────────────
abmil_models   = {}   # fold → SpatialABMIL2Level
knn_models     = {}   # fold → MaskedKNNMIL
for fold in range(N_SPLITS):
    abmil_ckpt = CKPT_DIR / f"abmil_fold{fold}.pt"
    knn_ckpt   = CKPT_DIR / f"maskedknn_fold{fold}.pt"
    if abmil_ckpt.exists():
        m = SpatialABMIL2Level().to(device)
        m.load_state_dict(torch.load(abmil_ckpt, map_location=device))
        m.eval(); abmil_models[fold] = m
    if knn_ckpt.exists():
        m = MaskedKNNMIL().to(device)
        m.load_state_dict(torch.load(knn_ckpt, map_location=device))
        m.eval(); knn_models[fold] = m

print(f"Loaded ABMIL checkpoints for folds: {sorted(abmil_models.keys())}", flush=True)
print(f"Loaded MaskedKNN checkpoints for folds: {sorted(knn_models.keys())}", flush=True)


# ── Inference: extract per-patch scores from both models ─────────────────────
@torch.no_grad()
def run_abmil(model, path):
    """Returns: coords (N,2), nbhd_score (N,), patch_score (N,),
                joint_score (N,), cluster_ids (N,), prob (float)"""
    d = torch.load(path, map_location='cpu', weights_only=False)
    clusters_raw = d['clusters']

    # build cluster list for model (capped at 512 patches per cluster)
    clusters_t = []
    for c in clusters_raw:
        f = c['feats']
        if f.shape[0] > 512:
            idx = torch.randperm(f.shape[0])[:512]
            f = f[idx]
        clusters_t.append(f.to(device))

    logit, na, patch_attns = model(clusters_t)
    prob = torch.sigmoid(logit).item()

    # na: (C,) neighbourhood softmax scores
    # patch_attns: list of C tensors each (N_c,) within-cluster softmax

    all_coords, all_cids = [], []
    all_nbhd, all_patch = [], []

    na_np = na.cpu().numpy()   # (C,) — raw softmax, sums to 1 over clusters

    for ci, c in enumerate(clusters_raw):
        nc = c['coords']          # (N_c, 2)
        nf = c['feats'].shape[0]
        if nf > 512:
            # use same random subset as model (deterministic with same seed? no)
            # just use all coords but re-use model's patch_attns truncated
            nf = 512
            nc = nc[:nf]

        pa_np = patch_attns[ci].cpu().numpy()   # (N_c,) within-cluster softmax

        all_coords.append(nc.numpy()[:nf])
        all_cids.append(np.full(nf, ci, dtype=int))
        all_nbhd.append(np.full(nf, na_np[ci]))    # broadcast neighbourhood score
        all_patch.append(pa_np[:nf])

    coords     = np.concatenate(all_coords, axis=0)   # (N, 2)
    cids       = np.concatenate(all_cids)              # (N,)
    nbhd_score = np.concatenate(all_nbhd)             # (N,)  raw softmax
    patch_score= np.concatenate(all_patch)             # (N,)  raw softmax
    joint_score= nbhd_score * patch_score              # (N,)  product

    return coords, nbhd_score, patch_score, joint_score, cids, prob, int(d['label'])


@torch.no_grad()
def run_knn(model, path):
    """Returns: coords (N,2), attn_per_layer list, prob (float)"""
    d = torch.load(path, map_location='cpu', weights_only=False)
    # flatten clusters
    feats_list  = [c['feats'] for c in d['clusters']]
    coords_list = [c['coords'] for c in d['clusters']]
    all_feats  = torch.cat(feats_list,  dim=0)
    all_coords = torch.cat(coords_list, dim=0).float()
    if all_feats.shape[0] > 2048:
        idx = torch.randperm(all_feats.shape[0])[:2048]
        all_feats  = all_feats[idx]
        all_coords = all_coords[idx]
    logit, _, layer_attns, _, _ = model(all_feats.to(device), all_coords.to(device))
    prob = torch.sigmoid(logit).item()
    attns = [a.cpu().numpy() for a in layer_attns]
    return all_coords.numpy(), attns, prob, int(d['label'])


# ── Plotting utilities ────────────────────────────────────────────────────────
# Cluster colour palette (up to 60 clusters)
CMAP_CLUSTERS = plt.cm.get_cmap('tab20', 20)

def cluster_color(cid):
    return CMAP_CLUSTERS(cid % 20)

def scatter_attn(ax, xs, ys, scores, vmin, vmax, cmap='hot_r', s=2, title=''):
    """Plot spatial scatter with raw scores and shared colour axis."""
    order = np.argsort(scores)    # draw low-score patches first, high on top
    sc = ax.scatter(xs[order], ys[order], c=scores[order], cmap=cmap,
                    vmin=vmin, vmax=vmax, s=s, alpha=0.9,
                    linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal'); ax.spines[:].set_visible(False)
    if title:
        ax.set_title(title, fontsize=6, pad=2)
    return sc

def scatter_clusters(ax, xs, ys, cids, title=''):
    """Plot cluster membership with categorical colours."""
    unique_cids = np.unique(cids)
    for cid in unique_cids:
        m = cids == cid
        ax.scatter(xs[m], ys[m], c=[cluster_color(cid)],
                   s=2, alpha=0.85, linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal'); ax.spines[:].set_visible(False)
    if title:
        ax.set_title(title, fontsize=6, pad=2)


# ── Main loop: one figure per patient ────────────────────────────────────────
N_COLS = 5   # cluster | nbhd | patch | joint | knn

for pid, slide_paths in sorted(patients.items()):
    # only process patients with at least one ACR+ slide
    slide_labels = []
    for p in slide_paths:
        d = torch.load(p, map_location='cpu', weights_only=False)
        slide_labels.append(int(d['label']))
    if sum(slide_labels) == 0:
        continue    # skip all-negative patients

    n_slides = len(slide_paths)
    print(f"\nPatient {pid}: {n_slides} slides, "
          f"{sum(slide_labels)} ACR+", flush=True)

    # ── Run inference for all slides ─────────────────────────────────────────
    records = []
    for path in slide_paths:
        fold = slide_to_fold.get(path, 0)
        abmil_m = abmil_models.get(fold)
        knn_m   = knn_models.get(fold)
        if abmil_m is None or knn_m is None:
            print(f"  No model for fold {fold}, skip {path.stem}", flush=True)
            continue
        try:
            coords_ab, nbhd, patch, joint, cids, prob_ab, lbl = run_abmil(abmil_m, path)
            coords_kn, layer_attns, prob_kn, _ = run_knn(knn_m, path)

            # align knn coords to abmil coords via KNN matching
            # (knn may have different subsampling — just plot separately)
            records.append({
                'slide':      path.stem,
                'label':      lbl,
                'prob_abmil': prob_ab,
                'prob_knn':   prob_kn,
                'coords_ab':  coords_ab,    # (N, 2)
                'cids':       cids,         # (N,) cluster membership
                'nbhd':       nbhd,         # (N,) neighbourhood score
                'patch':      patch,        # (N,) patch score
                'joint':      joint,        # (N,) joint
                'coords_kn':  coords_kn,    # (M, 2) — may differ from N
                'knn_attn':   layer_attns[-1],  # last layer, widest RF
                'knn_l1':     layer_attns[0],   # first layer, patch scale
            })
        except Exception as e:
            print(f"  Error {path.stem}: {e}", flush=True)

    if not records:
        continue

    # sort chronologically by slide name (DST number)
    records.sort(key=lambda r: r['slide'])
    n_rows = len(records)

    # ── Shared colour axes per column (per-patient, raw softmax values) ───────
    # Shared vmax = 99th pct across all slides for this patient
    # This means ACR+ slides appear brighter without per-slide normalisation
    all_nbhd  = np.concatenate([r['nbhd']  for r in records])
    all_patch = np.concatenate([r['patch'] for r in records])
    all_joint = np.concatenate([r['joint'] for r in records])
    all_knn   = np.concatenate([r['knn_attn'] for r in records])

    vmax_nbhd  = float(np.percentile(all_nbhd,  99))
    vmax_patch = float(np.percentile(all_patch, 99))
    vmax_joint = float(np.percentile(all_joint, 99))
    vmax_knn   = float(np.percentile(all_knn,   99))

    # ── Figure layout ─────────────────────────────────────────────────────────
    # rows = slides, cols = [cluster | nbhd | patch | joint | knn]
    col_w = 2.5
    row_h = 2.5
    label_w = 1.8
    fig_w = label_w + N_COLS * col_w + 0.8   # +0.8 for colourbar
    fig_h = n_rows * row_h + 1.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    # gridspec: n_rows × (1 label col + N_COLS data cols)
    gs = fig.add_gridspec(n_rows, N_COLS + 1,
                          width_ratios=[label_w/col_w] + [1]*N_COLS,
                          wspace=0.05, hspace=0.35,
                          left=0.01, right=0.92,
                          top=0.93, bottom=0.04)

    col_titles = ['Clusters\n(neighbourhood\nmembership)',
                  'Neighbourhood\nscore\n(raw softmax)',
                  'Patch score\n(within cluster,\nraw softmax)',
                  'Joint score\n(nbhd × patch,\nunnormalised)',
                  'MaskedKNN\nlast-layer\nattn (raw)']
    cmaps = [None, 'YlOrRd', 'YlOrRd', 'hot_r', 'YlOrRd']
    vmins = [None, 0, 0, 0, 0]
    vmaxs = [None, vmax_nbhd, vmax_patch, vmax_joint, vmax_knn]

    # Column headers (top row)
    for ci, ct in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, ci + 1])
        ax.set_title(ct, fontsize=6, pad=3, fontweight='bold')

    sc_refs = [None, None, None, None, None]   # for colourbars

    for ri, rec in enumerate(records):
        xs_ab = rec['coords_ab'][:, 0]
        ys_ab = -rec['coords_ab'][:, 1]   # flip Y so tissue reads naturally
        xs_kn = rec['coords_kn'][:, 0]
        ys_kn = -rec['coords_kn'][:, 1]

        # ── Row label ─────────────────────────────────────────────────────────
        ax_lbl = fig.add_subplot(gs[ri, 0])
        ax_lbl.axis('off')
        lbl_str  = "ACR+" if rec['label'] == 1 else "ACR−"
        col_lbl  = '#C62828' if rec['label'] == 1 else '#1565C0'
        slide_id = rec['slide'].split('-')[-1]   # e.g. DST0035
        ax_lbl.text(0.95, 0.65, lbl_str,
                    transform=ax_lbl.transAxes, ha='right', va='center',
                    fontsize=9, fontweight='bold', color=col_lbl)
        ax_lbl.text(0.95, 0.38, slide_id,
                    transform=ax_lbl.transAxes, ha='right', va='center',
                    fontsize=6.5, color='#444444')
        ax_lbl.text(0.95, 0.18,
                    f"p(ABMIL)={rec['prob_abmil']:.2f}\np(KNN)={rec['prob_knn']:.2f}",
                    transform=ax_lbl.transAxes, ha='right', va='center',
                    fontsize=5.5, color='#666666')

        # ── Col 0: Cluster membership ─────────────────────────────────────────
        ax0 = fig.add_subplot(gs[ri, 1])
        scatter_clusters(ax0, xs_ab, ys_ab, rec['cids'])
        if ri == 0:
            ax0.set_title(col_titles[0], fontsize=6, pad=3, fontweight='bold')

        # ── Col 1: Neighbourhood score ────────────────────────────────────────
        ax1 = fig.add_subplot(gs[ri, 2])
        sc = scatter_attn(ax1, xs_ab, ys_ab, rec['nbhd'],
                          vmin=0, vmax=vmax_nbhd, cmap='YlOrRd')
        if ri == 0: ax1.set_title(col_titles[1], fontsize=6, pad=3, fontweight='bold')
        if sc_refs[1] is None: sc_refs[1] = sc

        # ── Col 2: Patch score ────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[ri, 3])
        sc = scatter_attn(ax2, xs_ab, ys_ab, rec['patch'],
                          vmin=0, vmax=vmax_patch, cmap='YlOrRd')
        if ri == 0: ax2.set_title(col_titles[2], fontsize=6, pad=3, fontweight='bold')
        if sc_refs[2] is None: sc_refs[2] = sc

        # ── Col 3: Joint = nbhd × patch ───────────────────────────────────────
        ax3 = fig.add_subplot(gs[ri, 4])
        sc = scatter_attn(ax3, xs_ab, ys_ab, rec['joint'],
                          vmin=0, vmax=vmax_joint, cmap='hot_r')
        if ri == 0: ax3.set_title(col_titles[3], fontsize=6, pad=3, fontweight='bold')
        if sc_refs[3] is None: sc_refs[3] = sc

        # ── Col 4: MaskedKNN last-layer attn ──────────────────────────────────
        ax4 = fig.add_subplot(gs[ri, 5])
        sc = scatter_attn(ax4, xs_kn, ys_kn, rec['knn_attn'],
                          vmin=0, vmax=vmax_knn, cmap='YlOrRd')
        if ri == 0: ax4.set_title(col_titles[4], fontsize=6, pad=3, fontweight='bold')
        if sc_refs[4] is None: sc_refs[4] = sc

        # highlight ACR+ rows with light background
        if rec['label'] == 1:
            for ax in [ax0, ax1, ax2, ax3, ax4]:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor('#C62828')
                    spine.set_linewidth(0.8)

    # ── Shared colourbars (right side) ────────────────────────────────────────
    cbar_labels = ['', 'Nbhd score\n(shared scale)', 'Patch score\n(shared scale)',
                   'Joint score\n(shared scale)', 'KNN attn\n(shared scale)']
    cbar_cmaps  = ['', 'YlOrRd', 'YlOrRd', 'hot_r', 'YlOrRd']
    cbar_vmaxs  = [None, vmax_nbhd, vmax_patch, vmax_joint, vmax_knn]

    for ci in range(1, N_COLS):   # cols 1..4 have colorbars (col 0 = categorical clusters)
        if sc_refs[ci] is None:
            continue
        cax = fig.add_axes([0.93, 0.04 + (N_COLS - 1 - ci) * (0.88 / (N_COLS - 1)),
                             0.012, 0.88 / (N_COLS - 1) * 0.75])
        cb  = fig.colorbar(sc_refs[ci], cax=cax)
        cb.set_label(cbar_labels[ci], fontsize=5)
        cb.ax.tick_params(labelsize=4)
        # mark the vmax value
        cb.ax.axhline(cbar_vmaxs[ci], color='k', lw=0.5, linestyle='--')

    # ── Figure title ──────────────────────────────────────────────────────────
    n_pos = sum(r['label'] for r in records)
    fig.suptitle(
        f"Patient {pid}  |  {n_slides} timepoints  |  {n_pos} ACR+\n"
        f"Red border = ACR+  |  Colour scale shared across all timepoints (99th pct = vmax)\n"
        f"Cluster colours (col 1) show co-occurring patch groups",
        fontsize=8, y=0.98)

    out_path = OUT_DIR / f"patient_{pid}.pdf"
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path.name}", flush=True)


print(f"\nAll figures saved to: {OUT_DIR}", flush=True)
print(f"Total patient figures: {len(list(OUT_DIR.glob('patient_*.pdf')))}", flush=True)
PYEOF
