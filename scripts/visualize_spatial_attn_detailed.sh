#!/usr/bin/env bash
#SBATCH --job-name=vis_detailed
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/spatial_attn_detailed/job_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/spatial_attn_detailed/job_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

mkdir -p /lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/spatial_attn_detailed

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
"""
Detailed spatial attention figure — per patient, all H&E biopsies.

One PDF per ACR+ patient.  Rows = H&E biopsies (sorted by date).
7 data columns (+ row label):

  1. Tissue type           h5ad annotation, categorical color per tissue type
  2. Neighbourhood ID      which spatial cluster (tab20 colors)
  3. ABMIL α              flat single-level ABMIL (v8 Phase-1) attention logit
  4. pa[c]                1st-level: within-neighbourhood softmax patch attention
  5. na[c]                2nd-level: raw neighbourhood attention logit
  6. na_s[c] × pa[c]     joint = softmax(na)[c] * pa[c][i]
  7. KNN-masked ABMIL     gated attention pooling over K=16 nearest spatial
                          neighbours; score = attention weight patch i receives
                          within its local neighbourhood bag

Row label shows: ACR grade | date | split-0 train/test | model p=...

Shared 99th-pct colour scale across all biopsies for each patient.
ACR+ rows: red spine border.
"""

import sys, warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
H5AD        = Path("/lustre/groups/aih/dinesh.haridoss/datasets/adata_v3.h5ad")
SAMPLES_DIR = Path("/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples")
FEAT_DIR    = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/slide_cluster_feats")
V8_BASE     = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/results_mm_abmil_v8")
ABMIL2_DIR  = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/soft_hier_mil")
SPLITS_CSV  = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
OUT_DIR     = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/spatial_attn_detailed")
OUT_DIR.mkdir(exist_ok=True)

HIDDEN  = 256
DROPOUT = 0.4
IN_DIM  = 1024
KNN_K   = 16          # spatial nearest neighbours for masked KNN ABMIL

TYPE_COLORS = {
    "Alveolar":                                  "#E53030",
    "Bronchial":                                 "#1A72CC",
    "Cartilage":                                 "#F5C800",
    "Alveolar with empty spaces":                "#00BB44",
    "Alveolar with hemorrhage and inflammation": "#FF6F00",
    "Lymphocytoplasmic inflammation":            "#CC00CC",
    "Unknown":                                   "#BBBBBB",
}
TYPE_ORDER = list(TYPE_COLORS.keys())
N_V      = len(TYPE_ORDER)
TYPE_IDX = {t: i for i, t in enumerate(TYPE_ORDER)}
_angles  = [2 * np.pi * k / N_V - np.pi / 2 for k in range(N_V)]
VERTEX   = np.array([(np.cos(a), np.sin(a)) for a in _angles])   # (7, 2)
KNN_MAX_DIST = 256   # pixel tolerance for coord→tissue matching

ACR_GRADE_COLOR = {
    "A0B0": "#2D8A2D", "A1B0": "#E8A020",
    "A2B0": "#CC3333", "A1B1": "#7722AA",
}
ACR_GRADE_LABEL = {
    "A0B0": "No ACR", "A1B0": "Mild ACR",
    "A2B0": "Moderate ACR", "A1B1": "ACR+LB",
}


# ── Model definitions ──────────────────────────────────────────────────────────
class GatedAttentionEncoder(nn.Module):
    """v8 Phase-1 single-level ABMIL — returns raw attention logits per patch."""
    def __init__(self, feat_dim=IN_DIM, hidden_dim=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.att_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.att_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.att_w = nn.Linear(hidden_dim, 1, bias=False)
        self.att_drop = nn.Dropout(dropout)
    def forward(self, x):
        h = self.backbone(x)
        return self.att_w(self.att_drop(self.att_V(h) * self.att_U(h))).squeeze(1), h

    def gate_scores(self, h):
        """Compute gated attention logits from backbone features h (N, H)."""
        return self.att_w(self.att_drop(self.att_V(h) * self.att_U(h))).squeeze(1)


class GatedAttentionPool(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(in_dim, hidden), nn.Sigmoid())
        self.w = nn.Linear(hidden, 1, bias=False)
    def forward(self, h):
        raw = self.w(self.V(h) * self.U(h))
        a   = F.softmax(raw, dim=1)
        return (a * h).sum(dim=1), raw.reshape(raw.shape[1])


class SpatialABMIL2Level(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_proj = nn.Sequential(
            nn.Linear(IN_DIM, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU(), nn.Dropout(DROPOUT))
        self.patch_attn = GatedAttentionPool(HIDDEN, HIDDEN // 2)
        self.nbhd_proj  = nn.Sequential(nn.LayerNorm(HIDDEN), nn.Dropout(DROPOUT))
        self.nbhd_attn  = GatedAttentionPool(HIDDEN, HIDDEN // 2)
        self.head       = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 1))
    def forward(self, clusters):
        reps, pa_raws = [], []
        for c in clusters:
            h = self.patch_proj(c.unsqueeze(0))
            rep, pa_raw = self.patch_attn(h)
            reps.append(rep)
            pa_raws.append(F.softmax(pa_raw.reshape(1, -1), dim=1).reshape(-1))
        H = self.nbhd_proj(torch.stack(reps, dim=1))
        slide_rep, na_raw = self.nbhd_attn(H)
        return self.head(slide_rep).squeeze(-1), na_raw.reshape(-1), pa_raws


# ── Load models ────────────────────────────────────────────────────────────────
print("[models] Loading v8 Phase-1 encoders ...", flush=True)
p1_encoders: dict = {}
for fold in range(4):
    ckpt = V8_BASE / "acr" / f"split0_fold{fold}" / "phase1" / "enc_HE.pt"
    if not ckpt.exists(): print(f"  [warn] Missing {ckpt}"); continue
    enc = GatedAttentionEncoder()
    sd  = torch.load(ckpt, map_location="cpu", weights_only=True)
    if any(k.startswith("encoder.") for k in sd):
        sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    try:    enc.load_state_dict(sd, strict=True)
    except: enc.load_state_dict(sd, strict=False)
    enc.eval(); p1_encoders[fold] = enc
print(f"[models] Phase-1 folds: {sorted(p1_encoders)}", flush=True)

print("[models] Loading SpatialABMIL2Level ...", flush=True)
abmil2_models: dict = {}
for fold in range(5):
    ckpt = ABMIL2_DIR / f"abmil_fold{fold}.pt"
    if not ckpt.exists(): continue
    m = SpatialABMIL2Level()
    m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval()
    abmil2_models[fold] = m
print(f"[models] 2-level ABMIL folds: {sorted(abmil2_models)}", flush=True)


# ── Load splits CSV ────────────────────────────────────────────────────────────
print("[splits] Loading ...", flush=True)
df_sp = pd.read_csv(SPLITS_CSV)
fold_cols = [c for c in df_sp.columns if c.startswith("fold_")][:4]

# stem → test fold (split 0)
stem_to_fold: dict = {}
for fi, col in enumerate(fold_cols):
    for s in df_sp.loc[df_sp[col] == "test", "file"].values:
        stem_to_fold[str(s)] = fi

stem_meta: dict = {}
for _, row in df_sp.iterrows():
    stem_meta[str(row["file"])] = {
        "patient_id": str(row["patient_id"]),
        "anchor_dt":  str(row["anchor_dt"]),
        "label":      int(row["label"]) if pd.notna(row["label"]) else 0,
        "has_HE":     bool(row.get("has_HE", False)),
    }
pat_to_stems: dict = defaultdict(list)
for stem, meta in stem_meta.items():
    pat_to_stems[meta["patient_id"]].append(stem)
print(f"[splits] {len(stem_meta)} stems, {len(stem_to_fold)} test assignments", flush=True)


# ── Load h5ad obs ──────────────────────────────────────────────────────────────
print("[h5ad] Loading obs ...", flush=True)
adata = ad.read_h5ad(H5AD, backed='r')
obs = adata.obs[[
    'chunk_left', 'chunk_top', 'tissue_type',
    'slide_name_clean', 'record_id', 'biopsy_date', 'acr_status', 'acr_binary'
]].copy()
adata.file.close()
obs['chunk_left']       = obs['chunk_left'].astype(float)
obs['chunk_top']        = obs['chunk_top'].astype(float)
obs['biopsy_date_str']  = obs['biopsy_date'].astype(str).str[:10]
obs['slide_name_clean'] = obs['slide_name_clean'].astype(str)
print(f"[h5ad] {len(obs):,} patches, {obs['record_id'].nunique()} patients", flush=True)

# Per-slide KD-trees
print("[h5ad] Building per-slide KD-trees ...", flush=True)
slide_kd: dict = {}
for sn, grp in obs.groupby('slide_name_clean'):
    coords = grp[['chunk_left', 'chunk_top']].values.astype(np.float32)
    slide_kd[str(sn)] = (cKDTree(coords), grp['tissue_type'].values)
print(f"[h5ad] {len(slide_kd)} slides", flush=True)

_uniq = obs[['record_id', 'biopsy_date_str', 'slide_name_clean']].drop_duplicates('slide_name_clean')
date_to_slide = {
    (str(r['record_id']), str(r['biopsy_date_str'])[:10]): str(r['slide_name_clean'])
    for _, r in _uniq.iterrows()
}
feat_files = {p.stem: p for p in sorted(FEAT_DIR.glob("*.pt"))}


def find_feat_file(patient_id, biopsy_date_str):
    key = (patient_id, str(biopsy_date_str)[:10])
    sn  = date_to_slide.get(key)
    if sn is None: return None, None
    return (feat_files[sn], sn) if sn in feat_files else (None, None)


# ── KNN-masked ABMIL ───────────────────────────────────────────────────────────
@torch.no_grad()
def compute_knn_masked_abmil(enc: GatedAttentionEncoder, feats: torch.Tensor,
                              coords: np.ndarray, K: int = KNN_K) -> np.ndarray:
    """
    For each patch i, form a local bag = {i} ∪ K nearest spatial neighbours.
    Run the encoder's gated attention mechanism on this local bag.
    The attention weight patch i receives within its local bag = its KNN score.

    Returns: knn_scores (N,) in [0, 1], softmax-normalised within each local bag.
    """
    N = feats.shape[0]
    if N == 0:
        return np.array([])

    # Backbone features — shared, computed once
    h = enc.backbone(feats).numpy()   # (N, H)

    # Spatial KNN graph — K+1 because nearest includes self
    K_actual = min(K + 1, N)
    tree = cKDTree(coords)
    _, idxs = tree.query(coords, k=K_actual)   # (N, K_actual)
    # idxs[:, 0] = self, idxs[:, 1:] = K neighbours
    # We include self so idxs is shape (N, K_actual), first column = self index

    knn_scores = np.zeros(N)

    # Vectorised: for each patch i, get local bag features and run gate scoring
    # local bag = idxs[i]  (K_actual patches including self)
    h_t = torch.from_numpy(h)   # (N, H)

    # Build batch: (N, K_actual, H)
    bag_h = h_t[idxs]   # (N, K_actual, H)

    # Gated attention logits for each local bag
    # att_V, att_U, att_w operate on (..., H) — apply over K_actual dimension
    # Use (N * K_actual, H) → reshape back
    NK = N * K_actual
    flat_h = bag_h.reshape(NK, -1)   # (N*K, H)
    logits = enc.gate_scores(flat_h).reshape(N, K_actual)   # (N, K_actual)

    # Softmax over local bag (dim=1)
    attn = F.softmax(logits, dim=1).numpy()   # (N, K_actual)

    # For each patch i, its score = attention weight at position 0 (self)
    # because idxs[:, 0] == i (self is first in KD-tree result for k=K+1)
    knn_scores = attn[:, 0]   # (N,)

    return knn_scores


# ── Flat ABMIL + KNN ──────────────────────────────────────────────────────────
@torch.no_grad()
def run_flat_abmil(enc, sample_path):
    """Returns: coords (N,2), alpha (N,) raw logits, knn_scores (N,)."""
    d = torch.load(sample_path, map_location='cpu', weights_only=False)
    feats  = d.get('inputs', {}).get('HE_cells')
    coords = d.get('instance_spatial_coords', {}).get('HE_cells')
    if coords is None:
        coords = d.get('inputs', {}).get('HE_coords')
    if feats is None or coords is None or feats.shape[0] == 0:
        return None, None, None
    feats = feats.float(); coords = coords.float()
    if feats.shape[0] > 3000:
        idx = torch.randperm(feats.shape[0])[:3000]
        feats = feats[idx]; coords = coords[idx]
    alpha, _ = enc(feats)
    alpha_np  = alpha.numpy()
    coords_np = coords.numpy()
    knn = compute_knn_masked_abmil(enc, feats, coords_np)
    return coords_np, alpha_np, knn


# ── 2-level neighbourhood ABMIL ───────────────────────────────────────────────
@torch.no_grad()
def run_nbhd_abmil(model, feat_path):
    d = torch.load(feat_path, map_location='cpu', weights_only=False)
    clusters_raw = d['clusters']
    clusters_t, coord_list = [], []
    for c in clusters_raw:
        f  = c['feats'].float()
        co = c['coords'].float()
        if f.shape[0] > 512:
            idx = torch.randperm(f.shape[0])[:512]
            f = f[idx]; co = co[idx]
        clusters_t.append(f); coord_list.append(co)

    logit, na_raw, pa_raws = model(clusters_t)
    prob  = torch.sigmoid(logit).item()
    na_np = na_raw.numpy()
    na_sm = np.exp(na_np - na_np.max()); na_sm /= na_sm.sum()

    all_coords, all_cidx, all_na, all_pa, all_joint = [], [], [], [], []
    for ci, (co, pa_t) in enumerate(zip(coord_list, pa_raws)):
        n = co.shape[0]; pa_np = pa_t.numpy()[:n]
        all_coords.append(co.numpy())
        all_cidx.append(np.full(n, ci, dtype=np.int32))
        all_na.append(np.full(n, na_np[ci]))
        all_pa.append(pa_np)
        all_joint.append(na_sm[ci] * pa_np)

    return (np.concatenate(all_coords),
            np.concatenate(all_cidx),
            np.concatenate(all_na),
            np.concatenate(all_pa),
            np.concatenate(all_joint),
            prob, len(clusters_raw),
            [co.numpy() for co in coord_list], [p.numpy() for p in pa_raws], na_np)


# ── Simplex helpers ───────────────────────────────────────────────────────────
def attn_weighted_composition(cluster_coords: np.ndarray, pa_weights: np.ndarray,
                               slide_name: str) -> np.ndarray:
    """Attention-weighted tissue composition → (N_V,) vector summing to 1."""
    if slide_name not in slide_kd:
        return np.ones(N_V) / N_V
    tree, types = slide_kd[slide_name]
    coords2d = np.atleast_2d(cluster_coords)
    dists, idxs = tree.query(coords2d, k=1, workers=-1)
    dists = np.atleast_1d(dists); idxs = np.atleast_1d(idxs)
    pa    = np.atleast_1d(pa_weights)
    valid = dists < KNN_MAX_DIST
    if valid.sum() == 0:
        return np.ones(N_V) / N_V
    matched_t = types[idxs[valid]]
    matched_w = pa[valid]
    matched_w = matched_w / (matched_w.sum() + 1e-12)
    w = np.zeros(N_V)
    for t, wt in zip(matched_t, matched_w):
        if t in TYPE_IDX:
            w[TYPE_IDX[t]] += wt
    s = w.sum()
    return w / s if s > 0 else np.ones(N_V) / N_V


def draw_simplex_mini(ax, nbhd_positions, na_vals, na_vmin, na_vmax,
                      bag_pos, label, sc_store=None):
    """
    Compact simplex panel for one biopsy row.
    nbhd_positions: (C, 2)  each neighbourhood's projected position
    na_vals:        (C,)    raw na[c] logits for colour
    bag_pos:        (2,)    na-softmax-weighted bag rep position
    label:          0 or 1  (ACR-/ACR+)
    """
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_xlim(-1.40, 1.40); ax.set_ylim(-1.40, 1.40)
    # polygon outline
    poly = np.vstack([VERTEX, VERTEX[0]])
    ax.plot(poly[:, 0], poly[:, 1], color='#CCCCCC', lw=0.8, zorder=1)
    # inner guide rings
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in [0.33, 0.67]:
        ax.plot(r * np.cos(theta), r * np.sin(theta), color='#EEEEEE', lw=0.3, zorder=1)
    # radial spokes
    for vx, vy in VERTEX:
        ax.plot([0, vx], [0, vy], color='#E8E8E8', lw=0.4, zorder=1)
    # vertex labels (very small, initials)
    SHORT = {
        "Alveolar":                                  "Alv",
        "Bronchial":                                 "Bro",
        "Cartilage":                                 "Car",
        "Alveolar with empty spaces":                "Alv+",
        "Alveolar with hemorrhage and inflammation": "Alv+H",
        "Lymphocytoplasmic inflammation":            "Lym",
        "Unknown":                                   "Unk",
    }
    for i, t in enumerate(TYPE_ORDER):
        vx, vy = VERTEX[i]
        sc_v = 1.18
        ha = 'center'
        if vx > 0.15: ha = 'left'
        elif vx < -0.15: ha = 'right'
        ax.text(vx * sc_v, vy * sc_v, SHORT[t], fontsize=4.0, ha=ha, va='center',
                color=TYPE_COLORS[t], fontweight='bold', zorder=5)
    # neighbourhood dots
    if len(nbhd_positions) > 0:
        order = np.argsort(na_vals)
        sc = ax.scatter(nbhd_positions[order, 0], nbhd_positions[order, 1],
                        c=na_vals[order], cmap='Reds',
                        vmin=na_vmin, vmax=na_vmax,
                        s=14, alpha=0.85, linewidths=0, zorder=3, rasterized=True)
        if sc_store is not None and sc_store.get('simplex') is None:
            sc_store['simplex'] = sc
    # bag rep
    mk = '^' if label == 1 else 'o'
    sz = 55 if label == 1 else 40
    col = '#E53935' if label == 1 else '#1E88E5'
    ax.scatter([bag_pos[0]], [bag_pos[1]], c=col, marker=mk,
               s=sz, alpha=0.95, linewidths=0.7, edgecolors='white', zorder=8)


# ── Scatter helpers ────────────────────────────────────────────────────────────
S = 1.2

def _ax_base(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')
    for sp in ax.spines.values(): sp.set_visible(False)

def _red_border(ax):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor('#C62828'); sp.set_linewidth(1.2)

def scatter_tissue(ax, xs, ys, types):
    _ax_base(ax)
    for tt in TYPE_ORDER:
        m = np.array(types) == tt
        if not m.any(): continue
        ax.scatter(xs[m], ys[m], c=TYPE_COLORS[tt], s=S,
                   alpha=0.80, linewidths=0, rasterized=True)

def scatter_cluster(ax, xs, ys, cidx, n_clusters):
    _ax_base(ax)
    cmap = plt.cm.get_cmap('tab20', max(n_clusters, 1))
    for ci in range(n_clusters):
        m = cidx == ci
        if not m.any(): continue
        ax.scatter(xs[m], ys[m], c=[cmap(ci % 20)], s=S,
                   alpha=0.80, linewidths=0, rasterized=True)

def scatter_score(ax, xs, ys, scores, vmin, vmax, cmap='hot_r',
                  sc_store=None, key=None):
    _ax_base(ax)
    order = np.argsort(scores)
    sc = ax.scatter(xs[order], ys[order], c=scores[order],
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    s=S, alpha=0.90, linewidths=0, rasterized=True)
    if sc_store is not None and key is not None and sc_store.get(key) is None:
        sc_store[key] = sc
    return sc


# ── Main loop ──────────────────────────────────────────────────────────────────
acr_patients = set(obs.loc[obs['acr_binary'].astype(str).isin(['1','1.0','True']),
                            'record_id'].values)
print(f"[main] {len(acr_patients)} ACR+ patients", flush=True)

N_DATA_COLS = 8   # tissue|nbhd_id|flat_abmil|pa|na|joint|knn|simplex
LABEL_W = 1.8; COL_W = 2.8; ROW_H = 2.8

n_done = 0
for pid in sorted(acr_patients):
    stems    = pat_to_stems.get(pid, [])
    he_stems = [s for s in stems if stem_meta[s].get('has_HE', False)]
    if not he_stems: continue
    he_stems.sort(key=lambda s: stem_meta[s]['anchor_dt'])

    print(f"\nPatient {pid}: {len(he_stems)} H&E biopsies", flush=True)

    records = []
    for stem in he_stems:
        meta      = stem_meta[stem]
        fold      = stem_to_fold.get(stem, None)
        label     = meta['label']
        anchor    = meta['anchor_dt']
        sample_p  = SAMPLES_DIR / stem
        feat_p, slide_name = find_feat_file(pid, anchor)

        # Train/test label for split 0
        if fold is not None:
            split_tag = f"Test f{fold}"
        else:
            split_tag = "Train"

        # Tissue type from h5ad
        xs_t = ys_t = tt = np.array([]); acr_status = 'A0B0'
        if slide_name and slide_name in slide_kd:
            sl_obs = obs[obs['slide_name_clean'] == slide_name]
            xs_t = sl_obs['chunk_left'].values
            ys_t = -sl_obs['chunk_top'].values
            tt   = sl_obs['tissue_type'].values
            if len(sl_obs): acr_status = sl_obs['acr_status'].values[0]

        # Flat ABMIL + KNN
        enc = p1_encoders.get(fold if fold is not None else 0, p1_encoders.get(0))
        coords_ab = alpha_ab = knn_ab = None
        if enc is not None and sample_p.exists():
            try:
                coords_ab, alpha_ab, knn_ab = run_flat_abmil(enc, sample_p)
                if coords_ab is not None:
                    coords_ab = np.c_[coords_ab[:, 0], -coords_ab[:, 1]]
            except Exception as e:
                print(f"  [warn] flat ABMIL {stem}: {e}")

        # 2-level ABMIL
        coords_nb = cidx = na_logit = pa = joint = None
        n_clusters = 0; prob_nb = float('nan')
        simp_pos_c = np.zeros((0, 2)); simp_bag = np.zeros(2); na_raw_c = np.zeros(0)
        m2 = abmil2_models.get(fold if fold is not None else 0, abmil2_models.get(0))
        if m2 is not None and feat_p is not None:
            try:
                coords_nb, cidx, na_logit, pa, joint, prob_nb, n_clusters, \
                    _coord_list, _pa_list, _na_raw = run_nbhd_abmil(m2, feat_p)
                coords_nb = np.c_[coords_nb[:, 0], -coords_nb[:, 1]]
                # Compute simplex positions using slide tissue annotation
                na_raw_c   = _na_raw   # (C,) raw na logits
                na_sm_c    = np.exp(na_raw_c - na_raw_c.max())
                na_sm_c   /= na_sm_c.sum()
                simp_pos_c = np.zeros((n_clusters, 2))
                for ci, (co, pa_np) in enumerate(zip(_coord_list, _pa_list)):
                    w = attn_weighted_composition(co, pa_np, slide_name or '')
                    simp_pos_c[ci] = w @ VERTEX
                simp_bag = na_sm_c @ simp_pos_c   # (2,) bag rep on simplex
            except Exception as e:
                print(f"  [warn] 2-level ABMIL {slide_name}: {e}")

        records.append({
            'stem': stem, 'label': label, 'acr_status': acr_status,
            'anchor': anchor, 'split_tag': split_tag,
            'xs_t': xs_t, 'ys_t': ys_t, 'tt': tt,
            'coords_ab': coords_ab, 'alpha_ab': alpha_ab, 'knn_ab': knn_ab,
            'coords_nb': coords_nb, 'cidx': cidx, 'n_clusters': n_clusters,
            'na_logit': na_logit, 'pa': pa, 'joint': joint, 'prob_nb': prob_nb,
            'simp_pos': simp_pos_c,   # (C, 2) simplex positions per neighbourhood
            'simp_bag': simp_bag,     # (2,)   bag rep on simplex
            'na_raw':   na_raw_c,     # (C,)   raw na[c] logits for colour
        })

    if not records: continue
    n_rows = len(records)
    n_pos  = sum(r['label'] == 1 for r in records)

    # ── Shared colour ranges ────────────────────────────────────────────────
    def _cat(*arrs):
        out = [a for a in arrs if a is not None and len(a) > 0]
        return np.concatenate(out) if out else np.array([0.0])

    def _vrange(arr, lo=1, hi=99):
        v0 = float(np.percentile(arr, lo)); v1 = float(np.percentile(arr, hi))
        return (v0, v0 + 1e-6) if v1 <= v0 else (v0, v1)

    all_ab  = _cat(*[r['alpha_ab'] for r in records if r['alpha_ab'] is not None])
    all_knn = _cat(*[r['knn_ab']   for r in records if r['knn_ab']   is not None])
    all_pa  = _cat(*[r['pa']       for r in records if r['pa']       is not None])
    all_na  = _cat(*[r['na_logit'] for r in records if r['na_logit'] is not None])
    all_jt  = _cat(*[r['joint']    for r in records if r['joint']    is not None])

    vmin_ab,  vmax_ab  = _vrange(all_ab)
    vmin_knn, vmax_knn = _vrange(all_knn)
    vmin_pa,  vmax_pa  = _vrange(all_pa)
    vmin_na,  vmax_na  = _vrange(all_na)
    vmin_jt,  vmax_jt  = _vrange(all_jt)

    # ── Shared simplex colour range across all biopsies for this patient ───
    all_na_simp = np.concatenate([r['na_raw'] for r in records if len(r['na_raw']) > 0]) \
                  if any(len(r['na_raw']) > 0 for r in records) else np.array([0.0])
    sna_vmin, sna_vmax = _vrange(all_na_simp)

    # ── Figure ─────────────────────────────────────────────────────────────
    fig_w = LABEL_W + N_DATA_COLS * COL_W + 0.6
    fig_h = n_rows * ROW_H + 1.6
    fig   = plt.figure(figsize=(fig_w, fig_h))

    width_ratios = [LABEL_W / COL_W] + [1] * N_DATA_COLS
    gs = GridSpec(n_rows, 1 + N_DATA_COLS, figure=fig,
                  width_ratios=width_ratios,
                  wspace=0.04, hspace=0.30,
                  left=0.01, right=0.99,
                  top=0.90, bottom=0.05)

    COL_TITLES = [
        '',
        'Tissue type\n(h5ad annotation)',
        'Neighbourhood ID\n(spatial cluster)',
        'ABMIL α\n(flat, v8 Ph-1)',
        'pa[c]\n(within-nbhd)',
        'na[c]\n(nbhd logit)',
        'na_s×pa\n(joint)',
        f'KNN-ABMIL\n(K={KNN_K}, local)',
        'Simplex\n(nbhd pos | na[c])',
    ]
    for ci, ct in enumerate(COL_TITLES[1:], start=1):
        ax_hdr = fig.add_subplot(gs[0, ci])
        ax_hdr.set_title(ct, fontsize=6.5, pad=3, fontweight='bold')

    sc_refs = {'ab': None, 'knn': None, 'pa': None, 'na': None, 'jt': None, 'simplex': None}

    for ri, rec in enumerate(records):
        is_pos    = rec['label'] == 1
        acr_col   = ACR_GRADE_COLOR.get(rec['acr_status'], '#888888')
        acr_lbl   = ACR_GRADE_LABEL.get(rec['acr_status'], rec['acr_status'])

        # ── Row label ─────────────────────────────────────────────────────
        ax_lbl = fig.add_subplot(gs[ri, 0])
        ax_lbl.axis('off')
        ax_lbl.text(0.95, 0.78, acr_lbl,
                    transform=ax_lbl.transAxes, ha='right', va='center',
                    fontsize=7.5, fontweight='bold', color=acr_col)
        ax_lbl.text(0.95, 0.55, rec['anchor'][:10],
                    transform=ax_lbl.transAxes, ha='right', va='center',
                    fontsize=6, color='#444444')
        ax_lbl.text(0.95, 0.34, rec['split_tag'],
                    transform=ax_lbl.transAxes, ha='right', va='center',
                    fontsize=5.5,
                    color='#1565C0' if 'Test' in rec['split_tag'] else '#555555')
        if not np.isnan(rec['prob_nb']):
            ax_lbl.text(0.95, 0.14, f"p={rec['prob_nb']:.2f}",
                        transform=ax_lbl.transAxes, ha='right', va='center',
                        fontsize=5.5, color='#666666')

        # ── Col 1: Tissue type ────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[ri, 1])
        if len(rec['xs_t']) > 0:
            scatter_tissue(ax1, rec['xs_t'], rec['ys_t'], rec['tt'])
        else: ax1.axis('off')
        if is_pos: _red_border(ax1)

        # ── Col 2: Neighbourhood ID ───────────────────────────────────────
        ax2 = fig.add_subplot(gs[ri, 2])
        if rec['coords_nb'] is not None and rec['cidx'] is not None:
            scatter_cluster(ax2, rec['coords_nb'][:, 0], rec['coords_nb'][:, 1],
                            rec['cidx'], rec['n_clusters'])
            ax2.set_title(f"{rec['n_clusters']} nbhds", fontsize=5, pad=1, color='#555')
        else: ax2.axis('off')
        if is_pos: _red_border(ax2)

        # ── Col 3: Flat ABMIL α ───────────────────────────────────────────
        ax3 = fig.add_subplot(gs[ri, 3])
        if rec['coords_ab'] is not None and rec['alpha_ab'] is not None:
            scatter_score(ax3, rec['coords_ab'][:, 0], rec['coords_ab'][:, 1],
                          rec['alpha_ab'], vmin_ab, vmax_ab, 'hot_r',
                          sc_store=sc_refs, key='ab')
        else: ax3.axis('off')
        if is_pos: _red_border(ax3)

        # ── Col 4: pa[c] ──────────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[ri, 4])
        if rec['coords_nb'] is not None and rec['pa'] is not None:
            scatter_score(ax4, rec['coords_nb'][:, 0], rec['coords_nb'][:, 1],
                          rec['pa'], vmin_pa, vmax_pa, 'hot_r',
                          sc_store=sc_refs, key='pa')
        else: ax4.axis('off')
        if is_pos: _red_border(ax4)

        # ── Col 5: na[c] ──────────────────────────────────────────────────
        ax5 = fig.add_subplot(gs[ri, 5])
        if rec['coords_nb'] is not None and rec['na_logit'] is not None:
            scatter_score(ax5, rec['coords_nb'][:, 0], rec['coords_nb'][:, 1],
                          rec['na_logit'], vmin_na, vmax_na, 'hot_r',
                          sc_store=sc_refs, key='na')
        else: ax5.axis('off')
        if is_pos: _red_border(ax5)

        # ── Col 6: Joint ──────────────────────────────────────────────────
        ax6 = fig.add_subplot(gs[ri, 6])
        if rec['coords_nb'] is not None and rec['joint'] is not None:
            scatter_score(ax6, rec['coords_nb'][:, 0], rec['coords_nb'][:, 1],
                          rec['joint'], vmin_jt, vmax_jt, 'hot_r',
                          sc_store=sc_refs, key='jt')
        else: ax6.axis('off')
        if is_pos: _red_border(ax6)

        # ── Col 7: KNN-masked ABMIL ───────────────────────────────────────
        ax7 = fig.add_subplot(gs[ri, 7])
        if rec['coords_ab'] is not None and rec['knn_ab'] is not None:
            scatter_score(ax7, rec['coords_ab'][:, 0], rec['coords_ab'][:, 1],
                          rec['knn_ab'], vmin_knn, vmax_knn, 'hot_r',
                          sc_store=sc_refs, key='knn')
        else: ax7.axis('off')
        if is_pos: _red_border(ax7)

        # ── Col 8: Simplex (neighbourhood tissue composition) ─────────────
        ax8 = fig.add_subplot(gs[ri, 8])
        if len(rec['simp_pos']) > 0:
            draw_simplex_mini(ax8, rec['simp_pos'], rec['na_raw'],
                              sna_vmin, sna_vmax,
                              rec['simp_bag'], rec['label'],
                              sc_store=sc_refs)
        else:
            ax8.axis('off')
        if is_pos: _red_border(ax8)

    # ── Shared colourbars (above grid) ────────────────────────────────────
    total_w = sum(width_ratios)
    col_starts = np.cumsum([0] + width_ratios) / total_w
    cbar_specs = [
        (3, 'ab',      'ABMIL α'),
        (4, 'pa',      'pa[c]'),
        (5, 'na',      'na[c]'),
        (6, 'jt',      'Joint'),
        (7, 'knn',     'KNN-ABMIL'),
        (8, 'simplex', 'na[c] (simplex)'),
    ]
    for ci, key, lbl in cbar_specs:
        sc = sc_refs.get(key)
        if sc is None: continue
        cx0 = col_starts[ci] + 0.005
        cw  = (width_ratios[ci] / total_w) * 0.85
        cax = fig.add_axes([cx0, 0.921, cw, 0.011])
        cb  = fig.colorbar(sc, cax=cax, orientation='horizontal')
        cb.set_label(lbl, fontsize=5)
        cb.ax.tick_params(labelsize=4, labelbottom=False, labeltop=True)

    # ── Tissue legend ─────────────────────────────────────────────────────
    handles = [mpatches.Patch(color=TYPE_COLORS[t], label=t)
               for t in TYPE_ORDER if t != 'Unknown']
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               fontsize=5.5, frameon=False, bbox_to_anchor=(0.5, -0.01),
               columnspacing=0.6, handlelength=0.9)

    fig.suptitle(
        f"Patient {pid}  |  {n_rows} H&E biopsies  |  {n_pos} ACR+  |  "
        "red border = ACR+  |  blue label = split-0 test set\n"
        "Col3: flat ABMIL α  |  Col4: 1st-lvl pa[c]  |  Col5: 2nd-lvl na[c]  |  "
        f"Col6: joint = softmax(na)×pa  |  Col7: KNN-ABMIL (K={KNN_K})  |  "
        "Col8: Simplex — each dot = neighbourhood, pos = tissue composition, "
        "colour = na[c], ▲/● = ACR+/− bag rep",
        fontsize=6.5, y=0.99)

    out_pdf = OUT_DIR / f"{pid}.pdf"
    out_png = OUT_DIR / f"{pid}.png"
    fig.savefig(out_pdf, dpi=150, bbox_inches='tight')
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)
    n_done += 1
    print(f"  Saved {pid}.pdf  ({n_rows} rows, {n_pos} ACR+)", flush=True)

print(f"\n[done] {n_done} patient PDFs → {OUT_DIR}")
PYEOF
