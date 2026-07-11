#!/usr/bin/env bash
#SBATCH --job-name=vis_spatial_attn
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/spatial_attn_combined/job_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/spatial_attn_combined/job_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

mkdir -p /lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/spatial_attn_combined

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
"""
Combined spatial + attention figure per patient.

For every ACR+ patient, one figure with rows = biopsies (sorted by date) and
6 columns:
  1. Tissue type             (h5ad tissue_type annotation, chunk_left/top coords)
  2. ABMIL attention         (v8 Phase-1 HE encoder, flat softmax alpha)
  3. Neighbourhood score     (SpatialABMIL2Level na[c] broadcast to cluster patches)
  4. Instance score          (SpatialABMIL2Level pa[c][i] within-cluster softmax)
  5. Joint = nbhd × instance (unnormalised product)
  6. Composition bar         (tissue type fractions per biopsy)

Colour axes for columns 2–5: shared 99th-pct vmax across ALL timepoints for
the patient (raw softmax, not per-slide normalised) so ACR+ slides appear
genuinely brighter.

ACR+ rows: red spines.
"""

import sys, warnings, re
from pathlib import Path
from collections import defaultdict

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
from matplotlib.gridspec import GridSpec
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
H5AD        = Path("/lustre/groups/aih/dinesh.haridoss/datasets/adata_v3.h5ad")
SAMPLES_DIR = Path("/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples")
FEAT_DIR    = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/slide_cluster_feats")
V8_BASE     = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/results_mm_abmil_v8")
SPLITS_CSV  = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
OUT_DIR     = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/spatial_attn_combined")
OUT_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN   = 256
DROPOUT  = 0.4
IN_DIM   = 1024
KNN_K    = 8

# ── Tissue-type colour palette (same as plot_spatial_patches.sh) ───────────────
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
    """V8 Phase-1 single-level ABMIL encoder."""
    def __init__(self, feat_dim=IN_DIM, hidden_dim=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.att_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.att_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.att_w = nn.Linear(hidden_dim, 1, bias=False)
        self.att_drop = nn.Dropout(dropout)
    def forward(self, x):
        h     = self.backbone(x)
        gate  = self.att_V(h) * self.att_U(h)
        alpha = self.att_w(self.att_drop(gate)).squeeze(1)   # raw logits, NOT softmax
        return alpha


class GatedPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(dim, dim // 2), nn.Sigmoid())
        self.w = nn.Linear(dim // 2, 1, bias=False)
    def forward(self, h):
        a = F.softmax(self.w(self.V(h) * self.U(h)), dim=0)
        return (a * h).sum(0), a.squeeze(-1)


class GatedAttentionPool(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(in_dim, hidden), nn.Sigmoid())
        self.w = nn.Linear(hidden, 1, bias=False)
    def forward(self, h):
        # pool with softmax for the bag rep, but also return raw logits
        raw = self.w(self.V(h) * self.U(h))   # (1, C, 1) or (1, N, 1)
        a   = F.softmax(raw, dim=1)
        return (a * h).sum(dim=1), raw.squeeze(-1).squeeze(0)   # rep, raw_logits


class SpatialABMIL2Level(nn.Module):
    """Neighbourhood ABMIL: returns (logit, na_raw, patch_attns_raw)."""
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.patch_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.patch_attn = GatedAttentionPool(hidden, hidden // 2)
        self.nbhd_proj  = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.nbhd_attn  = GatedAttentionPool(hidden, hidden // 2)
        self.head       = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, clusters):
        nbhd_reps, patch_attns_raw = [], []
        for c in clusters:
            h = self.patch_proj(c.unsqueeze(0))
            rep, pa_raw = self.patch_attn(h)
            nbhd_reps.append(rep)
            patch_attns_raw.append(pa_raw.squeeze(0))
        H = self.nbhd_proj(torch.stack(nbhd_reps, dim=1))
        slide_rep, na_raw = self.nbhd_attn(H)
        return self.head(slide_rep).squeeze(-1), na_raw.squeeze(0), patch_attns_raw


# ── Load splits CSV → stem→(patient_id, anchor_dt, label, fold) ───────────────
print("[splits] Loading ...", flush=True)
df_sp = pd.read_csv(SPLITS_CSV)
# Use split0: fold column is fold_0..fold_3 → "split0_fold{f}" style
# Detect column naming
fold_cols = [c for c in df_sp.columns if c.startswith("fold_") or c.startswith("split0_fold")]
# fall back to any fold_X columns
if not fold_cols:
    fold_cols = [c for c in df_sp.columns if "fold" in c.lower()]

# Build stem → test_fold (which fold is this sample the test set for, in split 0)
stem_to_fold: dict = {}
for fi, col in enumerate(fold_cols[:4]):
    test_mask = df_sp[col] == "test"
    for stem in df_sp.loc[test_mask, "file"].values:
        stem_to_fold[str(stem)] = fi

# Build stem → meta (include has_HE flag)
stem_meta: dict = {}
for _, row in df_sp.iterrows():
    stem_meta[str(row["file"])] = {
        "patient_id": str(row["patient_id"]),
        "anchor_dt":  str(row["anchor_dt"]),
        "label":      int(row["label"]) if pd.notna(row["label"]) else 0,
        "has_HE":     bool(row.get("has_HE", False)),
    }

print(f"[splits] {len(stem_meta)} stems, {len(stem_to_fold)} test assignments", flush=True)


# ── Load v8 Phase-1 HE encoders (split0, folds 0-3) ──────────────────────────
print("[models] Loading v8 Phase-1 HE encoders ...", flush=True)
p1_encoders: dict = {}   # fold → GatedAttentionEncoder
for fold in range(4):
    ckpt = V8_BASE / "acr" / f"split0_fold{fold}" / "phase1" / "enc_HE.pt"
    if not ckpt.exists():
        print(f"  [warn] Missing {ckpt}"); continue
    enc = GatedAttentionEncoder()
    sd  = torch.load(ckpt, map_location="cpu", weights_only=True)
    # strip "encoder." prefix if saved as full SingleModalMIL state_dict
    if any(k.startswith("encoder.") for k in sd):
        sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    try:
        enc.load_state_dict(sd, strict=True)
    except RuntimeError:
        enc.load_state_dict(sd, strict=False)
    enc.eval()
    p1_encoders[fold] = enc
print(f"[models] Phase-1 encoders for folds: {sorted(p1_encoders)}", flush=True)

# ── Load SpatialABMIL2Level checkpoints (folds 0-4) ──────────────────────────
ABMIL2_DIR = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/soft_hier_mil")
print("[models] Loading SpatialABMIL2Level checkpoints ...", flush=True)
abmil2_models: dict = {}
for fold in range(5):
    ckpt = ABMIL2_DIR / f"abmil_fold{fold}.pt"
    if not ckpt.exists():
        continue
    m = SpatialABMIL2Level()
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    abmil2_models[fold] = m
print(f"[models] 2-level ABMIL for folds: {sorted(abmil2_models)}", flush=True)


# ── Read h5ad obs only (tissue type + coords) ─────────────────────────────────
print("[h5ad] Loading obs ...", flush=True)
adata = ad.read_h5ad(H5AD, backed='r')
obs = adata.obs[[
    'chunk_left', 'chunk_top', 'tissue_type',
    'slide_name_clean', 'record_id', 'biopsy_date', 'acr_status', 'acr_binary'
]].copy()
adata.file.close()
obs['chunk_left']  = obs['chunk_left'].astype(float)
obs['chunk_top']   = obs['chunk_top'].astype(float)
# biopsy_date may be Categorical — convert via string to avoid dtype issues
obs['biopsy_date_str'] = obs['biopsy_date'].astype(str).str[:10]   # YYYY-MM-DD
print(f"[h5ad] {len(obs):,} patches, {obs['record_id'].nunique()} patients", flush=True)

# Build: (patient_id, date_str YYYY-MM-DD) → slide_name_clean
_uniq = (obs[['record_id','biopsy_date_str','slide_name_clean']]
         .drop_duplicates('slide_name_clean'))
date_to_slide: dict = {}
for _, row in _uniq.iterrows():
    date_to_slide[(str(row['record_id']), str(row['biopsy_date_str'])[:10])] = \
        str(row['slide_name_clean'])

# ── Build patient → list of stems from splits CSV ─────────────────────────────
pat_to_stems: dict = defaultdict(list)
for stem, meta in stem_meta.items():
    pat_to_stems[meta["patient_id"]].append(stem)

# ── Cluster feat file lookup ───────────────────────────────────────────────────
feat_files = {p.stem: p for p in sorted(FEAT_DIR.glob("*.pt"))}

def find_feat_file(patient_id: str, biopsy_date_str: str):
    """
    Find cluster feat file for a patient's biopsy via h5ad date→slide_name_clean.
    slide_name_clean is like 'LT001-DST0035' which is exactly the feat file stem.
    Returns (feat_path, slide_name_clean) or (None, None) if no HE slide for this date.
    """
    date_key   = (patient_id, str(biopsy_date_str)[:10])
    slide_name = date_to_slide.get(date_key)
    if slide_name is None:
        return None, None   # no HE data for this biopsy date — skip row
    sn = str(slide_name)
    if sn in feat_files:
        return feat_files[sn], sn
    return None, None


# ── Inference helpers ──────────────────────────────────────────────────────────
@torch.no_grad()
def run_flat_abmil(enc: GatedAttentionEncoder, sample_path: Path):
    """
    Run v8 Phase-1 encoder on flat HE_cells features.
    Returns: coords (N,2), alpha (N,) raw softmax.
    """
    d = torch.load(sample_path, map_location='cpu', weights_only=False)
    feats  = d.get('inputs', {}).get('HE_cells')
    coords = d.get('instance_spatial_coords', {}).get('HE_cells')
    if feats is None or coords is None or feats.shape[0] == 0:
        return None, None
    feats  = feats.float()
    coords = coords.float()
    if feats.shape[0] > 3000:
        idx    = torch.randperm(feats.shape[0])[:3000]
        feats  = feats[idx]
        coords = coords[idx]
    alpha = enc(feats).numpy()
    return coords.numpy(), alpha


@torch.no_grad()
def run_nbhd_abmil(model: SpatialABMIL2Level, feat_path: Path):
    """
    Run SpatialABMIL2Level on cluster feat file.
    Returns: coords (N,2), nbhd_score (N,), inst_score (N,), joint (N,), prob float.
    """
    d = torch.load(feat_path, map_location='cpu', weights_only=False)
    clusters_raw = d['clusters']
    clusters_t   = []
    coord_list   = []

    for c in clusters_raw:
        f  = c['feats'].float()
        co = c['coords'].float()
        if f.shape[0] > 512:
            idx = torch.randperm(f.shape[0])[:512]
            f   = f[idx]; co = co[idx]
        clusters_t.append(f)
        coord_list.append(co)

    logit, na_raw, patch_attns = model(clusters_t)
    prob  = torch.sigmoid(logit).item()
    na_np = na_raw.cpu().numpy()   # (C,) raw neighbourhood logits, NOT softmax

    all_coords, all_nbhd, all_inst = [], [], []
    for ci, (co, pa) in enumerate(zip(coord_list, patch_attns)):
        n = co.shape[0]
        all_coords.append(co.numpy())
        all_nbhd.append(np.full(n, na_np[ci]))
        all_inst.append(pa.cpu().numpy()[:n])

    coords     = np.concatenate(all_coords, axis=0)
    nbhd_score = np.concatenate(all_nbhd)
    inst_score = np.concatenate(all_inst)
    joint      = nbhd_score * inst_score
    return coords, nbhd_score, inst_score, joint, prob


# ── Scatter plotting helpers ───────────────────────────────────────────────────
S_TISSUE = 1.2   # marker size for tissue type
S_ATTN   = 1.2   # marker size for attention

def _scatter_tissue(ax, xs, ys, types, title=''):
    for tt in TYPE_ORDER:
        m = np.array(types) == tt
        if m.sum() == 0: continue
        ax.scatter(xs[m], ys[m], c=TYPE_COLORS[tt],
                   s=S_TISSUE, alpha=0.75, linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')
    for sp in ax.spines.values(): sp.set_visible(False)
    if title: ax.set_title(title, fontsize=6, pad=2, fontweight='bold')


def _scatter_attn(ax, xs, ys, scores, vmin, vmax, cmap='hot_r',
                  title='', sc_store=None, key=None):
    order = np.argsort(scores)
    sc = ax.scatter(xs[order], ys[order], c=scores[order],
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    s=S_ATTN, alpha=0.9, linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')
    for sp in ax.spines.values(): sp.set_visible(False)
    if title: ax.set_title(title, fontsize=6, pad=2, fontweight='bold')
    if sc_store is not None and key is not None and sc_store.get(key) is None:
        sc_store[key] = sc
    return sc


def _red_border(ax):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor('#C62828')
        sp.set_linewidth(1.2)


# ── Figure legend handles ──────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=TYPE_COLORS[t], label=t)
    for t in TYPE_ORDER if t != 'Unknown'
]


# ── Main loop ─────────────────────────────────────────────────────────────────
# Only process patients with ≥1 ACR+ slide
acr_patients = set(obs.loc[obs['acr_binary'].astype(str).isin(['1','1.0','True']),
                             'record_id'].values)
print(f"[main] {len(acr_patients)} patients with ACR+ slides", flush=True)

N_COLS   = 6   # tissue | ABMIL | nbhd | instance | joint | comp-bar
COL_W    = 2.8
COMP_W   = 1.8
ROW_H    = 2.8
LABEL_W  = 1.6

n_done = 0
for pid in sorted(acr_patients):
    stems = pat_to_stems.get(pid, [])
    # Filter to stems that have a matching H&E slide (date_to_slide lookup)
    he_stems = [s for s in stems if stem_meta[s].get('has_HE', False)]
    if not he_stems:
        continue

    # Sort by biopsy date
    he_stems.sort(key=lambda s: stem_meta[s]['anchor_dt'])
    n_rows = len(he_stems)

    print(f"\nPatient {pid}: {n_rows} timepoints", flush=True)

    # ── Collect data for all timepoints ──────────────────────────────────────
    records = []
    for stem in he_stems:
        meta    = stem_meta[stem]
        fold    = stem_to_fold.get(stem, 0)
        label   = meta['label']
        anchor  = meta['anchor_dt']

        sample_path = SAMPLES_DIR / stem
        feat_path, slide_name = find_feat_file(pid, anchor)

        if not sample_path.exists():
            print(f"  skip {stem}: sample file missing"); continue

        # Tissue type from h5ad
        if slide_name is not None:
            sl_obs = obs[obs['slide_name_clean'] == slide_name]
            xs_t = sl_obs['chunk_left'].values
            ys_t = -sl_obs['chunk_top'].values
            tt   = sl_obs['tissue_type'].values
            acr_status = sl_obs['acr_status'].values[0] if len(sl_obs) else 'A0B0'
        else:
            xs_t = ys_t = tt = np.array([])
            acr_status = 'A0B0'

        # Flat ABMIL (v8 Phase-1)
        enc = p1_encoders.get(fold, p1_encoders.get(0))
        coords_ab = alpha_ab = None
        if enc is not None:
            coords_ab, alpha_ab = run_flat_abmil(enc, sample_path)
            if coords_ab is not None:
                coords_ab = np.c_[coords_ab[:, 0], -coords_ab[:, 1]]

        # 2-level neighbourhood ABMIL
        coords_nb = nbhd = inst = joint = None
        prob_nb = np.nan
        abmil2 = abmil2_models.get(fold, abmil2_models.get(0))
        if abmil2 is not None and feat_path is not None:
            try:
                c_nb, nb, ins, jt, prob_nb = run_nbhd_abmil(abmil2, feat_path)
                coords_nb = np.c_[c_nb[:, 0], -c_nb[:, 1]]
                nbhd = nb; inst = ins; joint = jt
            except Exception as e:
                print(f"  [warn] 2-level ABMIL error {stem}: {e}")

        records.append({
            'stem': stem, 'label': label, 'acr_status': acr_status,
            'anchor': anchor,
            'xs_t': xs_t, 'ys_t': ys_t, 'tt': tt,
            'coords_ab': coords_ab, 'alpha_ab': alpha_ab,
            'coords_nb': coords_nb, 'nbhd': nbhd, 'inst': inst, 'joint': joint,
            'prob_nb': prob_nb,
        })

    if not records:
        continue

    n_rows = len(records)
    n_pos  = sum(r['label'] == 1 for r in records)

    # ── Shared colour axes (99th pct across all timepoints) ──────────────────
    def _cat(*arrs):
        out = [a for a in arrs if a is not None and len(a) > 0]
        return np.concatenate(out) if out else np.array([0.0])

    all_ab   = _cat(*[r['alpha_ab'] for r in records if r['alpha_ab'] is not None])
    all_nbhd = _cat(*[r['nbhd']     for r in records if r['nbhd']     is not None])
    all_inst = _cat(*[r['inst']      for r in records if r['inst']     is not None])
    all_jt   = _cat(*[r['joint']     for r in records if r['joint']    is not None])

    def _vrange(arr, lo=1, hi=99):
        v0 = float(np.percentile(arr, lo))
        v1 = float(np.percentile(arr, hi))
        if v1 <= v0: v1 = v0 + 1e-6
        return v0, v1

    vmin_ab,   vmax_ab   = _vrange(all_ab)
    vmin_nbhd, vmax_nbhd = _vrange(all_nbhd)
    vmin_inst, vmax_inst = _vrange(all_inst)
    vmin_jt,   vmax_jt   = _vrange(all_jt)

    # ── Layout: rows = timepoints, cols = label|tissue|abmil|nbhd|inst|joint|comp ─
    fig_w = LABEL_W + 5 * COL_W + COMP_W + 0.5
    fig_h = n_rows * ROW_H + 1.2
    fig   = plt.figure(figsize=(fig_w, fig_h))

    width_ratios = [LABEL_W/COL_W] + [1]*5 + [COMP_W/COL_W]
    gs = GridSpec(n_rows, 7, figure=fig,
                  width_ratios=width_ratios,
                  wspace=0.04, hspace=0.30,
                  left=0.01, right=0.97,
                  top=0.91, bottom=0.06)

    col_titles = ['', 'Tissue type\n(h&e annotation)',
                  'ABMIL\n(v8 Phase-1, flat)',
                  'Neighbourhood\nscore (raw)',
                  'Instance score\n(raw, within nbhd)',
                  'Joint\n(nbhd × instance)',
                  'Composition']
    for ci, ct in enumerate(col_titles[1:6], start=1):
        ax_hdr = fig.add_subplot(gs[0, ci])
        ax_hdr.set_title(ct, fontsize=6.5, pad=3, fontweight='bold')

    sc_refs = {'ab': None, 'nbhd': None, 'inst': None, 'joint': None}

    for ri, rec in enumerate(records):
        xs_t, ys_t, tt = rec['xs_t'], rec['ys_t'], rec['tt']
        acr_lbl  = ACR_GRADE_LABEL.get(rec['acr_status'], rec['acr_status'])
        acr_col  = ACR_GRADE_COLOR.get(rec['acr_status'], '#888888')
        is_pos   = rec['label'] == 1

        # ── Row label ─────────────────────────────────────────────────────────
        ax_lbl = fig.add_subplot(gs[ri, 0])
        ax_lbl.axis('off')
        ax_lbl.text(0.95, 0.70, acr_lbl,
                    transform=ax_lbl.transAxes, ha='right', va='center',
                    fontsize=8, fontweight='bold', color=acr_col)
        ax_lbl.text(0.95, 0.45, rec['anchor'][:10],
                    transform=ax_lbl.transAxes, ha='right', va='center',
                    fontsize=6, color='#444444')
        if rec['prob_nb'] is not np.nan and not np.isnan(rec['prob_nb']):
            ax_lbl.text(0.95, 0.22, f"p={rec['prob_nb']:.2f}",
                        transform=ax_lbl.transAxes, ha='right', va='center',
                        fontsize=5.5, color='#666666')

        # ── Col 1: Tissue type ────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[ri, 1])
        if len(xs_t) > 0:
            _scatter_tissue(ax1, xs_t, ys_t, tt)
        if is_pos: _red_border(ax1)

        # ── Col 2: ABMIL flat alpha ───────────────────────────────────────────
        ax2 = fig.add_subplot(gs[ri, 2])
        if rec['coords_ab'] is not None and rec['alpha_ab'] is not None:
            _scatter_attn(ax2, rec['coords_ab'][:,0], rec['coords_ab'][:,1],
                          rec['alpha_ab'], vmin_ab, vmax_ab, 'RdYlGn',
                          sc_store=sc_refs, key='ab')
        if is_pos: _red_border(ax2)

        # ── Col 3: Neighbourhood score ────────────────────────────────────────
        ax3 = fig.add_subplot(gs[ri, 3])
        if rec['coords_nb'] is not None and rec['nbhd'] is not None:
            _scatter_attn(ax3, rec['coords_nb'][:,0], rec['coords_nb'][:,1],
                          rec['nbhd'], vmin_nbhd, vmax_nbhd, 'RdYlGn',
                          sc_store=sc_refs, key='nbhd')
        if is_pos: _red_border(ax3)

        # ── Col 4: Instance score ─────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[ri, 4])
        if rec['coords_nb'] is not None and rec['inst'] is not None:
            _scatter_attn(ax4, rec['coords_nb'][:,0], rec['coords_nb'][:,1],
                          rec['inst'], vmin_inst, vmax_inst, 'RdYlGn',
                          sc_store=sc_refs, key='inst')
        if is_pos: _red_border(ax4)

        # ── Col 5: Joint = nbhd × instance ───────────────────────────────────
        ax5 = fig.add_subplot(gs[ri, 5])
        if rec['coords_nb'] is not None and rec['joint'] is not None:
            _scatter_attn(ax5, rec['coords_nb'][:,0], rec['coords_nb'][:,1],
                          rec['joint'], vmin_jt, vmax_jt, 'hot_r',
                          sc_store=sc_refs, key='joint')
        if is_pos: _red_border(ax5)

    # ── Composition stacked bar (rightmost column, all rows merged) ───────────
    ax_comp = fig.add_subplot(gs[:, 6])
    dates_for_comp = [r['anchor'][:10] for r in records]
    comp_vals = {}
    for r in records:
        key = r['anchor'][:10]
        if len(r['tt']) > 0:
            from collections import Counter
            c = Counter(r['tt'])
            tot = sum(c.values())
            comp_vals[key] = {t: c.get(t, 0)/tot for t in TYPE_ORDER}
        else:
            comp_vals[key] = {t: 0.0 for t in TYPE_ORDER}

    bottoms = np.zeros(len(dates_for_comp))
    for tt in TYPE_ORDER:
        vals = np.array([comp_vals[d].get(tt, 0.0) for d in dates_for_comp])
        ax_comp.barh(range(len(dates_for_comp)), vals, left=bottoms,
                     color=TYPE_COLORS[tt], height=0.65)
        bottoms += vals
    ax_comp.set_yticks(range(len(dates_for_comp)))
    ax_comp.set_yticklabels(dates_for_comp, fontsize=6)
    ax_comp.set_xlim(0, 1); ax_comp.set_xlabel('Fraction', fontsize=7)
    ax_comp.set_title('Comp.', fontsize=7, pad=3)
    ax_comp.tick_params(axis='x', labelsize=6)
    for sp in ['top', 'right']: ax_comp.spines[sp].set_visible(False)

    # ── Shared colourbars (above the grid) ────────────────────────────────────
    cbar_specs = [
        ('ab',   sc_refs['ab'],   'ABMIL α\n(shared)',     'YlOrRd'),
        ('nbhd', sc_refs['nbhd'], 'Nbhd score\n(shared)',  'YlOrRd'),
        ('inst', sc_refs['inst'], 'Inst score\n(shared)',  'YlOrRd'),
        ('joint',sc_refs['joint'],'Joint\n(shared)',       'hot_r'),
    ]
    cbar_x0    = (LABEL_W/COL_W + 2) / sum(width_ratios)   # after tissue col
    cbar_width = 1 / sum(width_ratios) * 0.9
    gap        = cbar_x0
    for i, (key, sc, lbl, _) in enumerate(cbar_specs):
        if sc is None: continue
        cax = fig.add_axes([gap + i * cbar_width, 0.93,
                             cbar_width * 0.7, 0.012])
        cb  = fig.colorbar(sc, cax=cax, orientation='horizontal')
        cb.set_label(lbl, fontsize=5.5)
        cb.ax.tick_params(labelsize=4.5, labelbottom=False, labeltop=True)

    # ── Tissue legend ─────────────────────────────────────────────────────────
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=len(legend_handles), fontsize=6,
               frameon=False, bbox_to_anchor=(0.45, -0.01),
               columnspacing=0.7, handlelength=1.0)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Patient {pid}  |  {n_rows} biopsies  |  {n_pos} ACR+  |  Red border = ACR+\n"
        "Col 2: ABMIL v8 Phase-1 (flat α)  |  Cols 3-5: 2-level neighbourhood ABMIL  "
        "(shared scale = 99th pct across timepoints)",
        fontsize=7.5, y=0.99)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_pdf = OUT_DIR / f"{pid}.pdf"
    out_png = OUT_DIR / f"{pid}.png"
    fig.savefig(out_pdf, dpi=150, bbox_inches='tight')
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    n_done += 1
    print(f"  Saved {pid}.pdf  ({n_rows} rows, {n_pos} ACR+)", flush=True)

print(f"\n[done] {n_done} patient figures saved to {OUT_DIR}")
PYEOF
