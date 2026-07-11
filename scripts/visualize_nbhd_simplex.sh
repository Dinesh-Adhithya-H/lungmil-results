#!/usr/bin/env bash
#SBATCH --job-name=nbhd_simplex
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/spatial_attn_combined/simplex_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/spatial_attn_combined/simplex_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

mkdir -p /lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/spatial_attn_combined

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
"""
Neighbourhood Simplex — TISSUE COMPOSITION BASED
=================================================

Each spatial neighbourhood (cluster c in a slide) is projected onto a
heptagon whose 7 vertices = tissue types.

Projection:
  - Match cluster patch coords to h5ad chunk_left/chunk_top via KD-tree
  - w_t = fraction of patches in cluster c with tissue_type t
  - pos = sum_t  w_t * vertex_t

This gives naturally spread positions since tissue composition varies
across neighbourhoods and between ACR+/ACR- biopsies.

Colour:
  Fig A: na[c]      — 2nd-level ABMIL attention (raw logit)
  Fig B: ACR label  — ACR+ red, ACR- blue  (size proportional to |na[c]|)
  Fig C: hexbin density  ACR-/ACR+ side by side
  Fig D: per-patient longitudinal
"""

import sys, warnings
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
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
H5AD       = "/lustre/groups/aih/dinesh.haridoss/datasets/adata_v3.h5ad"
FEAT_DIR   = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/slide_cluster_feats")
ABMIL2_DIR = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/soft_hier_mil")
SPLITS_CSV = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
OUT_DIR    = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/spatial_attn_combined")
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "simplex_per_patient").mkdir(exist_ok=True)

HIDDEN  = 256
DROPOUT = 0.4
IN_DIM  = 1024
KNN_MAX_DIST = 256   # max pixel distance for coord matching (patch size ≈ 224px)

TYPE_ORDER = [
    "Alveolar",
    "Bronchial",
    "Cartilage",
    "Alveolar with empty spaces",
    "Alveolar with hemorrhage and inflammation",
    "Lymphocytoplasmic inflammation",
    "Unknown",
]
TYPE_COLORS = {
    "Alveolar":                                  "#E53030",
    "Bronchial":                                 "#1A72CC",
    "Cartilage":                                 "#F5C800",
    "Alveolar with empty spaces":                "#00BB44",
    "Alveolar with hemorrhage and inflammation": "#FF6F00",
    "Lymphocytoplasmic inflammation":            "#CC00CC",
    "Unknown":                                   "#AAAAAA",
}
N_V     = len(TYPE_ORDER)
_angles = [2*np.pi*k/N_V - np.pi/2 for k in range(N_V)]
VERTEX  = np.array([(np.cos(a), np.sin(a)) for a in _angles])   # (7, 2)
TYPE_IDX = {t: i for i, t in enumerate(TYPE_ORDER)}

ACR_COLORS  = {0: "#1E88E5", 1: "#E53935"}
ACR_LABELS  = {0: "ACR−", 1: "ACR+"}
ACR_MARKERS = {0: "o", 1: "^"}


# ── 2-level ABMIL (returns raw attention logits) ───────────────────────────────
class GatedAttentionPool(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(in_dim, hidden), nn.Sigmoid())
        self.w = nn.Linear(hidden, 1, bias=False)
    def forward(self, h):
        raw = self.w(self.V(h) * self.U(h))          # raw logits
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
            pa_raws.append(F.softmax(pa_raw.reshape(1, -1), dim=1).reshape(-1))  # (Nc,) softmax
        H = self.nbhd_proj(torch.stack(reps, dim=1))
        slide_rep, na_raw = self.nbhd_attn(H)
        return na_raw.reshape(-1), pa_raws   # (C,), list[(Nc,)]


# ── Load 2-level ABMIL checkpoints ────────────────────────────────────────────
print("[models] Loading ...", flush=True)
abmil2: dict = {}
for fold in range(5):
    ckpt = ABMIL2_DIR / f"abmil_fold{fold}.pt"
    if not ckpt.exists(): continue
    m = SpatialABMIL2Level()
    m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval()
    abmil2[fold] = m
print(f"[models] 2-level ABMIL folds: {sorted(abmil2)}", flush=True)


# ── Load splits CSV ────────────────────────────────────────────────────────────
df_sp = pd.read_csv(SPLITS_CSV)
fold_cols = [c for c in df_sp.columns if c.startswith("fold_")][:4]
stem_to_fold: dict = {}
for fi, col in enumerate(fold_cols):
    for s in df_sp.loc[df_sp[col]=="test","file"].values:
        stem_to_fold[str(s)] = fi
stem_meta: dict = {}
for _, row in df_sp.iterrows():
    stem_meta[str(row["file"])] = {
        "patient_id": str(row["patient_id"]),
        "anchor_dt":  str(row["anchor_dt"]),
        "label":      int(row["label"]) if pd.notna(row["label"]) else 0,
    }
print(f"[splits] {len(stem_to_fold)} test stems", flush=True)


# ── Load h5ad: slide → patch coords + tissue_type ─────────────────────────────
print("[h5ad] Reading obs ...", flush=True)
adata = ad.read_h5ad(H5AD, backed='r')
obs   = adata.obs[['slide_name_clean','chunk_left','chunk_top','tissue_type',
                    'record_id','biopsy_date','acr_binary']].copy()
adata.file.close()
obs['chunk_left'] = obs['chunk_left'].astype(float)
obs['chunk_top']  = obs['chunk_top'].astype(float)
obs['tissue_type'] = obs['tissue_type'].astype(str)
obs['slide_name_clean'] = obs['slide_name_clean'].astype(str)

# Per-slide KD-tree for fast coord matching
print("[h5ad] Building per-slide KD-trees ...", flush=True)
slide_data: dict = {}   # slide_name → (tree, tissue_type_array)
for sn, grp in obs.groupby('slide_name_clean'):
    coords = grp[['chunk_left','chunk_top']].values.astype(np.float32)
    types  = grp['tissue_type'].values
    slide_data[str(sn)] = (cKDTree(coords), types)
print(f"[h5ad] {len(slide_data)} slides with KD-trees", flush=True)


def attn_weighted_composition(cluster_coords: np.ndarray, pa_weights: np.ndarray,
                               slide_name: str):
    """
    Attention-weighted tissue composition.
    pa_weights: (Nc,) softmax patch attention weights within this neighbourhood.
    Returns w (N_V,) summing to 1 — high-attention patches drive the position.
    """
    if slide_name not in slide_data:
        return np.ones(N_V) / N_V
    tree, types = slide_data[slide_name]
    coords2d = np.atleast_2d(cluster_coords)
    dists, idxs = tree.query(coords2d, k=1, workers=-1)
    dists = np.atleast_1d(dists); idxs = np.atleast_1d(idxs)
    pa    = np.atleast_1d(pa_weights)
    valid = dists < KNN_MAX_DIST
    if valid.sum() == 0:
        return np.ones(N_V) / N_V
    matched_t  = types[idxs[valid]]
    matched_w  = pa[valid]
    matched_w  = matched_w / (matched_w.sum() + 1e-12)
    w = np.zeros(N_V)
    for t, wt in zip(matched_t, matched_w):
        if t in TYPE_IDX:
            w[TYPE_IDX[t]] += wt
    s = w.sum()
    return w / s if s > 0 else np.ones(N_V) / N_V


# ── Collect neighbourhood records ─────────────────────────────────────────────
print("\n[main] Collecting neighbourhood records ...", flush=True)
records = []
feat_files = {p.stem: p for p in FEAT_DIR.glob("*.pt")}

for slide_name, fp in sorted(feat_files.items()):
    # Find a test stem for this patient (for fold/label)
    pid = slide_name.split('-')[0]   # e.g. "LT001"
    # Find matching test stems for this patient
    pat_stems = [s for s, m in stem_meta.items() if m["patient_id"] == pid
                 and s in stem_to_fold]
    if not pat_stems: continue
    # Use the fold/label from the first matching test stem
    stem = pat_stems[0]
    fold  = stem_to_fold[stem]
    label = stem_meta[stem]["label"]
    anchor_dt = stem_meta[stem]["anchor_dt"]

    m2 = abmil2.get(fold, abmil2.get(0))
    if m2 is None: continue

    try:
        fd = torch.load(fp, map_location="cpu", weights_only=False)
    except: continue
    clusters_raw = fd.get("clusters", [])
    if not clusters_raw: continue

    clusters_t, coords_list = [], []
    for c in clusters_raw:
        f  = c["feats"].float()
        co = c["coords"].float()
        if f.shape[0] > 512:
            idx = torch.randperm(f.shape[0])[:512]
            f = f[idx]; co = co[idx]
        clusters_t.append(f); coords_list.append(co.numpy())

    with torch.no_grad():
        try:
            na_t, pa_raws = m2(clusters_t)
            na = na_t.numpy()   # (C,) raw neighbourhood logits
        except Exception as e:
            print(f"  [warn] {slide_name}: {e}"); continue

    for ci, (co, pa_t) in enumerate(zip(coords_list, pa_raws)):
        pa_np = pa_t.numpy()   # (Nc,) softmax patch attention weights
        w   = attn_weighted_composition(co, pa_np, slide_name)
        pos = w @ VERTEX   # (2,)
        dom = TYPE_ORDER[int(np.argmax(w))]

        records.append({
            "slide_name": slide_name,
            "patient_id": pid,
            "label":      label,
            "anchor_dt":  anchor_dt,
            "cluster_i":  ci,
            "na":         float(na[ci]),
            "pos_x":      float(pos[0]),
            "pos_y":      float(pos[1]),
            "dom_type":   dom,
            **{f"w_{t}": float(w[i]) for i, t in enumerate(TYPE_ORDER)},
        })
    if len(records) % 5000 == 0 and len(records) > 0:
        print(f"  {len(records)} records", flush=True)

df = pd.DataFrame(records)
df.to_csv(OUT_DIR / "nbhd_records.csv", index=False)
print(f"[main] {len(df)} neighbourhood records from {df['slide_name'].nunique()} slides",
      flush=True)
if df.empty: sys.exit("[error] No records")

# Slide-level bag rep: na-weighted mean position (use softmax of na for weighting)
def _bag_rep(g):
    na = g["na"].values
    w  = np.exp(na - na.max()); w /= w.sum()   # softmax of logits
    return pd.Series({
        "bag_x": (w * g["pos_x"].values).sum(),
        "bag_y": (w * g["pos_y"].values).sum(),
    })

slide_rep = (df.groupby(["slide_name","patient_id","label","anchor_dt"])
               .apply(_bag_rep).reset_index())

na_vmax = float(np.percentile(df["na"], 99))
na_vmin = float(np.percentile(df["na"],  1))
if na_vmax <= na_vmin: na_vmax = na_vmin + 1e-4


# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_polygon(ax, label_offset=0.20):
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.55, 1.55)
    poly = np.vstack([VERTEX, VERTEX[0]])
    ax.plot(poly[:,0], poly[:,1], color="#CCCCCC", lw=1.0, zorder=1)
    theta = np.linspace(0, 2*np.pi, 200)
    for r in [0.33, 0.67]:
        ax.plot(r*np.cos(theta), r*np.sin(theta), color="#EEEEEE", lw=0.4, zorder=1)
    for vx, vy in VERTEX:
        ax.plot([0, vx], [0, vy], color="#E8E8E8", lw=0.5, zorder=1)
    for i, t in enumerate(TYPE_ORDER):
        vx, vy = VERTEX[i]
        sc = 1.0 + label_offset
        short = (t.replace("Alveolar with ","Alv+")
                  .replace(" and inflammation","+inf")
                  .replace(" inflammation","inf"))
        ha = "center"
        if vx > 0.15: ha = "left"
        elif vx < -0.15: ha = "right"
        ax.text(vx*sc, vy*sc, short, fontsize=7, ha=ha, va="center",
                color=TYPE_COLORS[t], fontweight="bold", zorder=5)
    ax.scatter([0],[0], c="#AAAAAA", s=25, zorder=3, linewidths=0)


def _cbar(fig, ax, vmin, vmax, cmap, lbl):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.02, shrink=0.65)
    cb.set_label(lbl, fontsize=7); cb.ax.tick_params(labelsize=6)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure A — coloured by na[c] (raw neighbourhood attention logit)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[figA] Simplex by na[c] ...", flush=True)
figA, axs = plt.subplots(1, 3, figsize=(21, 8))
for ax, (title, mask) in zip(axs, [
    ("All",   slice(None)),
    ("ACR−",  df["label"]==0),
    ("ACR+",  df["label"]==1),
]):
    sub = df.loc[mask] if not isinstance(mask, slice) else df
    draw_polygon(ax)
    for lbl, alpha, z in [(0, 0.12, 2),(1, 0.45, 3)]:
        g = sub[sub["label"]==lbl]
        if g.empty: continue
        order = np.argsort(g["na"].values)
        ax.scatter(g["pos_x"].values[order], g["pos_y"].values[order],
                   c=g["na"].values[order], cmap="Reds",
                   vmin=na_vmin, vmax=na_vmax,
                   s=2.5, alpha=alpha, linewidths=0, rasterized=True, zorder=z)
    # bag reps
    for lbl, mk, sz in [(0,"o",55),(1,"^",75)]:
        sr = slide_rep[slide_rep["label"]==lbl]
        if sr.empty: continue
        ax.scatter(sr["bag_x"], sr["bag_y"], c=ACR_COLORS[lbl], marker=mk,
                   s=sz, alpha=0.88, linewidths=0.8, edgecolors="white", zorder=8)
    _cbar(figA, ax, na_vmin, na_vmax, "Reds", "na[c] logit")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)

figA.legend(handles=[
    mpatches.Patch(color=ACR_COLORS[0], label="ACR− bag rep ●"),
    mpatches.Patch(color=ACR_COLORS[1], label="ACR+ bag rep ▲"),
], fontsize=7, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5,-0.02))
figA.suptitle(
    "Neighbourhood Simplex — tissue composition projection\n"
    "Each point = spatial neighbourhood; position = tissue-type composition fraction  |  "
    f"Colour = 2nd-level ABMIL raw attention logit na[c]",
    fontsize=8, y=1.01)
plt.tight_layout()
figA.savefig(OUT_DIR/"simplex_all.pdf", dpi=200, bbox_inches="tight")
figA.savefig(OUT_DIR/"simplex_all.png", dpi=130, bbox_inches="tight")
plt.close(figA)
print("  → simplex_all.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure B — hexbin mean na[c], ACR- vs ACR+
# ═══════════════════════════════════════════════════════════════════════════════
print("[figB] Hexbin mean na[c] ...", flush=True)
figB, axsB = plt.subplots(1, 2, figsize=(14, 7))
for ax, (lbl, title) in zip(axsB, [(0,"ACR−"),(1,"ACR+")]):
    draw_polygon(ax)
    sub = df[df["label"]==lbl]
    if len(sub) >= 5:
        hb = ax.hexbin(sub["pos_x"], sub["pos_y"], C=sub["na"],
                       gridsize=25, mincnt=3, extent=[-1.1,1.1,-1.1,1.1],
                       reduce_C_function=np.mean, cmap="Reds",
                       vmin=na_vmin, vmax=na_vmax, alpha=0.88, linewidths=0.12, zorder=4)
        figB.colorbar(hb, ax=ax, fraction=0.038, pad=0.02,
                      shrink=0.65, label="mean na[c]").ax.tick_params(labelsize=6)
    sr = slide_rep[slide_rep["label"]==lbl]
    mk = "^" if lbl==1 else "o"
    ax.scatter(sr["bag_x"], sr["bag_y"], c=ACR_COLORS[lbl], marker=mk,
               s=70, alpha=0.85, linewidths=0.8, edgecolors="white", zorder=8)
    ax.set_title(f"Mean na[c] — {title}", fontsize=9, fontweight="bold")

figB.suptitle(
    "Hexbin: mean neighbourhood attention logit per tissue-composition region\n"
    "Hot regions = neighbourhood types the model attends to in ACR+ vs ACR−",
    fontsize=8, y=1.01)
plt.tight_layout()
figB.savefig(OUT_DIR/"simplex_hexbin_na.pdf", dpi=200, bbox_inches="tight")
figB.savefig(OUT_DIR/"simplex_hexbin_na.png", dpi=130, bbox_inches="tight")
plt.close(figB)
print("  → simplex_hexbin_na.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure C — ACR label + enrichment hexbin
# ═══════════════════════════════════════════════════════════════════════════════
print("[figC] ACR label + enrichment ...", flush=True)
figC, (axC1, axC2) = plt.subplots(1, 2, figsize=(16, 8))

draw_polygon(axC1)
na_norm = (df["na"].values - na_vmin) / (na_vmax - na_vmin + 1e-8)
s_vals  = 1.5 + np.clip(na_norm, 0, 1) * 8
for lbl in [0,1]:
    m = df["label"]==lbl
    axC1.scatter(df.loc[m,"pos_x"], df.loc[m,"pos_y"],
                 c=ACR_COLORS[lbl], s=s_vals[m.values],
                 alpha=0.15 if lbl==0 else 0.45,
                 linewidths=0, rasterized=True, zorder=2+lbl,
                 label=ACR_LABELS[lbl])
for lbl, mk, sz in [(0,"o",55),(1,"^",75)]:
    sr = slide_rep[slide_rep["label"]==lbl]
    axC1.scatter(sr["bag_x"], sr["bag_y"], c=ACR_COLORS[lbl], marker=mk,
                 s=sz, alpha=0.88, linewidths=0.8, edgecolors="white", zorder=8)
axC1.legend(fontsize=8, loc="upper right", framealpha=0.85, markerscale=2.5)
axC1.set_title("ACR label  |  size ∝ |na[c]|", fontsize=8, fontweight="bold")

# Enrichment hexbin
draw_polygon(axC2)
_tmp_fig, _tmp_ax = plt.subplots(figsize=(1,1))
EXT  = [-1.1, 1.1, -1.1, 1.1]
GS   = 28
n_pos = max((df["label"]==1).sum(), 1)
n_neg = max((df["label"]==0).sum(), 1)
def _hb(ax_h, sub):
    hb = ax_h.hexbin(sub["pos_x"], sub["pos_y"], gridsize=GS, extent=EXT, mincnt=0, visible=False)
    return np.array(hb.get_offsets()), np.array(hb.get_array())
offs, cp = _hb(_tmp_ax, df[df["label"]==1])
_,   cn  = _hb(_tmp_ax, df[df["label"]==0])
plt.close(_tmp_fig)
enrich = cp/n_pos - cn/n_neg
eabs   = float(np.percentile(np.abs(enrich), 98)); eabs = max(eabs, 1e-6)
norm_e = mcolors.TwoSlopeNorm(vmin=-eabs, vcenter=0, vmax=eabs)
sm_e   = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm_e); sm_e.set_array([])
hr     = 2.2 / (GS * 2)
for (ox, oy), ev in zip(offs, enrich):
    if ox**2+oy**2 > 1.15**2: continue
    axC2.scatter([ox],[oy], c=[sm_e.to_rgba(ev)], s=hr*2800, marker="h",
                 linewidths=0, alpha=0.82, zorder=3)
cb_e = figC.colorbar(sm_e, ax=axC2, fraction=0.038, pad=0.02, shrink=0.65)
cb_e.set_label("ACR+ density − ACR−\n(red = enriched in ACR+)", fontsize=7)
cb_e.ax.tick_params(labelsize=6)
axC2.set_title("Enrichment: ACR+ vs ACR− neighbourhood distribution",
               fontsize=8, fontweight="bold")

figC.suptitle(
    "Neighbourhood Simplex — ACR label perspective  |  "
    "Tissue-composition simplex projection\n"
    "Enriched regions show tissue types over-represented in ACR+ neighbourhoods",
    fontsize=8, y=1.01)
plt.tight_layout()
figC.savefig(OUT_DIR/"simplex_acr_label.pdf", dpi=200, bbox_inches="tight")
figC.savefig(OUT_DIR/"simplex_acr_label.png", dpi=130, bbox_inches="tight")
plt.close(figC)
print("  → simplex_acr_label.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure D — composition stacked bar: dominant type per attention quartile
# ═══════════════════════════════════════════════════════════════════════════════
print("[figD] Composition by attention quartile ...", flush=True)
df["na_q"] = pd.qcut(df["na"], 4, labels=["Q1\nlow","Q2","Q3","Q4\nhigh"])
w_cols = [f"w_{t}" for t in TYPE_ORDER]

figD, axes4 = plt.subplots(1, 2, figsize=(14, 6))
for ax, (acr_val, acr_lbl) in zip(axes4, [(0,"ACR−"),(1,"ACR+")]):
    sub = df[df["label"]==acr_val]
    if sub.empty: ax.set_visible(False); continue
    q_groups = sub.groupby("na_q", observed=False)[w_cols].mean()
    bot = np.zeros(len(q_groups))
    for t, wcol in zip(TYPE_ORDER, w_cols):
        vals = q_groups[wcol].values
        ax.bar(range(len(q_groups)), vals, bottom=bot,
               color=TYPE_COLORS[t], label=t, edgecolor="none")
        bot += vals
    ax.set_xticks(range(len(q_groups)))
    ax.set_xticklabels(q_groups.index, fontsize=8)
    ax.set_ylabel("Mean tissue composition weight", fontsize=8)
    ax.set_title(f"{acr_lbl} — tissue composition by attention quartile",
                 fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

axes4[0].legend(fontsize=6, loc="upper right", ncol=1, framealpha=0.8)
figD.suptitle(
    "Which tissue types are most attended? Q4 = top 25% na[c] neighbourhoods",
    fontsize=9, y=1.01)
plt.tight_layout()
figD.savefig(OUT_DIR/"simplex_composition_by_attn.pdf", dpi=180, bbox_inches="tight")
figD.savefig(OUT_DIR/"simplex_composition_by_attn.png", dpi=120, bbox_inches="tight")
plt.close(figD)
print("  → simplex_composition_by_attn.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure E — per-patient longitudinal simplex
# ═══════════════════════════════════════════════════════════════════════════════
print("[figE] Per-patient ...", flush=True)
acr_pids = sorted(df.loc[df["label"]==1, "patient_id"].unique())
for pid in acr_pids:
    sub_pid = df[df["patient_id"]==pid]
    if sub_pid.empty: continue
    slides_sorted = sorted(sub_pid["slide_name"].unique(),
                           key=lambda s: sub_pid.loc[sub_pid["slide_name"]==s,"anchor_dt"].values[0])
    n_sl  = len(slides_sorted)
    figP, axP = plt.subplots(figsize=(8, 8))
    draw_polygon(axP)
    na_vmax_p = float(np.percentile(sub_pid["na"], 99))
    na_vmin_p = float(np.percentile(sub_pid["na"],  1))
    if na_vmax_p <= na_vmin_p: na_vmax_p = na_vmin_p + 1e-4

    for sn in slides_sorted:
        sub_s = sub_pid[sub_pid["slide_name"]==sn]
        lbl   = sub_s["label"].iloc[0]
        axP.scatter(sub_s["pos_x"], sub_s["pos_y"],
                    c=sub_s["na"], cmap="Reds",
                    vmin=na_vmin_p, vmax=na_vmax_p,
                    s=3.5, alpha=0.55 if lbl==1 else 0.18,
                    linewidths=0, rasterized=True, zorder=2)
        sr_s = slide_rep[slide_rep["slide_name"]==sn]
        if not sr_s.empty:
            mk = "^" if lbl==1 else "o"
            axP.scatter(sr_s["bag_x"], sr_s["bag_y"],
                        c=[ACR_COLORS[lbl]], marker=mk,
                        s=130, alpha=0.9, linewidths=1.2, edgecolors="white", zorder=9)

    sm_p = plt.cm.ScalarMappable(cmap="Reds",
                                   norm=mcolors.Normalize(na_vmin_p, na_vmax_p))
    sm_p.set_array([])
    figP.colorbar(sm_p, ax=axP, fraction=0.038, pad=0.02,
                  shrink=0.55, label="na[c] logit (higher = more red)").ax.tick_params(labelsize=6)
    axP.legend(handles=[
        mpatches.Patch(color=ACR_COLORS[0], label="ACR− bag rep ●"),
        mpatches.Patch(color=ACR_COLORS[1], label="ACR+ bag rep ▲"),
    ], fontsize=7, loc="lower right", framealpha=0.85)
    n_pos = int((sub_pid["label"]==1).sum())
    axP.set_title(
        f"Patient {pid}  |  {n_sl} H&E slides  |  {n_pos} ACR+ neighbourhoods\n"
        "▲=ACR+ bag rep  ●=ACR− bag rep  |  colour = na[c] logit",
        fontsize=8, pad=6)
    figP.savefig(OUT_DIR/"simplex_per_patient"/f"{pid}.pdf", dpi=180, bbox_inches="tight")
    figP.savefig(OUT_DIR/"simplex_per_patient"/f"{pid}.png", dpi=110, bbox_inches="tight")
    plt.close(figP)
print(f"  → simplex_per_patient/ ({len(acr_pids)} patients)")


print(f"\n[done] All figures → {OUT_DIR}")
PYEOF
