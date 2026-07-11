#!/usr/bin/env bash
#SBATCH --job-name=sc_simplex
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/spatial_attn_combined/sc_simplex_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/spatial_attn_combined/sc_simplex_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

mkdir -p /lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/spatial_attn_combined
mkdir -p /lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/spatial_attn_combined/sc_simplex_per_patient

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
"""
Subcluster Simplex — SUBCLUSTER COMPOSITION BASED
==================================================

54 HE subclusters (subcluster_renamed from h5ad) as vertices on a circle,
grouped into 7 tissue-type sectors.

For each spatial neighbourhood c:
  - Match cluster patch coords to h5ad patches via KD-tree
  - w_s = fraction of patches in cluster c with subcluster_renamed == s
  - pos = sum_s  w_s * vertex_s

Colour:
  Fig A: na[c] — 2nd-level ABMIL raw attention logit
  Fig B: ACR label + enrichment hexbin
  Fig C: hexbin mean na[c]
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
(OUT_DIR / "sc_simplex_per_patient").mkdir(exist_ok=True)

HIDDEN  = 256
DROPOUT = 0.4
IN_DIM  = 1024
KNN_MAX_DIST = 256

TYPE_COLORS = {
    "Alveolar":                                  "#E53030",
    "Bronchial":                                 "#1A72CC",
    "Cartilage":                                 "#F5C800",
    "Alveolar with empty spaces":                "#00BB44",
    "Alveolar with hemorrhage and inflammation": "#FF6F00",
    "Lymphocytoplasmic inflammation":            "#CC00CC",
    "Unknown":                                   "#888888",
}
TYPE_SHORT = {
    "Alveolar":                                  "Alveolar",
    "Bronchial":                                 "Bronchial",
    "Cartilage":                                 "Cartilage",
    "Alveolar with empty spaces":                "Alv. empty",
    "Alveolar with hemorrhage and inflammation": "Alv. hem+inf",
    "Lymphocytoplasmic inflammation":            "Lympho.",
    "Unknown":                                   "Unknown",
}
ACR_COLORS = {0: "#1E88E5", 1: "#E53935"}


# ── 2-level ABMIL (raw logits) ─────────────────────────────────────────────────
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
            pa_raws.append(F.softmax(pa_raw.reshape(1, -1), dim=1).reshape(-1))  # (Nc,) softmax weights
        H   = self.nbhd_proj(torch.stack(reps, dim=1))
        _, na_raw = self.nbhd_attn(H)
        return na_raw.reshape(-1), pa_raws   # (C,), list[(Nc,)]


# ── Load 2-level ABMIL ─────────────────────────────────────────────────────────
print("[models] Loading ...", flush=True)
abmil2: dict = {}
for fold in range(5):
    ckpt = ABMIL2_DIR / f"abmil_fold{fold}.pt"
    if not ckpt.exists(): continue
    m = SpatialABMIL2Level()
    m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval()
    abmil2[fold] = m
print(f"[models] 2-level ABMIL folds: {sorted(abmil2)}", flush=True)


# ── Read h5ad: subcluster order + per-slide KD-trees ──────────────────────────
print("[h5ad] Reading obs ...", flush=True)
adata = ad.read_h5ad(H5AD, backed='r')
obs_raw = adata.obs[['slide_name_clean','chunk_left','chunk_top',
                      'subcluster_renamed','tissue_type']].copy()
adata.file.close()
obs_raw['chunk_left']        = obs_raw['chunk_left'].astype(float)
obs_raw['chunk_top']         = obs_raw['chunk_top'].astype(float)
obs_raw['subcluster_renamed'] = obs_raw['subcluster_renamed'].astype(str)
obs_raw['tissue_type']        = obs_raw['tissue_type'].astype(str)
obs_raw['slide_name_clean']   = obs_raw['slide_name_clean'].astype(str)

sc_to_type = (obs_raw[['subcluster_renamed','tissue_type']]
              .drop_duplicates('subcluster_renamed')
              .set_index('subcluster_renamed')['tissue_type']
              .to_dict())

# Build ordered subcluster list (grouped by tissue type, most populous first)
type_to_scs = defaultdict(list)
for sc, tt in sc_to_type.items():
    type_to_scs[tt].append(sc)
for tt in type_to_scs:
    type_to_scs[tt].sort(key=lambda s: (s.split('_')[0],
                                         int(s.split('_')[1]) if '_' in s and s.split('_')[1].isdigit() else 0))
TISSUE_ORDER = sorted(type_to_scs, key=lambda t: (-len(type_to_scs[t]), t))
ALL_SC  = [sc for tt in TISSUE_ORDER for sc in type_to_scs[tt]]
N_SC    = len(ALL_SC)
SC_IDX  = {sc: i for i, sc in enumerate(ALL_SC)}

print(f"[h5ad] {N_SC} subclusters:", flush=True)
for tt in TISSUE_ORDER:
    print(f"  {TYPE_SHORT.get(tt,tt):20s} ({len(type_to_scs[tt]):2d}): {type_to_scs[tt]}", flush=True)

# Circle layout
GAP_DEG   = 8.0
remain    = 360.0 - GAP_DEG * len(TISSUE_ORDER)
sc_deg    = remain / N_SC
SC_ANGLES = np.zeros(N_SC)
SC_COLORS = []
TYPE_ARC  = []
angle = -90.0
for tt in TISSUE_ORDER:
    scs = type_to_scs[tt]
    a_start = angle
    for sc in scs:
        SC_ANGLES[SC_IDX[sc]] = np.radians(angle + sc_deg/2)
        SC_COLORS.append(TYPE_COLORS.get(tt, "#888888"))
        angle += sc_deg
    TYPE_ARC.append((a_start, angle, (a_start+angle)/2, tt))
    angle += GAP_DEG
SC_VERTICES = np.stack([np.cos(SC_ANGLES), np.sin(SC_ANGLES)], axis=1)  # (N_SC, 2)

# Per-slide KD-trees
print("[h5ad] Building per-slide KD-trees ...", flush=True)
slide_data: dict = {}   # slide → (tree, subcluster_array)
for sn, grp in obs_raw.groupby('slide_name_clean'):
    coords  = grp[['chunk_left','chunk_top']].values.astype(np.float32)
    subcls  = grp['subcluster_renamed'].values
    slide_data[str(sn)] = (cKDTree(coords), subcls)
print(f"[h5ad] {len(slide_data)} slides", flush=True)


def attn_weighted_composition(cluster_coords: np.ndarray, pa_weights: np.ndarray,
                               slide_name: str):
    """
    Attention-weighted subcluster composition.
    pa_weights: (Nc,) softmax patch attention weights within this neighbourhood.
    Returns w (N_SC,) summing to 1 — position on the simplex is driven by which
    subclusters the model actually attends to, not raw patch counts.
    """
    if slide_name not in slide_data:
        return np.ones(N_SC) / N_SC
    tree, subcls = slide_data[slide_name]
    coords2d = np.atleast_2d(cluster_coords)
    dists, idxs = tree.query(coords2d, k=1, workers=-1)
    dists = np.atleast_1d(dists); idxs = np.atleast_1d(idxs)
    pa    = np.atleast_1d(pa_weights)
    valid = dists < KNN_MAX_DIST
    if valid.sum() == 0:
        return np.ones(N_SC) / N_SC
    matched_sc  = subcls[idxs[valid]]
    matched_w   = pa[valid]
    matched_w   = matched_w / (matched_w.sum() + 1e-12)   # re-normalise after filtering
    w = np.zeros(N_SC)
    for sc, wt in zip(matched_sc, matched_w):
        if sc in SC_IDX:
            w[SC_IDX[sc]] += wt
    s = w.sum()
    return w / s if s > 0 else np.ones(N_SC) / N_SC


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


# ── Collect records ────────────────────────────────────────────────────────────
print("\n[main] Collecting records ...", flush=True)
records = []
feat_files = {p.stem: p for p in FEAT_DIR.glob("*.pt")}

for slide_name, fp in sorted(feat_files.items()):
    pid = slide_name.split('-')[0]
    pat_stems = [s for s, m in stem_meta.items() if m["patient_id"]==pid and s in stem_to_fold]
    if not pat_stems: continue
    stem      = pat_stems[0]
    fold      = stem_to_fold[stem]
    label     = stem_meta[stem]["label"]
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
        pos = w @ SC_VERTICES
        dom_sc = ALL_SC[int(np.argmax(w))]
        records.append({
            "slide_name": slide_name, "patient_id": pid,
            "label": label, "anchor_dt": anchor_dt,
            "cluster_i": ci, "na": float(na[ci]),
            "pos_x": float(pos[0]), "pos_y": float(pos[1]),
            "dom_sc": dom_sc, "dom_tt": sc_to_type.get(dom_sc,"Unknown"),
        })
    if len(records) % 5000 == 0 and len(records) > 0:
        print(f"  {len(records)} records", flush=True)

df = pd.DataFrame(records)
df.to_csv(OUT_DIR/"sc_subcluster_records.csv", index=False)
print(f"[main] {len(df)} records from {df['slide_name'].nunique()} slides", flush=True)
if df.empty: sys.exit("[error] No records")

def _bag_rep(g):
    na = g["na"].values
    w  = np.exp(na - na.max()); w /= w.sum()
    return pd.Series({"bag_x": (w*g["pos_x"].values).sum(),
                       "bag_y": (w*g["pos_y"].values).sum()})

slide_rep = df.groupby(["slide_name","patient_id","label","anchor_dt"])\
              .apply(_bag_rep).reset_index()

na_vmax = float(np.percentile(df["na"], 99))
na_vmin = float(np.percentile(df["na"],  1))
if na_vmax <= na_vmin: na_vmax = na_vmin + 1e-4

OUTER_R = 1.05; ARC_R = 1.18; LABEL_R = 1.32; GROUP_R = 1.50; LIM = 1.65


def draw_sc_polygon(ax, show_labels=True):
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
    theta = np.linspace(0, 2*np.pi, 400)
    for r in [0.33, 0.67, 1.0]:
        ax.plot(r*np.cos(theta), r*np.sin(theta), color="#EEEEEE", lw=0.5, zorder=1)
    for (a_start, a_end, a_mid, tt) in TYPE_ARC:
        col   = TYPE_COLORS.get(tt, "#888888")
        short = TYPE_SHORT.get(tt, tt)
        arc_t = np.linspace(np.radians(a_start), np.radians(a_end), 60)
        ax.plot(ARC_R*np.cos(arc_t), ARC_R*np.sin(arc_t),
                color=col, lw=4, solid_capstyle="round", alpha=0.8, zorder=2)
        ang = np.radians(a_mid)
        ax.text(GROUP_R*np.cos(ang), GROUP_R*np.sin(ang), short,
                fontsize=6.5, ha="center", va="center", color=col,
                fontweight="bold",
                rotation=np.degrees(ang) if abs(np.degrees(ang))<90 else np.degrees(ang)+180,
                rotation_mode="anchor", zorder=5)
    for i, sc in enumerate(ALL_SC):
        ang = SC_ANGLES[i]
        col = TYPE_COLORS.get(sc_to_type.get(sc,"Unknown"), "#888888")
        ax.scatter([OUTER_R*np.cos(ang)], [OUTER_R*np.sin(ang)],
                   c=[col], s=12, zorder=6, linewidths=0)
        if show_labels:
            rot = np.degrees(ang)
            if np.cos(ang) < -0.1: rot += 180
            ax.text(LABEL_R*np.cos(ang), LABEL_R*np.sin(ang), sc,
                    fontsize=4.0, ha="center", va="center", color=col,
                    rotation=rot, rotation_mode="anchor", zorder=5)
    for i in range(0, N_SC, max(1, N_SC//20)):
        ang = SC_ANGLES[i]
        ax.plot([0, np.cos(ang)], [0, np.sin(ang)], color="#F0F0F0", lw=0.4, zorder=1)
    ax.scatter([0],[0], c=["#CCCCCC"], s=20, zorder=3, linewidths=0)


def _save(fig, name):
    fig.savefig(OUT_DIR/f"{name}.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR/f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {name}.pdf")


def _cbar(fig, ax, vmin, vmax, cmap, lbl):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, shrink=0.65)
    cb.set_label(lbl, fontsize=7); cb.ax.tick_params(labelsize=6)


# ── Figure A — na[c] ──────────────────────────────────────────────────────────
print("\n[figA] na[c] coloured simplex ...", flush=True)
figA, axs = plt.subplots(1, 3, figsize=(21, 8))
for ax, (title, mask) in zip(axs, [
    ("All",  slice(None)),
    ("ACR−", df["label"]==0),
    ("ACR+", df["label"]==1),
]):
    sub = df.loc[mask] if not isinstance(mask, slice) else df
    draw_sc_polygon(ax)
    for lbl, alpha, z in [(0, 0.12, 2),(1, 0.45, 3)]:
        g = sub[sub["label"]==lbl]
        if g.empty: continue
        order = np.argsort(g["na"].values)
        ax.scatter(g["pos_x"].values[order], g["pos_y"].values[order],
                   c=g["na"].values[order], cmap="Reds",
                   vmin=na_vmin, vmax=na_vmax,
                   s=2.5, alpha=alpha, linewidths=0, rasterized=True, zorder=z)
    for lbl, mk, sz in [(0,"o",50),(1,"^",70)]:
        sr = slide_rep[slide_rep["label"]==lbl]
        ax.scatter(sr["bag_x"], sr["bag_y"], c=ACR_COLORS[lbl], marker=mk,
                   s=sz, alpha=0.88, linewidths=0.8, edgecolors="white", zorder=8)
    _cbar(figA, ax, na_vmin, na_vmax, "Reds", "na[c] logit")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)

figA.legend(handles=[
    mpatches.Patch(color=ACR_COLORS[0], label="ACR− bag rep ●"),
    mpatches.Patch(color=ACR_COLORS[1], label="ACR+ bag rep ▲"),
], fontsize=7, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5,-0.02))
figA.suptitle(
    "Subcluster Simplex — subcluster composition projection\n"
    f"54 HE subclusters as vertices, grouped by tissue type sectors  |  "
    "Colour = 2nd-level ABMIL attention logit na[c]",
    fontsize=8, y=1.01)
plt.tight_layout()
_save(figA, "sc_simplex_attn")


# ── Figure B — ACR label + enrichment ─────────────────────────────────────────
print("[figB] ACR label ...", flush=True)
figB, (axB1, axB2) = plt.subplots(1, 2, figsize=(16, 8))

draw_sc_polygon(axB1)
na_norm = (df["na"].values - na_vmin) / (na_vmax - na_vmin + 1e-8)
s_vals  = 1.5 + np.clip(na_norm, 0, 1) * 8
for lbl in [0,1]:
    m = df["label"]==lbl
    axB1.scatter(df.loc[m,"pos_x"], df.loc[m,"pos_y"],
                 c=ACR_COLORS[lbl], s=s_vals[m.values],
                 alpha=0.15 if lbl==0 else 0.45,
                 linewidths=0, rasterized=True, zorder=2+lbl,
                 label=f"ACR{'−' if lbl==0 else '+'}")
for lbl, mk, sz in [(0,"o",50),(1,"^",70)]:
    sr = slide_rep[slide_rep["label"]==lbl]
    axB1.scatter(sr["bag_x"], sr["bag_y"], c=ACR_COLORS[lbl], marker=mk,
                 s=sz, alpha=0.88, linewidths=0.8, edgecolors="white", zorder=8)
axB1.legend(fontsize=8, loc="upper right", framealpha=0.85, markerscale=2.5)
axB1.set_title("ACR label  |  size ∝ |na[c]|", fontsize=8, fontweight="bold")

draw_sc_polygon(axB2)
_tmp_fig, _tmp_ax = plt.subplots(figsize=(1,1))
EXT  = [-1.1, 1.1, -1.1, 1.1]; GS = 30
n_pos = max((df["label"]==1).sum(), 1); n_neg = max((df["label"]==0).sum(), 1)
def _hb(ax_h, sub):
    hb = ax_h.hexbin(sub["pos_x"], sub["pos_y"], gridsize=GS, extent=EXT, mincnt=0, visible=False)
    return np.array(hb.get_offsets()), np.array(hb.get_array())
offs, cp = _hb(_tmp_ax, df[df["label"]==1])
_,   cn  = _hb(_tmp_ax, df[df["label"]==0])
plt.close(_tmp_fig)
enrich = cp/n_pos - cn/n_neg
eabs   = max(float(np.percentile(np.abs(enrich), 98)), 1e-6)
norm_e = mcolors.TwoSlopeNorm(vmin=-eabs, vcenter=0, vmax=eabs)
sm_e   = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm_e); sm_e.set_array([])
hr     = 2.2 / (GS * 2)
for (ox, oy), ev in zip(offs, enrich):
    if ox**2+oy**2 > 1.15**2: continue
    axB2.scatter([ox],[oy], c=[sm_e.to_rgba(ev)], s=hr*2800, marker="h",
                 linewidths=0, alpha=0.82, zorder=3)
cb_e = figB.colorbar(sm_e, ax=axB2, fraction=0.035, pad=0.02, shrink=0.65)
cb_e.set_label("ACR+ density − ACR−\n(red = enriched in ACR+)", fontsize=7)
cb_e.ax.tick_params(labelsize=6)
axB2.set_title("Enrichment: which subclusters over-represented in ACR+?",
               fontsize=8, fontweight="bold")
figB.suptitle(f"Subcluster Simplex — ACR label perspective", fontsize=8, y=1.01)
plt.tight_layout()
_save(figB, "sc_simplex_acr")


# ── Figure C — hexbin mean na[c] ──────────────────────────────────────────────
print("[figC] Hexbin ...", flush=True)
figC, axsC = plt.subplots(1, 2, figsize=(14, 7))
for ax, (lbl, title) in zip(axsC, [(0,"ACR−"),(1,"ACR+")]):
    draw_sc_polygon(ax)
    sub = df[df["label"]==lbl]
    if len(sub) >= 5:
        hb = ax.hexbin(sub["pos_x"], sub["pos_y"], C=sub["na"],
                       gridsize=28, mincnt=3, extent=[-1.1,1.1,-1.1,1.1],
                       reduce_C_function=np.mean, cmap="Reds",
                       vmin=na_vmin, vmax=na_vmax, alpha=0.85, linewidths=0.15, zorder=4)
        figC.colorbar(hb, ax=ax, fraction=0.035, pad=0.02,
                      shrink=0.65, label="mean na[c]").ax.tick_params(labelsize=6)
    sr = slide_rep[slide_rep["label"]==lbl]
    mk = "^" if lbl==1 else "o"
    ax.scatter(sr["bag_x"], sr["bag_y"], c=ACR_COLORS[lbl], marker=mk,
               s=65, alpha=0.85, linewidths=0.8, edgecolors="white", zorder=8)
    ax.set_title(f"Mean na[c] — {title}", fontsize=9, fontweight="bold")
figC.suptitle("Hexbin: mean attention per subcluster-simplex region", fontsize=8, y=1.01)
plt.tight_layout()
_save(figC, "sc_simplex_hexbin")


# ── Figure D — per-patient ─────────────────────────────────────────────────────
print("[figD] Per-patient ...", flush=True)
acr_pids = sorted(df.loc[df["label"]==1,"patient_id"].unique())
for pid in acr_pids:
    sub_pid = df[df["patient_id"]==pid]
    if sub_pid.empty: continue
    n_sl  = sub_pid["slide_name"].nunique()
    figP, axP = plt.subplots(figsize=(9, 9))
    draw_sc_polygon(axP)
    na_vmax_p = float(np.percentile(sub_pid["na"], 99))
    na_vmin_p = float(np.percentile(sub_pid["na"],  1))
    if na_vmax_p <= na_vmin_p: na_vmax_p = na_vmin_p + 1e-4

    # All neighbourhoods — ACR+ brighter, ACR- faint; all circles, colored by na[c]
    for lbl, alpha in [(0, 0.12), (1, 0.55)]:
        sub_s = sub_pid[sub_pid["label"]==lbl]
        if sub_s.empty: continue
        order = np.argsort(sub_s["na"].values)
        axP.scatter(sub_s["pos_x"].values[order], sub_s["pos_y"].values[order],
                    c=sub_s["na"].values[order], cmap="Reds",
                    vmin=na_vmin_p, vmax=na_vmax_p,
                    s=4, alpha=alpha, linewidths=0, rasterized=True, zorder=2+lbl)

    # Slide bag reps — triangle=ACR+, circle=ACR-
    for lbl in [0, 1]:
        sr_s = slide_rep[(slide_rep["patient_id"]==pid) & (slide_rep["label"]==lbl)]
        if sr_s.empty: continue
        mk = "^" if lbl==1 else "o"
        axP.scatter(sr_s["bag_x"], sr_s["bag_y"],
                    c=ACR_COLORS[lbl], marker=mk,
                    s=160, alpha=0.92, linewidths=1.0, edgecolors="white", zorder=9)

    sm_p = plt.cm.ScalarMappable(cmap="Reds",
                                   norm=mcolors.Normalize(na_vmin_p, na_vmax_p))
    sm_p.set_array([])
    figP.colorbar(sm_p, ax=axP, fraction=0.035, pad=0.02,
                  shrink=0.55, label="na[c] logit (higher = more red)").ax.tick_params(labelsize=6)
    # ACR label legend
    axP.legend(handles=[
        mpatches.Patch(color=ACR_COLORS[0], label="ACR− bag rep ●"),
        mpatches.Patch(color=ACR_COLORS[1], label="ACR+ bag rep ▲"),
    ], fontsize=7, loc="lower right", framealpha=0.85)
    n_pos = int((sub_pid["label"]==1).sum())
    axP.set_title(
        f"Patient {pid}  |  {n_sl} H&E slides  |  {n_pos} ACR+ neighbourhoods\n"
        "▲=ACR+ bag rep  ●=ACR− bag rep  |  colour = na[c] logit (higher = more red)",
        fontsize=8, pad=6)
    figP.savefig(OUT_DIR/"sc_simplex_per_patient"/f"{pid}.pdf", dpi=180, bbox_inches="tight")
    figP.savefig(OUT_DIR/"sc_simplex_per_patient"/f"{pid}.png", dpi=110, bbox_inches="tight")
    plt.close(figP)
print(f"  → sc_simplex_per_patient/ ({len(acr_pids)} patients)")


print(f"\n[done] All figures → {OUT_DIR}")
PYEOF
