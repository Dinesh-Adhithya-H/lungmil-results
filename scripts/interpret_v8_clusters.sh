#!/usr/bin/env bash
#SBATCH --job-name=interpret_v8
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/analysis/v8_full/interpret_v8_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/analysis/v8_full/interpret_v8_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
"""
V8 Multimodal Interpretability — Cluster-Level Attention Analysis
=================================================================

Figures produced
----------------
1. fig_umap_hexbin_{acr|clad|death}.pdf
   UMAP with cls_prob + hazard hexbin subplots (2 rows × 3 cols each endpoint).
   Uses pre-computed UMAP embeddings; adds hexbin panels for model outputs.

2. fig_cluster_attn_acr.pdf
   Heatmap: rows = coarse cluster types per modality, cols = ACR+/−.
   Color = mean Phase-1 ABMIL attention weight.

3. fig_cluster_attn_survival.pdf
   Heatmap: coarse cluster types × endpoint (CLAD/Death).
   Color = Spearman ρ between mean attention per slide and predicted hazard.

4. fig_cluster_discriminability.pdf
   Signed discriminability = mean_attn(ACR+) − mean_attn(ACR−), per cluster.
   Separate panels per modality.  Shows which patch/cell/region types the
   model relies on for each endpoint.

5. fig_cluster_violin_acr.pdf
   Violin plots of mean-attention distributions for top-10 discriminable
   coarse types per modality, split ACR+/−.

6. fig_modality_contribution_endpoint.pdf
   Mean total attention mass per modality (sum of alpha) split by ACR label
   and by quartile of CLAD/Death hazard.
"""

import sys, warnings, json, pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
V8_BASE    = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/results_mm_abmil_v8")
SAMPLES    = Path("/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples")
SPLITS_CSV = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
ANALYSIS   = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/analysis/v8_full")
OUT_DIR    = ANALYSIS / "cluster_interpretability_v8"
OUT_DIR.mkdir(exist_ok=True)

# Optional: CSV mapping HE leiden coarse cluster ID → tissue type name.
# Columns: leiden_coarse (str, e.g. "0") and tissue_type (str, e.g. "alveolar").
# If not found, falls back to using the raw coarse-ID labels.
HE_ANNOT_CSV = Path("/home/aih/dinesh.haridoss/chicago_mil/data_prep/he_leiden_coarse_annotation.csv")

sys.path.insert(0, "/home/aih/dinesh.haridoss/chicago_mil/src")
from mil.data.registry import MODALITIES, MODALITY_REGISTRY

# ── Constants ──────────────────────────────────────────────────────────────────
HIDDEN_DIM = 256
DROPOUT    = 0.4
FEAT_DIMS  = {m: MODALITY_REGISTRY[m][1] for m in MODALITIES}
FEAT_KEYS  = {m: MODALITY_REGISTRY[m][0] for m in MODALITIES}   # e.g. HE→HE_cells
N_SPLITS   = 5
N_FOLDS    = 4
DEVICE     = torch.device("cpu")

ENDPOINT_TASK = {
    "acr":   ("acr",   "both_alt", "middle", "cls_prob"),
    "clad":  ("clad",  "clad_surv","late",   "hazard"),
    "death": ("death", "death_surv","late",  "hazard"),
}

MOD_COLORS = {"HE": "#2196F3", "BAL": "#4CAF50", "CT": "#FF9800", "Clinical": "#9C27B0"}

# ── Load HE leiden coarse annotation ──────────────────────────────────────────
he_annot: dict = {}   # str(coarse_id) → tissue_type_str
if HE_ANNOT_CSV.exists():
    ann = pd.read_csv(HE_ANNOT_CSV)
    if "leiden_coarse" in ann.columns and "tissue_type" in ann.columns:
        he_annot = dict(zip(ann["leiden_coarse"].astype(str), ann["tissue_type"]))
        print(f"[annot] Loaded {len(he_annot)} HE leiden→tissue annotations")
    else:
        print(f"[annot] HE annotation CSV missing expected columns: {ann.columns.tolist()}")
else:
    print("[annot] HE_ANNOT_CSV not found — using raw cluster IDs as coarse type")


# ── Coarse-type extraction ─────────────────────────────────────────────────────
def coarse_label(mod: str, raw_label: str) -> str:
    """Map raw cluster label to interpretable coarse type name."""
    if mod == "HE":
        coarse_id = raw_label.split("_")[0]
        return he_annot.get(coarse_id, f"HE_{coarse_id}")
    if mod == "BAL":
        # BAL_cells: cell type names like "DC2", "NK", "CD8 T cells-5"
        # Group to coarse by stripping trailing numeric suffix
        import re
        base = re.sub(r"[-_]\d+$", "", raw_label).strip()
        return base if base else raw_label
    if mod == "CT":
        return f"CT_c{raw_label}"
    if mod == "Clinical":
        # "fvc_q0" → "fvc"
        return raw_label.split("_q")[0].split("_bin")[0]
    return raw_label


# ── Model definition ──────────────────────────────────────────────────────────
class GatedAttentionEncoder(nn.Module):
    def __init__(self, feat_dim=1024, hidden_dim=256, dropout=0.4):
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
        raw   = self.att_w(self.att_drop(gate))
        alpha = F.softmax(raw, dim=0).squeeze(1)   # (N,)
        return alpha, h


# ── Load splits → fold → test stems ───────────────────────────────────────────
df_splits = pd.read_csv(SPLITS_CSV)
stem_col  = "file"   # e.g. "00001.pt"

# Build: (split_idx, fold_idx) → test stem set
split_fold_test: dict = {}
for s in range(N_SPLITS):
    for f in range(N_FOLDS):
        col = f"split{s}_fold{f}"
        if col in df_splits.columns:
            mask = df_splits[col].values == "test"
            test_stems = set(df_splits.loc[mask, stem_col].values)
            split_fold_test[(s, f)] = test_stems

print(f"[splits] Loaded {len(split_fold_test)} split×fold combos")


# ── Load Phase-1 encoders for each endpoint × split × fold × modality ─────────
# Returns dict: (split, fold, mod) → GatedAttentionEncoder (eval mode)
print("[models] Loading Phase-1 encoders ...", flush=True)

def _load_enc(ep: str, split: int, fold: int, mod: str) -> GatedAttentionEncoder | None:
    ckpt = V8_BASE / ep / f"split{split}_fold{fold}" / "phase1" / f"enc_{mod}.pt"
    if not ckpt.exists():
        return None
    enc = GatedAttentionEncoder(feat_dim=FEAT_DIMS[mod], hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    sd  = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    # strip "encoder." prefix if saved as full SingleModalMIL state dict
    if any(k.startswith("encoder.") for k in sd):
        sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    try:
        enc.load_state_dict(sd, strict=True)
    except RuntimeError:
        enc.load_state_dict(sd, strict=False)
    enc.eval()
    return enc

encoders: dict = {}   # (ep, split, fold, mod) → enc
for ep in ["acr", "clad", "death"]:
    for s in range(N_SPLITS):
        for f in range(N_FOLDS):
            for mod in MODALITIES:
                enc = _load_enc(ep, s, f, mod)
                if enc is not None:
                    encoders[(ep, s, f, mod)] = enc

print(f"[models] Loaded {len(encoders)} encoders", flush=True)


# ── Load inference-cache meta (cls_prob, hazard per stem per variant) ──────────
# We read the ANALYSIS-level pkl once per endpoint — gives us cls_prob and hazard
# for all samples so we don't need to re-run Phase-2 inference.
meta_cache: dict = {}   # endpoint → {variant_tag → {stem: {cls_prob, hazard}}}
for ep, (ep_dir, task, best_var, pred_key) in ENDPOINT_TASK.items():
    pkl = ANALYSIS / f"inference_cache_{ep_dir}_{task}_meta.pkl"
    if not pkl.exists():
        print(f"[cache] Missing {pkl} — skipping {ep}"); continue
    raw = pickle.load(open(pkl, "rb"))
    var_data = raw.get(best_var, {})
    stem_pred: dict = {}
    for i, stem in enumerate(var_data.get("stem", [])):
        stem_pred[str(stem)] = {
            "cls_prob": float(var_data["cls_prob"][i]) if "cls_prob" in var_data else np.nan,
            "hazard":   float(var_data["hazard"][i])   if "hazard"   in var_data else np.nan,
            "label":    float(var_data["label"][i])    if "label"    in var_data else np.nan,
        }
        # also copy survival event keys
        for surv_key in ["acr_event", "clad_event", "death_event",
                          "event_next_acr", "tte_next_acr"]:
            if surv_key in var_data:
                stem_pred[str(stem)][surv_key] = float(var_data[surv_key][i]) if var_data[surv_key][i] is not None else np.nan
    meta_cache[ep] = stem_pred
    print(f"[cache] {ep}: {len(stem_pred)} samples loaded")


# ── Run Phase-1 inference, collect per-cluster attention ─────────────────────
print("[inference] Extracting per-cluster attention weights ...", flush=True)

# records: list of dicts
# {stem, modality, coarse_type, mean_alpha, sum_alpha, n_instances,
#  label, cls_prob, hazard_acr, hazard_clad, hazard_death}
records = []

# Which (split, fold) pairs have encoders for ALL endpoints?
valid_sf = {(s,f) for (ep,s,f,m) in encoders}

all_pts = sorted(SAMPLES.glob("*.pt"))
print(f"[data] {len(all_pts)} sample files", flush=True)

# stem→fold assignment: use split0 (consistent with benchmark)
stem_to_sf: dict = {}
for (s, f), test_stems in split_fold_test.items():
    if s == 0:   # use split 0 for test set
        for stem in test_stems:
            stem_to_sf[stem.replace(".pt","")] = (s, f)

@torch.no_grad()
def extract_attn(enc: GatedAttentionEncoder, feats: torch.Tensor):
    """Return alpha (N,) for the given feature tensor."""
    x = feats.to(DEVICE)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    alpha, _ = enc(x)
    return alpha.cpu().numpy()


processed = 0
for pt_path in all_pts:
    stem = pt_path.stem
    sf   = stem_to_sf.get(stem)
    if sf is None:
        continue   # not in test set for split 0
    s, f = sf

    try:
        d = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [warn] Load error {stem}: {e}"); continue

    # Prediction values from cache
    pred_acr   = meta_cache.get("acr",   {}).get(stem, {})
    pred_clad  = meta_cache.get("clad",  {}).get(stem, {})
    pred_death = meta_cache.get("death", {}).get(stem, {})

    base_row = {
        "stem":         stem,
        "label":        pred_acr.get("label", float(d.get("label", 0))),
        "cls_prob":     pred_acr.get("cls_prob",    np.nan),
        "hazard_acr":   pred_acr.get("hazard",      np.nan),
        "hazard_clad":  pred_clad.get("hazard",     np.nan),
        "hazard_death": pred_death.get("hazard",    np.nan),
        "acr_event":    pred_acr.get("event_next_acr",  float(d.get("label",0))),
    }

    inputs        = d.get("inputs", {})
    cluster_labels= d.get("cluster_labels", {})
    clin_token_ids= d.get("clinical_token_ids")   # (102,)
    clin_vocab    = d.get("clinical_vocab", [])

    for mod in MODALITIES:
        feat_key = FEAT_KEYS[mod]

        # ── Clinical tokens use token_id → vocab label ───────────────────────
        if mod == "Clinical":
            if clin_token_ids is None:
                continue
            enc = encoders.get(("acr", s, f, mod))
            if enc is None:
                continue
            feats = d.get("clinical_onehot")   # (102, 408)
            if feats is None:
                continue
            alpha = extract_attn(enc, feats)   # (102,)
            id_to_label = {e["id"]: e["label"] for e in clin_vocab}
            if isinstance(clin_token_ids, torch.Tensor):
                tids = clin_token_ids.numpy().tolist()
            else:
                tids = list(clin_token_ids)
            for i, (tid, alph) in enumerate(zip(tids, alpha)):
                raw_lbl = id_to_label.get(int(tid), f"tok_{tid}")
                ctype   = coarse_label("Clinical", raw_lbl)
                records.append({**base_row,
                    "modality":    "Clinical",
                    "coarse_type": ctype,
                    "mean_alpha":  float(alph),
                    "sum_alpha":   float(alph),
                    "n_instances": 1,
                })
            continue

        # ── Patch/cell modalities (HE_cells, BAL_cells, CT_cells) ────────────
        feats = inputs.get(feat_key)
        if feats is None or feats.shape[0] == 0:
            continue

        raw_labels = cluster_labels.get(feat_key, [])
        if len(raw_labels) == 0:
            continue

        enc = encoders.get(("acr", s, f, mod))
        if enc is None:
            continue

        # subsample for memory
        N = feats.shape[0]
        if N > 2048:
            idx    = torch.randperm(N)[:2048]
            feats  = feats[idx]
            raw_labels = [raw_labels[int(i)] for i in idx]

        alpha = extract_attn(enc, feats)   # (N,)

        # group by coarse type
        type_alpha: dict = defaultdict(list)
        for lbl, a in zip(raw_labels, alpha):
            ctype = coarse_label(mod, str(lbl))
            type_alpha[ctype].append(a)

        for ctype, alphas in type_alpha.items():
            records.append({**base_row,
                "modality":    mod,
                "coarse_type": ctype,
                "mean_alpha":  float(np.mean(alphas)),
                "sum_alpha":   float(np.sum(alphas)),
                "n_instances": len(alphas),
            })

    processed += 1
    if processed % 100 == 0:
        print(f"  {processed} slides done, {len(records)} records", flush=True)


df = pd.DataFrame(records)
df.to_csv(OUT_DIR / "cluster_attn_records.csv", index=False)
print(f"[records] {len(df)} rows saved → {OUT_DIR / 'cluster_attn_records.csv'}", flush=True)

if df.empty:
    print("[error] No records — check encoder loading / sample paths"); sys.exit(0)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {name}.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — UMAP + hexbin panels per endpoint
# Reuses pre-computed UMAP embeddings; adds cls_prob + hazard hexbins.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[fig1] UMAP hexbin panels ...", flush=True)

UMAP_DIRS = {
    "acr":   ANALYSIS / "umap_acr_both_alt" / "middle",
    "clad":  ANALYSIS / "umap_clad_surv"    / "late",
    "death": ANALYSIS / "umap_death_surv"   / "late",
}

def _hex_extent(xy, pad=0.04):
    xr = xy[:,0].ptp(); yr = xy[:,1].ptp()
    return [xy[:,0].min()-pad*xr, xy[:,0].max()+pad*xr,
            xy[:,1].min()-pad*yr, xy[:,1].max()+pad*yr]

def _hexbin_panel(ax, xy, values, title, cmap, vmin, vmax, vcenter=None,
                  xy_ref=None, extent=None, gridsize=35, cbar_label=""):
    ref   = xy_ref if xy_ref is not None else xy
    ext   = extent  if extent  is not None else _hex_extent(ref)
    mask  = ~np.isnan(values.astype(float))
    # Reference hexbin (all points → defines grid)
    hb = ax.hexbin(ref[:,0], ref[:,1], gridsize=gridsize, mincnt=1,
                   extent=ext, visible=False, cmap=cmap)
    offsets = np.array(hb.get_offsets())
    n_bins  = len(offsets)
    merged  = np.full(n_bins, np.nan)
    if mask.sum() > 0:
        hbv = ax.hexbin(xy[mask,0], xy[mask,1], C=values[mask],
                        gridsize=gridsize, mincnt=1, extent=ext,
                        reduce_C_function=np.mean, visible=False, cmap=cmap)
        v_offs = np.array(hbv.get_offsets()); v_arr = np.array(hbv.get_array())
        for bi, off in enumerate(offsets):
            hit = np.where(np.all(np.abs(v_offs - off) < 1e-8, axis=1))[0]
            if len(hit):
                merged[bi] = v_arr[hit[0]]
    if vcenter is None:
        vcenter = float(np.nanmedian(merged[~np.isnan(merged)])) if np.any(~np.isnan(merged)) else 0
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax) if (vmin < vcenter < vmax) else None
    hb2 = ax.hexbin(ref[:,0], ref[:,1], gridsize=gridsize, mincnt=1, extent=ext,
                    cmap=cmap, norm=norm)
    hb2.set_array(np.where(np.isnan(merged), np.nan, merged))
    hb2.set_clim(vmin, vmax)
    cb = plt.colorbar(hb2, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    cb.ax.tick_params(labelsize=6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=7, fontweight="bold")
    for sp in ax.spines.values(): sp.set_visible(False)
    return hb2

ep_labels = {"acr": "ACR", "clad": "CLAD", "death": "Death"}
ep_colors = {"acr": "#E53935", "clad": "#FB8500", "death": "#C62828"}

for ep, umap_dir in UMAP_DIRS.items():
    ep_label = ep_labels[ep]
    # Load embeddings CSV
    emb_csv = list(umap_dir.glob("embeddings_*.csv"))
    if not emb_csv:
        print(f"  [umap] No embeddings CSV in {umap_dir} — skip"); continue
    emb_df = pd.read_csv(emb_csv[0])
    if "umap1" not in emb_df.columns or "umap2" not in emb_df.columns:
        print(f"  [umap] UMAP columns not found in {emb_csv[0]}"); continue

    xy  = emb_df[["umap1","umap2"]].values
    ext = _hex_extent(xy)

    # Merge prediction values from inference cache
    ep_dir_name, task, best_var, _ = ENDPOINT_TASK[ep]
    pred_map = meta_cache.get(ep, {})

    # enrich emb_df with prediction values
    def _get(col):
        return np.array([pred_map.get(str(row), {}).get(col, np.nan)
                         for row in emb_df.get("stem", emb_df.index).values], dtype=float)

    cls_prob  = _get("cls_prob")
    hazard    = _get("hazard")
    label_arr = _get("label")
    if ep == "acr":
        ev_arr = _get("acr_event")
    elif ep == "clad":
        ev_arr = np.array([pred_map.get(str(r), {}).get("clad_event", np.nan)
                           for r in emb_df.get("stem", emb_df.index).values], dtype=float)
    else:
        ev_arr = np.array([pred_map.get(str(r), {}).get("death_event", np.nan)
                           for r in emb_df.get("stem", emb_df.index).values], dtype=float)

    # If emb_df already has the prediction columns, prefer those
    if "cls_prob" in emb_df.columns:
        cls_prob = emb_df["cls_prob"].values.astype(float)
    if "hazard" in emb_df.columns:
        hazard = emb_df["hazard"].values.astype(float)

    haz_valid = hazard[~np.isnan(hazard)]
    hmin = float(np.percentile(haz_valid, 1))  if len(haz_valid) > 0 else -1
    hmax = float(np.percentile(haz_valid, 99)) if len(haz_valid) > 0 else 1
    hmed = float(np.median(haz_valid))          if len(haz_valid) > 0 else 0

    # 2-row figure: row0 = scatter (event status | cls_prob | hazard), row1 = hexbins
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

    # Row 0: scatter plots
    ax = axes[0, 0]
    ev_mask = ~np.isnan(ev_arr)
    if ev_mask.sum() > 0:
        c_ev = np.where(ev_arr[ev_mask] == 1.0, ep_colors[ep], "#1E88E5")
        ax.scatter(xy[ev_mask,0], xy[ev_mask,1], c=c_ev, s=8, alpha=0.7, linewidths=0, rasterized=True)
    ax.legend(handles=[Patch(color=ep_colors[ep], label=f"{ep_label} event"),
                        Patch(color="#1E88E5", label="Censored/−")],
              fontsize=6, loc="best", framealpha=0.8)
    ax.set_title(f"{ep_label} event status", fontsize=7, fontweight="bold")

    ax = axes[0, 1]
    cp_valid = ~np.isnan(cls_prob)
    if cp_valid.sum() > 0:
        sc = ax.scatter(xy[cp_valid,0], xy[cp_valid,1], c=cls_prob[cp_valid],
                        cmap="RdBu_r", vmin=0, vmax=1,
                        s=8, alpha=0.75, linewidths=0, rasterized=True)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
    ax.set_title("Classification prob P(event+)", fontsize=7, fontweight="bold")

    ax = axes[0, 2]
    hv_mask = ~np.isnan(hazard)
    if hv_mask.sum() > 0:
        sc = ax.scatter(xy[hv_mask,0], xy[hv_mask,1], c=hazard[hv_mask],
                        cmap="RdBu_r", vmin=hmin, vmax=hmax,
                        s=8, alpha=0.75, linewidths=0, rasterized=True)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="hazard").ax.tick_params(labelsize=6)
    ax.set_title("Hazard score (high=red)", fontsize=7, fontweight="bold")

    # Row 1: hexbin enrichment
    _hexbin_panel(axes[1,0], xy, ev_arr.astype(float),
                  f"{ep_label} event density (hexbin)", "RdBu_r", 0.0, 1.0,
                  xy_ref=xy, extent=ext, cbar_label=f"prop {ep_label}")
    _hexbin_panel(axes[1,1], xy, cls_prob,
                  "Mean classification prob (hexbin)", "RdBu_r", 0.0, 1.0, vcenter=0.5,
                  xy_ref=xy, extent=ext, cbar_label="mean P(event+)")
    _hexbin_panel(axes[1,2], xy, hazard,
                  "Mean hazard (hexbin)", "RdBu_r", hmin, hmax, vcenter=hmed,
                  xy_ref=xy, extent=ext, cbar_label="mean hazard")

    n_tot  = len(xy)
    n_ev   = int(np.nansum(ev_arr == 1.0))
    n_cens = n_tot - n_ev
    fig.suptitle(
        f"V8 {ep_label} UMAP — best variant: {best_var}  |  n={n_tot}  events={n_ev}  censored={n_cens}\n"
        "Row 1: scatter (event status, cls prob, hazard)  |  Row 2: hexbin enrichment",
        fontsize=8, y=1.01)
    plt.tight_layout()
    _save(fig, f"fig_umap_hexbin_{ep}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Cluster attention heatmap for ACR (ACR+ vs ACR−)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[fig2] Cluster attention heatmap — ACR ...", flush=True)

df_acr = df[df["label"].notna()].copy()
df_acr["acr_label"] = (df_acr["label"] == 1.0).astype(int)

# For each modality: pivot table of mean_alpha × coarse_type
fig2, axes2 = plt.subplots(1, len(MODALITIES), figsize=(5 * len(MODALITIES), 10))
if len(MODALITIES) == 1:
    axes2 = [axes2]

for ax, mod in zip(axes2, MODALITIES):
    sub = df_acr[df_acr["modality"] == mod]
    if sub.empty:
        ax.set_visible(False); continue

    # pivot: rows=coarse_type, cols={0,1}
    piv = sub.pivot_table(index="coarse_type", columns="acr_label",
                          values="mean_alpha", aggfunc="mean")
    piv = piv.reindex(sorted(piv.index))

    # Keep top-30 coarse types by variance across label
    if len(piv) > 30:
        var = piv.var(axis=1).fillna(0)
        piv = piv.loc[var.nlargest(30).index]

    mat   = piv.values
    vmax  = float(np.nanpercentile(mat, 98))
    im    = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=6)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(["ACR−", "ACR+"] if list(piv.columns) == [0,1] else
                       [f"label={c}" for c in piv.columns], fontsize=8)
    ax.set_title(f"{mod}\nCluster attention\n(test set mean)", fontsize=8, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="mean α").ax.tick_params(labelsize=6)

fig2.suptitle("V8 Phase-1 ABMIL — mean attention per coarse cluster type  |  ACR+/−",
              fontsize=9, fontweight="bold", y=1.01)
plt.tight_layout()
_save(fig2, "fig_cluster_attn_acr")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Cluster attention vs survival hazard (Spearman ρ, per modality)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[fig3] Cluster attention vs survival hazard ...", flush=True)

fig3, axes3 = plt.subplots(2, len(MODALITIES), figsize=(5 * len(MODALITIES), 12))
if len(MODALITIES) == 1:
    axes3 = axes3.reshape(2,1)

for col_i, mod in enumerate(MODALITIES):
    sub = df[df["modality"] == mod]
    if sub.empty:
        for row_i in range(2): axes3[row_i, col_i].set_visible(False)
        continue

    for row_i, (ep, hz_col) in enumerate([("clad","hazard_clad"), ("death","hazard_death")]):
        ax = axes3[row_i, col_i]
        sub_ep = sub[sub[hz_col].notna() & (sub[hz_col] != 0)]
        if sub_ep.empty:
            ax.set_visible(False); continue

        # Per-coarse type: slide-level mean_alpha, then Spearman with hazard
        corrs = {}
        for ctype, grp in sub_ep.groupby("coarse_type"):
            slide_mean = grp.groupby("stem")[["mean_alpha", hz_col]].mean()
            if len(slide_mean) < 5: continue
            rho, pval = spearmanr(slide_mean["mean_alpha"], slide_mean[hz_col])
            corrs[ctype] = (rho, pval)

        if not corrs:
            ax.set_visible(False); continue

        corr_df = pd.DataFrame({"coarse_type": list(corrs.keys()),
                                 "rho":  [v[0] for v in corrs.values()],
                                 "pval": [v[1] for v in corrs.values()]})
        corr_df = corr_df.sort_values("rho")
        # Top 20 by absolute correlation
        if len(corr_df) > 20:
            corr_df = corr_df.loc[corr_df["rho"].abs().nlargest(20).index].sort_values("rho")

        colors = ["#E53935" if r > 0 else "#1E88E5" for r in corr_df["rho"]]
        bars = ax.barh(corr_df["coarse_type"], corr_df["rho"],
                       color=colors, edgecolor="none", height=0.7)
        ax.axvline(0, color="black", lw=0.8, linestyle="--")
        ax.set_xlabel("Spearman ρ  (mean α vs hazard)", fontsize=7)
        ep_lbl = ep_labels.get(ep, ep.upper())
        ax.set_title(f"{mod}\n{ep_lbl} hazard correlation", fontsize=8, fontweight="bold")
        ax.tick_params(axis="y", labelsize=6)
        ax.tick_params(axis="x", labelsize=6)
        # mark significance (p<0.05)
        for i, (_, row) in enumerate(corr_df.iterrows()):
            if row["pval"] < 0.05:
                ax.text(row["rho"] + 0.01 * np.sign(row["rho"]), i, "*",
                        ha="center", va="center", fontsize=8, color="#333333")

fig3.suptitle("V8 Phase-1 — Cluster attention × survival hazard (Spearman ρ)\n"
              "Red=positive corr (high attn → short survival), Blue=negative\n* p<0.05",
              fontsize=8, fontweight="bold", y=1.01)
plt.tight_layout()
_save(fig3, "fig_cluster_attn_survival")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Signed discriminability (ACR+ mean α − ACR− mean α) per modality
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[fig4] Cluster discriminability ...", flush=True)

fig4, axes4 = plt.subplots(1, len(MODALITIES), figsize=(5 * len(MODALITIES), 8))
if len(MODALITIES) == 1:
    axes4 = [axes4]

for ax, mod in zip(axes4, MODALITIES):
    sub = df_acr[df_acr["modality"] == mod]
    if sub.empty:
        ax.set_visible(False); continue

    # Per slide: mean attention per coarse type
    slide_ct = sub.groupby(["stem","coarse_type","acr_label"])["mean_alpha"].mean().reset_index()
    # Per coarse type: mean over slides separately for ACR+ and ACR−
    pos_mean = slide_ct[slide_ct["acr_label"]==1].groupby("coarse_type")["mean_alpha"].mean()
    neg_mean = slide_ct[slide_ct["acr_label"]==0].groupby("coarse_type")["mean_alpha"].mean()
    disc     = (pos_mean - neg_mean).dropna().sort_values()

    if disc.empty:
        ax.set_visible(False); continue

    # Top-20 by absolute discriminability
    if len(disc) > 20:
        disc = disc.loc[disc.abs().nlargest(20).index].sort_values()

    colors = ["#E53935" if v > 0 else "#1E88E5" for v in disc.values]
    ax.barh(disc.index, disc.values, color=colors, edgecolor="none", height=0.7)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("mean α(ACR+) − mean α(ACR−)", fontsize=7)
    ax.set_title(f"{mod}\nSigned discriminability", fontsize=8, fontweight="bold")
    ax.tick_params(axis="y", labelsize=6)
    ax.tick_params(axis="x", labelsize=6)

fig4.suptitle("V8 Phase-1 — Cluster type discriminability for ACR\n"
              "Red = higher in ACR+,  Blue = higher in ACR−",
              fontsize=8, fontweight="bold", y=1.01)
plt.tight_layout()
_save(fig4, "fig_cluster_discriminability")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Violin plots: top discriminable coarse types per modality
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[fig5] Violin plots ...", flush=True)

TOP_N = 8

fig5, axes5 = plt.subplots(len(MODALITIES), 1, figsize=(14, 4 * len(MODALITIES)))
if len(MODALITIES) == 1:
    axes5 = [axes5]

for ax, mod in zip(axes5, MODALITIES):
    sub = df_acr[df_acr["modality"] == mod]
    if sub.empty:
        ax.set_visible(False); continue

    slide_ct = sub.groupby(["stem","coarse_type","acr_label"])["mean_alpha"].mean().reset_index()
    pos_mean = slide_ct[slide_ct["acr_label"]==1].groupby("coarse_type")["mean_alpha"].mean()
    neg_mean = slide_ct[slide_ct["acr_label"]==0].groupby("coarse_type")["mean_alpha"].mean()
    disc     = (pos_mean - neg_mean).dropna()
    top_types= disc.abs().nlargest(TOP_N).index.tolist()

    sub_top = slide_ct[slide_ct["coarse_type"].isin(top_types)].copy()

    x_positions = {ct: i*2 for i, ct in enumerate(top_types)}
    for lbl, color, offset in [(0, "#1E88E5", -0.3), (1, "#E53935", 0.3)]:
        for ct in top_types:
            vals = sub_top.loc[(sub_top["coarse_type"]==ct) & (sub_top["acr_label"]==lbl),
                               "mean_alpha"].values
            if len(vals) < 3: continue
            xp = x_positions[ct] + offset
            vp = ax.violinplot([vals], positions=[xp], widths=0.55,
                               showmedians=True, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(color); b.set_alpha(0.55)
            vp["cmedians"].set_color(color); vp["cmedians"].set_linewidth(1.5)

    ax.set_xticks([x_positions[ct] for ct in top_types])
    ax.set_xticklabels(top_types, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("mean α per slide", fontsize=7)
    ax.set_title(f"{mod} — top-{TOP_N} discriminable coarse types",
                 fontsize=8, fontweight="bold")
    ax.legend(handles=[Patch(color="#E53935", label="ACR+"),
                        Patch(color="#1E88E5", label="ACR−")],
              fontsize=7, loc="upper right")
    ax.tick_params(axis="y", labelsize=6)

fig5.suptitle("V8 Phase-1 — Attention distribution by coarse cluster type  (top discriminable per modality)",
              fontsize=8, fontweight="bold", y=1.01)
plt.tight_layout()
_save(fig5, "fig_cluster_violin_acr")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Modality-level total attention mass vs endpoint
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[fig6] Modality contribution by endpoint ...", flush=True)

# Per slide per modality: total attention mass = sum of alpha values
slide_mod = df.groupby(["stem","modality"])["sum_alpha"].sum().reset_index()
slide_mod = slide_mod.merge(
    df[["stem","label","hazard_acr","hazard_clad","hazard_death","acr_event"]].drop_duplicates("stem"),
    on="stem", how="left")

# Normalise per-slide total mass to sum to 1 across modalities
total_mass = slide_mod.groupby("stem")["sum_alpha"].transform("sum")
slide_mod["norm_alpha"] = slide_mod["sum_alpha"] / total_mass.clip(lower=1e-9)

fig6, axes6 = plt.subplots(1, 3, figsize=(15, 5))
for ax, (ep, hz_col, ep_lbl) in zip(axes6, [
        ("acr",   "label",        "ACR status"),
        ("clad",  "hazard_clad",  "CLAD hazard quartile"),
        ("death", "hazard_death", "Death hazard quartile"),
]):
    sub = slide_mod.copy()
    if ep == "acr":
        sub = sub[sub["label"].notna()]
        sub["group"] = sub["label"].map({0.0: "ACR−", 1.0: "ACR+"})
        order = ["ACR−", "ACR+"]
    else:
        sub = sub[sub[hz_col].notna()]
        sub["group"] = pd.qcut(sub[hz_col], 4,
                               labels=["Q1 low","Q2","Q3","Q4 high"])
        order = ["Q1 low","Q2","Q3","Q4 high"]

    piv = sub.groupby(["group","modality"])["norm_alpha"].mean().unstack("modality").reindex(order)
    piv = piv.fillna(0)
    bottom = np.zeros(len(piv))
    x = np.arange(len(piv))
    for mod in MODALITIES:
        if mod not in piv.columns: continue
        vals = piv[mod].values
        ax.bar(x, vals, bottom=bottom, label=mod,
               color=MOD_COLORS[mod], edgecolor="none", width=0.6)
        bottom += vals

    ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Normalised attention mass", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(f"Modality contribution\nby {ep_lbl}", fontsize=8, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    for sp in ax.spines.values(): sp.set_visible(False)

fig6.suptitle("V8 Phase-1 — Normalised modality attention mass by endpoint group",
              fontsize=8, fontweight="bold")
plt.tight_layout()
_save(fig6, "fig_modality_contribution_endpoint")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Combined endpoint summary heatmap (all endpoints × modality)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[fig7] Combined endpoint × modality summary heatmap ...", flush=True)

# For each (modality, coarse_type): compute 3 scores:
#   disc_acr  = mean_alpha(ACR+) - mean_alpha(ACR-)
#   rho_clad  = Spearman rho between mean_alpha per slide and hazard_clad
#   rho_death = Spearman rho between mean_alpha per slide and hazard_death

summary_rows = []
for mod in MODALITIES:
    sub = df[df["modality"] == mod]
    for ctype, grp in sub.groupby("coarse_type"):
        slide_grp = grp.groupby("stem")[["mean_alpha","label",
                                          "hazard_clad","hazard_death"]].mean()
        if len(slide_grp) < 5: continue

        lbl = slide_grp["label"]
        ma  = slide_grp["mean_alpha"]

        pos_mean = ma[lbl == 1].mean() if (lbl==1).sum() > 0 else np.nan
        neg_mean = ma[lbl == 0].mean() if (lbl==0).sum() > 0 else np.nan
        disc_acr = (pos_mean - neg_mean) if (not np.isnan(pos_mean) and not np.isnan(neg_mean)) else np.nan

        rho_clad = rho_death = np.nan
        hz_c = slide_grp["hazard_clad"].dropna()
        if len(hz_c) > 5:
            rho_clad, _ = spearmanr(ma.loc[hz_c.index], hz_c)
        hz_d = slide_grp["hazard_death"].dropna()
        if len(hz_d) > 5:
            rho_death, _ = spearmanr(ma.loc[hz_d.index], hz_d)

        summary_rows.append({"modality": mod, "coarse_type": ctype,
                              "disc_acr": disc_acr,
                              "rho_clad": rho_clad,
                              "rho_death": rho_death})

if summary_rows:
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(OUT_DIR / "cluster_endpoint_summary.csv", index=False)

    fig7, axes7 = plt.subplots(1, 3, figsize=(6 * 3, max(8, len(sdf)//3)))
    for ax, (col, title, cmap) in zip(axes7, [
            ("disc_acr",  "ACR discriminability\n(mean_α ACR+ − ACR−)", "RdBu_r"),
            ("rho_clad",  "CLAD hazard ρ\n(Spearman)", "RdBu_r"),
            ("rho_death", "Death hazard ρ\n(Spearman)", "RdBu_r"),
    ]):
        piv7 = sdf.pivot_table(index="coarse_type", columns="modality",
                               values=col, aggfunc="mean")
        piv7 = piv7.reindex(sorted(piv7.index))
        if len(piv7) > 40:
            var7 = piv7.var(axis=1).fillna(0)
            piv7 = piv7.loc[var7.nlargest(40).index]

        vabs = float(np.nanpercentile(np.abs(piv7.values), 98))
        im7  = ax.imshow(piv7.values, aspect="auto", cmap=cmap,
                         vmin=-vabs, vmax=vabs)
        ax.set_yticks(range(len(piv7))); ax.set_yticklabels(piv7.index, fontsize=6)
        ax.set_xticks(range(len(piv7.columns)))
        ax.set_xticklabels(piv7.columns, fontsize=8, rotation=30, ha="right")
        ax.set_title(title, fontsize=8, fontweight="bold")
        plt.colorbar(im7, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=6)

    fig7.suptitle("V8 Phase-1 — Cluster type importance across endpoints × modality",
                  fontsize=9, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig7, "fig_cluster_endpoint_summary")

print(f"\n[done] All figures saved to {OUT_DIR}")
print(f"[done] Files: {sorted([f.name for f in OUT_DIR.glob('*.pdf')])}")
PYEOF
