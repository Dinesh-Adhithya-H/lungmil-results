"""
attention_maps.py — Extract real ABMIL attention weights from trained benchmark models.

For each modality (HE, BAL, CT) and task (acr):
  1. Load the best-fold phase1 ABMIL checkpoint
  2. Forward each .pt sample through the encoder
  3. Extract per-patch attention weights
  4. Analyse which patches/clusters are most attended

Outputs: fig12_attention_*.png, attention_stats.csv
"""

import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("/lustre/groups/aih/dinesh.haridoss/mil/dataset_cache_latest_fixed_large/samples")
RESULTS_DIR = Path("/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8")
OUT_DIR     = Path("/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NATURE_STYLE = {
    "font.family": "sans-serif", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
}
plt.rcParams.update(NATURE_STYLE)

MOD_COLORS = {"HE": "#4e79a7", "BAL": "#f28e2b", "CT": "#59a14f"}
MOD_TO_KEY = {"HE": "HE_cells", "BAL": "BAL_cells", "CT": "CT_cells"}

# ── ABMIL model definition ────────────────────────────────────────────────────

class ABMILEncoder(nn.Module):
    def __init__(self, in_dim, hidden=256, att_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.att_V    = nn.Sequential(nn.Linear(hidden, att_dim), nn.Tanh())
        self.att_U    = nn.Sequential(nn.Linear(hidden, att_dim), nn.Sigmoid())
        self.att_w    = nn.Linear(att_dim, 1, bias=False)

    def forward(self, x):
        h   = self.backbone(x)                            # (N, hidden)
        A   = self.att_w(self.att_V(h) * self.att_U(h))  # (N, 1)
        A   = F.softmax(A, dim=0)
        bag = (A * h).sum(0)
        return bag, A.squeeze(-1)                         # bag, attn (N,)


def build_encoder_from_ckpt(state_dict):
    in_dim  = state_dict["encoder.backbone.0.weight"].shape[1]
    hidden  = state_dict["encoder.backbone.0.weight"].shape[0]
    att_dim = state_dict["encoder.att_V.0.weight"].shape[0]
    enc = ABMILEncoder(in_dim, hidden, att_dim)
    enc_sd = {k.replace("encoder.", ""): v
              for k, v in state_dict.items() if k.startswith("encoder.")}
    enc.load_state_dict(enc_sd)
    enc.eval()
    return enc


def best_fold_checkpoint(task="acr", modality="HE", split=1):
    best_auc, best_path = -1, None
    for fold in range(4):
        ckpt = RESULTS_DIR / f"phase1/split{split}_fold{fold}/{task}/{modality}/final/best_model.pt"
        if not ckpt.exists():
            continue
        # pick AUC from unimodal_ablation[modality] in cls metrics
        auc = 0.5
        for mf in RESULTS_DIR.glob(f"metrics_split{split}_fold{fold}*cls*.json"):
            try:
                d = json.loads(mf.read_text())
                # unimodal_ablation has per-modality AUC — most relevant
                mod_auc = d.get("unimodal_ablation", {}).get(modality, {}).get("auc", -1)
                if mod_auc > 0:
                    auc = max(auc, float(mod_auc))
                else:
                    # fallback: overall test AUC
                    auc = max(auc, float(d.get("test", {}).get("auc", 0.5)))
            except Exception:
                pass
        if auc > best_auc:
            best_auc, best_path = auc, ckpt
    return best_path, best_auc


# ── load .pt samples ──────────────────────────────────────────────────────────

def safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None


print("Loading .pt files ...", flush=True)
pt_files = sorted(p for p in DATA_DIR.iterdir() if p.suffix == ".pt" and "_" not in p.stem)
samples  = [s for p in pt_files if (s := safe_load(p)) is not None]
print(f"  {len(samples)} samples loaded", flush=True)

# ── extract attention weights ─────────────────────────────────────────────────

print("\nExtracting attention weights ...", flush=True)
records = []

for mod in ["HE", "BAL", "CT"]:
    cells_key   = MOD_TO_KEY[mod]
    cluster_key = cells_key

    ckpt_path, best_auc = best_fold_checkpoint(task="acr", modality=mod)
    if ckpt_path is None:
        print(f"  [{mod}] no checkpoint found — skipping", flush=True)
        continue

    print(f"  [{mod}] checkpoint: {ckpt_path}  AUC={best_auc:.3f}", flush=True)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    try:
        encoder = build_encoder_from_ckpt(sd)
    except Exception as e:
        print(f"  [{mod}] failed to build encoder: {e}", flush=True)
        continue

    n_done = 0
    with torch.no_grad():
        for s in samples:
            x = s.get("inputs", {}).get(cells_key)
            if x is None or not isinstance(x, torch.Tensor) or x.numel() == 0:
                continue
            if x.ndim == 1:
                x = x.unsqueeze(0)

            acr_bin = int(s.get("label", 0) > 0)

            try:
                _, attn = encoder(x.float())
            except Exception:
                continue

            attn_np = attn.numpy()
            N       = len(attn_np)
            entropy = float(-np.sum(attn_np * np.log(attn_np + 1e-12)))
            top_k   = min(10, N)
            top_idx = np.argsort(attn_np)[-top_k:]

            cl = (s.get("cluster_labels") or {}).get(cluster_key)
            top_clusters = []
            if cl is not None and len(cl) == N:
                top_clusters = [str(cl[i]) for i in top_idx]

            records.append({
                "stem":            str(s.get("identifier", "")),
                "anchor_time":     str(s.get("anchor_time", "")),
                "modality":        mod,
                "n_patches":       N,
                "acr_binary":      acr_bin,
                "attn_entropy":    entropy,
                "attn_max":        float(attn_np.max()),
                "attn_top10_mean": float(attn_np[top_idx].mean()),
                "top_clusters":    "|".join(top_clusters),
            })
            n_done += 1

    print(f"  [{mod}] {n_done} samples processed", flush=True)

df_attn = pd.DataFrame(records)
if df_attn.empty:
    print("No attention data extracted — check checkpoint paths.", flush=True)
    raise SystemExit(0)

df_attn.to_csv(OUT_DIR / "attention_stats.csv", index=False)
print(f"  attention_stats.csv: {len(df_attn)} rows", flush=True)

mods_present = df_attn["modality"].unique().tolist()


def savefig(fig, name):
    p = OUT_DIR / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved → {p.name}", flush=True)


# ── Fig 12a: max attention violin by modality × ACR ──────────────────────────
print("\nFig 12a ...", flush=True)
fig, axes = plt.subplots(1, len(mods_present), figsize=(4 * len(mods_present), 3.5))
if len(mods_present) == 1:
    axes = [axes]
for ax, mod in zip(axes, mods_present):
    sub  = df_attn[df_attn["modality"] == mod]
    g0   = sub[sub["acr_binary"] == 0]["attn_max"].values
    g1   = sub[sub["acr_binary"] == 1]["attn_max"].values
    if len(g0) == 0 or len(g1) == 0:
        continue
    parts = ax.violinplot([g0, g1], positions=[0, 1], showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], ["#2196F3", "#F44336"]):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No ACR", "ACR"])
    ax.set_title(f"{mod}"); ax.set_ylabel("Max attention weight")
fig.suptitle("Figure 12a: ABMIL attention weight distribution by ACR", fontsize=9)
savefig(fig, "fig12a_attn_violin")

# ── Fig 12b: top-attended cluster heatmap per modality ───────────────────────
print("Fig 12b ...", flush=True)
for mod in mods_present:
    sub = df_attn[(df_attn["modality"] == mod) & (df_attn["top_clusters"].str.len() > 0)]
    if sub.empty:
        continue
    rows = []
    for _, r in sub.iterrows():
        for cl in r["top_clusters"].split("|"):
            if cl:
                rows.append({"acr": int(r["acr_binary"]), "cluster": cl})
    if not rows:
        continue
    cl_df = pd.DataFrame(rows)
    ct    = cl_df.groupby(["cluster", "acr"]).size().unstack(fill_value=0)
    ct    = ct.div(ct.sum(axis=0).replace(0, 1), axis=1)
    ct.columns = [f"No ACR (n={int(sub[sub['acr_binary']==0].shape[0])})" if c == 0
                  else f"ACR (n={int(sub[sub['acr_binary']==1].shape[0])})" for c in ct.columns]

    fig, ax = plt.subplots(figsize=(4, max(3, len(ct) * 0.4)))
    im = ax.imshow(ct.values, aspect="auto", cmap="RdYlBu_r", vmin=0)
    ax.set_xticks(range(len(ct.columns))); ax.set_xticklabels(ct.columns, fontsize=7)
    ax.set_yticks(range(len(ct.index)));   ax.set_yticklabels(ct.index, fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Fraction of top-attended patches")
    ax.set_title(f"{mod} — cluster composition of top-attended patches")
    savefig(fig, f"fig12b_cluster_heatmap_{mod.lower()}")

# ── Fig 12c: attention entropy violin by ACR ──────────────────────────────────
print("Fig 12c ...", flush=True)
from scipy.stats import mannwhitneyu
fig, axes = plt.subplots(1, len(mods_present), figsize=(4 * len(mods_present), 3.5))
if len(mods_present) == 1:
    axes = [axes]
for ax, mod in zip(axes, mods_present):
    sub = df_attn[df_attn["modality"] == mod]
    g0  = sub[sub["acr_binary"] == 0]["attn_entropy"].values
    g1  = sub[sub["acr_binary"] == 1]["attn_entropy"].values
    if len(g0) == 0 or len(g1) == 0:
        continue
    parts = ax.violinplot([g0, g1], positions=[0, 1], showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], ["#2196F3", "#F44336"]):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    _, p = mannwhitneyu(g0, g1, alternative="two-sided")
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.set_title(f"{mod}  p={p:.3f} {star}")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No ACR", "ACR"])
    ax.set_ylabel("Attention entropy (lower = more focused)")
fig.suptitle("Figure 12c: Attention entropy by ACR status", fontsize=9)
savefig(fig, "fig12c_entropy_violin")

# ── Fig 12d: top-10 mean attention vs ACR scatter ────────────────────────────
print("Fig 12d ...", flush=True)
fig, axes = plt.subplots(1, len(mods_present), figsize=(4 * len(mods_present), 3.5))
if len(mods_present) == 1:
    axes = [axes]
rng = np.random.default_rng(42)
for ax, mod in zip(axes, mods_present):
    sub = df_attn[df_attn["modality"] == mod].copy()
    jitter = rng.uniform(-0.08, 0.08, len(sub))
    colors = ["#F44336" if a else "#2196F3" for a in sub["acr_binary"]]
    ax.scatter(sub["attn_top10_mean"], sub["acr_binary"] + jitter,
               c=colors, alpha=0.4, s=10, linewidths=0)
    ax.set_xlabel("Mean attention (top-10 patches)")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No ACR", "ACR"])
    ax.set_title(mod)
patches_ = [mpatches.Patch(color="#2196F3", label="No ACR"),
            mpatches.Patch(color="#F44336", label="ACR")]
axes[-1].legend(handles=patches_, fontsize=7)
fig.suptitle("Figure 12d: Top-patch attention weight vs ACR outcome", fontsize=9)
savefig(fig, "fig12d_attn_vs_acr")

print("\nattention_maps.py COMPLETE", flush=True)
