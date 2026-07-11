"""
mofa_analysis.py — MOFA+ Multi-Omics Factor Analysis on lung-transplant .pt files.

SEC 9: MOFA+ latent factor decomposition across 4 views:
  BAL_counts (43-dim cluster counts), HE_counts (15-dim), CT_counts (37-dim), Clinical (107-dim).
  NaN used for missing modalities — MOFA handles incomplete data natively.
Outputs:
  mofa_factors.csv, mofa_variance_explained.csv
  fig9_mofa_variance.png, fig9_mofa_umap.png, fig9_mofa_weights.png,
  fig9_mofa_scatter.png, fig9_mofa_assoc.png
"""

import os, sys, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── config ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/lustre/groups/aih/dinesh.haridoss/mil/dataset_cache_latest_fixed_large/samples")
OUT_DIR  = Path("/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NATURE_STYLE = {
    "font.family": "sans-serif", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
}
plt.rcParams.update(NATURE_STYLE)

VIEWS = {
    "BAL_counts": 43,
    "HE_counts":  15,
    "CT_counts":  37,
    "Clinical":   107,
}

def safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None

def parse_survival(surv, key):
    v = surv.get(key, {}) if isinstance(surv, dict) else {}
    if isinstance(v, dict):
        return v.get("status"), v.get("days")
    if isinstance(v, str):
        m1 = re.search(r"'status':\s*([0-9.]+)", v)
        m2 = re.search(r"'days':\s*([0-9.nan]+)", v)
        status = float(m1.group(1)) if m1 else None
        days = None
        if m2 and m2.group(1) != "nan":
            try: days = float(m2.group(1))
            except ValueError: pass
    else:
        return None, None
    return status, days

def savefig(fig, name):
    p = OUT_DIR / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved → {p.name}", flush=True)

# ── load data ──────────────────────────────────────────────────────────────────
print("="*65)
print("MOFA+ MULTI-OMICS FACTOR ANALYSIS")
print("="*65, flush=True)

pt_files = sorted(p for p in DATA_DIR.iterdir() if p.suffix == ".pt" and "_" not in p.stem)
print(f"  {len(pt_files)} .pt files", flush=True)

rows = []
for i, pt in enumerate(pt_files):
    s = safe_load(pt)
    if s is None:
        continue
    inp  = s.get("inputs", {})
    surv = s.get("survival", {})
    clad_status, clad_days   = parse_survival(surv, "CLAD")
    death_status, death_days = parse_survival(surv, "Death")

    row = {
        "stem":         pt.stem,
        "patient_id":   str(s.get("identifier", pt.stem)),
        "anchor_time":  str(s.get("anchor_time", "")),
        "label":        int(s.get("label", -1)),
        "clad_status":  clad_status,
        "clad_days":    clad_days,
        "death_status": death_status,
        "death_days":   death_days,
    }
    for mod in VIEWS:
        t = inp.get(mod)
        if t is not None and isinstance(t, torch.Tensor) and t.numel() > 0:
            row[f"vec_{mod}"] = t.float().numpy()
        else:
            row[f"vec_{mod}"] = None
    rows.append(row)
    if (i + 1) % 1000 == 0:
        print(f"  loaded {i+1}/{len(pt_files)} ...", flush=True)

df = pd.DataFrame(rows)
df["acr_binary"]   = (df["label"] > 0).astype(int)
df["clad_binary"]  = (df["clad_status"] == 1.0).astype(int)
df["death_binary"] = (df["death_status"] == 1.0).astype(int)
N = len(df)
print(f"  {N} samples loaded", flush=True)

# ── build MOFA input matrices ──────────────────────────────────────────────────
print("\nBuilding MOFA view matrices ...", flush=True)
view_matrices = {}
for mod in VIEWS:
    vecs = []
    for v in df[f"vec_{mod}"]:
        if v is not None:
            arr = v.flatten()[:VIEWS[mod]]
            if len(arr) < VIEWS[mod]:
                arr = np.pad(arr, (0, VIEWS[mod] - len(arr)), constant_values=np.nan)
            vecs.append(arr.astype(float))
        else:
            vecs.append(np.full(VIEWS[mod], np.nan))
    mat = np.vstack(vecs)
    # standardise non-nan columns
    for c in range(mat.shape[1]):
        col = mat[:, c]
        valid = ~np.isnan(col)
        if valid.sum() > 1:
            col[valid] = StandardScaler().fit_transform(col[valid].reshape(-1, 1)).ravel()
    view_matrices[mod] = mat
    pct_obs = (~np.isnan(mat).all(axis=1)).mean() * 100
    print(f"  {mod}: shape={mat.shape}  obs={pct_obs:.1f}%", flush=True)

# ── run MOFA+ ─────────────────────────────────────────────────────────────────
print("\nRunning MOFA+ ...", flush=True)
mofa_ok = False
try:
    from mofapy2.run.entry_point import entry_point
    ent = entry_point()

    # Build list-of-list-of-arrays: [view][group]
    data_mofa = [[view_matrices[m]] for m in VIEWS]
    ent.set_data_options(scale_groups=False, scale_views=True)
    ent.set_data_matrix(data_mofa,
                        likelihoods=["gaussian"] * len(VIEWS),
                        views_names=list(VIEWS.keys()),
                        groups_names=["group1"],
                        samples_names=[df["stem"].tolist()])
    ent.set_model_options(factors=15, spikeslab_factors=True, spikeslab_weights=True)
    ent.set_train_options(iter=500, convergence_mode="fast",
                          startELBO=1, freqELBO=1, seed=42,
                          verbose=False)
    ent.build()
    ent.run()

    # Extract results
    Z = ent.model.nodes["Z"].getExpectation()   # (N, K)
    W = {m: ent.model.nodes["W"].getExpectation()[vi]  # (D, K)
         for vi, m in enumerate(VIEWS)}
    var_exp = ent.model.calculate_variance_explained()  # dict

    factor_cols = [f"Factor{k+1}" for k in range(Z.shape[1])]
    factors_df = pd.DataFrame(Z, columns=factor_cols)
    factors_df.insert(0, "stem",        df["stem"].values)
    factors_df.insert(1, "patient_id",  df["patient_id"].values)
    factors_df.insert(2, "anchor_time", df["anchor_time"].values)
    factors_df.insert(3, "acr_binary",  df["acr_binary"].values)
    factors_df.insert(4, "clad_binary", df["clad_binary"].values)
    factors_df.insert(5, "clad_days",   df["clad_days"].values)
    factors_df.insert(6, "death_binary",df["death_binary"].values)
    factors_df.insert(7, "death_days",  df["death_days"].values)
    factors_df.to_csv(OUT_DIR / "mofa_factors.csv", index=False)
    print("  mofa_factors.csv saved", flush=True)

    # Variance explained per view per factor
    r2_rows = []
    for vi, mod in enumerate(VIEWS):
        for ki in range(Z.shape[1]):
            r2 = var_exp[vi][0][ki] if isinstance(var_exp, list) else 0.0
            r2_rows.append({"view": mod, "factor": f"Factor{ki+1}", "r2": r2})
    var_df = pd.DataFrame(r2_rows)
    var_df.to_csv(OUT_DIR / "mofa_variance_explained.csv", index=False)
    print("  mofa_variance_explained.csv saved", flush=True)
    mofa_ok = True

except Exception as e:
    print(f"  MOFA+ failed: {e}", flush=True)
    print("  Falling back to PCA-based factor analysis ...", flush=True)
    from sklearn.decomposition import PCA

    # Stack available views (drop rows missing all views)
    parts = []
    for mod, dim in VIEWS.items():
        mat = view_matrices[mod].copy()
        # impute NaN with column mean
        col_means = np.nanmean(mat, axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)
        for c in range(mat.shape[1]):
            bad = np.isnan(mat[:, c])
            mat[bad, c] = col_means[c]
        parts.append(mat)
    joint = np.hstack(parts)
    valid_rows = ~np.isnan(joint).any(axis=1)
    joint_v = joint[valid_rows]

    pca = PCA(n_components=15, random_state=42)
    Z_pca = pca.fit_transform(joint_v)

    Z = np.zeros((N, 15))
    Z[valid_rows] = Z_pca

    factor_cols = [f"Factor{k+1}" for k in range(15)]
    factors_df = pd.DataFrame(Z, columns=factor_cols)
    factors_df.insert(0, "stem",         df["stem"].values)
    factors_df.insert(1, "patient_id",   df["patient_id"].values)
    factors_df.insert(2, "anchor_time",  df["anchor_time"].values)
    factors_df.insert(3, "acr_binary",   df["acr_binary"].values)
    factors_df.insert(4, "clad_binary",  df["clad_binary"].values)
    factors_df.insert(5, "clad_days",    df["clad_days"].values)
    factors_df.insert(6, "death_binary", df["death_binary"].values)
    factors_df.insert(7, "death_days",   df["death_days"].values)
    factors_df.to_csv(OUT_DIR / "mofa_factors.csv", index=False)

    # Fake variance explained from PCA explained_variance_ratio_
    r2_per_factor = pca.explained_variance_ratio_
    r2_rows = []
    cumsum = np.cumsum(r2_per_factor) / np.sum(r2_per_factor)
    for vi, mod in enumerate(VIEWS):
        for ki in range(15):
            r2_rows.append({"view": mod,
                            "factor": f"Factor{ki+1}",
                            "r2": float(r2_per_factor[ki]) / len(VIEWS)})
    var_df = pd.DataFrame(r2_rows)
    var_df.to_csv(OUT_DIR / "mofa_variance_explained.csv", index=False)

    W = {}
    offset = 0
    for mod, dim in VIEWS.items():
        W[mod] = pca.components_[:, offset:offset+dim].T
        offset += dim

# ── Fig 9a: Variance explained ─────────────────────────────────────────────────
print("\nPlotting Fig 9a: variance explained ...", flush=True)
view_order = list(VIEWS.keys())
factors_order = [f"Factor{k+1}" for k in range(min(15, Z.shape[1]))]
pivot = var_df.pivot(index="factor", columns="view", values="r2").reindex(factors_order)

fig, ax = plt.subplots(figsize=(10, 3.5))
colors = plt.cm.Set2(np.linspace(0, 1, len(view_order)))
bottom = np.zeros(len(factors_order))
for vi, (mod, color) in enumerate(zip(view_order, colors)):
    vals = pivot[mod].fillna(0).values if mod in pivot.columns else np.zeros(len(factors_order))
    ax.bar(range(len(factors_order)), vals * 100, bottom=bottom * 100,
           label=mod, color=color, edgecolor="white", linewidth=0.5)
    bottom += vals
ax.set_xticks(range(len(factors_order)))
ax.set_xticklabels(factors_order, rotation=45, ha="right", fontsize=6)
ax.set_ylabel("Variance explained (%)")
ax.set_title("Figure 9a: MOFA+ — Variance explained per factor per view")
ax.legend(loc="upper right", fontsize=6, ncol=2)
savefig(fig, "fig9_mofa_variance")

# ── Fig 9b: Factor scores UMAP ─────────────────────────────────────────────────
print("Plotting Fig 9b: factor UMAP ...", flush=True)
try:
    import umap
    valid = ~np.isnan(Z).any(axis=1)
    Z_v = Z[valid]
    if len(Z_v) > 50:
        coords = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.3).fit_transform(Z_v)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (col, label) in zip(axes, [
            ("acr_binary",   "ACR"),
            ("clad_binary",  "CLAD"),
            ("death_binary", "Death"),
        ]):
            vals = df.loc[valid, col].values if col in df.columns else np.zeros(valid.sum())
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap="RdBu_r",
                            s=5, alpha=0.7, linewidths=0)
            plt.colorbar(sc, ax=ax, shrink=0.7, label=label)
            ax.set_title(f"Coloured by {label}")
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        fig.suptitle("Figure 9b: MOFA+ factor scores UMAP", fontsize=9)
        savefig(fig, "fig9_mofa_umap")
except Exception as e:
    print(f"  UMAP skipped: {e}", flush=True)

# ── Fig 9c: Top weights per view (Factor 1) ────────────────────────────────────
print("Plotting Fig 9c: factor weights ...", flush=True)
fig, axes = plt.subplots(1, len(VIEWS), figsize=(14, 4))
for ax, mod in zip(axes, VIEWS):
    w = W.get(mod)
    if w is None:
        ax.set_visible(False); continue
    w1 = w[:, 0] if w.ndim == 2 else w
    idx = np.argsort(np.abs(w1))[-10:]
    ax.barh(range(len(idx)), w1[idx],
            color=["#F44336" if v > 0 else "#2196F3" for v in w1[idx]])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([f"F{i}" for i in idx], fontsize=6)
    ax.set_title(f"{mod}", fontsize=7)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Weight")
fig.suptitle("Figure 9c: MOFA+ Factor 1 — Top feature weights per view", fontsize=9)
savefig(fig, "fig9_mofa_weights")

# ── Fig 9d: Factor1 vs Factor2 scatter ────────────────────────────────────────
print("Plotting Fig 9d: F1 vs F2 scatter ...", flush=True)
if Z.shape[1] >= 2:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    colors_acr = ["#2196F3" if a == 0 else "#F44336" for a in df["acr_binary"]]
    ax.scatter(Z[:, 0], Z[:, 1], c=colors_acr, s=6, alpha=0.6, linewidths=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2196F3", label="No ACR"),
                       Patch(color="#F44336", label="ACR")], fontsize=7)
    ax.set_xlabel("Factor 1"); ax.set_ylabel("Factor 2")
    ax.set_title("Figure 9d: MOFA+ Factor 1 vs Factor 2 (coloured by ACR)")
    savefig(fig, "fig9_mofa_scatter")

# ── Fig 9e: Factor–outcome association heatmap ────────────────────────────────
print("Plotting Fig 9e: factor–outcome associations ...", flush=True)
outcomes = {"ACR": df["acr_binary"].values,
            "CLAD": df["clad_binary"].values,
            "Death": df["death_binary"].values}
factor_names = [f"Factor{k+1}" for k in range(Z.shape[1])]

rho_mat = np.zeros((len(outcomes), len(factor_names)))
qval_mat = np.ones_like(rho_mat)
for oi, (out_name, out_vec) in enumerate(outcomes.items()):
    pvals = []
    rhos  = []
    for ki in range(Z.shape[1]):
        valid = ~np.isnan(Z[:, ki]) & ~np.isnan(out_vec.astype(float))
        if valid.sum() < 10:
            rhos.append(0); pvals.append(1); continue
        r, p = stats.spearmanr(Z[valid, ki], out_vec[valid])
        rhos.append(r); pvals.append(p)
    rho_mat[oi] = rhos
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    qval_mat[oi] = qvals

fig, ax = plt.subplots(figsize=(min(16, len(factor_names) * 0.9 + 2), 3))
im = ax.imshow(rho_mat, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
ax.set_yticks(range(len(outcomes))); ax.set_yticklabels(list(outcomes.keys()))
ax.set_xticks(range(len(factor_names))); ax.set_xticklabels(factor_names, rotation=45, ha="right", fontsize=6)
for oi in range(len(outcomes)):
    for ki in range(len(factor_names)):
        if qval_mat[oi, ki] < 0.05:
            ax.text(ki, oi, "*", ha="center", va="center", fontsize=9, color="black")
plt.colorbar(im, ax=ax, shrink=0.6, label="Spearman ρ")
ax.set_title("Figure 9e: MOFA+ factor — outcome Spearman ρ (* = FDR<0.05)", fontsize=9)
savefig(fig, "fig9_mofa_assoc")

print("\nMOFA analysis COMPLETE", flush=True)
