"""
pseudobulk_deg.py — Pseudobulk DEG + GSEA from BAL scRNA h5ad.

SEC 11: Aggregate BAL cells per patient × cell-type → pseudobulk DESeq2 → GSEA.
Fallback to Mann-Whitney if raw counts unavailable.
Outputs:
  bal_deg_results.csv, bal_gsea_results.csv
  fig11_deg_composition.png, fig11_deg_volcano.png,
  fig11_deg_gsea.png, fig11_deg_heatmap.png
"""

import warnings, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

OUT_DIR      = Path("/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper")
SAMPLE_TABLE = OUT_DIR / "sample_table.csv"
BAL_H5AD     = Path("/lustre/groups/aih/dinesh.haridoss/datasets/"
                    "08clad-4-annotated-v2-FIXED_date_from_id_dropped_recipient.h5ad")

NATURE_STYLE = {
    "font.family": "sans-serif", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
}
plt.rcParams.update(NATURE_STYLE)

def savefig(fig, name):
    p = OUT_DIR / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved → {p.name}", flush=True)

print("="*65)
print("PSEUDOBULK DEG + GSEA")
print("="*65, flush=True)

# ── Load sample table for ACR labels ──────────────────────────────────────────
if SAMPLE_TABLE.exists():
    sample_df = pd.read_csv(SAMPLE_TABLE)
    print(f"  Sample table: {sample_df.shape}", flush=True)
else:
    print("  WARNING: sample_table.csv not found", flush=True)
    sample_df = pd.DataFrame()

# ── Load BAL h5ad ─────────────────────────────────────────────────────────────
if not BAL_H5AD.exists():
    print(f"  ERROR: BAL h5ad not found: {BAL_H5AD}", flush=True)
    raise SystemExit(1)

import anndata as ad
print(f"  Loading BAL h5ad: {BAL_H5AD}", flush=True)
adata = ad.read_h5ad(str(BAL_H5AD))
print(f"  Shape: {adata.shape}", flush=True)
print(f"  obs cols: {list(adata.obs.columns)}", flush=True)
print(f"  layers: {list(adata.layers.keys()) if adata.layers else 'none'}", flush=True)
print(f"  obsm keys: {list(adata.obsm.keys())}", flush=True)

# ── Detect cluster column ──────────────────────────────────────────────────────
clust_col = None
for c in ["resolution_v2", "leiden", "subcluster", "cluster", "cell_type", "celltype"]:
    if c in adata.obs.columns:
        clust_col = c
        break
if clust_col is None:
    # find any column with <30 unique values
    for c in adata.obs.columns:
        if 1 < adata.obs[c].nunique() <= 30:
            clust_col = c
            break
print(f"  Cluster column: {clust_col}", flush=True)

# ── Detect patient ID column ───────────────────────────────────────────────────
pid_col = None
for c in ["record_id", "patient_id", "Patient", "sample"]:
    if c in adata.obs.columns:
        pid_col = c
        break
print(f"  Patient ID column: {pid_col}", flush=True)

# ── Detect raw counts ─────────────────────────────────────────────────────────
use_raw_counts = False
count_matrix = None
gene_names   = None

if adata.layers and any(k in adata.layers for k in ["counts", "raw_counts", "spliced"]):
    key = next(k for k in ["counts", "raw_counts", "spliced"] if k in adata.layers)
    import scipy.sparse as sp
    mat = adata.layers[key]
    count_matrix = mat.toarray() if sp.issparse(mat) else np.array(mat)
    gene_names   = list(adata.var_names)
    use_raw_counts = True
    print(f"  Using counts from layers['{key}'] — shape {count_matrix.shape}", flush=True)
elif adata.raw is not None:
    import scipy.sparse as sp
    mat = adata.raw.X
    count_matrix = mat.toarray() if sp.issparse(mat) else np.array(mat)
    gene_names   = list(adata.raw.var_names)
    use_raw_counts = True
    print(f"  Using raw.X counts — shape {count_matrix.shape}", flush=True)
else:
    import scipy.sparse as sp
    mat = adata.X
    count_matrix = mat.toarray() if sp.issparse(mat) else np.array(mat)
    gene_names   = list(adata.var_names)
    print(f"  WARNING: No raw counts layer found. Using X (may be normalised/scVI). "
          f"Shape: {count_matrix.shape}", flush=True)

# ── Pseudobulk: sum cells per patient × cell_type ────────────────────────────
print("\nBuilding pseudobulk matrix ...", flush=True)
obs = adata.obs.copy().reset_index(drop=True)
obs["_cell_idx"] = np.arange(len(obs))

# Link patient → ACR label
acr_map = {}
if pid_col and not sample_df.empty and "patient_id" in sample_df.columns:
    if "acr_binary" in sample_df.columns:
        pt_acr = sample_df.groupby("patient_id")["acr_binary"].max()
        acr_map = pt_acr.to_dict()
    elif "label" in sample_df.columns:
        pt_acr = (sample_df.groupby("patient_id")["label"].max() > 0).astype(int)
        acr_map = pt_acr.to_dict()

if pid_col:
    obs["_patient"] = obs[pid_col].astype(str)
else:
    obs["_patient"] = "all"

obs["_cluster"] = obs[clust_col].astype(str) if clust_col else "all"
obs["_acr"] = obs["_patient"].map(acr_map).fillna(-1).astype(int)

# Pseudobulk: keep only patients with ACR label
valid_pats = obs[obs["_acr"].isin([0, 1])]["_patient"].unique()
print(f"  Patients with ACR label: {len(valid_pats)}", flush=True)

# Subset to top 3 most abundant cell types for speed
top_clusters = (obs[obs["_patient"].isin(valid_pats)]
                ["_cluster"].value_counts().index[:3].tolist())
print(f"  Top cell types: {top_clusters}", flush=True)

pseudobulk_rows = []
for pat in valid_pats:
    pat_mask = obs["_patient"] == pat
    acr_val  = obs.loc[pat_mask, "_acr"].iloc[0]
    for ct in top_clusters:
        ct_mask = pat_mask & (obs["_cluster"] == ct)
        if ct_mask.sum() < 3:
            continue
        agg = count_matrix[ct_mask.values].sum(axis=0)
        pseudobulk_rows.append({
            "patient": pat, "cell_type": ct, "acr": acr_val,
            "counts": agg,
        })

print(f"  Pseudobulk samples: {len(pseudobulk_rows)}", flush=True)

# ── DEG per cell type ─────────────────────────────────────────────────────────
print("\nDifferential expression analysis ...", flush=True)
deg_results = []

for ct in top_clusters:
    ct_rows = [r for r in pseudobulk_rows if r["cell_type"] == ct]
    acr_pos = [r for r in ct_rows if r["acr"] == 1]
    acr_neg = [r for r in ct_rows if r["acr"] == 0]
    if len(acr_pos) < 2 or len(acr_neg) < 2:
        continue
    print(f"  {ct}: n_ACR+={len(acr_pos)} n_ACR-={len(acr_neg)}", flush=True)

    mat_pos = np.vstack([r["counts"] for r in acr_pos])
    mat_neg = np.vstack([r["counts"] for r in acr_neg])
    n_genes = mat_pos.shape[1]

    # Try PyDESeq2 if available, else fall back to Mann-Whitney
    deg_ok = False
    if use_raw_counts and n_genes > 0:
        try:
            from pydeseq2.dds import DeseqDataSet
            from pydeseq2.stat_tests import DeseqStats
            import scipy.sparse as sp

            counts_all = np.vstack([mat_pos, mat_neg]).astype(int)
            # Filter low-count genes
            gene_mask = (counts_all.sum(0) >= 10) & (counts_all.sum(0) > 0)
            counts_all = counts_all[:, gene_mask]
            gene_sub   = [gene_names[i] for i, m in enumerate(gene_mask) if m]
            if len(gene_sub) < 5:
                raise ValueError("Too few expressed genes")

            meta_all = pd.DataFrame({
                "sample": [f"pos_{i}" for i in range(len(acr_pos))] +
                          [f"neg_{i}" for i in range(len(acr_neg))],
                "condition": ["ACR+"] * len(acr_pos) + ["ACR-"] * len(acr_neg),
            }).set_index("sample")

            dds = DeseqDataSet(
                counts=pd.DataFrame(counts_all,
                                    index=meta_all.index,
                                    columns=gene_sub),
                metadata=meta_all,
                design_factors="condition",
            )
            dds.deseq2()
            stat = DeseqStats(dds, contrast=["condition", "ACR+", "ACR-"])
            stat.summary()
            res = stat.results_df.dropna()
            for g, row in res.iterrows():
                deg_results.append({
                    "gene": g, "cell_type": ct,
                    "log2fc": row["log2FoldChange"],
                    "pval":   row["pvalue"],
                    "padj":   row["padj"],
                })
            deg_ok = True
            print(f"    DESeq2 complete: {len(res)} genes tested", flush=True)
        except Exception as e:
            print(f"    DESeq2 failed: {e} — falling back to Mann-Whitney", flush=True)

    if not deg_ok:
        # Mann-Whitney fallback on all genes (or first 5000)
        n_test = min(n_genes, 5000)
        for gi in range(n_test):
            g_pos = mat_pos[:, gi].astype(float)
            g_neg = mat_neg[:, gi].astype(float)
            if g_pos.sum() + g_neg.sum() == 0:
                continue
            try:
                u, p = stats.mannwhitneyu(g_pos, g_neg, alternative="two-sided")
                m_pos = g_pos.mean(); m_neg = g_neg.mean()
                lfc = np.log2((m_pos + 0.1) / (m_neg + 0.1))
                deg_results.append({
                    "gene": gene_names[gi] if gi < len(gene_names) else f"Gene{gi}",
                    "cell_type": ct, "log2fc": lfc, "pval": p, "padj": np.nan,
                })
            except Exception:
                pass

if deg_results:
    deg_df = pd.DataFrame(deg_results)
    if "padj" in deg_df.columns and deg_df["padj"].isna().all():
        _, padj, _, _ = multipletests(deg_df["pval"].fillna(1), method="fdr_bh")
        deg_df["padj"] = padj
    deg_df.to_csv(OUT_DIR / "bal_deg_results.csv", index=False)
    print(f"  bal_deg_results.csv saved ({len(deg_df)} rows)", flush=True)
else:
    deg_df = pd.DataFrame()
    print("  No DEG results", flush=True)

# ── Fig 11a: Cell type composition ────────────────────────────────────────────
print("\nPlotting Fig 11a: cell type composition ...", flush=True)
if pid_col and clust_col:
    obs_valid = obs[obs["_patient"].isin(valid_pats)].copy()
    comp = obs_valid.groupby(["_patient","_cluster"]).size().unstack(fill_value=0)
    comp = comp.div(comp.sum(axis=1), axis=0)
    comp["_acr"] = obs_valid.groupby("_patient")["_acr"].first()
    comp = comp.sort_values("_acr")

    plot_cols = [c for c in comp.columns if c != "_acr"][:min(10, len(comp.columns))]
    fig, ax = plt.subplots(figsize=(min(14, len(comp) * 0.25 + 2), 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_cols)))
    bottom = np.zeros(len(comp))
    for ci, col in enumerate(plot_cols):
        ax.bar(range(len(comp)), comp[col].values, bottom=bottom,
               label=str(col)[:15], color=colors[ci], edgecolor="none")
        bottom += comp[col].values
    ax.set_xticks([])
    ax.set_ylabel("Proportion")
    ax.legend(bbox_to_anchor=(1.01, 1), fontsize=6, ncol=1)
    ax.set_title("Figure 11a: BAL cell type composition per patient (sorted by ACR)")
    # mark ACR+ patients
    acr_x = [i for i, a in enumerate(comp["_acr"].values) if a == 1]
    for x in acr_x:
        ax.axvline(x, color="#F44336", alpha=0.3, lw=0.5)
    savefig(fig, "fig11_deg_composition")

# ── Fig 11b: Volcano plot ─────────────────────────────────────────────────────
print("Plotting Fig 11b: volcano ...", flush=True)
if not deg_df.empty:
    ct_plot = deg_df[deg_df["cell_type"] == top_clusters[0]].copy() if top_clusters else deg_df.copy()
    ct_plot["-log10p"] = -np.log10(ct_plot["pval"].clip(lower=1e-300))
    ct_plot["sig"] = (ct_plot["padj"].fillna(1) < 0.05) & (ct_plot["log2fc"].abs() > 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ct_plot.loc[~ct_plot["sig"], "log2fc"],
               ct_plot.loc[~ct_plot["sig"], "-log10p"],
               s=4, alpha=0.4, color="#9E9E9E", linewidths=0)
    ax.scatter(ct_plot.loc[ct_plot["sig"] & (ct_plot["log2fc"] > 0), "log2fc"],
               ct_plot.loc[ct_plot["sig"] & (ct_plot["log2fc"] > 0), "-log10p"],
               s=6, alpha=0.8, color="#F44336", linewidths=0, label="Up in ACR+")
    ax.scatter(ct_plot.loc[ct_plot["sig"] & (ct_plot["log2fc"] < 0), "log2fc"],
               ct_plot.loc[ct_plot["sig"] & (ct_plot["log2fc"] < 0), "-log10p"],
               s=6, alpha=0.8, color="#2196F3", linewidths=0, label="Down in ACR+")
    # label top 10
    top10 = ct_plot[ct_plot["sig"]].nlargest(10, "-log10p")
    for _, r in top10.iterrows():
        ax.annotate(r["gene"], (r["log2fc"], r["-log10p"]),
                    fontsize=5, xytext=(3, 2), textcoords="offset points")
    ax.axvline(0, color="black", lw=0.5); ax.axvline(-1, color="grey", lw=0.5, ls="--"); ax.axvline(1, color="grey", lw=0.5, ls="--")
    ax.axhline(-np.log10(0.05), color="grey", lw=0.5, ls="--")
    ax.set_xlabel("log₂ Fold Change (ACR+ vs ACR−)")
    ax.set_ylabel("−log₁₀(p-value)")
    cell_label = top_clusters[0] if top_clusters else ""
    ax.set_title(f"Figure 11b: DEG Volcano — {cell_label}")
    ax.legend(fontsize=7)
    savefig(fig, "fig11_deg_volcano")

# ── Fig 11c: GSEA ─────────────────────────────────────────────────────────────
print("Running GSEA ...", flush=True)
gsea_df = pd.DataFrame()
if not deg_df.empty and deg_df["pval"].notna().any():
    try:
        import gseapy as gp
        ct_plot = deg_df[deg_df["cell_type"] == (top_clusters[0] if top_clusters else deg_df["cell_type"].iloc[0])].copy()
        # rank metric: sign(lfc) * -log10(p)
        ct_plot["rank"] = np.sign(ct_plot["log2fc"]) * (-np.log10(ct_plot["pval"].clip(lower=1e-300)))
        rnk = ct_plot.set_index("gene")["rank"].sort_values(ascending=False)
        rnk = rnk[~rnk.index.duplicated()]

        pre_res = gp.prerank(
            rnk=rnk, gene_sets="MSigDB_Hallmark_2020",
            threads=4, min_size=10, max_size=500,
            permutation_num=100, seed=42, verbose=False,
        )
        gsea_df = pre_res.res2d.sort_values("NES", key=abs, ascending=False).head(20)
        gsea_df.to_csv(OUT_DIR / "bal_gsea_results.csv", index=False)
        print(f"  GSEA: {len(gsea_df)} pathways", flush=True)

        # plot top 10
        top10_gsea = gsea_df.head(10).copy()
        top10_gsea["color"] = top10_gsea["NES"].apply(lambda x: "#F44336" if x > 0 else "#2196F3")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(range(len(top10_gsea)), top10_gsea["NES"].values,
                color=top10_gsea["color"].values)
        ax.set_yticks(range(len(top10_gsea)))
        ax.set_yticklabels(top10_gsea["Term"].str[:40].values, fontsize=6)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Normalised Enrichment Score (NES)")
        ax.set_title("Figure 11c: GSEA — Top hallmark pathways (ACR+ vs ACR−)")
        savefig(fig, "fig11_deg_gsea")
    except Exception as e:
        print(f"  GSEA failed: {e}", flush=True)

# ── Fig 11d: Heatmap of top DEGs ─────────────────────────────────────────────
print("Plotting Fig 11d: DEG heatmap ...", flush=True)
if not deg_df.empty and len(pseudobulk_rows) >= 4:
    ct_deg = deg_df[deg_df["cell_type"] == (top_clusters[0] if top_clusters else deg_df["cell_type"].iloc[0])].copy()
    top30_genes = ct_deg.nlargest(30, "-log10p" if "-log10p" in ct_deg.columns else "log2fc")["gene"].tolist()

    if top30_genes and gene_names:
        gene_idx = {g: i for i, g in enumerate(gene_names)}
        valid_genes = [g for g in top30_genes if g in gene_idx][:30]
        if valid_genes:
            ct = top_clusters[0] if top_clusters else None
            rows_ct = [r for r in pseudobulk_rows if r["cell_type"] == ct] if ct else pseudobulk_rows
            if len(rows_ct) >= 4:
                mat_hm = np.vstack([r["counts"][[gene_idx[g] for g in valid_genes]]
                                    for r in rows_ct]).astype(float)
                mat_hm = np.log1p(mat_hm)
                acr_labels = [r["acr"] for r in rows_ct]
                # zscore per gene
                mat_z = (mat_hm - mat_hm.mean(0)) / (mat_hm.std(0) + 1e-8)
                row_order = np.argsort(acr_labels)
                fig, ax = plt.subplots(figsize=(min(14, len(valid_genes) * 0.45 + 2), 5))
                im = ax.imshow(mat_z[row_order].T, aspect="auto", cmap="RdBu_r",
                               vmin=-2, vmax=2)
                ax.set_yticks(range(len(valid_genes)))
                ax.set_yticklabels(valid_genes, fontsize=5)
                ax.set_xticks([])
                ax.set_xlabel("Pseudobulk samples (sorted by ACR)")
                plt.colorbar(im, ax=ax, shrink=0.5, label="z-score")
                ax.set_title("Figure 11d: Top 30 DEGs — pseudobulk heatmap")
                savefig(fig, "fig11_deg_heatmap")

print("\nPseudobulk DEG + GSEA COMPLETE", flush=True)
