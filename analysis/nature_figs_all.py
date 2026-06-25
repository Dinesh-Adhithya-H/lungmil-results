#!/usr/bin/env python3
"""
nature_figs_all.py — All Nature-submission figures in one script.
Submit via:  sbatch analysis/submit_nature_figs_all.sh
Do NOT run directly on the login node.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA & LABELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source CSVs (precomputed from .pt bag files):
  bal_cluster_prop.csv   134 BAL scRNA samples  × 43 cell-type clusters
  he_cluster_prop.csv    632 H&E biopsy samples × 54 tissue sub-clusters
  ct_cluster_prop.csv   1835 CT scan samples    × 37 patch clusters
  clinical_features.csv 3340 samples × 106 features (extracted from mil_v2 .pt)
  clinical_feature_names.csv  106 feature names
  *_cluster_names.csv    human-readable names for each cluster index
  HE_cluster_map.json    maps H&E sub-cluster code → tissue type category

Splits file (multimodal_splits_nested_cv.csv, 4210 rows):
  label column:  0 = A0B0 (ACR−)  |  1 = A1/A2 (ACR+)  |  NaN = no biopsy (excluded)
  acr_days / acr_status   — time-to-first-ACR and event flag
  clad_days / clad_status — time-to-CLAD and event flag
  death_days/ death_status— time-to-death and event flag

H&E tissue aggregation:
  54 sub-clusters → 6 tissue types via HE_cluster_map.json:
  Alveolar inflamed | Alveolar | Alveolar (clear) | Bronchial | Vascular | Unknown

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROCESSING PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Load raw cluster proportions (each row sums to 1 by construction).
2. Merge with splits via stem key (zero-padded 5-digit integer).
3. For Figs 7–19: use RAW proportions (0–1 scale, shown as %).
4. For Figs 20–24: apply CLR transform (Centered Log-Ratio):
     CLR(xᵢ) = log(xᵢ) − mean_j[log(xⱼ)]
   Zeros replaced with pseudocount 1e-5 before log; row re-normalised to
   sum=1 first. CLR maps compositions to unconstrained real space, removing
   the spurious negative correlations induced by the unit-sum constraint.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATISTICAL TESTS USED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mann-Whitney U (two-sided)
  Used in: Figs 7–9, 12, 20 (bar/violin pairwise comparisons, volcano).
  Tests whether the distribution of a cluster proportion differs between
  two groups (e.g. ACR+ vs ACR−) without assuming normality.
  Requires n ≥ 3 per group; returns p-value only (no effect size stored).

FDR correction — Benjamini-Hochberg (BH)
  Applied after Mann-Whitney across all clusters in a modality × task
  combination (Figs 10, 12, 20) and after Fisher-z tests (Fig 21).
  Threshold q < 0.05 used throughout. Implemented via
  statsmodels.stats.multitest.multipletests(method="fdr_bh").

Spearman rank correlation
  Used in: Figs 15, 18 (cross-modal), Fig 19 TTE column, Figs 22, 24
  (pairwise product vs outcome).
  Non-parametric, robust to outliers and non-linear monotone relationships.
  Requires n ≥ 5 paired samples.

Fisher's z-test for differential correlation (Fig 21)
  Tests whether Spearman r in group A differs significantly from group B:
    z = (arctanh(r₁) − arctanh(r₂)) / sqrt(1/(n₁−3) + 1/(n₂−3))
  Marks rewired pairs with × on the heatmap (p < 0.05, BH-corrected).

Kruskal-Wallis H-test (Fig TTE)
  Non-parametric one-way ANOVA across ≥ 2 TTE strata (Q1/Q2/Q3/Q4/Censored).
  Tests whether cluster proportions differ across survival time strata.
  BH-corrected q-values shown as dot plot; q < 0.05 marked *.

Log-rank test (Fig 17)
  Compares KM survival curves of high vs low tertile groups.
  Implemented via lifelines.statistics.logrank_test.
  p-value annotated on each KM panel.

LASSO logistic regression (Fig 23)
  L1-penalised logistic regression on CLR-transformed features (StandardScaler
  applied first). Regularisation strength C selected by StratifiedKFold CV
  (k = min(5, n_pos)). Solver: saga. Reports cross-val ROC-AUC and
  coefficient magnitude (signed: positive = enriched in positive group).

Random Forest importance (Fig 23)
  200 trees, max_depth=4, balanced class weights, Gini impurity.
  Feature importance = mean decrease in Gini across all splits.
  Cross-val ROC-AUC reported alongside bar chart.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIGURE DESCRIPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fig 0  — Project overview schematic
  Panel A: Matplotlib-drawn schematic of the multimodal MIL pipeline:
    patient cohort → serial biopsies → 4 modalities → instance clustering
    → MIL aggregation (Shared Slot Attention) → 4 clinical tasks.
  Panel B: Modality availability heatmap (patients × modalities, sorted by
    ACR burden). Violin of instance counts per sample per modality.
    Cohort summary table (n patients, events, timepoints).
  Panel C: Bar chart of top cluster proportions (ACR+ vs ACR−) per modality.

Fig 0b — Serial patient timeline
  Rows = patients (sorted by CLAD/death burden), columns = timepoints
  (chronological). Each circle = one H&E biopsy visit.
  Circle colour: ACR+ = red, ACR− = blue, no biopsy = grey.
  Circle opacity = alveolar-inflammation proportion at that visit.
  Value inside circle = alveolar-inflamed % (integer).
  CLAD/death labels appended to right of timeline.

Fig 7  — BAL scRNA cluster proportions × clinical tasks
  Layout: 4 rows (ACR, CLAD, Death, TTE-quartile) × 2 cols (bar, violin).
  Bar: top-15 clusters by mean proportion; dark bar = positive group,
    light bar = negative group; * = MW p < 0.05.
  Violin: top-8 clusters by |mean difference|; jittered points overlaid;
    significance stars above each pair.
  Row 4 col 1: Q1 vs Q4 vs Censored bar for endpoint with most events.
  Row 4 col 2: Spearman r of each cluster with TTE (events only).

Fig 8  — H&E tissue proportions × clinical tasks  (same layout as Fig 7)
Fig 9  — CT scan cluster proportions × clinical tasks  (same layout as Fig 7)

Fig 10 — Cross-modal enrichment heatmap
  Rows = top 12 clusters per modality (BAL/H&E/CT, concatenated).
  Cols = 3 tasks (ACR, CLAD, Death).
  Colour = log₂FC (positive group mean / negative group mean).
  * = BH-corrected q < 0.05.
  Rows sorted by max |log₂FC| across tasks.

Fig 11 — Compact stacked-bar summary
  3 × 3 grid (modalities × tasks). Each cell: two horizontal stacked bars
  (positive group top, negative group bottom). Bar segments = top-15 clusters
  coloured by cluster palette. Proportions sum to 100%.

Fig 12 — Volcano plots (raw proportions)
  3 × 3 grid. x-axis = log₂FC (pos/neg mean + 1e-6). y-axis = −log₁₀(BH q).
  Red dots = FDR < 5% and log₂FC > 60th percentile (enriched in positive).
  Blue dots = FDR < 5% and log₂FC < −threshold (depleted).
  Top 5 significant clusters labelled by name.
  Dashed line at q = 0.05; dotted lines at ±LFC threshold.

Fig 13 — Serial trajectory heatmaps
  One figure per modality (BAL, H&E). Rows = patients (sorted by ACR burden),
  columns = timepoints (chronological, NaN = missing visit).
  Colour = cluster proportion. Top-6 highest-variance clusters shown.
  White dashed line separates ACR+ from ACR− patients.

Fig 14 — BAL cell-type stacked area over time post-transplant
  x-axis = time bins (0–6 mo, 6–12 mo, 12–24 mo, >24 mo from clad_days).
  Stacked bars = top-10 cell types by mean proportion.
  Separate panels for each task × group (ACR+/−, CLAD+/−, Death+/−).

Fig 15 — BAL × H&E cross-modal Spearman correlation
  Rows = H&E tissue types (6), columns = top-12 BAL cell types.
  Only samples with BOTH BAL and H&E at same timepoint (n = 94 paired).
  Colour = Spearman r. * = BH-corrected q < 0.05.

Fig 16 — ACR episode cluster fingerprint
  For each consecutive visit pair where visit i = ACR− and visit i+1 = ACR+,
  compute Δ proportion = proportion_at_ACR − proportion_pre_ACR.
  Bar chart sorted by |mean Δ|; error bars = SEM across pairs.
  Red = increased at ACR episode, blue = decreased.
  (BAL skipped if < 5 consecutive pairs available.)

Fig 17 — Cluster-stratified KM survival curves
  Top-3 BAL clusters by |log₂FC| (ACR+ vs ACR−) used for stratification.
  High = ≥ 67th percentile, Low = ≤ 33rd percentile of that cluster.
  KM curves for CLAD-free and overall survival.
  Log-rank p-value annotated per panel.

Fig 18 — BAL × CT cross-modal Spearman correlation
  Same as Fig 15 but for BAL × CT paired samples (n = 81).

Fig 19 — Instance counts per sample × task
  Violin + jitter of raw instance counts (log scale) for positive vs negative
  group per modality × task. MW p-value annotated.

Fig TTE — TTE 4-quartile cluster landscape (9 panels + summary)
  For each modality × endpoint (3 × 3 = 9 figures):
    Panel A: heatmap of z-scored CLR proportions (rows = top-15 variable
      clusters, cols = Q1/Q2/Q3/Q4/Censored strata). * = KW q < 0.05.
    Panel B: line plot of top-6 most dynamic clusters across strata (mean ±
      SEM). X-axis colour matches stratum colour.
    Panel C: Kruskal-Wallis −log₁₀(q) dot plot, coloured by significance.
  TTE strata: Q1 = earliest 25% events, Q2/Q3/Q4 = successive quartiles,
    Censored = no event. Requires ≥ 8 events to produce quartiles.
  Summary figure: 3 × 3 bar chart, Q1 vs Q4 vs Censored only.

Fig 20 — CLR volcano (replaces Fig 12 with CLR-transformed features)
  x-axis = ΔCLR = mean_CLR(pos) − mean_CLR(neg)  (additive on log scale,
    equivalent to log ratio of geometric means in original space).
  y-axis = −log₁₀(BH q) from Mann-Whitney on CLR values.
  Threshold = 60th percentile of |ΔCLR| across clusters.

Fig 21 — Differential co-occurrence (rewiring) networks
  One figure per modality (3 × 3 tasks per figure).
  For top-n most variable clusters (CLR), compute Spearman correlation matrix
  separately in positive group and negative group.
  Heatmap shows ΔR = R_pos − R_neg.  × marks = Fisher-z BH q < 0.05
  (pair whose co-expression is significantly rewired between groups).

Fig 22 — Pairwise cluster interaction → outcome
  For top-10 variable clusters (CLR): form all pairwise products CLR(i)×CLR(j)
  as an interaction feature. Compute Spearman r of each product with binary
  outcome label. Bar chart shows top pairs ranked by |r|; BH q annotated.
  Network: nodes = clusters (size ∝ summed |r| of connected edges),
  edges = top-10 pairs (thickness ∝ |r|, red = positive, blue = negative).

Fig 23 — LASSO + RF multivariate signature (CLR features)
  LASSO: L1 logistic regression on StandardScaler-normalised CLR values.
    C selected by StratifiedKFold CV. Signed coefficients shown (top 15).
    Cross-val ROC-AUC in title.
  RF: 200-tree forest on raw CLR values. Gini impurity importance shown
    (top 15, always positive). Cross-val ROC-AUC in title.
  Both run per modality × task.

Fig 24 — Q1 early-event co-occurrence fingerprint
  Same pairwise-product Spearman approach as Fig 22 but target = binary flag
  Q1 (earliest 25% of events) vs Q4 + Censored (reference).
  Identifies which cluster PAIRS are jointly elevated specifically in patients
  who experience the event soonest. One panel per endpoint (ACR, CLAD, Death).

Fig 25 — H&E × CT cross-modal co-occurrence → outcome
Fig 26 — BAL × H&E cross-modal co-occurrence → outcome
Fig 27 — BAL × CT cross-modal co-occurrence → outcome
Fig 28a — Clinical × BAL cross-modal co-occurrence → outcome
Fig 28b — Clinical × H&E cross-modal co-occurrence → outcome
Fig 28c — Clinical × CT cross-modal co-occurrence → outcome
  These three figures answer: does co-elevation of cluster i in modality A
  AND cluster j in modality B predict ACR+/CLAD/Death?
  Only samples measured on BOTH modalities at the same timepoint are used
  (matched by stem_int_str = same bag/visit).
  Interaction feature = CLR(A_i) × CLR(B_j)  (large only when BOTH elevated).
  Left panel: heatmap rows=top-10 clusters from mod A, cols=top-10 from mod B.
    Colour = Spearman r of interaction feature with binary outcome.
    * = BH FDR q < 0.05.
  Right panel: ranked bar chart of top-20 cross-modal pairs by |r|.
    Red = pair co-elevated in positive group. Blue = anti-correlated.
    Significance stars + q-value annotated per bar.

Fig 28 — Multi-modal interaction LASSO (one figure per task)
  For each of the three cross-modal combinations (H&E×CT, BAL×H&E, BAL×CT):
  Build a feature matrix of all cross-modal CLR products (top-7 clusters from
  each modality → up to 49 interaction features per pair). Fit L1-penalised
  logistic regression (LASSO, C selected by StratifiedKFold CV).
  Signed coefficients show which specific cross-modal interactions drive
  prediction: positive coef = co-elevation of that pair predicts the positive
  class; negative coef = co-elevation predicts the negative class.
  Cross-val ROC-AUC annotated in panel title.
  Stat: same LASSO as Fig 23 (solver=saga, max_iter=500, scoring=roc_auc).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import gc
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, kruskal
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
CLUSTER_PROP_DIR = Path("/home/aih/dinesh.haridoss/chicago_mil/results/cluster_proportions")
HE_CLMAP         = Path("/home/aih/dinesh.haridoss/chicago_mil/results/cluster_name_maps/HE_cluster_map.json")
SPLITS_CSV       = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
SAMPLE_TABLE     = Path("/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/sample_table.csv")
COHORT_JSON      = Path("/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/cohort_summary.json")
OUT_DIR          = Path("/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8,
    "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
})

ACR_POS   = "#E53935"
ACR_NEG   = "#1E88E5"
CLAD_COL  = "#FB8C00"
DEATH_COL = "#5C6BC0"
GREY      = "#90A4AE"
DARK      = "#37474F"

TISSUE_MERGE = {
    "Alveolar with hemorrhage and inflammation": "Alveolar inflamed",
    "Alveolar with empty spaces":               "Alveolar (clear)",
    "Alveolar":                                 "Alveolar",
    "Bronchial":                                "Bronchial",
    "Vascular":                                 "Vascular",
    "Unknown":                                  "Unknown",
}
TISSUE_COLORS = {
    "Alveolar inflamed": "#E53935",
    "Alveolar (clear)":  "#42A5F5",
    "Alveolar":          "#66BB6A",
    "Bronchial":         "#FFA726",
    "Vascular":          "#AB47BC",
    "Unknown":           "#B0BEC5",
}
TISSUE_ORDER = ["Alveolar inflamed", "Alveolar", "Alveolar (clear)", "Bronchial", "Vascular", "Unknown"]
MOD_COLORS   = {"H&E": "#66BB6A", "BAL scRNA": "#42A5F5", "CT scan": "#FFA726", "Clinical": "#AB47BC"}
TTE_Q_COLORS = ["#B71C1C", "#EF5350", "#EF9A9A", "#90CAF9", "#B0BEC5"]
TTE_Q_LABELS = ["Q1 (earliest)", "Q2", "Q3", "Q4 (latest)", "Censored"]

cmap20 = cm.get_cmap("tab20", 20)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {name}.png/.pdf", flush=True)


def pval_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def mw_test(a, b):
    try:
        if len(a) < 3 or len(b) < 3:
            return 1.0
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except Exception:
        return 1.0


def get_col(df, base):
    """Return first existing column variant: base, base_s, base_p."""
    for s in ["", "_s", "_p", "_sp", "_prop"]:
        if base + s in df.columns:
            return base + s
    return base


def prep_group(df, group_col_base, cluster_cols):
    """Return tmp df with integer group label; NaN rows dropped."""
    gc = get_col(df, group_col_base)
    tmp = df[[gc] + cluster_cols].copy()
    tmp[gc] = tmp[gc].where(tmp[gc].notna(), other=np.nan)
    tmp = tmp[tmp[gc].isin([0, 1])].copy()
    tmp[gc] = tmp[gc].astype(int)
    return tmp, gc


def violin_clusters(ax, df, cluster_cols, cluster_labels, group_col,
                    pos_color, neg_color, pos_label, neg_label, title, n_top=8):
    pos = df[df[group_col] == 1]
    neg = df[df[group_col] == 0]
    if len(pos) < 3 or len(neg) < 3:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color=GREY)
        ax.set_title(title); return
    diffs = [abs(pos[c].mean() - neg[c].mean()) for c in cluster_cols]
    order = np.argsort(diffs)[::-1][:n_top]
    top_cols   = [cluster_cols[i] for i in order]
    top_labels = [cluster_labels[i] for i in order]
    xs = np.arange(len(top_cols)) * 2.5
    np.random.seed(42)
    for xi, (col, lbl) in enumerate(zip(top_cols, top_labels)):
        vp = neg[col].dropna().values * 100
        vq = pos[col].dropna().values * 100
        p  = mw_test(vp / 100, vq / 100)
        for vals, xp, col_c, al in [(vp, xs[xi]-0.35, neg_color, 0.55),
                                     (vq, xs[xi]+0.35, pos_color, 0.75)]:
            if len(vals) > 2:
                viol = ax.violinplot([vals], [xp], showmedians=True,
                                     widths=0.55, showextrema=False)
                for pc in viol["bodies"]:
                    pc.set_facecolor(col_c); pc.set_alpha(al)
                viol["cmedians"].set_color(DARK); viol["cmedians"].set_linewidth(1.2)
                jx = np.random.normal(xp, 0.07, len(vals))
                ax.scatter(jx, vals, color=col_c, alpha=0.35, s=3, linewidths=0, zorder=3)
        stars = pval_stars(p)
        ytop = max(max(vp) if len(vp) else 0, max(vq) if len(vq) else 0) * 1.08
        ax.text(xs[xi], ytop, stars if stars != "ns" else "ns",
                ha="center", va="bottom",
                fontsize=8 if stars != "ns" else 6,
                color=DARK if stars != "ns" else GREY,
                fontweight="bold" if stars != "ns" else "normal")
    ax.set_xticks(xs)
    ax.set_xticklabels([l[:14] for l in top_labels], rotation=35, ha="right", fontsize=5.5)
    ax.set_ylabel("Proportion (%)")
    ax.set_title(title)


def bar_comparison(ax, df, cluster_cols, cluster_labels, palette,
                   group_col, pos_label, neg_label, title, n_top=15):
    pos = df[df[group_col] == 1]
    neg = df[df[group_col] == 0]
    n_pos, n_neg = len(pos), len(neg)
    mean_all = df[cluster_cols].mean()
    top_cols   = mean_all.nlargest(n_top).index.tolist()
    top_labels = [cluster_labels[cluster_cols.index(c)] for c in top_cols]
    colors     = [palette.get(l, GREY) for l in top_labels]
    x = np.arange(len(top_cols)); w = 0.35
    for ci, (col, lbl, col_c) in enumerate(zip(top_cols, top_labels, colors)):
        pm = pos[col].mean() * 100 if n_pos > 0 else 0
        nm = neg[col].mean() * 100 if n_neg > 0 else 0
        ax.bar(ci - w/2, pm, w, color=col_c, alpha=0.9, linewidth=0)
        ax.bar(ci + w/2, nm, w, color=col_c, alpha=0.45, linewidth=0)
        if n_pos > 2 and n_neg > 2:
            p = mw_test(pos[col].dropna().values, neg[col].dropna().values)
            if p < 0.05:
                ax.text(ci, max(pm, nm) + 0.3, pval_stars(p),
                        ha="center", fontsize=7, color=DARK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([l[:16] for l in top_labels], rotation=40, ha="right", fontsize=5.5)
    ax.set_ylabel("Mean proportion (%)")
    ax.set_title(title)
    ax.legend(handles=[mpatches.Patch(color="#555", alpha=0.9,
                                       label=f"{pos_label} (n={n_pos})"),
                        mpatches.Patch(color="#555", alpha=0.45,
                                       label=f"{neg_label} (n={n_neg})")],
               frameon=False, fontsize=6.5, loc="upper right")


def assign_tte_strata(df, tte_col_base, status_col_base):
    """0-3 = event quartiles Q1-Q4, 4 = censored, -1 = TTE missing."""
    tc = get_col(df, tte_col_base)
    sc = get_col(df, status_col_base)
    strat = pd.Series(-1, index=df.index)
    if tc not in df.columns or sc not in df.columns:
        return strat, 0
    ev  = df[sc].fillna(0).astype(int) == 1
    cen = ~ev
    tte_ev = df.loc[ev, tc].dropna()
    n_ev = len(tte_ev)
    if n_ev >= 8:
        qs = tte_ev.quantile([0.25, 0.5, 0.75]).values
        strat.loc[ev & df[tc].notna() & (df[tc] <= qs[0])]                   = 0
        strat.loc[ev & df[tc].notna() & (df[tc] > qs[0]) & (df[tc] <= qs[1])]= 1
        strat.loc[ev & df[tc].notna() & (df[tc] > qs[1]) & (df[tc] <= qs[2])]= 2
        strat.loc[ev & df[tc].notna() & (df[tc] > qs[2])]                    = 3
    else:
        strat.loc[ev & df[tc].notna()] = 0
    strat.loc[cen] = 4
    return strat, n_ev


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (done once — all figures share these objects)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60, flush=True)
print("Loading data …", flush=True)

cohort = json.load(open(COHORT_JSON)) if COHORT_JSON.exists() else {}
mod_n  = cohort.get("modality_n", {"Clinical": 3340, "CT_cells": 1835,
                                    "HE_cells": 632, "BAL_cells": 134})

splits = pd.read_csv(SPLITS_CSV)
splits["stem_int_str"] = splits["file"].str.replace(".pt", "", regex=False)
splits = splits.drop_duplicates("stem_int_str").copy()
# ACR binary: label=0 → A0B0 (ACR−), label=1 → A1/A2 (ACR+), NaN → missing (excluded)
splits["acr_binary"] = splits["label"].where(splits["label"].notna())
n_acr_pos = int(splits["acr_binary"].eq(1).sum())
n_acr_neg = int(splits["acr_binary"].eq(0).sum())
print(f"  splits={len(splits)}  acr+={n_acr_pos}  acr-={n_acr_neg}  "
      f"no_biopsy={int(splits['acr_binary'].isna().sum())}", flush=True)

# Clinical extracted from mil_v2 .pt files (106 features, NaN = missing)
CLIN_CSV   = Path("/home/aih/dinesh.haridoss/chicago_mil/results/cluster_proportions/clinical_features.csv")
CLIN_NAMES = Path("/home/aih/dinesh.haridoss/chicago_mil/results/cluster_proportions/clinical_feature_names.csv")

bal_prop  = pd.read_csv(CLUSTER_PROP_DIR / "bal_cluster_prop.csv")
bal_names = pd.read_csv(CLUSTER_PROP_DIR / "bal_cluster_names.csv")
he_prop   = pd.read_csv(CLUSTER_PROP_DIR / "he_cluster_prop.csv")
he_names  = pd.read_csv(CLUSTER_PROP_DIR / "he_cluster_names.csv")
ct_prop   = pd.read_csv(CLUSTER_PROP_DIR / "ct_cluster_prop.csv")
ct_names  = pd.read_csv(CLUSTER_PROP_DIR / "ct_cluster_names.csv")
he_clmap  = json.load(open(HE_CLMAP)) if HE_CLMAP.exists() else {}

bal_cluster_cols = [c for c in bal_prop.columns if c.startswith("cluster_")]
he_cluster_cols  = [c for c in he_prop.columns  if c.startswith("cluster_")]
ct_cluster_cols  = [c for c in ct_prop.columns  if c.startswith("cluster_")]

bal_cluster_labels = [
    bal_names.loc[bal_names["idx"] == int(c.split("_")[1]), "name"].iloc[0]
    if len(bal_names[bal_names["idx"] == int(c.split("_")[1])]) else f"C{c.split('_')[1]}"
    for c in bal_cluster_cols
]
ct_cluster_labels = [
    f"CT-{ct_names.loc[ct_names['idx']==int(c.split('_')[1]),'name'].iloc[0]}"
    if len(ct_names[ct_names["idx"]==int(c.split("_")[1])]) else f"CT-{c.split('_')[1]}"
    for c in ct_cluster_cols
]

he_subcluster_to_tissue = {
    int(r["idx"]): TISSUE_MERGE.get(he_clmap.get(str(r["name"]), "Unknown"), "Unknown")
    for _, r in he_names.iterrows()
}
for tt in TISSUE_ORDER:
    cols = [c for c in he_cluster_cols
            if he_subcluster_to_tissue.get(int(c.split("_")[1]), "Unknown") == tt]
    he_prop[f"tissue_{tt}"] = he_prop[cols].sum(axis=1) if cols else 0.0

he_tissue_cols   = [f"tissue_{tt}" for tt in TISSUE_ORDER]
he_tissue_labels = TISSUE_ORDER

def merge_splits(prop_df):
    prop_df = prop_df.copy()
    prop_df["stem_int_str"] = prop_df["stem"].astype(str).str.zfill(5)
    sp = splits[["stem_int_str", "acr_binary", "acr_days", "acr_status",
                  "clad_status", "clad_days", "death_status", "death_days",
                  "patient_id", "anchor_dt"]].copy()
    m = prop_df.merge(sp, on="stem_int_str", how="left", suffixes=("_p", "_s"))
    for col in ["acr_binary", "clad_status", "death_status"]:
        sc, pc = col + "_s", col + "_p"
        if sc in m.columns:
            m[col] = m[sc].where(m[sc].notna(), m.get(pc, np.nan))
        elif pc in m.columns:
            m.rename(columns={pc: col}, inplace=True)
    return m

bal = merge_splits(bal_prop)
he  = merge_splits(he_prop)
ct  = merge_splits(ct_prop)

# ── Clinical features ──────────────────────────────────────────────────────
# 106 raw features per sample extracted from mil_v2 .pt clinical_raw_tensor.
# NaN = missing (native, no encoding tricks). Features include:
#   0-8  : PFT (fvc, fev1, fev1_fvc, fev1p, fvcp,
#                delta_fvc, pseudoslope_fvc, delta_fev1, pseudoslope_fev1)
#   9-24 : Blood chemistry (albumin, sodium, glucose, creatinine, ALT, AST…)
#   25-73: CBC (absolute counts + differentials)
#   74-80: Vitals (BMI, BP, HR, SpO2, temp)
#   81-105: Immunology/transplant (DSA, CMV/EBV, BAL differentials,
#            donor risk, age, PGD grades, prev_tx)
clin_raw   = pd.read_csv(CLIN_CSV)
clin_names = pd.read_csv(CLIN_NAMES)
clin_feat_cols = [c for c in clin_raw.columns if c.startswith("feat_")]
clin_raw["stem_int_str"] = clin_raw["stem"].astype(str).str.zfill(5)
n_clin_feats = len(clin_feat_cols)
clin_name_map = {f"feat_{int(r['idx'])}": r["name"]
                 for _, r in clin_names.iterrows()}
clin_labels = [clin_name_map.get(c, c) for c in clin_feat_cols]

# Merge outcomes from splits
sp_clin = splits[["stem_int_str","acr_binary","acr_days","acr_status",
                   "clad_status","clad_days","death_status","death_days"]].copy()
clin = clin_raw.merge(sp_clin, on="stem_int_str", how="left")

# Feature category groups — updated for 106 features
CLIN_GROUPS = {
    "PFT":          [f"feat_{i}" for i in range(0, 9)],    # 0-8: lung function + deltas
    "Blood chem":   [f"feat_{i}" for i in range(9, 25)],   # 9-24
    "CBC":          [f"feat_{i}" for i in range(25, 74)],   # 25-73
    "Vitals":       [f"feat_{i}" for i in range(74, 81)],   # 74-80
    "Immunology":   [f"feat_{i}" for i in range(81, n_clin_feats)],
}
CLIN_GROUP_COLORS = {
    "PFT":        "#1565C0",
    "Blood chem": "#E65100",
    "CBC":        "#6A1B9A",
    "Vitals":     "#37474F",
    "Immunology": "#B71C1C",
}
clin_palette = {}
for grp, cols_g in CLIN_GROUPS.items():
    for c_g in cols_g:
        lbl = clin_name_map.get(c_g, c_g)
        clin_palette[lbl] = CLIN_GROUP_COLORS[grp]

print(f"  BAL={len(bal)}  HE={len(he)}  CT={len(ct)}  Clinical={len(clin)}", flush=True)

# Palettes
bal_palette = {lbl: cmap20(i % 20) for i, lbl in enumerate(bal_cluster_labels)}
he_palette  = TISSUE_COLORS
ct_palette  = {lbl: cmap20(i % 20) for i, lbl in enumerate(ct_cluster_labels)}

# Patient-level for cohort overview
pat_df = splits.groupby("patient_id").agg(
    n_tp      =("stem_int_str", "count"),
    n_acr_pos =("acr_binary",  lambda x: (x == 1).sum()),
    clad_st   =("clad_status", "max"),
    death_st  =("death_status","max"),
    clad_days =("clad_days",   "first"),
    death_days=("death_days",  "first"),
).reset_index()
pat_df["ever_acr"] = (pat_df["n_acr_pos"] > 0).astype(int)
pat_df = pat_df.sort_values(["ever_acr","n_acr_pos"],
                              ascending=[False, False]).reset_index(drop=True)
n_pat   = len(pat_df)
n_acr   = int(pat_df["ever_acr"].sum())
n_clad  = int(pat_df["clad_st"].sum())
n_death = int(pat_df["death_st"].sum())
n_tp    = len(splits)
n_tp_pos= int(splits["acr_binary"].eq(1).sum())

print(f"  n_pat={n_pat}  ever_acr+={n_acr}  clad={n_clad}  death={n_death}", flush=True)
print("Data loaded.", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# TASK DEFINITIONS  (shared across many figures)
# ══════════════════════════════════════════════════════════════════════════════
TASKS = [
    ("acr_binary",  "ACR",   "ACR+",   "ACR−",   ACR_POS,   ACR_NEG,   "acr_days",  "acr_status"),
    ("clad_status", "CLAD",  "CLAD+",  "CLAD−",  CLAD_COL,  ACR_NEG,  "clad_days", "clad_status"),
    ("death_status","Death", "Death+", "Death−", DEATH_COL, ACR_NEG,  "death_days","death_status"),
]

MODALITIES = [
    ("BAL scRNA", bal, bal_cluster_cols, bal_cluster_labels, bal_palette),
    ("H&E",       he,  he_tissue_cols,   he_tissue_labels,   he_palette),
    ("CT scan",   ct,  ct_cluster_cols,  ct_cluster_labels,  ct_palette),
    ("Clinical",  clin, clin_feat_cols,  clin_labels,        clin_palette),
]


# ══════════════════════════════════════════════════════════════════════════════
# FIG 0 — Project overview
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 0] Project overview …", flush=True)

fig = plt.figure(figsize=(22, 18))
gs_main = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45,
                            height_ratios=[1.6, 1.4, 1.2])

# --- Schematic row ---
ax_sc = fig.add_subplot(gs_main[0])
ax_sc.set_xlim(0, 22); ax_sc.set_ylim(-0.7, 5.5); ax_sc.axis("off")
ax_sc.set_title("(A) Multimodal MIL Framework — Lung Transplant Monitoring",
                fontsize=11, fontweight="bold", loc="left")

def _box(ax, x, y, w, h, lbl, sub, col, fs=8):
    ax.add_patch(plt.Rectangle((x-w/2, y-h/2), w, h,
                  fc=col, alpha=0.85, ec="white", lw=1.5, zorder=3, clip_on=False))
    ax.text(x, y+0.08, lbl, ha="center", va="center", fontsize=fs,
            fontweight="bold", color="white", zorder=4)
    if sub:
        ax.text(x, y-0.32, sub, ha="center", va="center", fontsize=fs-1,
                color="white", alpha=0.9, zorder=4)

def _arr(ax, x0, x1, y, col=GREY):
    ax.annotate("", xy=(x1-0.08, y), xytext=(x0+0.08, y),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.1), zorder=5)

nodes = [
    (1.1, 2.5, 1.8, 0.9, "Patient\ncohort",  f"n={n_pat}",     "#455A64"),
    (3.8, 2.5, 2.0, 0.9, "Serial\nbiopsies", f"n={n_tp} tps",  "#546E7A"),
    (7.2, 4.2, 1.7, 0.85,"H&E biopsy",      f"n={mod_n.get('HE_cells',632)}", MOD_COLORS["H&E"]),
    (7.2, 2.8, 1.7, 0.85,"BAL scRNA",       f"n={mod_n.get('BAL_cells',134)}",MOD_COLORS["BAL scRNA"]),
    (7.2, 1.4, 1.7, 0.85,"CT scan",         f"n={mod_n.get('CT_cells',1835)}",MOD_COLORS["CT scan"]),
    (7.2, 0.0, 1.7, 0.85,"Clinical labs",   f"n={mod_n.get('Clinical',3340)}",MOD_COLORS["Clinical"]),
    (11.5,4.2, 1.8, 0.85,"Tissue patches",  "k≈54 sub-clusters",  MOD_COLORS["H&E"]),
    (11.5,2.8, 1.8, 0.85,"scRNA cells",     "k=43 cell types",     MOD_COLORS["BAL scRNA"]),
    (11.5,1.4, 1.8, 0.85,"CT patches",      "k=37 clusters",       MOD_COLORS["CT scan"]),
    (11.5,0.0, 1.8, 0.85,"Lab values",      "106 features",        MOD_COLORS["Clinical"]),
]
for x, y, w, h, l, s, c in nodes:
    _box(ax_sc, x, y, w, h, l, s, c)

_arr(ax_sc, 2.0, 2.8, 2.5)
for ym in [4.2, 2.8, 1.4, 0.0]:
    ax_sc.annotate("", xy=(6.3, ym), xytext=(4.8, 2.5),
                   arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9), zorder=2)
    _arr(ax_sc, 8.1, 10.6, ym)
    _arr(ax_sc, 12.4, 14.75, ym)

_box(ax_sc, 16.0, 2.1, 2.5, 4.5, "MIL\nAggregation", "Shared Slot\nAttention", "#7B1FA2", 9)

for tx, ty, tl, tc in [
    (19.2, 4.2, "ACR\nclass.",  ACR_POS),
    (20.2, 3.0, "ACR\nTTE",    "#D81B60"),
    (19.2, 1.4, "CLAD\nTTE",   CLAD_COL),
    (20.2, 0.2, "Death\nTTE",  DEATH_COL),
]:
    _box(ax_sc, tx, ty, 1.5, 0.7, tl, "", tc, 7.5)
    _arr(ax_sc, 17.25, tx-0.75, ty)

for i, (mod, col) in enumerate(MOD_COLORS.items()):
    ax_sc.add_patch(plt.Rectangle((0.5+i*3.0, -0.65), 2.5, 0.35,
                    fc=col, alpha=0.8, ec="white", clip_on=False))
    ax_sc.text(0.5+i*3.0+1.25, -0.47, mod, ha="center", va="center",
               fontsize=7, color="white", fontweight="bold")

# --- Data overview row ---
gs2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[1],
                                        wspace=0.38, width_ratios=[2, 1, 1])
ax_avail = fig.add_subplot(gs2[0])
ax_cnt   = fig.add_subplot(gs2[1])
ax_stats = fig.add_subplot(gs2[2])

# Modality availability heatmap
has_he  = splits.groupby("patient_id")["has_HE"].max() if "has_HE" in splits.columns else pd.Series()
has_bal = splits.groupby("patient_id")["has_BAL"].max() if "has_BAL" in splits.columns else pd.Series()
has_ct  = splits.groupby("patient_id")["has_CT"].max() if "has_CT" in splits.columns else pd.Series()
has_cl  = splits.groupby("patient_id")["has_Clinical"].max() if "has_Clinical" in splits.columns else pd.Series()
pat_avail = pd.DataFrame({"H&E": has_he, "BAL": has_bal,
                           "CT": has_ct, "Clin": has_cl}).fillna(0).astype(int)
ever_acr_s = pat_df.set_index("patient_id")["ever_acr"].reindex(pat_avail.index).fillna(0)
n_acr_s    = pat_df.set_index("patient_id")["n_acr_pos"].reindex(pat_avail.index).fillna(0)
pat_avail_s = pat_avail.iloc[np.argsort(-(ever_acr_s.values*100 + n_acr_s.values))]
from matplotlib.colors import ListedColormap
cmap_mod = ListedColormap(["#ECEFF1", "#00897B"])
mod_mat = pat_avail_s.values.T.astype(float)
ax_avail.imshow(mod_mat, aspect="auto", cmap=cmap_mod, vmin=0, vmax=1, interpolation="nearest")
ax_avail.set_yticks(range(4)); ax_avail.set_yticklabels(["H&E","BAL","CT","Clin"], fontsize=8)
ax_avail.set_xlabel(f"Patients (n={len(pat_avail_s)}, sorted by ACR burden)", fontsize=8)
ax_avail.set_title("(B) Modality availability per patient", fontsize=9)
for i, lab in enumerate(["H&E","BAL","CT","Clin"]):
    ax_avail.text(len(pat_avail_s)+1, i, f"n={int(mod_mat[i].sum())}",
                  va="center", fontsize=7, color=DARK)
n_acr_pat = int(ever_acr_s.sum())
ax_avail.axvline(n_acr_pat-0.5, color=ACR_POS, lw=1.2, ls="--", alpha=0.8)
ax_avail.text(n_acr_pat/2, -0.8, f"ACR+ (n={n_acr})", ha="center", fontsize=7, color=ACR_POS)
ax_avail.text(n_acr_pat+(n_pat-n_acr)/2, -0.8, f"ACR− (n={n_pat-n_acr})",
              ha="center", fontsize=7, color=ACR_NEG)
ax_avail.spines["top"].set_visible(False); ax_avail.spines["right"].set_visible(False)
ax_avail.spines["bottom"].set_visible(False)

# Instance counts violin
cnt_data, cnt_labels, cnt_colors = [], [], []
for df_m, lbl, col in [(bal,"BAL\ncells",MOD_COLORS["BAL scRNA"]),
                        (he, "H&E\npatches",MOD_COLORS["H&E"]),
                        (ct, "CT\npatches",MOD_COLORS["CT scan"])]:
    if "n_instances" in df_m.columns:
        v = df_m["n_instances"].dropna().values
        if len(v): cnt_data.append(np.log10(v+1)); cnt_labels.append(lbl); cnt_colors.append(col)
if cnt_data:
    vp = ax_cnt.violinplot(cnt_data, positions=range(len(cnt_data)),
                           showmedians=True, showextrema=True, widths=0.65)
    for pc, c in zip(vp["bodies"], cnt_colors):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    for part in ["cmedians","cbars","cmins","cmaxes"]:
        if part in vp: vp[part].set_color(DARK); vp[part].set_linewidth(1.2)
    for xi, d in enumerate(cnt_data):
        med = np.median(d)
        ax_cnt.text(xi, med+0.05, f"{10**med:.0f}", ha="center", va="bottom",
                    fontsize=6.5, color=DARK)
ax_cnt.set_xticks(range(len(cnt_labels))); ax_cnt.set_xticklabels(cnt_labels, fontsize=7.5)
ax_cnt.set_ylabel("log₁₀(instance count)"); ax_cnt.set_title("(B2) Instances per sample")

# Cohort stats
ax_stats.axis("off"); ax_stats.set_title("(B3) Cohort summary")
stats = [
    ("Patients",          f"{n_pat}",                          DARK),
    ("  Ever ACR+",       f"{n_acr} ({100*n_acr/n_pat:.0f}%)",ACR_POS),
    ("  CLAD events",     f"{n_clad} ({100*n_clad/n_pat:.0f}%)",CLAD_COL),
    ("  Death events",    f"{n_death} ({100*n_death/n_pat:.0f}%)",DEATH_COL),
    ("──────────────",    "",                                  GREY),
    ("Timepoints total",  f"{n_tp}",                          DARK),
    ("  ACR+ (A1/A2)",    f"{n_tp_pos} ({100*n_tp_pos/n_tp:.0f}%)",ACR_POS),
    ("  ACR− (A0B0)",     f"{n_acr_neg} ({100*n_acr_neg/n_tp:.0f}%)",ACR_NEG),
    ("──────────────",    "",                                  GREY),
    ("H&E biopsies",      f"n={mod_n.get('HE_cells',632)}",   MOD_COLORS["H&E"]),
    ("BAL scRNA",         f"n={mod_n.get('BAL_cells',134)}",  MOD_COLORS["BAL scRNA"]),
    ("CT scans",          f"n={mod_n.get('CT_cells',1835)}",  MOD_COLORS["CT scan"]),
    ("BAL cell types",    "43 clusters",                       DARK),
    ("H&E tissue types",  "6 (54 sub-cl.)",                    DARK),
    ("CT clusters",       "37 clusters",                       DARK),
]
ax_stats.set_xlim(0,1); ax_stats.set_ylim(0,len(stats)+1)
for i, (lab, val, col) in enumerate(reversed(stats)):
    y = i+0.5
    ax_stats.text(0.02, y, lab, fontsize=7, color="#546E7A", va="center")
    ax_stats.text(0.98, y, val, fontsize=7.5, color=col, va="center",
                  ha="right", fontweight="bold" if "──" not in lab else "normal")

# --- Cluster composition row ---
gs3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[2], wspace=0.35)
ax_c1 = fig.add_subplot(gs3[0])
ax_c2 = fig.add_subplot(gs3[1])
ax_c3 = fig.add_subplot(gs3[2])

for ax_c, df_m, c_cols, c_labels, pal, n_top, title in [
    (ax_c1, bal, bal_cluster_cols, bal_cluster_labels, bal_palette, 12,
     "(C1) BAL scRNA cell types\n(ACR+ vs ACR−)"),
    (ax_c2, he,  he_tissue_cols,   he_tissue_labels,   he_palette,
     len(he_tissue_cols), "(C2) H&E tissue types\n(ACR+ vs ACR−)"),
    (ax_c3, ct,  ct_cluster_cols,  ct_cluster_labels,  ct_palette, 10,
     "(C3) CT clusters\n(ACR+ vs ACR−)"),
]:
    tmp, gc_ = prep_group(df_m, "acr_binary", c_cols)
    bar_comparison(ax_c, tmp, c_cols, c_labels, pal, gc_, "ACR+", "ACR−", title, n_top)

fig.suptitle("Figure 0 — Project Overview: Multimodal MIL for Lung Transplant Monitoring",
             fontsize=12, fontweight="bold", y=1.01)
save_fig(fig, "fig0_project_overview")
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 0b — Serial patient timeline
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 0b] Serial patient timeline …", flush=True)

pid_col_he  = "patient_id_s"  if "patient_id_s"  in he.columns else "patient_id"
dte_col_he  = "anchor_dt_s"   if "anchor_dt_s"   in he.columns else "anchor_dt_p"
clad_sc     = "clad_status_s" if "clad_status_s" in he.columns else "clad_status"
death_sc    = "death_status_s"if "death_status_s"in he.columns else "death_status"
acr_bc      = "acr_binary"

if pid_col_he in he.columns:
    n_tp_he = he.groupby(pid_col_he).size()
    serial_pats = n_tp_he[n_tp_he >= 2].index.tolist()
    serial_pats = sorted(serial_pats,
                         key=lambda p: (-he[he[pid_col_he]==p][clad_sc].max()
                                        if clad_sc in he.columns else 0,
                                        -len(he[he[pid_col_he]==p])))[:30]

    fig2, ax2 = plt.subplots(figsize=(16, max(8, len(serial_pats)*0.55)))
    ax2.set_title("Figure 0b — Serial H&E Tissue Composition per Patient\n"
                  "Circle shade = Alveolar-inflamed proportion; red fill = ACR+ biopsy",
                  fontsize=10, fontweight="bold")
    yi = 0
    infl_col = "tissue_Alveolar inflamed"
    for pid in serial_pats:
        grp = he[he[pid_col_he]==pid].copy()
        if dte_col_he in grp.columns:
            grp = grp.sort_values(dte_col_he)
        clad_flag  = int(grp[clad_sc].max()) if clad_sc in grp.columns else 0
        death_flag = int(grp[death_sc].max()) if death_sc in grp.columns else 0
        for ti in range(len(grp)):
            infl = float(grp[infl_col].iloc[ti]) if infl_col in grp.columns else 0
            acr  = int(grp[acr_bc].iloc[ti]) if acr_bc in grp.columns and not pd.isna(grp[acr_bc].iloc[ti]) else -1
            col  = ACR_POS if acr == 1 else (ACR_NEG if acr == 0 else "#CFD8DC")
            ax2.add_patch(plt.Circle((ti*1.2, yi), 0.42, color=col,
                                     alpha=max(0.2, 0.3+0.7*infl), zorder=3))
            ax2.text(ti*1.2, yi, f"{infl*100:.0f}", ha="center", va="center",
                     fontsize=4.5, color="white" if acr==1 else DARK, zorder=4)
        n_tp_p = len(grp)
        ax2.text(-1.5, yi, pid, ha="right", va="center", fontsize=6, color=DARK)
        if clad_flag:
            ax2.text(n_tp_p*1.2+0.1, yi, "CLAD", ha="left", va="center",
                     fontsize=5.5, color=CLAD_COL, fontweight="bold")
        if death_flag:
            ax2.text(n_tp_p*1.2+0.9, yi, "†", ha="left", va="center",
                     fontsize=9, color=DEATH_COL)
        yi -= 1.3
    max_tps = max((len(he[he[pid_col_he]==p]) for p in serial_pats), default=1)
    ax2.set_xlim(-2, max_tps*1.2+2); ax2.set_ylim(yi, 1.5)
    ax2.set_xlabel("Visit index (chronological)"); ax2.set_yticks([])
    ax2.legend(handles=[mpatches.Patch(color=ACR_POS, alpha=0.7, label="ACR+ (A1/A2)"),
                         mpatches.Patch(color=ACR_NEG, alpha=0.7, label="ACR− (A0B0)"),
                         mpatches.Patch(color="#CFD8DC", alpha=0.7, label="No biopsy")],
               frameon=False, fontsize=7, loc="lower right")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    save_fig(fig2, "fig0b_serial_patient_timeline")
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# FIGS 7–9 — Cluster proportions × task, one figure per modality
# Layout: 4 rows (ACR, CLAD, Death, TTE-quartile) × 2 cols (bar, violin)
# ══════════════════════════════════════════════════════════════════════════════
for fig_num, (mod_name, df_m, c_cols, c_labels, pal) in zip([7, 8, 9, "9b"], MODALITIES):
    print(f"\n[Fig {fig_num}] {mod_name} cluster proportions by task …", flush=True)
    n_top_bar   = min(15, len(c_cols))
    n_top_viol  = min(8,  len(c_cols))

    fig, axes = plt.subplots(4, 2, figsize=(16, 22))
    fig.suptitle(f"Figure {fig_num} — {mod_name} Cluster Proportions Across Tasks",
                 fontsize=11, fontweight="bold", y=1.002)

    for row_i, (gcb, task_name, pos_lbl, neg_lbl, pos_c, neg_c, tte_c, st_c) in enumerate(TASKS):
        tmp, gc_ = prep_group(df_m, gcb, c_cols)
        ax_b = axes[row_i, 0]; ax_v = axes[row_i, 1]
        bar_comparison(ax_b, tmp, c_cols, c_labels, pal, gc_,
                       pos_lbl, neg_lbl,
                       f"(a{row_i+1}) {task_name} — proportions", n_top_bar)
        violin_clusters(ax_v, tmp, c_cols, c_labels, gc_, pos_c, neg_c,
                        pos_lbl, neg_lbl,
                        f"(b{row_i+1}) {task_name} — top differential", n_top_viol)

    # Row 4: TTE-quartile + Spearman correlation
    ax_tte  = axes[3, 0]; ax_cor = axes[3, 1]
    # Use first endpoint with sufficient events for the TTE row
    strat_col = None
    for gcb, task_name, _, _, _, _, tte_c, st_c in TASKS:
        strat, n_ev = assign_tte_strata(df_m, tte_c, st_c)
        if n_ev >= 8:
            strat_col = strat; tte_col_use = tte_c; ep_name = task_name; break
    if strat_col is not None:
        tmp_tte = df_m[c_cols].copy(); tmp_tte["stratum"] = strat_col.values
        tmp_tte = tmp_tte[tmp_tte["stratum"] >= 0]
        valid_s = [s for s in range(5) if (tmp_tte["stratum"]==s).sum() >= 3]
        mean_all = df_m[c_cols].mean()
        top_cols  = mean_all.nlargest(n_top_bar).index.tolist()
        top_lbl   = [c_labels[c_cols.index(c)] for c in top_cols]
        top_col_c = [pal.get(l, GREY) for l in top_lbl]
        x = np.arange(len(top_cols)); w = 0.22
        gi_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        for gi, (s_val, s_lbl, s_col) in enumerate(
                zip([0,3,4], ["Q1 earliest","Q4 latest","Censored"],
                    ["#B71C1C","#90CAF9","#B0BEC5"])):
            sub = tmp_tte[tmp_tte["stratum"]==s_val]
            if len(sub) < 3: continue
            for ci, col in enumerate(top_cols):
                ax_tte.bar(ci+(gi-1)*w, sub[col].mean()*100, w,
                           color=top_col_c[ci], alpha=[0.9,0.5,0.3][gi], linewidth=0)
        ax_tte.set_xticks(x)
        ax_tte.set_xticklabels([l[:14] for l in top_lbl], rotation=40, ha="right", fontsize=5.5)
        ax_tte.set_ylabel("Mean proportion (%)")
        ax_tte.set_title(f"(a4) {ep_name} TTE — Q1 vs Q4 vs Censored")
        ax_tte.legend(handles=[mpatches.Patch(color="#B71C1C", alpha=0.9, label="Q1 earliest"),
                                mpatches.Patch(color="#90CAF9", alpha=0.5, label="Q4 latest"),
                                mpatches.Patch(color="#B0BEC5", alpha=0.3, label="Censored")],
                      frameon=False, fontsize=6.5)
        ax_tte.spines["top"].set_visible(False); ax_tte.spines["right"].set_visible(False)

        # Spearman correlation bar
        tte_col_name = get_col(df_m, tte_col_use)
        st_col_name  = get_col(df_m, st_c if 'st_c' in dir() else "acr_status")
        if tte_col_name in df_m.columns:
            ev_tmp = df_m[(df_m.get(st_col_name, pd.Series(0, index=df_m.index)) == 1) &
                          df_m[tte_col_name].notna()]
            if len(ev_tmp) >= 10:
                corrs = []
                for col, lbl in zip(c_cols, c_labels):
                    v = ev_tmp[col].values
                    if v.std() > 0:
                        r, p = spearmanr(ev_tmp[tte_col_name].values, v)
                        corrs.append((lbl, r, p))
                corrs.sort(key=lambda x: abs(x[1]), reverse=True)
                top_c = corrs[:min(10, len(corrs))]
                rs  = [c[1] for c in top_c]
                lbs = [c[0] for c in top_c]
                ax_cor.barh(np.arange(len(rs)), rs,
                            color=[ACR_POS if r<0 else ACR_NEG for r in rs], alpha=0.8)
                ax_cor.set_yticks(np.arange(len(rs)))
                ax_cor.set_yticklabels(lbs, fontsize=6)
                ax_cor.axvline(0, color=GREY, lw=0.8, ls="--")
                ax_cor.set_xlabel(f"Spearman r (TTE-{ep_name})")
                ax_cor.set_title(f"(b4) Correlation with TTE-{ep_name}")
                ax_cor.text(0.98, 0.02, "red=shorter TTE", transform=ax_cor.transAxes,
                            ha="right", fontsize=6, color=ACR_POS)
                ax_cor.spines["top"].set_visible(False); ax_cor.spines["right"].set_visible(False)
    else:
        for ax in [ax_tte, ax_cor]:
            ax.text(0.5, 0.5, "Insufficient TTE data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color=GREY)

    plt.tight_layout(rect=[0,0,1,1])
    save_fig(fig, f"fig{fig_num}_{mod_name.replace(' ','_').lower()}_by_task")
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Cross-modal enrichment heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 10] Cross-modal enrichment heatmap …", flush=True)

def enrichment_df(df_m, c_cols, c_labels, top_n=12):
    mean_all = df_m[c_cols].mean()
    top_cols  = mean_all.nlargest(top_n).index.tolist()
    top_lbl   = [c_labels[c_cols.index(c)] for c in top_cols]
    rows = []
    for col, lbl in zip(top_cols, top_lbl):
        row = {"cluster": lbl}
        for gcb, tn, _, _, _, _, _, _ in TASKS:
            tmp, gc_ = prep_group(df_m, gcb, [col])
            pos = tmp[tmp[gc_]==1][col].dropna().values
            neg = tmp[tmp[gc_]==0][col].dropna().values
            if len(pos)>=3 and len(neg)>=3:
                lfc = np.log2((pos.mean()+1e-6)/(neg.mean()+1e-6))
                row[f"{tn}_lfc"] = lfc
                row[f"{tn}_p"]   = mw_test(pos, neg)
            else:
                row[f"{tn}_lfc"] = np.nan
                row[f"{tn}_p"]   = 1.0
        rows.append(row)
    return pd.DataFrame(rows)

dfs_enrich = []
for mod_name, df_m, c_cols, c_labels, pal in MODALITIES:
    e = enrichment_df(df_m, c_cols, c_labels)
    e["modality"] = mod_name[:3]
    e["row_label"] = e["modality"] + " | " + e["cluster"].str[:20]
    dfs_enrich.append(e)
all_enrich = pd.concat(dfs_enrich, ignore_index=True)

task_names = [t[1] for t in TASKS]
lfc_cols   = [f"{t}_lfc" for t in task_names]
p_cols     = [f"{t}_p"   for t in task_names]

mat = all_enrich[lfc_cols].values
_, pfdr, _, _ = multipletests(np.where(np.isnan(all_enrich[p_cols].values.ravel()),
                                        1, all_enrich[p_cols].values.ravel()), method="fdr_bh")
sig = (pfdr < 0.05).reshape(mat.shape)
row_labels = all_enrich["row_label"].tolist()
order = np.argsort(np.nanmax(np.abs(mat), axis=1))[::-1]
mat = mat[order]; sig = sig[order]
row_labels = [row_labels[i] for i in order]

vmax = np.nanpercentile(np.abs(mat), 95) or 1.0
fig, ax = plt.subplots(figsize=(8, max(10, len(row_labels)*0.32)))
im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
for ri in range(len(row_labels)):
    for ci in range(len(task_names)):
        if sig[ri,ci] and not np.isnan(mat[ri,ci]):
            ax.text(ci, ri, "*", ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
ax.set_xticks(range(len(task_names)))
ax.set_xticklabels([f"{t} (+/−)" for t in task_names], fontsize=9)
ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=6.5)
ax.set_title("Figure 10 — Multi-modal Cluster Enrichment (log₂FC pos/neg; * FDR<5%)",
             fontsize=9, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="log₂FC", shrink=0.6)
plt.tight_layout()
save_fig(fig, "fig10_crossmodal_enrichment_heatmap")
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 11 — Compact stacked-bar summary (all modalities × all tasks)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 11] Compact stacked-bar summary …", flush=True)

fig, axes = plt.subplots(3, 3, figsize=(20, 14))
fig.suptitle("Figure 11 — Cluster Proportions Summary: All Modalities × All Tasks",
             fontsize=11, fontweight="bold")

for col_i, (mod_name, df_m, c_cols, c_labels, pal) in enumerate(MODALITIES):
    n_top = min(15, len(c_cols))
    mean_all = df_m[c_cols].mean()
    top_cols  = mean_all.nlargest(n_top).index.tolist()
    top_lbl   = [c_labels[c_cols.index(c)] for c in top_cols]
    col_c_list= [pal.get(l, GREY) for l in top_lbl]

    for row_i, (gcb, tn, pos_lbl, neg_lbl, _, _, _, _) in enumerate(TASKS):
        ax = axes[row_i, col_i]
        tmp, gc_ = prep_group(df_m, gcb, top_cols)
        groups = [(1, pos_lbl, 0.85), (0, neg_lbl, 0.5)]
        for yi, (gv, gl, al) in enumerate(groups):
            sub = tmp[tmp[gc_]==gv]
            left = 0.0
            for ci, (col, col_c) in enumerate(zip(top_cols, col_c_list)):
                m = sub[col].mean() if len(sub) > 0 else 0
                ax.barh(yi, m*100, left=left*100, color=col_c, alpha=al, linewidth=0)
                left += m
        ax.set_yticks([0,1])
        ax.set_yticklabels([f"{pos_lbl} (n={int((tmp[gc_]==1).sum())})",
                             f"{neg_lbl} (n={int((tmp[gc_]==0).sum())})"], fontsize=7)
        ax.set_xlabel("Proportion (%)")
        ax.set_xlim(0, 100)
        if row_i == 0: ax.set_title(mod_name, fontsize=9, fontweight="bold")
        if col_i == 0: ax.set_ylabel(tn, fontsize=8, fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    handles = [mpatches.Patch(color=c, label=l[:22]) for c, l in zip(col_c_list, top_lbl)]
    axes[2, col_i].legend(handles=handles, frameon=False, fontsize=5.5,
                           loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=3)
plt.tight_layout()
save_fig(fig, "fig11_stacked_summary_all_tasks")
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 12 — Volcano plots
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 12] Volcano plots …", flush=True)

fig, axes = plt.subplots(3, 3, figsize=(17, 16))
fig.suptitle("Figure 12 — Volcano: Differential Cluster Abundance per Modality × Task",
             fontsize=11, fontweight="bold")

for col_i, (mod_name, df_m, c_cols, c_labels, pal) in enumerate(MODALITIES):
    for row_i, (gcb, tn, _, _, task_col, _, _, _) in enumerate(TASKS):
        ax = axes[row_i, col_i]
        tmp, gc_ = prep_group(df_m, gcb, c_cols)
        pos = tmp[tmp[gc_]==1]; neg = tmp[tmp[gc_]==0]
        if len(pos)<3 or len(neg)<3:
            ax.text(0.5,0.5,f"n+={len(pos)},n−={len(neg)}\nInsufficient data",
                    ha="center",va="center",transform=ax.transAxes,fontsize=8,color=GREY)
            ax.set_title(f"{mod_name} | {tn}"); continue
        lfcs, pvals, labs = [], [], []
        for col, lbl in zip(c_cols, c_labels):
            pm = pos[col].mean(); nm = neg[col].mean()
            lfcs.append(np.log2((pm+1e-6)/(nm+1e-6)))
            pvals.append(mw_test(pos[col].dropna().values, neg[col].dropna().values))
            labs.append(lbl)
        _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
        neg_lq = -np.log10(np.maximum(qvals, 1e-10))
        lfcs = np.array(lfcs); neg_lq = np.array(neg_lq); qvals = np.array(qvals)
        cols_v = [ACR_POS if q<0.05 and l>0.5 else
                  ACR_NEG if q<0.05 and l<-0.5 else GREY
                  for l, q in zip(lfcs, qvals)]
        ax.scatter(lfcs, neg_lq, c=cols_v, s=np.where(qvals<0.05,25,10),
                   alpha=0.75, linewidths=0, zorder=3)
        ax.axhline(-np.log10(0.05), color=GREY, lw=0.8, ls="--", alpha=0.7)
        ax.axvline(0.5,  color=ACR_POS, lw=0.7, ls=":", alpha=0.6)
        ax.axvline(-0.5, color=ACR_NEG, lw=0.7, ls=":", alpha=0.6)
        for ti in np.argsort(neg_lq)[::-1][:5]:
            if qvals[ti] < 0.05:
                ax.annotate(labs[ti][:16], (lfcs[ti], neg_lq[ti]),
                            xytext=(3,3), textcoords="offset points", fontsize=5, color=DARK)
        n_up   = int(((qvals<0.05)&(lfcs>0.5)).sum())
        n_down = int(((qvals<0.05)&(lfcs<-0.5)).sum())
        ax.set_xlabel("log₂FC (pos/neg)", fontsize=7)
        ax.set_ylabel("−log₁₀(FDR q)", fontsize=7)
        ax.set_title(f"{mod_name} | {tn}\n↑{n_up} ↓{n_down} (FDR<5%, |LFC|>0.5)", fontsize=8)
        ax.text(0.01,0.99,f"n+={len(pos)}, n−={len(neg)}",
                transform=ax.transAxes,fontsize=6.5,va="top",color=DARK)
plt.tight_layout()
save_fig(fig, "fig12_volcano_differential")
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 13 — Serial trajectory heatmaps  (BAL + H&E)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 13] Serial trajectory heatmaps …", flush=True)

for mod_name, df_m, c_cols, c_labels, pal in MODALITIES[:2]:
    pid_c = "patient_id_s" if "patient_id_s" in df_m.columns else "patient_id"
    dte_c = "anchor_dt_s"  if "anchor_dt_s"  in df_m.columns else None
    if pid_c not in df_m.columns: continue
    var_ = df_m[c_cols].var()
    top6  = var_.nlargest(6).index.tolist()
    top6l = [c_labels[c_cols.index(c)] for c in top6]
    pat_g = df_m.groupby(pid_c)
    valid_pats = [p for p, g in pat_g if len(g) >= 2][:40]
    if not valid_pats: continue
    max_tp = max(len(pat_g.get_group(p)) for p in valid_pats)
    n_pats = len(valid_pats)
    acr_fl = np.zeros(n_pats)
    mats   = {l: np.full((n_pats, max_tp), np.nan) for l in top6l}
    for pi, pid in enumerate(valid_pats):
        g = df_m[df_m[pid_c]==pid].copy()
        if dte_c: g = g.sort_values(dte_c)
        acr_fl[pi] = int(g["acr_binary"].eq(1).any()) if "acr_binary" in g.columns else 0
        for col, lbl in zip(top6, top6l):
            v = g[col].values[:max_tp]
            mats[lbl][pi, :len(v)] = v
    sort_ord = np.argsort(-acr_fl)
    fig, axes13 = plt.subplots(1, len(top6), figsize=(4*len(top6), 7))
    fig.suptitle(f"Figure 13 — {mod_name} Serial Trajectories per Patient",
                 fontsize=10, fontweight="bold")
    for ci, (lbl, ax13) in enumerate(zip(top6l, axes13)):
        m = mats[lbl][sort_ord]
        a = acr_fl[sort_ord]
        im13 = ax13.imshow(m, aspect="auto", cmap="RdBu_r",
                           vmin=0, vmax=np.nanpercentile(m, 95) or 0.1,
                           interpolation="nearest")
        ax13.set_title(lbl[:20], fontsize=7.5, fontweight="bold")
        ax13.set_xlabel("Timepoint"); ax13.set_yticks([])
        if ci == 0: ax13.set_ylabel("Patient (sorted by ACR burden)")
        n_acr_traj = int(a.sum())
        ax13.axhline(n_acr_traj-0.5, color="white", lw=1.2, ls="--")
        plt.colorbar(im13, ax=ax13, fraction=0.04, pad=0.02, shrink=0.7, label="Proportion")
    plt.tight_layout()
    save_fig(fig, f"fig13_serial_{mod_name.replace(' ','_').lower()}")
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 14 — BAL cell-type stacked area over time × task
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 14] BAL cell-type over time …", flush=True)

if "clad_days" in bal.columns:
    bal_t = bal.copy()
    bal_t["tte_bin"] = pd.cut(
        bal_t["clad_days"].fillna(0),
        bins=[0, 180, 365, 730, 1e6],
        labels=["0–6 mo","6–12 mo","12–24 mo",">24 mo"], right=True)
    top_n_area = min(10, len(bal_cluster_cols))
    top_bal_a  = bal[bal_cluster_cols].mean().nlargest(top_n_area).index.tolist()
    top_bal_al = [bal_cluster_labels[bal_cluster_cols.index(c)] for c in top_bal_a]
    colors_area = [cmap20(i % 20) for i in range(top_n_area)]
    bins_lab = ["0–6 mo","6–12 mo","12–24 mo",">24 mo"]
    fig, axes14 = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Figure 14 — BAL Cell-Type Proportions Over Time Post-Transplant",
                 fontsize=10, fontweight="bold")
    for ri, (gcb, tn, pos_lbl, neg_lbl, _, _, _, _) in enumerate(TASKS):
        gc_ = get_col(bal_t, gcb)
        tmp14 = bal_t[[gc_, "tte_bin"] + top_bal_a].copy()
        tmp14[gc_] = tmp14[gc_].where(tmp14[gc_].notna())
        for cj, (gv, gl) in enumerate([(1, pos_lbl), (0, neg_lbl)]):
            ax14 = axes14[ri, cj]
            sub = tmp14[tmp14[gc_]==gv]
            if len(sub) < 3:
                ax14.text(0.5,0.5,f"n={len(sub)}\nInsufficient",
                          ha="center",va="center",transform=ax14.transAxes,fontsize=8,color=GREY)
                continue
            stats14 = sub.groupby("tte_bin", observed=True)[top_bal_a].mean()
            stats14 = stats14.reindex(bins_lab)
            bottom = np.zeros(len(bins_lab))
            for ci14, (col, col_c) in enumerate(zip(top_bal_a, colors_area)):
                vals = stats14[col].fillna(0).values * 100
                ax14.bar(np.arange(len(bins_lab)), vals, bottom=bottom,
                         color=col_c, alpha=0.85, linewidth=0, label=top_bal_al[ci14][:18])
                bottom += vals
            ax14.set_xticks(np.arange(len(bins_lab)))
            ax14.set_xticklabels(bins_lab, fontsize=7)
            ax14.set_ylabel("Mean proportion (%)")
            ax14.set_title(f"{tn} | {gl} (n={len(sub)})")
            ax14.spines["top"].set_visible(False); ax14.spines["right"].set_visible(False)
            if cj == 1 and ri == 0:
                ax14.legend(frameon=False, fontsize=5, loc="upper right",
                            bbox_to_anchor=(1.4,1), ncol=1)
    plt.tight_layout()
    save_fig(fig, "fig14_bal_celltype_over_time")
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 15 — BAL × H&E cross-modal correlation
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 15] BAL × H&E correlation …", flush=True)

both_bh = set(bal["stem_int_str"].astype(str)) & set(he["stem_int_str"].astype(str))
print(f"  BAL+HE paired: {len(both_bh)}", flush=True)
if len(both_bh) >= 5:
    bal_bh = bal[bal["stem_int_str"].astype(str).isin(both_bh)].set_index("stem_int_str")
    he_bh  = he[he["stem_int_str"].astype(str).isin(both_bh)].set_index("stem_int_str")
    shared = sorted(both_bh)
    n_bal_top = min(12, len(bal_cluster_cols))
    top_bal_c  = bal[bal_cluster_cols].mean().nlargest(n_bal_top).index.tolist()
    top_bal_l  = [bal_cluster_labels[bal_cluster_cols.index(c)] for c in top_bal_c]
    corr_mat = np.full((len(he_tissue_cols), n_bal_top), np.nan)
    pval_mat = np.ones_like(corr_mat)
    for hi, hc in enumerate(he_tissue_cols):
        for bi, bc in enumerate(top_bal_c):
            hv = he_bh.reindex(shared)[hc].values.astype(float)
            bv = bal_bh.reindex(shared)[bc].values.astype(float)
            mask = ~(np.isnan(hv)|np.isnan(bv))
            if mask.sum() >= 5:
                r, p = spearmanr(hv[mask], bv[mask])
                corr_mat[hi,bi]=r; pval_mat[hi,bi]=p
    _, pq, _, _ = multipletests(pval_mat.ravel(), method="fdr_bh")
    pq = pq.reshape(pval_mat.shape)
    vmax15 = np.nanpercentile(np.abs(corr_mat),95) or 0.5
    fig, ax15 = plt.subplots(figsize=(max(8,n_bal_top*0.7), max(5,len(he_tissue_cols)*0.8)))
    im15 = ax15.imshow(corr_mat, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax15, vmax=vmax15, interpolation="nearest")
    for hi in range(len(he_tissue_cols)):
        for bi in range(n_bal_top):
            if pq[hi,bi]<0.05 and not np.isnan(corr_mat[hi,bi]):
                ax15.text(bi, hi, "*", ha="center", va="center", fontsize=9,
                          color="white", fontweight="bold")
    ax15.set_xticks(range(n_bal_top)); ax15.set_xticklabels(top_bal_l, rotation=40, ha="right", fontsize=7)
    ax15.set_yticks(range(len(he_tissue_cols))); ax15.set_yticklabels(he_tissue_labels, fontsize=8)
    ax15.set_title(f"Figure 15 — BAL × H&E Cross-Modal Correlation\n"
                   f"(Spearman r, n={len(both_bh)} paired; * FDR<5%)",
                   fontsize=9, fontweight="bold")
    plt.colorbar(im15, ax=ax15, fraction=0.025, pad=0.02, label="Spearman r")
    plt.tight_layout()
    save_fig(fig, "fig15_crossmodal_bal_he")
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 16 — ACR episode cluster fingerprint
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 16] ACR episode fingerprint …", flush=True)

for mod_name, df_m, c_cols, c_labels, pal in MODALITIES[:2]:
    pid_c = "patient_id_s" if "patient_id_s" in df_m.columns else "patient_id"
    dte_c = "anchor_dt_s"  if "anchor_dt_s"  in df_m.columns else None
    if pid_c not in df_m.columns or "acr_binary" not in df_m.columns: continue
    tmp16 = df_m[[pid_c,"acr_binary"]+c_cols].copy()
    if dte_c: tmp16 = tmp16.merge(df_m[[dte_c]], left_index=True, right_index=True)
    acr_r, pre_r = [], []
    for pid, grp in tmp16.groupby(pid_c):
        if dte_c and dte_c in grp.columns:
            grp = grp.sort_values(dte_c)
        grp = grp.reset_index(drop=True)
        acr_idx = [i for i in grp.index if grp.loc[i,"acr_binary"] == 1]
        for ai in acr_idx:
            if ai > 0 and grp.loc[ai-1,"acr_binary"] == 0:
                acr_r.append(grp.loc[ai,  c_cols].values.astype(float))
                pre_r.append(grp.loc[ai-1,c_cols].values.astype(float))
    if len(acr_r) < 5:
        print(f"  [skip Fig16 {mod_name}] only {len(acr_r)} pairs", flush=True); continue
    delta   = np.array(acr_r) - np.array(pre_r)
    means   = delta.mean(0); sems = delta.std(0)/np.sqrt(len(delta))
    order16 = np.argsort(np.abs(means))[::-1]
    means_s = means[order16]; sems_s = sems[order16]
    lbl_s   = [c_labels[i] for i in order16]
    fig, ax16 = plt.subplots(figsize=(7, max(5, len(lbl_s)*0.42)))
    ax16.barh(np.arange(len(lbl_s)), means_s*100, xerr=sems_s*100,
              color=[ACR_POS if m>0 else ACR_NEG for m in means_s],
              alpha=0.8, linewidth=0,
              error_kw={"ecolor":DARK,"elinewidth":0.8,"capsize":2})
    ax16.axvline(0, color=DARK, lw=0.8)
    ax16.set_yticks(np.arange(len(lbl_s))); ax16.set_yticklabels(lbl_s, fontsize=7)
    ax16.set_xlabel("Mean Δ proportion at ACR event (%)")
    ax16.set_title(f"Figure 16 — {mod_name} Fingerprint at ACR Episode\n"
                   f"(n={len(acr_r)} paired ACR+/ACR− consecutive visits)",
                   fontsize=9, fontweight="bold")
    ax16.spines["top"].set_visible(False); ax16.spines["right"].set_visible(False)
    ax16.text(0.99,0.01,"↑ increased at ACR",transform=ax16.transAxes,
              ha="right",va="bottom",fontsize=7,color=ACR_POS)
    ax16.text(0.01,0.01,"↓ decreased at ACR",transform=ax16.transAxes,
              ha="left",va="bottom",fontsize=7,color=ACR_NEG)
    plt.tight_layout()
    save_fig(fig, f"fig16_acr_fingerprint_{mod_name.replace(' ','_').lower()}")
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 17 — Cluster-stratified KM curves
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 17] Cluster-stratified KM curves …", flush=True)

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    tmp17, gc17 = prep_group(bal, "acr_binary", bal_cluster_cols)
    pos17 = tmp17[tmp17[gc17]==1]; neg17 = tmp17[tmp17[gc17]==0]
    lfcs17 = {c: abs(np.log2((pos17[c].mean()+1e-6)/(neg17[c].mean()+1e-6)))
              for c in bal_cluster_cols}
    top3 = sorted(lfcs17, key=lfcs17.get, reverse=True)[:3]
    top3l = [bal_cluster_labels[bal_cluster_cols.index(c)] for c in top3]
    KM_EP = [("clad_days","clad_status","CLAD-free",CLAD_COL),
             ("death_days","death_status","Overall",  DEATH_COL)]
    fig, axes17 = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle("Figure 17 — BAL Cluster-Stratified Survival (high vs low tertile)",
                 fontsize=10, fontweight="bold")
    for ri, (col17, lbl17) in enumerate(zip(top3, top3l)):
        q33 = bal[col17].quantile(0.33); q67 = bal[col17].quantile(0.67)
        for ci, (dc, sc, et, ec) in enumerate(KM_EP):
            ax17 = axes17[ri,ci]
            dc_c = get_col(bal, dc); sc_c = get_col(bal, sc)
            if dc_c not in bal.columns: continue
            v17 = bal[[dc_c, sc_c, col17]].dropna()
            v17 = v17[v17[dc_c]>0]
            hi = v17[v17[col17]>=q67]; lo = v17[v17[col17]<=q33]
            for gdf, gn, gc_col in [(hi,f"High (n={len(hi)})",ACR_POS),
                                     (lo,f"Low  (n={len(lo)})",ACR_NEG)]:
                if len(gdf)<5: continue
                KaplanMeierFitter().fit(
                    gdf[dc_c]/365.25, gdf[sc_c], label=gn
                ).plot_survival_function(ax=ax17, color=gc_col, ci_show=True,
                                          ci_alpha=0.12, linewidth=2)
            if len(hi)>=5 and len(lo)>=5:
                res17 = logrank_test(hi[dc_c], lo[dc_c],
                                      hi[sc_c], lo[sc_c])
                p17 = res17.p_value
                ax17.text(0.97,0.55,f"log-rank p={p17:.3f} {pval_stars(p17)}",
                          transform=ax17.transAxes,ha="right",fontsize=7.5,
                          bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=GREY,alpha=0.8))
            ax17.set_xlabel("Years post-transplant"); ax17.set_ylabel("Survival probability")
            ax17.set_ylim(0,1.05); ax17.set_title(f"{lbl17[:20]}\n{et} survival", fontsize=8)
            ax17.legend(frameon=False, fontsize=7)
            ax17.spines["top"].set_visible(False); ax17.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "fig17_km_cluster_stratified")
    gc.collect()
except ImportError:
    print("  lifelines not installed — skip Fig17", flush=True)
except Exception as e:
    print(f"  [Fig17 ERROR] {e}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 18 — BAL × CT cross-modal correlation
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 18] BAL × CT correlation …", flush=True)

both_bc = set(bal["stem_int_str"].astype(str)) & set(ct["stem_int_str"].astype(str))
print(f"  BAL+CT paired: {len(both_bc)}", flush=True)
if len(both_bc) >= 5:
    bal_bc = bal[bal["stem_int_str"].astype(str).isin(both_bc)].set_index("stem_int_str")
    ct_bc  = ct[ct["stem_int_str"].astype(str).isin(both_bc)].set_index("stem_int_str")
    shared_bc = sorted(both_bc)
    n_bal18 = min(10, len(bal_cluster_cols)); n_ct18 = min(10, len(ct_cluster_cols))
    top_b18 = bal[bal_cluster_cols].mean().nlargest(n_bal18).index.tolist()
    top_bl18= [bal_cluster_labels[bal_cluster_cols.index(c)] for c in top_b18]
    top_c18 = ct[ct_cluster_cols].mean().nlargest(n_ct18).index.tolist()
    top_cl18= [ct_cluster_labels[ct_cluster_cols.index(c)] for c in top_c18]
    cm18 = np.full((n_ct18, n_bal18), np.nan); pm18 = np.ones_like(cm18)
    for ci18, cc in enumerate(top_c18):
        for bi18, bc in enumerate(top_b18):
            cv = ct_bc.reindex(shared_bc)[cc].values.astype(float)
            bv = bal_bc.reindex(shared_bc)[bc].values.astype(float)
            mask18 = ~(np.isnan(cv)|np.isnan(bv))
            if mask18.sum()>=5:
                r,p = spearmanr(cv[mask18],bv[mask18])
                cm18[ci18,bi18]=r; pm18[ci18,bi18]=p
    _,pq18,_,_ = multipletests(pm18.ravel(), method="fdr_bh"); pq18=pq18.reshape(pm18.shape)
    vm18 = np.nanpercentile(np.abs(cm18),95) or 0.5
    fig,ax18 = plt.subplots(figsize=(max(8,n_bal18*0.8),max(6,n_ct18*0.7)))
    im18 = ax18.imshow(cm18, aspect="auto", cmap="RdBu_r", vmin=-vm18, vmax=vm18, interpolation="nearest")
    for ci18 in range(n_ct18):
        for bi18 in range(n_bal18):
            if pq18[ci18,bi18]<0.05 and not np.isnan(cm18[ci18,bi18]):
                ax18.text(bi18,ci18,"*",ha="center",va="center",fontsize=9,color="white",fontweight="bold")
    ax18.set_xticks(range(n_bal18)); ax18.set_xticklabels(top_bl18, rotation=40, ha="right", fontsize=7)
    ax18.set_yticks(range(n_ct18)); ax18.set_yticklabels(top_cl18, fontsize=7)
    ax18.set_title(f"Figure 18 — BAL × CT Cross-Modal Correlation\n"
                   f"(Spearman r, n={len(both_bc)} paired; * FDR<5%)", fontsize=9, fontweight="bold")
    plt.colorbar(im18, ax=ax18, fraction=0.025, pad=0.02, label="Spearman r")
    plt.tight_layout()
    save_fig(fig, "fig18_crossmodal_bal_ct")
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 19 — Instance counts per sample × task
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 19] Instance count distributions …", flush=True)

fig, axes19 = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle("Figure 19 — Instance Counts per Sample Across Tasks", fontsize=10, fontweight="bold")

for row_i, (gcb, tn, pos_lbl, neg_lbl, pos_c, neg_c, _, _) in enumerate(TASKS):
    for col_i, (mod_title, df_m, _, _, _) in enumerate(MODALITIES):
        ax19 = axes19[row_i, col_i]
        gc_ = get_col(df_m, gcb)
        if "n_instances" not in df_m.columns or gc_ not in df_m.columns:
            ax19.text(0.5,0.5,"N/A",ha="center",va="center",transform=ax19.transAxes,fontsize=9,color=GREY)
            continue
        tmp19 = df_m[[gc_,"n_instances"]].copy()
        tmp19[gc_] = tmp19[gc_].where(tmp19[gc_].notna())
        pos19 = tmp19[tmp19[gc_]==1]["n_instances"].dropna().values
        neg19 = tmp19[tmp19[gc_]==0]["n_instances"].dropna().values
        if len(pos19)<2 or len(neg19)<2:
            ax19.text(0.5,0.5,f"n+={len(pos19)},n−={len(neg19)}",
                      ha="center",va="center",transform=ax19.transAxes,fontsize=8,color=GREY); continue
        vp19 = ax19.violinplot([neg19,pos19], positions=[0,1],
                               showmedians=True, showextrema=True, widths=0.65)
        for pc,c in zip(vp19["bodies"],[neg_c,pos_c]):
            pc.set_facecolor(c); pc.set_alpha(0.55)
        for part in ["cmedians","cbars","cmins","cmaxes"]:
            if part in vp19: vp19[part].set_color(DARK); vp19[part].set_linewidth(1.2)
        np.random.seed(42)
        for xi,(vals,c) in enumerate([(neg19,neg_c),(pos19,pos_c)]):
            ax19.scatter(np.random.normal(xi,0.07,len(vals)), vals,
                         color=c, alpha=0.3, s=4, linewidths=0, zorder=3)
        p19 = mw_test(pos19, neg19)
        ax19.set_xticks([0,1])
        ax19.set_xticklabels([f"{neg_lbl}\n(n={len(neg19)})",
                               f"{pos_lbl}\n(n={len(pos19)})"], fontsize=7)
        ax19.set_yscale("log")
        if col_i==0: ax19.set_ylabel("Instance count (log)")
        if row_i==0: ax19.set_title(mod_title[0], fontsize=9, fontweight="bold")
        ax19.set_title(f"{mod_title[0]}\n{tn}", fontsize=7.5)
        ax19.text(0.98,0.98,f"p={p19:.3f} {pval_stars(p19)}",
                  transform=ax19.transAxes,ha="right",va="top",fontsize=6.5,color=DARK)
        ax19.spines["top"].set_visible(False); ax19.spines["right"].set_visible(False)
plt.tight_layout()
save_fig(fig, "fig19_instance_counts_by_task")
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# FIG TTE — 4-Quartile TTE cluster landscape (3 modalities × 3 endpoints)
# Panel A: heatmap clusters×strata (z-scored), Panel B: line plot top clusters,
# Panel C: Kruskal-Wallis significance dot plot
# Plus compact 3×3 summary (Q1 vs Q4 vs Censored)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig TTE] TTE quartile cluster landscapes …", flush=True)

TTE_ENDPOINTS = [
    ("acr_days",   "acr_status",  "ACR"),
    ("clad_days",  "clad_status", "CLAD"),
    ("death_days", "death_status","Death"),
]

def tte_landscape(df_m, c_cols, c_labels, pal, tte_c, st_c, ep_name, mod_name, tag,
                  n_top=15, n_line=6):
    strat, n_ev = assign_tte_strata(df_m, tte_c, st_c)
    df_m = df_m.copy(); df_m["stratum"] = strat.values
    tmp = df_m[df_m["stratum"] >= 0]
    sizes = tmp["stratum"].value_counts()
    valid_s = [s for s in range(5) if int(sizes.get(s,0)) >= 3]
    if len(valid_s) < 2 or n_ev < 8:
        print(f"  [skip TTE {mod_name}/{ep_name}] n_ev={n_ev}", flush=True); return

    strat_means = tmp.groupby("stratum")[c_cols].mean().reindex(valid_s)
    var_across  = strat_means.var(axis=0, ddof=0)
    top_cols    = var_across.nlargest(n_top).index.tolist()
    top_lbl     = [c_labels[c_cols.index(c)] for c in top_cols]

    hz = strat_means[top_cols]
    cm = hz.mean(); cs = hz.std(ddof=0).replace(0,1)
    hz_z = ((hz - cm)/cs).values.T  # (n_clusters, n_strata)

    kw_p = []
    for col in top_cols:
        gs = [tmp[tmp["stratum"]==s][col].dropna().values for s in valid_s if len(tmp[tmp["stratum"]==s])>=3]
        try: _, p = kruskal(*gs) if len(gs)>=2 else (0,1.0)
        except: p = 1.0
        kw_p.append(p)
    _, kw_q, _, _ = multipletests(kw_p, method="fdr_bh")
    neg_log_kw = -np.log10(np.maximum(kw_q, 1e-10))

    x_lbl = [f"{TTE_Q_LABELS[s]}\n(n={int(sizes.get(s,0))})" for s in valid_s]

    fig_tte = plt.figure(figsize=(18, max(8, n_top*0.45+3)))
    gs_tte  = gridspec.GridSpec(1, 3, figure=fig_tte,
                                 width_ratios=[2.5, 2.5, 1.5], wspace=0.35)
    ax_h = fig_tte.add_subplot(gs_tte[0])
    ax_l = fig_tte.add_subplot(gs_tte[1])
    ax_k = fig_tte.add_subplot(gs_tte[2])

    # Heatmap
    vm_tte = np.nanpercentile(np.abs(hz_z), 95) or 1.0
    im_tte = ax_h.imshow(hz_z, aspect="auto", cmap="RdBu_r",
                          vmin=-vm_tte, vmax=vm_tte, interpolation="nearest")
    for ri2 in range(len(top_lbl)):
        if kw_q[ri2] < 0.05:
            ax_h.text(len(valid_s)-0.5, ri2, "*", ha="center", va="center",
                      fontsize=8, color="white")
    ax_h.set_xticks(range(len(valid_s))); ax_h.set_xticklabels(x_lbl, rotation=25, ha="right", fontsize=7)
    ax_h.set_yticks(range(len(top_lbl))); ax_h.set_yticklabels(top_lbl, fontsize=6.5)
    ax_h.set_title(f"(A) z-scored proportions\n{ep_name} TTE strata (* KW FDR<5%)", fontsize=8.5)
    plt.colorbar(im_tte, ax=ax_h, fraction=0.04, pad=0.02, label="z-score", shrink=0.6)

    # Line plot — top n_line most dynamic (by KW p)
    line_order = np.argsort(kw_p)[:n_line]
    xp = np.arange(len(valid_s))
    for li, idx in enumerate(line_order):
        col_ll = top_cols[idx]; lbl_ll = top_lbl[idx]
        color_ll = pal.get(lbl_ll, cmap20(li%20))
        means_ll = np.array([tmp[tmp["stratum"]==s][col_ll].mean() for s in valid_s])*100
        sems_ll  = np.array([tmp[tmp["stratum"]==s][col_ll].sem()  for s in valid_s])*100
        ax_l.plot(xp, means_ll, "o-", color=color_ll, lw=1.8, ms=5,
                  label=f"{lbl_ll[:16]} (q={kw_q[idx]:.3f})")
        ax_l.fill_between(xp, means_ll-sems_ll, means_ll+sems_ll, color=color_ll, alpha=0.15)
    ax_l.set_xticks(xp); ax_l.set_xticklabels(x_lbl, rotation=25, ha="right", fontsize=7)
    for tick, col_t in zip(ax_l.get_xticklabels(), [TTE_Q_COLORS[s] for s in valid_s]):
        tick.set_color(col_t)
    ax_l.set_ylabel("Mean proportion (%)"); ax_l.legend(frameon=False, fontsize=5.5)
    ax_l.set_title(f"(B) Top {n_line} dynamic clusters", fontsize=8.5)
    ax_l.spines["top"].set_visible(False); ax_l.spines["right"].set_visible(False)

    # KW dot plot
    sort_kw = np.argsort(neg_log_kw)[::-1]
    ys_kw = np.arange(len(top_lbl))
    ax_k.scatter(neg_log_kw[sort_kw], ys_kw,
                 c=[ACR_POS if kw_q[i]<0.05 else GREY for i in sort_kw],
                 s=np.where(kw_q[sort_kw]<0.05, 40, 15), alpha=0.85, linewidths=0, zorder=3)
    ax_k.axvline(-np.log10(0.05), color=GREY, lw=0.8, ls="--", alpha=0.7)
    ax_k.set_yticks(ys_kw); ax_k.set_yticklabels([top_lbl[i] for i in sort_kw], fontsize=6)
    ax_k.set_xlabel("−log₁₀(KW FDR q)")
    ax_k.set_title(f"(C) KW significance\n(● FDR<5%)", fontsize=8.5)
    ax_k.text(0.98,0.02,f"{int((kw_q<0.05).sum())}/{len(top_lbl)} FDR<5%",
              transform=ax_k.transAxes, ha="right", va="bottom", fontsize=7, color=ACR_POS)
    ax_k.spines["top"].set_visible(False); ax_k.spines["right"].set_visible(False)

    fig_tte.suptitle(
        f"{mod_name} — Cluster Proportions Across {ep_name} TTE Quartiles\n"
        f"(Q1=earliest, Q4=latest, Cen=censored; n_events={n_ev})",
        fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig_tte, f"figTTE_{tag}_{mod_name.replace(' ','_').lower()}")
    gc.collect()

for tte_c, st_c, ep_name in TTE_ENDPOINTS:
    for mod_name, df_m, c_cols, c_labels, pal in MODALITIES:
        tag = ep_name.lower()
        tte_landscape(df_m, c_cols, c_labels, pal, tte_c, st_c, ep_name, mod_name, tag)

# Compact 3×3 summary
print("\n[Fig TTE summary] …", flush=True)
fig_s, axes_s = plt.subplots(3, 3, figsize=(20, 18))
fig_s.suptitle("TTE Cluster Proportions Summary — Q1 (earliest) vs Q4 (latest) vs Censored",
               fontsize=11, fontweight="bold")
for row_i, (mod_name, df_m, c_cols, c_labels, pal) in enumerate(MODALITIES):
    for col_i, (tte_c, st_c, ep_name) in enumerate(TTE_ENDPOINTS):
        ax_s = axes_s[row_i, col_i]
        strat, n_ev = assign_tte_strata(df_m, tte_c, st_c)
        df_ms = df_m.copy(); df_ms["stratum"] = strat.values
        n_top_s = min(12, len(c_cols))
        top_s   = df_m[c_cols].mean().nlargest(n_top_s).index.tolist()
        top_ls  = [c_labels[c_cols.index(c)] for c in top_s]
        col_cs  = [pal.get(l, GREY) for l in top_ls]
        xs_s = np.arange(len(top_s)); w_s = 0.25
        ns = {}
        for gi, (sv, sl, sc_col, al) in enumerate(
                [(0,"Q1\nearliest","#B71C1C",0.9),
                 (3,"Q4\nlatest","#90CAF9",0.5),
                 (4,"Censored","#B0BEC5",0.3)]):
            sub_s = df_ms[df_ms["stratum"]==sv]
            ns[sl] = len(sub_s)
            if len(sub_s)<3: continue
            for ci, col in enumerate(top_s):
                ax_s.bar(ci+(gi-1)*w_s, sub_s[col].mean()*100, w_s,
                         color=col_cs[ci], alpha=al, linewidth=0)
        ax_s.set_xticks(xs_s)
        ax_s.set_xticklabels([l[:12] for l in top_ls], rotation=40, ha="right", fontsize=5.5)
        ax_s.spines["top"].set_visible(False); ax_s.spines["right"].set_visible(False)
        if row_i==0: ax_s.set_title(f"{ep_name} endpoint", fontsize=9, fontweight="bold")
        if col_i==0: ax_s.set_ylabel(f"{mod_name}\nMean prop (%)", fontsize=8)
        n_str = "  ".join(f"{k}:n={v}" for k,v in ns.items())
        ax_s.text(0.01,0.98,n_str,transform=ax_s.transAxes,fontsize=5.5,va="top",color=DARK)
        if col_i==0:
            ax_s.legend(handles=[mpatches.Patch(color="#B71C1C",alpha=0.9,label="Q1 earliest"),
                                   mpatches.Patch(color="#90CAF9",alpha=0.5,label="Q4 latest"),
                                   mpatches.Patch(color="#B0BEC5",alpha=0.3,label="Censored")],
                        frameon=False, fontsize=6, loc="upper right")
plt.tight_layout()
save_fig(fig_s, "figTTE_summary_q1_q4_censored")
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# CLR TRANSFORMATION
# Centered Log-Ratio: handles compositionality; all analyses below use CLR
# ══════════════════════════════════════════════════════════════════════════════

def clr_transform(df, cols, pseudocount=1e-5):
    """CLR transform: log(xi) - mean_j(log(xj)).  Returns df with same cols."""
    X = df[cols].values.astype(float).copy()
    X = np.where(X <= 0, pseudocount, X)
    X = X / X.sum(axis=1, keepdims=True)            # re-normalise to sum=1
    X = np.where(X <= 0, pseudocount, X)
    logX = np.log(X)
    clr  = logX - logX.mean(axis=1, keepdims=True)
    out  = df.copy()
    out[cols] = clr
    return out

print("\n[CLR] Transforming cluster proportions …", flush=True)
bal_clr = clr_transform(bal, bal_cluster_cols)
he_clr  = clr_transform(he,  he_tissue_cols)
ct_clr  = clr_transform(ct,  ct_cluster_cols)
# Clinical is z-scored continuous (not compositional) — no CLR needed;
# use as-is. NaN already set for 0.0 values during loading.
clin_clr = clin.copy()

MODALITIES_CLR = [
    ("BAL scRNA", bal_clr,  bal_cluster_cols, bal_cluster_labels, bal_palette),
    ("H&E",       he_clr,   he_tissue_cols,   he_tissue_labels,   he_palette),
    ("CT scan",   ct_clr,   ct_cluster_cols,  ct_cluster_labels,  ct_palette),
    ("Clinical",  clin_clr, clin_feat_cols,   clin_labels,        clin_palette),
]
print("  CLR done.", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 20 — CLR Volcano (replaces raw-proportion volcano with CLR-based)
# Differential CLR abundance × task × modality
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 20] CLR volcano …", flush=True)

fig20, axes20 = plt.subplots(3, 3, figsize=(17, 16))
fig20.suptitle("Figure 20 — CLR-Transformed Volcano: Differential Cluster Abundance",
               fontsize=11, fontweight="bold")

for col_i, (mod_name, df_m, c_cols, c_labels, pal) in enumerate(MODALITIES_CLR):
    for row_i, (gcb, tn, _, _, _, _, _, _) in enumerate(TASKS):
        ax20 = axes20[row_i, col_i]
        tmp20, gc20 = prep_group(df_m, gcb, c_cols)
        pos20 = tmp20[tmp20[gc20]==1]; neg20 = tmp20[tmp20[gc20]==0]
        if len(pos20)<3 or len(neg20)<3:
            ax20.text(0.5,0.5,f"n+={len(pos20)}, n−={len(neg20)}\nInsufficient",
                      ha="center",va="center",transform=ax20.transAxes,fontsize=8,color=GREY)
            ax20.set_title(f"{mod_name} | {tn}"); continue
        lfcs20, pvals20, labs20 = [], [], []
        for col, lbl in zip(c_cols, c_labels):
            pm = pos20[col].mean(); nm = neg20[col].mean()
            # CLR difference (additive on log scale = ratio in original)
            lfcs20.append(pm - nm)
            pvals20.append(mw_test(pos20[col].dropna().values, neg20[col].dropna().values))
            labs20.append(lbl)
        _, qvals20, _, _ = multipletests(pvals20, method="fdr_bh")
        nlq20 = -np.log10(np.maximum(qvals20, 1e-10))
        lfcs20 = np.array(lfcs20); qvals20 = np.array(qvals20)
        thresh = np.nanpercentile(np.abs(lfcs20), 60)
        c_pts = [ACR_POS if q<0.05 and l>thresh else
                 ACR_NEG if q<0.05 and l<-thresh else GREY
                 for l,q in zip(lfcs20,qvals20)]
        ax20.scatter(lfcs20, nlq20, c=c_pts,
                     s=np.where(qvals20<0.05,28,10), alpha=0.8, linewidths=0, zorder=3)
        ax20.axhline(-np.log10(0.05), color=GREY, lw=0.8, ls="--", alpha=0.7)
        ax20.axvline( thresh, color=ACR_POS, lw=0.7, ls=":", alpha=0.6)
        ax20.axvline(-thresh, color=ACR_NEG, lw=0.7, ls=":", alpha=0.6)
        for ti in np.argsort(nlq20)[::-1][:5]:
            if qvals20[ti] < 0.05:
                ax20.annotate(labs20[ti][:16], (lfcs20[ti], nlq20[ti]),
                              xytext=(3,3), textcoords="offset points", fontsize=5, color=DARK)
        n_up20   = int(((qvals20<0.05)&(lfcs20>thresh)).sum())
        n_down20 = int(((qvals20<0.05)&(lfcs20<-thresh)).sum())
        ax20.set_xlabel("ΔCLR (pos − neg)", fontsize=7)
        ax20.set_ylabel("−log₁₀(FDR q)", fontsize=7)
        ax20.set_title(f"{mod_name} | {tn}\n↑{n_up20} ↓{n_down20} (FDR<5%)", fontsize=8)
        ax20.text(0.01,0.99,f"n+={len(pos20)}, n−={len(neg20)}",
                  transform=ax20.transAxes,fontsize=6.5,va="top",color=DARK)
plt.tight_layout()
save_fig(fig20, "fig20_clr_volcano")
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 21 — Differential Co-occurrence Networks
# For each modality × task: Spearman cluster-cluster correlation separately in
# pos and neg groups (on CLR values); show differential correlation heatmap
# and highlight pairs that are rewired between groups.
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 21] Differential co-occurrence networks …", flush=True)

from scipy.stats import norm as _norm

def diff_corr_heatmap(ax, df_m, c_cols, c_labels, gcb, tn, n_top=12, min_n=6):
    """Differential Spearman correlation matrix: pos group minus neg group."""
    tmp21, gc21 = prep_group(df_m, gcb, c_cols)
    pos21 = tmp21[tmp21[gc21]==1]; neg21 = tmp21[tmp21[gc21]==0]
    if len(pos21)<min_n or len(neg21)<min_n:
        ax.text(0.5,0.5,f"n+={len(pos21)}, n−={len(neg21)}\nInsufficient",
                ha="center",va="center",transform=ax.transAxes,fontsize=8,color=GREY)
        ax.set_title(f"{tn}"); return

    # Select top n_top clusters by CLR variance across all samples
    var_top = tmp21[c_cols].var().nlargest(n_top).index.tolist()
    n_top   = len(var_top)   # clamp to actual available clusters
    lbl_top = [c_labels[c_cols.index(c)] for c in var_top]

    def spearman_matrix(df_sub, cols):
        n = len(cols)
        R = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                v1 = df_sub[cols[i]].dropna().values
                v2 = df_sub[cols[j]].dropna().values
                idx = np.arange(min(len(v1),len(v2)))
                if len(idx) >= 5:
                    r,_ = spearmanr(v1[:len(idx)], v2[:len(idx)])
                    R[i,j] = R[j,i] = r if not np.isnan(r) else 0
        return R

    Rp = spearman_matrix(pos21, var_top)
    Rn = spearman_matrix(neg21, var_top)

    # Fisher-z test for difference in correlations
    def fisherz(r): return np.arctanh(np.clip(r, -0.999, 0.999))
    def diff_pval(r1, r2, n1, n2):
        z1 = fisherz(r1); z2 = fisherz(r2)
        se = np.sqrt(1/(n1-3) + 1/(n2-3))
        z  = (z1-z2)/se if se > 0 else 0
        return float(2*(1-_norm.cdf(abs(z)))), float(z1-z2)

    dR = Rp - Rn          # differential correlation matrix
    pmat = np.ones_like(dR); zmat = np.zeros_like(dR)
    n1, n2 = len(pos21), len(neg21)
    for i in range(n_top):
        for j in range(i+1, n_top):
            p,z = diff_pval(Rp[i,j], Rn[i,j], n1, n2)
            pmat[i,j]=pmat[j,i]=p; zmat[i,j]=zmat[j,i]=z

    vmax21 = np.nanpercentile(np.abs(dR[~np.eye(n_top,dtype=bool)]), 95) or 0.5
    im21 = ax.imshow(dR, aspect="auto", cmap="RdBu_r", vmin=-vmax21, vmax=vmax21, interpolation="nearest")
    for i in range(n_top):
        for j in range(n_top):
            if i != j and pmat[i,j] < 0.05:
                ax.text(j, i, "×", ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")
    ax.set_xticks(range(n_top)); ax.set_xticklabels([l[:12] for l in lbl_top], rotation=35, ha="right", fontsize=5.5)
    ax.set_yticks(range(n_top)); ax.set_yticklabels([l[:12] for l in lbl_top], fontsize=5.5)
    ax.set_title(f"{tn}\n(× Fisher-z p<0.05)", fontsize=7.5)
    return im21

for mod_name, df_m, c_cols, c_labels, pal in MODALITIES_CLR:
    fig21, axes21 = plt.subplots(1, 3, figsize=(18, 6))
    fig21.suptitle(f"Figure 21 — {mod_name}: Differential CLR Co-occurrence\n"
                   f"(Δ Spearman r = ACR+/CLAD+/Death+ group minus control; × = Fisher-z p<0.05)",
                   fontsize=10, fontweight="bold")
    ims = []
    for ci21, (gcb, tn, _, _, _, _, _, _) in enumerate(TASKS):
        im = diff_corr_heatmap(axes21[ci21], df_m, c_cols, c_labels, gcb, tn)
        if im is not None: ims.append((axes21[ci21], im))
    for ax_c, im_c in ims:
        plt.colorbar(im_c, ax=ax_c, fraction=0.04, pad=0.02, label="ΔSpearman r", shrink=0.7)
    plt.tight_layout()
    save_fig(fig21, f"fig21_cooccurrence_{mod_name.replace(' ','_').lower()}")
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 22 — Pairwise cluster interaction for outcome prediction
# For each modality × task: rank cluster PAIRS by |correlation of product
# with label / early-event indicator|. Top pairs shown as network + bar chart.
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 22] Cluster pairwise interactions → outcome …", flush=True)

def pair_interactions(df_m, c_cols, c_labels, gcb, n_top_single=10, n_pairs=20):
    """
    Compute CLR(i) * CLR(j) products for top clusters and rank by
    point-biserial correlation with binary outcome.
    Returns list of (lbl_i, lbl_j, r, p, q).
    """
    gc22 = get_col(df_m, gcb)
    if gc22 not in df_m.columns: return []
    tmp22 = df_m[[gc22] + c_cols].copy()
    tmp22[gc22] = tmp22[gc22].where(tmp22[gc22].notna())
    tmp22 = tmp22[tmp22[gc22].isin([0,1])].copy()
    if len(tmp22) < 10: return []
    y = tmp22[gc22].values.astype(float)
    # Restrict to top-variance clusters for speed
    top_s = tmp22[c_cols].var().nlargest(n_top_single).index.tolist()
    top_l = [c_labels[c_cols.index(c)] for c in top_s]
    pairs = []
    for i in range(len(top_s)):
        for j in range(i+1, len(top_s)):
            prod = tmp22[top_s[i]].values * tmp22[top_s[j]].values
            if prod.std() < 1e-9: continue
            r,p = spearmanr(prod, y)
            if not np.isnan(r):
                pairs.append((top_l[i], top_l[j], float(r), float(p)))
    if not pairs: return []
    pairs_df = pd.DataFrame(pairs, columns=["A","B","r","p"])
    _, pairs_df["q"], _, _ = multipletests(pairs_df["p"], method="fdr_bh")
    pairs_df = pairs_df.reindex(pairs_df["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return pairs_df.head(n_pairs)

for mod_name, df_m, c_cols, c_labels, pal in MODALITIES_CLR:
    fig22, axes22 = plt.subplots(3, 2, figsize=(16, 18))
    fig22.suptitle(f"Figure 22 — {mod_name}: Cluster Pair Co-occurrence → Outcome\n"
                   f"(CLR product Spearman r; red=co-elevated in positive group, blue=anti-correlated)",
                   fontsize=10, fontweight="bold")
    for row22, (gcb, tn, pos_lbl, neg_lbl, pos_c, neg_c, _, _) in enumerate(TASKS):
        ax_bar22 = axes22[row22, 0]
        ax_net22 = axes22[row22, 1]
        pairs_df22 = pair_interactions(df_m, c_cols, c_labels, gcb)
        if pairs_df22 is None or len(pairs_df22) == 0:
            for ax in [ax_bar22, ax_net22]:
                ax.text(0.5,0.5,"Insufficient data",ha="center",va="center",
                        transform=ax.transAxes,fontsize=9,color=GREY)
            continue

        # Bar chart of top pairs
        top_show = pairs_df22.head(15)
        ys22 = np.arange(len(top_show))
        ax_bar22.barh(ys22, top_show["r"].values,
                      color=[ACR_POS if r>0 else ACR_NEG for r in top_show["r"].values],
                      alpha=0.8, linewidth=0)
        for yi, (_, row_p) in enumerate(top_show.iterrows()):
            q_str = f"q={row_p['q']:.3f}" if row_p["q"] < 0.1 else ""
            ax_bar22.text(row_p["r"]+0.002*(1 if row_p["r"]>=0 else -1), yi,
                          q_str, va="center", fontsize=5.5,
                          color=DARK if row_p["q"]<0.05 else GREY)
        ax_bar22.set_yticks(ys22)
        ax_bar22.set_yticklabels(
            [f"{r['A'][:12]} × {r['B'][:12]}" for _,r in top_show.iterrows()],
            fontsize=5.5)
        ax_bar22.axvline(0, color=GREY, lw=0.8)
        ax_bar22.set_xlabel(f"Spearman r (pair product vs {tn})")
        ax_bar22.set_title(f"{tn} — top interacting pairs")
        ax_bar22.spines["top"].set_visible(False); ax_bar22.spines["right"].set_visible(False)

        # Network: nodes=clusters, edges=top pairs coloured by direction
        import math
        unique_nodes = list(dict.fromkeys(
            [x for row_p in pairs_df22.head(10).itertuples() for x in [row_p.A, row_p.B]]))
        n_nd = len(unique_nodes)
        if n_nd < 2:
            ax_net22.text(0.5,0.5,"<2 nodes",ha="center",va="center",
                          transform=ax_net22.transAxes,fontsize=9,color=GREY); continue
        angles = [2*math.pi*i/n_nd for i in range(n_nd)]
        nx_pos = {nd: (math.cos(a), math.sin(a)) for nd, a in zip(unique_nodes, angles)}
        ax_net22.set_xlim(-1.6,1.6); ax_net22.set_ylim(-1.6,1.6); ax_net22.axis("off")
        ax_net22.set_title(f"{tn} — co-occurrence network (top 10 pairs)")

        # Determine node importance by sum |r| of connected edges
        node_r = {nd: 0.0 for nd in unique_nodes}
        for _, ep in pairs_df22.head(10).iterrows():
            node_r[ep["A"]] += abs(ep["r"]); node_r[ep["B"]] += abs(ep["r"])
        max_nr = max(node_r.values()) if node_r else 1.0

        for _, ep in pairs_df22.head(10).iterrows():
            x0,y0 = nx_pos[ep["A"]]; x1,y1 = nx_pos[ep["B"]]
            lw = 1.5 + 4*abs(ep["r"])
            col_e = ACR_POS if ep["r"]>0 else ACR_NEG
            ax_net22.plot([x0,x1],[y0,y1], color=col_e, lw=lw,
                          alpha=0.7 if ep["q"]<0.05 else 0.3, zorder=1)
        for nd in unique_nodes:
            x,y = nx_pos[nd]
            sz  = 60 + 200*(node_r[nd]/max_nr)
            gc22_ = get_col(df_m, gcb)
            tmp_nd = df_m[[gc22_, nd.split(" ×")[0]]].copy() if nd in c_cols else None
            ax_net22.scatter([x],[y], s=sz, color=pal.get(nd, GREY), alpha=0.9,
                             edgecolors="white", linewidths=1.2, zorder=3)
            ax_net22.text(x*1.22, y*1.22, nd[:14], ha="center", va="center",
                          fontsize=5, color=DARK, zorder=4)
    plt.tight_layout()
    save_fig(fig22, f"fig22_pair_interactions_{mod_name.replace(' ','_').lower()}")
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 23 — LASSO / RF feature importance on CLR for multivariate prediction
# Per modality × task: regularised logistic (ACR, CLAD status) or Cox-ridge
# substitute (Q1 early-event vs Q4/censored) to identify multi-cluster signatures
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 23] LASSO/RF importance on CLR …", flush=True)

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

def lasso_importance(df_m, c_cols, c_labels, gcb, n_top=15, cv=5):
    gc23 = get_col(df_m, gcb)
    if gc23 not in df_m.columns: return None, None
    tmp23 = df_m[[gc23]+c_cols].dropna(subset=[gc23]).copy()
    tmp23[gc23] = tmp23[gc23].where(tmp23[gc23].notna())
    tmp23 = tmp23[tmp23[gc23].isin([0,1])].copy()
    X = tmp23[c_cols].values.astype(float)
    y = tmp23[gc23].values.astype(int)
    if y.sum()<3 or (y==0).sum()<3 or X.shape[0]<10: return None, None
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    # LASSO logistic with CV regularisation
    try:
        lr = LogisticRegressionCV(Cs=10, cv=StratifiedKFold(min(cv,y.sum())),
                                   penalty="l1", solver="saga", max_iter=500,
                                   random_state=42, scoring="roc_auc")
        lr.fit(Xs, y)
        coefs = lr.coef_.ravel()
        auc_cv = cross_val_score(lr, Xs, y, cv=StratifiedKFold(min(cv,y.sum())),
                                  scoring="roc_auc").mean()
    except Exception:
        coefs = np.zeros(len(c_cols)); auc_cv = 0.5
    # RF importance
    try:
        rf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42,
                                     class_weight="balanced", n_jobs=2)
        rf.fit(X, y)
        rf_imp = rf.feature_importances_
        auc_rf = cross_val_score(rf, X, y, cv=StratifiedKFold(min(cv,y.sum())),
                                  scoring="roc_auc").mean()
    except Exception:
        rf_imp = np.zeros(len(c_cols)); auc_rf = 0.5
    return (coefs, auc_cv, c_labels), (rf_imp, auc_rf, c_labels)

for mod_name, df_m, c_cols, c_labels, pal in MODALITIES_CLR:
    fig23, axes23 = plt.subplots(3, 2, figsize=(15, 18))
    fig23.suptitle(f"Figure 23 — {mod_name}: Multi-cluster CLR Signature (LASSO + RF)\n"
                   f"(LASSO: L1-penalised logistic; RF: Gini impurity; CLR-normalised inputs)",
                   fontsize=10, fontweight="bold")
    for row23, (gcb, tn, pos_lbl, neg_lbl, pos_c, neg_c, _, _) in enumerate(TASKS):
        ax_l23 = axes23[row23, 0]; ax_r23 = axes23[row23, 1]
        lasso_res, rf_res = lasso_importance(df_m, c_cols, c_labels, gcb)
        n_top_show = min(15, len(c_cols))
        for ax_, res, method, color in [
            (ax_l23, lasso_res, "LASSO", ACR_POS),
            (ax_r23, rf_res,   "RF",    CLAD_COL)]:
            if res is None:
                ax_.text(0.5,0.5,"Insufficient data",ha="center",va="center",
                         transform=ax_.transAxes,fontsize=9,color=GREY)
                ax_.set_title(f"{tn} — {method}"); continue
            vals, auc, lbls = res
            order23 = np.argsort(np.abs(vals))[::-1][:n_top_show]
            vals_s = vals[order23]; lbls_s = [lbls[i] for i in order23]
            ys23 = np.arange(len(vals_s))
            ax_.barh(ys23, vals_s if method=="LASSO" else vals_s,
                     color=[ACR_POS if v>0 else ACR_NEG if method=="LASSO" else color
                             for v in vals_s],
                     alpha=0.8, linewidth=0)
            ax_.set_yticks(ys23); ax_.set_yticklabels([l[:22] for l in lbls_s], fontsize=6)
            if method == "LASSO":
                ax_.axvline(0, color=GREY, lw=0.8)
                ax_.set_xlabel("LASSO coefficient (std CLR)")
            else:
                ax_.set_xlabel("RF feature importance")
            ax_.set_title(f"{tn} — {method} (AUC={auc:.2f})")
            ax_.spines["top"].set_visible(False); ax_.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig23, f"fig23_lasso_rf_{mod_name.replace(' ','_').lower()}")
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 24 — Early-event (Q1) co-occurrence fingerprint
# Which cluster PAIRS are jointly elevated in Q1 (earliest events) vs
# censored/Q4? Analogous to Fig 22 but for survival endpoints.
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 24] Q1 early-event co-occurrence fingerprint …", flush=True)

def q1_pair_fingerprint(df_m, c_cols, c_labels, tte_c, st_c, n_top_single=10, n_pairs=20):
    strat24, n_ev24 = assign_tte_strata(df_m, tte_c, st_c)
    if n_ev24 < 8: return None
    df24 = df_m.copy(); df24["stratum"] = strat24.values
    q1  = df24[df24["stratum"]==0]   # earliest events
    ref = df24[df24["stratum"].isin([3,4])]  # latest + censored as reference
    if len(q1)<5 or len(ref)<5: return None
    # Top clusters by |ΔCLR| Q1 vs ref
    top_s = (q1[c_cols].mean() - ref[c_cols].mean()).abs().nlargest(n_top_single).index.tolist()
    top_l = [c_labels[c_cols.index(c)] for c in top_s]
    # binary outcome: Q1=1, ref=0
    combined = pd.concat([q1, ref])
    combined["q1_flag"] = (combined["stratum"]==0).astype(int)
    pairs = []
    for i in range(len(top_s)):
        for j in range(i+1, len(top_s)):
            prod = combined[top_s[i]].values * combined[top_s[j]].values
            if prod.std()<1e-9: continue
            r,p = spearmanr(prod, combined["q1_flag"].values)
            if not np.isnan(r):
                pairs.append((top_l[i], top_l[j], float(r), float(p)))
    if not pairs: return None
    pairs_df = pd.DataFrame(pairs, columns=["A","B","r","p"])
    _, pairs_df["q"],_,_ = multipletests(pairs_df["p"], method="fdr_bh")
    return pairs_df.reindex(pairs_df["r"].abs().sort_values(ascending=False).index).reset_index(drop=True).head(n_pairs)

for mod_name, df_m, c_cols, c_labels, pal in MODALITIES_CLR:
    fig24, axes24 = plt.subplots(1, 3, figsize=(18, 7))
    fig24.suptitle(f"Figure 24 — {mod_name}: Cluster Pairs Co-elevated in EARLIEST Events (Q1)\n"
                   f"(CLR product Spearman r vs Q1-flag; Q1=earliest 25%% of events, ref=Q4+censored)",
                   fontsize=10, fontweight="bold")
    for ci24, (tte_c, st_c, ep_name) in enumerate(TTE_ENDPOINTS):
        ax24 = axes24[ci24]
        pdf24 = q1_pair_fingerprint(df_m, c_cols, c_labels, tte_c, st_c)
        if pdf24 is None or len(pdf24)==0:
            ax24.text(0.5,0.5,"Insufficient data",ha="center",va="center",
                      transform=ax24.transAxes,fontsize=9,color=GREY)
            ax24.set_title(f"{ep_name}"); continue
        show24 = pdf24.head(12)
        ys24 = np.arange(len(show24))
        ax24.barh(ys24, show24["r"].values,
                  color=[ACR_POS if r>0 else ACR_NEG for r in show24["r"].values],
                  alpha=0.8, linewidth=0)
        for yi24, (_,rp) in enumerate(show24.iterrows()):
            if rp["q"] < 0.05:
                ax24.text(rp["r"]+(0.005 if rp["r"]>=0 else -0.005), yi24,
                          f"q={rp['q']:.3f}", va="center", fontsize=5.5,
                          color=DARK, ha="left" if rp["r"]>=0 else "right")
        ax24.set_yticks(ys24)
        ax24.set_yticklabels([f"{r['A'][:12]} × {r['B'][:12]}" for _,r in show24.iterrows()],
                              fontsize=5.5)
        ax24.axvline(0, color=GREY, lw=0.8)
        ax24.set_xlabel(f"Spearman r (pair × Q1 early-{ep_name})")
        ax24.set_title(f"{ep_name} (top {len(show24)} pairs)")
        ax24.spines["top"].set_visible(False); ax24.spines["right"].set_visible(False)
        ax24.text(0.98,0.02,"red=co-elevated in early events",transform=ax24.transAxes,
                  ha="right",va="bottom",fontsize=6.5,color=ACR_POS)
    plt.tight_layout()
    save_fig(fig24, f"fig24_q1_cooccurrence_{mod_name.replace(' ','_').lower()}")
    gc.collect()



# ══════════════════════════════════════════════════════════════════════════════
# FIGS 25–27 — TRUE CROSS-MODAL CO-OCCURRENCE
#
# Question: does high proportion of cluster i in modality A **jointly with**
# high proportion of cluster j in modality B predict ACR+ / CLAD / Death?
#
# Method:
#   1. Find samples that have BOTH modalities measured at the same timepoint
#      (matched by stem_int_str — same bag/visit).
#   2. CLR-transform each modality's proportions independently.
#   3. For every cross-modal cluster pair (i from mod A, j from mod B):
#        interaction feature = CLR_A_i × CLR_B_j
#   4. Spearman r of that interaction feature with binary outcome.
#   5. BH FDR correction across all pairs. q < 0.05 highlighted.
#
# This captures genuine co-occurrence: the product is large only when BOTH
# clusters are simultaneously elevated in the same sample.
#
# Modality pairs covered:
#   Fig 25 — H&E × CT
#   Fig 26 — BAL × H&E
#   Fig 27 — BAL × CT
# Each figure has 3 rows (ACR / CLAD / Death) × 2 panels:
#   Left  — interaction heatmap (clusters_A × clusters_B, colour = Spearman r)
#   Right — top-20 cross-modal pairs ranked by |r|, bar chart + mini-network
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Figs 25-27] Cross-modal co-occurrence …", flush=True)

MOD_COLORS_MAP = {
    "H&E":      MOD_COLORS["H&E"],
    "BAL scRNA":MOD_COLORS["BAL scRNA"],
    "CT scan":  MOD_COLORS["CT scan"],
}

def get_paired(df_a, df_b, c_cols_a, c_cols_b, outcome_col):
    """Return (X_a, X_b, y) numpy arrays for samples present in both modalities
    and with a valid binary outcome label."""
    key = "stem_int_str"
    # outcome from df_a (both dfs were merged from same splits)
    oc_a = get_col(df_a, outcome_col)
    sub_a = df_a[[key, oc_a] + c_cols_a].copy()
    sub_a[oc_a] = sub_a[oc_a].where(sub_a[oc_a].notna())
    sub_a = sub_a[sub_a[oc_a].isin([0, 1])].copy()
    sub_b = df_b[[key] + c_cols_b].copy()
    merged = sub_a.merge(sub_b, on=key, how="inner", suffixes=("_a","_b"))
    if len(merged) < 8:
        return None, None, None, 0
    y  = merged[oc_a].values.astype(int)
    Xa = merged[[c+"_a" if c+"_a" in merged.columns else c for c in c_cols_a]].values.astype(float)
    Xb = merged[[c+"_b" if c+"_b" in merged.columns else c for c in c_cols_b]].values.astype(float)
    return Xa, Xb, y, len(merged)

def crossmodal_interaction_heatmap(ax_heat, ax_bar,
                                    df_a, df_b,
                                    c_cols_a, c_labels_a, pal_a,
                                    c_cols_b, c_labels_b, pal_b,
                                    outcome_col, task_name,
                                    mod_name_a, mod_name_b,
                                    n_top_a=10, n_top_b=10, n_bar=20):
    """Compute cross-modal interaction Spearman r → heatmap + top-pairs bar."""
    Xa, Xb, y, n_paired = get_paired(df_a, df_b, c_cols_a, c_cols_b, outcome_col)
    if Xa is None:
        for ax in [ax_heat, ax_bar]:
            ax.text(0.5, 0.5, f"n_paired={n_paired}\nInsufficient",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color=GREY)
        return

    # Select top-n clusters by variance in paired samples
    var_a = Xa.var(axis=0); var_b = Xb.var(axis=0)
    top_a = np.argsort(var_a)[::-1][:n_top_a]
    top_b = np.argsort(var_b)[::-1][:n_top_b]
    lbl_a = [c_labels_a[i] for i in top_a]
    lbl_b = [c_labels_b[i] for i in top_b]
    Xa_t  = Xa[:, top_a]; Xb_t = Xb[:, top_b]

    # Cross-modal interaction matrix: Spearman r of (a_i × b_j) with y
    R = np.zeros((len(top_a), len(top_b)))
    P = np.ones_like(R)
    pairs_list = []
    for ai in range(len(top_a)):
        for bi in range(len(top_b)):
            prod = Xa_t[:, ai] * Xb_t[:, bi]
            if prod.std() < 1e-9:
                continue
            r, p = spearmanr(prod, y)
            if not np.isnan(r):
                R[ai, bi] = r
                P[ai, bi] = p
                pairs_list.append((lbl_a[ai], lbl_b[bi], float(r), float(p)))

    # FDR correction across all pairs
    if pairs_list:
        pdf = pd.DataFrame(pairs_list, columns=["A","B","r","p"])
        _, pdf["q"], _, _ = multipletests(pdf["p"], method="fdr_bh")
        _, Pq_flat, _, _ = multipletests(P.ravel(), method="fdr_bh")
        Pq = Pq_flat.reshape(P.shape)
    else:
        Pq = P.copy()
        pdf = pd.DataFrame(columns=["A","B","r","p","q"])

    # Heatmap
    vmax_h = max(np.nanpercentile(np.abs(R), 95), 0.1)
    im = ax_heat.imshow(R, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax_h, vmax=vmax_h, interpolation="nearest")
    for ai in range(len(top_a)):
        for bi in range(len(top_b)):
            if Pq[ai, bi] < 0.05 and abs(R[ai, bi]) > 0:
                ax_heat.text(bi, ai, "*", ha="center", va="center",
                             fontsize=8, color="white", fontweight="bold")
    ax_heat.set_xticks(range(len(top_b)))
    ax_heat.set_xticklabels([l[:14] for l in lbl_b], rotation=35, ha="right", fontsize=5.5)
    ax_heat.set_yticks(range(len(top_a)))
    ax_heat.set_yticklabels([l[:16] for l in lbl_a], fontsize=6)
    ax_heat.set_xlabel(mod_name_b, fontsize=8, color=MOD_COLORS_MAP.get(mod_name_b, GREY))
    ax_heat.set_ylabel(mod_name_a, fontsize=8, color=MOD_COLORS_MAP.get(mod_name_a, GREY))
    ax_heat.set_title(f"{task_name} (n={n_paired} paired)\n* = BH q<5%", fontsize=8)
    plt.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02,
                 label="Spearman r\n(CLR_A × CLR_B vs label)", shrink=0.7)

    # Top-pairs bar chart
    if len(pdf) == 0:
        ax_bar.text(0.5,0.5,"No pairs",ha="center",va="center",
                    transform=ax_bar.transAxes,fontsize=9,color=GREY); return
    top_pairs = pdf.reindex(pdf["r"].abs().sort_values(ascending=False).index).head(n_bar)
    ys  = np.arange(len(top_pairs))
    col = [ACR_POS if r > 0 else ACR_NEG for r in top_pairs["r"].values]
    ax_bar.barh(ys, top_pairs["r"].values, color=col, alpha=0.8, linewidth=0)
    for yi, (_, row_p) in enumerate(top_pairs.iterrows()):
        sig_str = pval_stars(row_p["p"])
        q_str   = f"q={row_p['q']:.3f}" if row_p["q"] < 0.1 else ""
        txt = f"{sig_str} {q_str}".strip()
        if txt:
            ax_bar.text(row_p["r"] + (0.004 if row_p["r"] >= 0 else -0.004),
                        yi, txt, va="center", fontsize=5,
                        color=DARK if row_p["q"] < 0.05 else GREY,
                        ha="left" if row_p["r"] >= 0 else "right")
    ax_bar.set_yticks(ys)
    ax_bar.set_yticklabels(
        [f"{r['A'][:13]}  ×  {r['B'][:13]}" for _, r in top_pairs.iterrows()],
        fontsize=5)
    # Colour the × separator to indicate modality boundary
    ax_bar.axvline(0, color=GREY, lw=0.8)
    ax_bar.set_xlabel(f"Spearman r  (interaction vs {task_name})", fontsize=7)
    ax_bar.set_title(f"Top cross-modal pairs\n"
                     f"red = co-elevated → {task_name}+", fontsize=7.5)
    n_sig = int((top_pairs["q"] < 0.05).sum())
    ax_bar.text(0.98, 0.02, f"{n_sig}/{len(top_pairs)} FDR<5%",
                transform=ax_bar.transAxes, ha="right", va="bottom",
                fontsize=6.5, color=ACR_POS if n_sig > 0 else GREY)
    ax_bar.spines["top"].set_visible(False); ax_bar.spines["right"].set_visible(False)


XMOD_PAIRS = [
    ("H&E",       he_clr,   he_tissue_cols,  he_tissue_labels,  he_palette,
     "CT scan",   ct_clr,   ct_cluster_cols, ct_cluster_labels, ct_palette,  "25"),
    ("BAL scRNA", bal_clr,  bal_cluster_cols,bal_cluster_labels,bal_palette,
     "H&E",       he_clr,   he_tissue_cols,  he_tissue_labels,  he_palette,  "26"),
    ("BAL scRNA", bal_clr,  bal_cluster_cols,bal_cluster_labels,bal_palette,
     "CT scan",   ct_clr,   ct_cluster_cols, ct_cluster_labels, ct_palette,  "27"),
    # Clinical cross-modal pairs — largest paired sets
    ("Clinical",  clin_clr, clin_feat_cols,  clin_labels,       clin_palette,
     "BAL scRNA", bal_clr,  bal_cluster_cols,bal_cluster_labels, bal_palette, "28a"),
    ("Clinical",  clin_clr, clin_feat_cols,  clin_labels,       clin_palette,
     "H&E",       he_clr,   he_tissue_cols,  he_tissue_labels,   he_palette,  "28b"),
    ("Clinical",  clin_clr, clin_feat_cols,  clin_labels,       clin_palette,
     "CT scan",   ct_clr,   ct_cluster_cols, ct_cluster_labels,  ct_palette,  "28c"),
]

for (mna, dfa, ca, la, pa,
     mnb, dfb, cb, lb, pb, fnum) in XMOD_PAIRS:
    # Check overlap
    shared_keys = set(dfa["stem_int_str"].astype(str)) & set(dfb["stem_int_str"].astype(str))
    print(f"\n[Fig {fnum}] {mna} × {mnb}  paired n={len(shared_keys)}", flush=True)

    fig_xm, axes_xm = plt.subplots(3, 2, figsize=(16, 22))
    fig_xm.suptitle(
        f"Figure {fnum} — Cross-Modal Co-occurrence: {mna} × {mnb}\n"
        f"Interaction = CLR({mna} cluster i) × CLR({mnb} cluster j)  "
        f"correlated with outcome\n"
        f"(Spearman r on {len(shared_keys)} paired samples; * BH FDR<5%)",
        fontsize=10, fontweight="bold", y=1.01)

    for row_xm, (gcb, tn, _, _, _, _, _, _) in enumerate(TASKS):
        ax_h = axes_xm[row_xm, 0]
        ax_b = axes_xm[row_xm, 1]
        crossmodal_interaction_heatmap(
            ax_h, ax_b,
            dfa, dfb,
            ca, la, pa,
            cb, lb, pb,
            gcb, tn,
            mna, mnb,
            n_top_a=min(10, len(ca)),
            n_top_b=min(10, len(cb)),
            n_bar=20,
        )

    plt.tight_layout()
    tag = f"fig{fnum}_xmod_{mna.replace(' ','_').lower()}_x_{mnb.replace(' ','_').lower()}"
    save_fig(fig_xm, tag)
    gc.collect()


# ── Fig 29 — Multi-modal logistic: cross-modal interaction features ────────
# LASSO logistic using ALL cross-modal interaction features jointly
# (BAL×HE + BAL×CT + HE×CT + Clin×BAL + Clin×HE + Clin×CT) to predict label.
# Shows which cross-modal pair adds most predictive power.
# ────────────────────────────────────────────────────────────────────────────
print("\n[Fig 29] Multi-modal interaction LASSO …", flush=True)

def build_xmod_feature_matrix(dfa, dfb, ca, la, cb, lb,
                               outcome_col, n_top=8):
    """Build matrix of cross-modal CLR products for samples with both modalities."""
    Xa, Xb, y, n_p = get_paired(dfa, dfb, ca, cb, outcome_col)
    if Xa is None or n_p < 10:
        return None, None, []
    var_a = Xa.var(axis=0); var_b = Xb.var(axis=0)
    sel_a = np.argsort(var_a)[::-1][:n_top]
    sel_b = np.argsort(var_b)[::-1][:n_top]
    lbl_a = [la[i] for i in sel_a]
    lbl_b = [lb[i] for i in sel_b]
    cols, feat_names = [], []
    for ai, ln_a in zip(sel_a, lbl_a):
        for bi, ln_b in zip(sel_b, lbl_b):
            cols.append(Xa[:, ai] * Xb[:, bi])
            feat_names.append(f"{ln_a[:12]}×{ln_b[:12]}")
    X = np.column_stack(cols)
    return X, y, feat_names

for gcb, tn, pos_lbl, neg_lbl, pos_c, neg_c, _, _ in TASKS:
    fig28 = plt.figure(figsize=(18, 14))
    fig28.suptitle(
        f"Figure 29 — Multi-Modal Interaction LASSO: {tn}\n"
        f"Features = z_A_i × CLR_B_j (Clinical × image) or CLR(A_i) × CLR(B_j) "
        f"(image × image); L1 logistic; signed coefs = direction of effect",
        fontsize=10, fontweight="bold")

    xmod_combos = [
        ("H&E×CT",      he_clr,   he_tissue_cols,  he_tissue_labels,
                        ct_clr,   ct_cluster_cols, ct_cluster_labels),
        ("BAL×H&E",     bal_clr,  bal_cluster_cols,bal_cluster_labels,
                        he_clr,   he_tissue_cols,  he_tissue_labels),
        ("BAL×CT",      bal_clr,  bal_cluster_cols,bal_cluster_labels,
                        ct_clr,   ct_cluster_cols, ct_cluster_labels),
        ("Clin×BAL",    clin_clr, clin_feat_cols,  clin_labels,
                        bal_clr,  bal_cluster_cols,bal_cluster_labels),
        ("Clin×H&E",    clin_clr, clin_feat_cols,  clin_labels,
                        he_clr,   he_tissue_cols,  he_tissue_labels),
        ("Clin×CT",     clin_clr, clin_feat_cols,  clin_labels,
                        ct_clr,   ct_cluster_cols, ct_cluster_labels),
    ]

    fig28, ax28 = plt.subplots(2, 3, figsize=(18, 14))
    ax28 = ax28.ravel()

    for ci28, (combo_name, dfa, ca, la, dfb, cb, lb) in enumerate(xmod_combos):
        ax = ax28[ci28]
        X28, y28, feat_names = build_xmod_feature_matrix(
            dfa, dfb, ca, la, cb, lb, gcb, n_top=7)
        if X28 is None or y28 is None or y28.sum() < 3 or (y28==0).sum() < 3:
            ax.text(0.5,0.5,"Insufficient paired\nsamples",
                    ha="center",va="center",transform=ax.transAxes,fontsize=9,color=GREY)
            ax.set_title(combo_name); continue

        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        scaler28 = StandardScaler()
        Xs28 = scaler28.fit_transform(X28)
        n_splits = min(5, int(y28.sum()))
        try:
            lr28 = LogisticRegressionCV(
                Cs=10, cv=StratifiedKFold(n_splits),
                penalty="l1", solver="saga", max_iter=500,
                random_state=42, scoring="roc_auc", n_jobs=2)
            lr28.fit(Xs28, y28)
            coefs28 = lr28.coef_.ravel()
            auc28 = cross_val_score(lr28, Xs28, y28,
                                     cv=StratifiedKFold(n_splits),
                                     scoring="roc_auc").mean()
        except Exception as e:
            print(f"  [Fig28 {combo_name} {tn}] {e}", flush=True)
            coefs28 = np.zeros(len(feat_names)); auc28 = 0.5

        # Show all non-zero coefs, else top 15 by magnitude
        nonzero = np.where(coefs28 != 0)[0]
        if len(nonzero) == 0:
            nonzero = np.argsort(np.abs(coefs28))[::-1][:15]
        order28 = nonzero[np.argsort(np.abs(coefs28[nonzero]))[::-1]][:20]
        vals28  = coefs28[order28]
        lbls28  = [feat_names[i] for i in order28]
        ys28    = np.arange(len(vals28))

        ax.barh(ys28, vals28,
                color=[ACR_POS if v > 0 else ACR_NEG for v in vals28],
                alpha=0.8, linewidth=0)
        ax.set_yticks(ys28)
        ax.set_yticklabels(lbls28, fontsize=6)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_xlabel("LASSO coefficient (std CLR product)", fontsize=7)
        ax.set_title(f"{combo_name}  (AUC={auc28:.2f})\n"
                     f"n_paired={X28.shape[0]}, "
                     f"n+={int(y28.sum())}, n−={int((y28==0).sum())}", fontsize=8)
        ax.text(0.98, 0.98,
                f"red = co-elevated → {pos_lbl}\nblue = co-elevated → {neg_lbl}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                color=DARK)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_fig(fig28, f"fig29_xmod_lasso_{tn.lower()}")
    gc.collect()


print("\n" + "="*60, flush=True)
print("=== All figures complete ===", flush=True)
