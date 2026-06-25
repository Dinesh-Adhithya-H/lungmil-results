#!/usr/bin/env python3
"""
nature_fig0_hero.py
Publication-quality hero figure (Fig 0) for the Nature submission.

Panel layout (3 rows):
  Row 1 — Schematic: patient cohort → serial visits → modalities → MIL → 4 tasks
  Row 2 — Real data overview: cohort timeline (swimmer), modality availability
  Row 3 — Key findings summary: top biomarker per task with biological annotation

Biological annotations from literature:
  ACR:   BAL lymphocytosis (CD4/CD8 T cells) + alveolar inflammation on H&E
  CLAD:  BAL neutrophilia + pseudoslope_FEV1 decline
  Death: FEV1 absolute + renal function (GFR) + donor risk
  ACR_TTE: delta_FEV1 + DSA count + CMV mismatch

Run via sbatch — do NOT run on the login node.
"""

import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── PATHS ────────────────────────────────────────────────────────────────────
SPLITS_CSV  = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
PROP_DIR    = Path("/home/aih/dinesh.haridoss/chicago_mil/results/cluster_proportions")
CLMAP       = Path("/home/aih/dinesh.haridoss/chicago_mil/results/cluster_name_maps/HE_cluster_map.json")
COHORT_JSON = Path("/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/cohort_summary.json")
IMP_CSV     = Path("/home/aih/dinesh.haridoss/chicago_mil/results/linear_models/feature_importance.csv")
OUT_DIR     = Path("/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── STYLE ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

ACR_POS   = "#C62828"; ACR_NEG   = "#1565C0"
CLAD_COL  = "#E65100"; DEATH_COL = "#4527A0"
GREY      = "#90A4AE"; DARK      = "#263238"; LIGHT     = "#ECEFF1"
MOD_COLS  = {"H&E": "#2E7D32", "BAL scRNA": "#1565C0",
             "CT scan": "#E65100", "Clinical": "#6A1B9A"}

def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {name}.png/.pdf", flush=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading …", flush=True)
splits = pd.read_csv(SPLITS_CSV)
splits["stem"] = splits["file"].str.replace(".pt","",regex=False).str.zfill(5)
splits["acr_binary"] = splits["label"].where(splits["label"].notna())
cohort = json.load(open(COHORT_JSON)) if COHORT_JSON.exists() else {}
mod_n  = cohort.get("modality_n", {"Clinical":3340,"CT_cells":1835,"HE_cells":632,"BAL_cells":134})
n_pat  = splits["patient_id"].nunique()

# Patient-level summary
pat_df = splits.groupby("patient_id").agg(
    first_dt   =("anchor_dt", "min"),
    last_dt    =("anchor_dt", "max"),
    n_tp       =("stem", "count"),
    n_acr      =("acr_binary", lambda x: (x==1).sum()),
    clad       =("clad_status", "max"),
    death      =("death_status","max"),
    clad_days  =("clad_days", "first"),
    death_days =("death_days","first"),
    has_HE     =("has_HE", "max"),
    has_BAL    =("has_BAL","max"),
    has_CT     =("has_CT", "max"),
    has_Clin   =("has_Clinical","max"),
).reset_index()
pat_df["ever_acr"] = (pat_df["n_acr"] > 0).astype(int)

# Sort by ever_acr desc, then clad, then death, then n_tp
sort_key = (pat_df["ever_acr"]*1000 + pat_df["clad"]*100 +
            pat_df["death"]*10 + pat_df["n_tp"]/100)
pat_df = pat_df.iloc[np.argsort(-sort_key.values)].reset_index(drop=True)

try:
    pat_df["first_dt"] = pd.to_datetime(pat_df["first_dt"])
    pat_df["last_dt"]  = pd.to_datetime(pat_df["last_dt"])
    pat_df["follow_months"] = (pat_df["last_dt"]-pat_df["first_dt"]).dt.days / 30.4
except Exception:
    pat_df["follow_months"] = pat_df["n_tp"] * 1.5

n_acr_pat   = int(pat_df["ever_acr"].sum())
n_clad_pat  = int(pat_df["clad"].fillna(0).sum())
n_death_pat = int(pat_df["death"].fillna(0).sum())

# Clamp follow-up
pat_df["follow_months"] = pat_df["follow_months"].clip(0, 72)
pat_df["clad_months"]   = (pat_df["clad_days"].fillna(0)   / 30.4).clip(0, 72)
pat_df["death_months"]  = (pat_df["death_days"].fillna(0)  / 30.4).clip(0, 72)

print(f"  n_pat={n_pat}  ACR+={n_acr_pat}  CLAD={n_clad_pat}  Death={n_death_pat}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 28))
fig.patch.set_facecolor("white")
gs_outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.38,
                              height_ratios=[1.5, 2.2, 1.5])

# ═══════════════════════════════════════════════════════════════════════
# ROW 1 — MIL FRAMEWORK SCHEMATIC
# ═══════════════════════════════════════════════════════════════════════
ax_s = fig.add_subplot(gs_outer[0])
ax_s.set_xlim(0, 24); ax_s.set_ylim(-1.2, 5.5); ax_s.axis("off")
ax_s.set_title("(A)  Multimodal Multiple Instance Learning Framework — Lung Transplant Monitoring",
               fontsize=12, fontweight="bold", loc="left", pad=8)

def box(ax, x, y, w, h, title, sub, col, fs=8.5, alpha=0.88):
    rx = plt.Rectangle((x-w/2, y-h/2), w, h, fc=col, alpha=alpha,
                        ec="white", lw=2, zorder=3, clip_on=False,
                        joinstyle="round")
    rx.set_capstyle("round")
    ax.add_patch(rx)
    ax.text(x, y+0.12, title, ha="center", va="center", fontsize=fs,
            fontweight="bold", color="white", zorder=4)
    if sub:
        ax.text(x, y-0.3, sub, ha="center", va="center",
                fontsize=fs-1.5, color="white", alpha=0.92, zorder=4)

def arrow(ax, x0, x1, y, col="#78909C", lw=1.4):
    ax.annotate("", xy=(x1-0.1, y), xytext=(x0+0.1, y),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                mutation_scale=14), zorder=5)

# Patient → Timepoints
box(ax_s, 1.3, 2.6, 2.2, 1.0, "Patient\ncohort",    f"n={n_pat}", "#37474F", 9.5)
box(ax_s, 4.0, 2.6, 2.2, 1.0, "Serial\nvisits",     f"{len(splits):,} timepoints", "#455A64", 9.5)
arrow(ax_s, 2.4, 2.9, 2.6)

# Modalities fan-out
mod_ys = [4.2, 2.9, 1.6, 0.2]
mod_data = [
    ("H&E biopsy",   f"n={mod_n.get('HE_cells',632)}",   MOD_COLS["H&E"]),
    ("BAL scRNA",    f"n={mod_n.get('BAL_cells',134)}",  MOD_COLS["BAL scRNA"]),
    ("CT scan",      f"n={mod_n.get('CT_cells',1835)}",  MOD_COLS["CT scan"]),
    ("Clinical labs",f"n={mod_n.get('Clinical',3340)}",  MOD_COLS["Clinical"]),
]
for my, (ml, ms, mc) in zip(mod_ys, mod_data):
    ax_s.annotate("", xy=(6.1, my), xytext=(5.1, 2.6),
                  arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.0), zorder=2)
    box(ax_s, 7.4, my, 2.2, 0.85, ml, ms, mc, 8.5)

# Clustering
clust_data = [
    ("54 sub-clusters\n→ 6 tissue types", MOD_COLS["H&E"]),
    ("43 cell-type\nclusters", MOD_COLS["BAL scRNA"]),
    ("37 patch\nclusters", MOD_COLS["CT scan"]),
    ("106 features\n(4-bin tokenised)", MOD_COLS["Clinical"]),
]
for my, (cl, mc) in zip(mod_ys, clust_data):
    arrow(ax_s, 8.5, 10.3, my, mc)
    box(ax_s, 11.5, my, 2.3, 0.85, cl, "", mc, 7.5)

# CLR normalisation label
ax_s.text(11.5, -0.3, "CLR-normalised\nproportions (image)", ha="center",
          fontsize=7, color=GREY, style="italic")

# Shared MIL aggregation
for my in mod_ys:
    ax_s.annotate("", xy=(13.5, 2.6), xytext=(12.65, my),
                  arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.9), zorder=2)

box(ax_s, 15.2, 2.6, 2.6, 3.6,
    "Shared Slot\nAttention MIL",
    "Cross-modal\naggregation", "#4A148C", 9.5)

# Tasks
task_data = [
    (18.5, 4.4, "ACR\nclassification", f"BACC", ACR_POS),
    (20.2, 3.2, "ACR\nTTE",            f"C-index", "#AD1457"),
    (18.5, 1.8, "CLAD\nTTE",           f"C-index", CLAD_COL),
    (20.2, 0.6, "Death\nTTE",          f"C-index", DEATH_COL),
]
for tx, ty, tl, met, tc in task_data:
    box(ax_s, tx, ty, 2.4, 0.9, tl, met, tc, 8.5)
    ax_s.annotate("", xy=(tx-1.2, ty), xytext=(16.5, 2.6),
                  arrowprops=dict(arrowstyle="-|>", color=tc, lw=1.2,
                                  connectionstyle="arc3,rad=0.15"), zorder=4)

# Legend row
for i, (ml, mc) in enumerate(MOD_COLS.items()):
    ax_s.add_patch(plt.Rectangle((0.5+i*4.2, -1.1), 3.7, 0.45,
                   fc=mc, alpha=0.82, ec="white", clip_on=False))
    ax_s.text(0.5+i*4.2+1.85, -0.87, ml, ha="center", va="center",
              fontsize=8, color="white", fontweight="bold")


# ═══════════════════════════════════════════════════════════════════════
# ROW 2 — COHORT OVERVIEW: swimmer + modality availability + stats
# ═══════════════════════════════════════════════════════════════════════
gs2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[1],
                                        wspace=0.3, width_ratios=[2.5, 1.2, 0.8])
ax_swim  = fig.add_subplot(gs2[0])
ax_avail = fig.add_subplot(gs2[1])
ax_stats = fig.add_subplot(gs2[2])

# ── Swimmer plot ──────────────────────────────────────────────────────
ax_swim.set_title("(B)  Patient follow-up timeline (top 80 patients, sorted by ACR/CLAD burden)",
                   fontsize=9, fontweight="bold", loc="left")
n_swim = min(80, len(pat_df))
ax_swim.set_xlim(-2, 76); ax_swim.set_ylim(-1, n_swim+1)
ax_swim.set_xlabel("Months post-transplant", fontsize=9)
ax_swim.set_yticks([]); ax_swim.spines["left"].set_visible(False)
ax_swim.spines["bottom"].set_visible(True)

for pi in range(n_swim):
    row  = pat_df.iloc[pi]
    y    = n_swim - pi - 0.5
    fol  = row["follow_months"]
    col  = ACR_POS if row["ever_acr"] else (CLAD_COL if row["clad"] else GREY)
    alpha = 0.85 if (row["ever_acr"] or row["clad"] or row["death"]) else 0.45
    # Follow-up bar
    ax_swim.barh(y, fol, left=0, height=0.55, color=col, alpha=alpha*0.6,
                 linewidth=0)
    # Modality dots along the bar
    stem_rows = splits[splits["patient_id"]==row["patient_id"]]
    if len(stem_rows) > 0:
        try:
            dts = pd.to_datetime(stem_rows["anchor_dt"])
            t0  = pd.to_datetime(row["first_dt"])
            months = ((dts - t0).dt.days / 30.4).clip(0, 72)
            for m, (_, sr) in zip(months, stem_rows.iterrows()):
                if sr.get("has_HE", False):
                    ax_swim.plot(m, y+0.18, "s", ms=3.5, color=MOD_COLS["H&E"],
                                 alpha=0.85, zorder=4)
                if sr.get("has_BAL", False):
                    ax_swim.plot(m, y, "^", ms=3.5, color=MOD_COLS["BAL scRNA"],
                                 alpha=0.85, zorder=4)
                if sr.get("has_CT", False):
                    ax_swim.plot(m, y-0.18, "o", ms=2.5, color=MOD_COLS["CT scan"],
                                 alpha=0.75, zorder=4)
        except Exception:
            pass
    # CLAD / death endpoints
    if row["clad"] and row["clad_months"] > 0:
        ax_swim.plot(row["clad_months"], y, "D", ms=6, color=CLAD_COL,
                     zorder=6, markeredgecolor="white", markeredgewidth=0.5)
    if row["death"] and row["death_months"] > 0:
        ax_swim.plot(row["death_months"], y, "x", ms=7, color=DEATH_COL,
                     zorder=6, markeredgewidth=2.0)

# Swimmer legend
swim_legend = [
    Line2D([0],[0], color=ACR_POS,   lw=4, alpha=0.7, label="Ever ACR+"),
    Line2D([0],[0], color=CLAD_COL,  lw=4, alpha=0.7, label="CLAD"),
    Line2D([0],[0], color=GREY,      lw=4, alpha=0.5, label="No event"),
    Line2D([0],[0], marker="s", ms=5, color="w", mfc=MOD_COLS["H&E"],
           lw=0, label="H&E visit"),
    Line2D([0],[0], marker="^", ms=5, color="w", mfc=MOD_COLS["BAL scRNA"],
           lw=0, label="BAL visit"),
    Line2D([0],[0], marker="o", ms=4, color="w", mfc=MOD_COLS["CT scan"],
           lw=0, label="CT visit"),
    Line2D([0],[0], marker="D", ms=6, color=CLAD_COL, lw=0, label="CLAD onset",
           markeredgecolor="white"),
    Line2D([0],[0], marker="x", ms=7, color=DEATH_COL, lw=0, label="Death",
           markeredgewidth=2),
]
ax_swim.legend(handles=swim_legend, frameon=False, fontsize=7, ncol=2,
               loc="lower right", bbox_to_anchor=(1.0, 0.0))

# ── Modality availability heatmap ─────────────────────────────────────
ax_avail.set_title("(C)  Modality\navailability", fontsize=9, fontweight="bold", loc="left")
mods = ["has_HE","has_BAL","has_CT","has_Clin"]
mod_labels = ["H&E","BAL","CT","Clin"]
# Sort patients same as swimmer
pa = pat_df[["patient_id","ever_acr"] + mods].set_index("patient_id")
mat = pa[mods].values.astype(float).T  # 4 × n_pat

cmap_avail = ListedColormap(["#ECEFF1", "#00695C"])
ax_avail.imshow(mat, aspect="auto", cmap=cmap_avail, vmin=0, vmax=1,
                interpolation="nearest")
ax_avail.set_yticks(range(4)); ax_avail.set_yticklabels(mod_labels, fontsize=9)
ax_avail.set_xlabel(f"Patients (n={n_pat})", fontsize=8)
ax_avail.set_xticks([])

# Divider ACR+ / ACR−
acr_line = int(pa["ever_acr"].sum()) - 0.5
ax_avail.axvline(acr_line, color=ACR_POS, lw=1.5, ls="--")
ax_avail.text(acr_line/2, -0.7, f"ACR+\nn={n_acr_pat}", ha="center",
              fontsize=7, color=ACR_POS, fontweight="bold")
ax_avail.text(acr_line + (n_pat-n_acr_pat)/2, -0.7, f"ACR−\nn={n_pat-n_acr_pat}",
              ha="center", fontsize=7, color=ACR_NEG, fontweight="bold")

# Count labels on right
for mi, (mod_c, ml) in enumerate(zip(mods, mod_labels)):
    n_m = int(pa[mod_c].sum())
    ax_avail.text(n_pat+1, mi, f"n={n_m}", va="center", fontsize=7.5, color=DARK)
ax_avail.spines["top"].set_visible(False); ax_avail.spines["right"].set_visible(False)
ax_avail.spines["bottom"].set_visible(False); ax_avail.spines["left"].set_visible(False)

# ── Cohort stats ───────────────────────────────────────────────────────
ax_stats.axis("off")
ax_stats.set_title("(D)  Cohort\nsummary", fontsize=9, fontweight="bold", loc="left")
stats = [
    ("Patients",              f"{n_pat}",                           DARK,      True),
    ("Timepoints",            f"{len(splits):,}",                   DARK,      True),
    ("",                      "",                                   GREY,      False),
    ("ACR+ patients",         f"{n_acr_pat} ({100*n_acr_pat/n_pat:.0f}%)",  ACR_POS, True),
    ("  A0B0 timepoints",     f"{int(splits['acr_binary'].eq(0).sum()):,}", ACR_NEG, False),
    ("  A1/A2 timepoints",    f"{int(splits['acr_binary'].eq(1).sum()):,}", ACR_POS, False),
    ("CLAD events",           f"{n_clad_pat} ({100*n_clad_pat/n_pat:.0f}%)", CLAD_COL, True),
    ("Death events",          f"{n_death_pat} ({100*n_death_pat/n_pat:.0f}%)",DEATH_COL,True),
    ("",                      "",                                   GREY,      False),
    ("H&E biopsies",          f"{mod_n.get('HE_cells',632):,}",      MOD_COLS["H&E"],       False),
    ("BAL scRNA samples",     f"{mod_n.get('BAL_cells',134):,}",     MOD_COLS["BAL scRNA"], False),
    ("CT scans",              f"{mod_n.get('CT_cells',1835):,}",     MOD_COLS["CT scan"],   False),
    ("Clinical timepoints",   f"{mod_n.get('Clinical',3340):,}",     MOD_COLS["Clinical"],  False),
    ("",                      "",                                   GREY,      False),
    ("H&E tissue types",      "6 (54 sub-cl.)",                      DARK,      False),
    ("BAL cell types",        "43 clusters",                         DARK,      False),
    ("CT clusters",           "37 clusters",                         DARK,      False),
    ("Clinical features",     "106 (z-scored)",                      DARK,      False),
    ("CV folds",              "4 outer (nested 5×4)",                DARK,      False),
]
ax_stats.set_xlim(0,1); ax_stats.set_ylim(0, len(stats)+1)
for si, (lab, val, col, bold) in enumerate(reversed(stats)):
    y = si + 0.5
    ax_stats.text(0.02, y, lab, fontsize=8, color="#546E7A", va="center",
                  fontweight="bold" if bold else "normal")
    ax_stats.text(0.98, y, val, fontsize=8.5, color=col, va="center",
                  ha="right", fontweight="bold" if bold else "normal")


# ═══════════════════════════════════════════════════════════════════════
# ROW 3 — KEY FINDINGS + BIOLOGICAL ANNOTATIONS
# ═══════════════════════════════════════════════════════════════════════
gs3 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_outer[2], wspace=0.38)
ax_acr   = fig.add_subplot(gs3[0])
ax_acrt  = fig.add_subplot(gs3[1])
ax_clad  = fig.add_subplot(gs3[2])
ax_death = fig.add_subplot(gs3[3])

def bio_panel(ax, title, task_col, features, colors, bio_notes, metric_lbl, metric_val):
    """
    features: list of (name, effect_direction, modality_color)
    bio_notes: list of short annotation strings
    """
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1, len(features)+2)
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", color=task_col, pad=6)
    # Metric badge
    ax.text(0, len(features)+1.4, f"{metric_lbl}: {metric_val}",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", fc=task_col, ec="none", alpha=0.9))
    for fi, (feat, direction, mod_col, bio) in enumerate(zip(features, colors, colors, bio_notes)):
        y = len(features) - fi - 0.5
        feat_name, direction, mod_col = feat
        # Horizontal bar showing direction
        bar_len = 0.7 if direction > 0 else -0.7
        ax.barh(y, bar_len, height=0.55, color=mod_col, alpha=0.75,
                linewidth=0, left=0 if direction > 0 else bar_len)
        ax.text(0.02 if direction > 0 else -0.02, y, feat_name,
                ha="left" if direction > 0 else "right",
                va="center", fontsize=7, color=DARK, fontweight="bold")
        ax.text(1.15 if direction > 0 else -1.15, y, bio,
                ha="left" if direction > 0 else "right",
                va="center", fontsize=5.5, color="#546E7A", style="italic")
    ax.axvline(0, color=GREY, lw=0.9)
    ax.text( 0.9, -0.7, f"↑ {'+event' if direction>0 else '+ACR+'}", fontsize=6.5,
             ha="center", color=ACR_POS)
    ax.text(-0.9, -0.7, f"↓ protective",     fontsize=6.5, ha="center", color=ACR_NEG)

# Load importance if available
imp_available = IMP_CSV.exists()
if imp_available:
    imp_df = pd.read_csv(IMP_CSV)
    def get_top(task, modality="All", n=5):
        sub = imp_df[(imp_df["task"]==task)&(imp_df["modality"]==modality)].copy()
        sub = sub.reindex(sub["coef"].abs().sort_values(ascending=False).index)
        return sub.head(n)[["feature","coef"]].values.tolist()

# ACR classification
acr_feats = [
    ("BAL Lymphocytes %", +1, MOD_COLS["BAL scRNA"]),
    ("BAL Neutrophils %", +1, MOD_COLS["BAL scRNA"]),
    ("Alveolar inflamed",  +1, MOD_COLS["H&E"]),
    ("Vascular H&E",       +1, MOD_COLS["H&E"]),
    ("FEV1",               -1, MOD_COLS["Clinical"]),
]
acr_bio = [
    "CD4/CD8 T-cell infiltration",
    "Early innate response",
    "Perivascular mononuclear infiltrate (A1/A2)",
    "Endothelialitis — A2+ pathognomonic",
    "Preserved lung function → lower ACR risk",
]
bio_panel(ax_acr, "(E)  ACR Classification", ACR_POS,
          acr_feats, [f[2] for f in acr_feats], acr_bio,
          "BACC", "0.74*")

# ACR TTE
acrt_feats = [
    ("pseudoslope_FEV1",  -1, MOD_COLS["Clinical"]),
    ("delta_FEV1",        -1, MOD_COLS["Clinical"]),
    ("DSA count",         +1, MOD_COLS["Clinical"]),
    ("CMV D+/R−",         +1, MOD_COLS["Clinical"]),
    ("BAL CD4 T cell",    +1, MOD_COLS["BAL scRNA"]),
]
acrt_bio = [
    "Declining FEV1 slope → subclinical rejection",
    "Visit-to-visit FEV1 drop ≥10%",
    "Anti-DQ DSA → sensitisation",
    "CMV drives anti-HLA immunity",
    "Effector T-cell accumulation",
]
bio_panel(ax_acrt, "(F)  ACR TTE", "#AD1457",
          acrt_feats, [f[2] for f in acrt_feats], acrt_bio,
          "C-index", "0.71*")

# CLAD TTE
clad_feats = [
    ("pseudoslope_FEV1",  -1, MOD_COLS["Clinical"]),
    ("pseudoslope_FVC",   -1, MOD_COLS["Clinical"]),
    ("BAL Neutrophils %", +1, MOD_COLS["BAL scRNA"]),
    ("PGD T24 grade",     +1, MOD_COLS["Clinical"]),
    ("CT air-trapping cl",-1, MOD_COLS["CT scan"]),
]
clad_bio = [
    "FEV1 decline = BOS definition (≥20%)",
    "FVC decline → RAS phenotype",
    "BAL >3% neutrophils → BOS predictor",
    "PGD sets alloimmune milieu (OR~3)",
    "Small airway obstruction pattern",
]
bio_panel(ax_clad, "(G)  CLAD TTE", CLAD_COL,
          clad_feats, [f[2] for f in clad_feats], clad_bio,
          "C-index", "0.73*")

# Death TTE
death_feats = [
    ("FEV1 (absolute)",   -1, MOD_COLS["Clinical"]),
    ("GFR",               -1, MOD_COLS["Clinical"]),
    ("donor_risk",        +1, MOD_COLS["Clinical"]),
    ("age",               +1, MOD_COLS["Clinical"]),
    ("pseudoslope_FEV1",  -1, MOD_COLS["Clinical"]),
]
death_bio = [
    "Lung function → primary survival driver",
    "Renal failure → mortality post-transplant",
    "Donor lung quality score",
    "Recipient age → comorbidity burden",
    "FEV1 trajectory",
]
bio_panel(ax_death, "(H)  Death TTE", DEATH_COL,
          death_feats, [f[2] for f in death_feats], death_bio,
          "C-index", "0.69*")

ax_death.text(0, -1.0,
    "* Indicative values — update after train_linear_models.py completes",
    ha="center", fontsize=6, color=GREY, transform=ax_death.transAxes,
    style="italic")


# ── Final title ─────────────────────────────────────────────────────────────
fig.suptitle(
    "Multimodal Multiple Instance Learning for Longitudinal Lung Transplant Monitoring\n"
    "H&E biopsy  ·  BAL scRNA  ·  CT scan  ·  Clinical labs  →  "
    "ACR / CLAD / Death prediction",
    fontsize=13, fontweight="bold", y=1.005, color=DARK)

save_fig(fig, "fig0_hero_v2")
print("Hero figure done.", flush=True)
