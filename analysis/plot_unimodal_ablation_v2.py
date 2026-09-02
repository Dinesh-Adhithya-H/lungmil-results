"""
Unimodal ablation — LongMIL only.
For each task: Full multimodal bar + 4 unimodal bars + best-ABMIL reference line.
Tells the story: LongMIL with all modalities beats the best single-modality model.
Run via: sbatch analysis/submit_unimodal_ablation_v2.sh
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch, FancyBboxPatch
from scipy import stats

ROOT     = Path(__file__).resolve().parent.parent
PRED     = ROOT / "results" / "predictions"
LIN_CSV  = ROOT / "results" / "linear_models" / "metrics_summary.csv"
ABL_CSV  = ROOT / "interpretability" / "unimodal_ablation" / "unimodal_ablation_summary.csv"
OUT_DIR  = ROOT / "figures" / "interpretability" / "unimodal_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG = "#FAF6F2"

LONGMIL_COLOR  = "#B71C1C"   # dark red — full multimodal LongMIL
MOD_COLORS     = {
    "HE":       "#E64A19",
    "BAL":      "#1565C0",
    "CT":       "#2E7D32",
    "Clinical": "#7B1FA2",
}
ABMIL_LINE_COLOR = "#0D47A1"   # dark blue reference line

MOD_ORDER  = ["HE", "BAL", "CT", "Clinical"]
SPLIT_COLS = ["s0", "s1", "s2", "s3", "s4"]

TASK_CFG = {
    "acr_cls":   {
        "bench_file": "comparison_acr_cls.csv",
        "metric_col": "mean",  "split_metric": "bacc",
        "abl_metric": "bacc",  "abl_suffix":   "acr_cls",
        "lin_task":   "ACR",   "lin_metric":   "bacc",
        "metric_label": "BACC",
        "label": "ACR Classification",
    },
    "acr_surv":  {
        "bench_file": "comparison_acr_surv.csv",
        "metric_col": "mean",  "split_metric": "c_index",
        "abl_metric": "c_index", "abl_suffix": "acr_surv",
        "lin_task":   "ACR_TTE", "lin_metric": "cindex",
        "metric_label": "C-index",
        "label": "ACR Survival",
    },
    "clad_surv": {
        "bench_file": "comparison_clad.csv",
        "metric_col": "mean",  "split_metric": "c_index",
        "abl_metric": "c_index", "abl_suffix": "clad_surv",
        "lin_task":   "CLAD",  "lin_metric":  "cindex",
        "metric_label": "C-index",
        "label": "CLAD Survival",
    },
    "death_surv": {
        "bench_file": "comparison_death.csv",
        "metric_col": "mean",  "split_metric": "c_index",
        "abl_metric": "c_index", "abl_suffix": "death_surv",
        "lin_task":   "Death", "lin_metric":  "cindex",
        "metric_label": "C-index",
        "label": "Post-Transplant Mortality",
    },
}

CSV_TO_DISPLAY = {
    "P1 HE":       "ABMIL HE",   "P1 BAL":      "ABMIL BAL",
    "P1 CT":       "ABMIL CT",   "P1 Clinical":  "ABMIL Clinical",
    "LongMK-v2":   "LongMIL",
}
ABMIL_UNIMODAL = {"ABMIL HE", "ABMIL BAL", "ABMIL CT", "ABMIL Clinical"}


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_benchmark(bench_file):
    """Load LongMIL full-multimodal splits and per-modality ABMIL means from benchmark CSV."""
    df = pd.read_csv(PRED / bench_file)
    longmil_splits = []
    # per-modality ABMIL: {mod: mean}
    abmil_per_mod = {}

    for _, row in df.iterrows():
        name = CSV_TO_DISPLAY.get(str(row["model"]).strip())
        if name is None:
            continue
        splits = []
        for s in SPLIT_COLS:
            try:
                splits.append(float(row.get(s, np.nan)))
            except (TypeError, ValueError):
                splits.append(np.nan)
        if name == "LongMIL":
            longmil_splits = splits
        elif name in ABMIL_UNIMODAL:
            mod = name.replace("ABMIL ", "")   # "HE", "BAL", "CT", "Clinical"
            abmil_per_mod[mod] = float(np.nanmean(splits))

    # best ABMIL (for BH test) — highest mean across modalities
    if abmil_per_mod:
        best_mod        = max(abmil_per_mod, key=abmil_per_mod.get)
        best_abmil_mean = abmil_per_mod[best_mod]
    else:
        best_mod, best_abmil_mean = "", 0.0

    return longmil_splits, best_abmil_mean, abmil_per_mod, best_mod


def load_longmil_unimodal(abl_suffix, abl_metric):
    """Load LongMIL unimodal ablation per-modality splits from summary CSV."""
    if not ABL_CSV.exists():
        return {}
    df = pd.read_csv(ABL_CSV)
    rows = df[(df["variant"] == "longitudinal_mk_no_alibi_v2") &
              (df["task"]    == abl_suffix) &
              (df["metric"]  == abl_metric)]
    out = {}
    for mod, grp in rows.groupby("modality"):
        if mod not in MOD_ORDER:
            continue
        if "value" in grp.columns and grp["value"].notna().any():
            vals = grp["value"].dropna().tolist()
        elif "mean" in grp.columns and grp["mean"].notna().any():
            r = grp.iloc[0]
            vals = [float(r["mean"])]
        else:
            continue
        vals_clean = [v for v in vals if not np.isnan(float(v)) and 0.0 <= float(v) <= 1.0]
        if not vals_clean:
            continue
        out[mod] = {"mean": float(np.nanmean(vals_clean)), "std": float(np.nanstd(vals_clean)),
                    "splits": vals_clean}
    return out


def bh_correct(pvals):
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    adj = np.minimum(1.0, pvals * n / ranked)
    adj_sorted = adj[order]
    for k in range(n - 2, -1, -1):
        adj_sorted[k] = min(adj_sorted[k], adj_sorted[k + 1])
    adj[order] = adj_sorted
    return adj


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ── Main figure: 4-panel ──────────────────────────────────────────────────────

FS_TITLE  = 28
FS_LABEL  = 22
FS_TICK   = 19
FS_LEGEND = 18
FS_ANNOT  = 15
FS_SIG    = 18

fig, axes = plt.subplots(1, 4, figsize=(52, 14), facecolor=BG)
fig.patch.set_facecolor(BG)
fig.suptitle(
    "LongMIL — Unimodal Ablation  ·  Full multimodal vs. single-modality input",
    fontsize=32, fontweight="bold", y=1.01,
)

# Collect all (longmil_splits, abmil_splits) pairs across tasks for BH correction
all_tests = []   # list of (task_key, lm_splits, ab_splits)
for tk, cfg in TASK_CFG.items():
    lm_splits, best_ab_mean, ab_splits, best_ab_mod = load_benchmark(cfg["bench_file"])
    all_tests.append((tk, lm_splits, ab_splits, best_ab_mod))

raw_pvals = []
for _, lm_s, ab_per_mod, best_mod in all_tests:
    lm = np.array([v for v in lm_s if not np.isnan(v)])
    best_ab_mean = max(ab_per_mod.values()) if ab_per_mod else 0.0
    # use best ABMIL mean as a scalar reference; paired test not meaningful here
    # so just test LongMIL splits against best ABMIL mean as a threshold
    if len(lm) >= 2 and best_ab_mean > 0:
        from scipy.stats import ttest_1samp
        _, p = ttest_1samp(lm, best_ab_mean, alternative="greater")
        raw_pvals.append(p)
    else:
        raw_pvals.append(1.0)

adj_pvals = bh_correct(np.array(raw_pvals))

for ax_idx, ((task_key, cfg), (_, lm_splits, ab_per_mod, best_mod), adj_p) in enumerate(
        zip(TASK_CFG.items(), all_tests, adj_pvals)):

    ax = axes[ax_idx]
    ax.set_facecolor(BG)

    lm_unimodal = load_longmil_unimodal(cfg["abl_suffix"], cfg["abl_metric"])
    lm_full_mean = float(np.nanmean([v for v in lm_splits if not np.isnan(v)]))
    lm_full_std  = float(np.nanstd( [v for v in lm_splits if not np.isnan(v)]))

    # ── X positions: Full | HE | BAL | CT | Clinical ─────────────────────────
    labels   = ["Full\n(all 4 mod.)"] + MOD_ORDER
    x_pos    = np.arange(len(labels))
    bar_w    = 0.55

    # Full multimodal bar
    ax.bar(x_pos[0], lm_full_mean, width=bar_w, color=LONGMIL_COLOR,
           alpha=0.90, zorder=2, linewidth=2, edgecolor="#7B0000")
    if lm_full_std > 0:
        ax.errorbar(x_pos[0], lm_full_mean, yerr=lm_full_std,
                    fmt="none", ecolor="#222", elinewidth=1.8, capsize=6, zorder=4)
    for sv in lm_splits:
        if not np.isnan(sv):
            ax.scatter(x_pos[0], sv, s=40, color="white", edgecolors=LONGMIL_COLOR,
                       linewidths=1.2, zorder=5)

    # Unimodal bars
    for mi, mod in enumerate(MOD_ORDER):
        xi   = x_pos[mi + 1]
        col  = MOD_COLORS[mod]
        d    = lm_unimodal.get(mod)
        if d is None:
            continue
        ax.bar(xi, d["mean"], width=bar_w, color=col, alpha=0.80, zorder=2,
               linewidth=1.2, edgecolor=col)
        if d["std"] > 0:
            ax.errorbar(xi, d["mean"], yerr=d["std"],
                        fmt="none", ecolor="#222", elinewidth=1.8, capsize=6, zorder=4)
        for sv in d["splits"]:
            if not np.isnan(sv):
                ax.scatter(xi, sv, s=40, color="white", edgecolors=col,
                           linewidths=1.2, zorder=5)

    # ── Per-modality ABMIL reference diamonds ─────────────────────────────────
    for mi, mod in enumerate(MOD_ORDER):
        xi = x_pos[mi + 1]
        abmil_val = ab_per_mod.get(mod)
        if abmil_val is not None and abmil_val > 0:
            ax.scatter(xi, abmil_val, marker="D", s=110, color=ABMIL_LINE_COLOR,
                       edgecolors="white", linewidths=1.2, zorder=7)
            ax.text(xi, abmil_val + 0.025, f"{abmil_val:.2f}",
                    ha="center", va="bottom", fontsize=FS_ANNOT - 1,
                    color=ABMIL_LINE_COLOR, fontweight="bold")

    # ── Significance star above full multimodal bar ───────────────────────────
    stars = sig_stars(adj_p)
    y_star = lm_full_mean + lm_full_std + 0.04
    if stars != "ns":
        ax.text(x_pos[0], y_star, stars, ha="center", va="bottom",
                fontsize=FS_SIG + 2, color=LONGMIL_COLOR, fontweight="bold", zorder=6)
    else:
        ax.text(x_pos[0], y_star, "ns", ha="center", va="bottom",
                fontsize=FS_ANNOT, color="#888888", zorder=6)

    # ── Chance line ───────────────────────────────────────────────────────────
    ax.axhline(0.5, color="#aaa", linewidth=1.2, linestyle=":", alpha=0.7)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=FS_TICK, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(cfg["label"], fontsize=FS_TITLE, fontweight="bold", pad=12)
    if ax_idx == 0:
        ax.set_ylabel(cfg["metric_label"], fontsize=FS_LABEL)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Subtle rectangle around full multimodal bar ───────────────────────────
    rect = FancyBboxPatch(
        (x_pos[0] - bar_w / 2 - 0.03, 0.0),
        bar_w + 0.06,
        lm_full_mean + lm_full_std + 0.07,
        boxstyle="round,pad=0.01",
        linewidth=2.5, edgecolor=LONGMIL_COLOR,
        facecolor="none", zorder=6, clip_on=False,
    )
    ax.add_patch(rect)

# ── Legend + caption ──────────────────────────────────────────────────────────
legend_handles = [
    Patch(facecolor=LONGMIL_COLOR, edgecolor="#7B0000", linewidth=2,
          label="LongMIL — Full multimodal (all 4 modalities)"),
] + [
    Patch(facecolor=MOD_COLORS[m], label=f"LongMIL — {m} only") for m in MOD_ORDER
] + [
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=ABMIL_LINE_COLOR,
               markersize=10, label="ABMIL (same modality) baseline"),
]

fig.legend(handles=legend_handles, fontsize=FS_LEGEND, ncol=3,
           loc="lower center", bbox_to_anchor=(0.5, -0.04),
           framealpha=0.95, edgecolor="#ccc",
           handlelength=2.0, handleheight=1.4, borderpad=1.0, columnspacing=2.0)

ablation_note = (
    "Ablation: at inference, only one modality is provided to LongMIL — remaining modalities are masked.\n"
    "Compares full multimodal LongMIL (red, outlined) vs. each single-modality input. "
    "Blue diamonds show the ABMIL unimodal baseline for the same modality.\n"
    "* p<0.05  ** p<0.01  *** p<0.001  (one-sided t-test vs. best ABMIL mean, BH-corrected across 4 tasks)"
)
fig.text(0.5, -0.10, ablation_note, ha="center", va="top",
         fontsize=FS_ANNOT, color="#555", style="italic",
         transform=fig.transFigure)

fig.tight_layout(rect=[0, 0.06, 1, 1])
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"unimodal_ablation_v2_all.{ext}", dpi=200,
                bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved unimodal_ablation_v2_all")

# ── Per-task figures (same design, single panel) ──────────────────────────────
for tk_idx, (task_key, cfg) in enumerate(TASK_CFG.items()):
    _, lm_splits, ab_per_mod2, _ = all_tests[tk_idx]
    lm_unimodal  = load_longmil_unimodal(cfg["abl_suffix"], cfg["abl_metric"])
    def _safe_float(v):
        try: return float(v)
        except (TypeError, ValueError): return None
    lm_splits_f  = [_safe_float(v) for v in lm_splits]
    lm_splits_f  = [v for v in lm_splits_f if v is not None and not np.isnan(v)]
    lm_full_mean = float(np.nanmean(lm_splits_f)) if lm_splits_f else 0.0
    lm_full_std  = float(np.nanstd(lm_splits_f))  if lm_splits_f else 0.0
    best_ab_mean2 = max(ab_per_mod2.values()) if ab_per_mod2 else 0.0  # kept for rect height

    fig2, ax2 = plt.subplots(figsize=(14, 9), facecolor=BG)
    fig2.patch.set_facecolor(BG)
    ax2.set_facecolor(BG)

    labels  = ["Full\n(all 4 mod.)"] + MOD_ORDER
    x_pos   = np.arange(len(labels))
    bar_w   = 0.55

    ax2.bar(x_pos[0], lm_full_mean, width=bar_w, color=LONGMIL_COLOR,
            alpha=0.90, zorder=2, linewidth=2, edgecolor="#7B0000")
    if lm_full_std > 0:
        ax2.errorbar(x_pos[0], lm_full_mean, yerr=lm_full_std,
                     fmt="none", ecolor="#222", elinewidth=2.0, capsize=8, zorder=4)
    for sv in lm_splits_f:
        ax2.scatter(x_pos[0], sv, s=60, color="white", edgecolors=LONGMIL_COLOR,
                        linewidths=1.5, zorder=5)

    for mi, mod in enumerate(MOD_ORDER):
        xi  = x_pos[mi + 1]
        col = MOD_COLORS[mod]
        d   = lm_unimodal.get(mod)
        if d is None:
            continue
        ax2.bar(xi, d["mean"], width=bar_w, color=col, alpha=0.80, zorder=2,
                linewidth=1.2, edgecolor=col)
        if d["std"] > 0:
            ax2.errorbar(xi, d["mean"], yerr=d["std"],
                         fmt="none", ecolor="#222", elinewidth=2.0, capsize=8, zorder=4)
        for sv in d["splits"]:
            if not np.isnan(sv):
                ax2.scatter(xi, sv, s=60, color="white", edgecolors=col,
                            linewidths=1.5, zorder=5)

    for mi, mod in enumerate(MOD_ORDER):
        xi2 = x_pos[mi + 1]
        abmil_val2 = ab_per_mod2.get(mod)
        if abmil_val2 is not None and abmil_val2 > 0:
            ax2.scatter(xi2, abmil_val2, marker="D", s=140, color=ABMIL_LINE_COLOR,
                        edgecolors="white", linewidths=1.5, zorder=7)
            ax2.text(xi2, abmil_val2 + 0.025, f"{abmil_val2:.2f}",
                     ha="center", va="bottom", fontsize=FS_ANNOT,
                     color=ABMIL_LINE_COLOR, fontweight="bold")

    adj_p_task = adj_pvals[tk_idx]
    stars_task = sig_stars(adj_p_task)
    y_star2 = lm_full_mean + lm_full_std + 0.04
    color_star = LONGMIL_COLOR if stars_task != "ns" else "#888888"
    ax2.text(x_pos[0], y_star2, stars_task, ha="center", va="bottom",
             fontsize=FS_SIG + 4, color=color_star, fontweight="bold", zorder=6)

    ax2.axhline(0.5, color="#aaa", linewidth=1.5, linestyle=":", alpha=0.7)

    rect2 = FancyBboxPatch(
        (x_pos[0] - bar_w / 2 - 0.04, 0.3),
        bar_w + 0.08,
        lm_full_mean - 0.3 + lm_full_std + 0.08,
        boxstyle="round,pad=0.01",
        linewidth=3, edgecolor=LONGMIL_COLOR,
        facecolor="none", zorder=6, clip_on=False,
    )
    ax2.add_patch(rect2)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=FS_TICK + 2, fontweight="bold")
    ax2.set_ylim(0.3, 1.05)
    ax2.set_title(f"LongMIL Unimodal Ablation — {cfg['label']}",
                  fontsize=FS_TITLE, fontweight="bold", pad=14)
    ax2.set_ylabel(cfg["metric_label"], fontsize=FS_LABEL)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax2.tick_params(labelsize=FS_TICK)
    ax2.spines[["top", "right"]].set_visible(False)

    fig2.legend(handles=legend_handles, fontsize=FS_LEGEND - 1, ncol=3,
                loc="lower center", bbox_to_anchor=(0.5, -0.06),
                framealpha=0.95, edgecolor="#ccc",
                handlelength=2.0, handleheight=1.4, borderpad=1.0)
    fig2.text(0.5, -0.13,
              "Ablation: only one modality provided — rest masked.  "
              "Blue diamonds = ABMIL baseline for same modality.  "
              "* p<0.05  ** p<0.01  *** p<0.001  (one-sided t-test vs. best ABMIL, BH-corrected)",
              ha="center", fontsize=FS_ANNOT, color="#555", style="italic",
              transform=fig2.transFigure)

    fig2.tight_layout(rect=[0, 0.08, 1, 1])
    for ext in ("png", "pdf"):
        fig2.savefig(OUT_DIR / f"unimodal_ablation_v2_{task_key}.{ext}", dpi=200,
                     bbox_inches="tight", facecolor=BG)
    plt.close(fig2)
    print(f"Saved unimodal_ablation_v2_{task_key}")

print("Done.")
