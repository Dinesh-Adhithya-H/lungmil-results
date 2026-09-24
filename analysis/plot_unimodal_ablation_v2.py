"""
Unimodal ablation — LongMIL only.
Horizontal-bar layout matching benchmark style.

For each task: Full multimodal bar + 4 unimodal bars + per-modality ABMIL diamond.

Fair comparison: LongMIL(mod only) and ABMIL(mod) are BOTH evaluated on the same
modality-specific test subset (patients that have that modality). The paired
t-test compares LongMIL(mod) vs ABMIL(mod) per split, BH-corrected over 4 tasks × 4 mods.

Run via: sbatch analysis/submit_unimodal_ablation_v2.sh
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import matplotlib.lines as mlines
from scipy import stats

ROOT     = Path(__file__).resolve().parent.parent
PRED     = ROOT / "results" / "predictions"
ABL_CSV  = ROOT / "interpretability" / "unimodal_ablation" / "unimodal_ablation_summary.csv"
OUT_DIR  = ROOT / "figures" / "interpretability" / "unimodal_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG = "#FAF6F2"

LONGMIL_COLOR = "#B71C1C"
MOD_COLORS    = {
    "HE":       "#E64A19",
    "BAL":      "#1565C0",
    "CT":       "#2E7D32",
    "Clinical": "#7B1FA2",
}
ABMIL_COLOR = "#0D47A1"

MOD_ORDER  = ["HE", "BAL", "CT", "Clinical"]
SPLIT_COLS = ["s0", "s1", "s2", "s3", "s4"]

ROW_LABELS = ["Full (all 4 mod.)"] + [f"{m} only" for m in MOD_ORDER]
ROW_COLORS = [LONGMIL_COLOR] + [MOD_COLORS[m] for m in MOD_ORDER]
N_ROWS = len(ROW_LABELS)

TASK_CFG = {
    "acr_cls": {
        "bench_file": "comparison_acr_cls.csv",
        "abl_metric": "bacc",    "abl_suffix": "acr_cls",
        "metric_label": "BACC  (0.5 = chance,  1.0 = perfect)",
        "label": "ACR classification",
    },
    "acr_surv": {
        "bench_file": "comparison_acr_surv.csv",
        "abl_metric": "c_index", "abl_suffix": "acr_surv",
        "metric_label": "C-index  (0.5 = chance,  1.0 = perfect)",
        "label": "Time to first ACR",
    },
    "clad_surv": {
        "bench_file": "comparison_clad.csv",
        "abl_metric": "c_index", "abl_suffix": "clad_surv",
        "metric_label": "C-index  (0.5 = chance,  1.0 = perfect)",
        "label": "Time to CLAD",
    },
    "death_surv": {
        "bench_file": "comparison_death.csv",
        "abl_metric": "c_index", "abl_suffix": "death_surv",
        "metric_label": "C-index  (0.5 = chance,  1.0 = perfect)",
        "label": "Time to death",
    },
}

CSV_TO_DISPLAY = {
    "P1 HE":      "ABMIL HE",   "P1 BAL":     "ABMIL BAL",
    "P1 CT":      "ABMIL CT",   "P1 Clinical": "ABMIL Clinical",
    "LongMK-v2":  "LongMIL",
}
ABMIL_UNIMODAL = {"ABMIL HE", "ABMIL BAL", "ABMIL CT", "ABMIL Clinical"}

# ── Font sizes (matching benchmark) ──────────────────────────────────────────
FS_TITLE   = 36
FS_YLABEL  = 26
FS_XLABEL  = 24
FS_TICK    = 23
FS_LEGEND  = 22
FS_SIG     = 21
FS_CAPTION = 18
FS_ANNOT   = 18


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_benchmark(bench_file):
    df = pd.read_csv(PRED / bench_file)
    longmil_splits = []
    abmil_per_mod  = {}
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
            mod = name.replace("ABMIL ", "")
            abmil_per_mod[mod] = {
                "splits": splits,
                "mean":   float(np.nanmean([v for v in splits if not np.isnan(v)])),
                "std":    float(np.nanstd( [v for v in splits if not np.isnan(v)])),
            }
    return longmil_splits, abmil_per_mod


def load_longmil_unimodal(abl_suffix, abl_metric):
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
            vals = [float(grp.iloc[0]["mean"])]
        else:
            continue
        vals_clean = [float(v) for v in vals if not np.isnan(float(v)) and 0.0 <= float(v) <= 1.0]
        if not vals_clean:
            continue
        out[mod] = {
            "mean":   float(np.nanmean(vals_clean)),
            "std":    float(np.nanstd(vals_clean)),
            "splits": vals_clean,
        }
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
    return ""


def _bracket_label(ax, rows, text, color):
    if not rows:
        return
    y_top = min(rows) - 0.45
    y_bot = max(rows) + 0.45
    y_mid = (y_top + y_bot) / 2
    ax.plot([1.04, 1.04], [y_top, y_bot], color=color, linewidth=2.5,
            clip_on=False, zorder=8)
    ax.text(1.06, y_mid, text, ha="left", va="center", fontsize=14,
            fontweight="bold", color=color, clip_on=False, rotation=90)


def plot_ablation_task(ax, task_key, sig_mod_map, all_data, show_ylabel=True):
    cfg = TASK_CFG[task_key]
    lm_splits  = all_data[task_key]["lm_splits"]
    ab_per_mod = all_data[task_key]["ab_per_mod"]
    lm_unimodal = all_data[task_key]["lm_uni"]

    y = np.arange(N_ROWS)

    # ── Group backgrounds ─────────────────────────────────────────────────────
    ax.axhspan(-0.5, 0.5, facecolor="#FCE4EC", alpha=0.40, zorder=0)
    ax.axhspan(0.5, N_ROWS - 0.5, facecolor="#E8F5E9", alpha=0.35, zorder=0)
    ax.axhline(0.5, color="#bbbbbb", linewidth=1.2, linestyle="--", zorder=1)

    # ── Chance line ───────────────────────────────────────────────────────────
    ax.axvline(0.5, color="#888", linewidth=1.2, linestyle=":", alpha=0.8)

    # ── Full multimodal bar (row 0) ───────────────────────────────────────────
    lm_full_valid = [float(v) for v in lm_splits if not np.isnan(float(v))]
    lm_full_mean  = float(np.nanmean(lm_full_valid)) if lm_full_valid else np.nan
    lm_full_std   = float(np.nanstd(lm_full_valid))  if lm_full_valid else 0.0

    if not np.isnan(lm_full_mean):
        ax.barh(y[0], lm_full_mean, height=0.60, color=LONGMIL_COLOR, alpha=0.90, zorder=2)
        if lm_full_std > 0:
            ax.errorbar(lm_full_mean, y[0], xerr=lm_full_std, fmt="none",
                        ecolor="#222", elinewidth=1.5, capsize=5, zorder=4)
        for sv in lm_full_valid:
            ax.scatter(sv, y[0], s=22, color="white", edgecolors=LONGMIL_COLOR,
                       linewidths=1.0, zorder=5, alpha=0.95)

    # ── Unimodal bars + ABMIL diamonds + significance stars ──────────────────
    for mi, mod in enumerate(MOD_ORDER):
        row_y = y[mi + 1]
        col   = ROW_COLORS[mi + 1]
        d     = lm_unimodal.get(mod)

        if d is not None:
            ax.barh(row_y, d["mean"], height=0.60, color=col, alpha=0.82, zorder=2)
            if d["std"] > 0:
                ax.errorbar(d["mean"], row_y, xerr=d["std"], fmt="none",
                            ecolor="#222", elinewidth=1.5, capsize=5, zorder=4)
            for sv in d["splits"]:
                if not np.isnan(float(sv)):
                    ax.scatter(float(sv), row_y, s=22, color="white", edgecolors=col,
                               linewidths=1.0, zorder=5, alpha=0.95)
            # Significance star (LongMIL(mod) > ABMIL(mod), BH-corrected)
            adj_p = sig_mod_map.get(task_key, {}).get(mod, 1.0)
            stars = sig_stars(adj_p)
            if stars:
                x_star = min(d["mean"] + (d["std"] if d["std"] > 0 else 0) + 0.03, 0.98)
                ax.text(x_star, row_y, stars, va="center", ha="left",
                        fontsize=FS_SIG, color="#B71C1C", fontweight="bold", zorder=7)

        # ABMIL reference diamond (slightly above the bar, y + 0.2 which is below after invert)
        ab_d = ab_per_mod.get(mod)
        if ab_d is not None and ab_d["mean"] > 0:
            ax.scatter(ab_d["mean"], row_y - 0.20, marker="D", s=90,
                       color=ABMIL_COLOR, edgecolors="white", linewidths=1.2, zorder=7)
            if ab_d["std"] > 0:
                ax.errorbar(ab_d["mean"], row_y - 0.20, xerr=ab_d["std"], fmt="none",
                            ecolor=ABMIL_COLOR, elinewidth=1.3, capsize=4, zorder=6, alpha=0.7)

    # ── Bracket labels ────────────────────────────────────────────────────────
    _bracket_label(ax, [0],            "Full multimodal", "#B71C1C")
    _bracket_label(ax, list(range(1, N_ROWS)), "Unimodal",  "#2E7D32")

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_yticks(y)
    ax.set_yticklabels(ROW_LABELS, fontsize=FS_YLABEL)
    if show_ylabel:
        ax.set_ylabel("Input modality", fontsize=FS_XLABEL)
    ax.set_xlabel(cfg["metric_label"], fontsize=FS_XLABEL)
    ax.set_title(cfg["label"], fontsize=FS_TITLE, fontweight="bold", pad=10)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_YLABEL)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor(BG)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.18)


def make_legend():
    handles = [Patch(facecolor=LONGMIL_COLOR, label="LongMIL — Full (all 4 mod.)")]
    for m in MOD_ORDER:
        handles.append(Patch(facecolor=MOD_COLORS[m], label=f"LongMIL — {m} only"))
    handles.append(mlines.Line2D([0], [0], marker="D", color="w",
                                 markerfacecolor=ABMIL_COLOR, markersize=10,
                                 label="ABMIL baseline (same modality subset)"))
    handles.append(mlines.Line2D([], [], color="#888", linewidth=1.2,
                                 linestyle=":", label="Chance (0.5)"))
    return handles


# ── Pre-load all data ─────────────────────────────────────────────────────────
all_data = {}
for tk, cfg in TASK_CFG.items():
    lm_splits, ab_per_mod = load_benchmark(cfg["bench_file"])
    lm_uni = load_longmil_unimodal(cfg["abl_suffix"], cfg["abl_metric"])
    all_data[tk] = {"lm_splits": lm_splits, "ab_per_mod": ab_per_mod, "lm_uni": lm_uni}

# ── Paired t-test: LongMIL(mod) > ABMIL(mod), BH over 4 tasks × 4 mods ──────
raw_pvals  = []
test_keys  = []
for tk, cfg in TASK_CFG.items():
    lm_uni    = all_data[tk]["lm_uni"]
    ab_per_mod = all_data[tk]["ab_per_mod"]
    for mod in MOD_ORDER:
        lm_d = lm_uni.get(mod)
        ab_d = ab_per_mod.get(mod)
        if lm_d is None or ab_d is None:
            raw_pvals.append(1.0)
            test_keys.append((tk, mod))
            continue
        lm_s = np.array([v for v in lm_d["splits"] if not np.isnan(float(v))], dtype=float)
        ab_s = np.array([v for v in ab_d["splits"] if not np.isnan(float(v))], dtype=float)
        n = min(len(lm_s), len(ab_s))
        if n >= 2:
            _, p = stats.ttest_rel(lm_s[:n], ab_s[:n], alternative="greater")
        else:
            p = 1.0
        raw_pvals.append(p)
        test_keys.append((tk, mod))

adj_arr = bh_correct(np.array(raw_pvals))
sig_mod_map = {tk: {} for tk in TASK_CFG}
for (tk, mod), adj_p in zip(test_keys, adj_arr):
    sig_mod_map[tk][mod] = adj_p

print("=== Significance (BH-corrected, paired t-test: LongMIL(mod) > ABMIL(mod)) ===")
for (tk, mod), adj_p in zip(test_keys, adj_arr):
    stars = sig_stars(adj_p) or "ns"
    print(f"  {tk:12s}  {mod:10s}  adj_p={adj_p:.4f}  {stars}")

# ── Per-task figures ──────────────────────────────────────────────────────────
CAPTION = (
    "Ablation: at inference, only one modality is provided to LongMIL — remaining modalities are masked to zero.\n"
    "LongMIL (coloured bars) and ABMIL diamonds are both evaluated on the same modality-specific test subset.\n"
    "Stars: * p<0.05  ** p<0.01  *** p<0.001  (one-sided paired t-test, LongMIL(mod) > ABMIL(mod), BH-corrected over 4 tasks × 4 modalities)"
)

legend_handles = make_legend()

for task_key in TASK_CFG:
    fig, ax = plt.subplots(figsize=(22, 9), facecolor=BG)
    fig.patch.set_facecolor(BG)
    plot_ablation_task(ax, task_key, sig_mod_map, all_data, show_ylabel=True)
    fig.legend(handles=legend_handles, fontsize=FS_LEGEND, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.01),
               framealpha=0.95, edgecolor="#ccc",
               handlelength=1.8, handleheight=1.4, borderpad=1.0, columnspacing=1.6)
    fig.text(0.5, -0.10, CAPTION, ha="center", va="top",
             fontsize=FS_CAPTION, color="#555", style="italic",
             transform=fig.transFigure)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"unimodal_ablation_v2_{task_key}.{ext}", dpi=300,
                    bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved unimodal_ablation_v2_{task_key}")

# ── 4-panel combined figure (task-centric, existing layout) ──────────────────
fig, axes = plt.subplots(1, 4, figsize=(60, 12), facecolor=BG)
fig.patch.set_facecolor(BG)
fig.suptitle(
    "LongMIL — Unimodal Ablation  ·  Full multimodal vs. single-modality input  "
    "(5-fold CV, Northwestern University)   "
    "|   * p<0.05  ** p<0.01  *** p<0.001  (BH-corrected, LongMIL(mod) > ABMIL(mod))",
    fontsize=22, fontweight="bold", y=1.01,
)
for ax, task_key in zip(axes, TASK_CFG):
    plot_ablation_task(ax, task_key, sig_mod_map, all_data, show_ylabel=False)
fig.tight_layout(rect=[0, 0.08, 1, 0.97])
fig.legend(handles=legend_handles, loc="lower center", ncol=6, fontsize=FS_LEGEND,
           bbox_to_anchor=(0.5, 0.01), frameon=False,
           handlelength=1.8, handleheight=1.4, columnspacing=1.6)
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"unimodal_ablation_v2_all.{ext}", dpi=300,
                bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved unimodal_ablation_v2_all")


# ══════════════════════════════════════════════════════════════════════════════
# NEW: Modality-centric layout — 4 panels (one per modality),
# each showing LongMIL and ABMIL performance across all 4 tasks.
# ══════════════════════════════════════════════════════════════════════════════

TASK_LABELS = {
    "acr_cls":    "ACR classification",
    "acr_surv":   "Time to first ACR",
    "clad_surv":  "Time to CLAD",
    "death_surv": "Time to death",
}
TASK_COLORS = {
    "acr_cls":    "#E07B20",  # amber
    "acr_surv":   "#952030",  # crimson
    "clad_surv":  "#1A5C8A",  # navy
    "death_surv": "#2D8A4E",  # forest green
}
TASK_ORDER = ["acr_cls", "acr_surv", "clad_surv", "death_surv"]


def plot_ablation_modality(ax, mod, sig_mod_map, all_data, show_ylabel=True, show_task_labels=True):
    """One panel: per-task bars for a single modality."""
    tasks = TASK_ORDER
    y = np.arange(len(tasks))

    # Task background bands
    for ti, tk in enumerate(tasks):
        col = TASK_COLORS[tk]
        ax.axhspan(ti - 0.5, ti + 0.5, facecolor=col, alpha=0.06, zorder=0)

    ax.axvline(0.5, color="#888", linewidth=1.2, linestyle=":", alpha=0.8, zorder=1)

    for ti, task_key in enumerate(tasks):
        col     = TASK_COLORS[task_key]
        lm_uni  = all_data[task_key]["lm_uni"].get(mod)
        ab_d    = all_data[task_key]["ab_per_mod"].get(mod)
        row_y   = y[ti]

        # LongMIL unimodal bar
        if lm_uni is not None:
            ax.barh(row_y, lm_uni["mean"], height=0.55,
                    color=col, alpha=0.85, zorder=2)
            if lm_uni["std"] > 0:
                ax.errorbar(lm_uni["mean"], row_y, xerr=lm_uni["std"], fmt="none",
                            ecolor="#222", elinewidth=1.5, capsize=5, zorder=4)
            for sv in lm_uni["splits"]:
                if not np.isnan(float(sv)):
                    ax.scatter(float(sv), row_y, s=22, color="white",
                               edgecolors=col, linewidths=1.0, zorder=5, alpha=0.95)
            # Value annotation
            ax.text(lm_uni["mean"] + 0.01, row_y + 0.17,
                    f"{lm_uni['mean']:.3f}", va="center", ha="left",
                    fontsize=FS_ANNOT, color="#222", fontweight="bold", zorder=6)

        # ABMIL reference diamond
        if ab_d is not None and ab_d["mean"] > 0:
            ax.scatter(ab_d["mean"], row_y - 0.18, marker="D", s=100,
                       color=ABMIL_COLOR, edgecolors="white", linewidths=1.2, zorder=7)
            if ab_d["std"] > 0:
                ax.errorbar(ab_d["mean"], row_y - 0.18, xerr=ab_d["std"], fmt="none",
                            ecolor=ABMIL_COLOR, elinewidth=1.3, capsize=4,
                            zorder=6, alpha=0.7)

        # Significance star
        adj_p = sig_mod_map.get(task_key, {}).get(mod, 1.0)
        stars = sig_stars(adj_p)
        if stars and lm_uni is not None:
            x_star = lm_uni["mean"] + (lm_uni["std"] if lm_uni["std"] > 0 else 0) + 0.03
            ax.text(min(x_star, 0.96), row_y, stars, va="center", ha="left",
                    fontsize=FS_SIG, color="#B71C1C", fontweight="bold", zorder=8)

    ax.set_yticks(y)
    if show_task_labels:
        ax.set_yticklabels([TASK_LABELS[tk] for tk in tasks],
                           fontsize=FS_YLABEL + 4, fontweight="bold")
        ax.yaxis.set_tick_params(length=0, pad=14)
        ax.yaxis.tick_right()
    else:
        ax.set_yticklabels([""] * len(tasks))
        ax.tick_params(axis="y", length=0)
    if show_ylabel:
        ax.set_ylabel("Task", fontsize=FS_XLABEL)
    ax.set_xlabel("Performance  (0.5 = chance,  1.0 = perfect)", fontsize=FS_XLABEL)
    ax.set_title(f"{mod}", fontsize=FS_TITLE, fontweight="bold", pad=10)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor(BG)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.12)


# Per-modality individual figures
for mod in MOD_ORDER:
    fig, ax = plt.subplots(figsize=(18, 9), facecolor=BG)
    fig.patch.set_facecolor(BG)
    plot_ablation_modality(ax, mod, sig_mod_map, all_data, show_ylabel=True)

    # Legend
    mod_handles = []
    for tk in TASK_ORDER:
        mod_handles.append(Patch(facecolor=TASK_COLORS[tk], alpha=0.85,
                                 label=f"LongMIL — {TASK_LABELS[tk]}"))
    mod_handles.append(mlines.Line2D([0], [0], marker="D", color="w",
                                     markerfacecolor=ABMIL_COLOR, markersize=10,
                                     label="ABMIL baseline"))
    mod_handles.append(mlines.Line2D([], [], color="#888", linewidth=1.2,
                                     linestyle=":", label="Chance (0.5)"))
    fig.legend(handles=mod_handles, fontsize=FS_LEGEND, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.01),
               framealpha=0.95, edgecolor="#ccc",
               handlelength=1.8, handleheight=1.4, borderpad=1.0, columnspacing=1.6)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"unimodal_ablation_by_mod_{mod}.{ext}",
                    dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved unimodal_ablation_by_mod_{mod}")

# Combined 4-modality figure (modality-centric)
mod_handles = []
for tk in TASK_ORDER:
    mod_handles.append(Patch(facecolor=TASK_COLORS[tk], alpha=0.85,
                             label=f"LongMIL — {TASK_LABELS[tk]}"))
mod_handles.append(mlines.Line2D([0], [0], marker="D", color="w",
                                 markerfacecolor=ABMIL_COLOR, markersize=10,
                                 label="ABMIL baseline (same modality subset)"))
mod_handles.append(mlines.Line2D([], [], color="#888", linewidth=1.2,
                                 linestyle=":", label="Chance (0.5)"))

fig, axes = plt.subplots(1, 4, figsize=(56, 12), facecolor=BG)
fig.patch.set_facecolor(BG)
fig.suptitle(
    "LongMIL — Unimodal Ablation by Modality\n"
    "Each panel: LongMIL and ABMIL (◆) performance across all 4 tasks for one modality  "
    "|  * p<0.05  ** p<0.01  *** p<0.001  (BH-corrected)",
    fontsize=24, fontweight="bold", y=1.02,
)
for ax, mod in zip(axes, MOD_ORDER):
    is_last = (mod == MOD_ORDER[-1])
    plot_ablation_modality(ax, mod, sig_mod_map, all_data,
                           show_ylabel=False, show_task_labels=is_last)
fig.tight_layout(rect=[0, 0.10, 1, 0.97])
fig.legend(handles=mod_handles, loc="lower center", ncol=6, fontsize=FS_LEGEND,
           bbox_to_anchor=(0.5, 0.01), frameon=False,
           handlelength=1.8, handleheight=1.4, columnspacing=1.6)
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"unimodal_ablation_by_mod_all.{ext}",
                dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved unimodal_ablation_by_mod_all")
print("Done.")
