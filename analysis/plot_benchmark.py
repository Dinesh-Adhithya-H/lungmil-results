"""
Benchmark bar plots — fixed model order, linear baselines included.
Produces per-task + combined 4-panel figures.
Run via: sbatch analysis/submit_benchmark_v2.sh
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

ROOT    = Path(__file__).resolve().parent.parent
PRED    = ROOT / "results" / "predictions"
LIN_CSV = ROOT / "results" / "linear_models" / "metrics_summary.csv"
OUT_DIR = ROOT / "figures" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
BG = "#FAF6F2"

SHARED_MODEL_COLORS = {
    "Linear HE":                    "#BDBDBD",
    "Linear BAL":                   "#9E9E9E",
    "Linear CT":                    "#757575",
    "Linear Clinical":              "#616161",
    "Wtd. avg. Linear":             "#424242",
    "ABMIL HE":                     "#90CAF9",
    "ABMIL BAL":                    "#42A5F5",
    "ABMIL CT":                     "#1976D2",
    "ABMIL Clinical":               "#1565C0",
    "Wtd. avg. ABMIL":              "#0D47A1",
    "Early fusion":                 "#80CBC4",
    "Middle fusion":                "#26A69A",
    "Late fusion":                  "#00796B",
    "LongMIL (no time, no cross-attn)": "#CE93D8",
    "LongMIL (no time, cross-attn)":    "#9C27B0",
    "LongMIL":                          "#B71C1C",
}

# ── Fixed model order ─────────────────────────────────────────────────────────
MODEL_DEFS = [
    ("Linear HE",                     SHARED_MODEL_COLORS["Linear HE"],                     "linear"),
    ("Linear BAL",                    SHARED_MODEL_COLORS["Linear BAL"],                    "linear"),
    ("Linear CT",                     SHARED_MODEL_COLORS["Linear CT"],                     "linear"),
    ("Linear Clinical",               SHARED_MODEL_COLORS["Linear Clinical"],               "linear"),
    ("Wtd. avg. Linear",              SHARED_MODEL_COLORS["Wtd. avg. Linear"],              "linear"),
    ("ABMIL HE",                      SHARED_MODEL_COLORS["ABMIL HE"],                      "unimodal"),
    ("ABMIL BAL",                     SHARED_MODEL_COLORS["ABMIL BAL"],                     "unimodal"),
    ("ABMIL CT",                      SHARED_MODEL_COLORS["ABMIL CT"],                      "unimodal"),
    ("ABMIL Clinical",                SHARED_MODEL_COLORS["ABMIL Clinical"],                "unimodal"),
    ("Wtd. avg. ABMIL",               SHARED_MODEL_COLORS["Wtd. avg. ABMIL"],               "unimodal"),
    ("Early fusion",                  SHARED_MODEL_COLORS["Early fusion"],                  "fusion"),
    ("Middle fusion",                 SHARED_MODEL_COLORS["Middle fusion"],                 "fusion"),
    ("Late fusion",                   SHARED_MODEL_COLORS["Late fusion"],                   "fusion"),
    ("LongMIL (no time, no cross-attn)", SHARED_MODEL_COLORS["LongMIL (no time, no cross-attn)"], "longi"),
    ("LongMIL (no time, cross-attn)",    SHARED_MODEL_COLORS["LongMIL (no time, cross-attn)"],    "longi"),
    ("LongMIL",                          SHARED_MODEL_COLORS["LongMIL"],                          "longi"),
]
MODEL_LABELS = [m[0] for m in MODEL_DEFS]
MODEL_COLORS = {m[0]: m[1] for m in MODEL_DEFS}
MODEL_GROUPS = {m[0]: m[2] for m in MODEL_DEFS}

GROUP_BREAKS = ["linear", "unimodal", "fusion"]

# Models that LongMIL is compared against for significance testing
MULTIMODAL_COMPARATORS = [
    "Early fusion",
    "Middle fusion",
    "Late fusion",
    "LongMIL (no time, no cross-attn)",
    "LongMIL (no time, cross-attn)",
]

SPLIT_COLS = ["s0", "s1", "s2", "s3", "s4"]

TASKS = {
    "acr_cls":   {"file": "comparison_acr_cls.csv",   "metric": "BACC",    "label": "ACR classification (BACC)",
                  "lin_task": "ACR",    "lin_metric": "bacc"},
    "acr_surv":  {"file": "comparison_acr_surv.csv",  "metric": "C-index", "label": "ACR survival (C-index)",
                  "lin_task": "ACR_TTE","lin_metric": "cindex"},
    "clad_surv": {"file": "comparison_clad.csv",      "metric": "C-index", "label": "CLAD survival (C-index)",
                  "lin_task": "CLAD",   "lin_metric": "cindex"},
    "death_surv":{"file": "comparison_death.csv",     "metric": "C-index", "label": "Death survival (C-index)",
                  "lin_task": "Death",  "lin_metric": "cindex"},
}

CSV_TO_DISPLAY = {
    "P1 HE":                "ABMIL HE",
    "P1 BAL":               "ABMIL BAL",
    "P1 CT":                "ABMIL CT",
    "P1 Clinical":          "ABMIL Clinical",
    "P1 wtd ensemble":      "Wtd. avg. ABMIL",
    "Early fusion":         "Early fusion",
    "Middle fusion":        "Middle fusion",
    "Late fusion":          "Late fusion",
    "SetMIL":               "LongMIL (no time, cross-attn)",
    "SetMIL-MT (no SAB)":   "LongMIL (no time, no cross-attn)",
    "LongMK-v2":            "LongMIL",
}

# ── Slide-ready font sizes ────────────────────────────────────────────────────
FS_TITLE   = 30
FS_YLABEL  = 22
FS_XLABEL  = 21
FS_TICK    = 20
FS_LEGEND  = 20
FS_STAR    = 32
FS_CAPTION = 16
FS_SIG     = 18   # significance star annotation


def load_linear(lin_task, lin_metric):
    df = pd.read_csv(LIN_CSV)
    df = df[df["task"] == lin_task].copy()
    MOD_MAP = {"HE": "Linear HE", "BAL": "Linear BAL", "CT": "Linear CT",
               "Clinical": "Linear Clinical", "All": "Wtd. avg. Linear",
               "H&E": "Linear HE"}
    out = {}
    for mod, disp in MOD_MAP.items():
        rows = df[df["modality"] == mod]
        splits = []
        for _, r in rows.iterrows():
            v = r.get(lin_metric, np.nan)
            try:
                splits.append(float(v))
            except (TypeError, ValueError):
                splits.append(np.nan)
        valid = [v for v in splits if not np.isnan(v)]
        if valid:
            out[disp] = {"mean": np.nanmean(splits), "std": np.nanstd(splits), "splits": splits}
    return out


def load_task_df(task_key):
    meta = TASKS[task_key]
    df = pd.read_csv(PRED / meta["file"])
    out = {}
    for _, row in df.iterrows():
        raw_name = str(row["model"]).strip()
        disp = CSV_TO_DISPLAY.get(raw_name)
        if disp is None:
            continue
        splits = []
        for s in SPLIT_COLS:
            try:
                splits.append(float(row.get(s, np.nan)))
            except (TypeError, ValueError):
                splits.append(np.nan)
        valid = [v for v in splits if not np.isnan(v)]
        if not valid:
            continue
        mean = float(row["mean"]) if not pd.isna(row.get("mean")) else np.nanmean(splits)
        std  = float(row["std"])  if not pd.isna(row.get("std"))  else np.nanstd(splits)
        out[disp] = {"mean": mean, "std": std, "splits": splits}
    return out


def bh_correct(pvals):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    adj = np.minimum(1.0, pvals * n / ranked)
    # enforce monotonicity (cumulative min from largest rank downward)
    adj_sorted = adj[order]
    for k in range(n - 2, -1, -1):
        adj_sorted[k] = min(adj_sorted[k], adj_sorted[k + 1])
    adj[order] = adj_sorted
    return adj


def compute_significance(all_data_per_task):
    """
    One-sided paired t-test: LongMIL > each other multimodal model.
    Applies BH correction across all task × comparator combinations.
    Returns dict: {task_key: {comparator_label: adj_pval}}
    """
    task_keys   = list(all_data_per_task.keys())
    raw_pvals   = []
    test_keys   = []   # (task_key, comparator)

    for tk in task_keys:
        all_data = all_data_per_task[tk]
        longmil  = all_data.get("LongMIL", {}).get("splits", [])
        lm_arr   = np.array([v for v in longmil if not np.isnan(v)], dtype=float)
        if len(lm_arr) < 2:
            continue
        for comp in MULTIMODAL_COMPARATORS:
            comp_splits = all_data.get(comp, {}).get("splits", [])
            co_arr = np.array([v for v in comp_splits if not np.isnan(v)], dtype=float)
            if len(co_arr) < 2:
                continue
            # align lengths (take paired observations)
            n_pair = min(len(lm_arr), len(co_arr))
            _, p = stats.ttest_rel(lm_arr[:n_pair], co_arr[:n_pair], alternative="greater")
            raw_pvals.append(p)
            test_keys.append((tk, comp))

    if not raw_pvals:
        return {tk: {} for tk in task_keys}

    adj = bh_correct(np.array(raw_pvals))

    result = {tk: {} for tk in task_keys}
    for (tk, comp), ap in zip(test_keys, adj):
        result[tk][comp] = ap
    return result


def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def plot_task(ax, task_key, fig, sig_map=None, show_ylabel=True):
    meta = TASKS[task_key]
    lin_data = load_linear(meta["lin_task"], meta["lin_metric"])
    mdl_data = load_task_df(task_key)
    all_data = {**lin_data, **mdl_data}

    means, stds, splits_list, colors = [], [], [], []
    for lbl in MODEL_LABELS:
        d = all_data.get(lbl)
        if d:
            means.append(d["mean"])
            stds.append(d["std"])
            splits_list.append(d["splits"])
        else:
            means.append(np.nan)
            stds.append(0)
            splits_list.append([np.nan] * 5)
        colors.append(MODEL_COLORS[lbl])

    means  = np.array(means)
    stds   = np.array(stds)
    n      = len(MODEL_LABELS)
    y      = np.arange(n)

    # ── Group band highlights ─────────────────────────────────────────────────
    GROUP_COLORS = {"linear": "#F5F5F5", "unimodal": "#E3F2FD", "fusion": "#E8F5E9",
                    "longi": "#FCE4EC"}
    prev_grp, band_start = None, 0
    group_spans = {}   # grp → (y_start, y_end)
    for i, lbl in enumerate(MODEL_LABELS):
        grp = MODEL_GROUPS[lbl]
        if prev_grp is not None and prev_grp != grp:
            ax.axhspan(band_start - 0.5, i - 0.5,
                       facecolor=GROUP_COLORS.get(prev_grp, "#FAFAFA"), alpha=0.35, zorder=0)
            group_spans[prev_grp] = (band_start, i - 1)
            band_start = i
        prev_grp = grp
    ax.axhspan(band_start - 0.5, n - 0.5,
               facecolor=GROUP_COLORS.get(prev_grp, "#FAFAFA"), alpha=0.35, zorder=0)
    group_spans[prev_grp] = (band_start, n - 1)

    # ── Group separator lines ─────────────────────────────────────────────────
    prev_grp = None
    for i, lbl in enumerate(MODEL_LABELS):
        grp = MODEL_GROUPS[lbl]
        if prev_grp is not None and prev_grp != grp and prev_grp in GROUP_BREAKS:
            ax.axhline(i - 0.5, color="#bbbbbb", linewidth=1.2, linestyle="--", zorder=0)
        prev_grp = grp

    # ── Unimodal / Multimodal bracket labels on right ────────────────────────
    uni_grps  = {"linear", "unimodal"}
    mm_grps   = {"fusion", "longi"}
    uni_rows  = [i for i, lbl in enumerate(MODEL_LABELS) if MODEL_GROUPS[lbl] in uni_grps]
    mm_rows   = [i for i, lbl in enumerate(MODEL_LABELS) if MODEL_GROUPS[lbl] in mm_grps]

    def _bracket_label(rows, text, color):
        if not rows:
            return
        y_top = min(rows) - 0.45
        y_bot = max(rows) + 0.45
        y_mid = (y_top + y_bot) / 2
        # vertical bar at right edge (x=1.04 in data coords)
        ax.plot([1.04, 1.04], [y_top, y_bot], color=color, linewidth=2.5,
                clip_on=False, zorder=8)
        ax.text(1.06, y_mid, text, ha="left", va="center", fontsize=14,
                fontweight="bold", color=color,
                clip_on=False, rotation=90)

    _bracket_label(uni_rows, "Unimodal", "#555555")
    _bracket_label(mm_rows,  "Multimodal", "#B71C1C")

    # ── Bars ──────────────────────────────────────────────────────────────────
    valid = ~np.isnan(means)
    for i in range(n):
        if valid[i]:
            ax.barh(y[i], means[i], color=colors[i], alpha=0.88, height=0.65, zorder=2)

    # ── Error bars ────────────────────────────────────────────────────────────
    for i in range(n):
        if valid[i] and stds[i] > 0:
            ax.errorbar(means[i], y[i], xerr=stds[i], fmt="none",
                        ecolor="#222", elinewidth=1.2, capsize=4, alpha=0.7, zorder=4)

    # ── Per-split dots ────────────────────────────────────────────────────────
    for i in range(n):
        for sv in splits_list[i]:
            if not np.isnan(sv):
                ax.scatter(sv, y[i], s=22, color="white", edgecolors=colors[i],
                           linewidths=1.0, zorder=5, alpha=0.95)

    # ── Chance line ───────────────────────────────────────────────────────────
    ax.axvline(0.5, color="#888", linewidth=1.2, linestyle=":", alpha=0.8, label="Chance (0.5)")

    # ── Significance stars (LongMIL > comparator, BH-corrected) ──────────────
    task_sig = (sig_map or {}).get(task_key, {})
    longmil_i = MODEL_LABELS.index("LongMIL")
    for comp, ap in task_sig.items():
        stars = sig_stars(ap)
        if not stars:
            continue
        if comp not in MODEL_LABELS:
            continue
        comp_i = MODEL_LABELS.index(comp)
        if np.isnan(means[comp_i]):
            continue
        # place stars just to the right of the comparator bar
        x_pos = min(means[comp_i] + stds[comp_i] + 0.03, 0.96)
        ax.text(x_pos, comp_i, stars, va="center", ha="left",
                fontsize=FS_SIG, color="#B71C1C", fontweight="bold", zorder=7)

    # ── Red rectangle around LongMIL row ─────────────────────────────────────
    if valid[longmil_i]:
        rect = FancyBboxPatch(
            (0.0, longmil_i - 0.42),
            means[longmil_i] + stds[longmil_i] + 0.02,
            0.84,
            boxstyle="round,pad=0.01",
            linewidth=2.8,
            edgecolor="#B71C1C",
            facecolor="none",
            zorder=7,
            clip_on=False,
        )
        ax.add_patch(rect)

    # ── Best-multimodal star ──────────────────────────────────────────────────
    MULTIMODAL_GROUPS = {"fusion", "longi"}
    mm_mask  = np.array([MODEL_GROUPS[lbl] in MULTIMODAL_GROUPS for lbl in MODEL_LABELS])
    mm_means = np.where(mm_mask, means, np.nan)
    if not np.all(np.isnan(mm_means)):
        best_i = int(np.nanargmax(mm_means))
        ax.text(means[best_i] + 0.012, best_i, "★", va="center", ha="left",
                fontsize=FS_STAR, color="#BF7320", zorder=6)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_yticks(y)
    ax.set_yticklabels(MODEL_LABELS, fontsize=FS_YLABEL)
    if show_ylabel:
        ax.set_ylabel("Model", fontsize=FS_XLABEL)
    metric_xlab = {
        "BACC":    "BACC  (0.5 = chance,  1.0 = perfect)",
        "C-index": "C-index  (0.5 = chance,  1.0 = perfect)",
    }
    ax.set_xlabel(metric_xlab.get(meta["metric"], meta["metric"]), fontsize=FS_XLABEL)
    ax.set_title(meta["label"], fontsize=FS_TITLE, fontweight="bold", pad=10)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_YLABEL)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor(BG)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.18)


def make_legend():
    import matplotlib.lines as mlines
    handles = [Patch(facecolor=col, label=lbl) for lbl, col, _ in MODEL_DEFS]
    handles.append(mlines.Line2D([], [], color="#888", linewidth=1.2,
                                 linestyle=":", label="Chance (0.5)"))
    handles.append(Patch(facecolor="none", edgecolor="#B71C1C",
                         linewidth=2, label="LongMIL (best model)"))
    return handles


# ── Pre-compute significance across all tasks (BH over 4×5=20 tests) ─────────
all_data_per_task = {}
for tk in TASKS:
    meta = TASKS[tk]
    lin  = load_linear(meta["lin_task"], meta["lin_metric"])
    mdl  = load_task_df(tk)
    all_data_per_task[tk] = {**lin, **mdl}

sig_map = compute_significance(all_data_per_task)

# Print significance summary
print("=== Significance (BH-corrected, one-sided paired t-test: LongMIL > comparator) ===")
for tk, comps in sig_map.items():
    for comp, ap in comps.items():
        stars = sig_stars(ap)
        marker = stars if stars else "ns"
        print(f"  {tk:12s}  {comp:40s}  adj_p={ap:.4f}  {marker}")

# ── Per-task figures ──────────────────────────────────────────────────────────
for task_key in TASKS:
    fig, ax = plt.subplots(figsize=(22, 16), facecolor=BG)
    fig.patch.set_facecolor(BG)
    plot_task(ax, task_key, fig, sig_map=sig_map)
    fig.legend(handles=make_legend(), fontsize=FS_LEGEND, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.01),
               framealpha=0.95, edgecolor="#ccc",
               handlelength=1.6, handleheight=1.4, borderpad=1.0,
               columnspacing=1.4)
    fig.text(0.5, -0.06,
             "n = 263 patients · 5-fold cross-validation · Northwestern University"
             "  |  * p<0.05  ** p<0.01  *** p<0.001  (one-sided paired t-test, BH-corrected,"
             " LongMIL vs. other multimodal)",
             ha="center", fontsize=FS_CAPTION, color="#666", style="italic",
             transform=ax.transAxes)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"benchmark_{task_key}.{ext}", dpi=200,
                    bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved benchmark_{task_key}")

# ── 4-panel combined ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(60, 18), facecolor=BG)
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Benchmark — all models, all tasks  (n=263 patients, 5-fold CV, Northwestern University)"
    "   |   * p<0.05  ** p<0.01  *** p<0.001  (BH-corrected, LongMIL vs. other multimodal)",
    fontsize=26, fontweight="bold", y=0.995)
for ax, task_key in zip(axes, TASKS):
    plot_task(ax, task_key, fig, sig_map=sig_map, show_ylabel=False)
fig.tight_layout(rect=[0, 0.08, 1, 0.97])
fig.legend(handles=make_legend(), loc="lower center", ncol=6, fontsize=FS_LEGEND,
           bbox_to_anchor=(0.5, 0.01), frameon=False,
           handlelength=1.6, handleheight=1.4, columnspacing=1.4)
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"benchmark_all.{ext}", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved benchmark_all")
print("Done.")
