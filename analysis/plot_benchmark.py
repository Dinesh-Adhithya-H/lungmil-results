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
from matplotlib.patches import Patch

ROOT    = Path(__file__).resolve().parent.parent
PRED    = ROOT / "results" / "predictions"
LIN_CSV = ROOT / "results" / "linear_models" / "metrics_summary.csv"
OUT_DIR = ROOT / "figures" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
BG = "#FAF6F2"

# Shared per-model colors — identical across benchmark, unimodal ablation, and combo plots
SHARED_MODEL_COLORS = {
    "Linear HE":          "#BDBDBD",
    "Linear BAL":         "#9E9E9E",
    "Linear CT":          "#757575",
    "Linear Clinical":    "#616161",
    "Wtd. avg. Linear":  "#424242",
    "ABMIL HE":           "#90CAF9",
    "ABMIL BAL":          "#42A5F5",
    "ABMIL CT":           "#1976D2",
    "ABMIL Clinical":     "#1565C0",
    "Wtd. avg. ABMIL":   "#0D47A1",
    "Early fusion":       "#80CBC4",
    "Middle fusion":      "#26A69A",
    "Late fusion":        "#00796B",
    "SetMIL":             "#CE93D8",
    "SetMIL-MT":          "#9C27B0",
    "SetMIL-MT (no SAB)": "#6A1B9A",
    "LongMK-MT":          "#EF9A9A",
    "LongMK":             "#C62828",
}

# ── Fixed model order ─────────────────────────────────────────────────────────
# Each entry: (display_label, colour, group_tag)
MODEL_DEFS = [
    ("Linear HE",             SHARED_MODEL_COLORS["Linear HE"],          "linear"),
    ("Linear BAL",            SHARED_MODEL_COLORS["Linear BAL"],         "linear"),
    ("Linear CT",             SHARED_MODEL_COLORS["Linear CT"],          "linear"),
    ("Linear Clinical",       SHARED_MODEL_COLORS["Linear Clinical"],    "linear"),
    ("Wtd. avg. Linear",          SHARED_MODEL_COLORS["Wtd. avg. Linear"],      "linear"),
    # separator
    ("ABMIL HE",              SHARED_MODEL_COLORS["ABMIL HE"],           "p1"),
    ("ABMIL BAL",             SHARED_MODEL_COLORS["ABMIL BAL"],          "p1"),
    ("ABMIL CT",              SHARED_MODEL_COLORS["ABMIL CT"],           "p1"),
    ("ABMIL Clinical",        SHARED_MODEL_COLORS["ABMIL Clinical"],     "p1"),
    ("Wtd. avg. ABMIL",       SHARED_MODEL_COLORS["Wtd. avg. ABMIL"],    "p1"),
    # separator
    ("Early fusion",          SHARED_MODEL_COLORS["Early fusion"],       "fusion"),
    ("Middle fusion",         SHARED_MODEL_COLORS["Middle fusion"],      "fusion"),
    ("Late fusion",           SHARED_MODEL_COLORS["Late fusion"],        "fusion"),
    # separator
    ("SetMIL",                SHARED_MODEL_COLORS["SetMIL"],             "setmil"),
    ("SetMIL-MT",             SHARED_MODEL_COLORS["SetMIL-MT"],          "setmil"),
    ("SetMIL-MT (no SAB)",    SHARED_MODEL_COLORS["SetMIL-MT (no SAB)"], "setmil"),
    # separator
    ("LongMK-MT",             SHARED_MODEL_COLORS["LongMK-MT"],          "longi"),
    ("LongMK",                SHARED_MODEL_COLORS["LongMK"],             "longi"),
]
MODEL_LABELS = [m[0] for m in MODEL_DEFS]
MODEL_COLORS = {m[0]: m[1] for m in MODEL_DEFS}
MODEL_GROUPS = {m[0]: m[2] for m in MODEL_DEFS}

# Group boundaries (after which to draw separator)
GROUP_BREAKS = ["linear", "p1", "fusion", "setmil"]

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

# Mapping from comparison CSV model names → display labels
CSV_TO_DISPLAY = {
    "P1 HE":           "ABMIL HE",
    "P1 BAL":          "ABMIL BAL",
    "P1 CT":           "ABMIL CT",
    "P1 Clinical":     "ABMIL Clinical",
    "P1 wtd ensemble": "Wtd. avg. ABMIL",
    "Early fusion":    "Early fusion",
    "Middle fusion":   "Middle fusion",
    "Late fusion":     "Late fusion",
    "SetMIL":          "SetMIL",
    "SetMIL-MT":       "SetMIL-MT",
    "SetMIL-MT (no SAB)":   "SetMIL-MT (no SAB)",
    "LongMK-MT (no ALiBi)": "LongMK-MT",
    "LongMK (no ALiBi) ★":  "LongMK",
}


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


def plot_task(ax, task_key, fig, show_legend=False, show_ylabel=True):
    meta = TASKS[task_key]
    lin_data = load_linear(meta["lin_task"], meta["lin_metric"])
    mdl_data = load_task_df(task_key)
    all_data = {**lin_data, **mdl_data}

    # Build ordered arrays
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

    # Draw group separator lines
    prev_grp = None
    for i, lbl in enumerate(MODEL_LABELS):
        grp = MODEL_GROUPS[lbl]
        if prev_grp is not None and prev_grp != grp and prev_grp in GROUP_BREAKS:
            ax.axhline(i - 0.5, color="#cccccc", linewidth=0.8, linestyle="--", zorder=0)
        prev_grp = grp

    # Bars
    valid = ~np.isnan(means)
    for i in range(n):
        if valid[i]:
            ax.barh(y[i], means[i], color=colors[i], alpha=0.85, height=0.6, zorder=2)

    # Error bars
    for i in range(n):
        if valid[i] and stds[i] > 0:
            ax.errorbar(means[i], y[i], xerr=stds[i], fmt="none",
                        ecolor="#333", elinewidth=0.9, capsize=3, alpha=0.6, zorder=4)

    # Per-split dots
    for i in range(n):
        for sv in splits_list[i]:
            if not np.isnan(sv):
                ax.scatter(sv, y[i], s=12, color="white", edgecolors=colors[i],
                           linewidths=0.7, zorder=5, alpha=0.9)

    # Chance / 0.5 line
    ax.axvline(0.5, color="#999", linewidth=0.9, linestyle=":", alpha=0.7, label="Chance (0.5)")

    # Group band highlights
    GROUP_COLORS = {"linear": "#F5F5F5", "p1": "#E3F2FD", "fusion": "#E8F5E9",
                    "setmil": "#F3E5F5", "longi": "#FCE4EC"}
    prev_grp, band_start = None, 0
    for i, lbl in enumerate(MODEL_LABELS):
        grp = MODEL_GROUPS[lbl]
        if prev_grp is not None and prev_grp != grp:
            ax.axhspan(band_start - 0.5, i - 0.5,
                       facecolor=GROUP_COLORS.get(prev_grp, "#FAFAFA"), alpha=0.35, zorder=0)
            band_start = i
        prev_grp = grp
    ax.axhspan(band_start - 0.5, len(MODEL_LABELS) - 0.5,
               facecolor=GROUP_COLORS.get(prev_grp, "#FAFAFA"), alpha=0.35, zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(MODEL_LABELS, fontsize=7.5)
    if show_ylabel:
        ax.set_ylabel("Model", fontsize=8)
    metric_xlab = {
        "BACC":    "BACC — balanced accuracy (0.5 = random chance, 1.0 = perfect)",
        "C-index": "C-index — concordance index (0.5 = random chance, 1.0 = perfect)",
    }
    ax.set_xlabel(metric_xlab.get(meta["metric"], meta["metric"]), fontsize=8)
    ax.set_title(meta["label"], fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor(BG)
    ax.invert_yaxis()

    ax.set_xlim(0.0, 1.0)

    # Star on best multimodal model bar
    MULTIMODAL_GROUPS = {"fusion", "setmil", "longi"}
    mm_mask = np.array([MODEL_GROUPS[lbl] in MULTIMODAL_GROUPS for lbl in MODEL_LABELS])
    mm_means = np.where(mm_mask, means, np.nan)
    if not np.all(np.isnan(mm_means)):
        best_i = int(np.nanargmax(mm_means))
        ax.text(means[best_i] + 0.01, best_i, "★", va="center", ha="left",
                fontsize=14, color="#BF7320", zorder=6)


def make_legend():
    import matplotlib.lines as mlines
    handles = [Patch(facecolor=col, label=lbl) for lbl, col, _ in MODEL_DEFS]
    handles.append(mlines.Line2D([], [], color="#999", linewidth=0.9,
                                 linestyle=":", label="Chance (0.5)"))
    return handles


# ── Per-task figures ──────────────────────────────────────────────────────────
for task_key in TASKS:
    fig, ax = plt.subplots(figsize=(8, 10), facecolor=BG)
    fig.patch.set_facecolor(BG)
    plot_task(ax, task_key, fig)
    ax.legend(handles=make_legend(), fontsize=8, loc="lower right",
              ncol=2, framealpha=0.9, edgecolor="#ccc")
    fig.text(0.5, 0.01, "n = 350 patients · 5-fold cross-validation · Helmholtz Munich",
             ha="center", fontsize=8, color="#666", style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"benchmark_{task_key}.{ext}", dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved benchmark_{task_key}")

# ── 4-panel combined ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(30, 11), facecolor=BG)
fig.patch.set_facecolor(BG)
fig.suptitle("Benchmark — all models, all tasks  (n=350 patients, 5-fold CV, Helmholtz Munich)",
             fontsize=12, fontweight="bold")
for ax, task_key in zip(axes, TASKS):
    plot_task(ax, task_key, fig, show_ylabel=False)
fig.tight_layout(rect=[0, 0.10, 1, 0.97])
fig.legend(handles=make_legend(), loc="lower center", ncol=5, fontsize=8,
           bbox_to_anchor=(0.5, 0.01), frameon=False)
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"benchmark_all.{ext}", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved benchmark_all")
print("Done.")
