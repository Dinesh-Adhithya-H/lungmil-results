"""
Benchmark multimodal table — bar plots with per-split strip.

Reads the 4 comparison_*.csv files from results/predictions/ and produces:
  - figures/benchmark/benchmark_multimodal_<task>.png/.pdf  (per-task)
  - figures/benchmark/benchmark_multimodal_all.png/.pdf     (4-panel)

Run via:  sbatch analysis/submit_benchmark_multimodal.sh
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

ROOT    = Path(__file__).resolve().parent.parent
PRED    = ROOT / "results" / "predictions"
OUT_DIR = ROOT / "figures" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_EDGE = "#BF7320"

SHARED_MODEL_COLORS = {
    "Linear HE":          "#BDBDBD",
    "Linear BAL":         "#9E9E9E",
    "Linear CT":          "#757575",
    "Linear Clinical":    "#616161",
    "wt avg Linear":      "#424242",
    "ABMIL HE":           "#90CAF9",
    "ABMIL BAL":          "#42A5F5",
    "ABMIL CT":           "#1976D2",
    "ABMIL Clinical":     "#1565C0",
    "wt avg ABMIL":       "#0D47A1",
    "Early fusion":       "#80CBC4",
    "Middle fusion":      "#26A69A",
    "Late fusion":        "#00796B",
    "SetMIL":             "#CE93D8",
    "SetMIL-MT":          "#9C27B0",
    "SetMIL-MT (no SAB)": "#6A1B9A",
    "LongMK-MT":          "#EF9A9A",
    "LongMK":             "#C62828",
}

TASKS = {
    "acr_cls":   {"file": "comparison_acr_cls.csv",   "metric": "BACC",    "label": "ACR classification (BACC)"},
    "acr_surv":  {"file": "comparison_acr_surv.csv",  "metric": "C-index", "label": "ACR survival (C-index)"},
    "clad_surv": {"file": "comparison_clad.csv",      "metric": "C-index", "label": "CLAD survival (C-index)"},
    "death_surv":{"file": "comparison_death.csv",     "metric": "C-index", "label": "Death survival (C-index)"},
}

MODEL_COLORS = {
    "P1 HE":             SHARED_MODEL_COLORS["ABMIL HE"],
    "P1 BAL":            SHARED_MODEL_COLORS["ABMIL BAL"],
    "P1 CT":             SHARED_MODEL_COLORS["ABMIL CT"],
    "P1 Clinical":       SHARED_MODEL_COLORS["ABMIL Clinical"],
    "P1 wtd ensemble":   SHARED_MODEL_COLORS["wt avg ABMIL"],
    "P2 early":          SHARED_MODEL_COLORS["Early fusion"],
    "P2 late":           SHARED_MODEL_COLORS["Late fusion"],
    "P2 middle":         SHARED_MODEL_COLORS["Middle fusion"],
    "P2 longitudinal_mk":SHARED_MODEL_COLORS["LongMK"],
}

MODEL_LABELS = {
    "P1 HE":             "HE (P1)",
    "P1 BAL":            "BAL (P1)",
    "P1 CT":             "CT (P1)",
    "P1 Clinical":       "Clinical (P1)",
    "P1 wtd ensemble":   "Ensemble (P1)",
    "P2 early":          "Early fusion",
    "P2 late":           "Late fusion",
    "P2 middle":         "Middle fusion",
    "P2 longitudinal_mk":"LongMK ★",
}

SPLIT_COLS = ["s0", "s1", "s2", "s3", "s4"]


def load_task(task_key):
    meta = TASKS[task_key]
    df = pd.read_csv(PRED / meta["file"])
    rows = []
    for _, row in df.iterrows():
        model = str(row["model"]).strip()
        splits = []
        for s in SPLIT_COLS:
            v = row.get(s, np.nan)
            try:
                v = float(v)
            except (ValueError, TypeError):
                v = np.nan
            splits.append(v)
        valid = [v for v in splits if not np.isnan(v)]
        if not valid:
            continue
        mean = float(row["mean"]) if not pd.isna(row["mean"]) else np.nanmean(splits)
        std  = float(row["std"])  if not pd.isna(row["std"])  else np.nanstd(splits)
        rows.append({"model": model, "mean": mean, "std": std, "splits": splits})
    return pd.DataFrame(rows)


def plot_task(ax, df, title, metric, show_legend=False):
    n = len(df)
    y = np.arange(n)
    colors = [MODEL_COLORS.get(m, "#7D6D78") for m in df["model"]]
    best_idx = df["mean"].idxmax()

    # Bars
    bars = ax.barh(y, df["mean"], color=colors, alpha=0.82, height=0.55, zorder=2)
    # Amber border on best
    bars[best_idx].set_edgecolor(BEST_EDGE)
    bars[best_idx].set_linewidth(2.2)

    # Error bars (std)
    ax.errorbar(df["mean"], y, xerr=df["std"], fmt="none",
                ecolor="#1A1018", elinewidth=1.0, capsize=3.5, alpha=0.6, zorder=3)

    # Per-split dots
    for i, (_, row) in enumerate(df.iterrows()):
        for sv in row["splits"]:
            if not np.isnan(sv):
                ax.scatter(sv, i, s=14, color="white", edgecolors=colors[i],
                           linewidths=0.8, zorder=4, alpha=0.9)

    # Chance line
    ax.axvline(0.5, color="#7D6D78", linewidth=0.8, linestyle="--", alpha=0.4, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in df["model"]], fontsize=8)
    ax.set_xlabel(metric, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0.0, 1.0)

    # Star on best bar
    best_i = int(df["mean"].idxmax() - df.index[0])
    ax.text(df["mean"].max() + 0.01, best_i, "★", va="center", ha="left",
            fontsize=9, color=BEST_EDGE, zorder=6)
    ax.invert_yaxis()

    if show_legend:
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=SHARED_MODEL_COLORS["ABMIL HE"],    label="Unimodal ABMIL (P1)"),
            Patch(facecolor=SHARED_MODEL_COLORS["Early fusion"], label="Fusion (P2)"),
            Patch(facecolor=SHARED_MODEL_COLORS["LongMK"],       label="LongMK (P2)"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.8)


# ── Per-task figures ─────────────────────────────────────────────────────────
for task_key, meta in TASKS.items():
    df = load_task(task_key)
    if df.empty:
        print(f"  [skip] {task_key}: no data")
        continue
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(df) + 1.2))
    fig.patch.set_facecolor("#FAF6F2")
    ax.set_facecolor("#FAF6F2")
    plot_task(ax, df, meta["label"], meta["metric"], show_legend=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"benchmark_multimodal_{task_key}.{ext}", dpi=180,
                    bbox_inches="tight", facecolor="#FAF6F2")
    plt.close(fig)
    print(f"  [done] {task_key}")

# ── 4-panel combined figure ───────────────────────────────────────────────────
task_keys = list(TASKS.keys())
dfs = [load_task(k) for k in task_keys]
max_n = max(len(d) for d in dfs)

fig, axes = plt.subplots(1, 4, figsize=(16, 0.42 * max_n + 1.6), sharey=False)
fig.patch.set_facecolor("#FAF6F2")
for ax, tk, df in zip(axes, task_keys, dfs):
    ax.set_facecolor("#FAF6F2")
    if not df.empty:
        plot_task(ax, df, TASKS[tk]["label"], TASKS[tk]["metric"], show_legend=(tk == task_keys[-1]))
    else:
        ax.set_visible(False)

# Shared legend above
from matplotlib.patches import Patch
handles = [
    Patch(facecolor=SHARED_MODEL_COLORS["ABMIL HE"],    label="Unimodal ABMIL (P1)"),
    Patch(facecolor=SHARED_MODEL_COLORS["Early fusion"], label="Fusion (P2)"),
    Patch(facecolor=SHARED_MODEL_COLORS["LongMK"],       label="LongMK (P2) ★"),
]
fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, 1.02), framealpha=0.85)
fig.suptitle("Multimodal benchmark — all tasks", fontsize=11, fontweight="bold", y=1.06)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"benchmark_multimodal_all.{ext}", dpi=180,
                bbox_inches="tight", facecolor="#FAF6F2")
plt.close(fig)
print("  [done] 4-panel all")

print(f"\nFigures saved to {OUT_DIR}")
