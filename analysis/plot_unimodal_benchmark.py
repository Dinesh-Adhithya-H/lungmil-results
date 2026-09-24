"""
Unimodal benchmark: Linear + ABMIL per modality + weighted average ensemble.
No LongMIL reference. Horizontal-bar layout.
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

ROOT    = Path(__file__).resolve().parent.parent
PRED    = ROOT / "results" / "predictions"
LIN_DIR = PRED / "figures"   # linear CSVs live here
OUT_DIR = ROOT / "figures" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG = "#FAF6F2"

MOD_COLORS = {
    "HE":       "#E64A19",
    "BAL":      "#1565C0",
    "CT":       "#2E7D32",
    "Clinical": "#7B1FA2",
}
WTD_COLOR = "#37474F"   # dark slate for weighted average

MOD_ORDER  = ["HE", "BAL", "CT", "Clinical"]
SPLIT_COLS = ["s0", "s1", "s2", "s3", "s4"]

TASK_CFG = {
    "acr_cls": {
        "comp_file":  "comparison_acr_cls.csv",
        "lin_file":   "linear_acr_cls.csv",
        "metric_col": "bacc",
        "metric":     "BACC  (0.5 = chance,  1.0 = perfect)",
        "label":      "ACR classification",
    },
    "acr_surv": {
        "comp_file":  "comparison_acr_surv.csv",
        "lin_file":   "linear_acr_surv.csv",
        "metric_col": "cindex",
        "metric":     "C-index  (0.5 = chance,  1.0 = perfect)",
        "label":      "Time to first ACR",
    },
    "clad_surv": {
        "comp_file":  "comparison_clad.csv",
        "lin_file":   "linear_clad.csv",
        "metric_col": "cindex",
        "metric":     "C-index  (0.5 = chance,  1.0 = perfect)",
        "label":      "Time to CLAD",
    },
    "death_surv": {
        "comp_file":  "comparison_death.csv",
        "lin_file":   "linear_death.csv",
        "metric_col": "cindex",
        "metric":     "C-index  (0.5 = chance,  1.0 = perfect)",
        "label":      "Time to death",
    },
}

TASK_COLORS = {
    "acr_cls":    "#E07B20",
    "acr_surv":   "#952030",
    "clad_surv":  "#1A5C8A",
    "death_surv": "#2D8A4E",
}
TASK_ORDER = ["acr_cls", "acr_surv", "clad_surv", "death_surv"]

# Modality name mapping: linear CSV uses "H&E", comparison CSV uses "P1 HE" etc.
LIN_MOD_MAP = {"H&E": "HE", "BAL": "BAL", "CT": "CT", "Clinical": "Clinical"}
COMP_ABMIL  = {"P1 HE": "HE", "P1 BAL": "BAL", "P1 CT": "CT", "P1 Clinical": "Clinical"}

FS_TITLE  = 26
FS_LABEL  = 18
FS_TICK   = 16
FS_ANNOT  = 13
FS_LEGEND = 16


def _splits_stats(vals):
    valid = [v for v in vals if not np.isnan(v)]
    return float(np.nanmean(valid)), float(np.nanstd(valid))


def load_task(cfg):
    """Returns dict keyed by row label → {splits, mean, std}."""
    rows = {}

    # ── Linear models ────────────────────────────────────────────────────────
    lin_path = LIN_DIR / cfg["lin_file"]
    mcol     = cfg["metric_col"]
    if lin_path.exists():
        ldf = pd.read_csv(lin_path)
        for mod_raw, mod in LIN_MOD_MAP.items():
            sub = ldf[ldf["modality"] == mod_raw]
            if sub.empty:
                continue
            splits = [float(sub[sub["split"] == s][mcol].values[0])
                      if not sub[sub["split"] == s].empty else np.nan
                      for s in SPLIT_COLS]
            m, s = _splits_stats(splits)
            rows[f"Linear {mod}"] = {"splits": splits, "mean": m, "std": s,
                                      "color": MOD_COLORS[mod], "alpha": 0.50,
                                      "hatch": "///"}

    # ── ABMIL unimodal + weighted average ────────────────────────────────────
    comp_path = PRED / cfg["comp_file"]
    cdf = pd.read_csv(comp_path)
    for _, row in cdf.iterrows():
        raw = str(row["model"]).strip()
        splits = []
        for s in SPLIT_COLS:
            try:
                splits.append(float(row.get(s, np.nan)))
            except (TypeError, ValueError):
                splits.append(np.nan)
        m, s = _splits_stats(splits)
        if not splits or all(np.isnan(splits)):
            continue
        if raw in COMP_ABMIL:
            mod = COMP_ABMIL[raw]
            rows[f"ABMIL {mod}"] = {"splits": splits, "mean": m, "std": s,
                                     "color": MOD_COLORS[mod], "alpha": 0.88,
                                     "hatch": None}
        elif raw == "P1 wtd ensemble":
            rows["Wtd. Avg. Ensemble"] = {"splits": splits, "mean": m, "std": s,
                                           "color": WTD_COLOR, "alpha": 0.88,
                                           "hatch": None}
    return rows


FS_TITLE  = 32
FS_LABEL  = 22
FS_TICK   = 20
FS_ANNOT  = 16
FS_LEGEND = 20


def _draw_bar_group(ax, y, mean, std, splits, color, hatch, bar_h=0.36):
    """Draw one bar with error bar and split scatter dots."""
    ax.barh(y, mean, height=bar_h, color=color,
            alpha=0.52 if hatch else 0.90,
            hatch=hatch, edgecolor="#444" if hatch else color,
            linewidth=0.9, zorder=2)
    if std > 0:
        ax.errorbar(mean, y, xerr=std, fmt="none",
                    ecolor="#222", elinewidth=1.5, capsize=4, zorder=4)
    for sv in splits:
        if not np.isnan(sv):
            ax.scatter(sv, y, s=22, color="white",
                       edgecolors=color, linewidths=1.1, zorder=5)
    ax.text(min(mean + max(std, 0.005) + 0.014, 1.12), y,
            f"{mean:.3f}", va="center", ha="left", fontsize=FS_ANNOT,
            color="#111", fontweight="bold")


def plot_modality(ax, mod, all_data, show_ylabel=True):
    """Modality-centric panel: for each task, show Linear + ABMIL bars (colored by task)."""
    ax.set_facecolor(BG)
    ax.axvline(0.5, color="#888", linewidth=1.0, linestyle=":", alpha=0.7, zorder=1)

    row_labels = []
    row_ys     = []
    task_group_centers = {}

    group_h   = 2.2    # vertical space per task group
    bar_gap   = 0.45   # gap between Linear and ABMIL within group
    sep_extra = 0.5    # extra space between task groups

    y = 0.0
    for tk in TASK_ORDER:
        task_rows = all_data.get(tk, {})
        lin_key   = f"Linear {mod}"
        abmil_key = f"ABMIL {mod}"

        lin_d   = task_rows.get(lin_key)
        abmil_d = task_rows.get(abmil_key)
        tcol    = TASK_COLORS[tk]

        y_lin   = y + bar_gap / 2
        y_abmil = y - bar_gap / 2
        task_group_centers[tk] = y

        if lin_d:
            _draw_bar_group(ax, y_lin, lin_d["mean"], lin_d["std"],
                            lin_d["splits"], tcol, hatch="///")
            row_ys.append(y_lin)
            row_labels.append("Linear")
        if abmil_d:
            _draw_bar_group(ax, y_abmil, abmil_d["mean"], abmil_d["std"],
                            abmil_d["splits"], tcol, hatch=None)
            row_ys.append(y_abmil)
            row_labels.append("ABMIL")

        y -= (group_h + sep_extra)

    # Task group labels on y-axis (right side) at group centers
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(list(task_group_centers.values()))
    ax2.set_yticklabels(
        [TASK_CFG[tk]["label"] for tk in TASK_ORDER],
        fontsize=FS_TICK, fontweight="bold",
    )
    for label_obj, tk in zip(ax2.get_yticklabels(), TASK_ORDER):
        label_obj.set_color(TASK_COLORS[tk])
    ax2.tick_params(axis="y", length=0, pad=8)
    ax2.spines[["top", "left", "bottom"]].set_visible(False)

    # Left y-axis: Linear / ABMIL labels
    ax.set_yticks(row_ys)
    ax.set_yticklabels(row_labels, fontsize=FS_TICK - 2)
    if show_ylabel:
        ax.set_ylabel("Model type", fontsize=FS_LABEL)

    # Separators between task groups
    sep_y = task_group_centers[TASK_ORDER[0]] + (group_h + sep_extra) / 2
    for i in range(1, len(TASK_ORDER)):
        ax.axhline(sep_y - (i - 1) * (group_h + sep_extra),
                   color="#bbb", linewidth=1.2, linestyle="--", zorder=1)

    ax.set_xlabel("Metric  (BACC / C-index)", fontsize=FS_LABEL - 2)
    ax.set_title(f"{mod}", fontsize=FS_TITLE, fontweight="bold", pad=12,
                 color=MOD_COLORS[mod])
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0.0, 1.20)


# ── Load all tasks ────────────────────────────────────────────────────────────
all_data = {}
for tk, cfg in TASK_CFG.items():
    rows = load_task(cfg)
    all_data[tk] = rows
    print(f"{tk}: {list(rows.keys())}")

# ── Legend handles for modality-centric plot ──────────────────────────────────
legend_mod_handles = []
for tk in TASK_ORDER:
    lbl = TASK_CFG[tk]["label"]
    col = TASK_COLORS[tk]
    legend_mod_handles += [
        Patch(facecolor=col, alpha=0.52, hatch="///",
              edgecolor="#444", label=f"Linear — {lbl}"),
        Patch(facecolor=col, alpha=0.90,
              label=f"ABMIL — {lbl}"),
    ]
legend_mod_handles.append(
    mlines.Line2D([], [], color="#888", linewidth=1.2, linestyle=":", label="Chance (0.5)")
)

# ── Modality-centric combined figure (main output) ────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(56, 16), facecolor=BG,
                         gridspec_kw={"wspace": 0.45})
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Unimodal Benchmark — Linear & ABMIL per Modality",
    fontsize=FS_TITLE + 2, fontweight="bold", y=1.02,
)
for ax, mod in zip(axes, MOD_ORDER):
    plot_modality(ax, mod, all_data, show_ylabel=(mod == "HE"))

fig.legend(handles=legend_mod_handles, loc="lower center", ncol=4,
           fontsize=FS_LEGEND - 1, bbox_to_anchor=(0.5, -0.04),
           frameon=True, framealpha=0.95, edgecolor="#ccc",
           handlelength=2.0, handleheight=1.5, columnspacing=1.4)
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"unimodal_benchmark_by_mod_all.{ext}", dpi=300,
                bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved unimodal_benchmark_by_mod_all")

# ── Per-modality individual figures ───────────────────────────────────────────
for mod in MOD_ORDER:
    fig, ax = plt.subplots(figsize=(14, 14), facecolor=BG)
    fig.patch.set_facecolor(BG)
    plot_modality(ax, mod, all_data, show_ylabel=True)
    fig.legend(handles=legend_mod_handles, fontsize=FS_LEGEND - 1, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               framealpha=0.95, edgecolor="#ccc",
               handlelength=2.0, handleheight=1.5, borderpad=0.9, columnspacing=1.2)
    fig.tight_layout(rect=[0, 0.22, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"unimodal_benchmark_by_mod_{mod}.{ext}", dpi=300,
                    bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved unimodal_benchmark_by_mod_{mod}")

print("Done.")
