"""
Per-task benchmark figures for linear model baselines.
Reads results/linear_models/metrics_summary.csv.
Creates one figure per task (ACR cls, ACR surv, CLAD, Death) in the same
visual style as the MIL benchmark figures.

Run via: sbatch analysis/submit_linear_benchmarks.sh
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO    = Path(__file__).resolve().parent.parent
CSV     = REPO / "results/linear_models/metrics_summary.csv"
OUT_DIR = REPO / "results/predictions/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ── colour palette ────────────────────────────────────────────────────────────
MOD_COLORS = {
    "H&E":      "#4e79a7",
    "BAL":      "#f28e2b",
    "CT":       "#e15759",
    "Clinical": "#76b7b2",
    "All":      "#59a14f",
}
CHANCE_COLOR = "#aaaaaa"

# ── task configuration ────────────────────────────────────────────────────────
TASK_CFG = {
    "acr_cls": {
        "label":       "ACR Classification  (BACC)",
        "csv_task":    "ACR",
        "csv_type":    "cls",
        "metric_col":  "bacc",
        "chance":      0.5,
        "ylim":        (0.40, 0.85),
    },
    "acr_surv": {
        "label":       "ACR Survival  (C-index)",
        "csv_task":    "ACR_TTE",
        "csv_type":    "surv",
        "metric_col":  "cindex",
        "chance":      0.5,
        "ylim":        (0.35, 0.80),
    },
    "clad": {
        "label":       "CLAD  (C-index)",
        "csv_task":    "CLAD",
        "csv_type":    "surv",
        "metric_col":  "cindex",
        "chance":      0.5,
        "ylim":        (0.20, 0.80),
    },
    "death": {
        "label":       "Death  (C-index)",
        "csv_task":    "Death",
        "csv_type":    "surv",
        "metric_col":  "cindex",
        "chance":      0.5,
        "ylim":        (0.40, 0.80),
    },
}

MODALITY_ORDER = ["H&E", "BAL", "CT", "Clinical", "All"]
MOD_LABELS     = {"H&E": "H&E", "BAL": "BAL", "CT": "CT",
                  "Clinical": "Clinical", "All": "All\n(multimodal)"}

SPLITS = [f"split{s}_fold0" for s in range(5)]


def load_task_data(df, task_key):
    cfg = TASK_CFG[task_key]
    sub = df[(df["task"] == cfg["csv_task"]) & (df["task_type"] == cfg["csv_type"])]
    sub = sub[sub["fold"].isin(SPLITS)]

    result = {}
    for mod in MODALITY_ORDER:
        rows = sub[sub["modality"] == mod].copy()
        rows = rows.set_index("fold").reindex(SPLITS)
        vals = rows[cfg["metric_col"]].values.astype(float)
        result[mod] = vals  # may contain NaN for missing splits
    return result


def plot_task(task_key, data, out_path):
    cfg  = TASK_CFG[task_key]
    fig, ax = plt.subplots(figsize=(8, 5))

    positions = np.arange(len(MODALITY_ORDER))
    width     = 0.55

    bps = []
    for i, mod in enumerate(MODALITY_ORDER):
        vals = data[mod]
        vals_clean = vals[~np.isnan(vals)]
        color = MOD_COLORS[mod]

        bp = ax.boxplot(
            vals_clean,
            positions=[positions[i]],
            widths=width,
            patch_artist=True,
            notch=False,
            showfliers=True,
            boxprops=dict(facecolor=color, alpha=0.6, linewidth=1.2),
            medianprops=dict(color="black", linewidth=1.8),
            whiskerprops=dict(linewidth=1.1),
            capprops=dict(linewidth=1.1),
            flierprops=dict(marker="o", markersize=3, markerfacecolor=color,
                            markeredgewidth=0.5, markeredgecolor="white", alpha=0.7),
        )
        bps.append(bp)

        # scatter individual split points
        x_jitter = positions[i] + np.random.default_rng(i).uniform(-0.12, 0.12, len(vals_clean))
        ax.scatter(x_jitter, vals_clean, color=color, s=22, zorder=4,
                   edgecolors="white", linewidths=0.5, alpha=0.85)

        # annotate mean±std
        m, s = np.nanmean(vals), np.nanstd(vals)
        ax.text(positions[i], cfg["ylim"][0] + 0.005,
                f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom",
                fontsize=6.5, color="#333333")

    # chance line
    ax.axhline(cfg["chance"], color=CHANCE_COLOR, lw=1.2, ls="--", zorder=0,
               label="Chance (0.5)")

    ax.set_xticks(positions)
    ax.set_xticklabels([MOD_LABELS[m] for m in MODALITY_ORDER], fontsize=8)
    ax.set_ylabel(cfg["label"].split("(")[1].rstrip(")"), fontsize=9)
    ax.set_title(f"Linear Baseline — {cfg['label']}", fontsize=11, fontweight="bold")
    ax.set_ylim(cfg["ylim"])
    ax.yaxis.grid(True, alpha=0.35, lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    # legend
    handles = [mpatches.Patch(facecolor=MOD_COLORS[m], alpha=0.7, label=m)
               for m in MODALITY_ORDER]
    handles.append(plt.Line2D([0], [0], color=CHANCE_COLOR, lw=1.2, ls="--",
                               label="Chance (0.5)"))
    ax.legend(handles=handles, loc="upper right", framealpha=0.8, ncol=2)

    # per-split values as footnote
    lines = []
    for mod in MODALITY_ORDER:
        vals = data[mod]
        vals_str = "  ".join(f"{v:.3f}" if not np.isnan(v) else " — " for v in vals)
        lines.append(f"{mod:10s}: {vals_str}")
    footnote = "Per-split (s0–s4):\n" + "\n".join(lines)
    fig.text(0.01, -0.05, footnote, fontsize=6, family="monospace",
             va="top", color="#555555")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"→ saved: {out_path}")


def save_summary_table(task_key, data, out_dir):
    cfg = TASK_CFG[task_key]
    rows = []
    for mod in MODALITY_ORDER:
        vals = data[mod]
        for si, v in enumerate(vals):
            rows.append({"modality": mod, "split": f"s{si}", cfg["metric_col"]: v})
    df = pd.DataFrame(rows)
    csv_path = out_dir / f"linear_{task_key}.csv"
    df.to_csv(csv_path, index=False)

    # summary
    summary_rows = []
    for mod in MODALITY_ORDER:
        vals = data[mod]
        m, s = np.nanmean(vals), np.nanstd(vals)
        summary_rows.append({
            "modality": mod,
            "metric": cfg["metric_col"],
            "mean": round(m, 4),
            "std": round(s, 4),
            "mean±std": f"{m:.3f} ± {s:.3f}",
        })
    df_sum = pd.DataFrame(summary_rows)
    sum_path = out_dir / f"linear_{task_key}_summary.csv"
    df_sum.to_csv(sum_path, index=False)
    print(f"→ saved: {csv_path}")
    print(f"→ saved: {sum_path}")


def main():
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} rows from {CSV}")

    for task_key in ["acr_cls", "acr_surv", "clad", "death"]:
        print(f"\n=== {task_key} ===")
        data = load_task_data(df, task_key)

        # diagnostics
        for mod, vals in data.items():
            valid = np.sum(~np.isnan(vals))
            m = np.nanmean(vals)
            print(f"  {mod:10s}: n={valid}  mean={m:.3f}  vals={[round(v,3) if not np.isnan(v) else None for v in vals]}")

        plot_path = OUT_DIR / f"linear_metrics_{task_key}.png"
        plot_task(task_key, data, plot_path)
        save_summary_table(task_key, data, OUT_DIR)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
