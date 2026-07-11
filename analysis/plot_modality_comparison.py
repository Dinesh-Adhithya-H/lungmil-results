#!/usr/bin/env python3
"""
Bar + scatter plots for unimodal vs multimodal comparison — one figure per task.

Layout per task (4 panels arranged 2×2 or single-row):
  - Bar chart: mean±std across splits for all models
  - Strip chart: per-split dots so variance is visible

Colour scheme (from analysis/CLAUDE.md):
  - P1 unimodal per-modality: distinct muted colours
  - P1 weighted ensemble: grey
  - P2 variants: blue gradient (early→late→middle→set_mil→longitudinal_mk)

Output: results/predictions/figures/comparison_{task}.pdf + .png
"""

import math, statistics
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR  = Path("/home/aih/dinesh.haridoss/chicago_mil/results/predictions")
FIG_DIR  = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = [0, 1, 2, 3, 4]

TASKS = [
    {"short": "acr_cls",   "name": "ACR Classification", "metric": "BACC",    "chance": 0.5},
    {"short": "acr_surv",  "name": "ACR Survival",        "metric": "C-index", "chance": 0.5},
    {"short": "clad",      "name": "CLAD",                "metric": "C-index", "chance": 0.5},
    {"short": "death",     "name": "Death",               "metric": "C-index", "chance": 0.5},
]

# ── Colour palette ────────────────────────────────────────────────────────────
MOD_COLORS = {
    "P1 HE":       "#E57373",   # soft red
    "P1 BAL":      "#81C784",   # soft green
    "P1 CT":       "#64B5F6",   # soft blue
    "P1 Clinical": "#FFB74D",   # soft orange
}
ENSEMBLE_COLOR = "#9E9E9E"      # grey
P2_COLORS = {
    "P2 early":          "#1565C0",
    "P2 late":           "#1976D2",
    "P2 middle":         "#42A5F5",
    "P2 set_mil":   "#0D47A1",
    "P2 longitudinal_mk":"#311B92",
}

def get_color(label):
    if label in MOD_COLORS:      return MOD_COLORS[label]
    if label == "P1 wtd ensemble": return ENSEMBLE_COLOR
    return P2_COLORS.get(label, "#757575")

def short_label(label):
    return (label
        .replace("P1 ", "")
        .replace("P2 ", "")
        .replace("longitudinal_mk", "long-MK")
        .replace("set_mil", "mario-K")
        .replace("wtd ensemble", "wtd ens.")
        .replace("Clinical", "Clin.")
    )

# ── Load CSV ──────────────────────────────────────────────────────────────────
def load_table(short):
    f = OUT_DIR / f"comparison_{short}.csv"
    if not f.exists():
        return None
    return pd.read_csv(f)

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_task(task):
    df = load_table(task["short"])
    if df is None:
        print(f"  [skip] {task['short']} — no CSV")
        return

    models  = df["model"].tolist()
    n_models = len(models)
    split_cols = [f"s{s}" for s in SPLITS]

    # per-split values (NaN where missing)
    vals = df[split_cols].values.astype(float)   # (n_models, 5)
    means = df["mean"].values.astype(float)
    stds  = df["std"].fillna(0).values.astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    fig.suptitle(f"{task['name']} — {task['metric']}", fontsize=14, fontweight="bold", y=1.01)

    colors = [get_color(m) for m in models]
    labels = [short_label(m) for m in models]
    x      = np.arange(n_models)

    # ── Left: bar chart mean±std ──────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(x, np.where(np.isnan(means), 0, means),
                  color=colors, width=0.65, alpha=0.85, zorder=3,
                  edgecolor="white", linewidth=0.5)
    # error bars only where std > 0 and mean valid
    for i, (m, s) in enumerate(zip(means, stds)):
        if not math.isnan(m) and s > 0:
            ax.errorbar(i, m, yerr=s, fmt="none", color="black",
                        capsize=3, linewidth=1.2, zorder=4)
        if math.isnan(m):
            ax.text(i, task["chance"] + 0.01, "—", ha="center", va="bottom",
                    fontsize=9, color="#AAAAAA")

    # per-split dots
    for i, row in enumerate(vals):
        jitter = np.linspace(-0.15, 0.15, len(SPLITS))
        for j, v in enumerate(row):
            if not math.isnan(v):
                ax.scatter(i + jitter[j], v, color="black", s=18, zorder=5,
                           alpha=0.6, linewidths=0)

    ax.axhline(task["chance"], color="#BBBBBB", linestyle="--", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(task["metric"], fontsize=11)
    ax.set_title("Mean ± std (dots = per-split)", fontsize=10)
    ax.set_ylim(max(0.3, np.nanmin(vals) - 0.05), min(1.0, np.nanmax(vals) + 0.08))
    ax.grid(axis="y", alpha=0.3, zorder=1)
    ax.spines[["top","right"]].set_visible(False)

    # ── Right: per-split strip + line for each model group ───────────────────
    ax2 = axes[1]
    # group: P1 mods, ensemble, P2 variants
    group_map = {m: (0 if m.startswith("P1 ") and "wtd" not in m
                     else 1 if "wtd" in m
                     else 2)
                 for m in models}

    for i, (model, row) in enumerate(zip(models, vals)):
        valid_splits = [s for s, v in zip(SPLITS, row) if not math.isnan(v)]
        valid_vals   = [v for v in row if not math.isnan(v)]
        if not valid_vals:
            continue
        col = get_color(model)
        lbl = short_label(model)
        ls  = "-" if model.startswith("P2") else "--"
        lw  = 2.0 if model.startswith("P2") else 1.2
        ax2.plot(valid_splits, valid_vals, marker="o", markersize=5,
                 color=col, linestyle=ls, linewidth=lw, label=lbl, alpha=0.85)

    ax2.axhline(task["chance"], color="#BBBBBB", linestyle="--", linewidth=0.8)
    ax2.set_xticks(SPLITS)
    ax2.set_xticklabels([f"s{s}" for s in SPLITS])
    ax2.set_xlabel("Split", fontsize=11)
    ax2.set_title("Per-split trajectories", fontsize=10)
    ax2.set_ylim(max(0.3, np.nanmin(vals) - 0.05), min(1.0, np.nanmax(vals) + 0.08))
    ax2.legend(fontsize=7.5, loc="lower right", framealpha=0.7,
               ncol=1, handlelength=1.5)
    ax2.grid(alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)

    # ── Legend strip for left panel ───────────────────────────────────────────
    handles = []
    for model in models:
        col = get_color(model)
        lbl = short_label(model)
        handles.append(mpatches.Patch(color=col, label=lbl, alpha=0.85))
    axes[0].legend(handles=handles, fontsize=7.5, loc="lower right",
                   framealpha=0.7, ncol=2, handlelength=1.2)

    plt.tight_layout()
    stem = FIG_DIR / f"comparison_{task['short']}"
    fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {stem}.png")


if __name__ == "__main__":
    for task in TASKS:
        print(f"\nPlotting {task['name']} ...")
        plot_task(task)
    print("\nDone.")
