#!/usr/bin/env python3
"""
analyze_v7_full.py — Unified comprehensive analysis suite for multimodal ABMIL v7.

Tasks
-----
  benchmark           JSON metrics → heatmap / bar / radar / combined figures
  combo_dist          Modality combo distribution figures from CSV
  umap                UMAP of r_cls / r_surv with rich annotation panels
  combo_perf          Per-combo prediction performance breakdown
  multimodal_analysis Multimodal vs unimodal subset performance comparison

All v7 models return 4-tuple: (logit, hazard, r_cls, r_surv).
Reuses helpers from analyze_v7.py: _load_v7_module, _build_and_load,
_tag_to_variant_kwargs, _run_inference, _collect_variant_data.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import importlib.util
import json
import math
import re as _re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef,
)

warnings.filterwarnings("ignore")

try:
    from umap import UMAP as UMAPTransform
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[warn] umap-learn not installed; umap task will be skipped.")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VARIANT_TAGS = [
    "early", "late", "middle",
    "crossattn_k8", "crossmodal_k8", "iterative_r2_k8",
]
VARIANT_DISPLAY = {
    "early":            "Early",
    "late":             "Late",
    "middle":           "Middle",
    "crossattn_k8":     "Cross-Attention",
    "crossmodal_k8":    "Cross-Modal",
    "iterative_r2_k8":  "Iterative",
}
TASK_NAMES = ["cls", "surv", "alt"]
TASK_DISPLAY = {"cls": "Classification (CLS)", "surv": "Survival (SURV)", "alt": "Alternating (ALT)"}
PRIMARY_METRIC = {"cls": "bacc", "surv": "c_index", "alt": "bacc"}
ALL_METRICS = ["auc", "auprc", "bacc", "mcc", "sens", "spec", "c_index"]
METRIC_DISPLAY = {
    "auc": "AUC", "auprc": "AUPRC", "bacc": "BAcc",
    "mcc": "MCC", "sens": "Sens", "spec": "Spec", "c_index": "C-index",
}

COMBO_COLORS = {
    "Clin":              "#546E7A",
    "CT":                "#1565C0",
    "CT+Clin":           "#0B6E4F",
    "HE":                "#7B2D8B",
    "HE+Clin":           "#C2185B",
    "HE+CT":             "#E65100",
    "HE+CT+Clin":        "#E63946",
    "BAL+Clin":          "#00838F",
    "BAL+CT":            "#558B2F",
    "BAL+CT+Clin":       "#6A1B9A",
    "HE+BAL":            "#AD1457",
    "HE+BAL+Clin":       "#F9A825",
    "HE+BAL+CT":         "#4527A0",
    "HE+BAL+CT+Clin":    "#BF360C",
    "BAL":               "#2E7D32",
}
DEFAULT_COMBO_COLOR = "#9E9E9E"

# Nature-style rcParams
NATURE_RC = {
    "font.family":       "sans-serif",
    "font.size":         8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.linewidth":    0.5,
    "grid.alpha":        0.4,
    "grid.linestyle":    "--",
    "grid.color":        "#AAAAAA",
    "figure.dpi":        150,
    "savefig.dpi":       300,
}

ONE_COL_W  = 3.5    # inches (~89mm)
TWO_COL_W  = 7.2    # inches (~183mm)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — combo label
# ─────────────────────────────────────────────────────────────────────────────

def _combo_from_row(row) -> str:
    """Build combo string from has_* CSV columns: HE, BAL, CT, Clin order."""
    parts = []
    for mod, col in [("HE", "has_HE"), ("BAL", "has_BAL"), ("CT", "has_CT"), ("Clin", "has_Clinical")]:
        val = row.get(col, False)
        if val is True or str(val).lower() in ("true", "1", "1.0"):
            parts.append(mod)
    return "+".join(parts) if parts else "Unknown"


def _n_mods(combo: str) -> int:
    return len(combo.split("+")) if combo and combo != "Unknown" else 0


def _save_fig(fig, path_stem: Path):
    """Save figure as both PDF (300 DPI) and PNG (150 DPI)."""
    fig.savefig(str(path_stem) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(path_stem) + ".png", dpi=150, bbox_inches="tight")


# ─────────────────────────────────────────────────────────────────────────────
# Metric loading (from JSON files, no inference)
# ─────────────────────────────────────────────────────────────────────────────

def _load_metrics_for_exp(results_base: Path, exp: str, folds: List[int]) -> Dict:
    """
    Load all metrics_*.json for one experiment (cls/surv/alt).
    Returns: data[variant_tag][fold] = test_metrics_dict
    """
    results_dir = results_base / f"results_mm_abmil_v7_{exp}_p1v3"
    if not results_dir.exists():
        print(f"[benchmark] Results dir not found: {results_dir}")
        return {}
    data: Dict = {}
    for f in folds:
        fold_dir = results_dir / f"split0_fold{f}"
        if not fold_dir.exists():
            continue
        for mfile in sorted(fold_dir.glob("metrics_*.json")):
            tag = mfile.stem.replace("metrics_", "")
            try:
                with open(mfile) as fi:
                    m = json.load(fi)
                test = m.get("test", {})
                data.setdefault(tag, {})[f] = test
            except Exception as e:
                print(f"[load] {mfile}: {e}")
    return data


def _fold_stats(data: Dict, tag: str, metric: str) -> Tuple[float, float]:
    """Mean and std of a metric across folds for a variant."""
    vals = []
    for fold_data in data.get(tag, {}).values():
        v = fold_data.get(metric)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(float(v))
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: benchmark
# ─────────────────────────────────────────────────────────────────────────────

def task_benchmark(results_base: Path, output_dir: Path, folds: List[int]):
    plt.rcParams.update(NATURE_RC)
    out = output_dir / "benchmark"
    out.mkdir(parents=True, exist_ok=True)

    # ── Load all experiments ──────────────────────────────────────────────────
    all_data: Dict[str, Dict] = {}
    for exp in TASK_NAMES:
        all_data[exp] = _load_metrics_for_exp(results_base, exp, folds)
        print(f"[benchmark] {exp}: {len(all_data[exp])} variants")

    # Canonical variant order
    def _ordered_tags(data):
        order = [t for t in VARIANT_TAGS if t in data]
        order += [t for t in sorted(data.keys()) if t not in order]
        return order

    # ── Text table ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS — V7 Fusion Variants (split 0, mean ± std over folds)")
    print("=" * 80)
    for exp in TASK_NAMES:
        data = all_data[exp]
        if not data:
            continue
        prim = PRIMARY_METRIC[exp]
        tags = _ordered_tags(data)
        print(f"\n  Task: {TASK_DISPLAY[exp]}  (primary metric: {METRIC_DISPLAY[prim]})")
        print(f"  {'Variant':<22}  " + "  ".join(f"{METRIC_DISPLAY[m]:>12}" for m in ALL_METRICS))
        print("  " + "-" * 100)

        # compute and rank
        rows_exp = []
        for tag in tags:
            row = {"tag": tag}
            for m in ALL_METRICS:
                mu, sd = _fold_stats(data, tag, m)
                row[m] = (mu, sd)
            rows_exp.append(row)

        # star best per metric
        for m in ALL_METRICS:
            vals = [r[m][0] for r in rows_exp]
            best_idx = int(np.nanargmax(vals)) if not all(np.isnan(v) for v in vals) else -1
            for i, r in enumerate(rows_exp):
                r[f"{m}_star"] = (i == best_idx)

        # sort by primary metric
        rows_exp.sort(key=lambda r: r[prim][0] if not np.isnan(r[prim][0]) else -1, reverse=True)
        for r in rows_exp:
            disp = VARIANT_DISPLAY.get(r["tag"], r["tag"])
            cells = []
            for m in ALL_METRICS:
                mu, sd = r[m]
                star = "★" if r[f"{m}_star"] else " "
                cells.append(f"{mu:.3f}±{sd:.3f}{star}" if not np.isnan(mu) else "   —    ")
            print(f"  {disp:<22}  " + "  ".join(f"{c:>12}" for c in cells))

    # ── Figure: heatmaps ──────────────────────────────────────────────────────
    def _build_heatmap_arrays(data, exp):
        tags = _ordered_tags(data)
        prim = PRIMARY_METRIC[exp]
        rows_data = []
        for tag in tags:
            row_means = []
            row_stds  = []
            for m in ALL_METRICS:
                mu, sd = _fold_stats(data, tag, m)
                row_means.append(mu)
                row_stds.append(sd)
            rows_data.append((tag, row_means, row_stds))
        # sort by primary
        prim_idx = ALL_METRICS.index(prim)
        rows_data.sort(key=lambda x: x[1][prim_idx] if not np.isnan(x[1][prim_idx]) else -1, reverse=True)
        tags_out  = [r[0] for r in rows_data]
        means_mat = np.array([r[1] for r in rows_data])
        stds_mat  = np.array([r[2] for r in rows_data])
        return tags_out, means_mat, stds_mat

    # 3-panel heatmap figure
    fig_hm, axes_hm = plt.subplots(1, 3, figsize=(TWO_COL_W * 1.5, 4.5))
    for ax, exp in zip(axes_hm, TASK_NAMES):
        data = all_data[exp]
        if not data:
            ax.set_visible(False)
            continue
        tags, means_mat, stds_mat = _build_heatmap_arrays(data, exp)
        prim = PRIMARY_METRIC[exp]
        prim_idx = ALL_METRICS.index(prim)
        n_tags = len(tags)

        valid_mask = ~np.all(np.isnan(means_mat), axis=0)
        show_metrics = [m for i, m in enumerate(ALL_METRICS) if valid_mask[i]]
        show_means = means_mat[:, valid_mask]
        show_stds  = stds_mat[:, valid_mask]

        im = ax.imshow(show_means, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=0.9)
        ax.set_xticks(range(len(show_metrics)))
        xlabels = []
        for m in show_metrics:
            lbl = METRIC_DISPLAY[m]
            if m == prim:
                lbl = f"{lbl} ★"
            xlabels.append(lbl)
        ax.set_xticklabels(xlabels, fontsize=7, rotation=30, ha="right")
        ax.set_yticks(range(n_tags))
        ax.set_yticklabels([VARIANT_DISPLAY.get(t, t) for t in tags], fontsize=7)

        # Annotate cells
        for i in range(n_tags):
            for j, m in enumerate(show_metrics):
                mu = show_means[i, j]
                sd = show_stds[i, j]
                if not np.isnan(mu):
                    txt = f"{mu:.3f}\n±{sd:.3f}"
                    color = "white" if (mu < 0.45 or mu > 0.82) else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=5.5,
                            color=color, fontweight="bold" if m == prim else "normal")

        # Bold primary metric column
        pm_j = show_metrics.index(prim) if prim in show_metrics else -1
        if pm_j >= 0:
            ax.axvline(pm_j - 0.5, color="#333333", lw=0.8, alpha=0.5)
            ax.axvline(pm_j + 0.5, color="#333333", lw=0.8, alpha=0.5)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
        ax.set_title(TASK_DISPLAY[exp], fontsize=8, fontweight="bold")
        ax.tick_params(axis="both", which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)

    fig_hm.suptitle("Fusion variant performance", fontsize=9, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_fig(fig_hm, out / "fig_benchmark_heatmap")
    plt.close(fig_hm)
    print(f"[benchmark] Heatmap → {out}/fig_benchmark_heatmap.pdf/.png")

    # ── Figure: grouped bar chart ─────────────────────────────────────────────
    fig_bar, axes_bar = plt.subplots(3, 1, figsize=(TWO_COL_W, 7.5))
    bar_colors = plt.cm.get_cmap("tab10", len(VARIANT_TAGS))
    for ax, exp in zip(axes_bar, TASK_NAMES):
        data = all_data[exp]
        if not data:
            continue
        prim = PRIMARY_METRIC[exp]
        tags = _ordered_tags(data)
        disp_names = [VARIANT_DISPLAY.get(t, t) for t in tags]
        show_metrics = [m for m in ALL_METRICS
                        if any(not np.isnan(_fold_stats(data, t, m)[0]) for t in tags)]

        n_t = len(tags)
        n_m = len(show_metrics)
        x = np.arange(n_t)
        width = 0.8 / n_m

        for j, m in enumerate(show_metrics):
            means_ = [_fold_stats(data, t, m)[0] for t in tags]
            stds_  = [_fold_stats(data, t, m)[1] for t in tags]
            lw = 2.0 if m == prim else 1.0
            alpha = 0.9 if m == prim else 0.65
            bars = ax.bar(x + (j - n_m / 2 + 0.5) * width, means_,
                          width=width * 0.9,
                          yerr=stds_, capsize=2,
                          color=[bar_colors(i % 10) for i in range(n_t)],
                          alpha=alpha,
                          linewidth=lw,
                          error_kw={"elinewidth": 0.8, "ecolor": "#555555"},
                          label=METRIC_DISPLAY[m] + (" ★" if m == prim else ""))

        ax.set_xticks(x)
        ax.set_xticklabels(disp_names, fontsize=7, rotation=25, ha="right")
        ax.set_ylim(0.3, 1.05)
        ax.set_ylabel("Score", fontsize=7)
        ax.set_title(TASK_DISPLAY[exp], fontsize=8, fontweight="bold")
        ax.axhline(0.5, color="#888888", lw=0.5, ls=":")
        ax.legend(fontsize=6, ncol=4, loc="upper right", framealpha=0.7)

    fig_bar.suptitle("Fusion variant performance by task", fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig_bar, out / "fig_benchmark_bars")
    plt.close(fig_bar)
    print(f"[benchmark] Bars → {out}/fig_benchmark_bars.pdf/.png")

    # ── Figure: radar for ALT task ─────────────────────────────────────────────
    exp_radar = "alt"
    data_radar = all_data[exp_radar]
    if data_radar:
        tags_r = _ordered_tags(data_radar)
        radar_metrics = ["auc", "bacc", "mcc", "sens", "spec", "c_index"]
        radar_labels  = [METRIC_DISPLAY[m] for m in radar_metrics]
        n_m = len(radar_metrics)
        angles = np.linspace(0, 2 * np.pi, n_m, endpoint=False).tolist()
        angles += angles[:1]

        fig_rad, ax_rad = plt.subplots(
            figsize=(ONE_COL_W * 1.5, ONE_COL_W * 1.5),
            subplot_kw={"polar": True}
        )
        cmap_r = plt.cm.get_cmap("tab10", len(tags_r))
        for i, tag in enumerate(tags_r):
            vals = [_fold_stats(data_radar, tag, m)[0] for m in radar_metrics]
            vals_plot = [(v if not np.isnan(v) else 0.0) for v in vals]
            vals_plot += vals_plot[:1]
            color = cmap_r(i)
            ax_rad.plot(angles, vals_plot, "o-", lw=1.2, color=color, markersize=3,
                        label=VARIANT_DISPLAY.get(tag, tag), alpha=0.85)
            ax_rad.fill(angles, vals_plot, alpha=0.08, color=color)

        ax_rad.set_xticks(angles[:-1])
        ax_rad.set_xticklabels(radar_labels, fontsize=7)
        ax_rad.set_ylim(0, 1)
        ax_rad.set_yticks([0.4, 0.6, 0.8, 1.0])
        ax_rad.set_yticklabels(["0.4", "0.6", "0.8", "1.0"], fontsize=5)
        ax_rad.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7, framealpha=0.8)
        ax_rad.set_title("Alternating task — fusion variant comparison", fontsize=8,
                          fontweight="bold", pad=12)
        plt.tight_layout()
        _save_fig(fig_rad, out / "fig_benchmark_radar")
        plt.close(fig_rad)
        print(f"[benchmark] Radar → {out}/fig_benchmark_radar.pdf/.png")

    # ── Combined publication figure ───────────────────────────────────────────
    fig_comb = plt.figure(figsize=(TWO_COL_W * 1.4, 9.0))
    gs = fig_comb.add_gridspec(2, 3, hspace=0.45, wspace=0.4,
                                height_ratios=[1.0, 1.2])
    # Top row: 3 heatmaps
    for col, exp in enumerate(TASK_NAMES):
        ax_c = fig_comb.add_subplot(gs[0, col])
        data_c = all_data[exp]
        if not data_c:
            continue
        prim_c = PRIMARY_METRIC[exp]
        tags_c, means_c, stds_c = _build_heatmap_arrays(data_c, exp)
        valid_c = ~np.all(np.isnan(means_c), axis=0)
        sm_c = [m for i, m in enumerate(ALL_METRICS) if valid_c[i]]
        sm_means = means_c[:, valid_c]

        im_c = ax_c.imshow(sm_means, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=0.9)
        ax_c.set_xticks(range(len(sm_c)))
        xlbls = [METRIC_DISPLAY[m] + (" ★" if m == prim_c else "") for m in sm_c]
        ax_c.set_xticklabels(xlbls, fontsize=5.5, rotation=35, ha="right")
        ax_c.set_yticks(range(len(tags_c)))
        ax_c.set_yticklabels([VARIANT_DISPLAY.get(t, t) for t in tags_c], fontsize=6)
        for i in range(len(tags_c)):
            for j in range(len(sm_c)):
                mu = sm_means[i, j]
                if not np.isnan(mu):
                    color = "white" if (mu < 0.46 or mu > 0.82) else "black"
                    ax_c.text(j, i, f"{mu:.2f}", ha="center", va="center", fontsize=4.5, color=color)
        ax_c.set_title(TASK_DISPLAY[exp], fontsize=7, fontweight="bold")
        ax_c.tick_params(length=0)
        for sp in ax_c.spines.values():
            sp.set_visible(False)
        ax_c.grid(False)

    # Bottom-left: bar chart (ALT task only, primary metric)
    ax_bbar = fig_comb.add_subplot(gs[1, :2])
    exp_bb = "alt"
    data_bb = all_data[exp_bb]
    if data_bb:
        prim_bb = PRIMARY_METRIC[exp_bb]
        tags_bb = _ordered_tags(data_bb)
        means_bb = [_fold_stats(data_bb, t, prim_bb)[0] for t in tags_bb]
        stds_bb  = [_fold_stats(data_bb, t, prim_bb)[1] for t in tags_bb]
        x_bb = np.arange(len(tags_bb))
        bar_c = [plt.cm.get_cmap("tab10")(i % 10) for i in range(len(tags_bb))]
        ax_bbar.bar(x_bb, means_bb, yerr=stds_bb, capsize=3,
                    color=bar_c, alpha=0.85, linewidth=1.2,
                    error_kw={"elinewidth": 1.0, "ecolor": "#333333"})
        ax_bbar.set_xticks(x_bb)
        ax_bbar.set_xticklabels([VARIANT_DISPLAY.get(t, t) for t in tags_bb],
                                 fontsize=7, rotation=20, ha="right")
        ax_bbar.set_ylim(0.4, 1.0)
        ax_bbar.set_ylabel(METRIC_DISPLAY[prim_bb], fontsize=7)
        ax_bbar.set_title(f"ALT task — {METRIC_DISPLAY[prim_bb]} by variant", fontsize=7,
                          fontweight="bold")
        ax_bbar.axhline(0.5, color="#888888", lw=0.5, ls=":")
        for xi, (mu, sd) in enumerate(zip(means_bb, stds_bb)):
            if not np.isnan(mu):
                ax_bbar.text(xi, mu + sd + 0.01, f"{mu:.3f}", ha="center",
                             va="bottom", fontsize=5.5)

    # Bottom-right: radar
    ax_brad = fig_comb.add_subplot(gs[1, 2], polar=True)
    if data_radar:
        for i, tag in enumerate(tags_r):
            vals_r = [_fold_stats(data_radar, tag, m)[0] for m in radar_metrics]
            vals_p = [(v if not np.isnan(v) else 0.0) for v in vals_r] + \
                     [(vals_r[0] if not np.isnan(vals_r[0]) else 0.0)]
            color_r = plt.cm.get_cmap("tab10")(i % 10)
            ax_brad.plot(angles, vals_p, "o-", lw=1.0, color=color_r,
                         markersize=2, label=VARIANT_DISPLAY.get(tag, tag), alpha=0.85)
            ax_brad.fill(angles, vals_p, alpha=0.07, color=color_r)
        ax_brad.set_xticks(angles[:-1])
        ax_brad.set_xticklabels(radar_labels, fontsize=5.5)
        ax_brad.set_ylim(0, 1)
        ax_brad.set_yticks([0.5, 0.7, 0.9])
        ax_brad.set_yticklabels(["0.5", "0.7", "0.9"], fontsize=4.5)
        ax_brad.legend(loc="upper right", bbox_to_anchor=(1.5, 1.15), fontsize=5.5,
                       framealpha=0.8)
        ax_brad.set_title("ALT — radar", fontsize=7, fontweight="bold", pad=10)

    fig_comb.suptitle("Multimodal ABMIL v7: fusion variant performance", fontsize=9,
                       fontweight="bold", y=1.01)
    _save_fig(fig_comb, out / "fig_benchmark_combined")
    plt.close(fig_comb)
    print(f"[benchmark] Combined → {out}/fig_benchmark_combined.pdf/.png")
    print(f"[benchmark] Done. Outputs in {out}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: combo_dist
# ─────────────────────────────────────────────────────────────────────────────

def task_combo_dist(splits_csv: Path, output_dir: Path):
    plt.rcParams.update(NATURE_RC)
    out = output_dir / "combo_dist"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(str(splits_csv))

    # Compute combo per sample
    df["combo"] = df.apply(_combo_from_row, axis=1)
    df["n_mods"] = df["combo"].apply(_n_mods)
    df["is_multimodal"] = df["n_mods"] >= 2

    # Assign split label from fold0
    def _split_label(row):
        for col in ["split0_fold0", "split0_fold1", "split0_fold2", "split0_fold3"]:
            if col in row and str(row[col]) == "test":
                return "test"
        for col in ["split0_fold0", "split0_fold1", "split0_fold2", "split0_fold3"]:
            if col in row and str(row[col]) == "val":
                return "val"
        return "train"

    df["data_split"] = df.apply(_split_label, axis=1)

    # Overall combo counts
    combo_counts = df["combo"].value_counts().sort_values(ascending=False)
    combos_sorted = combo_counts.index.tolist()
    total = len(df)

    # ── Figure 1: main combo distribution ────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(TWO_COL_W * 1.3, 5.0),
                                gridspec_kw={"width_ratios": [2.5, 2.5, 1.5]})

    # Left: horizontal bar chart of combo counts (all/train/test)
    ax_left = axes1[0]
    splits_show = ["all", "train", "test"]
    split_colors = {"all": "#37474F", "train": "#1565C0", "test": "#C62828"}
    n_combos = len(combos_sorted)
    y_pos = np.arange(n_combos)
    bar_h = 0.25

    for si, sp in enumerate(splits_show):
        if sp == "all":
            counts = [combo_counts.get(c, 0) for c in combos_sorted]
            denom = total
        else:
            sub = df[df["data_split"] == sp]
            counts = [sub[sub["combo"] == c].shape[0] for c in combos_sorted]
            denom = len(sub)
        bars = ax_left.barh(y_pos - si * bar_h, counts,
                             height=bar_h * 0.9,
                             color=split_colors[sp], alpha=0.8,
                             label=sp.capitalize())
        for yi, (cnt, c_name) in enumerate(zip(counts, combos_sorted)):
            pct = 100 * cnt / denom if denom > 0 else 0
            if cnt > 0:
                ax_left.text(cnt + 0.5, y_pos[yi] - si * bar_h,
                              f"{cnt} ({pct:.0f}%)", va="center", fontsize=5, color="#333333")

    ax_left.set_yticks(y_pos - bar_h)
    ax_left.set_yticklabels(combos_sorted, fontsize=6.5)
    ax_left.invert_yaxis()
    ax_left.set_xlabel("Sample count", fontsize=7)
    ax_left.set_title("Modality combo counts", fontsize=8, fontweight="bold")
    ax_left.legend(fontsize=6, loc="lower right")
    ax_left.tick_params(axis="y", which="both", length=0)
    for sp in ["top", "right"]:
        ax_left.spines[sp].set_visible(False)

    # Middle: stacked bar per split (multimodal vs unimodal)
    ax_mid = axes1[1]
    split_labels_show = ["train", "val", "test"]
    for xi, sp in enumerate(split_labels_show):
        sub = df[df["data_split"] == sp]
        multi = sub["is_multimodal"].sum()
        uni   = len(sub) - multi
        n_sub = len(sub)
        ax_mid.bar(xi, multi / n_sub * 100, color="#1565C0", alpha=0.85,
                   label="Multimodal (≥2)" if xi == 0 else "")
        ax_mid.bar(xi, uni / n_sub * 100, bottom=multi / n_sub * 100,
                   color="#E65100", alpha=0.7,
                   label="Unimodal (1 mod)" if xi == 0 else "")
        ax_mid.text(xi, 2, f"n={n_sub}", ha="center", va="bottom", fontsize=6, color="white")
        ax_mid.text(xi, multi / n_sub * 100 - 3,
                    f"{multi / n_sub * 100:.0f}%", ha="center", va="top",
                    fontsize=5.5, color="white", fontweight="bold")
    ax_mid.set_xticks(range(len(split_labels_show)))
    ax_mid.set_xticklabels([s.capitalize() for s in split_labels_show], fontsize=7)
    ax_mid.set_ylabel("Percentage (%)", fontsize=7)
    ax_mid.set_title("Multimodal proportion per split", fontsize=8, fontweight="bold")
    ax_mid.set_ylim(0, 110)
    ax_mid.legend(fontsize=6)
    for sp in ["top", "right"]:
        ax_mid.spines[sp].set_visible(False)

    # Right: overall donut
    ax_right = axes1[2]
    multi_n = df["is_multimodal"].sum()
    uni_n   = total - multi_n
    wedges, texts, autotexts = ax_right.pie(
        [multi_n, uni_n],
        labels=["Multimodal", "Unimodal"],
        autopct="%1.0f%%",
        colors=["#1565C0", "#E65100"],
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    for t in texts + autotexts:
        t.set_fontsize(7)
    ax_right.set_title(f"All samples\n(n={total})", fontsize=7, fontweight="bold")

    fig1.suptitle("Modality combination analysis", fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig1, out / "fig_combo_dist")
    plt.close(fig1)
    print(f"[combo_dist] fig_combo_dist → {out}")

    # ── Figure 2: detailed combo × split table + positive rate ───────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(TWO_COL_W * 1.5, 5.5),
                                gridspec_kw={"width_ratios": [1.5, 1.5, 1.5]})

    # Count heatmap: combo × split
    ax_ht = axes2[0]
    splits_detail = ["train", "val", "test"]
    heat_counts = np.array([
        [df[(df["combo"] == c) & (df["data_split"] == sp)].shape[0]
         for sp in splits_detail]
        for c in combos_sorted
    ], dtype=float)
    im_ht = ax_ht.imshow(heat_counts, aspect="auto", cmap="Blues")
    ax_ht.set_xticks(range(len(splits_detail)))
    ax_ht.set_xticklabels([s.capitalize() for s in splits_detail], fontsize=7)
    ax_ht.set_yticks(range(len(combos_sorted)))
    ax_ht.set_yticklabels(combos_sorted, fontsize=6)
    for i, c in enumerate(combos_sorted):
        for j, sp in enumerate(splits_detail):
            v = int(heat_counts[i, j])
            ax_ht.text(j, i, str(v), ha="center", va="center", fontsize=5.5,
                       color="white" if v > heat_counts.max() * 0.6 else "#333333")
    plt.colorbar(im_ht, ax=ax_ht, fraction=0.046, pad=0.04, shrink=0.8)
    ax_ht.set_title("Sample count\n(combo × split)", fontsize=7, fontweight="bold")
    ax_ht.tick_params(length=0)
    for sp in ax_ht.spines.values():
        sp.set_visible(False)
    ax_ht.grid(False)

    # Positive rate per combo
    ax_pos = axes2[1]
    pos_rates = []
    n_totals  = []
    for c in combos_sorted:
        sub_c = df[df["combo"] == c]
        valid_lbl = sub_c["label"].dropna()
        if len(valid_lbl) > 0:
            pos_rates.append(valid_lbl.mean())
            n_totals.append(len(valid_lbl))
        else:
            pos_rates.append(float("nan"))
            n_totals.append(0)

    colors_combo = [COMBO_COLORS.get(c, DEFAULT_COMBO_COLOR) for c in combos_sorted]
    y_p = np.arange(len(combos_sorted))
    valid_mask_pr = ~np.isnan(pos_rates)
    ax_pos.barh(y_p[valid_mask_pr],
                [pos_rates[i] for i in range(len(combos_sorted)) if valid_mask_pr[i]],
                color=[colors_combo[i] for i in range(len(combos_sorted)) if valid_mask_pr[i]],
                alpha=0.85, height=0.7)
    ax_pos.set_yticks(y_p)
    ax_pos.set_yticklabels(combos_sorted, fontsize=6)
    ax_pos.invert_yaxis()
    ax_pos.set_xlim(0, 1)
    ax_pos.axvline(0.5, color="#999999", lw=0.8, ls=":")
    ax_pos.set_xlabel("ACR positive rate", fontsize=7)
    ax_pos.set_title("ACR positive rate\nper combo", fontsize=7, fontweight="bold")
    ax_pos.tick_params(axis="y", length=0)
    for sp in ["top", "right"]:
        ax_pos.spines[sp].set_visible(False)

    # CLAD rate per combo
    ax_clad = axes2[2]
    clad_rates = []
    for c in combos_sorted:
        sub_c = df[df["combo"] == c]
        if "clad_status" in sub_c.columns:
            valid_clad = pd.to_numeric(sub_c["clad_status"], errors="coerce").dropna()
            clad_rates.append(valid_clad.mean() if len(valid_clad) > 0 else float("nan"))
        else:
            clad_rates.append(float("nan"))

    valid_mask_cl = ~np.isnan(clad_rates)
    ax_clad.barh(y_p[valid_mask_cl],
                 [clad_rates[i] for i in range(len(combos_sorted)) if valid_mask_cl[i]],
                 color=[colors_combo[i] for i in range(len(combos_sorted)) if valid_mask_cl[i]],
                 alpha=0.85, height=0.7)
    ax_clad.set_yticks(y_p)
    ax_clad.set_yticklabels(combos_sorted, fontsize=6)
    ax_clad.invert_yaxis()
    ax_clad.set_xlim(0, 1)
    ax_clad.axvline(0.5, color="#999999", lw=0.8, ls=":")
    ax_clad.set_xlabel("CLAD rate", fontsize=7)
    ax_clad.set_title("CLAD rate\nper combo", fontsize=7, fontweight="bold")
    ax_clad.tick_params(axis="y", length=0)
    for sp in ["top", "right"]:
        ax_clad.spines[sp].set_visible(False)

    fig2.suptitle("Detailed modality combination analysis", fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig2, out / "fig_combo_dist_detail")
    plt.close(fig2)
    print(f"[combo_dist] fig_combo_dist_detail → {out}")
    print(f"[combo_dist] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers (reused from analyze_v7.py)
# ─────────────────────────────────────────────────────────────────────────────

def _load_v7_module(chicago_mil_dir: Path):
    """Import train_mm_abmil_v7.py without triggering __main__."""
    script = chicago_mil_dir / "train_mm_abmil_v7.py"
    if not script.exists():
        raise FileNotFoundError(script)
    parent = str(chicago_mil_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec   = importlib.util.spec_from_file_location("tv7", str(script))
    module = importlib.util.module_from_spec(spec)
    _argv_bak = sys.argv[:]
    sys.argv   = [str(script)]
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    finally:
        sys.argv = _argv_bak
    return module


def _tag_to_variant_kwargs(tag: str):
    """Map ckpts_* directory tag → (variant_str, build_kwargs). None if unknown."""
    if tag in ("early", "early_cls", "late", "middle", "middle_cls"):
        return tag, {}
    m = _re.match(r"crossattn(_cls)?_k(\d+)$", tag)
    if m:
        return ("crossattn_cls" if m.group(1) else "crossattn"), {"slot_k": int(m.group(2))}
    m = _re.match(r"crossattn_k(\d+)(_cls)?$", tag)
    if m:
        return ("crossattn_cls" if m.group(2) else "crossattn"), {"slot_k": int(m.group(1))}
    m = _re.match(r"crossmodal(_cls)?_k(\d+)$", tag)
    if m:
        return ("crossmodal_cls" if m.group(1) else "crossmodal"), {"slot_k": int(m.group(2))}
    m = _re.match(r"crossmodal_k(\d+)(_cls)?$", tag)
    if m:
        return ("crossmodal_cls" if m.group(2) else "crossmodal"), {"slot_k": int(m.group(1))}
    m = _re.match(r"iterative(_cls)?_r(\d+)_k(\d+)$", tag)
    if m:
        base = "iterative_cls" if m.group(1) else "iterative"
        return base, {"iter_r": int(m.group(2)), "slot_k": int(m.group(3))}
    m = _re.match(r"iterative_r(\d+)_k(\d+)(_cls)?$", tag)
    if m:
        base = "iterative_cls" if m.group(3) else "iterative"
        return base, {"iter_r": int(m.group(1)), "slot_k": int(m.group(2))}
    return None, None


def _build_and_load(tv7, tag: str, ckpt_path: Path, device):
    import torch
    variant, kwargs = _tag_to_variant_kwargs(tag)
    if variant is None:
        raise ValueError(f"Cannot map tag '{tag}' to build_model_v7 variant")
    model = tv7.build_model_v7(variant, **kwargs)
    ckpt  = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def _build_stem_to_meta(splits_csv: Path) -> Tuple[dict, pd.DataFrame]:
    """Compute gap-time TTE from acr_grade for every sample in the CSV."""
    df = pd.read_csv(str(splits_csv))
    df["anchor_dt"] = pd.to_datetime(df["anchor_dt"])

    # combo column
    df["combo"] = df.apply(_combo_from_row, axis=1)
    df["n_mods"] = df["combo"].apply(_n_mods)

    acr_mask = df["acr_grade"].apply(
        lambda g: isinstance(g, str) and (g.startswith("A1") or g.startswith("A2"))
    )
    acr_dates: dict = {}
    for _, row in df[acr_mask].iterrows():
        acr_dates.setdefault(row["patient_id"], []).append(row["anchor_dt"])
    last_date: dict = df.groupby("patient_id")["anchor_dt"].max().to_dict()

    meta: dict = {}
    for _, row in df.iterrows():
        stem = Path(str(row["file"])).stem
        pid  = row["patient_id"]
        t    = row["anchor_dt"]
        future = sorted([d for d in acr_dates.get(pid, []) if d > t])
        if future:
            tte, ev = float((future[0] - t).days), 1
        else:
            last = last_date.get(pid, t)
            tte, ev = float(max((last - t).days, 0)), 0
        g   = str(row.get("acr_grade", ""))
        lbl = (0.0 if g.startswith("A0")
               else 1.0 if (g.startswith("A1") or g.startswith("A2"))
               else float("nan"))
        meta[stem] = {
            "patient_id":     row.get("patient_id"),
            "anchor_dt":      t,
            "label":          lbl,
            "acr_status":     float(row["acr_status"]) if pd.notna(row.get("acr_status")) else float("nan"),
            "tte_next_acr":   tte,
            "event_next_acr": float(ev),
            "days_to_acr":    tte if ev == 1 else float("nan"),
            "combo":          row.get("combo", "Unknown"),
            "n_mods":         int(row.get("n_mods", 0)),
        }
    return meta, df


def _run_inference(tv7, model, tag: str, stems: list,
                   bag_cache: dict, device, stem_to_meta: dict,
                   split: int, fold: int,
                   stem_to_datasplit: Optional[Dict] = None) -> list:
    """Forward pass for stems. All v7 models return (logit, hazard, r_cls, r_surv).
    stem_to_datasplit maps stem → 'train'/'val'/'test' for labeling."""
    import torch
    rows = []
    with torch.no_grad():
        for stem in stems:
            bags = {m: bag_cache.get(stem, {}).get(m) for m in tv7.MODALITIES}
            bags["HE_coords"] = bag_cache.get(stem, {}).get("HE_coords")
            if all(v is None for k, v in bags.items() if k != "HE_coords"):
                continue
            try:
                out = model(bags, device)
                if not isinstance(out, tuple) or len(out) < 4:
                    continue
                logit, hazard, r_cls, r_surv = out[0], out[1], out[2], out[3]
                prob = float(torch.sigmoid(logit.float()).item())
                haz  = float(hazard.float().item())
                meta = stem_to_meta.get(stem, {})
                dsplit = (stem_to_datasplit or {}).get(stem, "test")
                rows.append({
                    "stem":           stem,
                    "variant":        tag,
                    "patient_id":     meta.get("patient_id"),
                    "anchor_dt":      meta.get("anchor_dt"),
                    "split":          split,
                    "fold":           fold,
                    "data_split":     dsplit,        # train / val / test
                    "cls_prob":       prob,
                    "hazard":         haz,
                    "label":          meta.get("label"),
                    "acr_status":     meta.get("acr_status"),
                    "tte_next_acr":   meta.get("tte_next_acr"),
                    "event_next_acr": meta.get("event_next_acr"),
                    "days_to_acr":    meta.get("days_to_acr"),
                    "combo":          meta.get("combo", "Unknown"),
                    "n_mods":         meta.get("n_mods", 0),
                    "rep_cls":        r_cls.detach().float().cpu().numpy(),
                    "rep_surv":       r_surv.detach().float().cpu().numpy(),
                })
            except Exception as exc:
                print(f"[infer]    {stem}: {exc}")
    return rows


def _collect_variant_data(
    results_dir:    Path,
    splits_csv:     Path,
    samples_dir:    Path,
    splits:         List[int],
    folds:          List[int],
    device_str:     str = "cpu",
    chicago_mil_dir: Optional[Path] = None,
    target_tags:    Optional[List[str]] = None,
) -> Optional[Dict]:
    """
    Run inference for all (split, fold) and collect representations.
    Returns: {tag: {"df": DataFrame, "reps_cls": ndarray, "reps_surv": ndarray}}
    """
    import torch

    chicago_mil = chicago_mil_dir if chicago_mil_dir is not None else results_dir.parent.parent
    print(f"[infer] Loading train_mm_abmil_v7 from {chicago_mil} ...")
    try:
        tv7 = _load_v7_module(chicago_mil)
    except Exception as e:
        print(f"[infer] Failed to load v7 module: {e}")
        return None

    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[infer] Device: {device}")

    stem_to_meta, df_csv = _build_stem_to_meta(splits_csv)
    all_rows: Dict[str, list] = {}

    for s in splits:
        for f in folds:
            fold_dir = results_dir / f"split{s}_fold{f}"
            if not fold_dir.exists():
                continue
            split_col  = f"split{s}_fold{f}"
            # Collect ALL splits (train/val/test) for full UMAP coverage
            stem_to_datasplit = {
                Path(str(row["file"])).stem: str(row.get(split_col, ""))
                for _, row in df_csv.iterrows()
                if str(row.get(split_col, "")) in ("train", "val", "test")
            }
            all_stems = list(stem_to_datasplit.keys())
            if not all_stems:
                continue

            n_tr = sum(1 for v in stem_to_datasplit.values() if v == "train")
            n_va = sum(1 for v in stem_to_datasplit.values() if v == "val")
            n_te = sum(1 for v in stem_to_datasplit.values() if v == "test")
            print(f"[infer] split={s} fold={f}: preloading {len(all_stems)} bags "
                  f"(train={n_tr} val={n_va} test={n_te}) ...")
            bag_cache = tv7.preload_bags(all_stems, str(samples_dir))

            for ckpt_dir in sorted(fold_dir.glob("ckpts_*")):
                tag     = ckpt_dir.name[len("ckpts_"):]
                if target_tags is not None and tag not in target_tags:
                    continue
                variant, kwargs = _tag_to_variant_kwargs(tag)
                if variant is None:
                    continue
                ckpt_file = ckpt_dir / "best_model.pt"
                if not ckpt_file.exists():
                    print(f"[infer]   no checkpoint for {tag} s={s} f={f} — skip")
                    continue

                print(f"[infer]   {tag} s={s} f={f}: building model ...")
                try:
                    model = _build_and_load(tv7, tag, ckpt_file, device)
                except Exception as e:
                    print(f"[infer]   build/load failed: {e}")
                    continue

                rows = _run_inference(tv7, model, tag, all_stems,
                                      bag_cache, device, stem_to_meta, s, f,
                                      stem_to_datasplit=stem_to_datasplit)
                all_rows.setdefault(tag, []).extend(rows)
                print(f"[infer]   {len(rows)} samples collected for {tag}")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del bag_cache

    if not all_rows:
        print("[infer] No data collected.")
        return None

    result: Dict[str, dict] = {}
    for tag, rows in all_rows.items():
        if not rows:
            continue
        reps_cls  = np.stack([r.pop("rep_cls")  for r in rows])
        reps_surv = np.stack([r.pop("rep_surv") for r in rows])
        df        = pd.DataFrame(rows)
        result[tag] = {"df": df, "reps_cls": reps_cls, "reps_surv": reps_surv}
        print(f"[infer] {tag}: {len(df)} samples, rep_dim={reps_cls.shape[1]}")
    return result


def _save_cache(variant_data: Dict, cache_path: Path):
    """Save inference results to NPZ cache."""
    arrays = {}
    meta_dfs = {}
    for tag, vd in variant_data.items():
        safe_tag = tag.replace("/", "_")
        arrays[f"{safe_tag}__reps_cls"]  = vd["reps_cls"]
        arrays[f"{safe_tag}__reps_surv"] = vd["reps_surv"]
        meta_dfs[safe_tag] = vd["df"]

    np.savez_compressed(str(cache_path) + "_arrays.npz", **arrays)
    import pickle
    with open(str(cache_path) + "_meta.pkl", "wb") as f:
        import pickle
        pickle.dump(meta_dfs, f)
    print(f"[cache] Saved to {cache_path}_arrays.npz + _meta.pkl")


def _load_cache(cache_path: Path) -> Optional[Dict]:
    """Load inference results from NPZ cache."""
    arr_file = str(cache_path) + "_arrays.npz"
    pkl_file = str(cache_path) + "_meta.pkl"
    if not (Path(arr_file).exists() and Path(pkl_file).exists()):
        return None
    import pickle
    arrays = np.load(arr_file, allow_pickle=True)
    with open(pkl_file, "rb") as f:
        meta_dfs = pickle.load(f)

    result = {}
    for safe_tag, df in meta_dfs.items():
        if f"{safe_tag}__reps_cls" in arrays:
            result[safe_tag] = {
                "df":         df,
                "reps_cls":   arrays[f"{safe_tag}__reps_cls"],
                "reps_surv":  arrays[f"{safe_tag}__reps_surv"],
            }
    print(f"[cache] Loaded {len(result)} variants from {cache_path}_*.npz/pkl")
    return result


def _get_or_run_inference(
    results_dir: Path, splits_csv: Path, samples_dir: Path,
    splits: List[int], folds: List[int], device_str: str,
    chicago_mil_dir: Optional[Path], output_dir: Path,
    exp: str, target_tags: Optional[List[str]] = None,
) -> Optional[Dict]:
    """Load cached inference results or run inference if cache missing."""
    cache_path = output_dir / f"inference_cache_{exp}"
    cached = _load_cache(cache_path)
    if cached is not None:
        # Filter to target_tags if requested
        if target_tags is not None:
            cached = {t: v for t, v in cached.items() if t in target_tags}
        return cached

    print(f"[infer] No cache found at {cache_path} — running inference ...")
    variant_data = _collect_variant_data(
        results_dir, splits_csv, samples_dir, splits, folds, device_str,
        chicago_mil_dir=chicago_mil_dir,
        target_tags=target_tags,
    )
    if variant_data:
        _save_cache(variant_data, cache_path)
    return variant_data


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3: umap
# ─────────────────────────────────────────────────────────────────────────────

def _scatter(ax, xy, values, title, cmap, vmin=None, vmax=None, s=14, alpha=0.7):
    values = np.asarray(values, dtype=float)
    mask   = ~np.isnan(values)
    if mask.sum() == 0:
        ax.set_title(f"{title} (no data)", fontsize=7)
        return
    sc = ax.scatter(xy[mask, 0], xy[mask, 1],
                    c=values[mask], cmap=cmap, vmin=vmin, vmax=vmax,
                    s=s, alpha=alpha, linewidths=0)
    if (~mask).sum() > 0:
        ax.scatter(xy[~mask, 0], xy[~mask, 1], c="lightgrey",
                   s=s * 0.4, alpha=0.3, linewidths=0)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])


def _hexbin_enrichment(ax, xy, values, title, cmap, vmin, vmax,
                        gridsize=30, mincnt=3, cbar_label=""):
    values = np.asarray(values, dtype=float)
    mask   = ~np.isnan(values)
    if mask.sum() < mincnt:
        ax.set_title(f"{title} (no data)", fontsize=7)
        return
    hb = ax.hexbin(xy[mask, 0], xy[mask, 1], C=values[mask],
                   gridsize=gridsize, mincnt=mincnt,
                   reduce_C_function=np.mean,
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   linewidths=0.2)
    plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])


def _pct(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0, 1.0
    return float(np.percentile(arr, 2)), float(np.percentile(arr, 98))


SPLIT_COLORS = {"train": "#2196F3", "val": "#FF9800", "test": "#E53935"}

def _plot_umap_split(df: pd.DataFrame, xy: np.ndarray, dsplit: str,
                     title: str, out_path: Path):
    """
    3×3 UMAP for a single data split (train / val / test):
    Row 0: [ACR label, modality combo, n_mods (1/2/3/4)]
    Row 1: [cls_prob, TTE (event=1 only), hazard]
    Row 2: [hexbin ACR+ prop, hexbin mean TTE, hexbin mean hazard]
    """
    mask_sp = df["data_split"].values == dsplit
    if mask_sp.sum() == 0:
        print(f"[umap] No {dsplit} samples — skipping")
        return

    df_s = df[mask_sp].reset_index(drop=True)
    xy_s = xy[mask_sp]

    fig, axes = plt.subplots(3, 3, figsize=(TWO_COL_W * 1.4, TWO_COL_W * 1.3))
    fig.patch.set_facecolor("white")

    labels   = np.asarray(df_s["label"].values,         dtype=float)
    cls_prob = np.asarray(df_s["cls_prob"].values,       dtype=float)
    tte      = np.asarray(df_s["days_to_acr"].values,    dtype=float)
    haz      = np.asarray(df_s["hazard"].values,         dtype=float)
    n_mods_v = np.asarray(df_s["n_mods"].values,         dtype=float)
    combos   = df_s["combo"].values

    for ax in axes.flat:
        ax.set_facecolor("white")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ["top", "right", "left", "bottom"]:
            ax.spines[sp].set_visible(False)

    # ── Row 0 ─────────────────────────────────────────────────────────────────
    ax00 = axes[0, 0]
    lbl_known = ~np.isnan(labels)
    if lbl_known.sum() > 0:
        c_lbl = np.where(labels[lbl_known] == 1.0, "#E53935", "#1E88E5")
        ax00.scatter(xy_s[lbl_known, 0], xy_s[lbl_known, 1],
                     c=c_lbl, s=10, alpha=0.75, linewidths=0, rasterized=True)
    ax00.legend(handles=[Patch(color="#E53935", label="ACR+ (label=1)"),
                          Patch(color="#1E88E5", label="No ACR (label=0)")],
                fontsize=6, loc="best", framealpha=0.8)
    ax00.set_title("ACR label", fontsize=7, fontweight="bold")

    # Modality combo
    ax01 = axes[0, 1]
    for c_name in sorted(set(combos), key=lambda x: -np.sum(combos == x)):
        mask_c = combos == c_name
        ax01.scatter(xy_s[mask_c, 0], xy_s[mask_c, 1],
                     c=COMBO_COLORS.get(c_name, DEFAULT_COMBO_COLOR),
                     s=10, alpha=0.72, linewidths=0, label=c_name, rasterized=True)
    ax01.legend(fontsize=5, loc="upper right", ncol=2, framealpha=0.75,
                markerscale=1.2, handlelength=1, borderpad=0.4)
    ax01.set_title("Modality combination", fontsize=7, fontweight="bold")

    # n_mods
    ax02 = axes[0, 2]
    sc_nm = ax02.scatter(xy_s[:, 0], xy_s[:, 1], c=n_mods_v,
                          cmap="plasma", vmin=0.5, vmax=4.5,
                          s=10, alpha=0.75, linewidths=0, rasterized=True)
    cb = plt.colorbar(sc_nm, ax=ax02, fraction=0.046, pad=0.04)
    cb.set_ticks([1, 2, 3, 4]); cb.ax.tick_params(labelsize=6)
    ax02.set_title("# modalities present", fontsize=7, fontweight="bold")

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    _scatter(axes[1, 0], xy_s, cls_prob, "Classification probability", "RdYlGn_r", 0, 1)

    ax11 = axes[1, 1]
    tte_valid = ~np.isnan(tte)
    if tte_valid.sum() > 0:
        vmax_t = float(np.percentile(tte[tte_valid], 98))
        sc_t = ax11.scatter(xy_s[tte_valid, 0], xy_s[tte_valid, 1],
                             c=tte[tte_valid], cmap="viridis_r",
                             vmin=0, vmax=vmax_t, s=10, alpha=0.8, linewidths=0, rasterized=True)
        cb2 = plt.colorbar(sc_t, ax=ax11, fraction=0.046, pad=0.04)
        cb2.ax.tick_params(labelsize=6)
    ax11.set_title("Days to next ACR (event=1 only)", fontsize=7, fontweight="bold")

    hmin, hmax = _pct(haz)
    _scatter(axes[1, 2], xy_s, haz, "Hazard score", "hot_r", hmin, hmax)

    # ── Row 2 ─────────────────────────────────────────────────────────────────
    _hexbin_enrichment(axes[2, 0], xy_s, labels,
                       "Hexbin: ACR+ density", "RdBu_r", 0.0, 1.0,
                       cbar_label="prop ACR+")
    vmax_hex = (float(np.nanpercentile(tte[~np.isnan(tte)], 98))
                if (~np.isnan(tte)).sum() > 0 else 1.0)
    _hexbin_enrichment(axes[2, 1], xy_s, tte,
                       "Hexbin: mean days-to-ACR", "viridis_r", 0.0, vmax_hex,
                       cbar_label="mean days")
    _hexbin_enrichment(axes[2, 2], xy_s, haz,
                       "Hexbin: mean hazard", "hot_r", hmin, hmax,
                       cbar_label="mean hazard")

    n_sp = int(mask_sp.sum())
    n_pos = int((labels == 1.0).sum())
    fig.suptitle(f"{title}  [{dsplit.upper()}  n={n_sp}  ACR+={n_pos}]",
                 fontsize=8, fontweight="bold", y=1.002)
    plt.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)
    print(f"[umap] → {out_path}.pdf")

def task_umap(
    results_dir: Path, splits_csv: Path, samples_dir: Path,
    output_dir: Path, exp: str,
    splits: List[int], folds: List[int],
    device_str: str, umap_n_neighbors: int, umap_min_dist: float,
    chicago_mil_dir: Optional[Path],
):
    if not HAS_UMAP:
        print("[umap] umap-learn not installed — skipping.")
        return
    plt.rcParams.update(NATURE_RC)
    out = output_dir / "umap"
    out.mkdir(parents=True, exist_ok=True)

    variant_data = _get_or_run_inference(
        results_dir, splits_csv, samples_dir, splits, folds,
        device_str, chicago_mil_dir, output_dir, exp,
    )
    if not variant_data:
        return

    tags_sorted = [t for t in VARIANT_TAGS if t in variant_data]
    tags_sorted += [t for t in sorted(variant_data.keys()) if t not in tags_sorted]

    # Store UMAP coords in variant_data so comparison figure can use them
    stored_xy: Dict[str, Dict[str, np.ndarray]] = {}

    for tag in tags_sorted:
        vd     = variant_data[tag]
        df     = vd["df"]           # operate on actual df, not copy
        r_cls  = vd["reps_cls"]
        r_surv = vd["reps_surv"]
        v_out  = out / tag
        v_out.mkdir(exist_ok=True)
        disp   = VARIANT_DISPLAY.get(tag, tag)

        reducer = UMAPTransform(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist,
                                random_state=42, n_jobs=4)

        # ── Fit UMAP on ALL samples (train+val+test combined) ──────────────────
        print(f"[umap] {tag}: UMAP on r_cls ({r_cls.shape}) ...")
        xy_cls = reducer.fit_transform(r_cls)
        df["umap_cls_x"] = xy_cls[:, 0]
        df["umap_cls_y"] = xy_cls[:, 1]

        print(f"[umap] {tag}: UMAP on r_surv ({r_surv.shape}) ...")
        xy_surv = reducer.fit_transform(r_surv)
        df["umap_surv_x"] = xy_surv[:, 0]
        df["umap_surv_y"] = xy_surv[:, 1]

        # Store for comparison figure
        stored_xy[tag] = {"cls": xy_cls, "surv": xy_surv}

        # ── Per-split figures (train / val / test) ─────────────────────────────
        for dsplit in ["train", "val", "test"]:
            _plot_umap_split(df, xy_cls,  dsplit,
                             f"{disp} — classification representation (r_cls)",
                             v_out / f"fig_umap_{tag}_rcls_{dsplit}")
            _plot_umap_split(df, xy_surv, dsplit,
                             f"{disp} — survival representation (r_surv)",
                             v_out / f"fig_umap_{tag}_rsurv_{dsplit}")

        # ── Patient-level UMAP — TEST SET ONLY ────────────────────────────────
        test_mask = df["data_split"].values == "test"
        df_test   = df[test_mask].reset_index(drop=True)
        xy_test   = xy_cls[test_mask]
        pids_test = df_test["patient_id"].values
        unique_pids = list(dict.fromkeys(pids_test))

        if len(unique_pids) >= 4:
            r_cls_test = r_cls[test_mask]
            pat_reps, pat_labels, pat_probs, pat_combos, pat_clad = [], [], [], [], []
            for pid in unique_pids:
                mask_p  = pids_test == pid
                pat_reps.append(r_cls_test[mask_p].mean(axis=0))
                lbl_v   = df_test.loc[mask_p, "label"].dropna()
                pat_labels.append(float(lbl_v.mode()[0]) if len(lbl_v) > 0 else float("nan"))
                pat_probs.append(float(df_test.loc[mask_p, "cls_prob"].mean()))
                combo_v = df_test.loc[mask_p, "combo"]
                pat_combos.append(combo_v.mode()[0] if len(combo_v) > 0 else "Unknown")
                clad_v  = df_test.loc[mask_p, "acr_status"].dropna()
                pat_clad.append(float(clad_v.mode()[0]) if len(clad_v) > 0 else float("nan"))

            reducer_p  = UMAPTransform(
                n_neighbors=min(umap_n_neighbors, len(unique_pids) - 1),
                min_dist=umap_min_dist, random_state=42)
            xy_pat     = reducer_p.fit_transform(np.stack(pat_reps))
            labels_p   = np.array(pat_labels, dtype=float)
            probs_p    = np.array(pat_probs,  dtype=float)
            combos_arr = np.array(pat_combos)

            fig_p, axes_p = plt.subplots(1, 3, figsize=(TWO_COL_W, ONE_COL_W * 1.4),
                                          facecolor="white")
            for ax in axes_p:
                ax.set_facecolor("white")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ["top","right","left","bottom"]: ax.spines[sp].set_visible(False)

            # col 0: ACR label
            lbl_k = ~np.isnan(labels_p)
            if lbl_k.sum() > 0:
                c_p = np.where(labels_p[lbl_k] == 1.0, "#E53935", "#1E88E5")
                axes_p[0].scatter(xy_pat[lbl_k, 0], xy_pat[lbl_k, 1],
                                  c=c_p, s=40, alpha=0.85, linewidths=0, rasterized=True)
            axes_p[0].legend(handles=[Patch(color="#E53935", label="ACR+ patient"),
                                       Patch(color="#1E88E5", label="No-ACR patient")],
                             fontsize=6, loc="best", framealpha=0.8)
            axes_p[0].set_title("Patient ACR label", fontsize=7, fontweight="bold")

            # col 1: modality combo (most common combo per patient)
            for c_name in sorted(set(pat_combos), key=lambda x: -np.sum(combos_arr == x)):
                mask_pc = combos_arr == c_name
                axes_p[1].scatter(xy_pat[mask_pc, 0], xy_pat[mask_pc, 1],
                                  c=COMBO_COLORS.get(c_name, DEFAULT_COMBO_COLOR),
                                  s=40, alpha=0.8, linewidths=0, label=c_name, rasterized=True)
            axes_p[1].legend(fontsize=5, ncol=2, framealpha=0.75, loc="best",
                             markerscale=1.2, handlelength=1)
            axes_p[1].set_title("Dominant modality combo", fontsize=7, fontweight="bold")

            # col 2: mean cls_prob
            sc_pp = axes_p[2].scatter(xy_pat[:, 0], xy_pat[:, 1], c=probs_p,
                                       cmap="RdYlGn_r", vmin=0, vmax=1,
                                       s=40, alpha=0.85, linewidths=0, rasterized=True)
            cb = plt.colorbar(sc_pp, ax=axes_p[2], fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=6)
            axes_p[2].set_title("Mean classification probability", fontsize=7, fontweight="bold")

            n_pat = len(unique_pids)
            n_acr_pat = int((labels_p == 1.0).sum())
            fig_p.suptitle(
                f"{disp} — patient-level UMAP  [TEST SET: {n_pat} patients, {n_acr_pat} ACR+]",
                fontsize=8, fontweight="bold")
            plt.tight_layout()
            _save_fig(fig_p, v_out / f"fig_umap_patient_{tag}")
            plt.close(fig_p)

            pd.DataFrame({"patient_id": unique_pids, "umap_x": xy_pat[:, 0],
                          "umap_y": xy_pat[:, 1], "label": pat_labels,
                          "mean_cls_prob": pat_probs, "combo": pat_combos}
                         ).to_csv(v_out / f"patient_umap_{tag}.csv", index=False)

        # Save embeddings CSV
        df.drop(columns=["rep_cls","rep_surv"], errors="ignore").to_csv(
            v_out / f"embeddings_{tag}.csv", index=False)

    # ── Cross-variant comparison — TEST SET, r_cls only ───────────────────────
    # Uses stored UMAP coords (no KeyError)
    n_variants = len(tags_sorted)
    if n_variants > 0:
        n_cols = 4
        fig_cmp, axes_cmp = plt.subplots(
            n_variants, n_cols,
            figsize=(n_cols * 3.0, n_variants * 2.6),
            facecolor="white")
        if n_variants == 1:
            axes_cmp = axes_cmp[np.newaxis, :]

        col_titles = ["ACR label", "Modality combo", "Cls probability", "Hexbin ACR+ density"]

        for i, tag in enumerate(tags_sorted):
            df_v   = variant_data[tag]["df"]
            # Use only test set
            t_mask = df_v["data_split"].values == "test"
            xy     = stored_xy[tag]["cls"][t_mask]
            df_t   = df_v[t_mask].reset_index(drop=True)
            labels_v = np.asarray(df_t["label"].values, dtype=float)
            combos_v = df_t["combo"].values

            for ci in range(n_cols):
                axes_cmp[i, ci].set_facecolor("white")
                axes_cmp[i, ci].set_xticks([]); axes_cmp[i, ci].set_yticks([])
                for sp in ["top","right","left","bottom"]:
                    axes_cmp[i, ci].spines[sp].set_visible(False)

            # col 0: ACR label
            lk = ~np.isnan(labels_v)
            if lk.sum() > 0:
                cl = np.where(labels_v[lk] == 1.0, "#E53935", "#1E88E5")
                axes_cmp[i, 0].scatter(xy[lk, 0], xy[lk, 1],
                                       c=cl, s=5, alpha=0.65, linewidths=0, rasterized=True)
            axes_cmp[i, 0].set_ylabel(VARIANT_DISPLAY.get(tag, tag), fontsize=7, labelpad=4)
            if i == 0:
                axes_cmp[i, 0].set_title(col_titles[0], fontsize=7, fontweight="bold")
                axes_cmp[i, 0].legend(
                    handles=[Patch(color="#E53935", label="ACR+"),
                              Patch(color="#1E88E5", label="No ACR")],
                    fontsize=5, loc="upper right", framealpha=0.7, markerscale=0.9)

            # col 1: modality combo
            for c_name in sorted(set(combos_v), key=lambda x: -np.sum(combos_v == x)):
                mask_cv = combos_v == c_name
                axes_cmp[i, 1].scatter(xy[mask_cv, 0], xy[mask_cv, 1],
                                       c=COMBO_COLORS.get(c_name, DEFAULT_COMBO_COLOR),
                                       s=5, alpha=0.65, linewidths=0, label=c_name, rasterized=True)
            if i == 0:
                axes_cmp[i, 1].set_title(col_titles[1], fontsize=7, fontweight="bold")
                axes_cmp[i, 1].legend(fontsize=4, ncol=1, framealpha=0.6,
                                      loc="upper right", markerscale=0.9, borderpad=0.3)

            # col 2: cls prob (continuous)
            _scatter(axes_cmp[i, 2], xy, df_t["cls_prob"].values,
                     col_titles[2] if i == 0 else "", "RdYlGn_r", 0, 1, s=5)

            # col 3: hexbin ACR+ enrichment
            _hexbin_enrichment(axes_cmp[i, 3], xy, labels_v,
                               col_titles[3] if i == 0 else "",
                               "RdBu_r", 0.0, 1.0, gridsize=18, mincnt=3,
                               cbar_label="prop ACR+")

        fig_cmp.suptitle("Fusion variant comparison — r_cls UMAP (test set)",
                          fontsize=8, fontweight="bold")
        plt.tight_layout()
        _save_fig(fig_cmp, out / "fig_umap_comparison")
        plt.close(fig_cmp)
        print(f"[umap] Comparison → {out}/fig_umap_comparison.pdf")

    print(f"[umap] Done. All outputs in {out}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4: combo_perf
# ─────────────────────────────────────────────────────────────────────────────

def _safe_cindex(tte, event, hazard):
    """Concordance index: fraction of concordant pairs (higher hazard → shorter TTE)."""
    try:
        from sklearn.utils import check_consistent_length
    except ImportError:
        pass
    tte = np.asarray(tte, dtype=float)
    ev  = np.asarray(event, dtype=float)
    hz  = np.asarray(hazard, dtype=float)
    valid = ~np.isnan(tte) & ~np.isnan(ev) & ~np.isnan(hz) & (ev >= 0)
    tte, ev, hz = tte[valid], ev[valid], hz[valid]
    if len(tte) < 2:
        return float("nan")
    concordant = discordant = 0
    for i in range(len(tte)):
        if ev[i] == 0:
            continue
        for j in range(len(tte)):
            if i == j:
                continue
            if tte[i] < tte[j]:
                concordant += hz[i] > hz[j]
                discordant += hz[i] < hz[j]
    total = concordant + discordant
    return float(concordant / total) if total > 0 else float("nan")


def _compute_combo_metrics(df_sub: pd.DataFrame, exp: str = "alt", min_n: int = 5) -> Dict:
    """
    Compute primary metric per combo for a predictions DataFrame.
    PRIMARY metric: BAcc for cls/alt, C-index for surv.
    Always run on test split only.
    """
    prim = PRIMARY_METRIC.get(exp, "bacc")
    df_test = df_sub[df_sub["data_split"] == "test"] if "data_split" in df_sub.columns else df_sub
    results = {}
    for combo, grp in df_test.groupby("combo"):
        n       = len(grp)
        labels  = grp["label"].dropna().values.astype(float)
        probs   = grp.loc[grp["label"].notna(), "cls_prob"].values.astype(float)
        pos_rate = float(np.nanmean(labels)) if len(labels) > 0 else float("nan")

        if n < min_n or len(labels) < min_n:
            results[combo] = {"n": n, "primary": float("nan"), "auc": float("nan"),
                               "bacc": float("nan"), "cindex": float("nan"),
                               "pos_rate": pos_rate}
            continue

        # AUC
        try:
            auc_v = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan")
        except Exception:
            auc_v = float("nan")

        # BAcc
        try:
            preds = (probs >= 0.5).astype(int)
            bacc_v = (float(balanced_accuracy_score(labels.astype(int), preds))
                      if len(np.unique(labels)) == 2 else float("nan"))
        except Exception:
            bacc_v = float("nan")

        # C-index (uses hazard and tte)
        if "tte_next_acr" in grp.columns and "event_next_acr" in grp.columns:
            ci_v = _safe_cindex(grp["tte_next_acr"].values,
                                grp["event_next_acr"].values,
                                grp["hazard"].values)
        else:
            ci_v = float("nan")

        primary_v = ci_v if prim == "c_index" else bacc_v
        results[combo] = {"n": n, "primary": primary_v, "auc": auc_v,
                           "bacc": bacc_v, "cindex": ci_v, "pos_rate": pos_rate}
    return results


def task_combo_perf(
    results_dir: Path, splits_csv: Path, samples_dir: Path,
    output_dir: Path, exp: str, focus_variant: str,
    splits: List[int], folds: List[int],
    device_str: str, chicago_mil_dir: Optional[Path],
    min_combo_n: int = 10,
):
    plt.rcParams.update(NATURE_RC)
    out = output_dir / "combo_perf"
    out.mkdir(parents=True, exist_ok=True)

    prim_key   = PRIMARY_METRIC.get(exp, "bacc")
    prim_label = METRIC_DISPLAY.get(prim_key, prim_key)

    variant_data = _get_or_run_inference(
        results_dir, splits_csv, samples_dir, splits, folds,
        device_str, chicago_mil_dir, output_dir, exp,
    )
    if not variant_data:
        return

    tags_sorted = [t for t in VARIANT_TAGS if t in variant_data]
    tags_sorted += [t for t in sorted(variant_data.keys()) if t not in tags_sorted]

    focus_tag = focus_variant if focus_variant in variant_data else (tags_sorted[0] if tags_sorted else None)
    if focus_tag is None:
        return

    # ── CSV output (all variants, all combos, test set) ────────────────────────
    csv_rows = []
    for tag in tags_sorted:
        cm = _compute_combo_metrics(variant_data[tag]["df"], exp=exp, min_n=1)
        for combo, vals in cm.items():
            csv_rows.append({"variant": tag, "combo": combo, **vals})
    pd.DataFrame(csv_rows).to_csv(out / "combo_metrics.csv", index=False)

    # ── Metrics for focus variant ────────────────────────────────────────────
    df_focus_test = variant_data[focus_tag]["df"]
    df_focus_test = df_focus_test[df_focus_test["data_split"] == "test"]
    combo_metrics  = _compute_combo_metrics(variant_data[focus_tag]["df"], exp=exp, min_n=min_combo_n)
    combo_metrics  = {c: v for c, v in combo_metrics.items() if v["n"] >= min_combo_n}
    combos_sorted  = sorted(combo_metrics, key=lambda c: combo_metrics[c]["n"], reverse=True)

    if not combos_sorted:
        print(f"[combo_perf] No combos with n≥{min_combo_n} in test set.")
        return

    n_c  = len(combos_sorted)
    x_p  = np.arange(n_c)
    cols = [COMBO_COLORS.get(c, DEFAULT_COMBO_COLOR) for c in combos_sorted]

    # ── Figure 1: Primary metric + AUC + count/pos_rate (focus variant) ───────
    fig1, axes1 = plt.subplots(
        3, 1,
        figsize=(max(TWO_COL_W, n_c * 0.7 + 1.5), 9.0),
        gridspec_kw={"height_ratios": [1.4, 1.4, 0.9]},
        facecolor="white")

    # Panel 0: Primary metric (BAcc or C-index)
    prim_vals = [combo_metrics[c]["primary"] for c in combos_sorted]
    axes1[0].bar(x_p, prim_vals, color=cols, alpha=0.85, width=0.65, edgecolor="white", lw=0.5)
    axes1[0].axhline(0.5, color="#aaaaaa", lw=0.8, ls=":", zorder=0)
    for xi, v in enumerate(prim_vals):
        if not np.isnan(v):
            axes1[0].text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    axes1[0].set_xticks(x_p); axes1[0].set_xticklabels([], fontsize=0)
    axes1[0].set_ylim(0.25, 1.05); axes1[0].set_ylabel(f"★ {prim_label}", fontsize=8, fontweight="bold")
    axes1[0].set_title(
        f"{VARIANT_DISPLAY.get(focus_tag, focus_tag)} — {prim_label} per modality combination "
        f"({exp.upper()} task, test set)",
        fontsize=8, fontweight="bold")
    axes1[0].spines[["top","right"]].set_visible(False)
    axes1[0].grid(axis="y", lw=0.5, alpha=0.4, ls="--")

    # Panel 1: AUC
    auc_vals = [combo_metrics[c]["auc"] for c in combos_sorted]
    axes1[1].bar(x_p, auc_vals, color=cols, alpha=0.72, width=0.65, edgecolor="white", lw=0.5)
    axes1[1].axhline(0.5, color="#aaaaaa", lw=0.8, ls=":", zorder=0)
    for xi, v in enumerate(auc_vals):
        if not np.isnan(v):
            axes1[1].text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    axes1[1].set_xticks(x_p); axes1[1].set_xticklabels([], fontsize=0)
    axes1[1].set_ylim(0.25, 1.05); axes1[1].set_ylabel("AUC", fontsize=8)
    axes1[1].set_title("AUC per modality combination", fontsize=8, fontweight="bold")
    axes1[1].spines[["top","right"]].set_visible(False)
    axes1[1].grid(axis="y", lw=0.5, alpha=0.4, ls="--")

    # Panel 2: count bars + ACR positive rate line
    counts   = [combo_metrics[c]["n"]        for c in combos_sorted]
    posrates = [combo_metrics[c]["pos_rate"] for c in combos_sorted]
    axes1[2].bar(x_p, counts, color=cols, alpha=0.55, width=0.65, edgecolor="white", lw=0.5)
    ax2r = axes1[2].twinx()
    ax2r.plot(x_p, posrates, "o-", color="#C62828", lw=1.4, markersize=5,
              markeredgecolor="white", markeredgewidth=0.6, label="ACR pos rate")
    ax2r.set_ylim(0, 0.6); ax2r.set_ylabel("ACR+ rate", fontsize=7, color="#C62828")
    ax2r.tick_params(axis="y", labelcolor="#C62828", labelsize=6)
    axes1[2].set_xticks(x_p)
    axes1[2].set_xticklabels(combos_sorted, fontsize=6.5, rotation=30, ha="right")
    axes1[2].set_ylabel("n samples (test)", fontsize=7)
    axes1[2].set_title("Sample count and ACR positive rate", fontsize=8)
    axes1[2].spines[["top","right"]].set_visible(False)
    axes1[2].grid(axis="y", lw=0.5, alpha=0.4, ls="--")
    ax2r.legend(fontsize=6.5, loc="upper right")

    plt.tight_layout()
    _save_fig(fig1, out / "fig_combo_perf")
    plt.close(fig1)
    print(f"[combo_perf] fig_combo_perf → {out}")

    # ── Figure 2: primary metric heatmap — all variants × combos ──────────────
    all_combos_set = set()
    for tag in tags_sorted:
        cm = _compute_combo_metrics(variant_data[tag]["df"], exp=exp, min_n=min_combo_n)
        all_combos_set.update(cm.keys())
    n_total = sum(variant_data[tags_sorted[0]]["df"]["data_split"].eq("test"))
    all_combos_list = sorted(all_combos_set,
        key=lambda c: variant_data[tags_sorted[0]]["df"][
            (variant_data[tags_sorted[0]]["df"]["combo"]==c) &
            (variant_data[tags_sorted[0]]["df"]["data_split"]=="test")].shape[0],
        reverse=True)

    if all_combos_list:
        heat = np.full((len(all_combos_list), len(tags_sorted)), float("nan"))
        for j, tag in enumerate(tags_sorted):
            cm = _compute_combo_metrics(variant_data[tag]["df"], exp=exp, min_n=min_combo_n)
            for i, combo in enumerate(all_combos_list):
                heat[i, j] = cm.get(combo, {}).get("primary", float("nan"))

        fig2, ax2 = plt.subplots(
            figsize=(max(TWO_COL_W, len(tags_sorted) * 1.4 + 1.5),
                     max(3.5, len(all_combos_list) * 0.65 + 1.8)),
            facecolor="white")

        im2 = ax2.imshow(heat, aspect="auto", cmap="RdYlGn", vmin=0.35, vmax=0.95)
        ax2.set_xticks(range(len(tags_sorted)))
        ax2.set_xticklabels([VARIANT_DISPLAY.get(t, t) for t in tags_sorted],
                             fontsize=7.5, rotation=30, ha="right")
        ax2.set_yticks(range(len(all_combos_list)))
        # Add sample counts to y-labels
        df0 = variant_data[tags_sorted[0]]["df"]
        ylabels = []
        for c in all_combos_list:
            n = int(df0[(df0["combo"]==c) & (df0["data_split"]=="test")].shape[0])
            ylabels.append(f"{c}  (n={n})")
        ax2.set_yticklabels(ylabels, fontsize=6.5)
        for i in range(len(all_combos_list)):
            for j in range(len(tags_sorted)):
                v = heat[i, j]
                if not np.isnan(v):
                    tc = "white" if (v < 0.45 or v > 0.80) else "#1a1a2e"
                    ax2.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6, color=tc)
        cb = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, shrink=0.8)
        cb.set_label(f"★ {prim_label}", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        ax2.set_title(f"{prim_label} per modality combo × variant  ({exp.upper()} task, test set)",
                      fontsize=8, fontweight="bold")
        ax2.tick_params(length=0)
        for sp in ax2.spines.values(): sp.set_visible(False)

        plt.tight_layout()
        _save_fig(fig2, out / "fig_combo_perf_all_variants")
        plt.close(fig2)

    print(f"[combo_perf] Done. Outputs in {out}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5: multimodal_analysis
# ─────────────────────────────────────────────────────────────────────────────

def _compute_subset_metrics(df_sub: pd.DataFrame, label_col: str = "label",
                              prob_col: str = "cls_prob") -> Dict:
    """Compute AUC, BAcc, MCC for a predictions subset."""
    valid = df_sub[df_sub[label_col].notna()]
    labels = valid[label_col].values.astype(float)
    probs  = valid[prob_col].values.astype(float)
    n = len(labels)
    if n < 5 or len(np.unique(labels)) < 2:
        return {"n": n, "auc": float("nan"), "bacc": float("nan"), "mcc": float("nan")}
    try:
        auc_v  = float(roc_auc_score(labels, probs))
        preds  = (probs >= 0.5).astype(int)
        bacc_v = float(balanced_accuracy_score(labels.astype(int), preds))
        mcc_v  = float(matthews_corrcoef(labels.astype(int), preds))
    except Exception:
        auc_v  = float("nan")
        bacc_v = float("nan")
        mcc_v  = float("nan")
    return {"n": n, "auc": auc_v, "bacc": bacc_v, "mcc": mcc_v}


def _primary_metric_from_df(df_sub: pd.DataFrame, exp: str) -> Tuple[float, int]:
    """Compute task-appropriate primary metric on test split of df_sub."""
    prim = PRIMARY_METRIC.get(exp, "bacc")
    if "data_split" in df_sub.columns:
        df_sub = df_sub[df_sub["data_split"] == "test"]
    n = len(df_sub)
    if n < 5:
        return float("nan"), n
    if prim == "c_index":
        val = _safe_cindex(
            df_sub["tte_next_acr"].values,
            df_sub["event_next_acr"].values,
            df_sub["hazard"].values,
        )
    else:
        valid = df_sub[df_sub["label"].notna()]
        labels = valid["label"].values.astype(float)
        probs  = valid["cls_prob"].values.astype(float)
        if len(np.unique(labels)) < 2 or len(labels) < 5:
            return float("nan"), n
        try:
            preds = (probs >= 0.5).astype(int)
            val = float(balanced_accuracy_score(labels.astype(int), preds))
        except Exception:
            val = float("nan")
    return val, n


def _run_inference_single_mod(
    tv7, model, tag: str, stems: list,
    bag_cache: dict, device, stem_to_meta: dict,
    split: int, fold: int, target_mod: str,
    stem_to_datasplit: Optional[Dict] = None,
) -> list:
    """Forward pass with only target_mod; all other modality slots = None."""
    import torch
    rows = []
    with torch.no_grad():
        for stem in stems:
            target_bag = bag_cache.get(stem, {}).get(target_mod)
            if target_bag is None:
                continue
            bags: Dict = {m: None for m in tv7.MODALITIES}
            bags[target_mod] = target_bag
            bags["HE_coords"] = bag_cache.get(stem, {}).get("HE_coords")
            try:
                out = model(bags, device)
                if not isinstance(out, tuple) or len(out) < 2:
                    continue
                logit, hazard = out[0], out[1]
                prob = float(torch.sigmoid(logit.float()).item())
                haz  = float(hazard.float().item())
                meta = stem_to_meta.get(stem, {})
                dsplit = (stem_to_datasplit or {}).get(stem, "test")
                rows.append({
                    "stem":           stem,
                    "variant":        f"{tag}__single_{target_mod}",
                    "data_split":     dsplit,
                    "split":          split,
                    "fold":           fold,
                    "cls_prob":       prob,
                    "hazard":         haz,
                    "label":          meta.get("label"),
                    "tte_next_acr":   meta.get("tte_next_acr"),
                    "event_next_acr": meta.get("event_next_acr"),
                    "combo":          meta.get("combo", "Unknown"),
                    "n_mods":         meta.get("n_mods", 0),
                    "target_mod":     target_mod,
                })
            except Exception as exc:
                print(f"[singlemod] {stem}: {exc}")
    return rows


def _get_or_run_singlemod(
    results_dir: Path, splits_csv: Path, samples_dir: Path,
    splits: List[int], folds: List[int],
    device_str: str, chicago_mil_dir: Optional[Path],
    output_dir: Path, exp: str, focus_tag: str,
) -> Optional[pd.DataFrame]:
    """Load or run single-modality inference for focus_tag. Returns long DataFrame."""
    import torch
    cache_csv = output_dir / f"singlemod_cache_{exp}_{focus_tag}.csv"
    if cache_csv.exists():
        df = pd.read_csv(str(cache_csv))
        print(f"[singlemod] Loaded from {cache_csv}: {len(df)} rows")
        return df

    print(f"[singlemod] Running single-mod inference for {focus_tag} ...")
    chicago_mil = chicago_mil_dir if chicago_mil_dir is not None else results_dir.parent.parent
    try:
        tv7 = _load_v7_module(chicago_mil)
    except Exception as e:
        print(f"[singlemod] Cannot load v7 module: {e}")
        return None

    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    stem_to_meta, df_csv = _build_stem_to_meta(splits_csv)
    variant, kwargs = _tag_to_variant_kwargs(focus_tag)
    if variant is None:
        print(f"[singlemod] Unknown variant tag: {focus_tag}")
        return None

    all_rows: list = []
    for s in splits:
        for f in folds:
            fold_dir  = results_dir / f"split{s}_fold{f}"
            ckpt_file = fold_dir / f"ckpts_{focus_tag}" / "best_model.pt"
            if not ckpt_file.exists():
                print(f"[singlemod] No checkpoint: {ckpt_file} — skip")
                continue
            split_col = f"split{s}_fold{f}"
            stem_to_datasplit = {
                Path(str(row["file"])).stem: str(row.get(split_col, ""))
                for _, row in df_csv.iterrows()
                if str(row.get(split_col, "")) in ("train", "val", "test")
            }
            stems = list(stem_to_datasplit.keys())
            print(f"[singlemod] s={s} f={f}: loading {len(stems)} bags ...")
            bag_cache = tv7.preload_bags(stems, str(samples_dir))
            try:
                model = _build_and_load(tv7, focus_tag, ckpt_file, device)
            except Exception as e:
                print(f"[singlemod] build/load failed: {e}")
                del bag_cache
                continue
            for mod in tv7.MODALITIES:
                rows = _run_inference_single_mod(
                    tv7, model, focus_tag, stems, bag_cache,
                    device, stem_to_meta, s, f, mod, stem_to_datasplit,
                )
                all_rows.extend(rows)
                print(f"[singlemod]   {mod}: {len(rows)} rows")
            del model, bag_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not all_rows:
        print("[singlemod] No rows collected.")
        return None
    df_sm = pd.DataFrame(all_rows)
    df_sm.to_csv(str(cache_csv), index=False)
    print(f"[singlemod] Saved {len(df_sm)} rows → {cache_csv}")
    return df_sm


# ─── display names for modalities ─────────────────────────────────────────────
_MOD_DISPLAY = {"HE": "H&E", "BAL": "BAL", "CT": "CT", "Clinical": "Clinical"}
_MOD_COMBO_KEY = {"HE": "HE", "BAL": "BAL", "CT": "CT", "Clinical": "Clin"}


def task_multimodal_analysis(
    results_dir: Path, splits_csv: Path, samples_dir: Path,
    output_dir: Path, exp: str, focus_variant: str,
    splits: List[int], folds: List[int],
    device_str: str, chicago_mil_dir: Optional[Path],
    min_combo_n: int = 10,
):
    """Comprehensive modality comparison:
    Full-multimodal P2 on different data subsets + single-modality mode
    inference (P2 model fed only one modality at a time). Inspired by v6
    task_modality_bacc + task_unimodal_p2, adapted for v7 with no P1 heads."""
    plt.rcParams.update(NATURE_RC)
    out = output_dir / "multimodal_analysis"
    out.mkdir(parents=True, exist_ok=True)

    prim_label = PRIMARY_METRIC.get(exp, "bacc")
    metric_label = "C-index" if prim_label == "c_index" else "Balanced Accuracy"

    # ── 1. Full-multimodal inference (cached) ─────────────────────────────────
    variant_data = _get_or_run_inference(
        results_dir, splits_csv, samples_dir, splits, folds,
        device_str, chicago_mil_dir, output_dir, exp,
    )
    if not variant_data:
        print("[multimodal] No variant data — skipping"); return

    tags_sorted = [t for t in VARIANT_TAGS if t in variant_data]
    tags_sorted += [t for t in sorted(variant_data.keys()) if t not in tags_sorted]
    focus_tag = focus_variant if focus_variant in variant_data else (tags_sorted[0] if tags_sorted else None)
    if focus_tag is None:
        print("[multimodal] No focus variant found — skipping"); return

    # v7 MODALITIES list (model keys)
    MODS = ["HE", "BAL", "CT", "Clinical"]

    # ── 2. Single-mod inference (focus_tag only) ──────────────────────────────
    df_sm = _get_or_run_singlemod(
        results_dir, splits_csv, samples_dir, splits, folds,
        device_str, chicago_mil_dir, output_dir, exp, focus_tag,
    )

    # ── 3. Compute metrics per modality × mode × subset ──────────────────────
    # Subsets defined by who has the target modality + unimodal/multimodal split
    df_focus = variant_data[focus_tag]["df"].copy()
    df_focus_test = df_focus[df_focus["data_split"] == "test"]

    SUBSET_DEFS = [
        ("P2 full — all test",        lambda df, ck: df),
        ("P2 full — has modality",     lambda df, ck: df[df["combo"].str.contains(ck, regex=False)]),
        ("P2 full — unimodal only",    lambda df, ck: df[(df["n_mods"] == 1) & df["combo"].str.contains(ck, regex=False)]),
        ("P2 full — multimodal",       lambda df, ck: df[(df["n_mods"] >= 2) & df["combo"].str.contains(ck, regex=False)]),
        ("P2 single-mod — has mod",    None),  # handled separately via df_sm
    ]
    SUBSET_COLORS = ["#4e79a7", "#59a14f", "#edc948", "#f28e2b", "#e15759"]

    rows_all = []
    for mod in MODS:
        combo_key = _MOD_COMBO_KEY[mod]
        for si, (sname, fn) in enumerate(SUBSET_DEFS):
            if fn is not None:
                sub = fn(df_focus_test, combo_key)
                val, n = _primary_metric_from_df(sub.assign(data_split="test"), exp)
            else:
                # single-mod subset
                if df_sm is not None:
                    sub_sm = df_sm[
                        (df_sm["target_mod"] == mod) &
                        (df_sm["data_split"] == "test")
                    ]
                    val, n = _primary_metric_from_df(sub_sm.assign(data_split="test"), exp)
                else:
                    val, n = float("nan"), 0
            rows_all.append({
                "modality": mod, "subset": sname,
                "metric": val, "n": n, "color": SUBSET_COLORS[si],
            })

    # Also collect per-variant full-test metrics for the comparison heatmap
    variant_fulltest_rows = []
    for tag in tags_sorted:
        df_v = variant_data[tag]["df"]
        df_vt = df_v[df_v["data_split"] == "test"] if "data_split" in df_v.columns else df_v
        for mod in MODS:
            combo_key = _MOD_COMBO_KEY[mod]
            for sname, fn in [
                ("All test",   lambda d, ck: d),
                ("Has mod",    lambda d, ck: d[d["combo"].str.contains(ck, regex=False)]),
                ("Unimodal",   lambda d, ck: d[(d["n_mods"] == 1) & d["combo"].str.contains(ck, regex=False)]),
                ("Multimodal", lambda d, ck: d[(d["n_mods"] >= 2) & d["combo"].str.contains(ck, regex=False)]),
            ]:
                sub = fn(df_vt, combo_key)
                val, n = _primary_metric_from_df(sub.assign(data_split="test"), exp)
                variant_fulltest_rows.append({
                    "variant": tag, "modality": mod, "subset": sname,
                    "metric": val, "n": n,
                })

    df_metrics = pd.DataFrame(rows_all)
    df_metrics.to_csv(out / "multimodal_metrics.csv", index=False)
    df_var_metrics = pd.DataFrame(variant_fulltest_rows)
    df_var_metrics.to_csv(out / "variant_modality_metrics.csv", index=False)
    print(f"[multimodal] CSV saved → {out}")

    # ── Figure 1: Per-modality grouped bar (5 bars × 4 modalities) ───────────
    subset_names = [s for s, _ in SUBSET_DEFS]
    n_mods   = len(MODS)
    n_subs   = len(subset_names)
    x        = np.arange(n_mods)
    bw       = 0.14
    offsets  = np.linspace(-(n_subs - 1) / 2 * bw, (n_subs - 1) / 2 * bw, n_subs)

    fig1, ax1 = plt.subplots(figsize=(TWO_COL_W * 1.2, 3.8))
    for si, (sname, color) in enumerate(zip(subset_names, SUBSET_COLORS)):
        vals, ns = [], []
        for mod in MODS:
            row = df_metrics[(df_metrics["modality"] == mod) & (df_metrics["subset"] == sname)]
            if len(row) > 0:
                vals.append(row.iloc[0]["metric"])
                ns.append(int(row.iloc[0]["n"]))
            else:
                vals.append(float("nan"))
                ns.append(0)
        bars = ax1.bar(x + offsets[si], vals, width=bw * 0.92,
                       color=color, alpha=0.85, label=sname, zorder=3)
        for bar, v, n in zip(bars, vals, ns):
            if not np.isfinite(v):
                continue
            ypos = v + 0.012
            if ypos > 0.93:
                ypos = v - 0.04
                fc = "white"
            else:
                fc = "#222"
            ax1.text(bar.get_x() + bar.get_width() / 2, ypos,
                     f"{v:.3f}\n(n={n})",
                     ha="center", va="bottom", fontsize=4.5, color=fc, zorder=5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([_MOD_DISPLAY.get(m, m) for m in MODS], fontsize=8)
    ax1.axhline(0.5, color="#999", lw=0.6, ls=":", zorder=2)
    y_lo = 0.35 if prim_label != "c_index" else 0.4
    ax1.set_ylim(y_lo, 1.05)
    ax1.set_ylabel(metric_label, fontsize=7)
    ax1.set_title(
        f"Multimodal vs unimodal mode comparison — {exp.upper()} task\n"
        f"(focus variant: {VARIANT_DISPLAY.get(focus_tag, focus_tag)})",
        fontsize=8, fontweight="bold",
    )
    ax1.legend(fontsize=5.5, ncol=2, loc="upper right", framealpha=0.9,
               handlelength=1.2, handleheight=0.8)
    ax1.grid(axis="y", alpha=0.25, zorder=0)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    # Weighted average summary at right
    wt_avg_rows = {}
    for sname in subset_names:
        ws, vs = [], []
        for mod in MODS:
            row = df_metrics[(df_metrics["modality"] == mod) & (df_metrics["subset"] == sname)]
            if len(row) > 0 and np.isfinite(row.iloc[0]["metric"]):
                ws.append(row.iloc[0]["n"])
                vs.append(row.iloc[0]["metric"])
        if ws:
            wt_avg_rows[sname] = float(np.average(vs, weights=ws))
        else:
            wt_avg_rows[sname] = float("nan")

    ax1_right = ax1.inset_axes([1.01, 0.0, 0.16, 1.0])
    for si, (sname, color) in enumerate(zip(subset_names, SUBSET_COLORS)):
        val = wt_avg_rows[sname]
        ax1_right.bar([si], [val], color=color, alpha=0.85, width=0.7, zorder=3)
        if np.isfinite(val):
            ax1_right.text(si, val + 0.008, f"{val:.3f}",
                           ha="center", va="bottom", fontsize=4.5, zorder=5)
    ax1_right.set_xticks(range(n_subs))
    ax1_right.set_xticklabels(
        [s.replace("P2 full — ", "").replace("P2 single-mod — ", "single\n")
         for s in subset_names],
        fontsize=4, rotation=60, ha="right",
    )
    ax1_right.set_ylim(y_lo, 1.05)
    ax1_right.axhline(0.5, color="#999", lw=0.6, ls=":", zorder=2)
    ax1_right.set_title("Wt avg", fontsize=5.5)
    ax1_right.grid(axis="y", alpha=0.25, zorder=0)
    for sp in ["top", "right"]:
        ax1_right.spines[sp].set_visible(False)

    fig1.suptitle("Modality comparison: P2 multimodal vs single-modality mode", fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig1, out / "fig_modality_comparison")
    plt.close(fig1)
    print(f"[multimodal] fig_modality_comparison → {out}")

    # ── Figure 2: Unimodal patients vs multimodal patients per modality ───────
    fig2, axes2 = plt.subplots(1, 2, figsize=(TWO_COL_W * 1.3, 3.5))

    for ax_idx, (sname_pair, title_str) in enumerate([
        (("P2 full — unimodal only", "P2 single-mod — has mod"),
         "Unimodal patients only (n_mods=1)"),
        (("P2 full — multimodal", "P2 single-mod — has mod"),
         "Multimodal patients (n_mods≥2)"),
    ]):
        ax = axes2[ax_idx]
        sname_full, sname_sm = sname_pair
        vals_full, vals_sm, labels_m, ns_full, ns_sm = [], [], [], [], []
        for mod in MODS:
            r_full = df_metrics[(df_metrics["modality"] == mod) & (df_metrics["subset"] == sname_full)]
            r_sm   = df_metrics[(df_metrics["modality"] == mod) & (df_metrics["subset"] == sname_sm)]
            vf = r_full.iloc[0]["metric"] if len(r_full) else float("nan")
            vs = r_sm.iloc[0]["metric"]   if len(r_sm)   else float("nan")
            nf = int(r_full.iloc[0]["n"]) if len(r_full) and np.isfinite(r_full.iloc[0]["n"]) else 0
            ns = int(r_sm.iloc[0]["n"])   if len(r_sm)   and np.isfinite(r_sm.iloc[0]["n"])   else 0
            vals_full.append(vf); vals_sm.append(vs)
            labels_m.append(_MOD_DISPLAY.get(mod, mod))
            ns_full.append(nf); ns_sm.append(ns)

        xi = np.arange(len(MODS))
        bw2 = 0.35
        b1 = ax.bar(xi - bw2 / 2, vals_full, width=bw2,
                    color="#4e79a7", alpha=0.85, label="P2 full model", zorder=3)
        b2 = ax.bar(xi + bw2 / 2, vals_sm, width=bw2,
                    color="#e15759", alpha=0.85, label="P2 single-mod", zorder=3)
        for bar, v, n in [(b, vv, nn) for b, vv, nn in zip(b1, vals_full, ns_full)] + \
                         [(b, vv, nn) for b, vv, nn in zip(b2, vals_sm, ns_sm)]:
            if not np.isfinite(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}\n(n={n})", ha="center", va="bottom", fontsize=4.5, zorder=5)
        ax.set_xticks(xi)
        ax.set_xticklabels(labels_m, fontsize=7)
        ax.set_ylim(y_lo, 1.05)
        ax.axhline(0.5, color="#999", lw=0.6, ls=":", zorder=2)
        ax.set_ylabel(metric_label if ax_idx == 0 else "", fontsize=7)
        ax.set_title(title_str, fontsize=7, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.9)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    fig2.suptitle(
        f"Full multimodal vs single-mod mode: unimodal/multimodal patients\n"
        f"({exp.upper()}, {VARIANT_DISPLAY.get(focus_tag, focus_tag)})",
        fontsize=8, fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig2, out / "fig_unimodal_multimodal_breakdown")
    plt.close(fig2)
    print(f"[multimodal] fig_unimodal_multimodal_breakdown → {out}")

    # ── Figure 3: All-variant heatmap (full-model, per modality subset) ───────
    sub_order = ["All test", "Has mod", "Unimodal", "Multimodal"]
    n_var = len(tags_sorted)
    n_sub3 = len(sub_order)
    n_mod3 = len(MODS)

    fig3, axes3 = plt.subplots(1, n_mod3,
                                figsize=(TWO_COL_W * 0.55 * n_mod3, 3.5))
    if n_mod3 == 1:
        axes3 = [axes3]

    for mi, mod in enumerate(MODS):
        ax = axes3[mi]
        heat = np.full((n_sub3, n_var), float("nan"))
        for si, sname in enumerate(sub_order):
            for ti, tag in enumerate(tags_sorted):
                row = df_var_metrics[
                    (df_var_metrics["variant"] == tag) &
                    (df_var_metrics["modality"] == mod) &
                    (df_var_metrics["subset"] == sname)
                ]
                if len(row):
                    heat[si, ti] = row.iloc[0]["metric"]
        vmin = 0.35 if prim_label != "c_index" else 0.4
        im = ax.imshow(heat, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=0.95)
        ax.set_xticks(range(n_var))
        ax.set_xticklabels(
            [VARIANT_DISPLAY.get(t, t) for t in tags_sorted],
            fontsize=5.5, rotation=40, ha="right",
        )
        ax.set_yticks(range(n_sub3))
        ax.set_yticklabels(sub_order, fontsize=6) if mi == 0 else ax.set_yticks([])
        for si in range(n_sub3):
            for ti in range(n_var):
                v = heat[si, ti]
                if np.isfinite(v):
                    tc = "white" if (v < 0.47 or v > 0.85) else "black"
                    ax.text(ti, si, f"{v:.2f}", ha="center", va="center",
                            fontsize=4.5, color=tc)
        ax.set_title(_MOD_DISPLAY.get(mod, mod), fontsize=7.5, fontweight="bold")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03, shrink=0.8,
                     label=metric_label if mi == n_mod3 - 1 else "")

    fig3.suptitle(
        f"Per-modality performance heatmap — all variants, {exp.upper()}\n"
        f"(full multimodal model; metric = {metric_label})",
        fontsize=8, fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig3, out / "fig_variant_modality_heatmap")
    plt.close(fig3)
    print(f"[multimodal] fig_variant_modality_heatmap → {out}")

    print(f"\n[multimodal] Done. Outputs in {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified comprehensive analysis suite for multimodal ABMIL v7."
    )
    p.add_argument("--tasks", nargs="+",
                   choices=["benchmark", "combo_dist", "umap", "combo_perf",
                             "multimodal_analysis", "all"],
                   default=["all"],
                   help="Which analysis tasks to run.")
    p.add_argument("--exp", type=str, default="alt",
                   choices=["cls", "surv", "alt"],
                   help="Which experiment to use for inference tasks.")
    p.add_argument("--variant", type=str, default="middle",
                   help="Focus variant for deep-dive analyses.")
    p.add_argument("--results_base", type=str,
                   default="/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil",
                   help="Base dir containing results_mm_abmil_v7_* subdirs.")
    p.add_argument("--splits_csv", type=str,
                   default="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
    p.add_argument("--samples_dir", type=str,
                   default="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples")
    p.add_argument("--output_dir", type=str,
                   default="/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/analysis/v7_full")
    p.add_argument("--chicago_mil_dir", type=str,
                   default="/home/aih/dinesh.haridoss/chicago_mil",
                   help="Path to dir containing train_mm_abmil_v7.py.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--umap_n_neighbors", type=int, default=30)
    p.add_argument("--umap_min_dist", type=float, default=0.3)
    p.add_argument("--min_combo_n", type=int, default=10,
                   help="Minimum samples for a combo to appear in combo plots.")
    return p.parse_args()


def main():
    args = parse_args()

    results_base    = Path(args.results_base)
    splits_csv      = Path(args.splits_csv)
    samples_dir     = Path(args.samples_dir)
    output_dir      = Path(args.output_dir)
    chicago_mil_dir = Path(args.chicago_mil_dir) if args.chicago_mil_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    # results dir for selected experiment
    results_dir = results_base / f"results_mm_abmil_v7_{args.exp}_p1v3"

    run_all = "all" in args.tasks
    tasks   = set(args.tasks)

    print("=" * 80)
    print("  Multimodal ABMIL v7 — Comprehensive Analysis Suite")
    print(f"  Tasks:   {args.tasks}")
    print(f"  Exp:     {args.exp}")
    print(f"  Variant: {args.variant}")
    print(f"  Output:  {output_dir}")
    print("=" * 80)

    # ── TASK 1: benchmark ────────────────────────────────────────────────────
    if run_all or "benchmark" in tasks:
        print("\n── TASK 1: benchmark ─────────────────────────────────────────────")
        task_benchmark(results_base, output_dir, args.folds)

    # ── TASK 2: combo_dist ───────────────────────────────────────────────────
    if run_all or "combo_dist" in tasks:
        print("\n── TASK 2: combo_dist ────────────────────────────────────────────")
        task_combo_dist(splits_csv, output_dir)

    # ── TASK 3: umap ─────────────────────────────────────────────────────────
    if run_all or "umap" in tasks:
        print("\n── TASK 3: umap ──────────────────────────────────────────────────")
        if not HAS_UMAP:
            print("[umap] umap-learn not installed — skipping.")
        else:
            task_umap(
                results_dir=results_dir,
                splits_csv=splits_csv,
                samples_dir=samples_dir,
                output_dir=output_dir,
                exp=args.exp,
                splits=[0],
                folds=args.folds,
                device_str=args.device,
                umap_n_neighbors=args.umap_n_neighbors,
                umap_min_dist=args.umap_min_dist,
                chicago_mil_dir=chicago_mil_dir,
            )

    # ── TASK 4: combo_perf ───────────────────────────────────────────────────
    if run_all or "combo_perf" in tasks:
        print("\n── TASK 4: combo_perf ────────────────────────────────────────────")
        task_combo_perf(
            results_dir=results_dir,
            splits_csv=splits_csv,
            samples_dir=samples_dir,
            output_dir=output_dir,
            exp=args.exp,
            focus_variant=args.variant,
            splits=[0],
            folds=args.folds,
            device_str=args.device,
            chicago_mil_dir=chicago_mil_dir,
            min_combo_n=args.min_combo_n,
        )

    # ── TASK 5: multimodal_analysis ──────────────────────────────────────────
    if run_all or "multimodal_analysis" in tasks:
        print("\n── TASK 5: multimodal_analysis ───────────────────────────────────")
        task_multimodal_analysis(
            results_dir=results_dir,
            splits_csv=splits_csv,
            samples_dir=samples_dir,
            output_dir=output_dir,
            exp=args.exp,
            focus_variant=args.variant,
            splits=[0],
            folds=args.folds,
            device_str=args.device,
            chicago_mil_dir=chicago_mil_dir,
            min_combo_n=args.min_combo_n,
        )

    print(f"\n{'=' * 80}")
    print(f"  All done. Outputs in: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
