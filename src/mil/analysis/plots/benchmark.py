"""
Benchmark figures — all tasks, single vs multitask comparison.

Outputs in output_dir/benchmark/:
  fig_benchmark_all.pdf          — 6-panel bar chart (one per task)
  fig_benchmark_heatmap.pdf      — task × variant heatmap
  fig_benchmark_single_vs_multi.pdf — paired comparison: single vs multitask
  benchmark_summary.csv
"""
import math
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import NATURE_RC, TASKS, VARIANT_DISPLAY, VARIANT_TAGS, TWO_COL_W
from ..io import load_metrics_dir, fold_stats, ordered_tags


def _save(fig, path_stem: Path):
    fig.savefig(str(path_stem) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(path_stem) + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def task_benchmark(results_base: Path, output_dir: Path, folds: List[int],
                   task_keys: List[str] = None):
    plt.rcParams.update(NATURE_RC)
    out = output_dir / "benchmark"
    out.mkdir(parents=True, exist_ok=True)

    active_tasks = {k: v for k, v in TASKS.items()
                    if task_keys is None or k in task_keys}

    # Load requested tasks
    exp_data: Dict[str, Dict] = {}
    for task_key, (dir_suf, prim, label, color, _) in active_tasks.items():
        rd = results_base / dir_suf
        d  = load_metrics_dir(rd, folds)
        exp_data[task_key] = d
        tags = ordered_tags(d)
        if tags:
            mu, sd = fold_stats(d, tags[0], prim)
            print(f"[benchmark] {label:25s}: {len(d)} variants  "
                  f"best {prim}={mu:.3f}±{sd:.3f} ({tags[0]})")

    # ── Figure 1: bar chart (one panel per active task) ───────────────────
    n_tasks = len(active_tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(TWO_COL_W * max(1, n_tasks / 5) * 1.8, 4.5),
                             sharey=False, squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor("white")

    for ax, (task_key, (dir_suf, prim, label, color, _)) in zip(axes, active_tasks.items()):
        data  = exp_data[task_key]
        tags  = ordered_tags(data)
        if not tags:
            ax.set_title(label, fontsize=7, fontweight="bold", color=color); continue

        mus  = [fold_stats(data, t, prim)[0] for t in tags]
        sds  = [fold_stats(data, t, prim)[1] for t in tags]
        x    = np.arange(len(tags))

        ax.bar(x, mus, yerr=sds, capsize=3, color=color, alpha=0.82, width=0.6,
               edgecolor="white", lw=0.5, error_kw={"elinewidth": 1.0, "ecolor": "#555"})
        ax.axhline(0.5, color="#aaa", lw=0.8, ls=":", zorder=0)

        # Highlight best
        best_i = int(np.nanargmax(mus)) if not all(np.isnan(v) for v in mus) else -1
        if best_i >= 0 and not np.isnan(mus[best_i]):
            ax.bar(x[best_i], mus[best_i], yerr=sds[best_i], capsize=3,
                   color=color, alpha=1.0, width=0.6, edgecolor="#333", lw=1.2,
                   error_kw={"elinewidth": 1.0, "ecolor": "#333"})

        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_DISPLAY.get(t, t) for t in tags],
                           rotation=45, ha="right", fontsize=6)
        metric_label = "BAcc" if prim == "bacc" else "C-index"
        ax.set_ylabel(metric_label, fontsize=7)
        ax.set_title(label, fontsize=7, fontweight="bold", color=color)

        ax.set_ylim(0.0, 1.0)   # fixed 0-1 range for all metrics

        for xi, (mu, sd) in enumerate(zip(mus, sds)):
            if not np.isnan(mu):
                ax.text(xi, mu + (sd if not np.isnan(sd) else 0) + 0.01,
                        f"{mu:.3f}", ha="center", va="bottom", fontsize=5.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.5, alpha=0.4, ls="--")

    fig.suptitle("All tasks — primary metric per task (mean ± std, test set)\n"
                 "ACR Cls: BAcc  |  ACR TTE / CLAD / Death: C-index",
                 fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "fig_benchmark_all")
    print(f"  Saved: {out}/fig_benchmark_all.pdf")

    # ── Figure 2: heatmap task × variant ─────────────────────────────────
    all_tags = []
    for tk in active_tasks:
        for t in ordered_tags(exp_data[tk]):
            if t not in all_tags:
                all_tags.append(t)
    all_tags = [t for t in VARIANT_TAGS if t in all_tags] + \
               [t for t in all_tags if t not in VARIANT_TAGS]

    active_keys   = list(active_tasks.keys())
    task_labels = [active_tasks[k][2] for k in active_keys]
    mat    = np.full((len(active_keys), len(all_tags)), float("nan"))
    mat_sd = np.full_like(mat, float("nan"))
    for ei, tk in enumerate(active_keys):
        prim = active_tasks[tk][1]
        for ti, tag in enumerate(all_tags):
            mu, sd = fold_stats(exp_data[tk], tag, prim)
            mat[ei, ti] = mu; mat_sd[ei, ti] = sd

    fig2, ax2 = plt.subplots(figsize=(max(TWO_COL_W, len(all_tags) * 0.9 + 1.5),
                                      len(active_keys) * 0.6 + 1.5))
    fig2.patch.set_facecolor("white")
    vmin_h = float(np.nanmin(mat)); vmax_h = float(np.nanmax(mat))
    im = ax2.imshow(mat, cmap="RdYlGn", vmin=max(0.45, vmin_h),
                    vmax=min(1.0, vmax_h), aspect="auto")
    plt.colorbar(im, ax=ax2, label="BAcc / C-index", fraction=0.03, pad=0.02)
    ax2.set_xticks(range(len(all_tags)))
    ax2.set_xticklabels([VARIANT_DISPLAY.get(t, t) for t in all_tags],
                        rotation=40, ha="right", fontsize=8)
    ax2.set_yticks(range(len(active_keys)))
    ax2.set_yticklabels(task_labels, fontsize=8)
    for ei in range(len(active_keys)):
        for ti in range(len(all_tags)):
            mu = mat[ei, ti]; sd = mat_sd[ei, ti]
            if not np.isnan(mu):
                txt = f"{mu:.3f}" if np.isnan(sd) else f"{mu:.3f}\n±{sd:.3f}"
                ax2.text(ti, ei, txt, ha="center", va="center", fontsize=6,
                         color="white" if (mu < 0.52 or mu > 0.82) else "black")
    ax2.set_title("All tasks × variants — primary metric heatmap", fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save(fig2, out / "fig_benchmark_heatmap")
    print(f"  Saved: {out}/fig_benchmark_heatmap.pdf")

    # ── Figure 3: single vs multitask paired comparison (v7 only) ────────────
    pairs = [
        ("acr_cls", "acr_alt_cls", "ACR Cls: single vs multitask", "bacc",    "BAcc"),
        ("acr_tte", "acr_alt_tte", "ACR TTE: single vs multitask", "c_index", "C-index"),
    ]
    active_pairs = [(s, m, t, met, yl) for s, m, t, met, yl in pairs
                    if s in exp_data and m in exp_data]
    if active_pairs:
        fig3, axes3 = plt.subplots(1, len(active_pairs), figsize=(TWO_COL_W * 1.2, 4.5),
                                   squeeze=False)
        axes3 = axes3[0]
        fig3.patch.set_facecolor("white")
        for ax, (single_k, multi_k, ttl, metric, ylabel) in zip(axes3, active_pairs):
            d_s = exp_data[single_k]; d_m = exp_data[multi_k]
            tags_s = ordered_tags(d_s); tags_m = ordered_tags(d_m)
            all_t = list(dict.fromkeys(tags_s + tags_m))
            x = np.arange(len(all_t))
            w = 0.35
            mus_s = [fold_stats(d_s, t, metric)[0] for t in all_t]
            sds_s = [fold_stats(d_s, t, metric)[1] for t in all_t]
            mus_m = [fold_stats(d_m, t, metric)[0] for t in all_t]
            sds_m = [fold_stats(d_m, t, metric)[1] for t in all_t]
            ax.bar(x - w/2, mus_s, yerr=sds_s, width=w, capsize=2,
                   color=TASKS[single_k][3], alpha=0.85, label="Single",
                   error_kw={"elinewidth": 1.0, "ecolor": "#555"})
            ax.bar(x + w/2, mus_m, yerr=sds_m, width=w, capsize=2,
                   color=TASKS[multi_k][3], alpha=0.85, label="Multitask",
                   error_kw={"elinewidth": 1.0, "ecolor": "#555"})
            ax.axhline(0.5, color="#aaa", lw=0.8, ls=":")
            ax.set_xticks(x)
            ax.set_xticklabels([VARIANT_DISPLAY.get(t, t) for t in all_t],
                               rotation=45, ha="right", fontsize=6.5)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(ttl, fontsize=8, fontweight="bold")
            ax.legend(fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", lw=0.5, alpha=0.4, ls="--")
        fig3.suptitle("Single-task vs Multitask (both_alt) — mean ± std, test set",
                      fontsize=9, fontweight="bold")
        plt.tight_layout()
        _save(fig3, out / "fig_benchmark_single_vs_multi")
        print(f"  Saved: {out}/fig_benchmark_single_vs_multi.pdf")

    # ── CSV summary ────────────────────────────────────────────────────────
    rows = []
    for task_key, (dir_suf, prim, label, color, _) in active_tasks.items():
        data = exp_data[task_key]
        for tag in ordered_tags(data):
            row = {"task": label, "variant": VARIANT_DISPLAY.get(tag, tag)}
            for metric in ["auc", "bacc", "mcc", "c_index"]:
                mu, sd = fold_stats(data, tag, metric)
                row[f"{metric}_mean"] = round(mu, 4) if not np.isnan(mu) else None
                row[f"{metric}_std"]  = round(sd, 4) if not np.isnan(sd) else None
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "benchmark_summary.csv", index=False)
    print(f"  CSV: {out}/benchmark_summary.csv")
