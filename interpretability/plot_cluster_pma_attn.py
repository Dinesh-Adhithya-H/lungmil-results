"""
Effective HE instance cluster attention for LongMIL — bar chart per task.

Reads cluster_aff_data_*.json (from interpret_longitudinal_mk.py).
That file stores mean_aff (K, C) where each cell [k, c] =
  alpha_k × pma_attn_mass[k → cluster c]
i.e. the ABMIL-weighted PMA attention flowing from seed k to cluster c.

Summing over seeds k gives effective cluster attention (C,) — the total
attention the model allocates to each HE instance cluster.
hi and lo (high/low risk groups) are averaged to get an overall score.

Clusters are colored by biological category and sorted by attention descending.
Output: figures/interpretability/cluster_pma_attn/
Usage: sbatch interpretability/submit_cluster_pma_attn.sh
"""

import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from interpretability.shared import HE_BIO_MAP, HE_BIO_COLORS

INTERP_ROOT = ROOT / "interpretability" / "longitudinal_mk_interp"
OUT_ROOT    = ROOT / "figures" / "interpretability" / "cluster_pma_attn"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

VARIANT = "longitudinal_mk_no_alibi_v2"

TASK_CFG = {
    "acr_cls":   {"dir_suffix": "acr_cls",  "json_key": "acr_cls",  "label": "ACR Classification"},
    "acr_surv":  {"dir_suffix": "acr_surv", "json_key": "acr_surv", "label": "ACR Survival"},
    "clad_surv": {"dir_suffix": "clad_surv","json_key": "clad",     "label": "CLAD Survival"},
    "death_surv":{"dir_suffix": "death_surv","json_key": "death",   "label": "Death Survival"},
}

TASK_COLORS = {
    "acr_cls":   "#e53935",
    "acr_surv":  "#ff7043",
    "clad_surv": "#7e57c2",
    "death_surv":"#26a69a",
}

BG = "#FAF6F2"

# Biological category order (for consistent coloring)
BIO_ORDER = list(HE_BIO_COLORS.keys())


def load_cluster_attn(task_key, cfg):
    """
    Returns: mean_attn (C,), std_attn (C,), cluster_names (C,)
    mean_attn[c] = mean PMA attention flowing to cluster c,
                   averaged over seeds and over all patients (hi+lo)/2, over splits.
    """
    split_means = []
    cluster_names = None

    for split in range(5):
        p = (INTERP_ROOT / f"{VARIANT}_split{split}_fold0_{cfg['dir_suffix']}"
             / f"cluster_aff_data_{cfg['json_key']}.json")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        caff = d.get("cluster_aff", {}).get("HE")
        if caff is None:
            continue

        hi = np.array(caff["hi"])   # (K, C)  — each cell: alpha_k * pma_attn[k→c] for hi-risk
        lo = np.array(caff["lo"])   # (K, C)  — same for lo-risk

        # Sum over seeds → (C,): total ABMIL-weighted cluster attention
        # Average hi and lo to get overall (not split by risk group)
        cluster_attn = ((hi + lo) / 2).sum(axis=0)   # (C,)
        split_means.append(cluster_attn)

        if cluster_names is None:
            cluster_names = caff["names"]

    if not split_means:
        return None, None, None

    arr = np.stack(split_means)                    # (n_splits, C)
    return arr.mean(0), arr.std(0), cluster_names


def bio_cat(name):
    return HE_BIO_MAP.get(name, "Unknown")


def bio_color(name):
    return HE_BIO_COLORS.get(bio_cat(name), "#b0bec5")


def plot_all_tasks():
    """4-panel figure: one bar chart per task, sorted by mean attn descending."""
    fig, axes = plt.subplots(2, 2, figsize=(22, 14), facecolor=BG)
    axes = axes.flatten()
    fig.suptitle("Effective HE Instance Cluster Attention — LongMIL\n"
                 "ABMIL-weighted PMA attention summed to cluster level "
                 "(Σ_k α_k · pma[k→c], avg over patients and 5 splits)",
                 fontsize=13, fontweight="bold")

    for ax, (task_key, cfg) in zip(axes, TASK_CFG.items()):
        mean_attn, std_attn, names = load_cluster_attn(task_key, cfg)
        if mean_attn is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cfg["label"]); continue

        # Sort by mean attention descending
        order = np.argsort(mean_attn)[::-1]
        names_s    = [names[i] for i in order]
        attn_s     = mean_attn[order]
        std_s      = std_attn[order]
        colors_s   = [bio_color(n) for n in names_s]

        # Sort by bio category first, then by attention within category
        BIO_ORDER_LIST = [
            "Alveolar with hemorrhage and inflammation",
            "Alveolar with empty spaces", "Alveolar",
            "Bronchial", "Lymphocytoplasmic inflammation",
            "Cartilage", "Unknown",
        ]
        SHORT_BIO = {
            "Alveolar with hemorrhage and inflammation": "Alv+Inflam",
            "Alveolar with empty spaces": "Alv+Empty",
            "Alveolar": "Alveolar", "Bronchial": "Bronchial",
            "Lymphocytoplasmic inflammation": "Lymphocytic",
            "Cartilage": "Cartilage", "Unknown": "Unknown",
        }
        cats_s = [bio_cat(n) for n in names_s]

        # Re-sort: by bio category rank, then by attention descending within
        def sort_key_bio(i):
            cat = cats_s[i]
            cr = BIO_ORDER_LIST.index(cat) if cat in BIO_ORDER_LIST else len(BIO_ORDER_LIST)
            return (cr, -attn_s[i])
        idx2 = sorted(range(len(names_s)), key=sort_key_bio)
        names_s  = [names_s[i]  for i in idx2]
        attn_s   = attn_s[idx2]
        std_s    = std_s[idx2]
        colors_s = [colors_s[i] for i in idx2]
        cats_s   = [cats_s[i]   for i in idx2]

        x = np.arange(len(names_s))
        ax.bar(x, attn_s, color=colors_s, alpha=0.85, width=0.8, zorder=3)
        ax.errorbar(x, attn_s, yerr=std_s, fmt="none",
                    color="#333", capsize=2, lw=0.8, alpha=0.6, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(names_s, rotation=75, ha="right", fontsize=5.5)
        ax.set_ylabel("Effective cluster attention (Σ_k α_k·pma[k→c])", fontsize=9)
        ax.set_title(cfg["label"], fontsize=12, fontweight="bold",
                     color=TASK_COLORS.get(task_key, "#333"))
        ax.set_facecolor(BG)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2, lw=0.5)

        # Bio category dividers + top labels
        prev_cat, cat_start = None, 0
        ylim_top = ax.get_ylim()[1]
        for xi, cat in enumerate(cats_s):
            if cat != prev_cat:
                if prev_cat is not None:
                    ax.axvline(xi - 0.5, color="#aaa", lw=0.8, ls="--", alpha=0.5)
                    mid = (cat_start + xi - 1) / 2
                    ax.text(mid, ylim_top * 0.98, SHORT_BIO.get(prev_cat, prev_cat),
                            ha="center", va="top", fontsize=6,
                            color=HE_BIO_COLORS.get(prev_cat, "#888"), fontweight="bold")
                cat_start = xi; prev_cat = cat
        if prev_cat:
            mid = (cat_start + len(cats_s) - 1) / 2
            ax.text(mid, ylim_top * 0.98, SHORT_BIO.get(prev_cat, prev_cat),
                    ha="center", va="top", fontsize=6,
                    color=HE_BIO_COLORS.get(prev_cat, "#888"), fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(OUT_ROOT / f"cluster_pma_attn_all_tasks.{ext}",
                    dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Saved cluster_pma_attn_all_tasks", flush=True)


def plot_bio_aggregated():
    """
    Aggregate clusters to biological category level.
    One bar per bio category, 4 tasks overlaid or grouped.
    """
    # Collect per-bio-category mean attn per task
    task_bio_attn = {}
    all_bio_cats = set()

    for task_key, cfg in TASK_CFG.items():
        mean_attn, std_attn, names = load_cluster_attn(task_key, cfg)
        if mean_attn is None:
            continue
        bio_sum = defaultdict(float)
        for i, n in enumerate(names):
            cat = bio_cat(n)
            bio_sum[cat] += mean_attn[i]   # sum cluster attentions within category
        task_bio_attn[task_key] = dict(bio_sum)
        all_bio_cats.update(bio_sum.keys())

    # Sort bio cats by mean attn across all tasks
    bio_cats = sorted(all_bio_cats,
                      key=lambda c: np.mean([task_bio_attn[t].get(c, 0)
                                             for t in task_bio_attn]),
                      reverse=True)

    n_tasks = len(task_bio_attn)
    n_cats  = len(bio_cats)
    width   = 0.18
    x       = np.arange(n_cats)

    fig, ax = plt.subplots(figsize=(max(10, n_cats * 1.2 + 2), 5), facecolor=BG)
    fig.suptitle("PMA Attention by Biological Category — LongMIL\n"
                 "Mean cluster attention aggregated to tissue category, per task",
                 fontsize=12, fontweight="bold")

    for ti, (task_key, bio_attn) in enumerate(task_bio_attn.items()):
        offset = (ti - n_tasks / 2 + 0.5) * width
        vals = [bio_attn.get(c, 0) for c in bio_cats]
        ax.bar(x + offset, vals, width=width * 0.9,
               color=TASK_COLORS.get(task_key, "#888"), alpha=0.82,
               label=TASK_CFG[task_key]["label"])

    ax.set_xticks(x)
    ax.set_xticklabels(bio_cats, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Total cluster attention per bio category (Σ clusters)", fontsize=9)
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, framealpha=0.7)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("png", "pdf"):
        fig.savefig(OUT_ROOT / f"cluster_pma_attn_bio_agg.{ext}",
                    dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Saved cluster_pma_attn_bio_agg", flush=True)


def plot_per_task_separate():
    """One figure per task — larger, easier to read individual cluster bars."""
    for task_key, cfg in TASK_CFG.items():
        mean_attn, std_attn, names = load_cluster_attn(task_key, cfg)
        if mean_attn is None:
            continue

        BIO_ORDER_LIST2 = [
            "Alveolar with hemorrhage and inflammation",
            "Alveolar with empty spaces", "Alveolar",
            "Bronchial", "Lymphocytoplasmic inflammation",
            "Cartilage", "Unknown",
        ]
        all_cats = [bio_cat(n) for n in names]
        def sort_key2(i):
            cat = all_cats[i]
            cr = BIO_ORDER_LIST2.index(cat) if cat in BIO_ORDER_LIST2 else len(BIO_ORDER_LIST2)
            return (cr, -mean_attn[i])
        order    = sorted(range(len(names)), key=sort_key2)
        names_s  = [names[i]    for i in order]
        attn_s   = mean_attn[np.array(order)]
        std_s    = std_attn[np.array(order)]
        colors_s = [bio_color(n) for n in names_s]
        cats_s   = [bio_cat(n)   for n in names_s]

        x = np.arange(len(names_s))
        fig, ax = plt.subplots(figsize=(max(14, len(names_s) * 0.28), 5), facecolor=BG)
        bars = ax.bar(x, attn_s, color=colors_s, alpha=0.85, width=0.75, zorder=3)
        ax.errorbar(x, attn_s, yerr=std_s, fmt="none",
                    color="#555", capsize=3, lw=1.0, alpha=0.7, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(names_s, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("Effective cluster attention Σ_k α_k·pma[k→c] (5-split avg ± std)", fontsize=10)
        ax.set_title(f"PMA Attention per HE Instance Cluster — {cfg['label']}",
                     fontsize=12, fontweight="bold",
                     color=TASK_COLORS.get(task_key, "#333"))
        ax.set_facecolor(BG)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, lw=0.5)

        # Bio category label above each bar
        for xi, (n, cat) in enumerate(zip(names_s, cats_s)):
            short = cat.split()[0][:4]  # e.g. "Alve", "Bron", "Lymp"
            ax.text(xi, attn_s[xi] + std_s[xi] * 0.5, short,
                    ha="center", va="bottom", fontsize=4, rotation=90,
                    color=bio_color(n), alpha=0.8)

        # Legend
        seen = {}
        for n, c in zip(names_s, colors_s):
            cat = bio_cat(n)
            if cat not in seen:
                seen[cat] = c
        patches = [mpatches.Patch(color=c, label=b) for b, c in seen.items()]
        ax.legend(handles=patches, fontsize=7, loc="upper right",
                  framealpha=0.7, title="Biological category", title_fontsize=8)

        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT_ROOT / f"cluster_pma_attn_{task_key}.{ext}",
                        dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"Saved cluster_pma_attn_{task_key}", flush=True)


def main():
    print("Plotting cluster PMA attention histograms...", flush=True)
    plot_all_tasks()
    plot_bio_aggregated()
    plot_per_task_separate()
    print(f"\nDone. → {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
