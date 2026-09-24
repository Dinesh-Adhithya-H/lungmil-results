"""
LongMIL cluster attention across ALL modalities — for decision making.

For each task × modality: which instance clusters does LongMIL attend to?
Uses cluster_aff_data JSONs: (hi + lo).sum(axis=0) = total PMA-weighted attention
across all patients, summed over K seeds → per-cluster total.
Normalized to relative attention (1.0 = uniform baseline).

Output: figures/interpretability/multimodal_cluster_attn/
  - one PNG/PDF per task (4 modalities as subplots)
  - combined all-tasks figure
Usage: sbatch interpretability/submit_multimodal_cluster_attn.sh
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from interpretability.shared import HE_BIO_MAP, HE_BIO_COLORS

INTERP_ROOT = ROOT / "interpretability" / "longitudinal_mk_interp"
OUT_ROOT    = ROOT / "figures" / "interpretability" / "multimodal_cluster_attn"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

VARIANT = "longitudinal_mk_no_alibi_v2"
BG      = "#FAF6F2"

TASK_CFG = {
    "acr_cls":   {"dir_suffix": "acr_cls",   "json_key": "acr_cls",  "label": "ACR Classification", "color": "#e53935"},
    "acr_surv":  {"dir_suffix": "acr_surv",  "json_key": "acr_surv", "label": "ACR Survival",       "color": "#ff7043"},
    "clad_surv": {"dir_suffix": "clad_surv", "json_key": "clad",     "label": "CLAD Survival",      "color": "#7e57c2"},
    "death_surv":{"dir_suffix": "death_surv","json_key": "death",    "label": "Death Survival",     "color": "#26a69a"},
}

MOD_COLORS = {"HE": "#4C9BE8", "BAL": "#3FB950", "CT": "#D4A017", "Clinical": "#A97ECC"}
MOD_ORDER  = ["HE", "BAL", "CT", "Clinical"]

# HE biological categories
HE_BIO_ORDER = [
    "Alveolar with hemorrhage and inflammation",
    "Alveolar with empty spaces",
    "Alveolar",
    "Bronchial",
    "Lymphocytoplasmic inflammation",
    "Cartilage",
    "Unknown",
]
HE_SHORT = {
    "Alveolar with hemorrhage and inflammation": "Alv+Inflam",
    "Alveolar with empty spaces":               "Alv+Empty",
    "Alveolar":                                 "Alveolar",
    "Bronchial":                                "Bronchial",
    "Lymphocytoplasmic inflammation":           "Lymphocytic",
    "Cartilage":                                "Cartilage",
    "Unknown":                                  "Unknown",
}

# Top-N clusters to show per modality (avoids 106-bar Clinical plot)
TOP_N = {"HE": 54, "BAL": 43, "CT": 37, "Clinical": 30}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
})


def load_cluster_aff(task_key, cfg):
    """Load cluster_aff from all 5 splits. Returns {mod: {name: [vals_per_split]}}."""
    mod_cluster_vals = defaultdict(lambda: defaultdict(list))
    for split in range(5):
        p = (INTERP_ROOT / f"{VARIANT}_split{split}_fold0_{cfg['dir_suffix']}"
             / f"cluster_aff_data_{cfg['json_key']}.json")
        if not p.exists():
            print(f"  missing: {p}", flush=True); continue
        d  = json.loads(p.read_text())
        ca = d.get("cluster_aff", {})
        for mod, v in ca.items():
            names = v.get("names", [])
            hi    = np.array(v.get("hi", []))   # (K, C)
            lo    = np.array(v.get("lo", []))   # (K, C)
            if hi.size == 0 or not names: continue
            total = (hi + lo).sum(axis=0)       # (C,) raw combined attention
            for ci, name in enumerate(names):
                mod_cluster_vals[mod][name].append(float(total[ci]))
    return mod_cluster_vals


def rel_attn(cluster_vals):
    """Normalize to relative attention: 1.0 = uniform. Returns {name: (mean, sem)}."""
    all_vals = [v for vals in cluster_vals.values() for v in vals]
    overall  = np.mean(all_vals) if all_vals else 1.0
    return {n: (np.mean(v) / overall,
                np.std(v) / overall / np.sqrt(max(len(v), 1)))
            for n, v in cluster_vals.items()}


def he_sort_key(name, rel_mean):
    cat = HE_BIO_MAP.get(name, "Unknown")
    return (HE_BIO_ORDER.index(cat) if cat in HE_BIO_ORDER else len(HE_BIO_ORDER), -rel_mean)


def plot_mod_ax(ax, mod, rel, task_color, top_n):
    """Horizontal bar plot for one modality on given axis."""
    if not rel:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes); return

    names  = list(rel.keys())
    means  = np.array([rel[n][0] for n in names])
    sems   = np.array([rel[n][1] for n in names])

    # Sort: HE by bio category then attention; others by attention descending
    if mod == "HE":
        order = sorted(range(len(names)), key=lambda i: he_sort_key(names[i], means[i]))
    else:
        order = sorted(range(len(names)), key=lambda i: -means[i])

    order   = order[:top_n]
    names_s = [names[i] for i in order]
    means_s = means[order]
    sems_s  = sems[order]

    # Colors
    if mod == "HE":
        cats_s   = [HE_BIO_MAP.get(n, "Unknown") for n in names_s]
        colors_s = [HE_BIO_COLORS.get(c, "#b0bec5") for c in cats_s]
    else:
        colors_s = [MOD_COLORS[mod]] * len(names_s)

    y = np.arange(len(names_s))
    ax.barh(y, means_s, color=colors_s, alpha=0.88, height=0.72, zorder=3)
    ax.errorbar(means_s, y, xerr=sems_s, fmt="none", color="#555",
                lw=0.6, alpha=0.4, zorder=4)
    ax.axvline(1.0, color="#555", lw=0.8, ls="--", alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names_s, fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel("Relative attention (1.0 = uniform)", fontsize=7)
    ax.set_title(mod, fontsize=10, fontweight="bold",
                 color=MOD_COLORS.get(mod, "#333"), pad=4)
    ax.set_facecolor(BG)
    ax.grid(axis="x", alpha=0.15, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # HE: bio category labels on right
    if mod == "HE":
        prev_cat, cat_start = None, 0
        for yi, cat in enumerate(cats_s):
            if cat != prev_cat:
                if prev_cat is not None:
                    ax.axhline(yi - 0.5, color="#bbb", lw=0.7, alpha=0.6)
                    mid = (cat_start + yi - 1) / 2
                    ax.text(ax.get_xlim()[1], mid,
                            f"  {HE_SHORT.get(prev_cat, prev_cat)}",
                            ha="left", va="center", fontsize=5.5,
                            color=HE_BIO_COLORS.get(prev_cat, "#888"),
                            fontweight="bold", clip_on=False)
                cat_start = yi; prev_cat = cat
        if prev_cat:
            mid = (cat_start + len(cats_s) - 1) / 2
            ax.text(ax.get_xlim()[1], mid,
                    f"  {HE_SHORT.get(prev_cat, prev_cat)}",
                    ha="left", va="center", fontsize=5.5,
                    color=HE_BIO_COLORS.get(prev_cat, "#888"),
                    fontweight="bold", clip_on=False)


def main():
    all_task_rel = {}

    for task_key, cfg in TASK_CFG.items():
        print(f"\n=== {cfg['label']} ===", flush=True)
        mod_cluster_vals = load_cluster_aff(task_key, cfg)
        mod_rel = {}
        for mod in MOD_ORDER:
            if mod not in mod_cluster_vals:
                print(f"  {mod}: no data", flush=True); continue
            mod_rel[mod] = rel_attn(mod_cluster_vals[mod])
            print(f"  {mod}: {len(mod_rel[mod])} clusters", flush=True)
        all_task_rel[task_key] = mod_rel

        # Per-task figure: 1 row × 4 modality columns
        n_mods   = len(mod_rel)
        max_clus = max((TOP_N[m] for m in mod_rel), default=30)
        fig_h    = max(10, max_clus * 0.22)
        fig, axes = plt.subplots(1, len(MOD_ORDER), figsize=(len(MOD_ORDER) * 7, fig_h),
                                 facecolor=BG)
        fig.patch.set_facecolor(BG)
        fig.suptitle(
            f"LongMIL Cluster Attention — {cfg['label']}\n"
            f"Relative attention per instance cluster  ·  1.0 = uniform baseline  ·  "
            f"error bars = SEM across 5 splits",
            fontsize=12, fontweight="bold", color=cfg["color"]
        )

        for ax, mod in zip(axes, MOD_ORDER):
            rel = mod_rel.get(mod, {})
            plot_mod_ax(ax, mod, rel, cfg["color"], TOP_N.get(mod, 30))

        # HE bio category legend
        he_patches = [mpatches.Patch(color=HE_BIO_COLORS.get(cat, "#b0bec5"),
                                     label=HE_SHORT.get(cat, cat))
                      for cat in HE_BIO_ORDER]
        fig.legend(handles=he_patches, loc="lower center", ncol=len(he_patches),
                   fontsize=8, framealpha=0.85, title="HE biological category",
                   bbox_to_anchor=(0.5, 0.0))

        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        for ext in ("png", "pdf"):
            fig.savefig(OUT_ROOT / f"multimod_cluster_attn_{task_key}.{ext}",
                        dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"  Saved multimod_cluster_attn_{task_key}", flush=True)

    # Combined all-tasks: 4 tasks × 1 col each (HE only for overview, most interpretable)
    print("\nSaving HE-focused all-tasks overview...", flush=True)
    fig, axes = plt.subplots(1, 4, figsize=(36, max(10, TOP_N["HE"] * 0.22)), facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.suptitle("LongMIL HE Cluster Attention — All Tasks\n"
                 "Relative attention  ·  1.0 = uniform  ·  error bars = SEM (5 splits)",
                 fontsize=13, fontweight="bold")
    for ax, (task_key, cfg) in zip(axes, TASK_CFG.items()):
        rel = all_task_rel.get(task_key, {}).get("HE", {})
        plot_mod_ax(ax, "HE", rel, cfg["color"], TOP_N["HE"])
        ax.set_title(f"{cfg['label']}\n(HE clusters)", fontsize=10,
                     fontweight="bold", color=cfg["color"])
    he_patches = [mpatches.Patch(color=HE_BIO_COLORS.get(cat, "#b0bec5"),
                                 label=HE_SHORT.get(cat, cat))
                  for cat in HE_BIO_ORDER]
    fig.legend(handles=he_patches, loc="lower center", ncol=len(he_patches),
               fontsize=9, framealpha=0.85, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(OUT_ROOT / f"he_cluster_all_tasks.{ext}",
                    dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    print(f"\nDone. → {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
