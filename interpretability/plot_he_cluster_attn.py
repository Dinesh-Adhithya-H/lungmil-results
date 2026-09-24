"""
HE instance cluster attention from phase1 ABMIL HE — raw mean attention per cluster.

Loads SingleModalMIL (phase1, direct ABMIL on HE instances), runs on all test
patients across 5 splits, collects alpha per instance, aggregates by cluster label.

Output: one bar chart per task — x = HE cluster (grouped by biological category),
        y = mean ABMIL attention, colored by bio category.

No high/low risk split — just raw overall mean attention per cluster.

Output: figures/interpretability/he_cluster_attn/
Usage: sbatch interpretability/submit_he_cluster_attn.sh
"""

import json, math, sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mil.models.phase1 import SingleModalMIL
from mil.data.splits import build_splits_longitudinal
from mil.data.loader import preload_bags
from interpretability.shared import (SPLITS_CSV, SAMPLES_DIR, RESULTS_ROOT,
                                     HE_BIO_MAP, HE_BIO_COLORS)

# ── Paths ─────────────────────────────────────────────────────────────────────
PHASE1_ROOT = ROOT / "results" / "mm_abmil_v8" / "phase1"
OUT_ROOT    = ROOT / "figures" / "interpretability" / "he_cluster_attn"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

HE_MOD      = "HE_cells"   # key in .pt cluster_labels dict
HE_BAG_KEY  = "HE"        # key in bag_cache (preload_bags uses modality name)
BG     = "#FAF6F2"

TASK_CFG = {
    "acr_cls":   {"p1_task": "acr",      "label": "ACR Classification", "color": "#e53935"},
    "acr_surv":  {"p1_task": "acr_surv", "label": "ACR Survival",       "color": "#ff7043"},
    "clad_surv": {"p1_task": "clad",     "label": "CLAD Survival",      "color": "#7e57c2"},
    "death_surv":{"p1_task": "death",    "label": "Death Survival",     "color": "#26a69a"},
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ── Model loading ─────────────────────────────────────────────────────────────

def load_phase1(split, p1_task, device):
    ckpt = PHASE1_ROOT / f"split{split}_fold0" / p1_task / "HE" / "final_combined" / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))
    raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw.get("state_dict", raw))
    model = SingleModalMIL(feat_dim=1024)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


# ── Cluster label cache ────────────────────────────────────────────────────────

def build_cluster_cache(stems):
    """stem → list of cluster name strings (one per HE instance)."""
    cache = {}
    for stem in stems:
        pt = Path(SAMPLES_DIR) / f"{stem}.pt"
        if not pt.exists():
            continue
        try:
            d  = torch.load(str(pt), map_location="cpu", weights_only=False)
            cl = d.get("cluster_labels", {}).get(HE_MOD)
            if cl is not None and len(cl) > 0:
                cache[stem] = list(cl)
        except Exception:
            pass
    return cache


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_cluster_attn(task_key, cfg, device, bag_cache, cluster_cache):
    """
    Returns {cluster_name: [alpha_values_across_all_patients_and_biopsies]}.
    No risk-group split — just raw per-instance attention aggregated by cluster.
    """
    cluster_alphas = defaultdict(list)

    for split in range(5):
        print(f"  split {split}", flush=True)
        try:
            model = load_phase1(split, cfg["p1_task"], device)
        except FileNotFoundError as e:
            print(f"    [skip] {e}", flush=True)
            continue

        splits = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=split)
        test_pats = splits["test"]

        with torch.no_grad():
            for pat in test_pats:
                for stem in pat["stems"]:
                    x = bag_cache.get(stem, {}).get(HE_BAG_KEY)
                    if x is None or x.shape[0] == 0:
                        continue
                    cl_names = cluster_cache.get(stem)
                    if cl_names is None or len(cl_names) != x.shape[0]:
                        continue
                    try:
                        _, extras = model(x.to(device), return_extras=True)
                        alpha = extras["alpha"].squeeze().detach().cpu().numpy()  # (N,)
                    except Exception as e:
                        print(f"      [err] {stem}: {e}", flush=True)
                        continue
                    for a, cn in zip(alpha, cl_names):
                        cluster_alphas[cn].append(float(a))

        del model
        torch.cuda.empty_cache()

    return cluster_alphas


# ── Plotting ──────────────────────────────────────────────────────────────────

def bio_cat(name):
    return HE_BIO_MAP.get(name, "Unknown")


def bio_color(name):
    return HE_BIO_COLORS.get(bio_cat(name), "#b0bec5")


BIO_ORDER = [
    "Alveolar with hemorrhage and inflammation",
    "Alveolar with empty spaces",
    "Alveolar",
    "Bronchial",
    "Lymphocytoplasmic inflammation",
    "Cartilage",
    "Unknown",
]

SHORT_BIO = {
    "Alveolar with hemorrhage and inflammation": "Alv+Inflam",
    "Alveolar with empty spaces":               "Alv+Empty",
    "Alveolar":                                 "Alveolar",
    "Bronchial":                                "Bronchial",
    "Lymphocytoplasmic inflammation":           "Lymphocytic",
    "Cartilage":                                "Cartilage",
    "Unknown":                                  "Unknown",
}


def plot_task(ax, cluster_alphas, task_key, cfg):
    if not cluster_alphas:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(cfg["label"]); return

    # Normalize: relative to overall mean → 1.0 = average attention
    all_vals     = [v for vals in cluster_alphas.values() for v in vals]
    overall_mean = np.mean(all_vals) if all_vals else 1.0

    names  = sorted(cluster_alphas.keys())
    means  = np.array([np.mean(cluster_alphas[n]) / overall_mean for n in names])
    sems   = np.array([np.std(cluster_alphas[n]) / overall_mean
                       / np.sqrt(max(len(cluster_alphas[n]), 1)) for n in names])
    cats   = [bio_cat(n) for n in names]
    colors = [bio_color(n) for n in names]

    # Sort by bio category order, then by relative attention within category
    def sort_key(i):
        cat = cats[i]
        return (BIO_ORDER.index(cat) if cat in BIO_ORDER else len(BIO_ORDER), -means[i])

    order    = sorted(range(len(names)), key=sort_key)
    names_s  = [names[i]  for i in order]
    means_s  = means[order]
    sems_s   = sems[order]
    colors_s = [colors[i] for i in order]
    cats_s   = [cats[i]   for i in order]

    # ── Horizontal bar chart ──────────────────────────────────────────────────
    y = np.arange(len(names_s))
    ax.barh(y, means_s, color=colors_s, alpha=0.88, height=0.72, zorder=3)
    ax.errorbar(means_s, y, xerr=sems_s, fmt="none",
                color="#555", lw=0.7, alpha=0.45, zorder=4)
    ax.axvline(1.0, color="#555", lw=0.9, ls="--", alpha=0.5, zorder=2)  # baseline = uniform

    ax.set_yticks(y)
    ax.set_yticklabels(names_s, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Relative attention (1.0 = uniform)", fontsize=9)
    ax.set_title(cfg["label"], fontsize=12, fontweight="bold", color=cfg["color"], pad=8)
    ax.set_facecolor(BG)
    ax.grid(axis="x", alpha=0.18, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # Category block dividers on the y-axis
    prev_cat = None
    cat_start = 0
    for yi, cat in enumerate(cats_s):
        if cat != prev_cat:
            if prev_cat is not None:
                ax.axhline(yi - 0.5, color="#bbb", lw=0.8, ls="-", alpha=0.7, zorder=1)
                mid = (cat_start + yi - 1) / 2
                ax.text(ax.get_xlim()[1], mid, f"  {SHORT_BIO.get(prev_cat, prev_cat)}",
                        ha="left", va="center", fontsize=7,
                        color=HE_BIO_COLORS.get(prev_cat, "#888"), fontweight="bold",
                        clip_on=False)
            cat_start = yi
            prev_cat = cat
    # Last category
    if prev_cat:
        mid = (cat_start + len(cats_s) - 1) / 2
        xmax = ax.get_xlim()[1]
        ax.text(xmax, mid, f"  {SHORT_BIO.get(prev_cat, prev_cat)}",
                ha="left", va="center", fontsize=7,
                color=HE_BIO_COLORS.get(prev_cat, "#888"), fontweight="bold",
                clip_on=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Preload bags and cluster labels once
    print("Preloading bags...", flush=True)
    all_stems = set()
    for split in range(5):
        try:
            sp = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=split)
            for pats in sp.values():
                for pat in pats:
                    all_stems.update(pat["stems"])
        except Exception:
            pass
    bag_cache     = preload_bags(list(all_stems), SAMPLES_DIR, n_workers=4)
    cluster_cache = build_cluster_cache(list(all_stems))
    print(f"  {len(bag_cache)} bags, {len(cluster_cache)} cluster label files", flush=True)

    # Per-task: individual figures
    all_ca = {}
    for task_key, cfg in TASK_CFG.items():
        print(f"\n=== {cfg['label']} ===", flush=True)
        ca = extract_cluster_attn(task_key, cfg, device, bag_cache, cluster_cache)
        print(f"  {len(ca)} clusters with data", flush=True)
        all_ca[task_key] = ca

        n_clusters = len(ca)
        fig2, ax2 = plt.subplots(figsize=(10, max(8, n_clusters * 0.28)), facecolor=BG)
        fig2.patch.set_facecolor(BG)
        plot_task(ax2, ca, task_key, cfg)
        fig2.tight_layout()
        for ext in ("png", "pdf"):
            fig2.savefig(OUT_ROOT / f"he_cluster_attn_{task_key}.{ext}",
                         dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig2)
        print(f"  Saved he_cluster_attn_{task_key}", flush=True)

    # 4-panel combined (1×4 horizontal — each is a tall narrow panel)
    n_clusters_max = max((len(ca) for ca in all_ca.values()), default=40)
    fig, axes = plt.subplots(1, 4, figsize=(36, max(10, n_clusters_max * 0.28)), facecolor=BG)
    fig.suptitle("Phase1 ABMIL HE — Relative Cluster Attention (÷ uniform baseline)\n"
                 "Grouped by biological category · error bars = SEM",
                 fontsize=13, fontweight="bold")

    for ax, (task_key, cfg) in zip(axes, TASK_CFG.items()):
        plot_task(ax, all_ca.get(task_key, {}), task_key, cfg)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(OUT_ROOT / f"he_cluster_attn_all_tasks.{ext}",
                    dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("\nSaved he_cluster_attn_all_tasks", flush=True)
    print(f"Done. → {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
