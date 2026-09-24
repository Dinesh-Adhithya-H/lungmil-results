"""
HE instance cluster attention comparison: Unimodal HE ABMIL (phase1) vs LongMIL (phase2).

For each task shows side-by-side horizontal bars per HE cluster:
  - Unimodal: mean ABMIL alpha per instance, grouped by cluster (phase1 model)
  - LongMIL:  combined (n_hi*hi + n_lo*lo)/(n_hi+n_lo) PMA-weighted mass per cluster,
              summed over K seeds (phase2 model)
Both normalized to relative attention (1.0 = uniform baseline).

No high/low risk split — overall cluster attention for each model.

Output: figures/interpretability/he_cluster_comparison/
Usage: sbatch interpretability/submit_he_cluster_comparison.sh
"""

import json, sys, warnings
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

PHASE1_ROOT = ROOT / "results" / "mm_abmil_v8" / "phase1"
INTERP_ROOT = ROOT / "interpretability" / "longitudinal_mk_interp"
OUT_ROOT    = ROOT / "figures" / "interpretability" / "he_cluster_comparison"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

VARIANT    = "longitudinal_mk_no_alibi_v2"
HE_MOD     = "HE_cells"
HE_BAG_KEY = "HE"
BG         = "#FAF6F2"

TASK_CFG = {
    "acr_cls":   {"p1_task": "acr",      "dir_suffix": "acr_cls",   "json_key": "acr_cls",  "label": "ACR Classification", "color": "#e53935"},
    "acr_surv":  {"p1_task": "acr_surv", "dir_suffix": "acr_surv",  "json_key": "acr_surv", "label": "ACR Survival",       "color": "#ff7043"},
    "clad_surv": {"p1_task": "clad",     "dir_suffix": "clad_surv", "json_key": "clad",     "label": "CLAD Survival",      "color": "#7e57c2"},
    "death_surv":{"p1_task": "death",    "dir_suffix": "death_surv","json_key": "death",    "label": "Death Survival",     "color": "#26a69a"},
}

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

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10,
})


def bio_cat(name):   return HE_BIO_MAP.get(name, "Unknown")
def bio_color(name): return HE_BIO_COLORS.get(bio_cat(name), "#b0bec5")


# ── Phase1 (unimodal) ─────────────────────────────────────────────────────────

def load_phase1(split, p1_task, device):
    ckpt = PHASE1_ROOT / f"split{split}_fold0" / p1_task / "HE" / "final_combined" / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))
    raw   = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw.get("state_dict", raw))
    model = SingleModalMIL(feat_dim=1024)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


def build_cluster_cache(stems):
    cache = {}
    for stem in stems:
        pt = Path(SAMPLES_DIR) / f"{stem}.pt"
        if not pt.exists(): continue
        try:
            d  = torch.load(str(pt), map_location="cpu", weights_only=False)
            cl = d.get("cluster_labels", {}).get(HE_MOD)
            if cl is not None and len(cl) > 0:
                cache[stem] = list(cl)
        except Exception: pass
    return cache


def extract_unimodal_attn(cfg, device, bag_cache, cluster_cache):
    """Mean ABMIL alpha per HE instance cluster, averaged across all 5 splits × test patients."""
    cluster_alphas = defaultdict(list)
    for split in range(5):
        print(f"  [phase1] split {split}", flush=True)
        try:
            model = load_phase1(split, cfg["p1_task"], device)
        except FileNotFoundError as e:
            print(f"    [skip] {e}", flush=True); continue

        splits = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=split)
        with torch.no_grad():
            for pat in splits["test"]:
                for stem in pat["stems"]:
                    x        = bag_cache.get(stem, {}).get(HE_BAG_KEY)
                    cl_names = cluster_cache.get(stem)
                    if x is None or cl_names is None or len(cl_names) != x.shape[0]: continue
                    try:
                        _, extras = model(x.to(device), return_extras=True)
                        alpha = extras["alpha"].squeeze().detach().cpu().numpy()
                    except Exception as e:
                        print(f"      [err] {stem}: {e}", flush=True); continue
                    for a, cn in zip(alpha, cl_names):
                        cluster_alphas[cn].append(float(a))
        del model; torch.cuda.empty_cache()
    return cluster_alphas  # {cluster: [alpha values]}


# ── LongMIL (phase2) ──────────────────────────────────────────────────────────

def extract_longmil_attn(cfg):
    """
    Load cluster_aff JSONs for all 5 splits.
    Returns {cluster: [weighted_attn per split]}.
    Combined = (n_hi*hi + n_lo*lo) / (n_hi+n_lo), summed over K seeds.
    """
    cluster_vals = defaultdict(list)
    for split in range(5):
        p = (INTERP_ROOT / f"{VARIANT}_split{split}_fold0_{cfg['dir_suffix']}"
             / f"cluster_aff_data_{cfg['json_key']}.json")
        if not p.exists():
            print(f"  [LongMIL] missing: {p}", flush=True); continue

        d     = json.loads(p.read_text())
        ca    = d.get("cluster_aff", {}).get("HE", {})
        names = ca.get("names", [])
        hi    = np.array(ca.get("hi", []))   # (K, C)
        lo    = np.array(ca.get("lo", []))   # (K, C)

        if hi.size == 0 or lo.size == 0 or not names: continue

        # Sum all patients (hi + lo) then sum over K seeds → (C,)
        total = (hi + lo).sum(axis=0)

        print(f"  [LongMIL] split {split}: {len(names)} clusters", flush=True)
        for ci, name in enumerate(names):
            cluster_vals[name].append(float(total[ci]))

    return cluster_vals  # {cluster: [value per split]}


# ── Plotting ──────────────────────────────────────────────────────────────────

def rel_attn(cluster_vals):
    """Normalize each cluster's mean to relative attention (÷ overall mean = 1.0)."""
    all_vals    = [v for vals in cluster_vals.values() for v in vals]
    overall     = np.mean(all_vals) if all_vals else 1.0
    return {n: (np.mean(v) / overall, np.std(v) / overall / np.sqrt(max(len(v), 1)))
            for n, v in cluster_vals.items()}  # {name: (rel_mean, rel_sem)}


def plot_comparison(ax_uni, ax_long, uni_rel, long_rel, task_key, cfg):
    """Two axes: left = unimodal, right = LongMIL. Shared y-axis (cluster order)."""
    # Union of clusters, sorted by bio category then unimodal attention
    all_names = sorted(set(uni_rel) | set(long_rel))

    def sort_key(n):
        cat = bio_cat(n)
        return (BIO_ORDER.index(cat) if cat in BIO_ORDER else len(BIO_ORDER),
                -uni_rel.get(n, (0, 0))[0])

    names_s = sorted(all_names, key=sort_key)
    cats_s  = [bio_cat(n) for n in names_s]
    colors_s = [bio_color(n) for n in names_s]
    y = np.arange(len(names_s))

    for ax, rel, model_label in [
        (ax_uni,  uni_rel,  "Unimodal HE ABMIL\n(Phase 1)"),
        (ax_long, long_rel, "LongMIL PMA-weighted\n(Phase 2)"),
    ]:
        means = np.array([rel.get(n, (0, 0))[0] for n in names_s])
        sems  = np.array([rel.get(n, (0, 0))[1] for n in names_s])

        ax.barh(y, means, color=colors_s, alpha=0.88, height=0.72, zorder=3)
        ax.errorbar(means, y, xerr=sems, fmt="none", color="#555",
                    lw=0.7, alpha=0.4, zorder=4)
        ax.axvline(1.0, color="#555", lw=0.9, ls="--", alpha=0.5)
        ax.set_yticks(y)
        ax.invert_yaxis()
        ax.set_xlabel("Relative attention (1.0 = uniform)", fontsize=8)
        ax.set_title(f"{cfg['label']}\n{model_label}",
                     fontsize=10, fontweight="bold", color=cfg["color"])
        ax.set_facecolor(BG)
        ax.grid(axis="x", alpha=0.18, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        # Category dividers
        prev_cat, cat_start = None, 0
        for yi, cat in enumerate(cats_s):
            if cat != prev_cat:
                if prev_cat is not None:
                    ax.axhline(yi - 0.5, color="#bbb", lw=0.8, alpha=0.7)
                cat_start = yi; prev_cat = cat

    # Cluster names only on left axis
    ax_uni.set_yticklabels(names_s, fontsize=6)
    ax_long.set_yticklabels([])   # shared y — no repeat

    # Category labels on right of LongMIL panel
    prev_cat, cat_start = None, 0
    for yi, cat in enumerate(cats_s):
        if cat != prev_cat:
            if prev_cat is not None:
                mid = (cat_start + yi - 1) / 2
                ax_long.text(ax_long.get_xlim()[1], mid,
                             f"  {SHORT_BIO.get(prev_cat, prev_cat)}",
                             ha="left", va="center", fontsize=7,
                             color=HE_BIO_COLORS.get(prev_cat, "#888"),
                             fontweight="bold", clip_on=False)
            cat_start = yi; prev_cat = cat
    if prev_cat:
        mid = (cat_start + len(cats_s) - 1) / 2
        ax_long.text(ax_long.get_xlim()[1], mid,
                     f"  {SHORT_BIO.get(prev_cat, prev_cat)}",
                     ha="left", va="center", fontsize=7,
                     color=HE_BIO_COLORS.get(prev_cat, "#888"),
                     fontweight="bold", clip_on=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Preload bags + cluster labels (needed for phase1 inference)
    print("Preloading bags...", flush=True)
    all_stems = set()
    for split in range(5):
        try:
            sp = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=split)
            for pats in sp.values():
                for pat in pats:
                    all_stems.update(pat["stems"])
        except Exception: pass
    bag_cache     = preload_bags(list(all_stems), SAMPLES_DIR, n_workers=4)
    cluster_cache = build_cluster_cache(list(all_stems))
    print(f"  {len(bag_cache)} bags, {len(cluster_cache)} cluster label files", flush=True)

    # 4-task figure: each task = 2 columns (unimodal | LongMIL)
    # Layout: 4 tasks × 2 models = 8 columns total, shared y per task pair
    n_tasks   = len(TASK_CFG)
    n_clusters_est = 54  # typical
    fig_h = max(12, n_clusters_est * 0.28)
    fig, axes = plt.subplots(1, n_tasks * 2,
                             figsize=(n_tasks * 9, fig_h),
                             facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "HE Instance Cluster Attention — Unimodal HE ABMIL (Phase 1) vs LongMIL (Phase 2)\n"
        "Relative attention: 1.0 = uniform baseline  ·  error bars = SEM across 5 splits",
        fontsize=13, fontweight="bold",
    )

    for ti, (task_key, cfg) in enumerate(TASK_CFG.items()):
        print(f"\n=== {cfg['label']} ===", flush=True)
        ax_uni  = axes[ti * 2]
        ax_long = axes[ti * 2 + 1]

        # Phase1 unimodal
        uni_raw  = extract_unimodal_attn(cfg, device, bag_cache, cluster_cache)
        uni_rel  = rel_attn(uni_raw)
        print(f"  Phase1: {len(uni_rel)} clusters", flush=True)

        # LongMIL
        long_raw = extract_longmil_attn(cfg)
        long_rel = rel_attn(long_raw)
        print(f"  LongMIL: {len(long_rel)} clusters", flush=True)

        plot_comparison(ax_uni, ax_long, uni_rel, long_rel, task_key, cfg)

        # Individual per-task figure
        fig2, axes2 = plt.subplots(1, 2, figsize=(18, fig_h), facecolor=BG)
        fig2.patch.set_facecolor(BG)
        fig2.suptitle(f"HE Cluster Attention — {cfg['label']}\n"
                      f"Phase1 Unimodal vs LongMIL Phase2",
                      fontsize=12, fontweight="bold")
        plot_comparison(axes2[0], axes2[1], uni_rel, long_rel, task_key, cfg)
        fig2.tight_layout(rect=[0, 0, 1, 0.95])
        for ext in ("png", "pdf"):
            fig2.savefig(OUT_ROOT / f"he_cluster_cmp_{task_key}.{ext}",
                         dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig2)
        print(f"  Saved he_cluster_cmp_{task_key}", flush=True)

    # Bio category legend
    legend_patches = [mpatches.Patch(color=HE_BIO_COLORS.get(cat, "#b0bec5"),
                                     label=SHORT_BIO.get(cat, cat))
                      for cat in BIO_ORDER]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(legend_patches),
               fontsize=9, framealpha=0.85, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(OUT_ROOT / f"he_cluster_cmp_all_tasks.{ext}",
                    dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("\nSaved he_cluster_cmp_all_tasks", flush=True)
    print(f"Done. → {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
