"""
HE bio-category level attention: ABMIL (Phase1) vs LongMIL (Phase2).

Collapses all clusters within each biological category into one bar.
4 tasks × 2 models, 7 bio categories per panel.

Output: figures/interpretability/he_biocategory_attn/
Usage: sbatch interpretability/submit_he_biocategory_attn.sh
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

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mil.models.phase1 import SingleModalMIL
from mil.data.splits import build_splits_longitudinal
from mil.data.loader import preload_bags
from interpretability.shared import SPLITS_CSV, SAMPLES_DIR, HE_BIO_MAP, HE_BIO_COLORS

PHASE1_ROOT = ROOT / "results" / "mm_abmil_v8" / "phase1"
INTERP_ROOT = ROOT / "interpretability" / "longitudinal_mk_interp"
OUT_ROOT    = ROOT / "figures" / "interpretability" / "he_biocategory_attn"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

VARIANT = "longitudinal_mk_no_alibi_v2"
BG      = "#FAF6F2"

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
BIO_SHORT = {
    "Alveolar with hemorrhage and inflammation": "Alv+Inflam",
    "Alveolar with empty spaces":               "Alv+Empty",
    "Alveolar":                                 "Alveolar",
    "Bronchial":                                "Bronchial",
    "Lymphocytoplasmic inflammation":           "Lymphocytic",
    "Cartilage":                                "Cartilage",
    "Unknown":                                  "Unknown",
}

ABMIL_COLOR  = "#4C9BE8"
LONGMIL_COLOR = "#E8874C"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


# ── Phase1 ABMIL ─────────────────────────────────────────────────────────────

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


def extract_abmil_bio(p1_task, device, bag_cache, cluster_cache):
    """Per-split relative attention per bio category, aggregated at cluster level.

    Steps per split:
      1. Collect all per-instance alpha values per cluster across test set
      2. Compute mean alpha per cluster  → relative to overall cluster mean
      3. Group clusters by bio category, average relative attentions within bio
    This matches LongMIL normalisation (per-cluster level) for a fair comparison.
    """
    bio_split_means = defaultdict(list)  # bio → [rel_mean per split]

    for split in range(5):
        print(f"  [ABMIL] split {split}", flush=True)
        try:
            model = load_phase1(split, p1_task, device)
        except FileNotFoundError as e:
            print(f"    [skip] {e}", flush=True); continue

        splits = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=split)
        cluster_alphas = defaultdict(list)  # cluster_id → [alpha values]

        with torch.no_grad():
            for pat in splits["test"]:
                for stem in pat["stems"]:
                    x        = bag_cache.get(stem, {}).get("HE")
                    cl_names = cluster_cache.get(stem)
                    if x is None or cl_names is None or len(cl_names) != x.shape[0]: continue
                    try:
                        _, extras = model(x.to(device), return_extras=True)
                        alpha = extras["alpha"].squeeze().detach().cpu().numpy()
                    except Exception:
                        continue
                    for a, cn in zip(alpha, cl_names):
                        cluster_alphas[cn].append(float(a))

        del model
        if device.type == "cuda": torch.cuda.empty_cache()

        # Per-cluster mean alpha → normalize by overall cluster-mean
        cluster_means = {cn: np.mean(vals) for cn, vals in cluster_alphas.items()}
        overall = np.mean(list(cluster_means.values())) if cluster_means else 1.0

        # Group by bio category, average per-cluster relative attentions
        bio_rel = defaultdict(list)
        for cn, cm in cluster_means.items():
            bio = HE_BIO_MAP.get(cn, "Unknown")
            bio_rel[bio].append(cm / overall)

        for bio, rels in bio_rel.items():
            bio_split_means[bio].append(np.mean(rels))

    return bio_split_means  # bio → [rel_mean per split]


def extract_longmil_bio(dir_suffix, json_key):
    """Per-split summed attention per bio category, normalized. Returns {bio: [split_val, ...]}."""
    bio_split_vals = defaultdict(list)

    for split in range(5):
        p = (INTERP_ROOT / f"{VARIANT}_split{split}_fold0_{dir_suffix}"
             / f"cluster_aff_data_{json_key}.json")
        if not p.exists(): continue
        d    = json.loads(p.read_text())
        ca   = d.get("cluster_aff", {}).get("HE", {})
        names = ca.get("names", [])
        hi    = np.array(ca.get("hi", []))
        lo    = np.array(ca.get("lo", []))
        if hi.size == 0 or not names: continue

        total    = (hi + lo).sum(axis=0)   # (C,) raw
        overall  = total.mean() if total.size else 1.0

        bio_agg = defaultdict(float)
        bio_cnt = defaultdict(int)
        for ci, name in enumerate(names):
            bio = HE_BIO_MAP.get(name, "Unknown")
            bio_agg[bio] += float(total[ci])
            bio_cnt[bio] += 1

        for bio, s in bio_agg.items():
            n = bio_cnt[bio]
            bio_split_vals[bio].append((s / n) / overall if overall > 0 else 0.0)

    return bio_split_vals


def build_cluster_cache(stems):
    cache = {}
    for stem in stems:
        pt = Path(SAMPLES_DIR) / f"{stem}.pt"
        if not pt.exists(): continue
        try:
            d  = torch.load(str(pt), map_location="cpu", weights_only=False)
            cl = d.get("cluster_labels", {}).get("HE_cells")
            if cl is not None and len(cl) > 0:
                cache[stem] = list(cl)
        except Exception: pass
    return cache


def plot_panel(ax, abmil_bio, longmil_bio, task_cfg):
    cats   = BIO_ORDER
    n      = len(cats)
    height = 0.35
    y      = np.arange(n)

    abmil_means  = np.array([np.mean(abmil_bio.get(c, [1.0])) for c in cats])
    abmil_sems   = np.array([np.std(abmil_bio.get(c, [0.0])) / np.sqrt(max(len(abmil_bio.get(c, [1.0])), 1)) for c in cats])
    long_means   = np.array([np.mean(longmil_bio.get(c, [1.0])) for c in cats])
    long_sems    = np.array([np.std(longmil_bio.get(c, [0.0])) / np.sqrt(max(len(longmil_bio.get(c, [1.0])), 1)) for c in cats])

    ax.barh(y + height/2, abmil_means,  height, color=ABMIL_COLOR,   alpha=0.88, label="ABMIL (Phase1)", zorder=3)
    ax.barh(y - height/2, long_means,   height, color=LONGMIL_COLOR,  alpha=0.88, label="LongMIL (Phase2)", zorder=3)
    ax.errorbar(abmil_means, y + height/2, xerr=abmil_sems, fmt="none", color="#333", lw=0.8, alpha=0.5, zorder=4)
    ax.errorbar(long_means,  y - height/2, xerr=long_sems,  fmt="none", color="#333", lw=0.8, alpha=0.5, zorder=4)

    ax.axvline(1.0, color="#555", lw=1.0, ls="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([BIO_SHORT[c] for c in cats], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Relative attention (1.0 = uniform)", fontsize=9)
    ax.set_title(task_cfg["label"], fontsize=11, fontweight="bold",
                 color=task_cfg["color"], pad=5)
    ax.set_facecolor(BG)
    ax.grid(axis="x", alpha=0.18, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # Color y tick labels by bio category
    for tick, cat in zip(ax.get_yticklabels(), cats):
        tick.set_color(HE_BIO_COLORS.get(cat, "#333"))
        tick.set_fontweight("bold")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Preload
    all_stems = set()
    for split in range(5):
        sp = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=split)
        for pats in sp.values():
            for pat in pats: all_stems.update(pat["stems"])

    bag_cache     = preload_bags(list(all_stems), SAMPLES_DIR, n_workers=4)
    cluster_cache = build_cluster_cache(list(all_stems))
    print(f"  {len(bag_cache)} bags, {len(cluster_cache)} with HE cluster labels", flush=True)

    # 2×2 figure, one panel per task
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "HE Biological Category Attention: ABMIL (Phase1) vs LongMIL (Phase2)\n"
        "Relative attention per bio category  ·  1.0 = uniform  ·  error bars = SEM (5 splits)",
        fontsize=13, fontweight="bold"
    )

    axes_flat = axes.flatten()
    for ax, (task_key, tcfg) in zip(axes_flat, TASK_CFG.items()):
        print(f"\n=== {tcfg['label']} ===", flush=True)
        abmil_bio   = extract_abmil_bio(tcfg["p1_task"], device, bag_cache, cluster_cache)
        longmil_bio = extract_longmil_bio(tcfg["dir_suffix"], tcfg["json_key"])
        plot_panel(ax, abmil_bio, longmil_bio, tcfg)

    # Shared legend
    legend_handles = [
        mpatches.Patch(color=ABMIL_COLOR,   label="ABMIL (Phase1)"),
        mpatches.Patch(color=LONGMIL_COLOR,  label="LongMIL (Phase2)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(OUT_ROOT / f"he_biocategory_abmil_vs_longmil.{ext}",
                    dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nDone. → {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
