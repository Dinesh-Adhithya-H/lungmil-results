"""
Unimodal ABMIL (Phase1) vs LongMIL (Phase2) cluster attention — all modalities.

For HE, BAL, CT: side-by-side horizontal bars per cluster per task.
  Left  = Phase1 unimodal ABMIL alpha per instance, grouped by cluster
  Right = LongMIL PMA-weighted cluster attention (cluster_aff_data JSONs)
Both normalized to relative attention (1.0 = uniform baseline).

Clinical skipped — no per-instance cluster labels.

Output: figures/interpretability/multimodal_cluster_comparison/
Usage: sbatch interpretability/submit_multimodal_cluster_comparison.sh
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
OUT_ROOT    = ROOT / "figures" / "interpretability" / "multimodal_cluster_comparison"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

VARIANT = "longitudinal_mk_no_alibi_v2"
BG      = "#FAF6F2"

TASK_CFG = {
    "acr_cls":   {"p1_task": "acr",      "dir_suffix": "acr_cls",   "json_key": "acr_cls",  "label": "ACR Classification", "color": "#e53935"},
    "acr_surv":  {"p1_task": "acr_surv", "dir_suffix": "acr_surv",  "json_key": "acr_surv", "label": "ACR Survival",       "color": "#ff7043"},
    "clad_surv": {"p1_task": "clad",     "dir_suffix": "clad_surv", "json_key": "clad",     "label": "CLAD Survival",      "color": "#7e57c2"},
    "death_surv":{"p1_task": "death",    "dir_suffix": "death_surv","json_key": "death",    "label": "Death Survival",     "color": "#26a69a"},
}

MOD_CFG = {
    "HE":  {"feat_dim": 1024, "pt_key": "HE_cells",  "bag_key": "HE",  "color": "#4C9BE8"},
    "BAL": {"feat_dim": 10,   "pt_key": "BAL_cells", "bag_key": "BAL", "color": "#3FB950"},
    "CT":  {"feat_dim": 1024, "pt_key": "CT_cells",  "bag_key": "CT",  "color": "#D4A017"},
}

HE_BIO_ORDER = [
    "Alveolar with hemorrhage and inflammation", "Alveolar with empty spaces",
    "Alveolar", "Bronchial", "Lymphocytoplasmic inflammation", "Cartilage", "Unknown",
]
HE_SHORT = {
    "Alveolar with hemorrhage and inflammation": "Alv+Inflam",
    "Alveolar with empty spaces": "Alv+Empty",
    "Alveolar": "Alveolar", "Bronchial": "Bronchial",
    "Lymphocytoplasmic inflammation": "Lymphocytic",
    "Cartilage": "Cartilage", "Unknown": "Unknown",
}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


# ── Phase1 unimodal ───────────────────────────────────────────────────────────

def load_phase1(split, p1_task, mod, feat_dim, device):
    ckpt = PHASE1_ROOT / f"split{split}_fold0" / p1_task / mod / "final_combined" / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))
    raw   = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw.get("state_dict", raw))
    model = SingleModalMIL(feat_dim=feat_dim)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


def build_cluster_cache(stems, pt_key):
    cache = {}
    for stem in stems:
        pt = Path(SAMPLES_DIR) / f"{stem}.pt"
        if not pt.exists(): continue
        try:
            d  = torch.load(str(pt), map_location="cpu", weights_only=False)
            cl = d.get("cluster_labels", {}).get(pt_key)
            if cl is not None and len(cl) > 0:
                cache[stem] = list(cl)
        except Exception: pass
    return cache


def extract_unimodal(mod_cfg, p1_task, device, bag_cache, cluster_cache):
    """Raw ABMIL alpha per instance, grouped by cluster, across all 5 splits × test."""
    cluster_alphas = defaultdict(list)
    bag_key = mod_cfg["bag_key"]
    for split in range(5):
        print(f"    [uni-{bag_key}] split {split}", flush=True)
        try:
            model = load_phase1(split, p1_task, bag_key, mod_cfg["feat_dim"], device)
        except FileNotFoundError as e:
            print(f"      [skip] {e}", flush=True); continue
        splits = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=split)
        with torch.no_grad():
            for pat in splits["test"]:
                for stem in pat["stems"]:
                    x        = bag_cache.get(stem, {}).get(bag_key)
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
    return cluster_alphas


def extract_longmil(mod_key, dir_suffix, json_key):
    """LongMIL PMA-weighted cluster attention from cluster_aff JSONs, all 5 splits."""
    cluster_vals = defaultdict(list)
    for split in range(5):
        p = (INTERP_ROOT / f"{VARIANT}_split{split}_fold0_{dir_suffix}"
             / f"cluster_aff_data_{json_key}.json")
        if not p.exists(): continue
        d  = json.loads(p.read_text())
        ca = d.get("cluster_aff", {}).get(mod_key, {})
        names = ca.get("names", [])
        hi    = np.array(ca.get("hi", []))
        lo    = np.array(ca.get("lo", []))
        if hi.size == 0 or not names: continue
        total = (hi + lo).sum(axis=0)
        for ci, name in enumerate(names):
            cluster_vals[name].append(float(total[ci]))
    return cluster_vals


def rel_attn(cluster_vals):
    all_vals = [v for vals in cluster_vals.values() for v in vals]
    overall  = np.mean(all_vals) if all_vals else 1.0
    return {n: (np.mean(v) / overall, np.std(v) / overall / np.sqrt(max(len(v), 1)))
            for n, v in cluster_vals.items()}


# ── Plot helpers ──────────────────────────────────────────────────────────────

def he_sort(name, mean):
    cat = HE_BIO_MAP.get(name, "Unknown")
    return (HE_BIO_ORDER.index(cat) if cat in HE_BIO_ORDER else len(HE_BIO_ORDER), -mean)


def plot_pair(ax_uni, ax_long, uni_rel, long_rel, mod_key, mod_cfg, task_cfg):
    all_names = sorted(set(uni_rel) | set(long_rel))
    if mod_key == "HE":
        names_s = sorted(all_names, key=lambda n: he_sort(n, uni_rel.get(n, (0,0))[0]))
    else:
        names_s = sorted(all_names, key=lambda n: -uni_rel.get(n, (0,0))[0])

    y = np.arange(len(names_s))
    col = mod_cfg["color"]

    if mod_key == "HE":
        cats_s   = [HE_BIO_MAP.get(n, "Unknown") for n in names_s]
        colors_s = [HE_BIO_COLORS.get(c, "#b0bec5") for c in cats_s]
    else:
        colors_s = [col] * len(names_s)

    for ax, rel, model_lbl in [
        (ax_uni,  uni_rel,  f"Unimodal {mod_key}\n(Phase 1)"),
        (ax_long, long_rel, f"LongMIL {mod_key}\n(Phase 2)"),
    ]:
        means = np.array([rel.get(n, (0, 0))[0] for n in names_s])
        sems  = np.array([rel.get(n, (0, 0))[1] for n in names_s])
        ax.barh(y, means, color=colors_s, alpha=0.88, height=0.72, zorder=3)
        ax.errorbar(means, y, xerr=sems, fmt="none", color="#555", lw=0.6, alpha=0.4, zorder=4)
        ax.axvline(1.0, color="#555", lw=0.8, ls="--", alpha=0.5)
        ax.set_yticks(y); ax.invert_yaxis()
        ax.set_xlabel("Relative attention (1.0 = uniform)", fontsize=7)
        ax.set_title(model_lbl, fontsize=9, fontweight="bold", color=col, pad=3)
        ax.set_facecolor(BG); ax.grid(axis="x", alpha=0.15, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    ax_uni.set_yticklabels(names_s, fontsize=5)
    ax_long.set_yticklabels([])

    # HE bio category dividers + labels on right
    if mod_key == "HE":
        prev_cat, cat_start = None, 0
        for yi, cat in enumerate(cats_s):
            if cat != prev_cat:
                if prev_cat is not None:
                    for ax in (ax_uni, ax_long):
                        ax.axhline(yi - 0.5, color="#bbb", lw=0.7, alpha=0.6)
                    mid = (cat_start + yi - 1) / 2
                    ax_long.text(ax_long.get_xlim()[1], mid,
                                 f"  {HE_SHORT.get(prev_cat, prev_cat)}",
                                 ha="left", va="center", fontsize=5.5,
                                 color=HE_BIO_COLORS.get(prev_cat, "#888"),
                                 fontweight="bold", clip_on=False)
                cat_start = yi; prev_cat = cat
        if prev_cat:
            mid = (cat_start + len(cats_s) - 1) / 2
            ax_long.text(ax_long.get_xlim()[1], mid,
                         f"  {HE_SHORT.get(prev_cat, prev_cat)}",
                         ha="left", va="center", fontsize=5.5,
                         color=HE_BIO_COLORS.get(prev_cat, "#888"),
                         fontweight="bold", clip_on=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Preload bags + cluster caches for each modality
    print("Preloading bags...", flush=True)
    all_stems = set()
    for split in range(5):
        try:
            sp = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=split)
            for pats in sp.values():
                for pat in pats: all_stems.update(pat["stems"])
        except Exception: pass
    bag_cache = preload_bags(list(all_stems), SAMPLES_DIR, n_workers=4)
    print(f"  {len(bag_cache)} bags loaded", flush=True)

    cluster_caches = {mod: build_cluster_cache(list(all_stems), cfg["pt_key"])
                      for mod, cfg in MOD_CFG.items()}
    for mod, cc in cluster_caches.items():
        print(f"  {mod} cluster cache: {len(cc)} stems", flush=True)

    mods = list(MOD_CFG.keys())  # HE, BAL, CT
    n_cols = len(mods) * 2       # 2 axes per modality (uni | long)

    for task_key, tcfg in TASK_CFG.items():
        print(f"\n=== {tcfg['label']} ===", flush=True)
        n_clus_max = 54  # HE dominates height
        fig_h = max(12, n_clus_max * 0.26)
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 5.5, fig_h), facecolor=BG)
        fig.patch.set_facecolor(BG)
        fig.suptitle(
            f"Instance Cluster Attention — {tcfg['label']}\n"
            f"Left = Unimodal Phase1 ABMIL  ·  Right = LongMIL Phase2\n"
            f"Relative attention: 1.0 = uniform  ·  error bars = SEM (5 splits × test set)",
            fontsize=11, fontweight="bold", color=tcfg["color"]
        )

        for mi, (mod, mod_cfg) in enumerate(MOD_CFG.items()):
            ax_uni  = axes[mi * 2]
            ax_long = axes[mi * 2 + 1]

            print(f"  [{mod}] unimodal...", flush=True)
            uni_raw  = extract_unimodal(mod_cfg, tcfg["p1_task"], device,
                                        bag_cache, cluster_caches[mod])
            uni_rel  = rel_attn(uni_raw)
            print(f"    {len(uni_rel)} clusters", flush=True)

            print(f"  [{mod}] LongMIL...", flush=True)
            long_raw = extract_longmil(mod, tcfg["dir_suffix"], tcfg["json_key"])
            long_rel = rel_attn(long_raw)
            print(f"    {len(long_rel)} clusters", flush=True)

            plot_pair(ax_uni, ax_long, uni_rel, long_rel, mod, mod_cfg, tcfg)

        # HE bio legend
        he_patches = [mpatches.Patch(color=HE_BIO_COLORS.get(cat, "#b0bec5"),
                                     label=HE_SHORT.get(cat, cat))
                      for cat in HE_BIO_ORDER]
        fig.legend(handles=he_patches, loc="lower center", ncol=len(he_patches),
                   fontsize=8, framealpha=0.85, title="HE bio category",
                   bbox_to_anchor=(0.5, 0.0))

        fig.tight_layout(rect=[0, 0.04, 1, 0.94])
        for ext in ("png", "pdf"):
            fig.savefig(OUT_ROOT / f"multimod_cmp_{task_key}.{ext}",
                        dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"  Saved multimod_cmp_{task_key}", flush=True)

    print(f"\nDone. → {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
