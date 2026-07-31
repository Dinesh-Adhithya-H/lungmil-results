"""
Generate cross-split aggregated seed attribution plots for death_surv and clad_surv.
Handles the naming mismatch where dir is *_death_surv but JSON is seed_attribution_data_death.json.
"""
import sys
sys.path.insert(0, "/home/aih/dinesh.haridoss/chicago_mil")

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_ROOT = Path("/home/aih/dinesh.haridoss/chicago_mil/interpretability/longitudinal_mk_interp")
FIGURES  = Path("/home/aih/dinesh.haridoss/chicago_mil/figures/interpretability/agg")
FIGURES.mkdir(parents=True, exist_ok=True)

FONT_LABEL = 10

MOD_COLORS = {
    "HE":       "#E64A19",
    "BAL":      "#1565C0",
    "CT":       "#2E7D32",
    "Clinical": "#6A1B9A",
}

TASK_META = {
    "death": {"label": "Mortality (death)",
              "dir_suffix": "death_surv", "json_key": "death"},
    "clad":  {"label": "CLAD",
              "dir_suffix": "clad_surv",  "json_key": "clad"},
}

VARIANTS = ["longitudinal_mk_no_alibi", "longitudinal_mk_mt_no_alibi"]

def _savefig(fig, out_dir, stem):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    return out_dir / f"{stem}.png"


def aggregate_task(variant, task_key, task_meta, n_splits=5):
    split_diffs, seed_labels, present_mods = [], None, None
    for split in range(n_splits):
        cand = OUT_ROOT / f"{variant}_split{split}_fold0_{task_meta['dir_suffix']}"
        jpath = cand / f"seed_attribution_data_{task_meta['json_key']}.json"
        if jpath.exists():
            d = json.loads(jpath.read_text())
            split_diffs.append(np.array(d["alpha_diff"]))
            if seed_labels is None:
                seed_labels = d["seed_labels"]
                present_mods = d["present_mods"]
        else:
            print(f"  [agg] missing: {jpath}")

    if len(split_diffs) < 2:
        print(f"  [agg] {task_key} ({variant}): only {len(split_diffs)} splits, skipping")
        return

    min_len = min(a.shape[0] for a in split_diffs)
    arr = np.stack([a[:min_len] for a in split_diffs])
    mean_diff = arr.mean(0)
    std_diff  = arr.std(0)
    seed_labels = (seed_labels or [str(i) for i in range(min_len)])[:min_len]

    mod_spans = {}
    idx = 0
    for mo in (present_mods or []):
        cnt = sum(1 for lb in seed_labels if lb.startswith(mo[:3]))
        mod_spans[mo] = (idx, idx + cnt)
        idx += cnt

    x = np.arange(min_len)
    bar_cols = ["#E53935" if v > 0 else "#1E88E5" for v in mean_diff]

    fig, ax = plt.subplots(figsize=(max(14, min_len * 0.22), 4.5))
    ax.bar(x, mean_diff, color=bar_cols, width=0.75, alpha=0.85)
    ax.errorbar(x, mean_diff, yerr=std_diff, fmt="none",
                color="#333", capsize=2, lw=1.0, alpha=0.7)
    ax.axhline(0, color="#333", lw=0.8)

    for mo in (present_mods or [])[1:]:
        ax.axvline(mod_spans[mo][0] - 0.5, color="#aaa", lw=0.7, ls="--")
    for mo in (present_mods or []):
        mid = (mod_spans[mo][0] + mod_spans[mo][1]) / 2
        ylim = ax.get_ylim()
        ax.text(mid, ylim[1], mo, ha="center", va="bottom", fontsize=8,
                color=MOD_COLORS.get(mo, "#888"), fontweight="bold")

    ax.set_title(
        f"{task_meta['label']} — Seed attribution  Δα  (mean ± std, n={len(split_diffs)} splits)\n"
        "Red = enriched in high-risk,  Blue = enriched in low-risk",
        fontsize=FONT_LABEL, fontweight="bold")
    ax.set_xlabel("Seed (modality · seed_k)", fontsize=FONT_LABEL - 1)
    ax.set_ylabel("Δα  (high-risk − low-risk)", fontsize=FONT_LABEL - 1)
    ax.set_xticks(x)
    ax.set_xticklabels(seed_labels, rotation=60, ha="right", fontsize=5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    # Save to variant agg dir
    agg_dir = OUT_ROOT / f"{variant}_agg"
    agg_dir.mkdir(parents=True, exist_ok=True)
    stem = f"Lpop_K_agg_{task_meta['json_key']}_surv"
    _savefig(fig, agg_dir, stem)
    print(f"  [agg] saved {agg_dir / stem}.png  ({len(split_diffs)} splits)")

    # Also save to figures/interpretability/agg
    _savefig(fig, FIGURES, f"{variant}_{stem}")
    plt.close(fig)


if __name__ == "__main__":
    for variant in VARIANTS:
        print(f"\n=== {variant} ===")
        for task_key, task_meta in TASK_META.items():
            aggregate_task(variant, task_key, task_meta)

    print("\n[done] agg plots for death and clad written.")
