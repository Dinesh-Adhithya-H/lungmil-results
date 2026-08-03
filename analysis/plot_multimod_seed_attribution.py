"""
Per-modality PMA seed attribution across all surv tasks.
Reads seed_attribution_data_*.json from longitudinal_mk_no_alibi splits 0-4,
averages alpha_diff per seed across splits, plots one panel per modality (HE, BAL, CT)
per task showing which prototype seeds drive high vs low risk.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

INTERP_ROOT = Path("/home/aih/dinesh.haridoss/chicago_mil/interpretability/longitudinal_mk_interp")
FIG_ROOT = Path("/home/aih/dinesh.haridoss/chicago_mil/figures/interpretability")

TASK_CFG = {
    "acr_surv": {"variant": "longitudinal_mk_no_alibi", "json_key": "acr_surv", "title": "ACR Survival"},
    "clad_surv": {"variant": "longitudinal_mk_no_alibi", "json_key": "clad",    "title": "CLAD"},
    "death_surv": {"variant": "longitudinal_mk_no_alibi", "json_key": "death",   "title": "Death"},
}

MOD_COLORS = {
    "HE":       "#4C72B0",
    "BAL":      "#DD8452",
    "CT":       "#55A868",
    "Clinical": "#C44E52",
}
MOD_ORDER = ["HE", "BAL", "CT", "Clinical"]

HI_RISK_COLOR = "#D62728"
LO_RISK_COLOR = "#1F77B4"


def load_alpha_diffs_for_task(task_key, cfg):
    variant = cfg["variant"]
    json_key = cfg["json_key"]
    all_diffs = []
    seed_labels = None
    for split in range(5):
        p = INTERP_ROOT / f"{variant}_split{split}_fold0_{task_key}" / f"seed_attribution_data_{json_key}.json"
        if not p.exists():
            print(f"  missing: {p}")
            continue
        d = json.load(open(p))
        if seed_labels is None:
            seed_labels = d["seed_labels"]
        all_diffs.append(d["alpha_diff"])
    if not all_diffs:
        return None, None
    mean_diff = np.mean(all_diffs, axis=0)
    std_diff  = np.std(all_diffs,  axis=0)
    return seed_labels, mean_diff, std_diff


def parse_by_modality(seed_labels, mean_diff, std_diff):
    by_mod = defaultdict(lambda: {"labels": [], "diffs": [], "stds": []})
    for lbl, d, s in zip(seed_labels, mean_diff, std_diff):
        mod, seed_id = lbl.split("·")
        by_mod[mod]["labels"].append(seed_id)
        by_mod[mod]["diffs"].append(d)
        by_mod[mod]["stds"].append(s)
    return by_mod


def plot_task(task_key, cfg):
    out_dir = FIG_ROOT / task_key
    out_dir.mkdir(parents=True, exist_ok=True)

    result = load_alpha_diffs_for_task(task_key, cfg)
    if result[0] is None:
        print(f"No data for {task_key}, skipping.")
        return
    seed_labels, mean_diff, std_diff = result

    by_mod = parse_by_modality(seed_labels, mean_diff, std_diff)
    mods_present = [m for m in MOD_ORDER if m in by_mod]

    n_mods = len(mods_present)
    fig, axes = plt.subplots(1, n_mods, figsize=(4.5 * n_mods, 6))
    if n_mods == 1:
        axes = [axes]

    fig.suptitle(f"Seed Attribution by Modality — {cfg['title']}", fontsize=13, fontweight="bold", y=1.01)

    for ax, mod in zip(axes, mods_present):
        info = by_mod[mod]
        diffs = np.array(info["diffs"])
        stds  = np.array(info["stds"])
        labels = info["labels"]

        # sort by absolute Δα descending
        order = np.argsort(np.abs(diffs))[::-1]
        diffs_s  = diffs[order]
        stds_s   = stds[order]
        labels_s = [labels[i] for i in order]

        y = np.arange(len(labels_s))
        colors = [HI_RISK_COLOR if v > 0 else LO_RISK_COLOR for v in diffs_s]

        ax.barh(y, diffs_s, xerr=stds_s, color=colors, alpha=0.85,
                error_kw=dict(ecolor="grey", capsize=2, linewidth=0.8))
        ax.set_yticks(y)
        ax.set_yticklabels([f"{mod}·{l}" for l in labels_s], fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Δα (high-risk − low-risk)", fontsize=9)
        ax.set_title(mod, fontsize=11, fontweight="bold", color=MOD_COLORS.get(mod, "black"))
        ax.invert_yaxis()

        # subtle background stripe for positive bars
        ax.set_facecolor("#FAFAFA")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    hi_patch = mpatches.Patch(color=HI_RISK_COLOR, label="High-risk associated (Δα > 0)")
    lo_patch = mpatches.Patch(color=LO_RISK_COLOR, label="Low-risk associated (Δα < 0)")
    fig.legend(handles=[hi_patch, lo_patch], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = out_dir / f"multimod_seed_attribution_{task_key}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def plot_all_tasks_summary():
    """One combined figure: rows = tasks, cols = modalities."""
    tasks = list(TASK_CFG.items())
    all_data = {}
    all_mods_present = set()
    for task_key, cfg in tasks:
        result = load_alpha_diffs_for_task(task_key, cfg)
        if result[0] is None:
            continue
        seed_labels, mean_diff, std_diff = result
        by_mod = parse_by_modality(seed_labels, mean_diff, std_diff)
        all_data[task_key] = (cfg, by_mod)
        all_mods_present.update(by_mod.keys())

    mods_present = [m for m in MOD_ORDER if m in all_mods_present]
    n_rows = len(all_data)
    n_cols = len(mods_present)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 5 * n_rows), squeeze=False)
    fig.suptitle("PMA Seed Attribution by Modality — All Survival Tasks", fontsize=14, fontweight="bold", y=1.01)

    for row_idx, (task_key, (cfg, by_mod)) in enumerate(all_data.items()):
        for col_idx, mod in enumerate(mods_present):
            ax = axes[row_idx][col_idx]
            if mod not in by_mod:
                ax.axis("off")
                continue
            info = by_mod[mod]
            diffs = np.array(info["diffs"])
            stds  = np.array(info["stds"])
            labels = info["labels"]

            order = np.argsort(np.abs(diffs))[::-1]
            diffs_s  = diffs[order]
            stds_s   = stds[order]
            labels_s = [labels[i] for i in order]

            y = np.arange(len(labels_s))
            colors = [HI_RISK_COLOR if v > 0 else LO_RISK_COLOR for v in diffs_s]

            ax.barh(y, diffs_s, xerr=stds_s, color=colors, alpha=0.85,
                    error_kw=dict(ecolor="grey", capsize=2, linewidth=0.7))
            ax.set_yticks(y)
            ax.set_yticklabels([f"{mod}·{l}" for l in labels_s], fontsize=7)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.invert_yaxis()
            ax.set_facecolor("#FAFAFA")
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

            if row_idx == 0:
                ax.set_title(mod, fontsize=11, fontweight="bold", color=MOD_COLORS.get(mod, "black"))
            if col_idx == 0:
                ax.set_ylabel(cfg["title"], fontsize=10, fontweight="bold")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Δα (high − low risk)", fontsize=8)

    hi_patch = mpatches.Patch(color=HI_RISK_COLOR, label="High-risk (Δα > 0)")
    lo_patch = mpatches.Patch(color=LO_RISK_COLOR, label="Low-risk (Δα < 0)")
    fig.legend(handles=[hi_patch, lo_patch], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10)

    plt.tight_layout()
    out_dir = FIG_ROOT / "agg"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = out_dir / f"multimod_seed_attribution_all_tasks.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    for task_key, cfg in TASK_CFG.items():
        print(f"\n=== {task_key} ===")
        plot_task(task_key, cfg)
    print("\n=== Combined summary ===")
    plot_all_tasks_summary()
    print("Done.")
