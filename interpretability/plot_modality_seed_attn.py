"""
Modality-level ABMIL seed attention for LongMIL (longitudinal_mk_no_alibi_v2).

Reads existing seed_attribution_data_*.json from all 5 splits × 4 tasks and
produces two figures:

  1. Grouped bar chart  — for each task, mean sum(Δα) per modality (± std across splits)
     Shows which modality's seeds are differentially attended for high vs low risk.

  2. Heatmap  — 4 tasks × 4 modalities,  mean modality-level |Δα|  (absolute differential)

Output: figures/interpretability/modality_seed_attn/
Usage: sbatch interpretability/submit_modality_seed_attn.sh
"""

import json, sys, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT     = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

INTERP_ROOT = ROOT / "interpretability" / "longitudinal_mk_interp"
OUT_ROOT    = ROOT / "figures" / "interpretability" / "modality_seed_attn"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

VARIANT = "longitudinal_mk_no_alibi_v2"

TASK_CFG = {
    "acr_cls":   {"dir_suffix": "acr_cls",  "json_key": "acr_cls",  "label": "ACR classification"},
    "acr_surv":  {"dir_suffix": "acr_surv", "json_key": "acr_surv", "label": "Time to first ACR"},
    "clad_surv": {"dir_suffix": "clad_surv","json_key": "clad",     "label": "Time to CLAD"},
    "death_surv":{"dir_suffix": "death_surv","json_key": "death",   "label": "Time to death"},
}

MOD_ORDER  = ["HE", "BAL", "CT", "Clinical"]
MOD_COLORS = {"HE": "#4C72B0", "BAL": "#DD8452", "CT": "#55A868", "Clinical": "#C44E52"}
TASK_COLORS = {
    "acr_cls":   "#E07B20",
    "acr_surv":  "#952030",
    "clad_surv": "#1A5C8A",
    "death_surv": "#2D8A4E",
}
BG = "#FAF6F2"

FS_TITLE  = 20
FS_LABEL  = 16
FS_TICK   = 14
FS_ANNOT  = 12
FS_LEGEND = 14

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": FS_TICK,
    "axes.titlesize": FS_LABEL, "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
})


def load_task_data(task_key, cfg):
    """Load seed attribution data across all 5 splits for one task.
    Returns list of dicts with alpha_mean (raw weighted mean), seed_labels, present_mods.
    Falls back to alpha_diff if alpha_hi/lo not present (old JSONs).
    """
    results = []
    for split in range(5):
        p = (INTERP_ROOT / f"{VARIANT}_split{split}_fold0_{cfg['dir_suffix']}"
             / f"seed_attribution_data_{cfg['json_key']}.json")
        if not p.exists():
            print(f"  missing: {p}")
            continue
        d = json.loads(p.read_text())
        n_hi = d.get("n_hi", 1); n_lo = d.get("n_lo", 1)
        if "alpha_hi" in d and "alpha_lo" in d:
            alpha_hi = np.array(d["alpha_hi"])
            alpha_lo = np.array(d["alpha_lo"])
            # weighted mean across all patients (hi + lo)
            alpha_mean = (alpha_hi * n_hi + alpha_lo * n_lo) / max(n_hi + n_lo, 1)
        else:
            alpha_mean = np.array(d["alpha_diff"])  # fallback: diff (old JSONs)
        results.append({
            "alpha_mean":  alpha_mean,
            "seed_labels": d["seed_labels"],
            "present_mods": d["present_mods"],
        })
    return results


def aggregate_by_modality(alpha_mean, seed_labels, present_mods):
    """Group seed raw mean alpha by modality, return {mod: sum} and {mod: list}."""
    mod_sums = defaultdict(float)
    mod_vals = defaultdict(list)
    for v, lbl in zip(alpha_mean, seed_labels):
        mod = lbl.split("·")[0]
        if mod == "Cli":
            mod = "Clinical"
        mod_sums[mod] += v
        mod_vals[mod].append(v)
    return dict(mod_sums), dict(mod_vals)


def build_modality_table():
    """
    Returns:
        task_list: list of task keys
        mod_means: {task: {mod: mean sum(raw_alpha) across splits}}
        mod_stds:  {task: {mod: std  sum(raw_alpha) across splits}}
    """
    task_list = list(TASK_CFG.keys())
    mod_means = {}
    mod_stds  = {}

    for task_key, cfg in TASK_CFG.items():
        splits_data = load_task_data(task_key, cfg)
        if not splits_data:
            continue

        per_split_sums   = defaultdict(list)
        present_mods_all = splits_data[0]["present_mods"]

        for sd in splits_data:
            mod_sums, _ = aggregate_by_modality(sd["alpha_mean"], sd["seed_labels"], sd["present_mods"])
            for mod in present_mods_all:
                per_split_sums[mod].append(mod_sums.get(mod, 0.0))

        mod_means[task_key] = {mod: np.mean(v) for mod, v in per_split_sums.items()}
        mod_stds[task_key]  = {mod: np.std(v)  for mod, v in per_split_sums.items()}

    return task_list, mod_means, mod_stds


def plot_grouped_bars(task_list, mod_means, mod_stds):
    """Grouped bar chart: x = tasks, groups = modalities."""
    present_mods = [m for m in MOD_ORDER if any(m in mod_means.get(t, {}) for t in task_list)]
    n_tasks = len([t for t in task_list if t in mod_means])
    n_mods  = len(present_mods)

    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 9), facecolor=BG, sharey=False)

    if n_tasks == 1:
        axes = [axes]

    task_keys_present = [t for t in task_list if t in mod_means]
    for ax, task_key in zip(axes, task_keys_present):
        cfg   = TASK_CFG[task_key]
        means = mod_means[task_key]
        stds  = mod_stds[task_key]
        mods  = [m for m in present_mods if m in means]

        x    = np.arange(len(mods))
        vals = [means[m] for m in mods]
        errs = [stds[m]  for m in mods]
        colors = [MOD_COLORS.get(m, "#888") for m in mods]

        ax.bar(x, vals, color=colors, alpha=0.85, width=0.6, zorder=3,
               edgecolor="white", linewidth=0.5)
        ax.errorbar(x, vals, yerr=errs, fmt="none", color="#444",
                    capsize=5, lw=1.5, zorder=4)
        ax.axhline(0, color="#444", lw=1.0)

        # Value annotation inside each bar
        for xi, v in enumerate(vals):
            ax.text(xi, v / 2, f"{v:.3f}", ha="center", va="center",
                    fontsize=FS_ANNOT, color="white", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(mods, fontsize=FS_TICK, fontweight="bold")
        ax.set_title(cfg["label"], fontweight="bold", fontsize=FS_LABEL,
                     color=TASK_COLORS.get(task_key, "#333"), pad=10)
        if task_key == task_keys_present[0]:
            ax.set_ylabel("Mean α (sum over seeds)", fontsize=FS_LABEL)
        ax.grid(axis="y", alpha=0.2, lw=0.5)
        ax.set_facecolor(BG)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=FS_TICK)

    # Shared modality legend
    patches = [mpatches.Patch(color=MOD_COLORS[m], label=m)
               for m in MOD_ORDER if m in MOD_COLORS]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=FS_LEGEND, framealpha=0.85, title="Modality", title_fontsize=FS_LEGEND - 1,
               bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.07, 1, 1.0])
    for ext in ("png", "pdf"):
        fig.savefig(OUT_ROOT / f"modality_seed_attn_grouped.{ext}",
                    dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Saved modality_seed_attn_grouped", flush=True)


def plot_heatmap(task_list, mod_means, mod_stds):
    """Heatmap: rows = tasks, cols = modalities, cell = mean sum α."""
    present_mods = [m for m in MOD_ORDER if any(m in mod_means.get(t, {}) for t in task_list)]
    task_keys_present = [t for t in task_list if t in mod_means]
    task_labels = [TASK_CFG[t]["label"] for t in task_keys_present]

    mat = np.zeros((len(task_keys_present), len(present_mods)))
    for i, task_key in enumerate(task_keys_present):
        for j, mod in enumerate(present_mods):
            mat[i, j] = mod_means[task_key].get(mod, 0.0)

    vmin, vmax = mat.min(), mat.max()

    fig, ax = plt.subplots(figsize=(len(present_mods) * 3.0 + 2, len(task_keys_present) * 2.2 + 1.5),
                           facecolor=BG)
    im = ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, label="Mean α (sum over seeds)", fraction=0.046, pad=0.06)
    cbar.ax.tick_params(labelsize=FS_TICK)
    cbar.set_label("Mean α (sum over seeds)", fontsize=FS_LABEL)

    ax.set_xticks(range(len(present_mods)))
    ax.set_xticklabels(present_mods, fontsize=FS_LABEL, fontweight="bold")
    ax.set_yticks(range(len(task_labels)))
    ax.set_yticklabels(task_labels, fontsize=FS_LABEL)
    ax.set_title("ABMIL Seed Attention — Modality × Task",
                 fontsize=FS_TITLE, fontweight="bold", pad=14)

    # Annotate cells
    for i in range(len(task_keys_present)):
        for j in range(len(present_mods)):
            v = mat[i, j]
            c = "white" if abs(v) > 0.5 * vmax else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=FS_ANNOT + 2, color=c,
                    fontweight="bold")

    ax.set_facecolor(BG)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_ROOT / f"modality_seed_attn_heatmap.{ext}",
                    dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Saved modality_seed_attn_heatmap", flush=True)


def plot_per_seed_bars(task_list, task_mod_data):
    """
    One figure per task: for each seed, bar = mean alpha_diff (± std across splits),
    colored by modality. Like the existing Lpop_K_agg but with modality colors.
    """
    for task_key, cfg in TASK_CFG.items():
        splits_data = task_mod_data.get(task_key, [])
        if len(splits_data) < 2:
            continue

        # Align across splits
        min_len = min(len(sd["alpha_mean"]) for sd in splits_data)
        arr = np.stack([sd["alpha_mean"][:min_len] for sd in splits_data])  # (n_splits, S)
        mean_d = arr.mean(0)
        std_d  = arr.std(0)
        labels = splits_data[0]["seed_labels"][:min_len]

        # Modality colors per seed
        seed_colors = []
        for lbl in labels:
            mod = lbl.split("·")[0]
            if mod == "Cli":
                mod = "Clinical"
            seed_colors.append(MOD_COLORS.get(mod, "#888"))

        seed_colors_bar = []
        for lbl in labels:
            mod = lbl.split("·")[0]
            if mod == "Cli":
                mod = "Clinical"
            seed_colors_bar.append(MOD_COLORS.get(mod, "#888"))
        x = np.arange(min_len)

        fig, (ax_bar, ax_mod) = plt.subplots(2, 1, figsize=(max(20, min_len * 0.28), 9),
                                              gridspec_kw={"height_ratios": [5, 0.5]},
                                              facecolor=BG)
        fig.suptitle(f"{cfg['label']}  —  ABMIL attention per seed  (mean ± std, 5 splits)",
                     fontsize=FS_TITLE, fontweight="bold")

        ax_bar.bar(x, mean_d, color=seed_colors_bar, width=0.75, alpha=0.85)
        ax_bar.errorbar(x, mean_d, yerr=std_d, fmt="none", color="#555",
                        capsize=2, lw=1.0, alpha=0.7)
        ax_bar.axhline(0, color="#333", lw=0.8)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
        ax_bar.set_ylabel("Mean α (raw attention)", fontsize=FS_LABEL)
        ax_bar.tick_params(labelsize=FS_TICK)
        ax_bar.set_facecolor(BG)
        ax_bar.spines[["top", "right"]].set_visible(False)
        ax_bar.grid(axis="y", alpha=0.2, lw=0.6)

        # Modality color strip
        ax_mod.bar(x, np.ones(min_len), color=seed_colors, width=1.0)
        ax_mod.set_xlim(-0.5, min_len - 0.5)
        ax_mod.set_xticks([])
        ax_mod.set_yticks([])
        ax_mod.set_ylabel("Mod", fontsize=FS_TICK - 2, rotation=0, labelpad=22)
        ax_mod.set_facecolor(BG)

        # Legend
        patches = [mpatches.Patch(color=MOD_COLORS[m], label=m)
                   for m in MOD_ORDER if m in MOD_COLORS]
        ax_bar.legend(handles=patches, loc="upper right", fontsize=FS_LEGEND,
                      framealpha=0.7, title="Modality", title_fontsize=FS_LEGEND - 1)

        # Modality dividers
        prev_mod = None
        for xi, lbl in enumerate(labels):
            mod = lbl.split("·")[0]
            if mod != prev_mod and prev_mod is not None:
                ax_bar.axvline(xi - 0.5, color="#aaa", lw=0.7, ls="--", alpha=0.6)
            prev_mod = mod

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        for ext in ("png", "pdf"):
            fig.savefig(OUT_ROOT / f"seed_attn_per_seed_{task_key}.{ext}",
                        dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"Saved seed_attn_per_seed_{task_key}", flush=True)


def main():
    print("Loading seed attribution data...", flush=True)

    # Collect raw split data per task
    task_mod_data = {}
    for task_key, cfg in TASK_CFG.items():
        splits_data = load_task_data(task_key, cfg)
        task_mod_data[task_key] = splits_data
        print(f"  {task_key}: {len(splits_data)} splits", flush=True)

    print("Building modality-level table...", flush=True)
    task_list, mod_means, mod_stds = build_modality_table()

    print("Plotting grouped bar chart...", flush=True)
    plot_grouped_bars(task_list, mod_means, mod_stds)

    print("Plotting heatmap...", flush=True)
    plot_heatmap(task_list, mod_means, mod_stds)

    print("Plotting per-seed bars...", flush=True)
    plot_per_seed_bars(task_list, task_mod_data)

    print(f"\nDone. Figures → {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
