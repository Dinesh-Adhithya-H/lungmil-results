"""
Modality combination ablation — best model per task.
For each task, shows performance for all modality subsets that were tested
(single modalities + all 4 combined) across all model variants.
Shows which modality combinations drive performance.
Run via: sbatch analysis/submit_modality_combo_ablation.sh
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from itertools import combinations

ROOT    = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results" / "mm_abmil_v8"
OUT_DIR = ROOT / "figures" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG = "#FAF6F2"

TASK_CFG = {
    "acr_cls":   {"suffix": "cls",       "metric": "bacc",    "label": "ACR classification (BACC)",
                  "best_variant": "set_mil_mt_no_sab", "best_label": "SetMIL-MT (no SAB)"},
    "acr_surv":  {"suffix": "acr_surv",  "metric": "c_index", "label": "ACR survival (C-index)",
                  "best_variant": "longitudinal_mk_no_alibi", "best_label": "LongMK"},
    "clad_surv": {"suffix": "clad_surv", "metric": "c_index", "label": "CLAD survival (C-index)",
                  "best_variant": "set_mil_mt", "best_label": "SetMIL-MT"},
    "death_surv":{"suffix": "death_surv","metric": "c_index", "label": "Death survival (C-index)",
                  "best_variant": "longitudinal_mk_no_alibi", "best_label": "LongMK"},
}

ALL_VARIANTS = {
    "early":                       "Early fusion",
    "middle":                      "Middle fusion",
    "late":                        "Late fusion",
    "set_mil_no_sab":              "SetMIL",
    "set_mil_mt":                  "SetMIL-MT",
    "set_mil_mt_no_sab":           "SetMIL-MT (no SAB)",
    "longitudinal_mk_mt_no_alibi": "LongMK-MT",
    "longitudinal_mk_no_alibi":    "LongMK",
}

MOD_ORDER = ["HE", "BAL", "CT", "Clinical"]
MOD_COLORS = {"HE": "#E64A19", "BAL": "#1565C0", "CT": "#2E7D32", "Clinical": "#9b59b6"}

COMBO_LABELS = {
    frozenset(["HE"]):                              "HE",
    frozenset(["BAL"]):                             "BAL",
    frozenset(["CT"]):                              "CT",
    frozenset(["Clinical"]):                        "Clinical",
    frozenset(["HE", "BAL"]):                       "HE+BAL",
    frozenset(["HE", "CT"]):                        "HE+CT",
    frozenset(["HE", "Clinical"]):                  "HE+Clin",
    frozenset(["BAL", "CT"]):                       "BAL+CT",
    frozenset(["BAL", "Clinical"]):                 "BAL+Clin",
    frozenset(["CT", "Clinical"]):                  "CT+Clin",
    frozenset(["HE", "BAL", "CT"]):                 "HE+BAL+CT",
    frozenset(["HE", "BAL", "Clinical"]):           "HE+BAL+Clin",
    frozenset(["HE", "CT", "Clinical"]):            "HE+CT+Clin",
    frozenset(["BAL", "CT", "Clinical"]):           "BAL+CT+Clin",
    frozenset(["HE", "BAL", "CT", "Clinical"]):     "All 4",
}

SINGLE_MODS = [frozenset([m]) for m in MOD_ORDER]
COMBO_ORDER = (
    SINGLE_MODS +
    [frozenset(c) for c in combinations(MOD_ORDER, 2)] +
    [frozenset(c) for c in combinations(MOD_ORDER, 3)] +
    [frozenset(MOD_ORDER)]
)


def load_unimodal_per_split(variant, suffix, metric):
    per_mod = {m: [] for m in MOD_ORDER}
    all_vals = []
    for split in range(5):
        path = METRICS / f"metrics_split{split}_fold0_{variant}_{suffix}.json"
        if not path.exists():
            continue
        try:
            d = json.loads(path.read_text())
            ua = d.get("unimodal_ablation", {})
            for mod in MOD_ORDER:
                v = ua.get(mod, {}).get(metric)
                per_mod[mod].append(float(v) if v is not None else np.nan)
            # All-modality from test
            test = d.get("test", {})
            v_all = test.get(metric)
            if v_all is None and isinstance(test, dict):
                for k, sub in test.items():
                    if isinstance(sub, dict):
                        v_all = sub.get(metric)
                        if v_all is not None:
                            break
            all_vals.append(float(v_all) if v_all is not None else np.nan)
        except Exception:
            continue
    return per_mod, all_vals


def plot_combo_for_task(task_key, cfg):
    n_variants = len(ALL_VARIANTS)
    model_colors = plt.cm.tab20(np.linspace(0, 1, n_variants))
    variant_list = list(ALL_VARIANTS.items())

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bar_w = 0.7 / n_variants
    # Only plot: 4 single mods + all 4 combined (5 x-positions)
    x_labels = [COMBO_LABELS[c] for c in SINGLE_MODS] + ["All 4"]
    n_x = len(x_labels)
    x = np.arange(n_x)

    for vi, (variant, disp) in enumerate(variant_list):
        offset = (vi - n_variants / 2 + 0.5) * bar_w
        per_mod, all_vals = load_unimodal_per_split(variant, cfg["suffix"], cfg["metric"])
        col = model_colors[vi]

        for xi, mod in enumerate(MOD_ORDER):
            vals = per_mod[mod]
            valid = [v for v in vals if not np.isnan(v)]
            if not valid:
                continue
            mean = np.nanmean(vals)
            std  = np.nanstd(vals)
            ax.bar(x[xi] + offset, mean, width=bar_w * 0.85, color=col, alpha=0.85, zorder=2)
            if std > 0:
                ax.errorbar(x[xi] + offset, mean, yerr=std,
                            fmt="none", ecolor="#222", elinewidth=0.6, capsize=1.5, zorder=3)
            for sv in vals:
                if not np.isnan(sv):
                    ax.scatter(x[xi] + offset, sv, s=5, color="white",
                               edgecolors=col, linewidths=0.5, zorder=4, alpha=0.85)

        # All-4 column
        valid = [v for v in all_vals if not np.isnan(v)]
        if valid:
            mean = np.nanmean(all_vals)
            std  = np.nanstd(all_vals)
            ax.bar(x[-1] + offset, mean, width=bar_w * 0.85, color=col, alpha=0.85, zorder=2)
            if std > 0:
                ax.errorbar(x[-1] + offset, mean, yerr=std,
                            fmt="none", ecolor="#222", elinewidth=0.6, capsize=1.5, zorder=3)
            for sv in all_vals:
                if not np.isnan(sv):
                    ax.scatter(x[-1] + offset, sv, s=5, color="white",
                               edgecolors=col, linewidths=0.5, zorder=4, alpha=0.85)

    # separator line before "All 4"
    ax.axvline(n_x - 1 - 0.5, color="#aaa", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0.5, color="#999", linewidth=0.7, linestyle=":", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10, fontweight="bold")
    ax.set_ylabel(cfg["metric"].upper(), fontsize=9)
    ax.set_title(f"Modality contribution — {cfg['label']}", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(labelsize=8)

    legend_handles = [Patch(facecolor=model_colors[i], label=disp)
                      for i, (_, disp) in enumerate(variant_list)]
    ax.legend(handles=legend_handles, fontsize=6.5, ncol=3, framealpha=0.85,
              loc="lower right", edgecolor="#ccc")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"modality_combo_{task_key}.{ext}", dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved modality_combo_{task_key}")


def plot_combined():
    fig, axes = plt.subplots(2, 2, figsize=(26, 12), facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.suptitle("Modality contribution (unimodal → all) — all models, all tasks",
                 fontsize=12, fontweight="bold")
    for ax, (task_key, cfg) in zip(axes.flat, TASK_CFG.items()):
        ax.set_facecolor(BG)
        n_variants = len(ALL_VARIANTS)
        model_colors = plt.cm.tab20(np.linspace(0, 1, n_variants))
        bar_w = 0.7 / n_variants
        x_labels = [COMBO_LABELS[c] for c in SINGLE_MODS] + ["All 4"]
        x = np.arange(len(x_labels))

        for vi, (variant, disp) in enumerate(ALL_VARIANTS.items()):
            offset = (vi - n_variants / 2 + 0.5) * bar_w
            per_mod, all_vals = load_unimodal_per_split(variant, cfg["suffix"], cfg["metric"])
            col = model_colors[vi]
            for xi, mod in enumerate(MOD_ORDER):
                vals = per_mod[mod]
                valid = [v for v in vals if not np.isnan(v)]
                if not valid:
                    continue
                ax.bar(x[xi] + offset, np.nanmean(vals), width=bar_w * 0.85, color=col, alpha=0.85, zorder=2)
            valid = [v for v in all_vals if not np.isnan(v)]
            if valid:
                ax.bar(x[-1] + offset, np.nanmean(all_vals), width=bar_w * 0.85, color=col, alpha=0.85, zorder=2)

        ax.axvline(len(x) - 1 - 0.5, color="#aaa", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(0.5, color="#999", linewidth=0.7, linestyle=":", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8, fontweight="bold")
        ax.set_title(cfg["label"], fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=7)

    legend_handles = [Patch(facecolor=plt.cm.tab20(i / len(ALL_VARIANTS)), label=disp)
                      for i, (_, disp) in enumerate(ALL_VARIANTS.items())]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"modality_combo_all.{ext}", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  Saved modality_combo_all")


if __name__ == "__main__":
    for task_key, cfg in TASK_CFG.items():
        print(f"\n=== {task_key} ===")
        plot_combo_for_task(task_key, cfg)
    print("\n=== Combined ===")
    plot_combined()
    print("Done.")
