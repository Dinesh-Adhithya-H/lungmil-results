"""
Unimodal ablation — fixed model order, all models, all tasks.
For each task: grouped bars where each group = one modality,
bars within = model variants, showing per-split dots + mean±std.
Also produces per-task heatmap (models × modalities).
Run via: sbatch analysis/submit_unimodal_ablation_v2.sh
"""
from pathlib import Path
import json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

ROOT      = Path(__file__).resolve().parent.parent
METRICS   = ROOT / "results" / "mm_abmil_v8"
LIN_CSV   = ROOT / "results" / "linear_models" / "metrics_summary.csv"
ABL_CSV   = ROOT / "interpretability" / "unimodal_ablation" / "unimodal_ablation_summary.csv"
OUT_DIR   = ROOT / "figures" / "interpretability" / "unimodal_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG = "#FAF6F2"

# Shared per-model colors — identical across benchmark, unimodal ablation, and combo plots
SHARED_MODEL_COLORS = {
    "Linear HE":          "#BDBDBD",
    "Linear BAL":         "#9E9E9E",
    "Linear CT":          "#757575",
    "Linear Clinical":    "#616161",
    "wt avg Linear":     "#424242",
    "ABMIL HE":           "#90CAF9",
    "ABMIL BAL":          "#42A5F5",
    "ABMIL CT":           "#1976D2",
    "ABMIL Clinical":     "#1565C0",
    "wt avg ABMIL":      "#0D47A1",
    "Early fusion":       "#80CBC4",
    "Middle fusion":      "#26A69A",
    "Late fusion":        "#00796B",
    "SetMIL":             "#CE93D8",
    "SetMIL-MT":          "#9C27B0",
    "SetMIL-MT (no SAB)": "#6A1B9A",
    "LongMK-MT":          "#EF9A9A",
    "LongMK":             "#C62828",
    "Linear":             "#9E9E9E",  # aggregated linear (unimodal ablation only)
}

# Fixed model order (same as benchmark)
MODEL_ORDER = [
    "Linear",
    "Early fusion", "Middle fusion", "Late fusion",
    "SetMIL", "SetMIL-MT", "SetMIL-MT (no SAB)",
    "LongMK-MT", "LongMK",
]
# Internal variant name → display label
VARIANT_LABELS = {
    "early":                       "Early fusion",
    "middle":                      "Middle fusion",
    "late":                        "Late fusion",
    "set_mil_no_sab":              "SetMIL",
    "set_mil_mt":                  "SetMIL-MT",
    "set_mil_mt_no_sab":           "SetMIL-MT (no SAB)",
    "longitudinal_mk_mt_no_alibi": "LongMK-MT",
    "longitudinal_mk_no_alibi":    "LongMK",
}

MOD_ORDER  = ["HE", "BAL", "CT", "Clinical"]
MOD_COLORS = {"HE": "#E64A19", "BAL": "#1565C0", "CT": "#2E7D32", "Clinical": "#9b59b6"}

TASK_CFG = {
    "acr_cls":   {"suffix": "cls",       "metric": "bacc",    "label": "ACR cls (BACC)",
                  "lin_task": "ACR",    "lin_metric": "bacc"},
    "acr_surv":  {"suffix": "acr_surv",  "metric": "c_index", "label": "ACR surv (C-index)",
                  "lin_task": "ACR_TTE","lin_metric": "cindex"},
    "clad_surv": {"suffix": "clad_surv", "metric": "c_index", "label": "CLAD (C-index)",
                  "lin_task": "CLAD",   "lin_metric": "cindex"},
    "death_surv":{"suffix": "death_surv","metric": "c_index", "label": "Death (C-index)",
                  "lin_task": "Death",  "lin_metric": "cindex"},
}


def load_linear_unimodal(lin_task, lin_metric):
    df = pd.read_csv(LIN_CSV)
    df = df[df["task"] == lin_task].copy()
    MOD_MAP = {"H&E": "HE", "BAL": "BAL", "CT": "CT", "Clinical": "Clinical"}
    out = {}
    for lin_mod, mod in MOD_MAP.items():
        rows = df[df["modality"] == lin_mod]
        vals = []
        for _, r in rows.iterrows():
            try:
                vals.append(float(r.get(lin_metric, np.nan)))
            except (TypeError, ValueError):
                vals.append(np.nan)
        if vals:
            out[mod] = {"mean": np.nanmean(vals), "std": np.nanstd(vals), "splits": vals}
    return out


def load_dl_unimodal(suffix, metric):
    csv_df = pd.read_csv(ABL_CSV) if ABL_CSV.exists() else pd.DataFrame()
    data = {}
    for variant, disp in VARIANT_LABELS.items():
        per_split = {mod: [] for mod in MOD_ORDER}
        has_ua = False
        for split in range(5):
            path = METRICS / f"metrics_split{split}_fold0_{variant}_{suffix}.json"
            if not path.exists():
                continue
            try:
                d = json.loads(path.read_text())
                ua = d.get("unimodal_ablation", {})
                if ua:
                    has_ua = True
                for mod in MOD_ORDER:
                    v = ua.get(mod, {}).get(metric, np.nan)
                    per_split[mod].append(float(v) if v is not None else np.nan)
            except Exception:
                continue

        if not has_ua and not csv_df.empty:
            # JSON has no unimodal_ablation key (LongMK variants) — fall back to summary CSV
            rows = csv_df[(csv_df["variant"] == variant) &
                          (csv_df["task"] == suffix) &
                          (csv_df["metric"] == metric)]
            for _, r in rows.iterrows():
                mod = r["modality"]
                if mod not in MOD_ORDER:
                    continue
                if disp not in data:
                    data[disp] = {}
                data[disp][mod] = {"mean": float(r["mean"]), "std": float(r["std"]), "splits": []}
        else:
            for mod in MOD_ORDER:
                vals = per_split[mod]
                valid = [v for v in vals if not np.isnan(v)]
                if valid:
                    if disp not in data:
                        data[disp] = {}
                    data[disp][mod] = {"mean": np.nanmean(vals), "std": np.nanstd(vals), "splits": vals}
    return data


def plot_task_bars(task_key, cfg):
    lin_data = load_linear_unimodal(cfg["lin_task"], cfg["lin_metric"])
    dl_data  = load_dl_unimodal(cfg["suffix"], cfg["metric"])

    n_mods = len(MOD_ORDER)
    n_models = len(MODEL_ORDER)
    bar_w = 0.7 / n_models
    x = np.arange(n_mods)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for mi, mdl in enumerate(MODEL_ORDER):
        offset = (mi - n_models / 2 + 0.5) * bar_w
        for mi2, mod in enumerate(MOD_ORDER):
            if mdl == "Linear":
                d = lin_data.get(mod)
            else:
                d = dl_data.get(mdl, {}).get(mod)
            if d is None:
                continue
            col = SHARED_MODEL_COLORS.get(mdl, "#888888")
            ax.bar(x[mi2] + offset, d["mean"], width=bar_w * 0.85,
                   color=col, alpha=0.85, zorder=2)
            if d["std"] > 0:
                ax.errorbar(x[mi2] + offset, d["mean"], yerr=d["std"],
                            fmt="none", ecolor="#333", elinewidth=1.2, capsize=3, capthick=1.0, zorder=3)
            for sv in d["splits"]:
                if not np.isnan(sv):
                    ax.scatter(x[mi2] + offset, sv, s=8, color="white",
                               edgecolors=col, linewidths=0.6, zorder=4, alpha=0.9)
            # Mean±std text annotation above each bar
            txt = f"{d['mean']:.2f}±{d['std']:.2f}"
            ax.text(x[mi2] + offset, d["mean"] + d["std"] + 0.004, txt,
                    ha="center", va="bottom", fontsize=4.0, color="#333",
                    rotation=90, zorder=5)

    ax.axhline(0.5, color="#999", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(MOD_ORDER, fontsize=11, fontweight="bold")
    ax.set_ylabel(cfg["metric"].upper(), fontsize=9)
    ax.set_title(f"Unimodal ablation — {cfg['label']}", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(labelsize=8)

    legend_handles = [Patch(facecolor=SHARED_MODEL_COLORS.get(m, "#888"), label=m) for m in MODEL_ORDER]
    ax.legend(handles=legend_handles, fontsize=6.5, ncol=2, framealpha=0.85,
              loc="lower right", edgecolor="#ccc")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"unimodal_ablation_v2_{task_key}.{ext}", dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved unimodal_ablation_v2_{task_key}")


def plot_heatmap(task_key, cfg):
    lin_data = load_linear_unimodal(cfg["lin_task"], cfg["lin_metric"])
    dl_data  = load_dl_unimodal(cfg["suffix"], cfg["metric"])

    rows, row_labels = [], []
    for mdl in MODEL_ORDER:
        row = []
        for mod in MOD_ORDER:
            if mdl == "Linear":
                d = lin_data.get(mod)
            else:
                d = dl_data.get(mdl, {}).get(mod)
            row.append(d["mean"] if d else np.nan)
        rows.append(row)
        row_labels.append(mdl)

    mat = np.array(rows)
    fig, ax = plt.subplots(figsize=(6, 7), facecolor=BG)
    fig.patch.set_facecolor(BG)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=0.8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if 0.45 < v < 0.75 else "white")
    ax.set_xticks(range(len(MOD_ORDER)))
    ax.set_xticklabels(MOD_ORDER, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.6, label=cfg["metric"].upper())
    ax.set_title(f"Unimodal ablation heatmap — {cfg['label']}", fontsize=10, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"unimodal_ablation_v2_heatmap_{task_key}.{ext}", dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved unimodal_ablation_v2_heatmap_{task_key}")


def plot_combined():
    n_tasks = len(TASK_CFG)
    fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.suptitle("Unimodal ablation — all models, all tasks", fontsize=12, fontweight="bold")

    n_models = len(MODEL_ORDER)
    bar_w = 0.7 / n_models
    x = np.arange(len(MOD_ORDER))

    for ax, (task_key, cfg) in zip(axes, TASK_CFG.items()):
        ax.set_facecolor(BG)
        lin_data = load_linear_unimodal(cfg["lin_task"], cfg["lin_metric"])
        dl_data  = load_dl_unimodal(cfg["suffix"], cfg["metric"])
        for mi, mdl in enumerate(MODEL_ORDER):
            offset = (mi - n_models / 2 + 0.5) * bar_w
            for mi2, mod in enumerate(MOD_ORDER):
                d = lin_data.get(mod) if mdl == "Linear" else dl_data.get(mdl, {}).get(mod)
                if d is None:
                    continue
                col = SHARED_MODEL_COLORS.get(mdl, "#888888")
                ax.bar(x[mi2] + offset, d["mean"], width=bar_w * 0.85, color=col, alpha=0.85, zorder=2)
                if d["std"] > 0:
                    ax.errorbar(x[mi2] + offset, d["mean"], yerr=d["std"],
                                fmt="none", ecolor="#222", elinewidth=0.6, capsize=1.5, zorder=3)
        ax.axhline(0.5, color="#999", linewidth=0.7, linestyle=":", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(MOD_ORDER, fontsize=9, fontweight="bold")
        ax.set_title(cfg["label"], fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=7)

    legend_handles = [Patch(facecolor=SHARED_MODEL_COLORS.get(m, "#888"), label=m) for m in MODEL_ORDER]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"unimodal_ablation_v2_all.{ext}", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  Saved unimodal_ablation_v2_all")


if __name__ == "__main__":
    for task_key, cfg in TASK_CFG.items():
        print(f"\n=== {task_key} ===")
        plot_task_bars(task_key, cfg)
        plot_heatmap(task_key, cfg)
    print("\n=== Combined ===")
    plot_combined()
    print("Done.")
