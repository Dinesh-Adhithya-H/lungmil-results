"""
Cross-split aggregated cluster-level attribution for the best model per task.

Best models (benchmark):
  ACR classification -> set_mil_mt_no_sab  (BACC 0.623)
  ACR survival       -> longitudinal_mk_no_alibi  (C-index 0.679)
  CLAD survival      -> set_mil_mt         (C-index 0.563)
  Death survival     -> longitudinal_mk_no_alibi  (C-index 0.771)

For set_mil models: reads paper_interp_data.json (already has cluster delta).
For longitudinal models: reads cluster_aff_data_{task}.json (from re-inference jobs).

Output: figures/interpretability/cluster_agg/{task}_cluster_aff_agg.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO        = Path("/home/aih/dinesh.haridoss/chicago_mil")
LMK_ROOT    = REPO / "interpretability" / "longitudinal_mk_interp"
SMT_ROOT    = REPO / "interpretability" / "set_mil_mt_interp"
OUT_FIGURES = REPO / "figures" / "interpretability" / "cluster_agg"
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

HE_BIO_MAP_FILE = REPO / "results" / "cluster_name_maps" / "HE_cluster_map.json"
HE_BIO_MAP = json.loads(HE_BIO_MAP_FILE.read_text()) if HE_BIO_MAP_FILE.exists() else {}

HE_BIO_COLORS = {
    "Alveolar with hemorrhage and inflammation": "#C62828",
    "Alveolar with empty spaces":               "#EF6C00",
    "Alveolar":                                 "#2E7D32",
    "Bronchial":                                "#1565C0",
    "Lymphocytoplasmic inflammation":           "#6A1B9A",
    "Cartilage":                                "#795548",
    "Unknown":                                  "#9E9E9E",
}
MOD_COLORS = {"HE": "#E64A19", "BAL": "#1565C0", "CT": "#2E7D32", "Clinical": "#9b59b6"}
FONT = 12

# Plain-language display names for model variants
MODEL_DISPLAY = {
    "longitudinal_mk_no_alibi": "LongMK (longitudinal multimodal attention)",
    "set_mil_mt_no_sab":        "SetMIL-MT (multimodal set-based MIL)",
    "set_mil_mt":               "SetMIL-MT (multimodal set-based MIL)",
}

# Clinical variable name expansions
CLINICAL_NAME_MAP = {
    "fvc":        "FVC (forced vital capacity, L)",
    "fvcp":       "FVC% (% predicted)",
    "fev1":       "FEV₁ (L)",
    "fev1p":      "FEV₁% (% predicted)",
    "pgd_t72":    "PGD score at 72h",
    "donor_risk": "Donor risk score",
    "mpv":        "MPV (mean platelet volume)",
    "rdw":        "RDW (red cell distribution width)",
    "hct":        "HCT (haematocrit)",
    "gfr":        "GFR (glomerular filtration rate)",
    "albumin":    "Albumin (g/dL)",
    "MVV":        "MVV",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.labelsize":   FONT,
    "axes.titlesize":   FONT + 1,
    "xtick.labelsize":  FONT - 2,
    "ytick.labelsize":  FONT - 2,
    "legend.fontsize":  FONT - 2,
    "figure.dpi":       150,
})

TASK_CONFIG = {
    "acr_cls": {
        "model_family": "set_mil",
        "variant":      "set_mil_mt_no_sab",
        "dir_pattern":  "set_mil_mt_no_sab_split{s}_fold0_cls",
        "json_task":    "acr_cls",
        "label":        "ACR Classification  (ACR+ vs ACR-)",
        "hi_lbl":       "ACR+",
        "lo_lbl":       "ACR-",
        "performance":  "BACC 0.623 +/- 0.034",
    },
    "acr_surv": {
        "model_family": "longitudinal",
        "variant":      "longitudinal_mk_no_alibi",
        "dir_pattern":  "longitudinal_mk_no_alibi_split{s}_fold0_acr_surv",
        "json_key":     "acr_surv",
        "label":        "ACR Survival  (time to next rejection)",
        "hi_lbl":       "High risk (short TTE)",
        "lo_lbl":       "Low risk (long TTE)",
        "performance":  "C-index 0.679 +/- 0.064",
    },
    "clad": {
        "model_family": "set_mil",
        "variant":      "set_mil_mt",
        "dir_pattern":  "set_mil_mt_split{s}_fold0_clad_surv",
        "json_task":    "clad",
        "label":        "CLAD Survival  (time to CLAD onset)",
        "hi_lbl":       "High risk (short TTE)",
        "lo_lbl":       "Low risk (long TTE)",
        "performance":  "C-index 0.563 +/- 0.080",
    },
    "death": {
        "model_family": "longitudinal",
        "variant":      "longitudinal_mk_no_alibi",
        "dir_pattern":  "longitudinal_mk_no_alibi_split{s}_fold0_death_surv",
        "json_key":     "death",
        "label":        "Mortality  (post-transplant survival)",
        "hi_lbl":       "Non-survivor (short survival)",
        "lo_lbl":       "Long-term survivor",
        "performance":  "C-index 0.771 +/- 0.056",
    },
}


def load_set_mil(cfg, n_splits=5):
    result = {}
    task_key = cfg["json_task"]
    for split in range(n_splits):
        dpath = SMT_ROOT / cfg["dir_pattern"].format(s=split) / "paper_interp_data.json"
        if not dpath.exists():
            print(f"  [load] missing: {dpath}")
            continue
        d = json.loads(dpath.read_text())
        task_data = d.get("tasks", {}).get(task_key, {})
        ca = task_data.get("cluster_affinity", {})
        for mod, mdata in ca.items():
            delta = np.array(mdata.get("delta", []))
            names = mdata.get("cluster_names", [])
            if len(delta) == 0:
                continue
            if mod not in result:
                result[mod] = {"names": names, "deltas": []}
            result[mod]["deltas"].append(delta)
    return result


def load_longitudinal(cfg, n_splits=5):
    result = {}
    json_key = cfg["json_key"]
    for split in range(n_splits):
        dpath = LMK_ROOT / cfg["dir_pattern"].format(s=split) / f"cluster_aff_data_{json_key}.json"
        if not dpath.exists():
            print(f"  [load] missing: {dpath}")
            continue
        d = json.loads(dpath.read_text())
        caff = d.get("cluster_aff", {})
        for mod, mdata in caff.items():
            hi    = mdata.get("hi")
            lo    = mdata.get("lo")
            names = mdata.get("names", [])
            # Clinical delta may be stored directly (no hi/lo if extracted differently)
            delta_direct = mdata.get("delta")
            if delta_direct is not None:
                delta = np.array(delta_direct)
            elif hi is not None and lo is not None:
                hi_arr = np.array(hi)
                lo_arr = np.array(lo)
                delta = hi_arr.mean(0) - lo_arr.mean(0)
            else:
                continue
            if mod not in result:
                result[mod] = {"names": names, "deltas": []}
            result[mod]["deltas"].append(delta)
    return result


def aggregate(data_by_mod):
    agg = {}
    for mod, info in data_by_mod.items():
        deltas = info["deltas"]
        names = info["names"]
        if len(deltas) < 2:
            print(f"  [agg] {mod}: only {len(deltas)} splits, skipping")
            continue
        min_len = min(d.shape[0] for d in deltas)
        arr = np.stack([d[:min_len] for d in deltas])
        agg[mod] = {
            "names":      names[:min_len],
            "delta_mean": arr.mean(0),
            "delta_std":  arr.std(0),
            "n_splits":   len(deltas),
        }
    return agg


def bio_label(name, mod):
    """Return expanded label for HE (bio category) and Clinical (full name); raw otherwise."""
    if mod == "HE":
        cat = HE_BIO_MAP.get(name, "Unknown")
        return f"{cat} ({name})"
    if mod == "Clinical":
        return CLINICAL_NAME_MAP.get(name, CLINICAL_NAME_MAP.get(name.lower(), name.upper()))
    return name


def _fmt_sci(v):
    """Format a small float as e.g. 1.2e−4 for bar annotation."""
    if v == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(v))))
    mant = v / 10 ** exp
    return f"{mant:.1f}×10$^{{{exp}}}$"


def plot_task(agg, cfg, task_key, top_n=14):
    mods = [m for m in ["HE", "BAL", "CT", "Clinical"] if m in agg]
    if not mods:
        print(f"  [plot] {task_key}: no modality data")
        return

    n_cols = len(mods)
    panel_w = {"HE": 8.5, "BAL": 8.5, "CT": 6.5, "Clinical": 8.0}
    total_w = sum(panel_w.get(m, 7.5) for m in mods)
    fig, axes = plt.subplots(1, n_cols, figsize=(total_w, 10))
    if n_cols == 1:
        axes = [axes]

    # Shared x-axis limit across all modalities
    global_abs_max = 0.0
    for mod in mods:
        d = agg[mod]
        order = np.argsort(np.abs(d["delta_mean"]))[::-1][:top_n]
        vals = d["delta_mean"][order]
        errs = d["delta_std"][order]
        global_abs_max = max(global_abs_max, np.max(np.abs(vals) + errs))
    shared_xlim = global_abs_max * 1.15  # 15% padding

    for ax, mod in zip(axes, mods):
        d = agg[mod]
        delta = d["delta_mean"]
        err   = d["delta_std"]
        names = d["names"]
        n_sp  = d["n_splits"]

        bio_labels = [bio_label(nm, mod) for nm in names]

        order   = np.argsort(np.abs(delta))[::-1][:top_n]
        delta_s = delta[order]
        err_s   = err[order]
        labs_s  = [bio_labels[i] for i in order]

        x = np.arange(len(delta_s))
        bar_cols = ["#C62828" if v > 0 else "#1565C0" for v in delta_s]

        bars = ax.barh(x, delta_s, color=bar_cols, height=0.65, alpha=0.88,
                       edgecolor="white", linewidth=0.4)
        ax.errorbar(delta_s, x, xerr=err_s, fmt="none",
                    color="#222", capsize=3, capthick=0.9, lw=1.0, alpha=0.75)
        ax.axvline(0, color="#444", lw=1.1)

        # Light x-grid for readability
        ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color="#ccc", zorder=0)
        ax.set_axisbelow(True)

        # Y-tick labels: wrap long names at 32 chars
        def _wrap(s, n=32):
            if len(s) <= n:
                return s
            # break at last space before n
            cut = s.rfind(" ", 0, n)
            return s[:cut] + "\n" + s[cut + 1:] if cut > 0 else s[:n] + "-\n" + s[n:]

        ax.set_yticks(x)
        ax.set_yticklabels([_wrap(lb) for lb in labs_s], fontsize=FONT - 3)
        ax.tick_params(axis="y", pad=2)

        ax.set_xlabel("Attribution score  (positive = enriched in high-risk group)", fontsize=FONT - 1)
        ax.set_title(
            f"{mod}  [top {len(order)} clusters]\n"
            f"← {cfg['lo_lbl']}  |  {cfg['hi_lbl']} →",
            fontsize=FONT, fontweight="bold",
            color=MOD_COLORS.get(mod, "#333"), pad=8)
        ax.text(0.98, 0.01, f"n={n_sp} splits", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=FONT - 2, color="#555")
        ax.set_xlim(-shared_xlim, shared_xlim)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#C62828", alpha=0.88, label=f"Enriched in {cfg['hi_lbl']}"),
        Patch(facecolor="#1565C0", alpha=0.88, label=f"Enriched in {cfg['lo_lbl']}"),
    ]
    for ax in axes:
        ax.legend(handles=legend_elems, fontsize=FONT - 3, loc="lower right",
                  framealpha=0.85, edgecolor="#ccc")

    model_disp = MODEL_DISPLAY.get(cfg["variant"], cfg["variant"])
    fig.suptitle(
        f"{cfg['label']}\n"
        f"Model: {model_disp}  [{cfg['performance']}]  |  "
        f"Mean ± s.d. across {agg[mods[0]]['n_splits']} cross-validation splits",
        fontsize=FONT + 2, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 1])

    stem = OUT_FIGURES / f"{task_key}_cluster_aff_agg"
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {stem}.png")


def run_all():
    for task_key, cfg in TASK_CONFIG.items():
        print(f"\n=== {task_key} / {cfg['variant']} ===")
        if cfg["model_family"] == "set_mil":
            data = load_set_mil(cfg)
        else:
            data = load_longitudinal(cfg)

        if not data:
            print(f"  [skip] no data found")
            continue

        agg = aggregate(data)
        if not agg:
            print(f"  [skip] aggregation empty")
            continue

        plot_task(agg, cfg, task_key)

    print("\n[done]")


if __name__ == "__main__":
    run_all()
