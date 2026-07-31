"""
Cross-split aggregated cluster attribution figures.

For each task × modality, computes:
  Δaff[cluster] = mean_aff(high-risk) - mean_aff(low-risk)   (mean over ALL seeds, over ALL splits)

Produces a clean bar chart per task showing which biological clusters
distinguish high-risk from low-risk patients, with biological labels.

Reads: interpretability/longitudinal_mk_interp/{variant}_split{s}_fold0_{task}/cluster_aff_data_{task_key}.json
Writes: figures/interpretability/cluster_agg/{variant}_{task}_cluster_aff_agg.png
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path("/home/aih/dinesh.haridoss/chicago_mil")
INTERP_ROOT = REPO / "interpretability" / "longitudinal_mk_interp"
OUT_FIGURES  = REPO / "figures" / "interpretability" / "cluster_agg"
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

VARIANTS = ["longitudinal_mk_no_alibi", "longitudinal_mk_mt_no_alibi"]

TASK_META = {
    "acr_cls":  {"dir_suffix": "acr_cls",   "json_key": "acr_cls",
                 "label": "ACR classification (ACR+ vs ACR−)",
                 "hi_lbl": "ACR+",  "lo_lbl": "ACR−"},
    "acr_surv": {"dir_suffix": "acr_surv",  "json_key": "acr_surv",
                 "label": "ACR survival (shorter vs longer TTE)",
                 "hi_lbl": "High risk (short TTE)", "lo_lbl": "Low risk (long TTE)"},
    "clad":     {"dir_suffix": "clad_surv", "json_key": "clad",
                 "label": "CLAD survival (shorter vs longer TTE)",
                 "hi_lbl": "High risk (short TTE)", "lo_lbl": "Low risk (long TTE)"},
    "death":    {"dir_suffix": "death_surv","json_key": "death",
                 "label": "Mortality survival (shorter vs longer TTE)",
                 "hi_lbl": "High risk (shorter survival)", "lo_lbl": "Low risk (longer survival)"},
}

# Biological category colours for HE
HE_BIO_COLORS = {
    "Alveolar with hemorrhage and inflammation": "#D32F2F",
    "Alveolar with empty spaces":                "#F57C00",
    "Alveolar":                                  "#388E3C",
    "Bronchial":                                 "#1565C0",
    "Lymphocytoplasmic inflammation":            "#6A1B9A",
    "Cartilage":                                 "#795548",
    "Unknown":                                   "#9E9E9E",
}

HE_BIO_MAP_FILE = REPO / "results" / "cluster_name_maps" / "HE_cluster_map.json"
HE_BIO_MAP = json.loads(HE_BIO_MAP_FILE.read_text()) if HE_BIO_MAP_FILE.exists() else {}

MOD_COLORS = {"HE": "#E64A19", "BAL": "#1565C0", "CT": "#2E7D32", "Clinical": "#6A1B9A"}

FONT = 10


def _savefig(fig, path):
    fig.savefig(str(path) + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(str(path) + ".pdf", dpi=200, bbox_inches="tight")
    return Path(str(path) + ".png")


def load_cluster_aff(variant, task_key, dir_suffix, n_splits=5):
    """
    Load cluster_aff_data_{task_key}.json for all 5 splits.
    Returns list of dicts, one per split.
    """
    records = []
    for split in range(n_splits):
        cand = INTERP_ROOT / f"{variant}_split{split}_fold0_{dir_suffix}"
        jpath = cand / f"cluster_aff_data_{task_key}.json"
        if jpath.exists():
            records.append(json.loads(jpath.read_text()))
        else:
            print(f"  [load] missing: {jpath}")
    return records


def aggregate_cluster_aff(records, task_label):
    """
    For each modality × cluster, compute:
      Δaff = mean_over_seeds(hi_aff) - mean_over_seeds(lo_aff)
    averaged across splits, with std.

    Returns dict: {mod: {"names": [...], "delta_mean": array, "delta_std": array}}
    """
    if not records:
        return {}

    present_mods = records[0]["present_mods"]
    result = {}

    for mo in present_mods:
        split_deltas = []
        clust_names = None
        for rec in records:
            caff = rec.get("cluster_aff", {}).get(mo)
            if caff is None:
                continue
            names = caff.get("names", [])
            hi = caff.get("hi")
            lo = caff.get("lo")
            if hi is None or lo is None:
                continue
            hi_arr = np.array(hi)   # (K, n_c)
            lo_arr = np.array(lo)   # (K, n_c)
            # Mean over seeds → (n_c,)
            delta = hi_arr.mean(0) - lo_arr.mean(0)
            split_deltas.append(delta)
            if clust_names is None:
                clust_names = names

        if not split_deltas:
            continue

        min_len = min(d.shape[0] for d in split_deltas)
        arr = np.stack([d[:min_len] for d in split_deltas])  # (n_splits, n_c)
        result[mo] = {
            "names":      (clust_names or [str(i) for i in range(min_len)])[:min_len],
            "delta_mean": arr.mean(0),
            "delta_std":  arr.std(0),
            "n_splits":   len(split_deltas),
        }

    return result


def plot_cluster_agg(agg, task_label, hi_lbl, lo_lbl, out_stem, top_n=15):
    """
    One panel per modality. Bar chart: Δaff per cluster with error bars.
    Red = enriched in high-risk; blue = enriched in low-risk.
    For HE: colour-code bars by biological category.
    """
    mods = [m for m in agg if m != "Clinical"]   # Clinical clusters not biologically labelled
    if not mods:
        print(f"  [plot] no modality data, skipping")
        return

    fig, axes = plt.subplots(1, len(mods), figsize=(6 * len(mods), 5.5))
    if len(mods) == 1:
        axes = [axes]

    for ax, mo in zip(axes, mods):
        d = agg[mo]
        delta  = d["delta_mean"]
        err    = d["delta_std"]
        names  = d["names"]
        n_splt = d["n_splits"]

        # Sort by |delta|, take top_n
        order = np.argsort(np.abs(delta))[::-1][:top_n]
        delta_s = delta[order]
        err_s   = err[order]
        names_s = [names[i] for i in order]

        # Bar colours
        if mo == "HE":
            bar_cols = []
            for nm in names_s:
                cat = HE_BIO_MAP.get(nm, "Unknown")
                base = HE_BIO_COLORS.get(cat, "#9E9E9E")
                # Tint by direction
                if delta_s[list(names_s).index(nm)] > 0:
                    bar_cols.append("#E53935")
                else:
                    bar_cols.append("#1E88E5")
        else:
            bar_cols = ["#E53935" if v > 0 else "#1E88E5" for v in delta_s]

        x = np.arange(len(delta_s))
        ax.barh(x, delta_s[::-1], color=bar_cols[::-1], height=0.7, alpha=0.85)
        ax.errorbar(x[::-1] + 0.0, delta_s[::-1], xerr=err_s[::-1],
                    fmt="none", color="#333", capsize=2, lw=0.9, alpha=0.7)
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels([nm[:35] for nm in names_s[::-1]], fontsize=7)
        ax.set_xlabel("Δ cluster affinity (high-risk − low-risk)", fontsize=FONT - 1)
        ax.set_title(f"{mo}\n{hi_lbl} ← | → {lo_lbl}", fontsize=FONT, fontweight="bold",
                     color=MOD_COLORS.get(mo, "#333"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.98, 0.02, f"n={n_splt} splits", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color="#888")

    # Add HE legend
    if "HE" in mods:
        from matplotlib.patches import Patch
        legend_elems = [Patch(facecolor="#E53935", label="Enriched in high-risk"),
                        Patch(facecolor="#1E88E5", label="Enriched in low-risk")]
        axes[0].legend(handles=legend_elems, fontsize=7, loc="lower right")

    fig.suptitle(
        f"{task_label}\nCluster attribution: Δ mean affinity (high-risk − low-risk), top {top_n}",
        fontsize=FONT + 1, fontweight="bold", y=1.01)
    fig.tight_layout()
    png = _savefig(fig, out_stem)
    plt.close(fig)
    print(f"  [plot] {png.name}")
    return png


def run_all():
    for variant in VARIANTS:
        print(f"\n=== {variant} ===")
        for task_key, meta in TASK_META.items():
            print(f"  task={task_key}")
            records = load_cluster_aff(variant, meta["json_key"], meta["dir_suffix"])
            if len(records) < 2:
                print(f"  [skip] only {len(records)} split(s) with data")
                continue
            agg = aggregate_cluster_aff(records, meta["label"])
            if not agg:
                print(f"  [skip] no cluster affinity data found")
                continue
            stem = OUT_FIGURES / f"{variant}_{task_key}_cluster_aff_agg"
            plot_cluster_agg(agg, meta["label"], meta["hi_lbl"], meta["lo_lbl"], stem)

    print("\n[done]")


if __name__ == "__main__":
    run_all()
