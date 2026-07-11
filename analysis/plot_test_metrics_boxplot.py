"""
Boxplot of test-set metrics directly from metrics_*_final.json files.
No GPU inference needed — reads saved scalars only.

Three model groups per panel:
  1. P1 unimodal  — per-modality score on the subset of test samples that have
                    that modality (same subset logic as compare_modalities.py)
     + P1 wtd     — prevalence-weighted expected: Σ_m prevalence_m × metric_m
                    where prevalence_m = n_m / n_total (n_total = max n across mods)

  2. P2 full      — multimodal model test metrics from metrics_*_final.json

  3. P2 abl       — unimodal ablation of the P2 multimodal model (from
                    unimodal_ablation key in JSON): model run with one modality
                    at a time → per-modality metric
     + P2 abl wtd — prevalence-weighted expected: Σ_m n_m × metric_m / Σ_m n_m

4 panels: ACR Classification (BACC), ACR Survival (C-index), CLAD (C-index), Death (C-index)
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO     = Path(__file__).resolve().parent.parent
P2_DIR   = REPO / "results/mm_abmil_v8/phase2"
P1_DIR   = REPO / "results/mm_abmil_v8/phase1"
OUT_DIR  = REPO / "results/predictions/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS     = [0, 1, 2, 3, 4]
MODALITIES = ["HE", "BAL", "CT", "Clinical"]
MOD_COLORS = {"HE": "#4e79a7", "BAL": "#f28e2b", "CT": "#e15759", "Clinical": "#76b7b2"}

P2_VARIANTS = [
    ("early",           "Early"),
    ("late",            "Late"),
    ("middle",          "Middle"),
    ("set_mil",    "SlotAttn"),
    ("longitudinal_mk", "TimeSlot"),
]

# task folder suffix + metric path for standard P2 tasks
TASK_CFG = {
    "acr_cls": {
        "label":           "ACR Classification  (BACC)",
        "p1_folder":       "acr",
        "p1_metric":       "bacc",
        "p2_folder":       "cls",
        "p2_metric":       "bacc",
        "mega_sub":        "acr_cls",
        "mega_metric":     "bacc",
        "abl_metric":      "bacc",
        "mega_flat_key":     "bacc",      # flat key in fold1/3 mega JSONs (for full metric)
        # fold0 (fixed code): "bacc"  |  fold1 fallback: "bacc"
        "mega_abl_key":      "bacc",      # key in fold0 unimodal_ablation (new format)
        "mega_abl_flat_key": "bacc",      # key in fold1 unimodal_ablation (old format)
    },
    "acr_surv": {
        "label":             "ACR Survival  (C-index)",
        "p1_folder":         "acr_surv",
        "p1_metric":         "c_index",
        "p2_folder":         "acr_surv",
        "p2_metric":         "c_index",
        "mega_sub":          "acr_surv",
        "mega_metric":       "c_index",
        "abl_metric":        "c_index",
        "mega_flat_key":     "c_index",
        "mega_abl_key":      "acr_c_index",  # fold0 new format
        "mega_abl_flat_key": "c_index",      # fold1 fallback (surv_ep=acr → correct)
    },
    "clad": {
        "label":             "CLAD  (C-index)",
        "p1_folder":         "clad",
        "p1_metric":         "c_index",
        "p2_folder":         "clad_surv",
        "p2_metric":         "c_index",
        "mega_sub":          "clad",
        "mega_metric":       "c_index",
        "abl_metric":        "c_index",
        "mega_flat_key":     "clad_c_index",
        "mega_abl_key":      "clad_c_index",  # fold0 new format
        "mega_abl_flat_key": None,             # fold1 has no clad ablation → show —
    },
    "death": {
        "label":             "Death  (C-index)",
        "p1_folder":         "death",
        "p1_metric":         "c_index",
        "p2_folder":         "death_surv",
        "p2_metric":         "c_index",
        "mega_sub":          "death",
        "mega_metric":       "c_index",
        "abl_metric":        "c_index",
        "mega_flat_key":     "death_c_index",
        "mega_abl_key":      "death_c_index",  # fold0 new format
        "mega_abl_flat_key": None,              # fold1 has no death ablation → show —
    },
}

# Tracks which (var, split) pairs used a fallback fold (not fold0).
# Populated by load_p2_full(); used to annotate tables with †.
_FALLBACK_FOLDS = {}   # (var_internal, split) → fold_number


def get_nested(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    if isinstance(d, (int, float)) and not math.isnan(float(d)):
        return float(d)
    return None


# ── P1 unimodal ───────────────────────────────────────────────────────────────
def load_p1_per_modality(task_key):
    """Returns {mod: [val_s0..val_s4]}  — metric on modality-specific subset."""
    cfg = TASK_CFG[task_key]
    data = {}
    for mod in MODALITIES:
        vals = []
        for s in SPLITS:
            f = P1_DIR / f"split{s}_fold0" / cfg["p1_folder"] / mod / "final_combined" / "metrics.json"
            if not f.exists():
                vals.append(None); continue
            d = json.loads(f.read_text())
            vals.append(get_nested(d, "test", cfg["p1_metric"]))
        data[mod] = vals
    return data


def load_p1_weighted(task_key):
    """Prevalence-weighted expected P1 metric per split.
    weight_m = n_m / n_total  (n_total = max n_m across modalities, proxy for test set size)
    expected = Σ_m weight_m * metric_m  (normalised so weights sum to 1)
    """
    per_mod = load_p1_per_modality(task_key)
    cfg = TASK_CFG[task_key]

    # n_m per modality: read from ablation of the first available P2 model
    # We need prevalences — use P2 early ablation n values (they reflect full test set)
    def get_n(split, mod):
        for var, _ in P2_VARIANTS:
            folder = ("cls" if task_key == "acr_cls" else
                      "acr_surv" if task_key == "acr_surv" else
                      "clad_surv" if task_key == "clad" else "death_surv")
            if var in ("set_mil", "longitudinal_mk"):
                folder = "mega"
            f = P2_DIR / f"split{split}_fold0" / f"{var}_{folder}" / f"metrics_{var}_final.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            n = get_nested(d, "unimodal_ablation", mod, "n")
            if n is not None:
                return int(n)
        return None

    wtd = []
    for s in SPLITS:
        ns = {m: get_n(s, m) for m in MODALITIES}
        metrics = {m: per_mod[m][s] for m in MODALITIES}
        valid = [(m, ns[m], metrics[m]) for m in MODALITIES
                 if ns[m] is not None and metrics[m] is not None]
        if not valid:
            wtd.append(None); continue
        n_total = max(n for _, n, _ in valid)
        wsum = sum(n / n_total for _, n, _ in valid)
        wtd.append(sum((n / n_total) * v for _, n, v in valid) / wsum if wsum > 0 else None)
    return wtd


# ── P2 full multimodal ────────────────────────────────────────────────────────
def load_p2_full(task_key):
    """Returns {display_label: [val_s0..val_s4]}.

    For mega variants (set_mil, longitudinal_mk): tries fold0 with nested
    key structure first.  If fold0 is missing, falls back to the best available
    fold (1, 2, 3) which uses a flat key structure (mega_flat_key in TASK_CFG).
    Fallbacks are recorded in _FALLBACK_FOLDS for table annotation.
    """
    cfg = TASK_CFG[task_key]
    data = {}
    for var, lbl in P2_VARIANTS:
        vals = []
        for s in SPLITS:
            if var == "longitudinal_mk":
                # longitudinal_mk fold0 uses nested keys: test.acr_cls.bacc etc.
                f0 = P2_DIR / f"split{s}_fold0" / f"{var}_mega" / f"metrics_{var}_final.json"
                if f0.exists():
                    d = json.loads(f0.read_text())
                    v = get_nested(d, "test", cfg["mega_sub"], cfg["mega_metric"])
                    vals.append(v); continue
                vals.append(None)
            elif var == "set_mil":
                # set_mil (all folds) uses flat keys: test.bacc, test.clad_c_index etc.
                flat_key = cfg.get("mega_flat_key")
                f0 = P2_DIR / f"split{s}_fold0" / f"{var}_mega" / f"metrics_{var}_final.json"
                if f0.exists():
                    d = json.loads(f0.read_text())
                    v = get_nested(d, "test", flat_key) if flat_key else None
                    vals.append(v); continue

                # Fallback: try folds 1, 2, 3 (same flat format)
                fb_val = None; fb_fold = None
                for fb in [1, 2, 3]:
                    ff = P2_DIR / f"split{s}_fold{fb}" / f"{var}_mega" / f"metrics_{var}_final.json"
                    if ff.exists():
                        d = json.loads(ff.read_text())
                        v = get_nested(d, "test", flat_key) if flat_key else None
                        if v is not None:
                            fb_val = v; fb_fold = fb; break
                if fb_fold is not None:
                    _FALLBACK_FOLDS[(var, s)] = fb_fold
                vals.append(fb_val)
            else:
                f = P2_DIR / f"split{s}_fold0" / f"{var}_{cfg['p2_folder']}" / f"metrics_{var}_final.json"
                if not f.exists():
                    vals.append(None); continue
                d = json.loads(f.read_text())
                vals.append(get_nested(d, "test", cfg["p2_metric"]))
        data[lbl] = vals
    return data


# ── P2 unimodal ablation ──────────────────────────────────────────────────────
def load_p2_ablation(task_key):
    """Returns {(variant_label, mod): [val_s0..val_s4]}
    and {variant_label + ' wtd': [weighted_expected_s0..s4]}."""
    cfg = TASK_CFG[task_key]
    per_mod = {}   # (lbl, mod) → [s0..s4]
    wtd = {}       # lbl → [s0..s4]

    for var, lbl in P2_VARIANTS:
        mod_vals = {m: [] for m in MODALITIES}
        mod_ns   = {m: [] for m in MODALITIES}
        for s in SPLITS:
            if var in ("set_mil", "longitudinal_mk"):
                f0 = P2_DIR / f"split{s}_fold0" / f"{var}_mega" / f"metrics_{var}_final.json"
                abl_d = None

                if f0.exists():
                    abl_d = json.loads(f0.read_text()).get("unimodal_ablation", None)

                # Fallback for set_mil: try fold1 ablation if fold0 missing/empty
                # NOTE: fold1 ablation uses surv_ep="acr", so c_index = acr_surv only.
                # Fallback for set_mil: try fold1/2/3.
                # mega_abl_flat_key is None for clad/death if the old ablation only had acr.
                # After compute_slotattn_ablation.py patches the JSON, clad/death keys exist
                # as clad_c_index / death_c_index — handled by mega_abl_key remapping below.
                if abl_d is None and var == "set_mil":
                    for fb in [1, 2, 3]:
                        ff = P2_DIR / f"split{s}_fold{fb}" / f"{var}_mega" / f"metrics_{var}_final.json"
                        if ff.exists():
                            fb_abl = json.loads(ff.read_text()).get("unimodal_ablation", {})
                            if fb_abl and any(fb_abl.get(m, {}).get("n", 0) > 0 for m in MODALITIES):
                                abl_d = fb_abl; break

                # Remap mega_abl_key → abl_metric key for both fold0 and fold1 patched JSONs.
                # mega_abl_key: acr_cls→"bacc", acr_surv→"acr_c_index", clad→"clad_c_index", death→"death_c_index"
                if abl_d is not None and var in ("set_mil", "longitudinal_mk"):
                    mega_abl_key = cfg.get("mega_abl_key")
                    if mega_abl_key and mega_abl_key != cfg["abl_metric"]:
                        abl_d = {m: {cfg["abl_metric"]: get_nested(abl_d, m, mega_abl_key),
                                     "n": get_nested(abl_d, m, "n")}
                                 for m in MODALITIES}

                if abl_d is None:
                    for m in MODALITIES:
                        mod_vals[m].append(None); mod_ns[m].append(None)
                    continue
                f = None  # already loaded
            else:
                f = P2_DIR / f"split{s}_fold0" / f"{var}_{cfg['p2_folder']}" / f"metrics_{var}_final.json"
                abl_d = None
                if not f.exists():
                    for m in MODALITIES:
                        mod_vals[m].append(None); mod_ns[m].append(None)
                    continue
                abl_d = json.loads(f.read_text()).get("unimodal_ablation", {})

            for m in MODALITIES:
                mod_vals[m].append(get_nested(abl_d, m, cfg["abl_metric"]))
                n = get_nested(abl_d, m, "n")
                mod_ns[m].append(int(n) if n is not None else None)

        for m in MODALITIES:
            per_mod[(lbl, m)] = mod_vals[m]

        # prevalence-weighted expected ablation per split
        wv = []
        for si in range(len(SPLITS)):
            valid = [(m, mod_ns[m][si], mod_vals[m][si]) for m in MODALITIES
                     if mod_ns[m][si] is not None and mod_vals[m][si] is not None]
            if not valid:
                wv.append(None); continue
            n_total = max(n for _, n, _ in valid)
            wsum = sum(n / n_total for _, n, _ in valid)
            wv.append(sum((n / n_total) * v for _, n, v in valid) / wsum if wsum > 0 else None)
        wtd[lbl] = wv

    return per_mod, wtd


# ── Plotting helpers ──────────────────────────────────────────────────────────
def _scatter_col(ax, xi, vals, color, marker="o", size=36, alpha=0.85, zorder=3):
    vals = [v for v in vals if v is not None]
    if not vals:
        return
    rng = np.random.default_rng(int(xi * 100) + 42)
    jitter = rng.uniform(-0.13, 0.13, len(vals))
    ax.scatter([xi + j for j in jitter], vals, color=color, s=size, marker=marker,
               zorder=zorder, alpha=alpha, linewidths=0)
    ax.hlines(np.mean(vals), xi - 0.30, xi + 0.30, color=color, linewidth=2.0, zorder=zorder+1)


VAR_COLORS = {
    "Early":    "#59a14f",
    "Late":     "#edc948",
    "Middle":   "#b07aa1",
    "SlotAttn": "#ff9da7",
    "TimeSlot": "#9c755f",
}
# Alternating light backgrounds for ablation variant groups
_ABL_BG = ["#f7f7f7", "#eeeeee"]


def plot_panel(ax, task_key):
    cfg = TASK_CFG[task_key]

    p1_pm   = load_p1_per_modality(task_key)
    p1_wtd  = load_p1_weighted(task_key)
    p2_full = load_p2_full(task_key)
    p2_abl_pm, p2_abl_wtd = load_p2_ablation(task_key)

    # ── column layout ─────────────────────────────────────────────────────────
    # Group 1: P1 per-mod (4) + P1 wtd   | GAP |
    # Group 2: P2 full (5 variants)       | GAP |
    # Group 3: For each variant — 4 per-mod ablation cols + 1 wtd col (with shade)

    GAP  = 1.0   # gap between major groups
    SGAP = 0.4   # gap between variant sub-groups within ablation section
    xi = 0.0

    xticks = []; xlabels = []

    # ── Group 1: P1 ──────────────────────────────────────────────────────────
    g1_start = xi
    for mod in MODALITIES:
        _scatter_col(ax, xi, p1_pm[mod], MOD_COLORS[mod])
        xticks.append(xi); xlabels.append(f"P1\n{mod}")
        xi += 1.0
    # P1 prevalence-weighted expected
    _scatter_col(ax, xi, p1_wtd, "#555555", marker="D", size=50)
    xticks.append(xi); xlabels.append("P1\nwtd")
    xi += 1.0
    g1_end = xi

    xi += GAP

    # ── Group 2: P2 full ─────────────────────────────────────────────────────
    g2_start = xi
    for var, lbl in P2_VARIANTS:
        fb_splits = [s for s in SPLITS if (var, s) in _FALLBACK_FOLDS]
        tick_lbl = f"Full\n{lbl}†" if fb_splits else f"Full\n{lbl}"
        _scatter_col(ax, xi, p2_full[lbl], VAR_COLORS[lbl], marker="s", size=44)
        xticks.append(xi); xlabels.append(tick_lbl)
        xi += 1.0
    g2_end = xi

    xi += GAP

    # ── Group 3: per-variant ablation — explicit per-modality columns + wtd ──
    g3_start = xi
    sep_positions = []
    for vi, (var, lbl) in enumerate(P2_VARIANTS):
        sub_start = xi - 0.5

        # shaded background for this variant's ablation sub-group
        sub_width = len(MODALITIES) + 1 + SGAP  # 4 mods + 1 wtd
        ax.axvspan(sub_start, sub_start + sub_width, alpha=0.35,
                   color=_ABL_BG[vi % 2], zorder=0, linewidth=0)

        # per-modality ablation columns
        for mod in MODALITIES:
            key = (lbl, mod)
            vals = p2_abl_pm.get(key, [None]*5)
            _scatter_col(ax, xi, vals, MOD_COLORS[mod], size=32)
            xticks.append(xi)
            xlabels.append(f"{lbl}\nabl {mod[:2]}")
            xi += 1.0

        # prevalence-weighted expected for this variant
        _scatter_col(ax, xi, p2_abl_wtd[lbl], VAR_COLORS[lbl], marker="D", size=52, alpha=0.95)
        xticks.append(xi); xlabels.append(f"{lbl}\nabl wtd")
        xi += 1.0

        if vi < len(P2_VARIANTS) - 1:
            sep_positions.append(xi + SGAP / 2)
            xi += SGAP

    g3_end = xi

    # ── Aesthetics ────────────────────────────────────────────────────────────
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.45, zorder=1)

    # major group separators
    for sx in [g1_end + GAP / 2 - 0.5, g2_end + GAP / 2 - 0.5]:
        ax.axvline(sx, color="#aaaaaa", linewidth=1.2, linestyle="--", zorder=1)

    # minor separators between variant ablation sub-groups
    for sx in sep_positions:
        ax.axvline(sx, color="#cccccc", linewidth=0.7, linestyle=":", zorder=1)

    # group labels above axes
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.03
    for label, xs, xe in [
        ("P1 Unimodal", g1_start, g1_end),
        ("P2 Multimodal (full)", g2_start, g2_end),
        ("P2 Unimodal Ablation (per variant, per modality)", g3_start, g3_end),
    ]:
        ax.annotate(label, xy=((xs + xe) / 2, 1.01), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=8, color="#333333",
                    fontweight="semibold")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=6.5, rotation=30, ha="right")
    ax.set_ylabel("BACC" if "cls" in task_key else "C-index", fontsize=10)
    ax.set_xlim(g1_start - 0.7, g3_end + 0.3)
    ax.set_ylim(0.25, 1.03)
    ax.yaxis.grid(True, alpha=0.30, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # stdout summary
    print(f"\n  {cfg['label']}")
    for mod in MODALITIES:
        v = [x for x in p1_pm[mod] if x is not None]
        if v: print(f"    P1 {mod:10s}: {[round(x,3) for x in p1_pm[mod]]}  mean={np.mean(v):.3f}")
    v = [x for x in p1_wtd if x is not None]
    if v: print(f"    P1 wtd     : {[round(x,3) if x else None for x in p1_wtd]}  mean={np.mean(v):.3f}")
    for _, lbl in P2_VARIANTS:
        v = [x for x in p2_full[lbl] if x is not None]
        if v: print(f"    P2 {lbl:10s}: {[round(x,3) if x else None for x in p2_full[lbl]]}  mean={np.mean(v):.3f}")
        vw = [x for x in p2_abl_wtd[lbl] if x is not None]
        if vw: print(f"    Abl-wtd {lbl:7s}: {[round(x,3) if x else None for x in p2_abl_wtd[lbl]]}  mean={np.mean(vw):.3f}")


# ── Legend ────────────────────────────────────────────────────────────────────
def make_legend(fig):
    handles = []
    for mod, c in MOD_COLORS.items():
        handles.append(mpatches.Patch(color=c, label=f"Modality: {mod}"))
    handles.append(plt.Line2D([0],[0], marker="D", color="w", markerfacecolor="#555",
                               markersize=7, label="P1 prevalence-wtd expected"))
    for lbl, c in VAR_COLORS.items():
        handles.append(mpatches.Patch(color=c, label=f"P2: {lbl}"))
    handles.append(plt.Line2D([0],[0], marker="D", color="w", markerfacecolor="#333",
                               markersize=7, label="P2 abl prevalence-wtd expected (◆)"))
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=7.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))


# ── Per-task tables ───────────────────────────────────────────────────────────
import csv, statistics as _stats

def print_task_table(task_key):
    import math as _math
    cfg = TASK_CFG[task_key]
    metric_name = "BACC" if "cls" in task_key else "C-index"

    p1_pm        = load_p1_per_modality(task_key)
    p1_wtd       = load_p1_weighted(task_key)
    p2_full      = load_p2_full(task_key)
    p2_abl_pm, p2_abl_wtd = load_p2_ablation(task_key)

    W = 28; COL = 8
    total_w = W + 5*(COL+2) + 14 + 4

    def fv(v):
        if v is None or (isinstance(v, float) and _math.isnan(v)):
            return "   —   "
        return f"{v:.3f}"

    def ms(vals):
        v = [x for x in vals if x is not None and not _math.isnan(x)]
        if not v: return "      —      "
        flag = "*" if len(v) < len(SPLITS) else " "
        if len(v) == 1: return f"{v[0]:.3f}      {flag}"
        return f"{_stats.mean(v):.3f} ± {_stats.stdev(v):.3f}{flag}"

    def print_row(label, vals, indent=0):
        pad = " " * indent
        cells = "  ".join(f"{fv(v):>{COL}}" for v in vals)
        print(f"  {pad}{label:<{W-indent}}{cells}  {ms(vals):>14}")

    def section(title):
        print(f"\n  ── {title} {'─'*(total_w - len(title) - 6)}")

    def divider(char="·"):
        print(f"  {char*(total_w-2)}")

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n\n{'█'*total_w}")
    print(f"  {cfg['label'].upper()}  —  {metric_name}  "
          f"(fold-0 test set, 5 outer splits)")
    print(f"{'█'*total_w}")
    hdr = f"  {'Model':<{W}}" + "  ".join(f"{'s'+str(s):>{COL}}" for s in SPLITS) + f"  {'mean ± std':>14}"
    print(f"\n{hdr}")
    print(f"  {'─'*(total_w-2)}")

    csv_rows = []
    def record(label, vals, section_tag=""):
        v_clean = [x for x in vals if x is not None and not _math.isnan(x)]
        n_valid = len(v_clean)
        csv_rows.append({
            "section": section_tag, "model": label.strip(),
            **{f"s{i}": vals[i] for i in range(5)},
            "n_valid": n_valid,
            "mean": round(_stats.mean(v_clean), 4) if v_clean else None,
            "std":  round(_stats.stdev(v_clean), 4) if len(v_clean) >= 2 else None,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: P1 Unimodal
    # ─────────────────────────────────────────────────────────────────────────
    section("1. P1 Unimodal  (each model evaluated on samples with that modality)")
    for mod in MODALITIES:
        print_row(mod, p1_pm[mod])
        record(mod, p1_pm[mod], "P1 unimodal")
    divider()
    print_row("Wtd. expected (P1)", p1_wtd)
    record("Wtd. expected (P1)", p1_wtd, "P1 unimodal")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: P2 Multimodal — full model
    # ─────────────────────────────────────────────────────────────────────────
    section("2. P2 Multimodal — full model  (all modalities fused)")
    any_fallback = False
    for var, lbl in P2_VARIANTS:
        # Build per-split display: append † where a fallback fold was used
        vals = p2_full[lbl]
        fb_splits = [s for s in SPLITS if (var, s) in _FALLBACK_FOLDS]
        if fb_splits:
            any_fallback = True
            fb_folds = {s: _FALLBACK_FOLDS[(var, s)] for s in fb_splits}
            row_label = lbl + " †"
        else:
            row_label = lbl
        print_row(row_label, vals)
        record(row_label, vals, "P2 multimodal")
    if any_fallback:
        print(f"  † fold0 not yet available; values from best inner fold (HP-sweep model, same test set)")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: Unimodal ablation — P2 model with 1 modality vs P1 baseline
    # ─────────────────────────────────────────────────────────────────────────
    section("3. Unimodal ablation — P2 model (1 modality at a time) vs P1 baseline")
    print(f"   {'Model':<{W-3}}" +
          "  ".join(f"{'s'+str(s):>{COL}}" for s in SPLITS) +
          f"  {'mean ± std':>14}")
    print(f"   {'─'*(total_w-5)}")

    for mod in MODALITIES:
        print(f"\n   ┌ Modality: {mod}")
        # P1 baseline for this modality
        print_row(f"│  P1 {mod} (baseline)", p1_pm[mod], indent=3)
        record(f"P1 {mod} (baseline)", p1_pm[mod], f"ablation/{mod}")
        # P2 ablation per variant
        any_p2 = False
        for _, lbl in P2_VARIANTS:
            key = (lbl, mod)
            vals = p2_abl_pm.get(key, [None]*5)
            if any(v is not None for v in vals):
                any_p2 = True
                print_row(f"│  {lbl} ablation", vals, indent=3)
                record(f"{lbl} ablation", vals, f"ablation/{mod}")
        if not any_p2:
            print(f"     │  (no P2 ablation results yet)")
        print(f"   └{'─'*30}")

    # Weighted expected comparison
    print(f"\n   ┌ Prevalence-weighted expected")
    print_row("│  P1 wtd (baseline)", p1_wtd, indent=3)
    record("P1 wtd (baseline)", p1_wtd, "ablation/wtd")
    for _, lbl in P2_VARIANTS:
        vals = p2_abl_wtd[lbl]
        if any(v is not None for v in vals):
            print_row(f"│  {lbl} abl wtd", vals, indent=3)
            record(f"{lbl} abl wtd", vals, "ablation/wtd")
    print(f"   └{'─'*30}")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY: mean ± std only
    # ─────────────────────────────────────────────────────────────────────────
    section("Summary — mean ± std across splits")
    SW = 34
    print(f"  {'Model':<{SW}}  {'P1 baseline':>16}  {'P2 full':>16}  {'P2 abl wtd':>16}")
    print(f"  {'─'*SW}  {'─'*16}  {'─'*16}  {'─'*16}")

    def _ms2(vals):
        v = [x for x in vals if x is not None and not _math.isnan(x)]
        if not v: return "      —      "
        flag = f"(n={len(v)})" if len(v) < len(SPLITS) else ""
        if len(v) == 1: return f"  {v[0]:.3f}    {flag}"
        return f"  {_stats.mean(v):.3f} ± {_stats.stdev(v):.3f} {flag}"

    # Row: P1 wtd vs P2 full vs P2 abl wtd
    print(f"\n  {'Wtd. expected':<{SW}}  {_ms2(p1_wtd):>16}  {'':>16}  {'':>16}")
    for var, lbl in P2_VARIANTS:
        p2v = p2_full[lbl]
        ablv = p2_abl_wtd[lbl]
        fb_splits = [s for s in SPLITS if (var, s) in _FALLBACK_FOLDS]
        row_label = lbl + " †" if fb_splits else lbl
        print(f"  {row_label:<{SW}}  {'':>16}  {_ms2(p2v):>16}  {_ms2(ablv):>16}")

    print(f"\n  Per modality (P1 baseline  →  best P2 ablation):")
    print(f"  {'─'*(SW+54)}")
    for mod in MODALITIES:
        p1v = p1_pm[mod]
        best_lbl, best_vals, best_mean = None, None, -1
        for _, lbl in P2_VARIANTS:
            key = (lbl, mod)
            vals = p2_abl_pm.get(key, [None]*5)
            v = [x for x in vals if x is not None and not _math.isnan(x)]
            if v and _stats.mean(v) > best_mean:
                best_mean = _stats.mean(v); best_lbl = lbl; best_vals = vals
        best_str = f"{_ms2(best_vals)} ({best_lbl})" if best_lbl else "—"
        print(f"  {mod:<{SW}}  {_ms2(p1v):>16}  {'—':>16}  {best_str}")

    print(f"  * mean computed over <5 splits (one or more splits had 0 events for this modality/task)")
    print(f"\n{'═'*total_w}")

    # ── Save per-split detail CSV ─────────────────────────────────────────────
    out_csv = OUT_DIR / f"table_{task_key}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f,
            fieldnames=["section","model","s0","s1","s2","s3","s4","n_valid","mean","std"])
        writer.writeheader(); writer.writerows(csv_rows)
    print(f"  → saved: {out_csv}")

    # ── Save mean±std summary CSV ─────────────────────────────────────────────
    def _mean_std(vals):
        v = [x for x in vals if x is not None and not _math.isnan(x)]
        if not v: return None, None, 0
        return round(_stats.mean(v), 4), (round(_stats.stdev(v), 4) if len(v) >= 2 else None), len(v)

    summary_rows = []
    def srec(section, model, p1_vals=None, p2_vals=None, abl_vals=None):
        p1m, p1s, p1n   = _mean_std(p1_vals)   if p1_vals  is not None else (None, None, 0)
        p2m, p2s, p2n   = _mean_std(p2_vals)   if p2_vals  is not None else (None, None, 0)
        ablm, abls, abln = _mean_std(abl_vals)  if abl_vals is not None else (None, None, 0)
        summary_rows.append({
            "section": section, "model": model,
            "p1_mean": p1m,  "p1_std": p1s,  "p1_n": p1n,
            "p2_mean": p2m,  "p2_std": p2s,  "p2_n": p2n,
            "abl_mean": ablm, "abl_std": abls, "abl_n": abln,
        })

    # P1 unimodal
    for mod in MODALITIES:
        srec("P1 unimodal", mod, p1_vals=p1_pm[mod])
    srec("P1 unimodal", "Wtd. expected", p1_vals=p1_wtd)

    # P2 full + ablation
    for var, lbl in P2_VARIANTS:
        fb_splits = [s for s in SPLITS if (var, s) in _FALLBACK_FOLDS]
        row_label = lbl + " †" if fb_splits else lbl
        srec("P2 multimodal", row_label,
             p2_vals=p2_full[lbl], abl_vals=p2_abl_wtd[lbl])

    # Per-modality ablation best
    for mod in MODALITIES:
        for var, lbl in P2_VARIANTS:
            vals = p2_abl_pm.get((lbl, mod), [None]*5)
            fb_splits = [s for s in SPLITS if (var, s) in _FALLBACK_FOLDS]
            row_label = lbl + " †" if fb_splits else lbl
            srec(f"ablation/{mod}", row_label,
                 p1_vals=p1_pm[mod], abl_vals=vals)

    out_summary = OUT_DIR / f"table_{task_key}_summary.csv"
    with open(out_summary, "w", newline="") as f:
        writer = csv.DictWriter(f,
            fieldnames=["section","model",
                        "p1_mean","p1_std","p1_n",
                        "p2_mean","p2_std","p2_n",
                        "abl_mean","abl_std","abl_n"])
        writer.writeheader(); writer.writerows(summary_rows)
    print(f"  → saved: {out_summary}")

    # ── Save summary as image table ───────────────────────────────────────────
    _save_summary_image(task_key, summary_rows, cfg)


def _save_summary_image(task_key, summary_rows, cfg):
    """Render mean±std summary as a PNG table image."""
    import math as _math

    metric = "BACC" if "cls" in task_key else "C-index"

    def _fmt(mean, std):
        if mean is None: return "—"
        if std is None:  return f"{mean:.3f}"
        return f"{mean:.3f} ± {std:.3f}"

    # Build display rows: section header + data rows
    sections = {
        "P1 unimodal":   "P1 Unimodal",
        "P2 multimodal": "P2 Multimodal (full)",
    }
    # Collect rows grouped by section, then per-mod ablation blocks
    rows_by_section = {}
    for r in summary_rows:
        sec = r["section"]
        rows_by_section.setdefault(sec, []).append(r)

    display_rows = []   # (label, p1_str, p2_str, abl_str, is_header)

    for sec_key, sec_title in [
        ("P1 unimodal",   "P1 Unimodal"),
        ("P2 multimodal", "P2 Multimodal — full model"),
    ]:
        display_rows.append((sec_title, "", "", "", True))
        for r in rows_by_section.get(sec_key, []):
            display_rows.append((
                "  " + r["model"],
                _fmt(r["p1_mean"], r["p1_std"]),
                _fmt(r["p2_mean"], r["p2_std"]),
                _fmt(r["abl_mean"], r["abl_std"]),
                False,
            ))

    display_rows.append(("Unimodal Ablation (per modality)", "", "", "", True))
    for mod in MODALITIES:
        sec_key = f"ablation/{mod}"
        mod_rows = rows_by_section.get(sec_key, [])
        if not mod_rows: continue
        display_rows.append((f"  ── {mod}", "", "", "", True))
        for r in mod_rows:
            display_rows.append((
                "    " + r["model"],
                _fmt(r["p1_mean"], r["p1_std"]),
                "—",
                _fmt(r["abl_mean"], r["abl_std"]),
                False,
            ))

    col_labels = ["Model", f"P1 baseline ({metric})", f"P2 full ({metric})", f"P2 abl ({metric})"]
    cell_text  = [[r[0], r[1], r[2], r[3]] for r in display_rows]
    n_rows = len(cell_text)

    fig_h = max(3.0, n_rows * 0.32 + 1.2)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.35)

    # Style: header row
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Style data rows
    section_bg = "#dce6f1"
    alt_bg     = ["#ffffff", "#f4f8ff"]
    data_idx = 0
    for i, (label, p1, p2, abl, is_hdr) in enumerate(display_rows):
        row_idx = i + 1
        if is_hdr:
            for j in range(len(col_labels)):
                tbl[(row_idx, j)].set_facecolor(section_bg)
                tbl[(row_idx, j)].set_text_props(fontweight="bold", color="#1a252f")
        else:
            bg = alt_bg[data_idx % 2]; data_idx += 1
            for j in range(len(col_labels)):
                tbl[(row_idx, j)].set_facecolor(bg)
            # Highlight non-"—" cells in p2/abl columns
            for j in [2, 3]:
                cell = tbl[(row_idx, j)]
                if cell.get_text().get_text() not in ("—", ""):
                    cell.set_text_props(fontweight="semibold")

    fig.suptitle(
        f"Lung Transplant MIL — {cfg['label']}  |  mean ± std (5 outer splits)",
        fontsize=10, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / f"table_{task_key}_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
TASK_KEYS = ["acr_cls", "acr_surv", "clad", "death"]

for tk in TASK_KEYS:
    # Wider figure to accommodate explicit per-modality ablation columns:
    # 5 (P1) + 5 (P2 full) + 5 variants × 5 cols = 35 data columns → 26 inches wide
    fig, ax = plt.subplots(1, 1, figsize=(26, 7))
    plot_panel(ax, tk)
    fig.suptitle(
        f"Lung Transplant MIL — {TASK_CFG[tk]['label']}\n"
        "fold-0 test sets  |  ● = per-split  ◆ = prevalence-weighted expected  — = mean  "
        "† = HP-sweep fold (fold-0 pending)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    make_legend(fig)
    fig.tight_layout()
    out = OUT_DIR / f"test_metrics_{tk}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure → {out}\n")

for tk in TASK_KEYS:
    print_task_table(tk)
