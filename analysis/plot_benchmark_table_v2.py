"""
Benchmark table figure — all 18 models × 4 tasks.
Rows: fixed model order matching plot_benchmark_v2.py
Cols: ACR cls (BACC), ACR surv (C-index), CLAD (C-index), Death (C-index)
Cell: mean ± std; per-split values s0–s4 in small monospace below
Background: RdYlGn per column (relative rank)
Bold + green border = best per column
Group separator rows between model families

Run via: sbatch analysis/submit_benchmark_table_v2.sh
"""
from pathlib import Path
import json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
METRICS  = ROOT / "results" / "mm_abmil_v8"
LIN_CSV  = ROOT / "results" / "linear_models" / "metrics_summary.csv"
OUT_DIR  = ROOT / "figures" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared model palette (same as benchmark_v2) ────────────────────────────────
SHARED_MODEL_COLORS = {
    "Linear HE":          "#BDBDBD",
    "Linear BAL":         "#9E9E9E",
    "Linear CT":          "#757575",
    "Linear Clinical":    "#616161",
    "wt avg Linear":      "#424242",
    "ABMIL HE":           "#90CAF9",
    "ABMIL BAL":          "#42A5F5",
    "ABMIL CT":           "#1976D2",
    "ABMIL Clinical":     "#1565C0",
    "wt avg ABMIL":       "#0D47A1",
    "Early fusion":       "#80CBC4",
    "Middle fusion":      "#26A69A",
    "Late fusion":        "#00796B",
    "SetMIL":             "#CE93D8",
    "SetMIL-MT":          "#9C27B0",
    "SetMIL-MT (no SAB)": "#6A1B9A",
    "LongMK-MT":          "#EF9A9A",
    "LongMK":             "#C62828",
}

# ── Fixed model order with group tags ─────────────────────────────────────────
MODEL_DEFS = [
    ("Linear HE",          "linear"),
    ("Linear BAL",         "linear"),
    ("Linear CT",          "linear"),
    ("Linear Clinical",    "linear"),
    ("wt avg Linear",      "linear"),
    ("ABMIL HE",           "p1"),
    ("ABMIL BAL",          "p1"),
    ("ABMIL CT",           "p1"),
    ("ABMIL Clinical",     "p1"),
    ("wt avg ABMIL",       "p1"),
    ("Early fusion",       "fusion"),
    ("Middle fusion",      "fusion"),
    ("Late fusion",        "fusion"),
    ("SetMIL",             "setmil"),
    ("SetMIL-MT",          "setmil"),
    ("SetMIL-MT (no SAB)", "setmil"),
    ("LongMK-MT",          "longi"),
    ("LongMK",             "longi"),
]

GROUP_LABELS = {
    "linear": "Linear baselines",
    "p1":     "ABMIL unimodal",
    "fusion": "Fusion",
    "setmil": "SetMIL family",
    "longi":  "LongitudinalMK",
}

# ── Task definitions ───────────────────────────────────────────────────────────
TASKS = [
    {"key": "acr_cls",   "suffix": "cls",       "metric": "bacc",    "label": "ACR cls\n(BACC ↑)",    "lin_task": "ACR",    "lin_metric": "bacc"},
    {"key": "acr_surv",  "suffix": "acr_surv",  "metric": "c_index", "label": "ACR surv\n(C-idx ↑)",  "lin_task": "ACR_TTE","lin_metric": "cindex"},
    {"key": "clad_surv", "suffix": "clad_surv", "metric": "c_index", "label": "CLAD surv\n(C-idx ↑)", "lin_task": "CLAD",   "lin_metric": "cindex"},
    {"key": "death_surv","suffix": "death_surv","metric": "c_index", "label": "Death surv\n(C-idx ↑)","lin_task": "Death",  "lin_metric": "cindex"},
]

# ── Mapping CSV display labels → data keys ────────────────────────────────────
CSV_TO_DISPLAY = {
    "P1 HE":           "ABMIL HE",
    "P1 BAL":          "ABMIL BAL",
    "P1 CT":           "ABMIL CT",
    "P1 Clinical":     "ABMIL Clinical",
    "P1 wtd ensemble": "wt avg ABMIL",
    "Early fusion":    "Early fusion",
    "Middle fusion":   "Middle fusion",
    "Late fusion":     "Late fusion",
    "SetMIL":          "SetMIL",
    "SetMIL-MT":       "SetMIL-MT",
    "SetMIL-MT (no SAB)":   "SetMIL-MT (no SAB)",
    "LongMK-MT (no ALiBi)": "LongMK-MT",
    "LongMK (no ALiBi) ★":  "LongMK",
}

LONGI_TASK_KEYS = {
    "acr_cls":   "acr_cls",
    "acr_surv":  "acr_surv",
    "clad_surv": "clad",
    "death_surv":"death",
}

LONGI_VARIANTS = {
    "longitudinal_mk_mt_no_alibi": "LongMK-MT",
    "longitudinal_mk_no_alibi":    "LongMK",
}

NON_LONGI_VARIANTS = {
    "early":          "Early fusion",
    "middle":         "Middle fusion",
    "late":           "Late fusion",
    "set_mil_no_sab": "SetMIL",
    "set_mil_mt":     "SetMIL-MT",
    "set_mil_mt_no_sab": "SetMIL-MT (no SAB)",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_linear():
    df = pd.read_csv(LIN_CSV)
    MOD_MAP = {"H&E": "Linear HE", "HE": "Linear HE", "BAL": "Linear BAL",
               "CT": "Linear CT", "Clinical": "Linear Clinical", "All": "wt avg Linear"}
    data = {}
    for task in TASKS:
        task_df = df[df["task"] == task["lin_task"]].copy()
        for lin_mod, disp in MOD_MAP.items():
            rows = task_df[task_df["modality"] == lin_mod]
            vals = []
            for _, r in rows.iterrows():
                try:
                    vals.append(float(r.get(task["lin_metric"], np.nan)))
                except (TypeError, ValueError):
                    vals.append(np.nan)
            if vals:
                data.setdefault(disp, {})[task["key"]] = vals
    return data


def load_nonlongi():
    data = {}
    for variant, disp in NON_LONGI_VARIANTS.items():
        for task in TASKS:
            vals = []
            for s in range(5):
                path = METRICS / f"metrics_split{s}_fold0_{variant}_{task['suffix']}.json"
                v = np.nan
                if path.exists():
                    try:
                        d = json.loads(path.read_text())
                        test = d.get("test", {})
                        raw = test.get(task["metric"])
                        if raw is not None:
                            v = float(raw)
                    except Exception:
                        pass
                vals.append(v)
            data.setdefault(disp, {})[task["key"]] = vals
    return data


def load_longi():
    data = {}
    for variant, disp in LONGI_VARIANTS.items():
        for task in TASKS:
            nested_key = LONGI_TASK_KEYS[task["key"]]
            vals = []
            for s in range(5):
                path = METRICS / f"metrics_split{s}_fold0_{variant}_{task['suffix']}.json"
                v = np.nan
                if path.exists():
                    try:
                        d = json.loads(path.read_text())
                        test = d.get("test", {})
                        sub = test.get(nested_key, {})
                        raw = sub.get(task["metric"]) if isinstance(sub, dict) else None
                        if raw is not None:
                            v = float(raw)
                    except Exception:
                        pass
                vals.append(v)
            data.setdefault(disp, {})[task["key"]] = vals
    return data


def build_matrix():
    lin  = load_linear()
    nonl = load_nonlongi()
    longi = load_longi()
    combined = {**lin, **nonl, **longi}

    n_models = len(MODEL_DEFS)
    n_tasks  = len(TASKS)
    means  = np.full((n_models, n_tasks), np.nan)
    stds   = np.full((n_models, n_tasks), np.nan)
    splits_raw = [[None] * n_tasks for _ in range(n_models)]

    for mi, (label, _) in enumerate(MODEL_DEFS):
        mdata = combined.get(label, {})
        for ti, task in enumerate(TASKS):
            vals = mdata.get(task["key"])
            if vals is not None:
                valid = [v for v in vals if not np.isnan(v)]
                if valid:
                    means[mi, ti] = float(np.nanmean(vals))
                    stds[mi, ti]  = float(np.nanstd(vals))
                    splits_raw[mi][ti] = vals

    return means, stds, splits_raw


# ── Figure drawing ─────────────────────────────────────────────────────────────

def draw_table(means, stds, splits_raw):
    n_models = len(MODEL_DEFS)
    n_tasks  = len(TASKS)

    # Per-column normalisation for RdYlGn background
    col_min = np.nanmin(means, axis=0)
    col_max = np.nanmax(means, axis=0)
    col_rng = np.where(col_max - col_min < 1e-6, 1.0, col_max - col_min)
    norm    = (means - col_min) / col_rng
    best_idx = np.nanargmax(means, axis=0)

    # Build row sequence: add separator rows between groups
    rows = []   # each entry: ("data", mi) or ("sep", group_label) or ("header",)
    rows.append(("header",))
    prev_group = None
    for mi, (label, group) in enumerate(MODEL_DEFS):
        if group != prev_group:
            rows.append(("sep", GROUP_LABELS[group]))
            prev_group = group
        rows.append(("data", mi))

    CMAP     = cm.RdYlGn
    HDR_BG   = "#1c2833"
    HDR_FG   = "white"
    SEP_BG   = "#E8EAF6"
    SEP_FG   = "#3949AB"
    ODD_BG   = "#f8f9fa"
    EVEN_BG  = "#ffffff"
    BEST_EC  = "#1e8449"
    MISS_BG  = "#f5f5f5"

    DATA_H   = 0.95
    HEADER_H = 0.75
    SEP_H    = 0.38

    row_heights = []
    for r in rows:
        if r[0] == "header":
            row_heights.append(HEADER_H)
        elif r[0] == "sep":
            row_heights.append(SEP_H)
        else:
            row_heights.append(DATA_H)

    COL_W_LABEL = 2.6
    COL_W_TASK  = 2.3
    col_widths  = [COL_W_LABEL] + [COL_W_TASK] * n_tasks

    fig_w = sum(col_widths) + 0.4
    fig_h = sum(row_heights) + 0.65

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")

    PAD_L = 0.22
    PAD_T = 0.32

    def row_top(ri):
        return fig_h - PAD_T - sum(row_heights[:ri])

    def cell_rect(ri, ci):
        x = PAD_L + sum(col_widths[:ci])
        y = row_top(ri) - row_heights[ri]
        return x, y, col_widths[ci], row_heights[ri]

    def add_rect(ri, ci, fc, ec="#d0d0d0", lw=0.5):
        x, y, w, h = cell_rect(ri, ci)
        patch = mpatches.FancyBboxPatch(
            (x / fig_w, y / fig_h), w / fig_w, h / fig_h,
            boxstyle="square,pad=0",
            transform=fig.transFigure,
            facecolor=fc, edgecolor=ec, linewidth=lw, clip_on=False)
        fig.add_artist(patch)

    def add_text(ri, ci, main, sub="", fs=8.0, bold=False,
                 color="#111", halign="center"):
        x, y, w, h = cell_rect(ri, ci)
        cx = (x + w / 2) / fig_w
        cy = (y + h / 2) / fig_h
        kw = dict(ha=halign, transform=fig.transFigure,
                  fontweight="bold" if bold else "normal")
        if sub:
            ax.text(cx, cy + 0.012, main, va="center", fontsize=fs,
                    color=color, **kw)
            ax.text(cx, cy - 0.013, sub, va="center", fontsize=5.0,
                    color="#777", fontfamily="monospace",
                    ha=halign, transform=fig.transFigure)
        else:
            ax.text(cx, cy, main, va="center", fontsize=fs, color=color, **kw)

    data_row_parity = {}
    parity = 0
    for ri, r in enumerate(rows):
        if r[0] == "data":
            data_row_parity[ri] = parity
            parity ^= 1

    for ri, r in enumerate(rows):
        if r[0] == "header":
            for ci in range(n_tasks + 1):
                add_rect(ri, ci, HDR_BG, ec=HDR_BG)
            add_text(ri, 0, "Model", fs=9.5, bold=True, color=HDR_FG, halign="left")
            for ti, task in enumerate(TASKS):
                add_text(ri, ti + 1, task["label"], fs=8.5, bold=True, color=HDR_FG)

        elif r[0] == "sep":
            _, group_lbl = r
            for ci in range(n_tasks + 1):
                add_rect(ri, ci, SEP_BG, ec=SEP_BG)
            add_text(ri, 0, group_lbl, fs=7.8, bold=True, color=SEP_FG, halign="left")

        else:
            _, mi = r
            label, group = MODEL_DEFS[mi]
            model_color = SHARED_MODEL_COLORS.get(label, "#888888")
            bg = ODD_BG if data_row_parity[ri] == 0 else EVEN_BG

            # Label cell with left colour stripe
            add_rect(ri, 0, bg, ec="#d0d0d0")
            # Colour stripe on left edge
            x, y, w, h = cell_rect(ri, 0)
            stripe = mpatches.FancyBboxPatch(
                (x / fig_w, y / fig_h), 0.006, h / fig_h,
                boxstyle="square,pad=0",
                transform=fig.transFigure,
                facecolor=model_color, edgecolor="none", clip_on=False)
            fig.add_artist(stripe)
            add_text(ri, 0, label, fs=8.0, bold=False, color="#111", halign="left")

            # Task cells
            for ti in range(n_tasks):
                is_best = (best_idx[ti] == mi)
                nv = norm[mi, ti]
                m  = means[mi, ti]
                s  = stds[mi, ti]
                sv = splits_raw[mi][ti]

                if np.isnan(nv):
                    cell_bg = MISS_BG
                else:
                    rgba = CMAP(0.10 + nv * 0.80)
                    cell_bg = (*rgba[:3], 0.45)

                ec = BEST_EC if is_best else "#d0d0d0"
                lw = 2.2 if is_best else 0.5
                add_rect(ri, ti + 1, cell_bg, ec=ec, lw=lw)

                if np.isnan(m):
                    add_text(ri, ti + 1, "—", fs=8.5, color="#aaa")
                else:
                    main_txt = f"{m:.3f} ± {s:.3f}"
                    if sv is not None:
                        per = "  ".join(
                            f"s{si}:{v:.3f}" if not np.isnan(v) else f"s{si}:—"
                            for si, v in enumerate(sv))
                    else:
                        per = ""
                    add_text(ri, ti + 1, main_txt, sub=per,
                             fs=8.0, bold=is_best, color="#111")

    # Colorbar
    cb_ax = fig.add_axes([0.55, 0.008, 0.30, 0.013])
    cb = plt.colorbar(cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
                      cax=cb_ax, orientation="horizontal")
    cb.set_label("Relative rank per task column", fontsize=6)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Low", "Mid", "High"])
    cb.ax.tick_params(labelsize=5.5)

    # Legend
    lh = [
        mpatches.Patch(facecolor="none", edgecolor=BEST_EC, lw=2.2,
                       label="Best per task"),
        mpatches.Patch(facecolor=MISS_BG, edgecolor="#ccc", lw=0.5,
                       label="Not available"),
    ]
    fig.legend(handles=lh, loc="lower left",
               bbox_to_anchor=(0.02, 0.005), fontsize=6, framealpha=0.9)

    fig.suptitle("Benchmark — All Models × All Tasks  (mean ± std, 5 splits)",
                 fontsize=11.5, fontweight="bold", y=0.998)
    fig.text(0.02, 0.001,
             "BACC = balanced accuracy. C-idx = Harrell C-index. "
             "Per-split values s0–s4 shown below mean±std. "
             "Green border = best per column.",
             fontsize=5.5, color="#666", va="bottom")

    for ext in ("png", "pdf"):
        out = OUT_DIR / f"benchmark_table_v2.{ext}"
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"  Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("Loading data...")
    means, stds, splits_raw = build_matrix()
    print(f"  {np.sum(~np.isnan(means))}/{means.size} cells have data")
    print("Drawing table...")
    draw_table(means, stds, splits_raw)
    print("Done.")
