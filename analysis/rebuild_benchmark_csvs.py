"""
Rebuild all 4 comparison_*.csv files from raw metrics JSONs, then regenerate
benchmark bar+strip figures.  Includes ALL model variants.

Run via:  sbatch analysis/submit_rebuild_benchmark.sh
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

ROOT    = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results" / "mm_abmil_v8"
PRED    = ROOT / "results" / "predictions"
OUT_DIR = ROOT / "figures" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────────
P1_COL   = "#1A5C8A"  # unimodal P1
P2_COL   = "#17685A"  # non-temporal multimodal
SET_COL  = "#452870"  # SetMIL family (plum)
LK_COL   = "#952030"  # longitudinal family (crimson)
BEST_EDGE= "#BF7320"  # amber border on best bar
BG       = "#FAF6F2"

# ── Model variant metadata ────────────────────────────────────────────────────
# Maps internal variant name → (display label, color, group)
P2_VARIANTS = {
    "early":                    ("Early fusion",              P2_COL,  "p2"),
    "middle":                   ("Middle fusion",             P2_COL,  "p2"),
    "late":                     ("Late fusion",               P2_COL,  "p2"),
    "set_mil_no_sab":           ("SetMIL",                    SET_COL, "set"),
    "set_mil_mt":               ("SetMIL-MT",                 SET_COL, "set"),
    "set_mil_mt_no_sab":        ("SetMIL-MT (no SAB)",        SET_COL, "set"),
    "longitudinal_mk_mt":       ("LongMK-MT",                 LK_COL,  "longi"),
    "longitudinal_mk_mt_no_alibi":("LongMK-MT (no ALiBi)",   LK_COL,  "longi"),
    "longitudinal_mk_no_alibi": ("LongMK (no ALiBi) ★",      LK_COL,  "longi"),
}

# P1 unimodal rows are already in the CSVs — keep as-is
P1_VARIANTS_ORDER = [
    "P1 HE", "P1 BAL", "P1 CT", "P1 Clinical", "P1 wtd ensemble",
]

TASK_CFG = {
    # suffix = filename suffix; nested_key = key inside test{} for multi-task longi JSONs
    "acr_cls":   {"csv": "comparison_acr_cls.csv",   "suffix": "cls",       "metric_key": "bacc",    "nested_key": "acr_cls",  "metric_lbl": "BACC",    "label": "ACR classification (BACC)"},
    "acr_surv":  {"csv": "comparison_acr_surv.csv",  "suffix": "acr_surv",  "metric_key": "c_index", "nested_key": "acr_surv", "metric_lbl": "C-index", "label": "ACR survival (C-index)"},
    "clad_surv": {"csv": "comparison_clad.csv",      "suffix": "clad_surv", "metric_key": "c_index", "nested_key": "clad",     "metric_lbl": "C-index", "label": "CLAD survival (C-index)"},
    "death_surv":{"csv": "comparison_death.csv",     "suffix": "death_surv","metric_key": "c_index", "nested_key": "death",    "metric_lbl": "C-index", "label": "Death survival (C-index)"},
}


def extract_metric(variant, suffix, split, metric_key, nested_key=None):
    path = METRICS / f"metrics_split{split}_fold0_{variant}_{suffix}.json"
    if not path.exists():
        return float("nan")
    try:
        d = json.loads(path.read_text())
        test = d.get("test", {})
        # Flat structure (most models): test.metric_key
        val = test.get(metric_key)
        if val is not None:
            return float(val)
        # Nested structure (longitudinal multi-task): test.nested_key.metric_key
        if nested_key and nested_key in test:
            val = test[nested_key].get(metric_key)
            if val is not None:
                return float(val)
        return float("nan")
    except Exception:
        return float("nan")


def build_p2_rows(suffix, metric_key, nested_key=None):
    rows = []
    for variant, (label, color, group) in P2_VARIANTS.items():
        splits = [extract_metric(variant, suffix, s, metric_key, nested_key) for s in range(5)]
        valid = [v for v in splits if not np.isnan(v)]
        rows.append({
            "model":  label,
            "s0": splits[0], "s1": splits[1], "s2": splits[2],
            "s3": splits[3], "s4": splits[4],
            "mean": float(np.nanmean(splits)) if valid else float("nan"),
            "std":  float(np.nanstd(splits))  if valid else float("nan"),
            "_variant": variant,
            "_color": color,
            "_group": group,
        })
    return rows


def load_p1_rows(csv_path):
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        model = str(row["model"]).strip()
        if not model.startswith("P1"):
            continue
        s_vals = []
        for c in ["s0","s1","s2","s3","s4"]:
            try:
                v = float(row.get(c, "nan"))
            except (ValueError, TypeError):
                v = float("nan")
            s_vals.append(v)
        rows.append({
            "model": model, "s0": s_vals[0], "s1": s_vals[1], "s2": s_vals[2],
            "s3": s_vals[3], "s4": s_vals[4],
            "mean": float(row["mean"]) if not pd.isna(row.get("mean","")) else float(np.nanmean(s_vals)),
            "std":  float(row["std"])  if not pd.isna(row.get("std",""))  else float(np.nanstd(s_vals)),
            "_variant": model.replace(" ","_").lower(),
            "_color": P1_COL,
            "_group": "p1",
        })
    return rows


# ── Build / rewrite CSVs ──────────────────────────────────────────────────────
all_dfs = {}
for task_key, cfg in TASK_CFG.items():
    p1_rows = load_p1_rows(PRED / cfg["csv"])
    p2_rows = build_p2_rows(cfg["suffix"], cfg["metric_key"], cfg.get("nested_key"))
    all_rows = p1_rows + p2_rows
    df = pd.DataFrame(all_rows)
    # Write CSV (drop internal _* cols)
    csv_cols = ["model","s0","s1","s2","s3","s4","mean","std"]
    df[csv_cols].to_csv(PRED / cfg["csv"], index=False)
    print(f"[csv] {cfg['csv']} — {len(df)} models")
    all_dfs[task_key] = df


# ── Plotting ──────────────────────────────────────────────────────────────────
SPLIT_COLS_DOT = ["s0","s1","s2","s3","s4"]

def get_color(row):
    if "_color" in row and not pd.isna(row["_color"]):
        return row["_color"]
    m = str(row["model"])
    if m.startswith("P1"):
        return P1_COL
    if "LongMK" in m or "Longitudinal" in m:
        return LK_COL
    if "SetMIL" in m:
        return SET_COL
    return P2_COL


def plot_task(ax, df, title, metric_lbl, show_legend=False):
    # Drop rows with no valid mean
    df = df.dropna(subset=["mean"]).reset_index(drop=True)
    # Sort: p1 first, then p2 by group, then by mean desc within group
    grp_order = {"p1": 0, "p2": 1, "set": 2, "longi": 3}
    if "_group" in df.columns:
        df["_go"] = df["_group"].map(grp_order).fillna(4)
        df = df.sort_values(["_go", "mean"], ascending=[True, False]).reset_index(drop=True)

    n = len(df)
    y = np.arange(n)
    colors = [get_color(row) for _, row in df.iterrows()]
    best_idx = int(df["mean"].idxmax())

    bars = ax.barh(y, df["mean"], color=colors, alpha=0.82, height=0.55, zorder=2)
    bars[best_idx].set_edgecolor(BEST_EDGE)
    bars[best_idx].set_linewidth(2.5)

    ax.errorbar(df["mean"], y, xerr=df["std"], fmt="none",
                ecolor="#1A1018", elinewidth=0.9, capsize=3, alpha=0.5, zorder=3)

    for i, (_, row) in enumerate(df.iterrows()):
        for sc in SPLIT_COLS_DOT:
            sv = row.get(sc, float("nan"))
            try:
                sv = float(sv)
            except (ValueError, TypeError):
                sv = float("nan")
            if not np.isnan(sv):
                ax.scatter(sv, i, s=13, color="white", edgecolors=colors[i],
                           linewidths=0.8, zorder=4, alpha=0.9)

    ax.axvline(0.5, color="#7D6D78", linewidth=0.8, linestyle="--", alpha=0.4, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"].tolist(), fontsize=7.5)
    ax.set_xlabel(metric_lbl, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    ax.tick_params(axis="both", labelsize=7)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines[["top","right"]].set_visible(False)
    xleft = max(0.3, float(df["mean"].min()) - float(df["std"].max()) - 0.06)
    ax.set_xlim(left=xleft)
    ax.invert_yaxis()
    ax.set_facecolor(BG)

    if show_legend:
        handles = [
            Line2D([0],[0], color=P1_COL,  lw=7, alpha=.82, label="Unimodal (P1)"),
            Line2D([0],[0], color=P2_COL,  lw=7, alpha=.82, label="Fusion baselines"),
            Line2D([0],[0], color=SET_COL, lw=7, alpha=.82, label="SetMIL family"),
            Line2D([0],[0], color=LK_COL,  lw=7, alpha=.82, label="LongitudinalMK ★ best"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.85)


# Per-task figures
for task_key, cfg in TASK_CFG.items():
    df = all_dfs[task_key]
    fig, ax = plt.subplots(figsize=(7, 0.44 * len(df.dropna(subset=["mean"])) + 1.2))
    fig.patch.set_facecolor(BG)
    plot_task(ax, df, cfg["label"], cfg["metric_lbl"], show_legend=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"benchmark_multimodal_{task_key}.{ext}", dpi=180,
                    bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [fig] {task_key}")

# 4-panel combined
task_keys = list(TASK_CFG.keys())
dfs = [all_dfs[k].dropna(subset=["mean"]) for k in task_keys]
max_n = max(len(d) for d in dfs)

fig, axes = plt.subplots(1, 4, figsize=(20, 0.42 * max_n + 2.0), sharey=False)
fig.patch.set_facecolor(BG)
for ax, tk, df in zip(axes, task_keys, dfs):
    ax.set_facecolor(BG)
    if not df.empty:
        plot_task(ax, df, TASK_CFG[tk]["label"], TASK_CFG[tk]["metric_lbl"])

handles = [
    Line2D([0],[0], color=P1_COL,  lw=8, alpha=.82, label="Unimodal (P1)"),
    Line2D([0],[0], color=P2_COL,  lw=8, alpha=.82, label="Fusion baselines"),
    Line2D([0],[0], color=SET_COL, lw=8, alpha=.82, label="SetMIL family"),
    Line2D([0],[0], color=LK_COL,  lw=8, alpha=.82, label="LongitudinalMK ★"),
]
fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, 1.02), framealpha=0.88)
fig.suptitle("Multimodal benchmark — all tasks, all variants", fontsize=11, fontweight="bold", y=1.07)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"benchmark_multimodal_all.{ext}", dpi=180,
                bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  [fig] 4-panel all")
print(f"\n[done] figures → {OUT_DIR}")
