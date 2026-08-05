"""
One-sided paired t-test: LongMK > each other model, per task.
Produces a heatmap (models × tasks) coloured by p-value significance,
using the same model colours as plot_benchmark.py.

Run via: sbatch analysis/submit_benchmark_pvalues.sh
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

ROOT    = Path(__file__).resolve().parent.parent
PRED    = ROOT / "results" / "predictions"
LIN_CSV = ROOT / "results" / "linear_models" / "metrics_summary.csv"
OUT_DIR = ROOT / "figures" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG = "#FAF6F2"

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

# All models except LongMK itself (it's the reference)
MODEL_ORDER = [
    "Linear HE", "Linear BAL", "Linear CT", "Linear Clinical", "wt avg Linear",
    "ABMIL HE", "ABMIL BAL", "ABMIL CT", "ABMIL Clinical", "wt avg ABMIL",
    "Early fusion", "Middle fusion", "Late fusion",
    "SetMIL", "SetMIL-MT", "SetMIL-MT (no SAB)",
    "LongMK-MT",
]

TASKS = {
    "acr_cls":   {"file": "comparison_acr_cls.csv",  "metric": "BACC",    "label": "ACR cls\n(BACC)",
                  "lin_task": "ACR",    "lin_metric": "bacc"},
    "acr_surv":  {"file": "comparison_acr_surv.csv", "metric": "C-index", "label": "ACR surv\n(C-index)",
                  "lin_task": "ACR_TTE","lin_metric": "cindex"},
    "clad_surv": {"file": "comparison_clad.csv",     "metric": "C-index", "label": "CLAD\n(C-index)",
                  "lin_task": "CLAD",   "lin_metric": "cindex"},
    "death_surv":{"file": "comparison_death.csv",    "metric": "C-index", "label": "Death\n(C-index)",
                  "lin_task": "Death",  "lin_metric": "cindex"},
}

CSV_TO_DISPLAY = {
    "P1 HE":                 "ABMIL HE",
    "P1 BAL":                "ABMIL BAL",
    "P1 CT":                 "ABMIL CT",
    "P1 Clinical":           "ABMIL Clinical",
    "P1 wtd ensemble":       "wt avg ABMIL",
    "Early fusion":          "Early fusion",
    "Middle fusion":         "Middle fusion",
    "Late fusion":           "Late fusion",
    "SetMIL":                "SetMIL",
    "SetMIL-MT":             "SetMIL-MT",
    "SetMIL-MT (no SAB)":    "SetMIL-MT (no SAB)",
    "LongMK-MT (no ALiBi)":  "LongMK-MT",
    "LongMK (no ALiBi) ★":   "LongMK",
}

SPLIT_COLS = ["s0", "s1", "s2", "s3", "s4"]


def load_linear_splits(lin_task, lin_metric):
    df = pd.read_csv(LIN_CSV)
    df = df[df["task"] == lin_task].copy()
    MOD_MAP = {"HE": "Linear HE", "H&E": "Linear HE", "BAL": "Linear BAL",
               "CT": "Linear CT", "Clinical": "Linear Clinical", "All": "wt avg Linear"}
    out = {}
    for mod, disp in MOD_MAP.items():
        rows = df[df["modality"] == mod].sort_values("fold")
        vals = []
        for _, r in rows.iterrows():
            try:
                vals.append(float(r[lin_metric]))
            except (TypeError, ValueError, KeyError):
                vals.append(np.nan)
        if any(~np.isnan(v) for v in vals):
            out[disp] = np.array(vals[:5])
    return out


def load_csv_splits(task_key):
    meta = TASKS[task_key]
    df = pd.read_csv(PRED / meta["file"])
    out = {}
    for _, row in df.iterrows():
        disp = CSV_TO_DISPLAY.get(str(row["model"]).strip())
        if disp is None:
            continue
        vals = []
        for s in SPLIT_COLS:
            try:
                vals.append(float(row[s]))
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[disp] = np.array(vals)
    return out


def paired_ttest_onesided(a, b):
    """P(LongMK > other): one-sided paired t-test on valid paired obs."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    n = mask.sum()
    if n < 2:
        return np.nan, n
    d = a[mask] - b[mask]
    t_stat, p_two = stats.ttest_rel(a[mask], b[mask])
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    return p_one, n


# ── Collect p-values matrix ────────────────────────────────────────────────────
p_matrix = np.full((len(MODEL_ORDER), len(TASKS)), np.nan)
n_matrix = np.zeros_like(p_matrix, dtype=int)

for j, (task_key, meta) in enumerate(TASKS.items()):
    lin_splits = load_linear_splits(meta["lin_task"], meta["lin_metric"])
    csv_splits = load_csv_splits(task_key)
    all_splits = {**lin_splits, **csv_splits}

    longmk_vals = all_splits.get("LongMK")
    if longmk_vals is None:
        continue

    for i, mdl in enumerate(MODEL_ORDER):
        other_vals = all_splits.get(mdl)
        if other_vals is None:
            continue
        p, n = paired_ttest_onesided(longmk_vals, other_vals)
        p_matrix[i, j] = p
        n_matrix[i, j] = n

# ── Plot ───────────────────────────────────────────────────────────────────────
task_labels = [v["label"] for v in TASKS.values()]
n_models = len(MODEL_ORDER)
n_tasks  = len(TASKS)

fig, ax = plt.subplots(figsize=(6, 0.45 * n_models + 2.0), facecolor=BG)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Colour map: green = significant, fade to white/yellow at p=1
cmap = mcolors.LinearSegmentedColormap.from_list(
    "sig", ["#2e7d32", "#a5d6a7", "#fff9c4", "#ef9a9a"], N=256)
norm = mcolors.LogNorm(vmin=0.001, vmax=1.0)

im = ax.imshow(p_matrix, aspect="auto", cmap=cmap, norm=norm,
               interpolation="nearest")

# Cell annotations
for i in range(n_models):
    for j in range(n_tasks):
        p = p_matrix[i, j]
        n = n_matrix[i, j]
        if np.isnan(p):
            txt = "—"
            col = "#aaa"
        else:
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            txt = f"{p:.3f}{stars}\nn={n}"
            col = "white" if p < 0.15 else "#333"
        ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color=col,
                linespacing=1.4)

# Axis labels — colour y-tick labels by model family colour
ax.set_xticks(range(n_tasks))
ax.set_xticklabels(task_labels, fontsize=9, fontweight="bold")
ax.set_yticks(range(n_models))
ax.set_yticklabels(MODEL_ORDER, fontsize=8)
for tick, mdl in zip(ax.get_yticklabels(), MODEL_ORDER):
    tick.set_color(SHARED_MODEL_COLORS.get(mdl, "#333"))

ax.set_title("LongMK vs. all models — one-sided paired t-test\n"
             "p(LongMK > other), n = number of valid paired splits",
             fontsize=9, fontweight="bold", pad=8)

# Significance legend
sig_patches = [
    Patch(facecolor="#2e7d32", label="p < 0.01  (***)"),
    Patch(facecolor="#66bb6a", label="p < 0.05  (**)"),
    Patch(facecolor="#a5d6a7", label="p < 0.10  (*)"),
    Patch(facecolor="#fff9c4", label="p ≥ 0.10"),
]
ax.legend(handles=sig_patches, fontsize=7, loc="lower left",
          bbox_to_anchor=(0, -0.18), ncol=4, framealpha=0.9,
          edgecolor="#ccc", title="Significance", title_fontsize=7)

plt.colorbar(im, ax=ax, shrink=0.5, label="p-value", pad=0.02)
fig.tight_layout()

for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"benchmark_pvalues_longmk.{ext}",
                dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved benchmark_pvalues_longmk.png/pdf")
