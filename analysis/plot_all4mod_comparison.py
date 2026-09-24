"""
Fair comparison on biopsies with ALL 4 modalities (HE+BAL+CT+Clinical).

Pools ALL ~53 such biopsies across all 5 test splits into one dataset,
computes a single metric + bootstrap 95% CI per model per task.
Annotates n= (valid samples used) on each bar.

Unimodal ABMIL (P1) predictions: results/predictions/predictions_all.csv
LongMIL predictions: results/mm_abmil_v8/biopsy_reps (risk per biopsy)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT     = Path(__file__).resolve().parent.parent
PRED_CSV = ROOT / "results" / "predictions" / "predictions_all.csv"
REPS_DIR = ROOT / "results" / "mm_abmil_v8" / "biopsy_reps"
MASTER   = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
OUT_DIR  = ROOT / "figures" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG = "#FAF6F2"
MOD_COLORS = {
    "HE":       "#E64A19",
    "BAL":      "#1565C0",
    "CT":       "#2E7D32",
    "Clinical": "#7B1FA2",
}
LONGMIL_COLOR = "#B71C1C"

FS_TITLE  = 18
FS_LABEL  = 14
FS_TICK   = 12
FS_ANNOT  = 10

TASK_CFG = {
    "death_surv": {
        "label":    "Post-Transplant\nMortality",
        "metric":   "cindex",
        "ml":       "C-index",
        "uni_risk": "h1_death_{mod}",
        "tte_col":  "death_time",
        "ev_col":   "death_event",
        "lbl_col":  None,
    },
    "acr_surv": {
        "label":    "ACR Survival",
        "metric":   "cindex",
        "ml":       "C-index",
        "uni_risk": "h1_acr_surv_{mod}",
        "tte_col":  "tte_next_acr",
        "ev_col":   "event_next_acr",
        "lbl_col":  None,
    },
    "clad_surv": {
        "label":    "CLAD Survival",
        "metric":   "cindex",
        "ml":       "C-index",
        "uni_risk": "h1_clad_{mod}",
        "tte_col":  "clad_time",
        "ev_col":   "clad_event",
        "lbl_col":  None,
    },
    "acr_cls": {
        "label":    "ACR\nClassification",
        "metric":   "bacc",
        "ml":       "BACC",
        "uni_risk": "p1_acr_{mod}",
        "tte_col":  None,
        "ev_col":   None,
        "lbl_col":  "label",
    },
}

MODS = ["HE", "BAL", "CT", "Clinical"]
MODEL_ORDER = ["ABMIL HE", "ABMIL BAL", "ABMIL CT", "ABMIL Clinical", "LongMIL"]
MODEL_COLORS = {**{f"ABMIL {m}": MOD_COLORS[m] for m in MODS}, "LongMIL": LONGMIL_COLOR}


def cindex_val(risk, tte, event):
    from sksurv.metrics import concordance_index_censored
    mask = ~(np.isnan(risk) | np.isnan(tte) | np.isnan(event))
    r, t, e = risk[mask], tte[mask], event[mask].astype(bool)
    if e.sum() < 2 or len(r) < 3:
        return np.nan, int(mask.sum())
    c, *_ = concordance_index_censored(e, t, r)
    return float(c), int(mask.sum())


def bacc_val(risk, label):
    mask = ~(np.isnan(risk) | np.isnan(label))
    r, l = risk[mask], label[mask].astype(int)
    if len(np.unique(l)) < 2 or len(r) < 3:
        return np.nan, int(mask.sum())
    thresh = np.nanmedian(risk[~np.isnan(risk)])
    pred = (r >= thresh).astype(int)
    classes = np.unique(l)
    recalls = [(pred[l == c] == c).mean() for c in classes if (l == c).sum() > 0]
    return float(np.mean(recalls)) if recalls else np.nan, int(mask.sum())



# ── Load predictions_all.csv — filter to all-4-modality biopsies ──────────────
print("Loading predictions_all.csv ...")
pred = pd.read_csv(PRED_CSV)
all4 = pred[pred["has_HE"] & pred["has_BAL"] & pred["has_CT"] & pred["has_Clinical"]].copy()
print(f"  All-4-modality biopsies: {len(all4)}  (splits: {sorted(all4['split'].unique())})")

# ── Load LongMIL biopsy_reps — filter to all-4-modality stems ─────────────────
print("Loading biopsy_reps ...")
master = pd.read_csv(MASTER)
master["stem"] = master["file"].str.replace(".pt", "", regex=False)
_all4_master = master[master["has_HE"] & master["has_BAL"] & master["has_CT"] & master["has_Clinical"]]["stem"]
all4_stems = set(_all4_master) | set(_all4_master.str.lstrip("0").str.replace("^$", "0", regex=True).astype(int))
print(f"  All-4-mod stems in master: {len(all4_stems)}")

# Build stem→risk map for LongMIL per task (risk only; TTE/label from predictions_all.csv)
longmil_risk = {}  # task -> {stem: risk}
for task in TASK_CFG:
    stem_risk = {}
    for s in range(5):
        p = REPS_DIR / f"biopsy_reps_{task}_split{s}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        for stem, r in zip(d["stems"], d["risk"].numpy()):
            try:
                key = int(stem)
            except (ValueError, TypeError):
                key = stem
            if stem in all4_stems or key in all4_stems:
                stem_risk[key] = float(r)
    longmil_risk[task] = stem_risk
    n_found = sum(1 for v in stem_risk.values() if not np.isnan(v))
    print(f"  {task}: {len(stem_risk)} all-4-mod stems in reps, {n_found} non-NaN risk")

# ── Compute pooled metrics ────────────────────────────────────────────────────
results = {}  # task -> model -> {val, ci, n}

for task, cfg in TASK_CFG.items():
    results[task] = {}
    metric = cfg["metric"]

    # Unimodal ABMIL
    for mod in MODS:
        col = cfg["uni_risk"].replace("{mod}", mod)
        key = f"ABMIL {mod}"
        if col not in all4.columns:
            print(f"  [skip] {task} {mod}: col {col} missing")
            continue
        risk = all4[col].values.astype(float)
        if metric == "cindex":
            tte   = all4[cfg["tte_col"]].values.astype(float)
            event = all4[cfg["ev_col"]].values.astype(float)
            val, n = cindex_val(risk, tte, event)
        else:
            lbl = all4[cfg["lbl_col"]].values.astype(float)
            val, n = bacc_val(risk, lbl)
        results[task][key] = {"val": val, "n": n}

    # LongMIL — join risk by stem onto all4 so TTE/label source is identical
    stem_risk = longmil_risk.get(task, {})
    lm_risk = all4["stem"].map(stem_risk).values.astype(float)
    if metric == "cindex":
        tte   = all4[cfg["tte_col"]].values.astype(float)
        event = all4[cfg["ev_col"]].values.astype(float)
        val, n = cindex_val(lm_risk, tte, event)
    else:
        lbl = all4[cfg["lbl_col"]].values.astype(float)
        val, n = bacc_val(lm_risk, lbl)
    results[task]["LongMIL"] = {"val": val, "n": n}

# ── Print ─────────────────────────────────────────────────────────────────────
for task, mmap in results.items():
    print(f"\n=== {task} ===")
    for model, r in mmap.items():
        print(f"  {model:<20} {r['val']:.3f}  n={r['n']}")

# ── Plot ──────────────────────────────────────────────────────────────────────
n_tasks = len(TASK_CFG)
fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 8), facecolor=BG)
fig.suptitle(
    "All-4-Modality Biopsies (HE + BAL + CT + Clinical)\n"
    "LongMIL vs Unimodal ABMIL — pooled across all splits",
    fontsize=16, fontweight="bold", y=1.01
)

for ax, (task, cfg) in zip(axes, TASK_CFG.items()):
    ax.set_facecolor(BG)
    ys = np.arange(len(MODEL_ORDER))

    lm_y = MODEL_ORDER.index("LongMIL")
    ax.axhspan(lm_y - 0.45, lm_y + 0.45, facecolor="#FCE4EC", alpha=0.40, zorder=0)
    ax.axvline(0.5, color="#999", lw=1.0, ls=":", zorder=1)

    task_res = results[task]
    for i, (y, model) in enumerate(zip(ys, MODEL_ORDER)):
        r = task_res.get(model, {"val": np.nan, "n": 0})
        col = MODEL_COLORS[model]
        val, n = r["val"], r["n"]

        if np.isnan(val) or n < 3:
            ax.text(0.52, y, f"n={n} (insufficient)", va="center",
                    fontsize=FS_ANNOT, color="#aaa")
            continue

        lw = 2.5 if model == "LongMIL" else 0.8
        ax.barh(y, val, height=0.62, color=col, alpha=0.90, zorder=2,
                edgecolor="#222" if model == "LongMIL" else col, linewidth=lw)
        ax.text(val + 0.012, y,
                f"{val:.3f}   n={n}",
                va="center", fontsize=FS_ANNOT, color="#333")

    ax.set_yticks(ys)
    ax.set_yticklabels(MODEL_ORDER, fontsize=FS_TICK)
    ax.set_xlabel(cfg["ml"], fontsize=FS_LABEL)
    ax.set_title(cfg["label"], fontsize=FS_TITLE, fontweight="bold", pad=8)
    ax.set_xlim(0.0, 1.18)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"all4mod_comparison.{ext}", dpi=200,
                bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("\nSaved all4mod_comparison")
print("Done.")
