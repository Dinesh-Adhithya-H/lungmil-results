"""
Unimodal ABMIL vs LongMIL on biopsies with ALL 4 modalities (HE+BAL+CT+Clinical).

For each task, filter to biopsies that have all 4 modalities, then compare:
  - Unimodal ABMIL HE / BAL / CT / Clinical (from predictions_all.csv h1_* columns)
  - LongMIL (from biopsy_reps risk scores, joined by stem)

Metrics computed per split, shown as mean ± std across 5 splits.
One 4-panel figure.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sksurv.metrics import concordance_index_censored

ROOT     = Path(__file__).resolve().parent.parent
PRED_CSV = ROOT / "results" / "predictions" / "predictions_all.csv"
REPS_DIR = ROOT / "results" / "mm_abmil_v8" / "biopsy_reps"
OUT_DIR  = ROOT / "figures" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG = "#FAF6F2"
FS_TITLE = 16
FS_LABEL = 13
FS_TICK  = 11
FS_ANNOT = 10

MOD_COLORS = {
    "HE":       "#E64A19",
    "BAL":      "#1565C0",
    "CT":       "#2E7D32",
    "Clinical": "#7B1FA2",
    "LongMIL":  "#B71C1C",
}

TASK_CFG = {
    "death_surv": {
        "label":      "Post-Transplant Mortality",
        "metric":     "cindex",
        "metric_lbl": "C-index",
        "tte_col":    "death_time",
        "ev_col":     "death_event",
        "label_col":  None,
        "uni_cols":   {"HE":"h1_death_HE","BAL":"h1_death_BAL",
                       "CT":"h1_death_CT","Clinical":"h1_death_Clinical"},
    },
    "acr_surv": {
        "label":      "ACR Survival",
        "metric":     "cindex",
        "metric_lbl": "C-index",
        "tte_col":    "tte_next_acr",
        "ev_col":     "event_next_acr",
        "label_col":  None,
        "uni_cols":   {"HE":"h1_acr_surv_HE","BAL":"h1_acr_surv_BAL",
                       "CT":"h1_acr_surv_CT","Clinical":"h1_acr_surv_Clinical"},
    },
    "clad_surv": {
        "label":      "CLAD Survival",
        "metric":     "cindex",
        "metric_lbl": "C-index",
        "tte_col":    "clad_time",
        "ev_col":     "clad_event",
        "label_col":  None,
        "uni_cols":   {"HE":"h1_clad_HE","BAL":"h1_clad_BAL",
                       "CT":"h1_clad_CT","Clinical":"h1_clad_Clinical"},
    },
    "acr_cls": {
        "label":      "ACR Classification",
        "metric":     "bacc",
        "metric_lbl": "Balanced Accuracy",
        "tte_col":    None,
        "ev_col":     None,
        "label_col":  "label",
        "uni_cols":   {"HE":"p1_acr_HE","BAL":"p1_acr_BAL",
                       "CT":"p1_acr_CT","Clinical":"p1_acr_Clinical"},
    },
}

MOD_ORDER = ["HE", "BAL", "CT", "Clinical", "LongMIL"]


def cindex_safe(risk, tte, event):
    mask = ~(np.isnan(risk) | np.isnan(tte) | np.isnan(event))
    r, t, e = risk[mask], tte[mask], event[mask].astype(bool)
    if e.sum() < 2 or len(r) < 3:
        return np.nan
    try:
        c, *_ = concordance_index_censored(e, t, r)
        return float(c)
    except Exception:
        return np.nan


def bacc_safe(pred_prob, label):
    mask = ~(np.isnan(pred_prob) | np.isnan(label))
    p, l = pred_prob[mask], label[mask].astype(int)
    if len(np.unique(l)) < 2 or len(p) < 5:
        return np.nan
    thresh = np.nanmedian(pred_prob[~np.isnan(pred_prob)])
    pred = (p >= thresh).astype(int)
    classes = np.unique(l)
    recalls = [(pred[l == c] == c).mean() for c in classes if (l == c).sum() > 0]
    return float(np.mean(recalls))


# ── Load predictions_all, filter to all-4-modality biopsies ──────────────────
print("Loading predictions_all.csv ...")
pred = pd.read_csv(PRED_CSV)
all4 = pred[
    (pred["has_HE"] == True) &
    (pred["has_BAL"] == True) &
    (pred["has_CT"] == True) &
    (pred["has_Clinical"] == True)
].copy()
print(f"  All-4-modality biopsies: {len(all4)} across {all4['patient_id'].nunique()} patients")
print(f"  Splits: {sorted(all4['split'].unique())}")

# ── Load LongMIL risk scores from biopsy_reps, merge by stem ─────────────────
print("Loading LongMIL biopsy reps ...")
for task in TASK_CFG:
    risk_map = {}
    for s in range(5):
        p = REPS_DIR / f"biopsy_reps_{task}_split{s}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        stems = list(d["stems"])
        risks = d["risk"].numpy() if "risk" in d else np.full(len(stems), np.nan)
        for stem, r in zip(stems, risks):
            # biopsy_reps stems are zero-padded (e.g. '00019'); predictions use int
            risk_map[int(stem)] = float(r)
    all4[f"longmil_{task}"] = all4["stem"].map(risk_map)
    n_valid = all4[f"longmil_{task}"].notna().sum()
    print(f"  {task}: LongMIL risk available for {n_valid}/{len(all4)} biopsies")

# ── Compute metrics per split per model per task ──────────────────────────────
results = {}   # results[task][model] = list of per-split values

for task, cfg in TASK_CFG.items():
    results[task] = {m: [] for m in MOD_ORDER}
    for split, grp in all4.groupby("split"):
        tte_col   = cfg["tte_col"]
        ev_col    = cfg["ev_col"]
        label_col = cfg["label_col"]

        for mod, col in cfg["uni_cols"].items():
            if col not in grp.columns:
                results[task][mod].append(np.nan)
                continue
            risk = grp[col].values.astype(float)
            if cfg["metric"] == "cindex":
                tte   = grp[tte_col].values.astype(float)
                event = grp[ev_col].values.astype(float)
                val = cindex_safe(risk, tte, event)
            else:
                lbl = grp[label_col].values.astype(float)
                val = bacc_safe(risk, lbl)
            results[task][mod].append(val)

        # LongMIL
        lm_col = f"longmil_{task}"
        risk = grp[lm_col].values.astype(float)
        if cfg["metric"] == "cindex":
            tte   = grp[tte_col].values.astype(float)
            event = grp[ev_col].values.astype(float)
            val = cindex_safe(risk, tte, event)
        else:
            lbl = grp[label_col].values.astype(float)
            val = bacc_safe(risk, lbl)
        results[task]["LongMIL"].append(val)

# ── Print summary ─────────────────────────────────────────────────────────────
for task in TASK_CFG:
    print(f"\n{task}")
    for mod in MOD_ORDER:
        vals = [v for v in results[task][mod] if not np.isnan(v)]
        if vals:
            print(f"  {mod:12s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}  (n_splits={len(vals)})")
        else:
            print(f"  {mod:12s}: no data")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 7), facecolor=BG)
fig.suptitle(
    "Unimodal ABMIL vs LongMIL — biopsies with ALL 4 modalities (HE+BAL+CT+Clinical)\n"
    f"n={len(all4)} biopsies · {all4['patient_id'].nunique()} patients",
    fontsize=14, fontweight="bold", y=1.02
)

for ax, (task, cfg) in zip(axes, TASK_CFG.items()):
    ax.set_facecolor(BG)
    ys = np.arange(len(MOD_ORDER))
    for i, mod in enumerate(MOD_ORDER):
        vals = [v for v in results[task][mod] if not np.isnan(v)]
        col  = MOD_COLORS[mod]
        if not vals:
            ax.text(0.5, i, "no data", ha="center", va="center",
                    fontsize=FS_ANNOT, color="#aaa")
            continue
        mean_v = np.mean(vals)
        std_v  = np.std(vals)
        ax.barh(i, mean_v, height=0.60, color=col,
                alpha=0.90 if mod == "LongMIL" else 0.78, zorder=2)
        if std_v > 0:
            ax.errorbar(mean_v, i, xerr=std_v, fmt="none",
                        ecolor="#222", elinewidth=1.5, capsize=5, zorder=4)
        for sv in vals:
            ax.scatter(sv, i, s=25, color="white", edgecolors=col,
                       linewidths=1.0, zorder=5, alpha=0.95)
        ax.text(mean_v + std_v + 0.01, i,
                f"{mean_v:.3f}±{std_v:.3f}\nn={len(vals)}splits",
                va="center", fontsize=FS_ANNOT-1, color="#444")

    ax.set_yticks(ys)
    ax.set_yticklabels(MOD_ORDER, fontsize=FS_TICK)
    ax.set_xlabel(cfg["metric_lbl"], fontsize=FS_LABEL)
    ax.set_title(cfg["label"], fontsize=FS_TITLE, fontweight="bold", pad=6)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_xlim(0, 1.15)
    ax.axvline(0.5, color="#999", lw=0.9, ls="--", zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    # separator between unimodal and LongMIL
    ax.axhline(len(MOD_ORDER) - 1 - 0.5, color="#bbb", lw=1.2, ls=":", zorder=1)

legend_els = [Patch(facecolor=MOD_COLORS[m], label=f"{'ABMIL ' if m != 'LongMIL' else ''}{m}")
              for m in MOD_ORDER]
fig.legend(handles=legend_els, fontsize=FS_TICK, ncol=len(MOD_ORDER),
           loc="lower center", bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"all4mod_comparison.{ext}", dpi=200,
                bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("\nSaved all4mod_comparison")
print("All done.")
