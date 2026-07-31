"""
Unimodal ablation summary — all models, all tasks, all splits.

Reads the `unimodal_ablation` block stored in every metrics_*_final.json and
produces:
  1. Console tables: mean ± std across 5 splits per (model, task, modality)
  2. CSV: full split-level data + mean/std
  3. Markdown table: best-model-per-task unimodal breakdown (for paper Supplementary)
  4. Bar plots: per-task modality comparison across model families

Run via sbatch interpretability/submit_unimodal_ablation.sh
"""

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT    = Path(__file__).resolve().parent.parent
PHASE2  = ROOT / "results" / "mm_abmil_v8" / "phase2"
OUT_DIR = ROOT / "interpretability" / "unimodal_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODALITY_ORDER  = ["HE", "BAL", "CT", "Clinical"]
MOD_COLORS      = {"HE": "#e74c3c", "BAL": "#3498db", "CT": "#2ecc71", "Clinical": "#9b59b6"}

TASK_META = {
    "cls":       {"label": "ACR cls",       "metric": "bacc",    "metric_label": "BACC",    "chance": 0.5},
    "acr_surv":  {"label": "ACR surv",      "metric": "c_index", "metric_label": "C-index", "chance": 0.5},
    "clad_surv": {"label": "CLAD surv",     "metric": "c_index", "metric_label": "C-index", "chance": 0.5},
    "death_surv":{"label": "Death surv",    "metric": "c_index", "metric_label": "C-index", "chance": 0.5},
}

# Human-readable variant names
VARIANT_LABELS = {
    "early":            "Early fusion",
    "late":             "Late fusion",
    "middle":           "Middle fusion",
    "set_mil_mt":       "SetMIL-MT (SAB)",
    "set_mil_mt_no_sab":"SetMIL-MT (no SAB)",
    "set_mil_no_sab":   "SetMIL (single-task)",
}

# Best model per task (as reported in paper)
BEST_MODEL_PER_TASK = {
    "cls":       "set_mil_mt_no_sab",
    "acr_surv":  "early",          # best with ablation data; longitudinal has no ablation stored
    "clad_surv": "set_mil_mt",
    "death_surv":"early",          # best with ablation data
}

# ── 1. Collect all unimodal ablation rows ─────────────────────────────────────
rows = []
for mf in sorted(PHASE2.glob("split*_fold0/*/metrics_*_final.json")):
    split_fold   = mf.parent.parent.name
    variant_task = mf.parent.name
    split        = int(split_fold.replace("split","").replace("_fold0",""))

    d = json.loads(mf.read_text())
    ablation = d.get("unimodal_ablation")
    if not ablation:
        continue

    # Parse variant and task from directory name
    # e.g. set_mil_mt_no_sab_cls, early_acr_surv, late_death_surv
    for task_key in ["death_surv", "clad_surv", "acr_surv", "cls"]:
        if variant_task.endswith(f"_{task_key}") or variant_task.endswith(f"_{task_key.replace('_surv','')}"):
            task = task_key
            variant = variant_task[:-(len(task_key)+1)]
            break
    else:
        # fallback: last segment is task
        parts = variant_task.rsplit("_", 1)
        variant, task = parts[0], parts[1] if len(parts) == 2 else ("unknown", variant_task)
        # map surv suffix
        if task in ("surv",):
            task = variant_task.rsplit("_", 2)[-2] + "_surv"
            variant = variant_task.rsplit("_", 2)[0]

    for mod, metrics in ablation.items():
        metric_key = "bacc" if "bacc" in metrics else "c_index"
        rows.append({
            "split":   split,
            "variant": variant,
            "task":    task,
            "modality": mod,
            "metric":  metric_key,
            "value":   metrics.get(metric_key, np.nan),
            "n":       metrics.get("n", np.nan),
            "auc":     metrics.get("auc", np.nan),
        })

df = pd.DataFrame(rows)
df = df[df["variant"].isin(VARIANT_LABELS)]  # keep known variants only

print(f"Loaded {len(df)} ablation rows across "
      f"{df['split'].nunique()} splits, "
      f"{df['variant'].nunique()} variants, "
      f"{df['task'].nunique()} tasks.")

# ── 2. Compute mean ± std across splits ──────────────────────────────────────
summary = (
    df.groupby(["variant","task","modality","metric"])["value"]
    .agg(["mean","std","count"])
    .reset_index()
)
summary.columns = ["variant","task","modality","metric","mean","std","count"]
summary["mean_std"] = summary.apply(
    lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}" if not np.isnan(r['std']) else f"{r['mean']:.3f}", axis=1
)

# ── 3. Console tables ─────────────────────────────────────────────────────────
print("\n" + "="*80)
print("UNIMODAL ABLATION: mean ± std across 5 splits")
print("="*80)

for task_key, tmeta in TASK_META.items():
    sub = summary[summary["task"] == task_key]
    if sub.empty:
        continue
    print(f"\n── {tmeta['label']} ({tmeta['metric_label']}) ──")
    pivot = sub.pivot_table(index="variant", columns="modality", values="mean_std", aggfunc="first")
    # reorder columns
    pivot = pivot.reindex(columns=[m for m in MODALITY_ORDER if m in pivot.columns])
    # rename rows
    pivot.index = [VARIANT_LABELS.get(v, v) for v in pivot.index]
    print(pivot.to_string())

# ── 4. Markdown table for paper (best model per task) ────────────────────────
print("\n" + "="*80)
print("MARKDOWN TABLE — best model per task (for paper Supplementary Note 2)")
print("="*80)

md_lines = []
for task_key, tmeta in TASK_META.items():
    best_v = BEST_MODEL_PER_TASK.get(task_key)
    sub = summary[(summary["task"] == task_key) & (summary["variant"] == best_v)]
    if sub.empty:
        continue

    md_lines.append(f"\n**{tmeta['label']} ({VARIANT_LABELS.get(best_v, best_v)}) — {tmeta['metric_label']}**\n")
    md_lines.append(f"| Modality | Mean | s.d. | n (avg) |")
    md_lines.append(f"|---|---|---|---|")

    # n from raw data
    n_df = df[(df["task"]==task_key) & (df["variant"]==best_v)].groupby("modality")["n"].mean()

    for mod in MODALITY_ORDER:
        row = sub[sub["modality"] == mod]
        if row.empty:
            continue
        r = row.iloc[0]
        n_avg = n_df.get(mod, np.nan)
        n_str = f"{n_avg:.0f}" if not np.isnan(n_avg) else "—"
        md_lines.append(f"| {mod} | {r['mean']:.3f} | {r['std']:.3f} | {n_str} |")

for l in md_lines:
    print(l)

# ── 5. Save CSVs ──────────────────────────────────────────────────────────────
df.to_csv(OUT_DIR / "unimodal_ablation_raw.csv", index=False)
summary.to_csv(OUT_DIR / "unimodal_ablation_summary.csv", index=False)
print(f"\nSaved CSVs to {OUT_DIR}/")

# ── 6. Bar plots — per task, all models, all modalities ──────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Unimodal ablation: per-modality performance in multimodal model\n"
             "Mean ± std across 5 splits", fontsize=13, fontweight="bold")

for ax, (task_key, tmeta) in zip(axes.flat, TASK_META.items()):
    sub = summary[summary["task"] == task_key].copy()
    sub = sub[sub["variant"].isin(VARIANT_LABELS)]
    if sub.empty:
        ax.set_visible(False)
        continue

    variants_order = [v for v in VARIANT_LABELS if v in sub["variant"].unique()]
    n_variants = len(variants_order)
    n_mods = len(MODALITY_ORDER)
    width = 0.15
    x = np.arange(n_variants)

    for i, mod in enumerate(MODALITY_ORDER):
        mod_sub = sub[sub["modality"] == mod].set_index("variant")
        means = [mod_sub.loc[v, "mean"] if v in mod_sub.index else np.nan for v in variants_order]
        stds  = [mod_sub.loc[v, "std"]  if v in mod_sub.index else 0       for v in variants_order]
        offset = (i - n_mods/2 + 0.5) * width
        ax.bar(x + offset, means, width, label=mod, color=MOD_COLORS[mod],
               alpha=0.85, yerr=stds, capsize=3, error_kw={"linewidth": 0.8})

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants_order],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(tmeta["metric_label"], fontsize=10)
    ax.set_title(tmeta["label"], fontsize=11, fontweight="bold")
    ax.set_ylim(0.3, 1.0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if task_key == "cls":
        ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
fig.savefig(OUT_DIR / "unimodal_ablation_barplot.png", dpi=180, bbox_inches="tight")
fig.savefig(OUT_DIR / "unimodal_ablation_barplot.pdf", dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved bar plot to {OUT_DIR}/unimodal_ablation_barplot.png")

# ── 7. Per-task heatmap: variant × modality ───────────────────────────────────
fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
fig2.suptitle("Unimodal ablation heatmap (mean across splits)", fontsize=13, fontweight="bold")

import matplotlib.colors as mcolors
cmap = plt.cm.RdYlGn

for ax, (task_key, tmeta) in zip(axes2, TASK_META.items()):
    sub = summary[(summary["task"] == task_key) & summary["variant"].isin(VARIANT_LABELS)].copy()
    pivot = sub.pivot_table(index="variant", columns="modality", values="mean", aggfunc="mean")
    pivot = pivot.reindex(index=[v for v in VARIANT_LABELS if v in pivot.index],
                          columns=[m for m in MODALITY_ORDER if m in pivot.columns])
    pivot.index = [VARIANT_LABELS.get(v, v) for v in pivot.index]

    im = ax.imshow(pivot.values, cmap=cmap, vmin=0.4, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(f"{tmeta['label']}\n({tmeta['metric_label']})", fontsize=10, fontweight="bold")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.45 < val < 0.85 else "white")

    plt.colorbar(im, ax=ax, shrink=0.7)

plt.tight_layout()
fig2.savefig(OUT_DIR / "unimodal_ablation_heatmap.png", dpi=180, bbox_inches="tight")
fig2.savefig(OUT_DIR / "unimodal_ablation_heatmap.pdf", dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved heatmap to {OUT_DIR}/unimodal_ablation_heatmap.png")
print("\nDone.")
