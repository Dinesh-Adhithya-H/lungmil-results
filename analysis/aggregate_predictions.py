"""
Aggregate per-split predictions into two analysis tables.

Each sample appears in exactly ONE test split — no cross-split averaging.
Each sample has one test prediction per model.

Goal 1 — Unimodal vs multimodal fair comparison
  For each task (ACR cls, ACR surv, CLAD, Death), compare:
    - P1 per-modality (only on samples where that modality is present)
    - P1 ensemble (mean of available P1 probs/hazards per sample)
    - P2 variant/task (on same test set)
  Expected unimodal baseline: per-sample mean of available P1 scores,
  so samples missing a modality are not penalised for it.

Goal 2 — Reannotation candidates
  For ACR classification (the only task with discrete, reannotatable labels):
    error = prob_of_wrong_class (high = confidently wrong)
  Sorted by canonical model error (best available: mario_kempes > early_cls).
  Includes per-modality errors to show whether all models agree (label noise)
  or only some are wrong (hard samples).

Outputs
-------
results/predictions/
  predictions_all.csv           — full cohort, all predictions/hazards
  unimodal_comparison.csv       — per-split × per-task BACC / C-index table
  reannotation_candidates.csv   — ACR-labeled biopsies sorted by model error
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MODALITIES  = ["HE", "BAL", "CT", "Clinical"]
IN_DIR      = REPO / "results/predictions/raw"
OUT_DIR     = REPO / "results/predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# P2 (variant, task) specs — used to enumerate columns
P2_SPECS = [
    ("early",          "cls"),
    ("early",          "acr_surv"),
    ("early",          "clad_surv"),
    ("early",          "death_surv"),
    ("late",           "cls"),
    ("late",           "acr_surv"),
    ("late",           "clad_surv"),
    ("late",           "death_surv"),
    ("middle",         "cls"),
    ("middle",         "acr_surv"),
    ("middle",         "clad_surv"),
    ("middle",         "death_surv"),
    ("mario_kempes",   "mega"),
    ("longitudinal_mk","mega"),
]

# Endpoints: maps endpoint_name → (p1_hazard_col_prefix, time_col, event_col)
ENDPOINTS = {
    "acr_surv": ("h1_acr_surv",  "tte_next_acr",  "event_next_acr"),
    "clad":     ("h1_clad",      "clad_time",      "clad_event"),
    "death":    ("h1_death",     "death_time",      "death_event"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

dfs = []
for s in range(5):
    p = IN_DIR / f"split{s}_predictions.csv"
    if not p.exists():
        print(f"  [warn] missing split{s}")
        continue
    dfs.append(pd.read_csv(p, parse_dates=["anchor_dt"]))
    print(f"  Loaded split{s}: {len(dfs[-1])} rows")

if not dfs:
    print("No prediction files found. Run extract_predictions.py first.")
    sys.exit(1)

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal: {len(df)} rows  |  Labeled: {df['label'].notna().sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# Modality-wise P1 ensemble (mean of available preds per sample per task)
# ─────────────────────────────────────────────────────────────────────────────

def nanmean_row(row, cols):
    vals = [row[c] for c in cols if c in row.index and pd.notna(row[c])]
    return float(np.mean(vals)) if vals else float("nan")

# ACR classification: p1_acr_{mod}
p1_cls_cols = [f"p1_acr_{m}" for m in MODALITIES]
df["p1_ensemble_cls"] = df.apply(lambda r: nanmean_row(r, p1_cls_cols), axis=1)

# Survival endpoints: h1_{ep}_{mod}
for ep in ("acr_surv", "clad", "death"):
    cols = [f"h1_{ep}_{m}" for m in MODALITIES]
    df[f"p1_ensemble_h_{ep}"] = df.apply(lambda r: nanmean_row(r, cols), axis=1)

df["n_modalities"] = df[[f"has_{m}" for m in MODALITIES]].sum(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# ACR classification error
# error = prob assigned to wrong class (high → confidently wrong → reannotate)
#   label=1: error = 1 - pred_prob
#   label=0: error = pred_prob
#   label=None: NaN
# ─────────────────────────────────────────────────────────────────────────────

def cls_error(probs: pd.Series, labels: pd.Series) -> pd.Series:
    err = pd.Series(float("nan"), index=probs.index)
    labeled = labels.notna()
    pos = labeled & (labels == 1)
    neg = labeled & (labels == 0)
    err.loc[pos] = 1.0 - probs.loc[pos]
    err.loc[neg] = probs.loc[neg]
    return err

# P2 classification errors
for var, task in P2_SPECS:
    if task not in ("cls", "mega"):
        continue
    col = f"p2_{var}_{task}"
    if col in df.columns and df[col].notna().any():
        df[f"err_{col}"] = cls_error(df[col], df["label"])

# P1 per-modality errors
for mod in MODALITIES:
    col = f"p1_acr_{mod}"
    if col in df.columns:
        df[f"err_{col}"] = cls_error(df[col], df["label"])
df["err_p1_ensemble_cls"] = cls_error(df["p1_ensemble_cls"], df["label"])

# Canonical P2 error for ranking (best available model)
CANONICAL_ORDER = [
    ("mario_kempes",   "mega"),
    ("longitudinal_mk","mega"),
    ("early",          "cls"),
    ("late",           "cls"),
    ("middle",         "cls"),
]
canonical_err_col = None
for var, task in CANONICAL_ORDER:
    col = f"err_p2_{var}_{task}"
    if col in df.columns and df[col].notna().any():
        canonical_err_col = col
        break
print(f"Canonical error column: {canonical_err_col}")

if canonical_err_col:
    labeled_mask = df["label"].notna()
    df.loc[labeled_mask, "rank_err"] = (
        df.loc[labeled_mask, canonical_err_col]
        .rank(ascending=False, method="min", na_option="bottom")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Save full table
# ─────────────────────────────────────────────────────────────────────────────

df.to_csv(OUT_DIR / "predictions_all.csv", index=False)
print(f"Saved full table → {OUT_DIR / 'predictions_all.csv'}")


# ─────────────────────────────────────────────────────────────────────────────
# Reannotation candidates
# Only ACR-labeled biopsies, sorted by canonical model error
# ─────────────────────────────────────────────────────────────────────────────

labeled = df[df["label"].notna()].copy()
if canonical_err_col:
    labeled = labeled.sort_values(canonical_err_col, ascending=False)

base_cols = ["stem", "patient_id", "anchor_dt", "split", "acr_grade", "label",
             "n_modalities", "has_HE", "has_BAL", "has_CT", "has_Clinical",
             "rank_err"]
canon_col = [canonical_err_col] if canonical_err_col else []
p2_prob_cols  = [f"p2_{v}_{t}" for v, t in P2_SPECS
                 if t in ("cls","mega") and f"p2_{v}_{t}" in labeled.columns]
p2_err_cols   = [f"err_{c}" for c in p2_prob_cols if f"err_{c}" in labeled.columns]
p1_prob_cols  = [f"p1_acr_{m}" for m in MODALITIES if f"p1_acr_{m}" in labeled.columns]
p1_err_cols   = [f"err_p1_acr_{m}" for m in MODALITIES if f"err_p1_acr_{m}" in labeled.columns]
surv_cols     = ["tte_next_acr","event_next_acr","clad_time","clad_event",
                 "death_time","death_event"]
all_cols = (base_cols + canon_col + p2_prob_cols + p2_err_cols
            + p1_prob_cols + p1_err_cols
            + ["p1_ensemble_cls", "err_p1_ensemble_cls"]
            + [c for c in surv_cols if c in labeled.columns])
all_cols = [c for c in all_cols if c in labeled.columns]

labeled[all_cols].to_csv(OUT_DIR / "reannotation_candidates.csv", index=False)
print(f"Saved reannotation candidates ({len(labeled)} labeled biopsies) → "
      f"{OUT_DIR / 'reannotation_candidates.csv'}")


# ─────────────────────────────────────────────────────────────────────────────
# Unimodal vs multimodal comparison table
# Per split, per task: BACC (cls) and C-index (survival)
# ─────────────────────────────────────────────────────────────────────────────

def bacc(probs, labels, thr=0.5):
    probs  = np.asarray(probs,  dtype=float)
    labels = np.asarray(labels, dtype=float)
    preds  = (probs >= thr).astype(int)
    pos = labels == 1; neg = labels == 0
    tpr = preds[pos].mean()      if pos.any() else float("nan")
    tnr = (1-preds[neg]).mean()  if neg.any() else float("nan")
    if math.isnan(tpr) or math.isnan(tnr):
        return float("nan")
    return 0.5 * (tpr + tnr)


def c_index(hazards, times, events):
    h = np.asarray(hazards, dtype=float)
    t = np.asarray(times,   dtype=float)
    e = np.asarray(events,  dtype=float)
    valid = ~np.isnan(h) & ~np.isnan(t) & ~np.isnan(e)
    h, t, e = h[valid], t[valid], e[valid]
    if e.sum() == 0 or len(h) < 4:
        return float("nan")
    conc = disc = 0
    for i in range(len(h)):
        if not e[i]: continue
        for j in range(len(h)):
            if t[j] <= t[i] or i == j: continue
            conc += int(h[i] > h[j])
            disc += int(h[i] < h[j])
    total = conc + disc
    return conc / total if total > 0 else 0.5


records = []
for s in range(5):
    mask = (df["split"] == s)
    sub  = df[mask]
    if len(sub) == 0:
        continue
    labeled_sub = sub[sub["label"].notna()]
    row = {"split": s,
           "n_total": int(len(sub)),
           "n_labeled": int(len(labeled_sub))}

    # ── ACR CLASSIFICATION ──────────────────────────────────────────────────
    # P1 per modality (only on present samples)
    for mod in MODALITIES:
        col   = f"p1_acr_{mod}"
        valid = labeled_sub[col].notna() if col in sub.columns else pd.Series(False, index=sub.index)
        n_v   = valid.sum()
        row[f"bacc_p1_acr_{mod}"] = (bacc(labeled_sub.loc[valid, col].values,
                                           labeled_sub.loc[valid, "label"].values)
                                      if n_v > 5 else float("nan"))
        row[f"n_p1_acr_{mod}"] = int(n_v)

    # P1 ensemble (mean of available modality probs per sample)
    valid = labeled_sub["p1_ensemble_cls"].notna()
    row["bacc_p1_ensemble"] = (bacc(labeled_sub.loc[valid, "p1_ensemble_cls"].values,
                                     labeled_sub.loc[valid, "label"].values)
                                if valid.sum() > 5 else float("nan"))

    # P2 cls/mega variants
    for var, task in P2_SPECS:
        if task not in ("cls", "mega"):
            continue
        col = f"p2_{var}_{task}"
        if col not in sub.columns:
            continue
        valid = labeled_sub[col].notna()
        n_v   = valid.sum()
        row[f"bacc_p2_{var}_{task}"] = (bacc(labeled_sub.loc[valid, col].values,
                                              labeled_sub.loc[valid, "label"].values)
                                         if n_v > 5 else float("nan"))
        row[f"n_p2_{var}_{task}"] = int(n_v)

    # ── SURVIVAL ENDPOINTS ──────────────────────────────────────────────────
    for ep, (p1_h_prefix, t_col, e_col) in ENDPOINTS.items():
        t_vals = sub[t_col];  e_vals = sub[e_col]

        # P1 per modality
        for mod in MODALITIES:
            h_col = f"{p1_h_prefix}_{mod}"
            if h_col not in sub.columns:
                continue
            valid = sub[h_col].notna() & t_vals.notna() & e_vals.notna()
            row[f"ci_{ep}_p1_{mod}"] = (c_index(sub.loc[valid, h_col].values,
                                                  t_vals[valid].values,
                                                  e_vals[valid].values)
                                         if valid.sum() > 5 else float("nan"))

        # P1 ensemble hazard
        ens_col = f"p1_ensemble_h_{ep}"
        if ens_col in sub.columns:
            valid = sub[ens_col].notna() & t_vals.notna() & e_vals.notna()
            row[f"ci_{ep}_p1_ensemble"] = (c_index(sub.loc[valid, ens_col].values,
                                                     t_vals[valid].values,
                                                     e_vals[valid].values)
                                            if valid.sum() > 5 else float("nan"))

        # P2: pick relevant variant+task columns for this endpoint
        for var, task in P2_SPECS:
            # which P2 tasks are relevant for each endpoint?
            relevant = {
                "acr_surv": ["acr_surv", "mega"],
                "clad":     ["clad_surv", "mega"],
                "death":    ["death_surv", "mega"],
            }.get(ep, [])
            if task not in relevant:
                continue
            h_col = f"h2_{ep}_{var}_{task}"
            if h_col not in sub.columns:
                continue
            valid = sub[h_col].notna() & t_vals.notna() & e_vals.notna()
            row[f"ci_{ep}_p2_{var}_{task}"] = (c_index(sub.loc[valid, h_col].values,
                                                         t_vals[valid].values,
                                                         e_vals[valid].values)
                                                if valid.sum() > 5 else float("nan"))

    records.append(row)

comp_df = pd.DataFrame(records)

# mean ± std rows
numeric_cols = [c for c in comp_df.columns if c != "split"]
mean_row = {"split": "mean"}
std_row  = {"split": "std"}
for c in numeric_cols:
    vals = comp_df[c].dropna()
    mean_row[c] = round(float(vals.mean()), 4) if len(vals)    else float("nan")
    std_row[c]  = round(float(vals.std()),  4) if len(vals) > 1 else float("nan")
comp_df = pd.concat([comp_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

comp_df.to_csv(OUT_DIR / "unimodal_comparison.csv", index=False)
print(f"Saved comparison table → {OUT_DIR / 'unimodal_comparison.csv'}")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n=== ACR BACC (P1 per-modality vs P1 ensemble vs P2) ===")
bacc_cols = (["split", "n_labeled"]
             + [f"bacc_p1_acr_{m}" for m in MODALITIES]
             + ["bacc_p1_ensemble"]
             + [f"bacc_p2_{v}_{t}" for v, t in P2_SPECS
                if t in ("cls","mega") and f"bacc_p2_{v}_{t}" in comp_df.columns])
print(comp_df[[c for c in bacc_cols if c in comp_df.columns]].to_string(
    index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) and not math.isnan(x) else "   "))

print("\n=== Survival C-index (P1 per-modality vs P2) ===")
for ep in ("acr_surv", "clad", "death"):
    ci_cols = (["split"]
               + [f"ci_{ep}_p1_{m}" for m in MODALITIES]
               + [f"ci_{ep}_p1_ensemble"]
               + [c for c in comp_df.columns if c.startswith(f"ci_{ep}_p2_")])
    ci_cols = [c for c in ci_cols if c in comp_df.columns]
    if len(ci_cols) > 1:
        print(f"\n  {ep.upper()}")
        print(comp_df[ci_cols].to_string(
            index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) and not math.isnan(x) else "   "))
