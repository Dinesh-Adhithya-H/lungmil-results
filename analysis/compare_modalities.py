#!/usr/bin/env python3
"""
Unimodal vs multimodal comparison tables — one per task.

Rows
----
P1 unimodal     : per-modality score on subset of test samples that have that modality
P1 wtd ensemble : prevalence-weighted average of unimodal predictions across present modalities
P2 variants     : from final metrics JSONs (early/late/middle/mario_kempes/longitudinal_mk)

Columns: s0 s1 s2 s3 s4  mean±std

Tasks: ACR cls (BACC), ACR surv (C-index), CLAD (C-index), Death (C-index)
"""

import json, math, statistics
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

REPO    = Path("/home/aih/dinesh.haridoss/chicago_mil")
PRED_DIR = REPO / "results/predictions/raw"
P2_BASE  = REPO / "results/mm_abmil_v8/phase2"
OUT_DIR  = REPO / "results/predictions"

MODALITIES = ["HE", "BAL", "CT", "Clinical"]
SPLITS     = list(range(5))

# ── C-index (Harrell) ─────────────────────────────────────────────────────────
def concordance_index(times, hazards, events):
    """Returns Harrell's C-index. Higher hazard = higher risk."""
    times   = np.asarray(times,   dtype=float)
    hazards = np.asarray(hazards, dtype=float)
    events  = np.asarray(events,  dtype=float)
    mask = np.isfinite(times) & np.isfinite(hazards) & np.isfinite(events)
    times, hazards, events = times[mask], hazards[mask], events[mask]
    if events.sum() == 0:
        return float("nan")
    concordant = discordant = 0
    for i in range(len(times)):
        if events[i] == 0:
            continue
        for j in range(len(times)):
            if times[j] <= times[i]:
                continue
            if hazards[i] > hazards[j]:
                concordant += 1
            elif hazards[i] < hazards[j]:
                discordant += 1
            else:
                concordant += 0.5; discordant += 0.5
    total = concordant + discordant
    return concordant / total if total > 0 else float("nan")

def fast_cidx(times, hazards, events):
    """Vectorised C-index via lifelines if available, else slow loop."""
    try:
        from lifelines.utils import concordance_index as li_ci
        mask = np.isfinite(times) & np.isfinite(hazards) & np.isfinite(events)
        if mask.sum() == 0 or np.asarray(events)[mask].sum() == 0:
            return float("nan")
        return li_ci(times[mask], -hazards[mask], events[mask])
    except ImportError:
        return concordance_index(times, hazards, events)

# ── BACC at 0.5 threshold ─────────────────────────────────────────────────────
def bacc(labels, probs):
    mask = np.isfinite(probs) & (labels >= 0)
    if mask.sum() == 0:
        return float("nan")
    preds = (probs[mask] >= 0.5).astype(int)
    labs  = labels[mask].astype(int)
    if len(np.unique(labs)) < 2:
        return float("nan")
    return balanced_accuracy_score(labs, preds)

# ── Load per-split test predictions ──────────────────────────────────────────
def load_split(split):
    f = PRED_DIR / f"split{split}_predictions.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    # keep only test rows (split column = this split)
    df = df[df["split"] == split].copy()
    return df

# ── Weighted ensemble for one task ───────────────────────────────────────────
def weighted_ensemble(df, score_cols, weights):
    """
    score_cols : dict {mod: col_name}
    weights    : dict {mod: prevalence_weight}  (from test set, unnormalised per-sample)
    Returns array of per-sample weighted scores (NaN where no modality present).
    """
    n = len(df)
    scores = np.full(n, np.nan)
    for i, (_, row) in enumerate(df.iterrows()):
        w_sum = 0.0; s_sum = 0.0
        for mod, col in score_cols.items():
            if row.get(f"has_{mod}", 0) != 1:
                continue
            v = row.get(col, np.nan)
            if not math.isfinite(float(v)) if v is not None else True:
                continue
            w = weights[mod]
            w_sum += w; s_sum += w * float(v)
        if w_sum > 0:
            scores[i] = s_sum / w_sum
    return scores

# ── P2 metrics from final JSONs ───────────────────────────────────────────────
def get_p2_metric(split, variant, p2_task, metric_key, sub_key=None):
    """
    metric_key : 'bacc'|'cidx'|'c_index'
    sub_key    : task sub-dict key in JSON (e.g. 'acr_cls', 'acr_surv', 'clad', 'death')
                 None → flat dict
    """
    f = P2_BASE / f"split{split}_fold0" / f"{variant}_{p2_task}" / f"metrics_{variant}_final.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text()).get("test", {})
    if sub_key:
        d = d.get(sub_key, {})
    return d.get(metric_key) or d.get("cidx") or d.get("c_index") or d.get("bacc")

# ─────────────────────────────────────────────────────────────────────────────
# TASK DEFINITIONS
# each task: name, p1_score_cols, label_col, time_col, event_col, metric_fn,
#            p2_specs [(variant, p2_task, json_sub_key)]
# ─────────────────────────────────────────────────────────────────────────────
TASKS = [
    {
        "name":    "ACR Classification",
        "short":   "acr_cls",
        "metric":  "BACC",
        "p1_cols": {m: f"p1_acr_{m}" for m in MODALITIES},
        "label_col": "label",
        "eval": lambda df, col: bacc(df["label"].values, df[col].values),
        "p2_specs": [
            ("early",          "cls",  "bacc",    None),
            ("late",           "cls",  "bacc",    None),
            ("middle",         "cls",  "bacc",    None),
            ("mario_kempes",   "mega", "bacc",    "acr_cls"),
            ("longitudinal_mk","mega", "bacc",    "acr_cls"),
        ],
    },
    {
        "name":   "ACR Survival",
        "short":  "acr_surv",
        "metric": "C-index",
        "p1_cols": {m: f"h1_acr_surv_{m}" for m in MODALITIES},
        "time_col":  "tte_next_acr",
        "event_col": "event_next_acr",
        "eval": lambda df, col: fast_cidx(
            df["tte_next_acr"].values, df[col].values, df["event_next_acr"].values),
        "p2_specs": [
            ("early",          "acr_surv", "cidx",    None),
            ("late",           "acr_surv", "cidx",    None),
            ("middle",         "acr_surv", "cidx",    None),
            ("mario_kempes",   "mega",     "c_index", "acr_surv"),
            ("longitudinal_mk","mega",     "c_index", "acr_surv"),
        ],
    },
    {
        "name":   "CLAD",
        "short":  "clad",
        "metric": "C-index",
        "p1_cols": {m: f"h1_clad_{m}" for m in MODALITIES},
        "time_col":  "clad_time",
        "event_col": "clad_event",
        "eval": lambda df, col: fast_cidx(
            df["clad_time"].values, df[col].values, df["clad_event"].values),
        "p2_specs": [
            ("early",          "clad_surv", "cidx",    None),
            ("late",           "clad_surv", "cidx",    None),
            ("middle",         "clad_surv", "cidx",    None),
            ("mario_kempes",   "mega",      "c_index", "clad"),
            ("longitudinal_mk","mega",      "c_index", "clad"),
        ],
    },
    {
        "name":   "Death",
        "short":  "death",
        "metric": "C-index",
        "p1_cols": {m: f"h1_death_{m}" for m in MODALITIES},
        "time_col":  "death_time",
        "event_col": "death_event",
        "eval": lambda df, col: fast_cidx(
            df["death_time"].values, df[col].values, df["death_event"].values),
        "p2_specs": [
            ("early",          "death_surv", "cidx",    None),
            ("late",           "death_surv", "cidx",    None),
            ("middle",         "death_surv", "cidx",    None),
            ("mario_kempes",   "mega",       "c_index", "death"),
            ("longitudinal_mk","mega",       "c_index", "death"),
        ],
    },
]

# ─────────────────────────────────────────────────────────────────────────────

def fmt(v, n_samples=None):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  —  "
    s = f"{v:.3f}"
    if n_samples is not None:
        s += f" (n={n_samples})"
    return s

def mean_std(vals):
    v = [x for x in vals if x is not None and not math.isnan(x)]
    if len(v) == 0: return "  —  "
    if len(v) == 1: return f"{v[0]:.3f}     "
    return f"{statistics.mean(v):.3f}±{statistics.stdev(v):.3f}"

def print_pretty_table(task_name, metric, rows):
    """Print a clean publication-style table: model | s0 s1 s2 s3 s4 | mean±std."""
    # strip n=() annotations for the pretty table
    def clean(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "   —   "
        return f"{v:.3f}"

    COL = 9
    SEP = "─"
    sections = [
        ("Unimodal (P1)",   [r for r in rows if r[0].startswith("P1 ") and "wtd" not in r[0]]),
        ("Weighted ensemble", [r for r in rows if "wtd" in r[0]]),
        ("Multimodal (P2)", [r for r in rows if r[0].startswith("P2 ")]),
    ]

    width = 28 + 5 * (COL + 2) + 2 + 14
    print(f"\n{'┌' + SEP*width + '┐'}")
    title = f"  {task_name}  —  {metric}"
    print(f"│{title:<{width}}│")
    print(f"├{'─'*28}┬{'─'*(COL+2)}┬{'─'*(COL+2)}┬{'─'*(COL+2)}┬{'─'*(COL+2)}┬{'─'*(COL+2)}┬{'─'*14}┤")
    print(f"│{'Model':<28}│{'  s0':>{COL+2}}│{'  s1':>{COL+2}}│{'  s2':>{COL+2}}│{'  s3':>{COL+2}}│{'  s4':>{COL+2}}│{'  mean±std':>14}│")
    print(f"├{'─'*28}┼{'─'*(COL+2)}┼{'─'*(COL+2)}┼{'─'*(COL+2)}┼{'─'*(COL+2)}┼{'─'*(COL+2)}┼{'─'*14}┤")

    for sec_name, sec_rows in sections:
        if not sec_rows:
            continue
        print(f"│ {sec_name:<{width-2}}│")
        for label, vals, _ in sec_rows:
            short = (label.replace("P1 ","").replace("P2 ","")
                         .replace("longitudinal_mk","LongitudinalMK")
                         .replace("mario_kempes","MarioKempes")
                         .replace("wtd ensemble","Wtd. ensemble"))
            cells = [f"{clean(v):>{COL}}" for v in vals]
            ms = mean_std(vals)
            print(f"│  {short:<26}│{'  ' + cells[0]}│{'  ' + cells[1]}│{'  ' + cells[2]}│{'  ' + cells[3]}│{'  ' + cells[4]}│  {ms:<12}│")
        print(f"├{'─'*28}┼{'─'*(COL+2)}┼{'─'*(COL+2)}┼{'─'*(COL+2)}┼{'─'*(COL+2)}┼{'─'*(COL+2)}┼{'─'*14}┤")

    print(f"└{'─'*28}┴{'─'*(COL+2)}┴{'─'*(COL+2)}┴{'─'*(COL+2)}┴{'─'*(COL+2)}┴{'─'*(COL+2)}┴{'─'*14}┘")

def run_task(task):
    print(f"\n{'='*80}")
    print(f"  TASK: {task['name']}   metric: {task['metric']}")
    print(f"{'='*80}")

    rows = []    # list of (label, [s0..s4])

    # ── P1 unimodal ──────────────────────────────────────────────────────────
    for mod in MODALITIES:
        col   = task["p1_cols"][mod]
        vals  = []
        ns    = []
        for split in SPLITS:
            df = load_split(split)
            if df is None or col not in df.columns:
                vals.append(None); ns.append(0); continue
            sub = df[df[f"has_{mod}"] == 1].copy()
            ns.append(len(sub))
            vals.append(task["eval"](sub, col))
        row_vals = [fmt(v, n) for v, n in zip(vals, ns)]
        rows.append((f"P1 {mod}", vals, row_vals))

    # ── P1 weighted ensemble ──────────────────────────────────────────────────
    wtd_vals = []
    for split in SPLITS:
        df = load_split(split)
        if df is None:
            wtd_vals.append(None); continue
        # prevalence weights: fraction of test samples with each modality
        raw_w = {m: df[f"has_{m}"].mean() for m in MODALITIES}
        total_w = sum(raw_w.values())
        weights = {m: raw_w[m] / total_w for m in MODALITIES}  # sum to 1
        ens = weighted_ensemble(df, task["p1_cols"], weights)
        # keep only samples with at least one modality
        valid = ~np.isnan(ens)
        if valid.sum() == 0:
            wtd_vals.append(None); continue
        df2 = df[valid].copy()
        df2["_ens"] = ens[valid]
        wtd_vals.append(task["eval"](df2, "_ens"))

    rows.append(("P1 wtd ensemble", wtd_vals, [fmt(v) for v in wtd_vals]))

    # ── P2 variants (from metrics JSONs) ─────────────────────────────────────
    for variant, p2_task, metric_key, sub_key in task["p2_specs"]:
        vals = []
        for split in SPLITS:
            vals.append(get_p2_metric(split, variant, p2_task, metric_key, sub_key))
        label = f"P2 {variant}"
        rows.append((label, vals, [fmt(v) for v in vals]))

    # ── Print table ───────────────────────────────────────────────────────────
    col_w = 16
    header = f"{'Model':<28} " + "  ".join(f"s{s}" for s in SPLITS) + "   mean±std"
    print(header)
    print("-" * 90)
    for label, vals, fmts in rows:
        # mean±std (no n= suffix — raw vals)
        ms = mean_std([v for v, f in zip(vals, fmts) if "n=" not in f])
        # rebuild fmts without n= for mean±std rows, but keep n= for display
        display = "  ".join(f"{f:>12}" for f in fmts)
        print(f"{label:<28} {display}   {ms}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_rows = []
    for label, vals, _ in rows:
        row = {"model": label}
        for split, v in zip(SPLITS, vals):
            row[f"s{split}"] = round(v, 4) if v is not None and not math.isnan(v) else None
        v_clean = [v for v in vals if v is not None and not math.isnan(v)]
        row["mean"] = round(statistics.mean(v_clean), 4) if v_clean else None
        row["std"]  = round(statistics.stdev(v_clean), 4) if len(v_clean) >= 2 else None
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_path = OUT_DIR / f"comparison_{task['short']}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  → saved: {out_path}")

    print_pretty_table(task["name"], task["metric"], rows)

    return out_rows


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        run_task(task)
    print("\n\nDone.")
