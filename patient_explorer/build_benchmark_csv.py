"""Aggregate all P1 and P2 metrics JSON files → benchmark_results.csv."""

from pathlib import Path
import json, re
import numpy as np
import pandas as pd

RESULTS = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8")
OUT = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/patient_explorer/data/benchmark_results.csv")

TASK_NORM = {
    "cls": "acr_cls", "acr_cls": "acr_cls",
    "acr_surv": "acr_surv",
    "clad_surv": "clad_surv",
    "death_surv": "death_surv",
}

rows = []

# ── P2: phase2/split{s}_fold0/{variant}_{task}/metrics_*_final.json ──────────
for jf in sorted(RESULTS.glob("phase2/split*_fold0/*/metrics_*_final.json")):
    parts = jf.parent.name  # e.g. "set_mil_mt_cls"
    m = re.match(r"(early|late|middle|set_mil_mt|longitudinal_mk_mt|mario_kempes)_(.*)", parts)
    if not m:
        continue
    variant, raw_task = m.group(1), m.group(2)
    task = TASK_NORM.get(raw_task, raw_task)

    split_m = re.search(r"split(\d+)_fold0", str(jf))
    split = int(split_m.group(1)) if split_m else -1

    try:
        d = json.loads(jf.read_text())
    except Exception:
        continue
    test = d.get("test", {})

    bacc    = test.get("bacc")
    c_index = test.get("c_index")
    auc     = test.get("auc")

    for metric, val in [("BAcc", bacc), ("C-idx", c_index), ("AUC", auc)]:
        if val is None:
            continue
        rows.append({"phase": "P2", "model": variant, "task": task,
                     "metric": metric, f"s{split}": val})

# ── P1: phase1/split{s}_fold{f}/{task}/{modality}/final/metrics.json ─────────
for jf in sorted(RESULTS.glob("phase1/split*_fold0/*/*/final/metrics.json")):
    parts = jf.parts
    # find split index
    split_part = next((p for p in parts if re.match(r"split\d+_fold0", p)), None)
    split = int(re.search(r"split(\d+)", split_part).group(1)) if split_part else -1
    task_raw  = parts[-4]  # e.g. acr_cls, acr_surv, clad, death
    modality  = parts[-3]  # HE, BAL, CT, Clinical
    task = TASK_NORM.get(task_raw, task_raw)
    model = f"unimodal_{modality}"

    try:
        d = json.loads(jf.read_text())
    except Exception:
        continue
    test = d.get("test", {})
    bacc    = test.get("bacc")
    c_index = test.get("c_index")
    auc     = test.get("auc")

    for metric, val in [("BAcc", bacc), ("C-idx", c_index), ("AUC", auc)]:
        if val is None:
            continue
        rows.append({"phase": "P1", "model": model, "task": task,
                     "metric": metric, f"s{split}": val})

# ── Aggregate: pivot splits → mean ± std ─────────────────────────────────────
df = pd.DataFrame(rows)
if df.empty:
    print("ERROR: no rows found"); exit(1)

split_cols = ["s0", "s1", "s2", "s3", "s4"]
for c in split_cols:
    if c not in df.columns:
        df[c] = np.nan

df = df.groupby(["phase", "model", "task", "metric"])[split_cols].first().reset_index()

df["mean"] = df[split_cols].mean(axis=1, skipna=True)
df["std"]  = df[split_cols].std(axis=1, skipna=True)

# Reorder columns
df = df[["phase", "model", "task", "metric", "mean", "std"] + split_cols]
df = df.sort_values(["phase", "task", "metric", "mean"], ascending=[True, True, True, False])

df.to_csv(OUT, index=False)
print(f"Written {len(df)} rows → {OUT}")
print(df.groupby(["phase","metric"])["model"].count())
