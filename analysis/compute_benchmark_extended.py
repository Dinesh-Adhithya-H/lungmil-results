#!/usr/bin/env python3
"""
Extended benchmark table:

  P1-{mod}                 — unimodal baseline (single modality, P1 encoder)
  P1-weighted              — coverage-weighted average of P1 unimodal scores
                             weight_m = n_test_with_m / sum(n_test_with_any_m), per fold
  P2-{base}                — multimodal fusion model (all modalities)
  P2-{base}-abl-{mod}      — P2 model at inference with ONLY modality m visible
                             (measures how well fusion model utilises each modality)

Saves:
  results/analysis_v8_full/metrics_summary.csv        — all rows, mean±std across folds
  results/analysis_v8_full/metrics_summary_folds.csv  — per-fold values for plotting
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO    = Path(__file__).parent.parent
SPLITS  = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
BASE    = REPO / "results/mm_abmil_v8"
P1DIR   = BASE / "phase1"
P2DIR   = BASE / "phase2"
OUT_DIR = REPO / "results/analysis_v8_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODALITIES = ["HE", "BAL", "CT", "Clinical"]

TASKS = [
    # (task_key,  metrics_json_key,  display_label)
    ("acr_cls",   "bacc",          "ACR cls (BACC)"),
    ("acr_surv",  "c_index",       "ACR surv (C-idx)"),
    ("clad_surv", "c_index",       "CLAD surv (C-idx)"),
    ("death_surv","c_index",       "Death surv (C-idx)"),
]
TASK_KEYS = [t[0] for t in TASKS]

P1PT = {"acr_cls":"acr", "acr_surv":"acr_surv", "clad_surv":"clad", "death_surv":"death"}
SUFF = {"acr_cls":"_cls", "acr_surv":"_acr_surv", "clad_surv":"_clad_surv", "death_surv":"_death_surv"}

# Slot mega model: all tasks in one JSON, CLAD/death use distinct keys
SLOT_MEGA_DIR  = "slot_mega_tss"
SLOT_MEGA_KEYS = {
    "acr_cls":   "bacc",
    "acr_surv":  "c_index",
    "clad_surv": "clad_c_index",
    "death_surv": "death_c_index",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def get_test(d: dict, metric_key: str):
    v = d.get("test", {}).get(metric_key)
    return float(v) if v is not None else None


# ── 1. Collect raw per-fold values ────────────────────────────────────────────

def collect_all() -> dict[str, dict[str, list[float]]]:
    """Returns model_key → task_key → [fold0, fold1, fold2, fold3]."""
    res = defaultdict(lambda: defaultdict(list))

    for tl, mk, _ in TASKS:
        for mod in MODALITIES:
            for fold in range(4):
                mf = P1DIR / f"split1_fold{fold}/{P1PT[tl]}/{mod}/final/metrics.json"
                v  = get_test(load_json(mf), mk)
                if v is not None:
                    res[f"P1-{mod}"][tl].append(v)

    for tl, mk, _ in TASKS:
        for base in ["early", "middle", "late"]:
            for fold in range(4):
                mf = P2DIR / f"split1_fold{fold}/{base}{SUFF[tl]}/metrics_{base}.json"
                if not mf.exists():
                    mf = BASE / f"metrics_split1_fold{fold}_{base}{SUFF[tl]}.json"
                d  = load_json(mf)
                v  = get_test(d, mk)
                if v is not None:
                    res[f"P2-{base}"][tl].append(v)

                # unimodal ablation of this P2 model: only modality m is visible
                for mod, ab in d.get("unimodal_ablation", {}).items():
                    v2 = ab.get(mk)
                    if v2 is not None:
                        res[f"P2-{base}-abl-{mod}"][tl].append(float(v2))

    # Slot mega model: one JSON per fold, CLAD/death use distinct metric keys
    for tl, _, _ in TASKS:
        mega_mk = SLOT_MEGA_KEYS[tl]
        for fold in range(4):
            mf = P2DIR / f"split1_fold{fold}/{SLOT_MEGA_DIR}/metrics_slot.json"
            d  = load_json(mf)
            v  = get_test(d, mega_mk)
            if v is not None:
                res["P2-slot"][tl].append(v)

            # unimodal ablation — ablation uses same key as task primary metric
            for mod, ab in d.get("unimodal_ablation", {}).items():
                v2 = ab.get(mega_mk)
                if v2 is not None:
                    res[f"P2-slot-abl-{mod}"][tl].append(float(v2))

    return res


# ── 2. P1-weighted baseline ───────────────────────────────────────────────────

def compute_p1_weighted(res: dict) -> dict[str, list[float]]:
    """
    For each fold, compute:
        score = sum_m(  w_m  * P1-m_score  )
    where  w_m  = n_test_with_m / sum_m(n_test_with_m)   (coverage-normalized).

    n_test_with_m comes from unimodal_ablation["m"]["n"] averaged across P2 models
    that have it, or falls back to the splits CSV.
    """
    splits_df = pd.read_csv(SPLITS)

    fold_weighted: dict[str, list[float]] = defaultdict(list)

    for fold in range(4):
        # modality counts in this fold's test set
        test = splits_df[splits_df[f"fold_{fold}"] == "test"]
        n_total = len(test)
        raw_counts = {m: int(test[f"has_{m}"].sum()) for m in MODALITIES}
        total_cov  = sum(raw_counts.values())   # can exceed n_total (multi-modal samples)
        if total_cov == 0:
            continue
        weights = {m: raw_counts[m] / total_cov for m in MODALITIES}

        for tl, mk, _ in TASKS:
            score = 0.0
            for m in MODALITIES:
                fold_vals = res.get(f"P1-{m}", {}).get(tl, [])
                if len(fold_vals) <= fold:
                    continue
                score += weights[m] * fold_vals[fold]
            fold_weighted[tl].append(score)

    return dict(fold_weighted)


# ── 3. Build summary DataFrames ───────────────────────────────────────────────

def build_summary(res: dict, p1w: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_models = sorted(res.keys())

    rows_mean, rows_fold = [], []

    def add(model_key: str, fold_vals_by_task: dict[str, list[float]]):
        mean_row = {"model": model_key}
        fold_row = {"model": model_key}
        for tl, _, _ in TASKS:
            vals = fold_vals_by_task.get(tl, [])
            mean_row[tl + "_mean"] = float(np.mean(vals)) if vals else float("nan")
            mean_row[tl + "_std"]  = float(np.std(vals))  if vals else float("nan")
            mean_row[tl + "_n"]    = len(vals)
            for fi, v in enumerate(vals):
                fold_row[f"{tl}_fold{fi}"] = v
        rows_mean.append(mean_row)
        rows_fold.append(fold_row)

    # P1 unimodal
    for mod in MODALITIES:
        add(f"P1-{mod}", res.get(f"P1-{mod}", {}))

    # P1-weighted
    add("P1-weighted", p1w)

    # P2 multimodal
    for base in ["early", "middle", "late", "slot"]:
        add(f"P2-{base}", res.get(f"P2-{base}", {}))

    # P2 unimodal ablation
    for base in ["early", "middle", "late", "slot"]:
        for mod in MODALITIES:
            key = f"P2-{base}-abl-{mod}"
            if key in res:
                add(key, res[key])

    return pd.DataFrame(rows_mean), pd.DataFrame(rows_fold)


# ── 4. Print comparison table ─────────────────────────────────────────────────

def print_table(df_mean: pd.DataFrame):
    cols = [(tl, lbl) for tl, _, lbl in TASKS]

    header = f"{'Model':<28}" + "".join(f"  {lbl[:18]:<20}" for _, lbl in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    def section(title, models):
        print(f"\n  — {title}")
        for m in models:
            row = df_mean[df_mean["model"] == m]
            if row.empty:
                continue
            line = f"  {m:<26}"
            for tl, _ in cols:
                mn = row[tl + "_mean"].iloc[0]
                sd = row[tl + "_std"].iloc[0]
                line += f"  {mn:.3f}±{sd:.3f}          "[:22]
            print(line)

    section("P1 unimodal", [f"P1-{m}" for m in MODALITIES])
    section("P1 weighted (coverage-normalised)", ["P1-weighted"])
    section("P2 multimodal", [f"P2-{b}" for b in ["early","middle","late","slot"]])
    for base in ["early", "middle", "late", "slot"]:
        keys = [f"P2-{base}-abl-{m}" for m in MODALITIES
                if f"P2-{base}-abl-{m}" in df_mean["model"].values]
        if keys:
            section(f"P2-{base} unimodal ablation", keys)

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Collecting metrics …")
    res  = collect_all()
    p1w  = compute_p1_weighted(res)

    print("Building summary tables …")
    df_mean, df_fold = build_summary(res, p1w)

    out_mean = OUT_DIR / "metrics_summary.csv"
    out_fold = OUT_DIR / "metrics_folds.csv"
    df_mean.to_csv(out_mean, index=False)
    df_fold.to_csv(out_fold, index=False)

    print_table(df_mean)

    print(f"Saved:")
    print(f"  {out_mean}  ({len(df_mean)} model rows)")
    print(f"  {out_fold}")

    # Also write a compact version matching the old format (for downstream scripts)
    compact_rows = []
    for _, row in df_mean.iterrows():
        for tl, _, _ in TASKS:
            compact_rows.append({
                "model": row["model"],
                "task":  tl,
                "mean":  row[tl + "_mean"],
                "std":   row[tl + "_std"],
                "n_folds": int(row[tl + "_n"]),
            })
            fold_vals = df_fold[df_fold["model"] == row["model"]]
            if not fold_vals.empty:
                for fi in range(4):
                    col = f"{tl}_fold{fi}"
                    compact_rows[-1][f"fold{fi}"] = (
                        float(fold_vals[col].iloc[0]) if col in fold_vals.columns else float("nan")
                    )
    pd.DataFrame(compact_rows).to_csv(OUT_DIR / "metrics_summary_long.csv", index=False)
    print(f"  {OUT_DIR / 'metrics_summary_long.csv'}")


if __name__ == "__main__":
    main()
