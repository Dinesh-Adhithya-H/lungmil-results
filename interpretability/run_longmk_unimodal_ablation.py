"""
Unimodal ablation for LongMK (no_alibi) and LongMK-MT (no_alibi).

For each split × task × modality: pass bags with ONLY that modality present
(all others set to None) — the model handles missing modalities natively.
Computes BACC (acr_cls) or C-index (survival tasks) on full test set.

Appends results to interpretability/unimodal_ablation/unimodal_ablation_summary.csv.

Run via: sbatch interpretability/submit_longmk_unimodal_ablation.sh
"""
import sys, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mil.data.splits import build_splits_longitudinal
from interpretability.interpret_longitudinal_mk import (
    load_model, load_patient_bags,
)
from shared import MOD_ORDER

SAMPLES_DIR = "/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples/"
SPLITS_CSV  = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
OUT_CSV     = ROOT / "interpretability" / "unimodal_ablation" / "unimodal_ablation_summary.csv"

TASK_CFG = {
    "cls":       {"metric": "bacc",    "surv": False},
    "acr_surv":  {"metric": "c_index", "surv": True,  "ev_col": "acr_event",  "tte_col": "acr_tte"},
    "clad_surv": {"metric": "c_index", "surv": True,  "ev_col": "clad_event", "tte_col": "clad_tte"},
    "death_surv":{"metric": "c_index", "surv": True,  "ev_col": "death_event","tte_col": "death_tte"},
}

VARIANTS = {
    "longitudinal_mk_no_alibi":    ["cls", "acr_surv", "clad_surv", "death_surv"],
    "longitudinal_mk_mt_no_alibi": ["cls", "acr_surv", "clad_surv", "death_surv"],
}


def concordance_index(risks, events, times):
    from lifelines.utils import concordance_index as ci
    return ci(times, -np.array(risks), np.array(events))


def balanced_accuracy(logits, labels):
    preds = (np.array(logits) > 0).astype(int)
    labs  = np.array(labels)
    pos = labs == 1; neg = labs == 0
    tpr = preds[pos].mean() if pos.sum() > 0 else 0.0
    tnr = (1 - preds[neg]).mean() if neg.sum() > 0 else 0.0
    return (tpr + tnr) / 2


def run_ablation(model, tasks, patients, device, ablate_mod):
    """Run model with only ablate_mod present; return {task: [score per patient]}."""
    scores = {t: [] for t in tasks}
    for patient in patients:
        try:
            bags_list, transplant_days = load_patient_bags(patient, device)
            # Keep only ablate_mod, zero others
            masked = []
            for bags in bags_list:
                masked.append({m: (bags[m] if m == ablate_mod else None)
                               for m in MOD_ORDER})
            # Check patient has at least one biopsy with this mod
            has_mod = any(b[ablate_mod] is not None for b in masked)
            if not has_mod:
                continue

            with torch.no_grad():
                # Build token sequence (reuse extract logic inline)
                from interpretability.interpret_longitudinal_mk import extract_patient_longitudinal
                extr = extract_patient_longitudinal(
                    model, masked, transplant_days, device, tasks)

            for t in tasks:
                logit = extr["logits"].get(t)
                if logit is not None:
                    scores[t].append((patient, logit))
        except Exception as e:
            print(f"  [warn] patient {patient.get('patient_id','?')}: {e}")
    return scores


def compute_metric(scores, task_key, splits_df):
    cfg = TASK_CFG[task_key]
    results = []
    for patient, score in scores:
        pid = patient["patient_id"]
        row = splits_df[splits_df["patient_id"] == pid]
        if row.empty:
            continue
        row = row.iloc[0]
        if cfg["surv"]:
            ev  = float(row.get(cfg["ev_col"], np.nan))
            tte = float(row.get(cfg["tte_col"], np.nan))
            if np.isnan(ev) or np.isnan(tte):
                continue
            results.append((score, ev, tte))
        else:
            lbl = float(row.get("acr_label", np.nan))
            if np.isnan(lbl):
                continue
            results.append((score, lbl))

    if len(results) < 3:
        return np.nan

    if cfg["surv"]:
        risks, events, times = zip(*results)
        try:
            return concordance_index(risks, events, times)
        except Exception:
            return np.nan
    else:
        logits, labels = zip(*results)
        return balanced_accuracy(logits, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--fold",  type=int, default=0)
    parser.add_argument("--gpu",   type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[ablation] split={args.split}  fold={args.fold}  device={device}")

    splits_df = pd.read_csv(SPLITS_CSV)

    rows = []
    for variant, task_keys in VARIANTS.items():
        for task_key in task_keys:
            print(f"\n[ablation] {variant}  task={task_key}")
            try:
                model, tasks = load_model(args.split, args.fold, device,
                                          task=task_key, variant=variant)
            except Exception as e:
                print(f"  [skip] could not load model: {e}")
                continue

            split_data = build_splits_longitudinal(
                SAMPLES_DIR, SPLITS_CSV, fold=args.fold, split=args.split)
            test_patients = split_data.get("test", [])
            print(f"  {len(test_patients)} test patients")

            for mod in MOD_ORDER:
                print(f"  ablating mod={mod} ...")
                scores = run_ablation(model, [task_key], test_patients, device, mod)
                val = compute_metric(scores[task_key], task_key, splits_df)
                print(f"    {mod}: {val:.4f}" if not np.isnan(val) else f"    {mod}: nan")
                rows.append({
                    "variant": variant,
                    "task":    task_key,
                    "modality":mod,
                    "metric":  TASK_CFG[task_key]["metric"],
                    "split":   args.split,
                    "value":   val,
                })

    if not rows:
        print("[ablation] no results — exiting")
        return

    new_df = pd.DataFrame(rows)

    # Merge into existing summary CSV (replace matching rows)
    if OUT_CSV.exists():
        old_df = pd.read_csv(OUT_CSV)
        # Drop per-split rows (old format had mean/std aggregated — keep those)
        # If the CSV has a "split" column, remove matching rows and append
        if "split" in old_df.columns:
            mask = ~((old_df["variant"].isin(VARIANTS)) &
                     (old_df["split"] == args.split))
            old_df = old_df[mask]
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"\n[ablation] saved {len(new_df)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
