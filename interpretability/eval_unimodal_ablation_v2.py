"""
Unimodal ablation evaluation for longitudinal_mk_no_alibi_v2.

For each split × task, loads the v2 checkpoint and runs the model with only
one modality active at a time across all biopsy timepoints, computing BACC / C-index.
Appends results to interpretability/unimodal_ablation/unimodal_ablation_summary.csv.

Usage: sbatch interpretability/submit_eval_unimodal_ablation_v2.sh
       (Never run Python directly on the login node.)
"""

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from shared import SPLITS_CSV, SAMPLES_DIR, RESULTS_ROOT
from interpret_longitudinal_mk import load_model, load_patient_bags, extract_patient_longitudinal
from mil.data.splits import build_splits_longitudinal

VARIANT = "longitudinal_mk_no_alibi_v2"
OUT_CSV = ROOT / "interpretability" / "unimodal_ablation" / "unimodal_ablation_summary.csv"
MODALITIES = ["HE", "BAL", "CT", "Clinical"]

TASK_CFG = {
    "acr_cls":   {"dir_suffix": "cls",       "longi_key": "acr_cls",
                  "tte_key": None,            "event_key": None,  "metric": "bacc"},
    "acr_surv":  {"dir_suffix": "acr_surv",  "longi_key": "acr_surv",
                  "tte_key": "tte_next_acr",  "event_key": "event_next_acr", "metric": "c_index"},
    "clad_surv": {"dir_suffix": "clad_surv", "longi_key": "clad",
                  "tte_key": "clad_time",     "event_key": "clad_event",     "metric": "c_index"},
    "death_surv":{"dir_suffix": "death_surv","longi_key": "death",
                  "tte_key": "death_time",    "event_key": "death_event",    "metric": "c_index"},
}


def run_patient_unimodal(model, patient, bags_list, device, longi_key, active_mod):
    """
    Run model with only active_mod present. Returns (haz_traj, records) where
    haz_traj[t] is the per-biopsy hazard and records[t] is the biopsy record.
    Only biopsies where active_mod is available are included.
    """
    masked = []
    for biopsy_bags in bags_list:
        m = {k: (v if k in (active_mod, "HE_coords") else None)
             for k, v in biopsy_bags.items()}
        if active_mod != "HE":
            m["HE_coords"] = None
        masked.append(m)

    with torch.no_grad():
        extr = extract_patient_longitudinal(model, patient, masked, device, tasks=[longi_key])
    if extr is None:
        return [], []

    haz_traj = extr["hazard_traj"].get(longi_key, [])
    records  = patient["records"]
    T = min(len(haz_traj), len(records))

    # Filter to biopsies where active_mod was available
    out_haz, out_rec = [], []
    for t in range(T):
        if bags_list[t].get(active_mod) is None:
            continue
        h = haz_traj[t]
        if h is None or np.isnan(float(h)):
            continue
        out_haz.append(float(h))
        out_rec.append(records[t])

    return out_haz, out_rec


def eval_unimodal_for_task(model, test_patients, device, cfg):
    """Evaluate each unimodal ablation for one task. Returns {mod: {metric: value}}."""
    longi_key = cfg["longi_key"]
    tte_key   = cfg["tte_key"]
    evt_key   = cfg["event_key"]
    metric    = cfg["metric"]

    results = {}
    for active_mod in MODALITIES:
        all_scores, all_labels, all_ttes, all_events = [], [], [], []

        for patient in test_patients:
            bags_list, _ = load_patient_bags(patient, device)
            has_mod = any(b.get(active_mod) is not None for b in bags_list)
            if not has_mod:
                continue

            haz_list, rec_list = run_patient_unimodal(
                model, patient, bags_list, device, longi_key, active_mod)

            if metric == "bacc":
                # Patient-level: use last available biopsy hazard + its label
                if not haz_list:
                    continue
                risk = haz_list[-1]
                lbl  = rec_list[-1].get("label")
                if lbl is not None:
                    all_scores.append(risk)
                    all_labels.append(int(lbl))
            else:
                # Per-biopsy: pool all (hazard, tte, event) tuples
                for h, rec in zip(haz_list, rec_list):
                    try:
                        t_val = float(rec.get(tte_key, float("nan")))
                        e_val = float(rec.get(evt_key, float("nan")))
                    except (TypeError, ValueError):
                        continue
                    if np.isnan(t_val) or t_val < 0:
                        continue
                    all_scores.append(h)
                    all_ttes.append(t_val)
                    all_events.append(0.0 if np.isnan(e_val) else e_val)

        if metric == "bacc" and len(all_scores) >= 2:
            preds = [1 if s > 0.5 else 0 for s in all_scores]
            val = balanced_accuracy_score(all_labels, preds)
        elif metric == "c_index" and len(all_scores) >= 2 and sum(all_events) > 0:
            val = concordance_index(all_ttes, [-s for s in all_scores], all_events)
        else:
            val = float("nan")

        n = len(all_scores)
        results[active_mod] = {metric: val, "n": n}
        print(f"    {active_mod}: n={n}  {metric}={val:.4f}" if not np.isnan(val)
              else f"    {active_mod}: n={n}  {metric}=NaN")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--fold",  type=int, default=0)
    parser.add_argument("--tasks", nargs="+", default=list(TASK_CFG.keys()))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[v2 unimodal ablation] split={args.split} tasks={args.tasks} device={device}")

    rows = []

    for task_key in args.tasks:
        cfg = TASK_CFG[task_key]
        print(f"\n── {task_key} ──")

        model, _ = load_model(args.split, args.fold, device, task=task_key, variant=VARIANT)
        model.eval()

        split_info = build_splits_longitudinal(
            SAMPLES_DIR, SPLITS_CSV, fold=args.fold, split=args.split
        )
        test_patients = split_info.get("test", [])
        print(f"  {len(test_patients)} test patients")

        abl = eval_unimodal_for_task(model, test_patients, device, cfg)

        metric_key = cfg["metric"]
        for mod, metrics in abl.items():
            val = metrics.get(metric_key, float("nan"))
            rows.append({
                "variant":  VARIANT,
                "task":     task_key,
                "modality": mod,
                "metric":   metric_key,
                "split":    args.split,
                "value":    val,
                "mean":     float("nan"),
                "std":      float("nan"),
                "count":    float("nan"),
                "mean_std": "",
            })

        del model
        torch.cuda.empty_cache()

    # Append to CSV (remove old v2 rows for this split first, avoid race: write per-split file)
    split_csv = OUT_CSV.parent / f"unimodal_ablation_v2_split{args.split}.csv"
    pd.DataFrame(rows).to_csv(split_csv, index=False)
    print(f"\nSaved {len(rows)} rows → {split_csv}")


if __name__ == "__main__":
    main()
