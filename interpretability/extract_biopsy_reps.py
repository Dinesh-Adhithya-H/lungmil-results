"""
Extract per-biopsy (256-dim) representations and risk scores from LongMK (no_alibi)
single-task models across all 5 outer splits.

For each split s (0..4), for each test patient:
  - Run model with FULL biopsy sequence (SAB over all biopsies)
  - For each biopsy position i, compute:
      * 256-dim ABMIL rep anchored at biopsy i (causal prefix tokens[:end_i])
      * Scalar risk score from the task head
  - This gives N_biopsies biopsy-level data points across 5 splits (out-of-sample)

Tasks: acr_cls, acr_surv, clad_surv, death_surv  (one model per task per split)

Output per task: results/mm_abmil_v8/biopsy_reps_{task}_split{s}.pt
  {
    "patient_ids": list[str],   len = N_biopsies
    "stems":       list[str],   len = N_biopsies
    "biopsy_days": list[float], len = N_biopsies (days from patient t0)
    "reps":        Tensor (N, 256),
    "risk":        Tensor (N,),
    "tte":         Tensor (N,),   (NaN if missing)
    "event":       Tensor (N,),   (NaN if missing)
    "label":       Tensor (N,),   (NaN if not ACR-labeled)
    "split":       int,
  }

Usage: sbatch interpretability/submit_extract_biopsy_reps.sh
       (Never run Python directly on the login node.)
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from shared import SPLITS_CSV, SAMPLES_DIR, RESULTS_ROOT, MOD_ORDER
from interpret_longitudinal_mk import load_model, load_patient_bags, extract_patient_longitudinal
from mil.data.splits import build_splits_longitudinal

# ── Config ─────────────────────────────────────────────────────────────────────

VARIANT_BASE = "longitudinal_mk_no_alibi"

TASK_CFG = {
    "acr_cls":   {"dir_suffix": "cls",       "longi_key": "acr_cls",
                  "tte_key": None,            "event_key": None,
                  "label_key": "label"},
    "acr_surv":  {"dir_suffix": "acr_surv",  "longi_key": "acr_surv",
                  "tte_key": "acr_days",      "event_key": "acr_status",
                  "label_key": None},
    "clad_surv": {"dir_suffix": "clad_surv", "longi_key": "clad",
                  "tte_key": "clad_time",     "event_key": "clad_event",
                  "label_key": None},
    "death_surv":{"dir_suffix": "death_surv","longi_key": "death",
                  "tte_key": "death_time",    "event_key": "death_event",
                  "label_key": None},
}

OUT_DIR = RESULTS_ROOT.parent / "biopsy_reps"


# ── Per-patient extraction ──────────────────────────────────────────────────────

def extract_patient(model, patient, device, task_cfg):
    """
    Run model on a patient and return per-biopsy reps + risk scores.

    Uses extract_patient_longitudinal which computes reps causally at each biopsy
    position — guaranteed to produce T records per patient regardless of whether
    the model's forward() calls _abmil_rep internally.

    Returns list of dicts, one per biopsy position:
      { "stem", "biopsy_day", "rep": Tensor(256,), "risk", "tte", "event", "label" }
    """
    bags_list, _ = load_patient_bags(patient, device)
    bags_ok = [b for b in bags_list if any(v is not None for v in b.values())]
    if not bags_ok:
        return []

    longi_key = task_cfg["longi_key"]
    tte_key   = task_cfg["tte_key"]
    evt_key   = task_cfg["event_key"]
    lbl_key   = task_cfg["label_key"]
    records   = patient["records"]
    days      = patient["days"]

    with torch.no_grad():
        extr = extract_patient_longitudinal(
            model, patient, bags_list, device, tasks=[longi_key]
        )
    if extr is None:
        return []

    haz_traj  = extr["hazard_traj"].get(longi_key, [])
    rep_list  = extr["per_biopsy_reps"].get(longi_key, [])
    T         = extr["n_biopsies"]

    result = []
    for t_idx in range(T):
        if t_idx >= len(haz_traj):
            break
        rep = rep_list[t_idx] if t_idx < len(rep_list) and rep_list[t_idx] is not None else None
        if rep is None:
            continue

        rec  = records[t_idx]
        stem = patient["stems"][t_idx]
        day  = float(days[t_idx])

        tte   = float(rec.get(tte_key,  float("nan"))) if tte_key  else float("nan")
        event = float(rec.get(evt_key,  float("nan"))) if evt_key  else float("nan")
        label = float(rec.get(lbl_key,  float("nan"))) if lbl_key  else float("nan")
        if label is None:
            label = float("nan")

        result.append({
            "stem":       stem,
            "biopsy_day": day,
            "rep":        rep,
            "risk":       float(haz_traj[t_idx]),
            "tte":        tte,
            "event":      event,
            "label":      label,
        })

    return result


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  type=int, default=None,
                        help="Outer split index (0-4). If omitted, runs all 5.")
    parser.add_argument("--fold",   type=int, default=0)
    parser.add_argument("--tasks",  nargs="+",
                        default=["acr_cls","acr_surv","clad_surv","death_surv"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    splits_to_run = [args.split] if args.split is not None else list(range(5))

    for task_key in args.tasks:
        cfg = TASK_CFG[task_key]
        print(f"\n{'='*60}\nTask: {task_key}\n{'='*60}")

        for split in splits_to_run:
            out_path = OUT_DIR / f"biopsy_reps_{task_key}_split{split}.pt"
            if out_path.exists():
                print(f"  [split {split}] already exists, skipping: {out_path.name}")
                continue

            print(f"  [split {split}] loading model...")
            variant = VARIANT_BASE
            try:
                model, task_names = load_model(
                    split=split, fold=args.fold, device=device,
                    task=cfg["dir_suffix"],
                    variant=variant,
                )
            except FileNotFoundError as e:
                print(f"  [split {split}] checkpoint missing: {e}")
                continue

            print(f"  [split {split}] loading test patients...")
            longi_splits = build_splits_longitudinal(
                SAMPLES_DIR, SPLITS_CSV, fold=args.fold, split=split)
            test_patients = longi_splits.get("test", [])
            print(f"  [split {split}] N_patients_test = {len(test_patients)}")

            all_recs = []
            for pi, patient in enumerate(test_patients):
                if (pi + 1) % 10 == 0:
                    print(f"    patient {pi+1}/{len(test_patients)}")
                try:
                    recs = extract_patient(model, patient, device, cfg)
                except Exception as exc:
                    print(f"    WARNING: patient {patient['patient_id']} failed: {exc}")
                    continue
                for r in recs:
                    r["patient_id"] = patient["patient_id"]
                all_recs.extend(recs)

            if not all_recs:
                print(f"  [split {split}] no records extracted, skipping save.")
                continue

            n = len(all_recs)
            reps   = torch.stack([r["rep"] for r in all_recs])         # (N, 256)
            risk   = torch.tensor([r["risk"]       for r in all_recs])
            tte    = torch.tensor([r["tte"]        for r in all_recs])
            event  = torch.tensor([r["event"]      for r in all_recs])
            label  = torch.tensor([r["label"]      for r in all_recs])
            bdays  = torch.tensor([r["biopsy_day"] for r in all_recs])
            pids   = [r["patient_id"] for r in all_recs]
            stems  = [r["stem"]       for r in all_recs]

            payload = {
                "patient_ids": pids,
                "stems":       stems,
                "biopsy_days": bdays,
                "reps":        reps,
                "risk":        risk,
                "tte":         tte,
                "event":       event,
                "label":       label,
                "split":       split,
                "task":        task_key,
                "n_patients":  len(test_patients),
            }
            torch.save(payload, out_path)
            print(f"  [split {split}] saved {n} biopsy reps → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
