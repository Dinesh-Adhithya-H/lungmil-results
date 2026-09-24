"""
Extract per-biopsy reps from ALL patients (train + val + test) for UMAP train/test comparison.

For each split s (0..4), runs the split-s model on every patient in the cohort
and saves a split_type field ("train", "val", "test") so the UMAP can be coloured
to show that representations generalise across splits.

Output: results/mm_abmil_v8/biopsy_reps_all/biopsy_reps_all_{task}_split{s}.pt
  Same format as biopsy_reps_{task}_split{s}.pt, plus:
    "split_types": list[str]  ("train", "val", "test")

Usage: sbatch interpretability/submit_extract_biopsy_reps_all.sh
       (Never run Python directly on the login node.)
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from shared import SPLITS_CSV, SAMPLES_DIR, RESULTS_ROOT
from interpret_longitudinal_mk import load_model, load_patient_bags, extract_patient_longitudinal
from mil.data.splits import build_splits_longitudinal

VARIANT_BASE = "longitudinal_mk_no_alibi_v2"

TASK_CFG = {
    "acr_cls":   {"dir_suffix": "cls",       "longi_key": "acr_cls",
                  "tte_key": None,            "event_key": None,       "label_key": "label"},
    "acr_surv":  {"dir_suffix": "acr_surv",  "longi_key": "acr_surv",
                  "tte_key": "tte_next_acr",  "event_key": "event_next_acr", "label_key": None},
    "clad_surv": {"dir_suffix": "clad_surv", "longi_key": "clad",
                  "tte_key": "clad_time",     "event_key": "clad_event",    "label_key": None},
    "death_surv":{"dir_suffix": "death_surv","longi_key": "death",
                  "tte_key": "death_time",    "event_key": "death_event",   "label_key": None},
}

OUT_DIR = RESULTS_ROOT.parent / "biopsy_reps_all"


def extract_patient(model, patient, device, task_cfg):
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
    is_cls    = (longi_key == "acr_cls")

    with torch.no_grad():
        extr = extract_patient_longitudinal(
            model, patient, bags_list, device, tasks=[longi_key]
        )
    if extr is None:
        return []

    haz_traj = extr["hazard_traj"].get(longi_key, [])
    rep_list = extr["per_biopsy_reps"].get(longi_key, [])
    T        = extr["n_biopsies"]

    result = []
    for t_idx in range(T):
        rep = rep_list[t_idx] if t_idx < len(rep_list) and rep_list[t_idx] is not None else None
        if rep is None:
            continue

        rec  = records[t_idx]
        stem = patient["stems"][t_idx]
        day  = float(days[t_idx]) if t_idx < len(days) and days[t_idx] is not None else float("nan")

        def _f(v):
            return float(v) if v is not None else float("nan")

        tte   = _f(rec.get(tte_key))  if tte_key  else float("nan")
        event = _f(rec.get(evt_key))  if evt_key  else float("nan")
        label = _f(rec.get(lbl_key))  if lbl_key  else float("nan")

        raw = float(haz_traj[t_idx]) if t_idx < len(haz_traj) else float("nan")
        if is_cls and not math.isnan(raw):
            risk = 1.0 / (1.0 + math.exp(-raw))
        else:
            risk = raw

        result.append({
            "stem": stem, "biopsy_day": day, "rep": rep,
            "risk": risk, "tte": tte, "event": event, "label": label,
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  type=int, default=None)
    parser.add_argument("--fold",   type=int, default=0)
    parser.add_argument("--tasks",  nargs="+",
                        default=["acr_cls", "acr_surv", "clad_surv", "death_surv"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    splits_to_run = [args.split] if args.split is not None else list(range(5))

    for task_key in args.tasks:
        cfg = TASK_CFG[task_key]
        print(f"\n{'='*60}\nTask: {task_key}\n{'='*60}")

        for split in splits_to_run:
            out_path = OUT_DIR / f"biopsy_reps_all_{task_key}_split{split}.pt"
            if out_path.exists() and not args.overwrite:
                print(f"  [split {split}] already exists, skipping.")
                continue

            print(f"  [split {split}] loading model...")
            try:
                model, _ = load_model(
                    split=split, fold=args.fold, device=device,
                    task=cfg["dir_suffix"], variant=VARIANT_BASE,
                )
            except FileNotFoundError as e:
                print(f"  [split {split}] checkpoint missing: {e}")
                continue

            longi_splits = build_splits_longitudinal(
                SAMPLES_DIR, SPLITS_CSV, fold=args.fold, split=split)

            all_recs = []
            for split_type in ("train", "val", "test"):
                # val is merged into train for UMAP purposes
                patients = longi_splits.get(split_type, [])
                print(f"  [split {split}] {split_type}: {len(patients)} patients")
                for pi, patient in enumerate(patients):
                    if (pi + 1) % 20 == 0:
                        print(f"    {split_type} patient {pi+1}/{len(patients)}")
                    try:
                        recs = extract_patient(model, patient, device, cfg)
                    except Exception as exc:
                        import traceback
                        print(f"    WARNING: {patient['patient_id']} failed: {exc}")
                        traceback.print_exc()
                        continue
                    for r in recs:
                        r["patient_id"]  = patient["patient_id"]
                        r["split_type"]  = "train" if split_type == "val" else split_type
                    all_recs.extend(recs)

            if not all_recs:
                print(f"  [split {split}] no records, skipping.")
                continue

            n = len(all_recs)
            payload = {
                "patient_ids":  [r["patient_id"]  for r in all_recs],
                "stems":        [r["stem"]         for r in all_recs],
                "split_types":  [r["split_type"]   for r in all_recs],
                "biopsy_days":  torch.tensor([r["biopsy_day"] for r in all_recs]),
                "reps":         torch.stack([r["rep"]         for r in all_recs]),
                "risk":         torch.tensor([r["risk"]       for r in all_recs]),
                "tte":          torch.tensor([r["tte"]        for r in all_recs]),
                "event":        torch.tensor([r["event"]      for r in all_recs]),
                "label":        torch.tensor([r["label"]      for r in all_recs]),
                "split":        split,
                "task":         task_key,
            }
            torch.save(payload, out_path)
            print(f"  [split {split}] saved {n} biopsy reps → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
