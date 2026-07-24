"""
Export LongitudinalSetMIL per-visit predictions → patient_explorer/data/longi_preds.csv

Runs inference on all test patients across all 5 splits, using the task-specific
longitudinal models (cls / acr_surv / clad_surv / death_surv).
Output: per-biopsy rows with hazard scores for each task.

Run via sbatch only — never on the login node.
"""

import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "interpretability"))
sys.path.insert(0, str(ROOT))

from mil.models.builders import build_model_v8
from mil.data.splits import build_splits_longitudinal
from interpret_longitudinal_mk import (
    load_patient_bags,
    extract_patient_longitudinal,
    SAMPLES_DIR,
)

SPLITS_CSV = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
RESULTS_ROOT = ROOT / "results" / "mm_abmil_v8"
OUT_CSV = ROOT / "patient_explorer" / "data" / "longi_preds.csv"

# Build stem → anchor_dt lookup from splits CSV
_splits_df = pd.read_csv(SPLITS_CSV, parse_dates=["anchor_dt"])
STEM_TO_DATE = dict(zip(_splits_df["file"].astype(str).str.zfill(5), _splits_df["anchor_dt"]))

TASKS = ["acr_cls", "acr_surv", "clad_surv", "death_surv"]
TASK_DIR = {"acr_cls": "cls", "acr_surv": "acr_surv", "clad_surv": "clad_surv", "death_surv": "death_surv"}


def load_model(split: int, task_dir: str, device: torch.device):
    vtag = "longitudinal_mk_mt"
    ckpt_dir_name = f"{vtag}_{task_dir}"
    ckpt_dir = RESULTS_ROOT / "phase2" / f"split{split}_fold0" / ckpt_dir_name
    # Look for model checkpoint
    model_pt = ckpt_dir / f"model_{vtag}_final.pt"
    if not model_pt.exists():
        print(f"  [WARN] model not found: {model_pt}")
        return None

    # Determine task key for builder
    task_key_map = {
        "cls":        "acr_cls",
        "acr_surv":   "acr_surv",
        "clad_surv":  "clad_surv",
        "death_surv": "death_surv",
    }
    build_task = task_key_map[task_dir]

    model = build_model_v8("longitudinal_mk_mt", task=build_task)
    state = torch.load(model_pt, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"  Loaded model: {model_pt.name}")
    return model


def get_task_list(task_dir: str):
    """Map task_dir → task list used by extract function."""
    return {
        "cls":        ["acr_cls"],
        "acr_surv":   ["acr_surv"],
        "clad_surv":  ["clad_surv"],
        "death_surv": ["death_surv"],
    }[task_dir]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_rows = []

    for split in range(5):
        print(f"\n{'='*60}")
        print(f"SPLIT {split}")
        print(f"{'='*60}")

        # Load test patients
        splits_data = build_splits_longitudinal(
            SAMPLES_DIR, SPLITS_CSV, fold=0, split=split
        )
        test_patients = splits_data.get("test", [])
        print(f"  {len(test_patients)} test patients")

        if not test_patients:
            continue

        # Collect per-patient hazard trajectories across tasks
        # patient_id → task → list of (stem, day, hazard)
        patient_hazards = {}

        for task in TASKS:
            task_dir = TASK_DIR[task]
            print(f"\n  Task: {task} (dir: {task_dir})")

            model = load_model(split, task_dir, device)
            if model is None:
                continue

            task_list = get_task_list(task_dir)

            for patient in test_patients:
                pid = patient["patient_id"]
                T = len(patient["stems"])

                try:
                    bags_list, _ = load_patient_bags(patient, device)
                    extr = extract_patient_longitudinal(
                        model, patient, bags_list, device, task_list
                    )
                    if extr is None:
                        continue

                    hazard_traj = extr["hazard_traj"].get(task, [])
                    records = extr.get("records", [])

                    if pid not in patient_hazards:
                        patient_hazards[pid] = {
                            "stems": [r.get("stem", "") for r in records],
                            "anchor_dts": [r.get("anchor_dt") for r in records],
                            "days": extr.get("biopsy_days", []),
                            "records": records,
                        }
                        for t in TASKS:
                            patient_hazards[pid][t] = [float("nan")] * len(records)

                    n = min(len(hazard_traj), len(records))
                    for i in range(n):
                        patient_hazards[pid][task][i] = hazard_traj[i]

                except Exception as e:
                    print(f"    ERROR {pid}: {e}")
                    continue

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Build rows for this split
        for pid, pdata in patient_hazards.items():
            records = pdata["records"]
            for i, rec in enumerate(records):
                stem = rec.get("stem", "")
                adt = STEM_TO_DATE.get(str(stem).zfill(5))
                label = rec.get("label")
                tte_acr = rec.get("tte_acr")
                event_acr = rec.get("event_acr")
                tte_clad = rec.get("tte_clad")
                event_clad = rec.get("event_clad")
                tte_death = rec.get("tte_death")
                event_death = rec.get("event_death")

                h_cls = pdata["acr_cls"][i]
                h_acr = pdata["acr_surv"][i]
                h_clad = pdata["clad_surv"][i]
                h_death = pdata["death_surv"][i]

                # ACR cls: convert logit → probability
                score_cls = sigmoid(h_cls) if not np.isnan(h_cls) else float("nan")

                all_rows.append({
                    "stem": stem,
                    "patient_id": pid,
                    "anchor_dt": adt,
                    "split": split,
                    "score_acr_cls": score_cls,
                    "hazard_acr_surv": h_acr,
                    "hazard_clad_surv": h_clad,
                    "hazard_death_surv": h_death,
                    "true_acr_cls": label,
                    "event_acr": event_acr,
                    "tte_acr": tte_acr,
                    "event_clad": event_clad,
                    "tte_clad": tte_clad,
                    "event_death": event_death,
                    "tte_death": tte_death,
                })

    if not all_rows:
        print("ERROR: no rows collected!")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    df["anchor_dt"] = pd.to_datetime(df["anchor_dt"], errors="coerce")
    df = df.sort_values(["patient_id", "anchor_dt"]).reset_index(drop=True)
    df["days_from_tx"] = df.groupby("patient_id")["anchor_dt"].transform(
        lambda x: (x - x.min()).dt.days
    )

    # Percentile-rank survival hazards (like setmilmt_preds)
    for col in ["hazard_acr_surv", "hazard_clad_surv", "hazard_death_surv"]:
        arr = df[col].values.astype(float)
        valid = ~np.isnan(arr)
        ranks = np.full(len(arr), float("nan"))
        if valid.sum() > 0:
            order = np.argsort(arr[valid])
            pct = np.linspace(0, 1, valid.sum())
            pct_sorted = np.empty(valid.sum())
            pct_sorted[order] = pct
            ranks[valid] = pct_sorted
        df[f"pct_{col.replace('hazard_', '')}"] = ranks

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows → {OUT_CSV}")
    print(df[["patient_id", "anchor_dt", "score_acr_cls", "hazard_acr_surv", "hazard_death_surv"]].head(8).to_string())


if __name__ == "__main__":
    main()
