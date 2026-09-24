"""
Re-run ACR survival inference for LongMIL v2 with fixed per-biopsy evaluation.

Previously acr_surv was evaluated patient-level (one hazard per patient using
all biopsies). This script uses the corrected per-biopsy gap-time approach:
at each biopsy t, use all biopsies 1..t to predict time-to-next-ACR from t.
This matches how clad/death are evaluated and how the model is trained.

Updates results/predictions/comparison_acr_surv.csv in-place (replaces LongMK-v2 row).
Run via: sbatch analysis/submit_rerun_acr_surv.sh
"""
import sys, math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "interpretability"))

from mil.data.splits import build_splits_longitudinal
from shared import SPLITS_CSV, SAMPLES_DIR
from interpret_longitudinal_mk import load_model, load_patient_bags, extract_patient_longitudinal

VARIANT   = "longitudinal_mk_no_alibi_v2"
TASK      = "acr_surv"
N_SPLITS  = 5
FOLD      = 0
OUT_CSV   = ROOT / "results" / "predictions" / "comparison_acr_surv.csv"
MODEL_ROW = "LongMK-v2"   # name in comparison CSV to replace


def eval_split(split: int, device: torch.device) -> float:
    """Return C-index for one split using per-biopsy ACR surv evaluation."""
    from lifelines.utils import concordance_index

    model, _ = load_model(split, FOLD, device, task=TASK, variant=VARIANT)
    model.eval()

    split_info = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=FOLD, split=split)
    test_patients = split_info.get("test", [])
    print(f"  split={split}  {len(test_patients)} test patients")

    all_h, all_t, all_e = [], [], []

    for patient in test_patients:
        bags_list, _ = load_patient_bags(patient, device)
        records = patient["records"]

        with torch.no_grad():
            result = model(
                {"bags_list": bags_list,
                 "days":      patient["days"],
                 "records":   records},
                device,
            )

        biopsy_hazards = result.get("acr_surv", [])
        if not isinstance(biopsy_hazards, list):
            continue

        for hazard, tte, ev in biopsy_hazards:
            h = hazard.float().item() if isinstance(hazard, torch.Tensor) else float(hazard)
            if math.isnan(h) or math.isnan(tte) or tte < 0:
                continue
            all_h.append(h)
            all_t.append(float(tte))
            all_e.append(float(ev))

    del model
    torch.cuda.empty_cache()

    n_ev = sum(all_e)
    print(f"  split={split}  n_biopsies={len(all_h)}  n_events={int(n_ev)}")
    if len(all_h) < 2 or n_ev == 0:
        print(f"  split={split}  WARNING: too few events, returning 0.5")
        return float("nan")

    ci = concordance_index(all_t, [-h for h in all_h], all_e)
    print(f"  split={split}  C-index={ci:.4f}")
    return ci


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    split_cis = []
    for split in range(N_SPLITS):
        ci = eval_split(split, device)
        split_cis.append(ci)

    valid = [c for c in split_cis if not math.isnan(c)]
    mean  = float(np.mean(valid)) if valid else float("nan")
    std   = float(np.std(valid))  if valid else float("nan")
    print(f"\nACR Survival C-index (per-biopsy):  {mean:.4f} ± {std:.4f}")
    print(f"Per-split: {[f'{c:.4f}' for c in split_cis]}")

    # Update comparison CSV
    df = pd.read_csv(OUT_CSV)
    mask = df["model"] == MODEL_ROW
    if not mask.any():
        print(f"WARNING: row '{MODEL_ROW}' not found in {OUT_CSV}, appending.")
        new_row = {"model": MODEL_ROW}
        for i, c in enumerate(split_cis):
            new_row[f"s{i}"] = round(c, 4)
        new_row["mean"] = round(mean, 4)
        new_row["std"]  = round(std, 4)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        for i, c in enumerate(split_cis):
            df.loc[mask, f"s{i}"] = round(c, 4)
        df.loc[mask, "mean"] = round(mean, 4)
        df.loc[mask, "std"]  = round(std, 4)

    df.to_csv(OUT_CSV, index=False)
    print(f"Updated {OUT_CSV}")


if __name__ == "__main__":
    main()
