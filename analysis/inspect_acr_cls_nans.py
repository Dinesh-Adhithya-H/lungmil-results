"""Inspect why acr_cls biopsy_reps have NaN risk values."""
from pathlib import Path
import numpy as np
import torch

REPS_DIR = Path(__file__).resolve().parent.parent / "results" / "mm_abmil_v8" / "biopsy_reps"

total = 0
nan_total = 0
for s in range(5):
    p = REPS_DIR / f"biopsy_reps_acr_cls_split{s}.pt"
    if not p.exists():
        print(f"split{s}: FILE MISSING")
        continue
    d = torch.load(p, map_location="cpu", weights_only=False)
    risk  = d["risk"].numpy()
    label = d["label"].numpy() if "label" in d else None
    n_nan = int(np.isnan(risk).sum())
    n     = len(risk)
    total += n
    nan_total += n_nan

    print(f"\nsplit{s}: {n} biopsies, {n_nan} NaN risk ({100*n_nan/n:.1f}%)")
    print(f"  keys: {list(d.keys())}")
    print(f"  risk dtype: {risk.dtype}, min={np.nanmin(risk):.4f}, max={np.nanmax(risk):.4f}")
    if label is not None:
        lbl_nan = int(np.isnan(label).sum())
        print(f"  label: {n - lbl_nan} valid, {lbl_nan} NaN")
        unique_lbl, counts = np.unique(label[~np.isnan(label)].astype(int), return_counts=True)
        print(f"  label distribution: {dict(zip(unique_lbl.tolist(), counts.tolist()))}")

    # Check if NaN risk correlates with label
    nan_mask = np.isnan(risk)
    if label is not None and nan_mask.any():
        lbl_where_nan = label[nan_mask]
        lbl_valid = lbl_where_nan[~np.isnan(lbl_where_nan)].astype(int)
        if len(lbl_valid):
            unique_l, cnt_l = np.unique(lbl_valid, return_counts=True)
            print(f"  label dist among NaN-risk biopsies: {dict(zip(unique_l.tolist(), cnt_l.tolist()))}")
        else:
            print(f"  all NaN-risk biopsies also have NaN label")

    # Check if NaN biopsies have tte info
    if "tte" in d:
        tte = d["tte"].numpy()
        print(f"  tte NaN among NaN-risk: {int(np.isnan(tte[nan_mask]).sum())} / {int(nan_mask.sum())}")

print(f"\n=== TOTAL: {total} biopsies, {nan_total} NaN risk ({100*nan_total/total:.1f}%) ===")
