#!/usr/bin/env python3
"""
extract_clinical_features.py
Extract clinical_raw_tensor (106 features) from every mil_v2 .pt bag file.
Saves to:  results/cluster_proportions/clinical_features.csv
           results/cluster_proportions/clinical_feature_names.csv

Run via sbatch — do NOT run on the login node.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

SAMPLES_DIR = Path("/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples")
OUT_DIR     = Path("/home/aih/dinesh.haridoss/chicago_mil/results/cluster_proportions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pt_files = sorted(SAMPLES_DIR.glob("*.pt"))
print(f"Found {len(pt_files)} .pt files", flush=True)

rows = []
feat_names = None

for i, pt_path in enumerate(pt_files):
    if i % 500 == 0:
        print(f"  {i}/{len(pt_files)} …", flush=True)
    try:
        bag = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [SKIP] {pt_path.name}: {e}", flush=True)
        continue

    raw = bag.get("clinical_raw_tensor", None)
    if raw is None:
        continue

    arr = raw.numpy().astype(float)   # NaN already encoded as nan

    # Save feature names once
    if feat_names is None:
        feat_names = bag.get("clinical_feature_names", [f"feat_{j}" for j in range(len(arr))])
        print(f"  n_features = {len(feat_names)}", flush=True)

    stem = pt_path.stem                          # e.g. "00001"
    patient_id  = bag.get("identifier", "")
    anchor_dt   = bag.get("anchor_time", "")

    row = {"stem": stem, "patient_id": patient_id, "anchor_dt": anchor_dt}
    for j, name in enumerate(feat_names):
        row[f"feat_{j}"] = float(arr[j]) if not np.isnan(arr[j]) else np.nan
    rows.append(row)

print(f"Extracted {len(rows)} samples", flush=True)

df = pd.DataFrame(rows)
out_csv = OUT_DIR / "clinical_features.csv"
df.to_csv(out_csv, index=False)
print(f"Saved → {out_csv}", flush=True)

# Feature name mapping
if feat_names:
    nm_df = pd.DataFrame({"idx": range(len(feat_names)), "name": feat_names})
    out_nm = OUT_DIR / "clinical_feature_names.csv"
    nm_df.to_csv(out_nm, index=False)
    print(f"Saved → {out_nm}", flush=True)

print("Done.", flush=True)
