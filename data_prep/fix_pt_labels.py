#!/usr/bin/env python3
"""
Fix labels in .pt files using CSV as ground truth.

Updates per sample:
  - label               (binary ACR 0/1 from CSV)
  - metadata['ACR Status/Grade']  (string grade from CSV)
  - metadata['acr_encoded']       (float from CSV)
  - survival['CLAD']              (status + days from CSV columns)
  - survival['Death']             (status + days from CSV columns)
"""
import argparse
import os
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples_dir",
                   default="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples")
    p.add_argument("--splits_csv",
                   default="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
    p.add_argument("--dry_run", action="store_true",
                   help="Print diffs without saving")
    return p.parse_args()


def build_surv_entry(status, days):
    """Return clean survival dict (no string-wrapped dicts)."""
    return {
        "status": float(status) if pd.notna(status) else float("nan"),
        "days":   float(days)   if pd.notna(days)   else float("nan"),
    }


def main():
    args = parse_args()
    samples_dir = Path(args.samples_dir)

    df = pd.read_csv(args.splits_csv)
    # Index by filename (basename, e.g. "00000.pt")
    df["_fname"] = df["file"].apply(lambda x: Path(x).name)
    df = df.set_index("_fname")

    pt_files = sorted(samples_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} .pt files, {len(df)} CSV rows")

    n_fixed = 0
    n_missing = 0
    diffs = []

    for pt_path in tqdm(pt_files, desc="Fixing labels"):
        fname = pt_path.name
        if fname not in df.index:
            n_missing += 1
            continue

        row = df.loc[fname]

        d = torch.load(pt_path, map_location="cpu", weights_only=False)
        changed = False

        # ── ACR label ────────────────────────────────────────────────────
        csv_label = int(row["label"]) if pd.notna(row["label"]) else 0
        if d.get("label") != csv_label:
            if args.dry_run:
                diffs.append(f"{fname}: label {d.get('label')} → {csv_label}")
            d["label"] = csv_label
            changed = True

        # ── metadata ─────────────────────────────────────────────────────
        if not isinstance(d.get("metadata"), dict):
            d["metadata"] = {}

        csv_grade   = str(row["acr_grade"]) if pd.notna(row["acr_grade"]) else "A0B0"
        csv_encoded = float(row["acr_encoded"]) if pd.notna(row["acr_encoded"]) else 1.0

        if d["metadata"].get("ACR Status/Grade") != csv_grade:
            d["metadata"]["ACR Status/Grade"] = csv_grade
            changed = True

        if d["metadata"].get("acr_encoded") != csv_encoded:
            d["metadata"]["acr_encoded"] = csv_encoded
            changed = True

        # ── survival ─────────────────────────────────────────────────────
        new_surv = {
            "CLAD":  build_surv_entry(row.get("clad_status"),  row.get("clad_days")),
            "Death": build_surv_entry(row.get("death_status"), row.get("death_days")),
        }
        # Compare to existing (may be string-encoded dicts)
        old_surv = d.get("survival", {})
        if old_surv != new_surv:
            d["survival"] = new_surv
            changed = True

        if changed:
            n_fixed += 1
            if not args.dry_run:
                torch.save(d, pt_path)

    if args.dry_run and diffs:
        print("\nSample diffs (first 20):")
        for diff in diffs[:20]:
            print(" ", diff)

    print(f"\nDone: {n_fixed} files updated, {n_missing} not in CSV")


if __name__ == "__main__":
    main()
