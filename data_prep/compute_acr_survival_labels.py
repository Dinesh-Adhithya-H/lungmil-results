#!/usr/bin/env python3
"""
compute_acr_survival_labels.py
==============================
Adds acr_days and acr_status columns to the multimodal splits CSV for
time-to-first-ACR survival modelling.

acr_status : 1 if patient ever has label==1 (ACR+), else 0
acr_days   : days from this sample's anchor_dt to the start of the
             patient's first ACR episode (first anchor_dt where label==1).
             Positive  → pre-episode sample (use for survival training)
             1         → this IS the first ACR+ sample (event=1, t=1)
             Negative  → after first episode (dropped by trainer)
             NaN       → never-ACR patient (censored; trainer uses study_end proxy)

Usage:
    python compute_acr_survival_labels.py \
        --csv /path/to/multimodal_splits_nested_cv.csv \
        --out /path/to/multimodal_splits_nested_cv.csv   # overwrite in place
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None,
                    help="Output path (default: overwrite input)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path(args.csv)

    df = pd.read_csv(args.csv, parse_dates=["anchor_dt"])
    df = df.sort_values(["patient_id", "anchor_dt"]).reset_index(drop=True)

    # First ACR+ date per patient
    acr_pos = df[df["label"] == 1].groupby("patient_id")["anchor_dt"].min()
    acr_pos.name = "first_acr_dt"

    df = df.join(acr_pos, on="patient_id")

    df["acr_status"] = df["first_acr_dt"].notna().astype(float)
    raw_days = np.where(
        df["first_acr_dt"].notna(),
        (df["first_acr_dt"] - df["anchor_dt"]).dt.days.astype(float),
        np.nan
    )
    # First ACR+ sample has raw_days=0 → set to 1 so it passes the t>0 filter
    # as event=1 (the event occurs at this visit, TTE≈0, use 1 day)
    df["acr_days"] = np.where(raw_days == 0, 1.0, raw_days)

    df = df.drop(columns=["first_acr_dt"])

    # Summary
    n_event    = (df.groupby("patient_id")["acr_status"].first() == 1).sum()
    n_censor   = (df.groupby("patient_id")["acr_status"].first() == 0).sum()
    pre_rows   = (df["acr_status"] == 1) & (df["acr_days"] > 1)
    event_rows = (df["acr_status"] == 1) & (df["acr_days"] == 1)
    post_rows  = (df["acr_status"] == 1) & (df["acr_days"] < 0)
    print(f"Patients:  {n_event} with ACR  |  {n_censor} censored (never ACR)")
    print(f"Rows:      {pre_rows.sum()} pre-episode  |  {event_rows.sum()} event (ACR+, t=1)  |  {post_rows.sum()} post-episode (dropped)")
    print(f"           {(df['acr_status']==0).sum()} censored rows")

    df.to_csv(out, index=False)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
