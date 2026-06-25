#!/usr/bin/env python3
"""
make_tcga_splits.py — Stratified 5-fold CV splits for TCGA-GBM survival.

Stratification: survival time binned into quartiles (4 strata).
For each outer fold i  (i = 0..4):
  test  = fold_i
  val   = fold_{(i+1) % 5}
  train = remaining 3 folds

Output CSV columns:
  idx, identifier, os_time, os_status, fold_0 … fold_4
  (values: "train" | "val" | "test")
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

MANIFEST = "/lustre/groups/aih/dinesh.haridoss/mil/tcga_cache_gbm/manifest.csv"
OUT      = "/home/aih/dinesh.haridoss/chicago_mil/tcga_gbm_splits.csv"
N_FOLDS  = 5
N_BINS   = 4   # quartiles for stratification
SEED     = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--out",      default=OUT)
    ap.add_argument("--n_folds",  type=int, default=N_FOLDS)
    ap.add_argument("--n_bins",   type=int, default=N_BINS)
    ap.add_argument("--seed",     type=int, default=SEED)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    print(f"Loaded {len(df)} samples from {args.manifest}")
    print(f"  os_status unique: {df['os_status'].unique()}")
    print(f"  os_time  : min={df['os_time'].min():.2f}  "
          f"max={df['os_time'].max():.2f}  "
          f"median={df['os_time'].median():.2f}")

    # Stratification label: event_status × time_quartile
    t  = df["os_time"].values
    ev = df["os_status"].astype(int).values
    q_labels = pd.qcut(t, q=args.n_bins, labels=False, duplicates="drop")
    strat = ev * args.n_bins + q_labels.astype(int)
    print(f"  Strata distribution: {np.bincount(strat)}")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                          random_state=args.seed)
    fold_assignments = np.full(len(df), -1, dtype=int)
    for fold_idx, (_, test_idx) in enumerate(skf.split(df, strat)):
        fold_assignments[test_idx] = fold_idx

    assert (fold_assignments == -1).sum() == 0, "Some samples unassigned!"

    out_df = df[["idx", "identifier", "os_time", "os_status"]].copy()

    for outer_fold in range(args.n_folds):
        col = f"fold_{outer_fold}"
        split_label = []
        for assignment in fold_assignments:
            if assignment == outer_fold:
                split_label.append("test")
            elif assignment == (outer_fold + 1) % args.n_folds:
                split_label.append("val")
            else:
                split_label.append("train")
        out_df[col] = split_label

        tr = split_label.count("train")
        vl = split_label.count("val")
        te = split_label.count("test")
        print(f"  fold_{outer_fold}: train={tr}  val={vl}  test={te}")

    out_df.to_csv(args.out, index=False)
    print(f"\nSaved → {args.out}")

    # Sanity: check time distribution per split for fold_0
    for sp in ("train", "val", "test"):
        mask  = out_df["fold_0"] == sp
        times = out_df.loc[mask, "os_time"]
        print(f"  fold_0 [{sp:5s}] n={mask.sum()}  "
              f"os_time median={times.median():.1f}  "
              f"q25={times.quantile(.25):.1f}  "
              f"q75={times.quantile(.75):.1f}")


if __name__ == "__main__":
    main()
