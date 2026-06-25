#!/usr/bin/env python3
"""
make_tcga_multitask_splits.py
Create stratified 5-fold CV splits for TCGA multi-task benchmark.

Outputs one CSV per dataset to:
  /home/aih/dinesh.haridoss/chicago_mil/data/tcga_splits/<cancer>.csv

Columns:
  key         — unique sample key used as bag-cache key
  cancer      — e.g. "gbm" or "lgg"
  idx         — integer idx matching the .pt filename (00001.pt → 1)
  identifier  — TCGA barcode
  cls_label   — binary classification label (1=GBM, 0=LGG for gbmlgg; NaN for others)
  os_status, os_time
  dss_status, dss_time
  pfi_status, pfi_time
  fold_0 … fold_4  — "train" | "val" | "test"

Usage (via sbatch):
  python3 make_tcga_multitask_splits.py --cancer gbmlgg
  python3 make_tcga_multitask_splits.py --cancer blca
  python3 make_tcga_multitask_splits.py --cancer kirc
  python3 make_tcga_multitask_splits.py --cancer luad
  python3 make_tcga_multitask_splits.py --cancer brca
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

LUSTRE = "/lustre/groups/aih/dinesh.haridoss/mil"
OUT_DIR = Path("/home/aih/dinesh.haridoss/chicago_mil/data/tcga_splits")
N_FOLDS = 5
SEED = 42

CACHE_MAP = {
    "gbm":  f"{LUSTRE}/tcga_cache_gbm",
    "lgg":  f"{LUSTRE}/tcga_cache_lgg",
    "blca": f"{LUSTRE}/tcga_cache_blca",
    "brca": f"{LUSTRE}/tcga_cache_brca",
    "kirc": f"{LUSTRE}/tcga_cache_kirc",
    "luad": f"{LUSTRE}/tcga_cache_luad",
}

CANCER_CONFIGS = {
    "gbmlgg": {
        "cancers": ["gbm", "lgg"],
        "cls_label": {"gbm": 1, "lgg": 0},
    },
    "blca": {"cancers": ["blca"], "cls_label": {}},
    "brca": {"cancers": ["brca"], "cls_label": {}},
    "kirc": {"cancers": ["kirc"], "cls_label": {}},
    "luad": {"cancers": ["luad"], "cls_label": {}},
}


def load_survival_from_pt(pt_path: Path) -> dict:
    """Read OS/DSS/PFI from the .pt file's survival dict."""
    try:
        d = torch.load(pt_path, map_location="cpu", weights_only=False)
        surv = d.get("survival", {})
        out = {}
        for endpoint in ("OS", "DSS", "PFI"):
            ep = surv.get(endpoint, {})
            out[f"{endpoint.lower()}_status"] = ep.get("status", float("nan"))
            out[f"{endpoint.lower()}_time"]   = ep.get("time",   float("nan"))
        del d
        return out
    except Exception as e:
        print(f"  [warn] failed {pt_path.name}: {e}")
        return {
            "os_status": float("nan"), "os_time": float("nan"),
            "dss_status": float("nan"), "dss_time": float("nan"),
            "pfi_status": float("nan"), "pfi_time": float("nan"),
        }


def build_records_for_cancer(cancer: str, cls_label_val) -> list:
    cache_dir = Path(CACHE_MAP[cancer])
    manifest  = pd.read_csv(cache_dir / "manifest.csv")
    samples_dir = cache_dir / "samples"

    print(f"  [{cancer}] manifest rows={len(manifest)}  cache={cache_dir}")

    records = []
    for _, row in manifest.iterrows():
        idx  = int(row["idx"])
        key  = f"{cancer}_{idx:05d}"
        pt   = samples_dir / f"{idx:05d}.pt"
        surv = load_survival_from_pt(pt) if pt.exists() else {
            "os_status": float("nan"), "os_time": float("nan"),
            "dss_status": float("nan"), "dss_time": float("nan"),
            "pfi_status": float("nan"), "pfi_time": float("nan"),
        }
        records.append({
            "key":          key,
            "cancer":       cancer,
            "idx":          idx,
            "identifier":   str(row["identifier"]),
            "cls_label":    cls_label_val if cls_label_val is not None else float("nan"),
            **surv,
        })
        if len(records) % 100 == 0:
            print(f"    loaded {len(records)}/{len(manifest)} ...", flush=True)

    return records


def make_strat_label(df: pd.DataFrame) -> np.ndarray:
    """Stratification: event_status × time_quartile (uses OS)."""
    t  = df["os_time"].fillna(df["os_time"].median()).values
    ev = df["os_status"].fillna(0).astype(int).values
    q  = pd.qcut(t, q=4, labels=False, duplicates="drop").astype(int)
    # Also include cls_label if available
    has_cls = not df["cls_label"].isna().all()
    if has_cls:
        cls = df["cls_label"].fillna(0).astype(int).values
        return cls * 8 + ev * 4 + q
    return ev * 4 + q


def make_splits(df: pd.DataFrame) -> pd.DataFrame:
    strat = make_strat_label(df)
    print(f"  strat counts: {np.bincount(strat)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_assignments = np.full(len(df), -1, dtype=int)
    for fold_idx, (_, test_idx) in enumerate(skf.split(df, strat)):
        fold_assignments[test_idx] = fold_idx

    out_df = df.copy()
    for outer_fold in range(N_FOLDS):
        col = f"fold_{outer_fold}"
        labels = []
        for asgn in fold_assignments:
            if asgn == outer_fold:
                labels.append("test")
            elif asgn == (outer_fold + 1) % N_FOLDS:
                labels.append("val")
            else:
                labels.append("train")
        out_df[col] = labels
        tr = labels.count("train"); vl = labels.count("val"); te = labels.count("test")
        print(f"  fold_{outer_fold}: train={tr}  val={vl}  test={te}")

    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=list(CANCER_CONFIGS.keys()))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / f"{args.cancer}.csv"

    cfg = CANCER_CONFIGS[args.cancer]
    cls_map = cfg["cls_label"]

    print(f"\n{'='*60}")
    print(f"  Building splits: {args.cancer}")
    print(f"  Cancers: {cfg['cancers']}")
    print(f"  cls_label: {cls_map if cls_map else 'N/A'}")
    print(f"{'='*60}")

    all_records = []
    for cancer in cfg["cancers"]:
        cls_val = cls_map.get(cancer, None)
        recs = build_records_for_cancer(cancer, cls_val)
        all_records.extend(recs)
        print(f"  [{cancer}] {len(recs)} samples loaded")

    df = pd.DataFrame(all_records)
    print(f"\n  Total samples: {len(df)}")
    print(f"  OS events: {df['os_status'].sum():.0f} / {df['os_status'].notna().sum()}")
    if not df["cls_label"].isna().all():
        print(f"  cls_label: {df['cls_label'].value_counts().to_dict()}")

    df = make_splits(df)
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")

    # Print summary stats
    print(f"\n  Summary:")
    print(f"  {'col':20s}  {'non_null':>8}  {'min':>8}  {'max':>8}  {'mean':>8}")
    for col in ["os_time", "dss_time", "pfi_time"]:
        v = df[col].dropna()
        if len(v):
            print(f"  {col:20s}  {len(v):8d}  {v.min():8.2f}  {v.max():8.2f}  {v.mean():8.2f}")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
