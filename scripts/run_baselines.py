"""
run_baselines.py — comprehensive baseline benchmark for v8 nested-CV.

Classical baselines read .pt files directly (no bag_cache preload):
  mean_concat   — per-modality mean pool → concat → logistic/Cox
  cluster_prop  — instance_cluster_ids → proportions → CLR → logistic/Cox
                  n_clusters from cluster_count_onehot.shape[0] (global vocab)
                  Clinical: mean pool inputs

DL results loaded from saved metrics JSONs:
  early/late/middle_cls, *_acr_surv, *_clad_surv, *_death_surv
  slot_mega

Output: results/mm_abmil_v8/baselines_summary.json + printed table
"""

import argparse
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import Path
from collections import defaultdict

import numpy as np

from mil.data.splits import build_splits_multitask
from mil.training.classical_baselines import run_classical_baselines


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples-dir",  required=True)
    p.add_argument("--splits-csv",   required=True)
    p.add_argument("--split",        type=int, default=1)
    p.add_argument("--folds",        type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--out-dir",      required=True)
    p.add_argument("--props-csv",    default=None,
                   help="Pre-computed cluster proportions CSV (from save_cluster_props.sh). "
                        "Speeds up ClusterProp by skipping .pt loads for patch modalities.")
    return p.parse_args()


def load_dl_results(out_dir: Path, split: int, folds: list) -> dict:
    dl: dict = defaultdict(lambda: defaultdict(list))
    for fold in folds:
        pattern = f"metrics_split{split}_fold{fold}_*.json"
        for jf in sorted(out_dir.glob(pattern)):
            model = jf.stem.replace(f"metrics_split{split}_fold{fold}_", "")
            try:
                d = json.load(open(jf)).get("test", {})
            except Exception:
                continue
            for k, v in d.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    dl[model][k].append(v)
    return dl


def print_summary(classical_per_fold: list, dl_results: dict) -> dict:
    agg: dict = defaultdict(lambda: defaultdict(list))
    for fold_res in classical_per_fold:
        for model, metrics in fold_res.items():
            for k, v in metrics.items():
                agg[model][k].append(v)

    all_models = {}
    for model, metrics in agg.items():
        all_models[f"[classical] {model}"] = {k: float(np.mean(v)) for k, v in metrics.items()}
    for model, metrics in dl_results.items():
        all_models[f"[dl] {model}"] = {k: float(np.mean(v)) for k, v in metrics.items()}

    print(f"\n{'='*80}")
    print(f"  BASELINE SUMMARY  (mean across {len(classical_per_fold)} folds)")
    print(f"{'='*80}")
    print(f"  {'Model':<38} {'BACC':>6} {'AUC':>6} {'ci_acr':>7} {'ci_clad':>8} {'ci_death':>9}")
    print(f"  {'-'*73}")
    for name, m in sorted(all_models.items()):
        bacc     = m.get("bacc",          m.get("test_bacc",    float("nan")))
        auc      = m.get("auc",           m.get("test_auc",     float("nan")))
        ci_acr   = m.get("c_index",       m.get("test_ci_acr",  float("nan")))
        ci_clad  = m.get("clad_c_index",  m.get("test_ci_clad", float("nan")))
        ci_death = m.get("death_c_index", m.get("test_ci_death",float("nan")))
        def fmt(x): return f"{x:6.3f}" if not (x != x) else "    — "
        print(f"  {name:<38} {fmt(bacc)} {fmt(auc)} {fmt(ci_acr)} {fmt(ci_clad)} {fmt(ci_death)}")
    print(f"{'='*80}\n")
    return all_models


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    print(f"\n  Baseline benchmark  split={args.split}  folds={args.folds}")
    print(f"  samples: {args.samples_dir}")

    dl_results = load_dl_results(out_dir, args.split, args.folds)
    print(f"  DL models found: {sorted(dl_results.keys())}\n")

    classical_per_fold = []
    for fold in args.folds:
        print(f"\n{'─'*60}  fold={fold}  {'─'*60}")
        splits     = build_splits_multitask(args.samples_dir, args.splits_csv,
                                            fold, split=args.split)
        train_recs = splits["train"]
        val_recs   = splits["val"]
        test_recs  = splits["test"]

        fold_res = run_classical_baselines(
            args.samples_dir, train_recs, test_recs, val_recs=val_recs,
            props_csv=args.props_csv)
        classical_per_fold.append(fold_res)

    all_models = print_summary(classical_per_fold, dl_results)

    out_file = out_dir / "baselines_summary.json"
    def clean(d):
        if isinstance(d, dict): return {k: clean(v) for k, v in d.items()}
        if isinstance(d, float) and d != d: return None
        return d
    with open(out_file, "w") as f:
        json.dump(clean(all_models), f, indent=2)
    print(f"  Saved → {out_file}")


if __name__ == "__main__":
    main()
