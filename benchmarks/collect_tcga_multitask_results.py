#!/usr/bin/env python3
"""
Collect and print results table for TCGA multi-task benchmark.

Compares our TCGASetTransformerMIL against existing baselines.
Run after training is complete.
"""

import json
from pathlib import Path
import numpy as np

RESULTS_DIR = Path("/home/aih/dinesh.haridoss/chicago_mil/results_tcga_multitask")
BASELINE_DIR = Path("/lustre/groups/aih/dinesh.haridoss/mil/baseline_results")

CANCERS = ["gbmlgg", "blca", "kirc", "luad", "brca"]
FOLDS   = [0, 1, 2, 3, 4]


def load_ours(cancer: str) -> dict:
    summary = RESULTS_DIR / cancer / "summary.json"
    if not summary.exists(): return {}
    d = json.load(open(summary))
    folds = d.get("folds", {})

    out: dict = {}
    for fold_key, metrics in folds.items():
        for k, v in metrics.items():
            if k not in out: out[k] = []
            out[k].append(v)
    return {k: (np.mean(v), np.std(v), v) for k, v in out.items() if v}


def load_baseline_ci(model_prefix: str, cancer: str) -> dict:
    """Load C-index from AMIL/MCAT/SurvPath baseline dirs."""
    vals = []
    for fold in FOLDS:
        fold_dir = BASELINE_DIR / f"{model_prefix}_{cancer}_fold{fold}"
        # Look for experiment txt files with c-index result
        for txt in fold_dir.rglob("*.txt"):
            content = txt.read_text(errors="ignore")
            # Format: c_index lines like 'fold 0: c_index = 0.7234'
            for line in content.split("\n"):
                if "c_index" in line.lower() or "cindex" in line.lower():
                    parts = line.strip().split()
                    for i, p in enumerate(parts):
                        try:
                            v = float(p.replace(",", ""))
                            if 0.4 < v < 1.0:
                                vals.append(v)
                                break
                        except ValueError:
                            pass
            if len(vals) > len(FOLDS) - fold - 1:
                break
    if not vals: return {}
    return {"os_ci": (np.mean(vals), np.std(vals), vals)}


BASELINE_PREFIXES = {
    "AMIL (WSI+omics)": "amil",
    "MCAT": "mcat",
    "SurvPath": "survpath",
}


def print_table(cancer: str):
    print(f"\n{'═'*80}")
    print(f"  {cancer.upper()}")
    print(f"{'─'*80}")

    ours = load_ours(cancer)
    if not ours:
        print(f"  (no results yet)")
        return

    # Header
    tasks_keys = sorted(ours.keys())
    tasks_keys = [k for k in tasks_keys if k != "val_primary"]
    header = f"  {'Model':30s}"
    for t in tasks_keys:
        header += f"  {t:>14s}"
    print(header)
    print(f"  {'─'*30}" + "  " + "  ".join(["─"*14] * len(tasks_keys)))

    # Baselines (OS C-index only)
    for name, prefix in BASELINE_PREFIXES.items():
        bl = load_baseline_ci(prefix, cancer)
        row = f"  {name:30s}"
        for t in tasks_keys:
            if t == "os_ci" and bl.get("os_ci"):
                m, s, _ = bl["os_ci"]
                row += f"  {m:.4f}±{s:.4f}"
            else:
                row += f"  {'—':>14s}"
        print(row)

    # Ours
    row = f"  {'TCGASetTransformerMIL (ours)':30s}"
    for t in tasks_keys:
        if t in ours:
            m, s, _ = ours[t]
            row += f"  {m:.4f}±{s:.4f}"
        else:
            row += f"  {'—':>14s}"
    print(row)

    print(f"{'─'*80}")
    print(f"  Per-fold (ours):")
    for t in tasks_keys:
        if t in ours:
            _, _, vals = ours[t]
            print(f"    {t:20s}: {[round(v,4) for v in vals]}")


if __name__ == "__main__":
    for cancer in CANCERS:
        print_table(cancer)
    print()
