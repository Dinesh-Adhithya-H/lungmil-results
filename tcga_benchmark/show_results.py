#!/usr/bin/env python3
"""
Collect and display TCGA benchmark results.
Prints conference-style table: rows=methods, columns=cancer types.
"""
import json
import numpy as np
from pathlib import Path

BASE     = Path(__file__).parent.parent / "results" / "tcga_benchmark"
CANCERS  = ["KIRC", "BRCA", "BLCA", "LGG", "GBM"]
MODELS   = ["abmil", "transmil", "dsmil", "slotmil", "geomae_slotmil"]
MODEL_LABELS = {
    "abmil":          "ABMIL",
    "transmil":       "TransMIL",
    "dsmil":          "DSMIL",
    "slotmil":        "SlotMIL (ours)",
    "geomae_slotmil": "GeoMAE-SlotMIL (ours)",
}
N_FOLDS = 5


def collect():
    res = {}
    for model in MODELS:
        res[model] = {}
        for cancer in CANCERS:
            vals = []
            for fold in range(N_FOLDS):
                rf = BASE / cancer / model / f"fold{fold}" / "result.json"
                if rf.exists():
                    d = json.loads(rf.read_text())
                    vals.append(d["test_cidx"])
            res[model][cancer] = vals
    return res


def print_table(res):
    print("\n" + "=" * 85)
    print("  TCGA Survival Benchmark — OS C-index  (5-fold CV, mean ± std)")
    print("=" * 85)
    print(f"\n  {'Method':28s}", end="")
    for c in CANCERS:
        print(f"  {c:>12}", end="")
    print(f"  {'Avg':>10}  N")
    print("  " + "─" * 78)

    # Best per column for bolding
    best = {}
    for cancer in CANCERS:
        cands = []
        for model in MODELS:
            v = res[model][cancer]
            if v: cands.append(np.mean(v))
        best[cancer] = max(cands) if cands else None

    for model in MODELS:
        label = MODEL_LABELS[model]
        row_vals = []
        n_folds  = []
        for cancer in CANCERS:
            v = res[model][cancer]
            if v:
                row_vals.append(np.mean(v))
                n_folds.append(len(v))
            else:
                row_vals.append(None)
                n_folds.append(0)

        print(f"  {label:28s}", end="")
        for i, (cancer, rv) in enumerate(zip(CANCERS, row_vals)):
            if rv is None:
                print(f"  {'—':>12}", end="")
            else:
                s    = np.std(res[model][cancer])
                mark = "★" if best[cancer] and abs(rv - best[cancer]) < 0.001 else " "
                print(f"  {mark}{rv:.3f}±{s:.3f}", end="")

        valid = [v for v in row_vals if v is not None]
        avg   = np.mean(valid) if valid else None
        n_str = str(min(n_folds)) if n_folds else "0"
        avg_str = f"{avg:.3f}" if avg else "—"
        print(f"  {avg_str:>10}  {n_str}")

    print()


if __name__ == "__main__":
    res = collect()
    print_table(res)

    # Also show pending jobs
    pending = []
    for model in MODELS:
        for cancer in CANCERS:
            done = sum(1 for fold in range(N_FOLDS)
                       if (BASE/cancer/model/f"fold{fold}/result.json").exists())
            if done < N_FOLDS:
                pending.append(f"{cancer}/{model} ({done}/{N_FOLDS})")
    if pending:
        print(f"  Pending: {', '.join(pending[:8])}")
        if len(pending) > 8:
            print(f"  ... and {len(pending)-8} more")
