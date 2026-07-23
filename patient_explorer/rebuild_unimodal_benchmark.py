"""Rebuild unimodal rows in benchmark_results.csv from phase1 metrics across all 5 splits."""
import json
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/phase1")
OUT_CSV = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/patient_explorer/data/benchmark_results.csv")

MODS = ["HE", "BAL", "CT", "Clinical"]
TASK_MAP = {
    "acr":      ("acr_cls",   "BAcc",  "bacc"),
    "acr_surv": ("acr_surv",  "C-idx", "c_index"),
    "clad":     ("clad_surv", "C-idx", "c_index"),
    "death":    ("death_surv","C-idx", "c_index"),
}
AUC_TASK = ("acr_cls", "AUC", "auc")  # extra metric for acr only

rows = []
for task_dir, (task_name, metric_name, metric_key) in TASK_MAP.items():
    for mod in MODS:
        split_vals = {}
        for split in range(5):
            p = RESULTS / f"split{split}_fold0" / task_dir / mod / "final_combined" / "metrics.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            v = d.get("test", {}).get(metric_key)
            if v is not None:
                split_vals[split] = float(v)

        if not split_vals:
            continue

        vals = list(split_vals.values())
        mean = float(np.mean(vals))
        std  = float(np.std(vals)) if len(vals) > 1 else float("nan")
        row = {
            "phase": "P1", "model": f"unimodal_{mod}",
            "task": task_name, "metric": metric_name,
            "mean": mean, "std": std,
            "s0": split_vals.get(0), "s1": split_vals.get(1),
            "s2": split_vals.get(2), "s3": split_vals.get(3),
            "s4": split_vals.get(4),
        }
        rows.append(row)
        print(f"  {task_name} {mod} {metric_name}: {mean:.4f} ± {std:.4f}  n={len(vals)}")

    # AUC for acr_cls
    if task_dir == "acr":
        for mod in MODS:
            split_vals = {}
            for split in range(5):
                p = RESULTS / f"split{split}_fold0" / task_dir / mod / "final_combined" / "metrics.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                v = d.get("test", {}).get("auc")
                if v is not None:
                    split_vals[split] = float(v)
            if not split_vals:
                continue
            vals = list(split_vals.values())
            row = {
                "phase": "P1", "model": f"unimodal_{mod}",
                "task": "acr_cls", "metric": "AUC",
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)) if len(vals) > 1 else float("nan"),
                "s0": split_vals.get(0), "s1": split_vals.get(1),
                "s2": split_vals.get(2), "s3": split_vals.get(3),
                "s4": split_vals.get(4),
            }
            rows.append(row)

df_new = pd.DataFrame(rows)

# Load existing CSV, drop old unimodal rows, append fresh ones
df_old = pd.read_csv(OUT_CSV)
df_old = df_old[~df_old["model"].str.startswith("unimodal_")]
df_out = pd.concat([df_old, df_new], ignore_index=True)
df_out.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(df_new)} unimodal rows → {OUT_CSV}")
