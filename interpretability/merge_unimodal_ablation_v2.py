"""
Merge per-split unimodal ablation v2 CSVs into unimodal_ablation_summary.csv.
Removes any existing longitudinal_mk_no_alibi_v2 rows then appends new ones.
Run after submit_eval_unimodal_ablation_v2.sh completes.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
ABL_DIR = ROOT / "interpretability" / "unimodal_ablation"
OUT_CSV = ABL_DIR / "unimodal_ablation_summary.csv"
VARIANT = "longitudinal_mk_no_alibi_v2"

split_files = sorted(ABL_DIR.glob("unimodal_ablation_v2_split*.csv"))
print(f"Found {len(split_files)} split files: {[f.name for f in split_files]}")

if not split_files:
    print("ERROR: no per-split CSV files found — did the array job finish?")
    raise SystemExit(1)

new_rows = pd.concat([pd.read_csv(f) for f in split_files], ignore_index=True)
print(f"  {len(new_rows)} total rows from splits")

# Compute mean/std per (variant, task, modality)
agg = (new_rows.groupby(["variant", "task", "modality", "metric"])["value"]
       .agg(["mean", "std", "count"]).reset_index())
agg["mean_std"] = agg.apply(lambda r: f"{r['mean']:.3f}±{r['std']:.3f}", axis=1)
# Merge back per-split values
merged = new_rows.merge(agg[["variant","task","modality","metric","mean","std","count","mean_std"]],
                        on=["variant","task","modality","metric"], how="left",
                        suffixes=("_raw", ""))
merged = merged.drop(columns=["mean_raw","std_raw","count_raw","mean_std_raw"], errors="ignore")

# Load existing summary, drop old v2 rows, append new
if OUT_CSV.exists():
    existing = pd.read_csv(OUT_CSV)
    existing = existing[existing["variant"] != VARIANT]
    final = pd.concat([existing, merged], ignore_index=True)
else:
    final = merged

final.to_csv(OUT_CSV, index=False)
print(f"Saved {len(final)} total rows → {OUT_CSV}")
print(f"  v2 rows: {(final['variant'] == VARIANT).sum()}")
