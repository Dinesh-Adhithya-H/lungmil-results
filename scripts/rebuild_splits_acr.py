"""
Rebuild multimodal_splits_nested_cv.csv with corrected ACR labels.

Fixes:
  1. NaN out acr_grade/label/acr_encoded for 966 ghost-labelled samples
     (not in annotation file).
  2. Fill in acr_grade from annotation file for 8 samples that were
     annotation-confirmed but had NaN grade in splits CSV.
"""

import pandas as pd
import numpy as np
import re

SPLITS_CSV = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
ANNOT_CSV  = "/home/aih/dinesh.haridoss/chicago/ACR_biopsies_clean_dates_fixed (1).csv"
OUT_CSV    = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
BACKUP_CSV = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv_backup.csv"

splits = pd.read_csv(SPLITS_CSV)
annot  = pd.read_csv(ANNOT_CSV)

splits["anchor_dt"] = pd.to_datetime(splits["anchor_dt"]).dt.date
annot["Biopsy Date"] = pd.to_datetime(annot["Biopsy Date"]).dt.date

# Unique (patient, date) → grade from annotation
# Take the most severe grade if multiple rows (different core levels) per visit
def most_severe(grades):
    """A2 > A1 > A0; within same A level keep first."""
    order = {"A2": 3, "A1": 2, "A0": 1}
    best, best_score = None, 0
    for g in grades:
        if not isinstance(g, str):
            continue
        score = order.get(g[:2], 0)
        if score > best_score:
            best, best_score = g, score
    return best

annot_grade = (
    annot.groupby(["Record id", "Biopsy Date"])["ACR Status/Grade"]
    .apply(lambda s: most_severe(list(s)))
    .reset_index()
    .rename(columns={"Record id": "patient_id", "Biopsy Date": "anchor_dt",
                     "ACR Status/Grade": "annot_grade"})
)
annot_map = {
    (str(r["patient_id"]), r["anchor_dt"]): r["annot_grade"]
    for _, r in annot_grade.iterrows()
}
print(f"Annotation: {len(annot_map)} unique (patient, date) → grade entries")

# Mark annotation-confirmed rows
splits["acr_annotated"] = splits.apply(
    lambda r: (str(r["patient_id"]), r["anchor_dt"]) in annot_map, axis=1
)

# --- Backup ---
splits_orig = pd.read_csv(SPLITS_CSV)
splits_orig.to_csv(BACKUP_CSV, index=False)
print(f"Backed up original → {BACKUP_CSV}")

# --- Fix 1: NaN out ghost labels (not in annotation) ---
mask_ghost = ~splits["acr_annotated"]
n_ghost = (mask_ghost & splits["acr_grade"].notna()).sum()
splits.loc[mask_ghost, "acr_grade"]   = np.nan
splits.loc[mask_ghost, "label"]       = np.nan
splits.loc[mask_ghost, "acr_encoded"] = np.nan
print(f"Fix 1: NaN'd {n_ghost} ghost-labelled samples")

# --- Fix 2: Fill missing grades from annotation for confirmed samples ---
def acr_label(grade_str):
    if grade_str is None or (isinstance(grade_str, float) and np.isnan(grade_str)):
        return None
    g = str(grade_str).strip()
    if not g or g.lower() in ("nan", "none"):
        return None
    if g.startswith("A0"):
        return 0
    if g.startswith("A1") or g.startswith("A2"):
        return 1
    return None

def acr_encoded(grade_str):
    """Same encoding as original: A0→1, A1→2, A2→3, else NaN."""
    if grade_str is None or (isinstance(grade_str, float) and np.isnan(grade_str)):
        return np.nan
    g = str(grade_str).strip()
    if g.startswith("A0"): return 1.0
    if g.startswith("A1"): return 2.0
    if g.startswith("A2"): return 3.0
    return np.nan

n_filled = 0
for idx, row in splits[splits["acr_annotated"] & splits["acr_grade"].isna()].iterrows():
    key = (str(row["patient_id"]), row["anchor_dt"])
    grade = annot_map.get(key)
    if grade is None:
        continue
    splits.at[idx, "acr_grade"]   = grade
    lbl = acr_label(grade)
    splits.at[idx, "label"]       = float(lbl) if lbl is not None else np.nan
    splits.at[idx, "acr_encoded"] = acr_encoded(grade)
    n_filled += 1
    print(f"  Filled: {row['file']}  {row['patient_id']}  {row['anchor_dt']}  → {grade}")

print(f"Fix 2: Filled {n_filled} missing grades from annotation")

# --- Summary ---
print()
acr_df = splits[splits["acr_grade"].notna()]
print(f"After fix:")
print(f"  acr_grade present:   {len(acr_df)} samples")
print(f"  with HE:             {acr_df['has_HE'].sum()}")
print(f"  without HE:          {(~acr_df['has_HE']).sum()}")
print(f"  ACR positive (A1/A2): {(acr_df['label']==1).sum()}")
print(f"  ACR negative (A0):    {(acr_df['label']==0).sum()}")
print(f"  acr_annotated=True:  {splits['acr_annotated'].sum()}")

splits.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}")

# --- Uncertain grade audit ---
print()
print("=== Uncertain / incomplete grades in annotation file ===")
all_grades = annot["ACR Status/Grade"].dropna().astype(str)
uncertain = all_grades[all_grades.str.contains(r'BX|NA\b|B\?|\?', regex=True)]
print(f"Rows with BX/NA/? in grade: {len(uncertain)}")
print(uncertain.value_counts().to_string())

print()
print("=== Uncertain grades in CLEANED splits CSV ===")
acr_grades_clean = splits["acr_grade"].dropna().astype(str)
unc_splits = acr_grades_clean[acr_grades_clean.str.contains(r'BX|NA\b|B\?|\?', regex=True)]
print(f"Samples with uncertain grade: {len(unc_splits)}")
print(unc_splits.value_counts().to_string())
print()
# Show which samples these are
unc_df = splits[splits["acr_grade"].notna() & splits["acr_grade"].astype(str).str.contains(r'BX|NA\b|B\?|\?', regex=True, na=False)]
print(unc_df[['file','patient_id','anchor_dt','acr_grade','has_HE','label']].to_string(index=False))
