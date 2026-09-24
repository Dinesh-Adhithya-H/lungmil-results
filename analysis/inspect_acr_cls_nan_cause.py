"""
Check what causes NaN risk in acr_cls biopsy_reps.
For NaN-risk biopsies: load their .pt files and check for NaN features.
Also check if it's a TTE/label filtering issue in the dataset builder.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT       = Path(__file__).resolve().parent.parent
REPS_DIR   = ROOT / "results" / "mm_abmil_v8" / "biopsy_reps"
SAMPLES_DIR = Path("/lustre/groups/aih/dinesh.haridoss/mil/samples")

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "interpretability"))

# Load split0 reps
p = REPS_DIR / "biopsy_reps_acr_cls_split0.pt"
d = torch.load(p, map_location="cpu", weights_only=False)
stems  = list(d["stems"])
risk   = d["risk"].numpy()
labels = d["label"].numpy()

nan_stems  = [s for s, r in zip(stems, risk) if np.isnan(r)]
good_stems = [s for s, r in zip(stems, risk) if not np.isnan(r)]
print(f"Split0: {len(stems)} biopsies, {len(nan_stems)} NaN risk, {len(good_stems)} valid")

# Check .pt files for NaN features
def check_pt(stem):
    pt = SAMPLES_DIR / f"{stem}.pt"
    if not pt.exists():
        return {"exists": False}
    data = torch.load(pt, map_location="cpu", weights_only=False)
    inp = data.get("inputs", data)
    result = {"exists": True}
    for mod in ["HE", "BAL", "CT"]:
        feat = inp.get(f"{mod}_cells")
        if feat is not None:
            result[f"{mod}_shape"] = tuple(feat.shape)
            result[f"{mod}_nan"]   = int(torch.isnan(feat).sum().item())
        else:
            result[f"{mod}_shape"] = None
    clin = data.get("clinical_onehot")
    if clin is not None:
        result["clin_shape"] = tuple(clin.shape)
        result["clin_nan"]   = int(torch.isnan(clin).sum().item())
    else:
        result["clin_shape"] = None
    return result

print("\n--- Checking 10 NaN-risk stems ---")
nan_clin_nan  = 0
nan_feat_nan  = 0
nan_missing   = 0
for stem in nan_stems[:10]:
    r = check_pt(stem)
    if not r["exists"]:
        print(f"  {stem}: FILE MISSING")
        nan_missing += 1
        continue
    clin_nan = r.get("clin_nan", "N/A")
    print(f"  {stem}: clin={r.get('clin_shape')} nan={clin_nan} | "
          f"HE={r.get('HE_shape')} BAL={r.get('BAL_shape')} CT={r.get('CT_shape')}")

print("\n--- Checking 10 valid-risk stems ---")
for stem in good_stems[:10]:
    r = check_pt(stem)
    if not r["exists"]:
        print(f"  {stem}: FILE MISSING")
        continue
    clin_nan = r.get("clin_nan", "N/A")
    print(f"  {stem}: clin={r.get('clin_shape')} nan={clin_nan} | "
          f"HE={r.get('HE_shape')} BAL={r.get('BAL_shape')} CT={r.get('CT_shape')}")

# Full scan: count NaN clinical features across all NaN-risk stems in split0
print(f"\n--- Full scan of all {len(nan_stems)} NaN-risk stems in split0 ---")
clin_nan_count  = 0
clin_missing    = 0
any_feat_nan    = 0
file_missing    = 0
for stem in nan_stems:
    r = check_pt(stem)
    if not r["exists"]:
        file_missing += 1
        continue
    if r.get("clin_shape") is None:
        clin_missing += 1
    elif r.get("clin_nan", 0) > 0:
        clin_nan_count += 1
    # check any modality feat nan
    if any(r.get(f"{m}_nan", 0) > 0 for m in ["HE","BAL","CT"] if r.get(f"{m}_shape")):
        any_feat_nan += 1

print(f"  file missing:       {file_missing}")
print(f"  clin_onehot None:   {clin_missing}")
print(f"  clin_onehot NaN>0:  {clin_nan_count}")
print(f"  any feat NaN>0:     {any_feat_nan}")
print(f"  no NaN issues:      {len(nan_stems) - file_missing - clin_missing - clin_nan_count - any_feat_nan}")

# Now check: does acr_cls dataset builder exclude biopsies by TTE?
print("\n--- Checking build_splits_longitudinal for acr_cls (split0) ---")
from shared import SPLITS_CSV, SAMPLES_DIR as SD
from mil.data.splits import build_splits_longitudinal

longi_splits = build_splits_longitudinal(SD, SPLITS_CSV, fold=0, split=0)
test_patients = longi_splits.get("test", [])
print(f"  test patients: {len(test_patients)}")

# Count total biopsies and those with label in dataset
total_bio = sum(len(p["stems"]) for p in test_patients)
labeled_bio = 0
tte_nan_bio = 0
for pat in test_patients:
    for rec in pat["records"]:
        lbl = rec.get("label")
        tte = rec.get("tte_next_acr", float("nan"))
        if lbl is not None and not (isinstance(lbl, float) and np.isnan(lbl)):
            labeled_bio += 1
        if isinstance(tte, float) and np.isnan(tte):
            tte_nan_bio += 1
        elif tte is None:
            tte_nan_bio += 1

print(f"  total biopsies in test split0: {total_bio}")
print(f"  biopsies with label:           {labeled_bio}")
print(f"  biopsies with NaN/None tte:    {tte_nan_bio}")
print(f"  (biopsy_reps has {len(stems)} entries, {len(nan_stems)} NaN risk)")
