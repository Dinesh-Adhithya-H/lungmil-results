"""Check acr_cls model head keys and what hazard_traj produces."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "interpretability"))

from shared import SPLITS_CSV, SAMPLES_DIR, RESULTS_ROOT
from interpret_longitudinal_mk import load_model, load_patient_bags, extract_patient_longitudinal
from mil.data.splits import build_splits_longitudinal

VARIANT = "longitudinal_mk_no_alibi_v2"
TASK_CFG = {
    "acr_cls": {"dir_suffix": "cls", "longi_key": "acr_cls"}
}

device = torch.device("cpu")
cfg = TASK_CFG["acr_cls"]

print("Loading model...")
model, task_names = load_model(split=0, fold=0, device=device,
                                task=cfg["dir_suffix"], variant=VARIANT)
print(f"  task_names: {task_names}")
print(f"  model.heads keys: {list(model.heads.keys()) if hasattr(model, 'heads') else 'NO heads attr'}")
if hasattr(model, 'heads'):
    for k, v in model.heads.items():
        print(f"    heads['{k}']: {v}")

# Run on one patient and inspect hazard_traj
print("\nLoading test patients (split0)...")
longi_splits = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=0)
test_patients = longi_splits.get("test", [])

# Find a patient that has NaN risk in biopsy_reps
REPS_DIR = ROOT / "results" / "mm_abmil_v8" / "biopsy_reps"
d = torch.load(REPS_DIR / "biopsy_reps_acr_cls_split0.pt", map_location="cpu", weights_only=False)
nan_stems = set(s for s, r in zip(d["stems"], d["risk"].numpy()) if np.isnan(r))
good_stems = set(s for s, r in zip(d["stems"], d["risk"].numpy()) if not np.isnan(r))
print(f"  NaN risk stems: {len(nan_stems)}, valid: {len(good_stems)}")

# Find patient with both NaN and valid stems
target_pat = None
for pat in test_patients:
    stems = set(pat["stems"])
    if stems & nan_stems and stems & good_stems:
        target_pat = pat
        break

if target_pat is None:
    # Just use first patient with nan stems
    for pat in test_patients:
        if set(pat["stems"]) & nan_stems:
            target_pat = pat
            break

if target_pat:
    print(f"\nRunning extraction on patient {target_pat['patient_id']} "
          f"({len(target_pat['stems'])} biopsies)...")
    bags_list, _ = load_patient_bags(target_pat, device)
    with torch.no_grad():
        extr = extract_patient_longitudinal(model, target_pat, bags_list, device,
                                            tasks=[cfg["longi_key"]])
    if extr:
        haz_traj = extr["hazard_traj"].get(cfg["longi_key"], [])
        reps     = extr["per_biopsy_reps"].get(cfg["longi_key"], [])
        print(f"  hazard_traj length: {len(haz_traj)}")
        print(f"  per_biopsy_reps length: {len(reps)}")
        for i, (h, stem) in enumerate(zip(haz_traj, target_pat["stems"])):
            in_nan = stem in nan_stems
            in_good = stem in good_stems
            rep = reps[i] if i < len(reps) else None
            rep_has_nan = (rep is not None and torch.isnan(rep).any().item())
            print(f"  [{i}] stem={stem} haz={h:.4f if not (isinstance(h,float) and np.isnan(h)) else 'NaN'} "
                  f"nan_risk={in_nan} good_risk={in_good} rep_nan={rep_has_nan}")
        # Also check: run head directly on a rep
        print(f"\n  logits_out: {extr.get('logits', {})}")
        # Try running head on first valid rep
        for i, rep in enumerate(reps):
            if rep is not None and not torch.isnan(rep).any():
                for key in (cfg["longi_key"], "cls", "acr"):
                    if hasattr(model, "heads") and key in model.heads:
                        out = model.heads[key](rep.unsqueeze(0) if rep.dim()==1 else rep)
                        print(f"  heads['{key}'](rep[{i}]) = {out}")
                break
