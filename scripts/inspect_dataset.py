"""
inspect_dataset.py — Full dataset inspection from .pt files only.
Produces a JSON summary of modality availability, outcome distributions,
patch/cell count shapes, and sample statistics.
"""
import os, sys, json, collections
import torch
import numpy as np
from pathlib import Path

DATA_DIR = Path("/lustre/groups/aih/dinesh.haridoss/mil/dataset_cache_latest_fixed_large/samples")
OUT_DIR  = Path("/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return None


MODALITIES = [
    "BAL_cells", "BAL_centroids", "BAL_counts",
    "HE_cells",  "HE_centroids",  "HE_counts",
    "CT_cells",  "CT_centroids",  "CT_counts", "CT_radiomics",
    "Clinical",
]

COUNT_MODS = ["BAL_counts", "HE_counts", "CT_counts"]
PATCH_MODS = ["BAL_cells", "HE_cells", "CT_cells"]

# ── main loop ─────────────────────────────────────────────────────────────────
pt_files = sorted(p for p in DATA_DIR.iterdir()
                  if p.suffix == ".pt" and "_" not in p.stem)

print(f"Found {len(pt_files)} main .pt files", flush=True)

# per-sample records for summary
records = []

mod_n        = collections.Counter()          # how many samples have each mod
label_dist   = collections.Counter()          # ACR label distribution
patients     = collections.defaultdict(list)  # patient_id → [anchor_times]
clad_events  = 0;  clad_censored  = 0
death_events = 0;  death_censored = 0

# track patch counts (for MIL context)
patch_counts = {m: [] for m in PATCH_MODS}
count_dims   = {}                             # mod → dim (from first sample)

for i, pt in enumerate(pt_files):
    s = safe_load(pt)
    if s is None:
        continue

    inp  = s.get("inputs", {})
    surv = s.get("survival", {})
    label = s.get("label", -1)
    pid   = s.get("identifier", pt.stem)
    atime = str(s.get("anchor_time", ""))

    label_dist[int(label)] += 1
    patients[pid].append(atime)

    # survival
    for key, ev_cnt, cen_cnt in [
        ("CLAD",  "clad_events",  "clad_censored"),
        ("Death", "death_events", "death_censored"),
    ]:
        val = surv.get(key, "")
        status = None
        if isinstance(val, dict):
            status = val.get("status")
        elif isinstance(val, str) and "status" in val:
            # parse "{'status': 1.0, ...}"  safely
            try:
                import re
                m = re.search(r"'status':\s*([0-9.]+)", val)
                if m: status = float(m.group(1))
            except Exception:
                pass
        if status == 1.0:
            if key == "CLAD":  clad_events  += 1
            else:              death_events += 1
        else:
            if key == "CLAD":  clad_censored  += 1
            else:              death_censored += 1

    # modality availability
    avail = {}
    for mod in MODALITIES:
        t = inp.get(mod)
        has = (t is not None and isinstance(t, torch.Tensor) and t.numel() > 0)
        avail[mod] = has
        if has:
            mod_n[mod] += 1
            # record dims for count mods (first time only)
            if mod in COUNT_MODS and mod not in count_dims:
                count_dims[mod] = t.shape[-1]
            if mod in PATCH_MODS:
                patch_counts[mod].append(t.shape[0])

    rec = {"stem": pt.stem, "patient_id": pid, "anchor_time": atime,
           "label": int(label)}
    rec.update({f"has_{m}": avail[m] for m in MODALITIES})
    records.append(rec)

    if (i + 1) % 500 == 0:
        print(f"  processed {i+1}/{len(pt_files)}", flush=True)

# ── summary ───────────────────────────────────────────────────────────────────
N = len(records)
n_patients = len(patients)
n_serial   = sum(1 for v in patients.values() if len(v) > 1)  # patients with >1 sample

summary = {
    "n_samples":   N,
    "n_patients":  n_patients,
    "n_with_serial_samples": n_serial,
    "label_distribution_acr": dict(sorted(label_dist.items())),
    "survival": {
        "CLAD_events":    clad_events,
        "CLAD_censored":  clad_censored,
        "Death_events":   death_events,
        "Death_censored": death_censored,
    },
    "modality_availability": {
        mod: {"n": mod_n[mod], "pct": round(100 * mod_n[mod] / N, 1)}
        for mod in MODALITIES
    },
    "count_feature_dims": count_dims,
    "patch_counts": {
        mod: {
            "n_samples": len(patch_counts[mod]),
            "mean":  round(float(np.mean(patch_counts[mod])), 1) if patch_counts[mod] else 0,
            "median": round(float(np.median(patch_counts[mod])), 1) if patch_counts[mod] else 0,
            "min":   int(min(patch_counts[mod])) if patch_counts[mod] else 0,
            "max":   int(max(patch_counts[mod])) if patch_counts[mod] else 0,
        }
        for mod in PATCH_MODS
    },
}

# co-availability matrix (fraction of samples with both mods present)
key_mods = ["Clinical", "HE_cells", "CT_cells", "BAL_cells"]
co_avail = {}
for a in key_mods:
    for b in key_mods:
        n_both = sum(1 for r in records if r[f"has_{a}"] and r[f"has_{b}"])
        co_avail[f"{a}+{b}"] = {"n": n_both, "pct": round(100 * n_both / N, 1)}
summary["co_availability"] = co_avail

# samples per patient distribution
spp = [len(v) for v in patients.values()]
summary["samples_per_patient"] = {
    "mean":   round(float(np.mean(spp)), 2),
    "median": float(np.median(spp)),
    "max":    int(max(spp)),
    "distribution": dict(sorted(collections.Counter(spp).items())),
}

# save JSON
out_json = OUT_DIR / "dataset_inspection.json"
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary → {out_json}")

# save per-sample table (CSV)
import csv
csv_path = OUT_DIR / "sample_table.csv"
if records:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
print(f"Saved sample table → {csv_path}")

# ── print to stdout ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"DATASET INSPECTION SUMMARY")
print("="*60)
print(f"Total samples : {N}")
print(f"Total patients: {n_patients}  (serial: {n_serial})")
print(f"\nACR label dist: {dict(sorted(label_dist.items()))}")
print(f"  (0=No ACR, 1=ACR A0 triggered, 2+=higher grade)")
print(f"\nSurvival events:")
print(f"  CLAD  events={clad_events}  censored={clad_censored}")
print(f"  Death events={death_events}  censored={death_censored}")
print(f"\nModality availability:")
for mod in MODALITIES:
    n = mod_n[mod]; pct = 100 * n / N
    bar = "█" * int(pct / 5)
    print(f"  {mod:20s}: {n:4d}/{N} ({pct:4.1f}%) {bar}")
print(f"\nCount feature dims: {count_dims}")
print(f"\nPatch counts per modality:")
for mod, stats in summary["patch_counts"].items():
    print(f"  {mod}: n={stats['n_samples']}  mean={stats['mean']}  "
          f"median={stats['median']}  min={stats['min']}  max={stats['max']}")
print(f"\nSamples per patient: mean={summary['samples_per_patient']['mean']}  "
      f"max={summary['samples_per_patient']['max']}")
print(f"\nCo-availability (key modalities):")
for pair, v in co_avail.items():
    print(f"  {pair:40s}: {v['n']:4d} ({v['pct']}%)")
print("="*60)
print("Done.", flush=True)
