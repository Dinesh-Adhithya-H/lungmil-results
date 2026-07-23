#!/usr/bin/env bash
#SBATCH --job-name=check_trajectory
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=192G
#SBATCH --time=00:30:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.err

set -euo pipefail
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
export PYTHONUNBUFFERED=1

conda run -n chicago python - <<'PYEOF'
import numpy as np
from collections import Counter
from pathlib import Path

npy = "interpretability/set_mil_mt_interp/all_splits_merged/results_raw.npy"
print(f"Loading {npy} ...")
data = list(np.load(npy, allow_pickle=True))
print(f"Total records: {len(data)}")

# Check anchor_dt injection
n_with_adt = sum(1 for r in data if r.get("anchor_dt") is not None)
adt_types = Counter(type(r.get("anchor_dt")).__name__ for r in data)
print(f"anchor_dt present: {n_with_adt}/{len(data)}  types: {dict(adt_types)}")

# First 3 records
print("\n=== First 3 records ===")
for r in data[:3]:
    print(f"  stem={r['stem']}  pid={r.get('patient_id')}  anchor_dt={r.get('anchor_dt')!r}")

# Multi-visit patients
pid_to_stems = {}
for r in data:
    pid = r.get("patient_id", r["stem"])
    pid_to_stems.setdefault(pid, []).append(r["stem"])
multi = {pid: stems for pid, stems in pid_to_stems.items() if len(stems) > 1}
print(f"\nUnique patients: {len(pid_to_stems)}  Multi-visit: {len(multi)}")

# Show top 3 multi-visit patients and their trajectory
top3 = sorted(multi.items(), key=lambda x: -len(x[1]))[:3]
for pid, stems in top3:
    visits = sorted([r for r in data if r.get("patient_id", r["stem"]) == pid],
                    key=lambda x: x.get("anchor_dt") or x["stem"])
    t0 = visits[0].get("anchor_dt")
    print(f"\n  pid={pid}  n_visits={len(stems)}")
    for v in visits:
        adt = v.get("anchor_dt")
        try:
            days = (adt - t0).days if (adt and t0) else "?"
        except Exception as e:
            days = f"ERR:{e}"
        print(f"    stem={v['stem']}  anchor_dt={adt!r}  days_from_t0={days}")

# Check a patient summary PNG exists and is non-zero size
png_dir = Path("interpretability/set_mil_mt_interp/all_splits_merged/patient_summaries")
pngs = list(png_dir.glob("*.png"))
print(f"\nPatient summary PNGs: {len(pngs)}")
if pngs:
    ex = pngs[0]
    print(f"  Example: {ex.name}  size={ex.stat().st_size/1024:.1f}KB")
PYEOF
