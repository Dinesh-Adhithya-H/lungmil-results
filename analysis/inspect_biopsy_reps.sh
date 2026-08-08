#!/bin/bash
#SBATCH --job-name=inspect_biopsy
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/inspect_biopsy_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/inspect_biopsy_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=1 --mem=8G
#SBATCH --time=00:05:00

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 - <<'EOF'
import torch
from pathlib import Path

p = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/biopsy_reps/biopsy_reps_acr_surv_split0.pt")
d = torch.load(p, map_location="cpu", weights_only=False)

print("Keys:", list(d.keys()))
print("reps shape:", d["reps"].shape)
print("risk:", d["risk"][:8].tolist())
print("tte:", d["tte"][:8].tolist())
print("event:", d["event"][:8].tolist())
print("biopsy_days:", d["biopsy_days"][:8].tolist())
print("patient_ids:", d["patient_ids"][:5])
print("stems:", d["stems"][:5])

# Check if TTE is per-biopsy (time from biopsy to event) or per-patient (total TTE)
import numpy as np
pids = d["patient_ids"]
tte  = d["tte"].numpy()
bdays = d["biopsy_days"].numpy()

print("\n--- Same patient, multiple biopsies ---")
from collections import defaultdict
pid_idx = defaultdict(list)
for i, pid in enumerate(pids):
    pid_idx[pid].append(i)

for pid, idxs in list(pid_idx.items())[:3]:
    if len(idxs) > 1:
        print(f"  {pid}: biopsy_days={bdays[idxs].tolist()}, tte={tte[idxs].tolist()}, risk={d['risk'].numpy()[idxs].tolist()}")
EOF
