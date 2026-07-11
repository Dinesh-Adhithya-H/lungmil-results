#!/usr/bin/env bash
# inspect_pt_clusters.sh — inspect cluster fields in .pt files via sbatch
#SBATCH --job-name=inspect_pt
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_inspect_pt.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_inspect_pt.err
#SBATCH --mail-type=NONE

set -uo pipefail
export PYTHONUNBUFFERED=1

SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 - <<'PYEOF'
import torch, os, numpy as np
from pathlib import Path
from collections import defaultdict

SAMPLES = "/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
MOD_KEYS = {
    "HE":       "HE_cells",
    "BAL":      "BAL_cells",
    "CT":       "CT_cells",
    "Clinical": "clinical_onehot",
}

files = sorted(Path(SAMPLES).glob("*.pt"))
print(f"Total .pt files: {len(files)}")

# Inspect first 200 files to understand cluster structure
n_clusters_seen  = defaultdict(set)   # mod -> set of n_clusters values
has_iid          = defaultdict(int)   # mod -> count of files with instance_cluster_ids
has_count_onehot = defaultdict(int)
has_inputs       = defaultdict(int)

for f in files[:200]:
    d = torch.load(f, map_location="cpu", weights_only=False)
    iid  = d.get("instance_cluster_ids", {})
    coh  = d.get("cluster_count_onehot", {})
    inp  = d.get("inputs", {})

    for mod, key in MOD_KEYS.items():
        v = iid.get(key)
        if v is not None and (isinstance(v, torch.Tensor) and v.numel() > 0):
            has_iid[mod] += 1
            nc = int(v.max().item()) + 1
            n_clusters_seen[mod].add(nc)

        c = coh.get(key)
        if c is not None and isinstance(c, torch.Tensor) and c.numel() > 0:
            has_count_onehot[mod] += 1

        i = inp.get(key)
        if i is not None and isinstance(i, torch.Tensor) and i.numel() > 0:
            has_inputs[mod] += 1

print("\n=== Modality field availability (first 200 files) ===")
print(f"  {'Modality':<12} {'has_inputs':>12} {'has_iid':>10} {'has_count_onehot':>18} {'n_clusters_seen'}")
for mod in MOD_KEYS:
    print(f"  {mod:<12} {has_inputs[mod]:>12} {has_iid[mod]:>10} {has_count_onehot[mod]:>18} {sorted(n_clusters_seen.get(mod, set()))}")

# Deep-inspect one file per modality that has cluster data
print("\n=== Detailed field shapes for one example per modality ===")
for mod, key in MOD_KEYS.items():
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        iid = d.get("instance_cluster_ids", {}).get(key)
        inp = d.get("inputs", {}).get(key)
        cco = d.get("cluster_count_onehot", {}).get(key)
        ccv = d.get("cluster_count_vocab", {}).get(key)
        if iid is not None and isinstance(iid, torch.Tensor) and iid.numel() > 0:
            print(f"\n  {mod} ({f.name}):")
            print(f"    inputs:                {getattr(inp, 'shape', None)}")
            print(f"    instance_cluster_ids:  {getattr(iid, 'shape', None)}  "
                  f"unique={iid.unique().numel()}  max={iid.max().item()}")
            print(f"    cluster_count_onehot:  {getattr(cco, 'shape', None)}")
            if ccv is not None:
                print(f"    cluster_count_vocab:   {type(ccv).__name__}  len={len(ccv) if hasattr(ccv,'__len__') else '?'}")
                if isinstance(ccv, list) and len(ccv) > 0:
                    print(f"      first entry: {ccv[0]}")
            break
PYEOF
