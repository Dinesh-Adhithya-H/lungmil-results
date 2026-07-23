#!/usr/bin/env bash
#SBATCH --job-name=check_npy_fields
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/ictstr01/home/aih/dinesh.haridoss/chicago_mil
cd "${REPO}"

conda run -n chicago python - <<'EOF'
import numpy as np
from pathlib import Path

# Check merged npy for panel H required fields
npy = Path("interpretability/set_mil_mt_interp/all_splits_merged/results_raw.npy")
data = np.load(npy, allow_pickle=True)
r = data[0]
print("=== Fields in merged results_raw.npy (first record) ===")
for k, v in r.items():
    if isinstance(v, dict):
        print(f"  {k}: dict with keys {list(v.keys())[:6]}")
    elif hasattr(v, 'shape'):
        print(f"  {k}: array {v.shape}")
    else:
        print(f"  {k}: {type(v).__name__} = {str(v)[:60]}")

print()
panel_h_fields = ['inst_reps', 'pma_attn', 'pma_bcos', 'cluster_ids', 'cluster_names']
print("=== Panel H required fields ===")
for f in panel_h_fields:
    v = r.get(f)
    if v is None:
        print(f"  {f}: MISSING")
    elif isinstance(v, dict):
        present = {k: getattr(vv,'shape',type(vv).__name__) for k,vv in v.items() if vv is not None}
        print(f"  {f}: present — {present}")
    else:
        print(f"  {f}: {type(v).__name__}")

# Check single-task npy for comparison
print()
npy2 = Path("interpretability/set_mil_mt_interp/all_splits_cls/results_raw.npy")
if npy2.exists():
    r2 = np.load(npy2, allow_pickle=True)[0]
    print("=== Panel H fields in single-task cls npy ===")
    for f in panel_h_fields:
        v = r2.get(f)
        if v is None:
            print(f"  {f}: MISSING")
        elif isinstance(v, dict):
            present = {k: getattr(vv,'shape',type(vv).__name__) for k,vv in v.items() if vv is not None}
            print(f"  {f}: present — {present}")
        else:
            print(f"  {f}: {type(v).__name__}")
EOF
