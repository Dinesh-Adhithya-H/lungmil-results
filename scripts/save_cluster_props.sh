#!/usr/bin/env bash
#SBATCH --job-name=cluster_props
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_cluster_props.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_cluster_props.err
#SBATCH --mail-type=NONE

set -uo pipefail
export PYTHONUNBUFFERED=1

SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
OUT_CSV="/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/cluster_proportions.csv"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 - <<PYEOF
import torch, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os

SAMPLES = "$SAMPLES"
OUT_CSV = "$OUT_CSV"
CLR_EPS = 1e-6

MOD_KEYS = {
    "HE":  "HE_cells",
    "BAL": "BAL_cells",
    "CT":  "CT_cells",
}

# ── Pass 1: determine global n_clusters per modality from first file that has it
print("Pass 1: finding global cluster vocabulary sizes...")
n_clusters = {}
files = sorted(Path(SAMPLES).glob("*.pt"))
print(f"  Total .pt files: {len(files)}")

for f in files:
    if len(n_clusters) == len(MOD_KEYS):
        break
    d = torch.load(f, map_location="cpu", weights_only=False)
    cco = d.get("cluster_count_onehot", {})
    for mod, key in MOD_KEYS.items():
        if mod in n_clusters:
            continue
        v = cco.get(key)
        if v is not None and isinstance(v, torch.Tensor) and v.ndim == 2:
            n_clusters[mod] = v.shape[0]
            print(f"  {mod}: n_clusters={v.shape[0]}")

print(f"  Cluster vocab: {n_clusters}")

# ── Pass 2: compute proportions for every sample
def clr(prop):
    p = prop + CLR_EPS
    lp = np.log(p)
    return lp - lp.mean()

def process_file(f):
    d = torch.load(f, map_location="cpu", weights_only=False)
    stem = f.stem   # unique file stem ("00000") — patient ID duplicates across timepoints
    identifier = d.get("identifier", "")
    anchor_dt  = d.get("anchor_dt", d.get("anchor_time", d.get("date", "")))
    iid  = d.get("instance_cluster_ids", {})
    inp  = d.get("inputs", {})
    row  = {"stem": stem, "patient_id": identifier, "anchor_dt": anchor_dt}

    # Clinical: 1-D (102,) vector directly in inputs["Clinical"]
    clin_t = inp.get("Clinical")
    if clin_t is not None and isinstance(clin_t, torch.Tensor):
        ct = clin_t.float()
        clin_mean = ct.mean(0).numpy() if ct.ndim == 2 else ct.numpy()
    else:
        clin_mean = None

    if clin_mean is not None:
        for k, v in enumerate(clin_mean):
            row[f"Clinical_mean_{k}"] = float(v)

    # Patch modalities: cluster proportions + CLR
    for mod, key in MOD_KEYS.items():
        nc = n_clusters.get(mod, 0)
        ids_t = iid.get(key)

        if ids_t is not None and isinstance(ids_t, torch.Tensor) and ids_t.numel() > 0:
            ids    = ids_t.long().numpy()
            counts = np.bincount(ids, minlength=nc).astype(np.float32)
            prop   = counts / max(counts.sum(), 1)
            prop_c = clr(prop)
        else:
            prop   = np.zeros(nc, dtype=np.float32)
            prop_c = np.zeros(nc, dtype=np.float32)

        for k in range(nc):
            row[f"{mod}_prop_{k}"]  = float(prop[k])
            row[f"{mod}_clr_{k}"]   = float(prop_c[k])

    return row

print(f"\nPass 2: computing proportions for {len(files)} files (8 threads)...")
rows = []
done = 0
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(process_file, f): f for f in files}
    for fut in futs:
        try:
            rows.append(fut.result())
        except Exception as e:
            print(f"  ERROR {futs[fut].name}: {e}")
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(files)} done...")

df = pd.DataFrame(rows)
# sort columns: stem first, then HE_prop_*, HE_clr_*, BAL_*, CT_*
prop_cols = [c for c in df.columns if c != "stem"]
prop_cols.sort()
df = df[["stem"] + prop_cols]

Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(df)} rows x {len(df.columns)} cols → {OUT_CSV}")
print(df.head(3).to_string())
PYEOF
