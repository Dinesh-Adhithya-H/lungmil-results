#!/usr/bin/env bash
#SBATCH --job-name=clust_names
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_clust_names.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_clust_names.err
#SBATCH --mail-type=NONE

set -uo pipefail
export PYTHONUNBUFFERED=1

SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
OUT_JSON="/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/interpretability/cluster_names.json"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 - <<'PYEOF'
import torch, json
from pathlib import Path
from collections import Counter, defaultdict

SAMPLES = "/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
OUT_JSON = "/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/interpretability/cluster_names.json"
MOD_KEYS = {"HE": "HE_cells", "BAL": "BAL_cells", "CT": "CT_cells"}

# Per modality/cluster_id: count how often each name appears
id_name_counts = {mod: defaultdict(Counter) for mod in MOD_KEYS}
files = sorted(Path(SAMPLES).glob("*.pt"))
print(f"Scanning {len(files)} files...")

for i, f in enumerate(files):
    try:
        d = torch.load(f, map_location="cpu", weights_only=False)
        cn  = d.get("cluster_names", {})
        iid = d.get("instance_cluster_ids", {})
        for mod, key in MOD_KEYS.items():
            names = cn.get(key)
            ids   = iid.get(key)
            if names is None or ids is None:
                continue
            if not isinstance(ids, torch.Tensor):
                continue
            for j, cid in enumerate(ids.long().tolist()):
                if j < len(names) and names[j]:
                    id_name_counts[mod][cid][names[j]] += 1
    except Exception as e:
        pass
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(files)} done")

print("Done scanning.")

# Build final mapping: cluster_id -> majority name + purity
result = {}
for mod in MOD_KEYS:
    result[mod] = {}
    for cid in sorted(id_name_counts[mod].keys()):
        counter = id_name_counts[mod][cid]
        top_name, top_count = counter.most_common(1)[0]
        total = sum(counter.values())
        result[mod][str(cid)] = {
            "name":    top_name,
            "purity":  round(top_count / total, 3),
            "n_patches": total,
            "all_names": dict(counter.most_common(5)),
        }
    print(f"\n{mod}: {len(result[mod])} clusters with names")
    for cid, info in list(result[mod].items())[:10]:
        print(f"  {cid:3s}: {info['name']:40s}  purity={info['purity']:.0%}  n={info['n_patches']}")

Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved -> {OUT_JSON}")
PYEOF
