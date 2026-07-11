#!/usr/bin/env bash
# Submit one SLURM job per model × fold — all run in parallel.
# Usage:
#   bash submit_spatial_mil.sh                  # all models, all folds
#   bash submit_spatial_mil.sh maskedknn        # one model, all folds
#   bash submit_spatial_mil.sh maskedknn 0 1 2  # one model, specific folds
set -euo pipefail

SCRIPT="/home/aih/dinesh.haridoss/chicago_mil/scripts/train_soft_hier_mil.sh"
OUTDIR="/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/soft_hier_mil"
LOGS="${OUTDIR}/slurm_logs"
MAIL="dinesh.haridoss@helmholtz-munich.de"
mkdir -p "$LOGS"

ALL_MODELS=(hier abmil knnmil maskedknn abmil3 abmil4)
ALL_FOLDS=(0 1 2 3 4)

# Parse args
if [[ $# -ge 1 ]]; then
    MODELS=("$1"); shift
else
    MODELS=("${ALL_MODELS[@]}")
fi
if [[ $# -ge 1 ]]; then
    FOLDS=("$@")
else
    FOLDS=("${ALL_FOLDS[@]}")
fi

JOB_IDS=()

for MODEL in "${MODELS[@]}"; do
    for FOLD in "${FOLDS[@]}"; do
        JID=$(MODEL="$MODEL" FOLD="$FOLD" sbatch \
            --job-name="smil_${MODEL}_f${FOLD}" \
            --partition=gpu_p \
            --qos=gpu_normal \
            --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=120G \
            --gres=gpu:1 --constraint="a100_40gb|a100_80gb|h100_80gb" \
            --time=06:00:00 \
            --output="${LOGS}/${MODEL}_f${FOLD}_%j.out" \
            --error="${LOGS}/${MODEL}_f${FOLD}_%j.err" \
            --mail-type=FAIL \
            --mail-user="$MAIL" \
            --export=ALL,MODEL="$MODEL",FOLD="$FOLD" \
            "$SCRIPT" | awk '{print $NF}')
        echo "  Submitted ${MODEL} fold=${FOLD}  JID=${JID}"
        JOB_IDS+=("$JID")
    done
done

echo ""
echo "All jobs: ${JOB_IDS[*]}"
echo "Monitor: squeue -u \$USER"
echo ""

# Submit collect+plot job with dependency on all model jobs
DEP=$(IFS=:; echo "${JOB_IDS[*]}")
COLLECT_JOB=$(cat <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=smil_collect
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

OUTDIR="/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/soft_hier_mil"

python3 -u << 'PYEOF'
import json, glob
import numpy as np
from pathlib import Path

OUTDIR = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/soft_hier_mil")
ALL_MODELS = [
    ('hier',      'SoftHier (Gaussian)',     '#E53030'),
    ('knnmil',    'KNN-MIL (K=8, 4L)',       '#9B30FF'),
    ('maskedknn', 'Masked KNN-MIL (30%)',    '#FF69B4'),
    ('abmil',     'ABMIL 2-level',           '#4477CC'),
    ('abmil3',    'ABMIL 3-level',           '#22AA44'),
    ('abmil4',    'ABMIL 4-level',           '#FF8800'),
]

print("\n" + "="*60)
print("FINAL COMPARISON (all folds)")
print("="*60)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

results = {}
for tag, name, color in ALL_MODELS:
    folds = []
    for jf in sorted(OUTDIR.glob(f"result_{tag}_fold*.json")):
        folds.append(json.loads(jf.read_text()))
    if not folds:
        continue
    results[tag] = folds
    aucs  = [r['auc']  for r in folds]
    baccs = [r['bacc'] for r in folds]
    print(f"\n  {name}")
    for r in folds:
        print(f"    fold {r['fold']}: AUC={r['auc']:.4f}  BACC={r['bacc']:.4f}")
    print(f"    MEAN AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"    MEAN BACC= {np.mean(baccs):.4f} ± {np.std(baccs):.4f}")

# Bar chart
active = [(t, n, c) for t, n, c in ALL_MODELS if t in results]
n_m = len(active)
if n_m == 0:
    print("No results found."); exit()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
w = 0.7 / n_m
offsets = np.linspace(-(n_m-1)*w/2, (n_m-1)*w/2, n_m)
folds_list = sorted({r['fold'] for res in results.values() for r in res})
x = np.arange(len(folds_list))

for ax_i, metric in enumerate(['auc', 'bacc']):
    ax = axes[ax_i]
    for (tag, name, color), offset in zip(active, offsets):
        vals = {r['fold']: r[metric] for r in results[tag]}
        ys = [vals.get(f, float('nan')) for f in folds_list]
        ax.bar(x + offset, ys, w, label=f'{name} (μ={np.nanmean(ys):.3f})',
               color=color, alpha=0.82, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels([f'F{f}' for f in folds_list])
    ax.set_ylabel('AUC' if metric == 'auc' else 'BAcc')
    ax.set_ylim(0.4, 1.0); ax.set_title(metric.upper())
    ax.legend(fontsize=7); ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0.5, color='gray', ls='--', lw=0.8)

fig.suptitle("Spatial MIL Benchmark — ACR Classification (H&E)", fontsize=11)
plt.tight_layout()
fig.savefig(OUTDIR / "comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved comparison.png")
PYEOF
EOF
)

COLLECT_SCRIPT=$(mktemp /tmp/smil_collect_XXXX.sh)
echo "$COLLECT_JOB" > "$COLLECT_SCRIPT"
chmod +x "$COLLECT_SCRIPT"

COLLECT_JID=$(sbatch --dependency="afterok:${DEP}" \
    --output="${LOGS}/collect_%j.out" \
    --error="${LOGS}/collect_%j.err" \
    --mail-type=END,FAIL \
    --mail-user="$MAIL" \
    "$COLLECT_SCRIPT" | awk '{print $NF}')
rm -f "$COLLECT_SCRIPT"
echo "Collect+plot job: JID=${COLLECT_JID} (runs after all model jobs finish)"
