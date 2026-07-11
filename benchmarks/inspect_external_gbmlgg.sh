#!/usr/bin/env bash
#SBATCH --job-name=inspect_gbmlgg_models
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_inspect_gbmlgg.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_inspect_gbmlgg.err

set -e
LUSTRE=/lustre/groups/aih/dinesh.haridoss/mil

echo "================================================================"
echo " MCAT full model files"
echo "================================================================"
find "$LUSTRE/MCAT/models" -name "*.py" | sort | while read f; do
    echo ""
    echo "===== $f ====="
    cat "$f"
done

echo ""
echo "================================================================"
echo " SurvPath full model files"
echo "================================================================"
find "$LUSTRE/SurvPath/models" -name "*.py" | sort | while read f; do
    echo ""
    echo "===== $f ====="
    cat "$f"
done

echo ""
echo "================================================================"
echo " MOTCat model files"
echo "================================================================"
find "$LUSTRE/MOTCat" -name "*.py" 2>/dev/null | sort | head -20
find "$LUSTRE/MOTCat" -name "model*.py" 2>/dev/null | sort | while read f; do
    echo ""
    echo "===== $f ====="
    cat "$f"
done

echo ""
echo "================================================================"
echo " PORPOISE model files"
echo "================================================================"
find "$LUSTRE/PORPOISE" -name "*.py" 2>/dev/null | sort | head -20
find "$LUSTRE/PORPOISE" -name "model*.py" 2>/dev/null | sort | while read f; do
    echo ""
    echo "===== $f ====="
    cat "$f"
done

echo ""
echo "================================================================"
echo " External packages top-level layout"
echo "================================================================"
for pkg in MCAT SurvPath MOTCat PORPOISE; do
    d="$LUSTRE/$pkg"
    if [ -d "$d" ]; then
        echo "--- $pkg ---"
        ls "$d"
    else
        echo "--- $pkg --- NOT FOUND"
    fi
done

echo ""
echo "================================================================"
echo " Sample .pt structure (GBM 00000.pt)"
echo "================================================================"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
python3 - << 'EOF'
import torch, json
s = torch.load('/lustre/groups/aih/dinesh.haridoss/mil/tcga_cache_gbm/samples/00000.pt',
               map_location='cpu', weights_only=False)
print("Top-level keys:", list(s.keys()))
if 'inputs' in s:
    print("inputs keys:")
    for k, v in s['inputs'].items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}  dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")
if 'survival' in s:
    print("survival:", s['survival'])
if 'label' in s:
    print("label:", s['label'])
EOF

echo ""
echo "================================================================"
echo " Sample .pt structure (LGG 00000.pt)"
echo "================================================================"
python3 - << 'EOF'
import torch
s = torch.load('/lustre/groups/aih/dinesh.haridoss/mil/tcga_cache_lgg/samples/00000.pt',
               map_location='cpu', weights_only=False)
print("Top-level keys:", list(s.keys()))
if 'inputs' in s:
    print("inputs keys:")
    for k, v in s['inputs'].items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}  dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")
if 'survival' in s:
    print("survival:", s['survival'])
if 'label' in s:
    print("label:", s['label'])
EOF

echo ""
echo "================================================================"
echo " src/mil/models/phase2.py class signatures"
echo "================================================================"
grep -E "^class |def __init__|def forward" \
    /home/aih/dinesh.haridoss/chicago_mil/src/mil/models/phase2.py | head -80
