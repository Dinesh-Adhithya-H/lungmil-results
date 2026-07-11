#!/usr/bin/env bash
#SBATCH --job-name=inspect_ext_models
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_inspect_ext_models.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_inspect_ext_models.err

echo "=== MCAT directory structure ==="
find /lustre/groups/aih/dinesh.haridoss/mil/MCAT/models -name "*.py" | sort
echo ""

echo "=== MCAT model files ==="
for f in /lustre/groups/aih/dinesh.haridoss/mil/MCAT/models/*.py; do
    echo "--- $f ---"
    head -80 "$f"
    echo ""
done

echo "=== SurvPath directory structure ==="
find /lustre/groups/aih/dinesh.haridoss/mil/SurvPath/models -name "*.py" | sort
echo ""

echo "=== SurvPath model files ==="
for f in /lustre/groups/aih/dinesh.haridoss/mil/SurvPath/models/*.py; do
    echo "--- $f ---"
    head -80 "$f"
    echo ""
done

echo "=== MCAT main.py args ==="
grep -E "add_argument|model_type|fusion|modality|path_input|omic" \
    /lustre/groups/aih/dinesh.haridoss/mil/MCAT/main.py | head -40

echo ""
echo "=== SurvPath main.py args ==="
grep -E "add_argument|model_type|fusion|modality|path_input|omic|wsi" \
    /lustre/groups/aih/dinesh.haridoss/mil/SurvPath/main.py | head -40

echo ""
echo "=== GBMLGG splits summary ==="
python3 - << 'EOF'
import pandas as pd
df = pd.read_csv('/home/aih/dinesh.haridoss/chicago_mil/data/tcga_splits/gbmlgg.csv')
print(f"Total: {len(df)}")
print(f"GBM: {(df.cls_label==1).sum()}, LGG: {(df.cls_label==0).sum()}")
print(f"OS events: {df.os_status.sum():.0f} / {df.os_status.notna().sum()}")
print(f"OS time range: {df.os_time.min():.1f} - {df.os_time.max():.1f} months")
for fold in range(5):
    col = f'fold_{fold}'
    print(f"  fold_{fold}: train={( df[col]=='train').sum()} val={(df[col]=='val').sum()} test={(df[col]=='test').sum()}")
EOF
