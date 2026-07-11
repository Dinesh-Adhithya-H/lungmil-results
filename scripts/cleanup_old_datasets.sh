#!/usr/bin/env bash
# cleanup_old_datasets.sh
# Remove old .pt dataset caches after mil_v2 precompute is verified complete.
# Run AFTER benchmarks finish and mil_v2 has been spot-checked.
#
# Keeps: /lustre/groups/aih/dinesh.haridoss/datasets/mil_v2  (the new enriched dataset)
# Removes old caches in /lustre/groups/aih/dinesh.haridoss/mil/:
#   dataset_cache          (~6 GB)
#   dataset_cache_latest   (~127 GB)
#   dataset_cache_latest_fixed (~123 GB)
#   dataset_cache_v2       (if present)
#   dataset_cache_latest_fixed_large  (128 GB pkl — only after benchmarks finish)

set -euo pipefail

MIL_DIR="/lustre/groups/aih/dinesh.haridoss/mil"
MIL_V2="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2"

# --- Sanity checks ---
MIL_V2_COUNT=$(ls "$MIL_V2/samples/" 2>/dev/null | grep -c '\.pt$' || true)
if [[ "$MIL_V2_COUNT" -lt 4000 ]]; then
    echo "ERROR: mil_v2 only has $MIL_V2_COUNT samples — looks incomplete. Aborting."
    exit 1
fi
echo "mil_v2 verified: $MIL_V2_COUNT samples"

# Check that none of the benchmark jobs are running
RUNNING_V8=$(squeue -u "$USER" --name=v8_split0,v8_split1,v8_split2,v8_split3,v8_split4 -h 2>/dev/null | wc -l)
if [[ "$RUNNING_V8" -gt 0 ]]; then
    echo "WARNING: $RUNNING_V8 benchmark v8 jobs are still running."
    echo "Wait for them to finish before removing datasets they may be reading."
    echo "Run this script again after: squeue -u \$USER | grep v8_split"
    read -rp "Continue anyway? [y/N] " _ans
    [[ "${_ans,,}" == "y" ]] || exit 0
fi

OLD_DIRS=(
    "$MIL_DIR/dataset_cache"
    "$MIL_DIR/dataset_cache_latest"
    "$MIL_DIR/dataset_cache_latest_fixed"
    "$MIL_DIR/dataset_cache_v2"
    "$MIL_DIR/dataset_cache_latest_fixed_large"
)

echo ""
echo "Will remove:"
TOTAL=0
for D in "${OLD_DIRS[@]}"; do
    if [[ -d "$D" ]]; then
        SZ=$(du -sh "$D" 2>/dev/null | cut -f1)
        echo "  $D  ($SZ)"
        TOTAL=$((TOTAL + 1))
    fi
done
if [[ "$TOTAL" -eq 0 ]]; then
    echo "  (nothing to remove)"
    exit 0
fi

echo ""
read -rp "Delete all $TOTAL directories? [y/N] " ANS
[[ "${ANS,,}" == "y" ]] || { echo "Aborted."; exit 0; }

for D in "${OLD_DIRS[@]}"; do
    if [[ -d "$D" ]]; then
        echo "Removing $D ..."
        rm -rf "$D"
        echo "  done."
    fi
done

echo ""
echo "Cleanup complete. Remaining space:"
df -h "$MIL_DIR" 2>/dev/null | tail -1
