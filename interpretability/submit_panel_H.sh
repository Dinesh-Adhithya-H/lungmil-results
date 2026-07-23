#!/usr/bin/env bash
#SBATCH --job-name=interp_panel_H
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.err

# Panel H: instance-cluster → seed affinity heatmap.
# Shows which patch clusters each PMA seed attends to (B-cos mass per cluster).
# Uses the single-task cls npy which contains inst_reps and pma_attn per patient.
# (The merged npy omits these fields because they are too large.)

set -euo pipefail
REPO=/ictstr01/home/aih/dinesh.haridoss/chicago_mil
cd "${REPO}"
export PYTHONUNBUFFERED=1

BASE="${REPO}/interpretability/set_mil_mt_interp"

echo "=== Panel H (instance-cluster→seed): $(date) ==="
conda run -n chicago python interpretability/interpret_set_mil_mt.py \
    --json-only \
    --out-dir "${BASE}/all_splits_cls" \
    --panels H \
    --wandb-project chicago-mil-interpretability
echo "=== Done: $(date) ==="
