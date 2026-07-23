#!/usr/bin/env bash
#SBATCH --job-name=extract_paper_json
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=00:45:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%x_%j.err

# Loads results_raw.npy from one variant dir and writes paper_interp_data.json.
# No GPU needed. Run after interp_smt_allsplits jobs finish.
# Usage: sbatch interpretability/submit_extract_paper_json.sh --variant cls

set -euo pipefail
REPO=/ictstr01/home/aih/dinesh.haridoss/chicago_mil
cd "${REPO}"
export PYTHONUNBUFFERED=1

VARIANT="cls"  # default; override with --variant flag below

# Parse --variant from args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        *) shift ;;
    esac
done

NPY="${REPO}/interpretability/set_mil_mt_interp/all_splits_${VARIANT}/results_raw.npy"
if [[ -f "${NPY}" ]]; then
    echo "=== Extracting JSON for variant=${VARIANT} ==="
    conda run -n chicago python interpretability/interpret_set_mil_mt.py \
        --json-only \
        --variant "${VARIANT}" \
        --wandb-project none
    echo "=== Done: ${VARIANT} $(date) ==="
else
    echo "ERROR: ${NPY} not found"
    exit 1
fi
