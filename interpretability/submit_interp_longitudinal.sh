#!/bin/bash
#SBATCH --job-name=interp_long_mk
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_interp_longitudinal.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_interp_longitudinal.err
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --qos=gpu_normal

set -euo pipefail

# ── Parse args ────────────────────────────────────────────────────────────────
SPLIT=0
FOLD=0
TASK="mega"
N_PATIENTS=30
MIN_BIOPSIES=2
GPU=0
WANDB_PROJECT="chicago-mil-interpretability"

while [[ $# -gt 0 ]]; do
    case $1 in
        --split)          SPLIT="$2";          shift 2 ;;
        --fold)           FOLD="$2";           shift 2 ;;
        --task)           TASK="$2";           shift 2 ;;
        --n-patients)     N_PATIENTS="$2";     shift 2 ;;
        --min-biopsies)   MIN_BIOPSIES="$2";   shift 2 ;;
        --gpu)            GPU="$2";            shift 2 ;;
        --wandb-project)  WANDB_PROJECT="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "===  Longitudinal-MK-MT Interpretability  ==="
echo "    split=${SPLIT}  fold=${FOLD}  task=${TASK}  n_patients=${N_PATIENTS}"
echo "    min_biopsies=${MIN_BIOPSIES}  gpu=${GPU}"
echo "    job_id=${SLURM_JOB_ID}  node=$(hostname)"
echo "=============================================="

# ── Environment ───────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

REPO=/ictstr01/home/aih/dinesh.haridoss/chicago_mil
cd "${REPO}"

mkdir -p results/mm_abmil_v8/slurm_logs

# ── Run ───────────────────────────────────────────────────────────────────────
conda run -n chicago python interpretability/interpret_longitudinal_mk.py \
    --split         "${SPLIT}" \
    --fold          "${FOLD}" \
    --task          "${TASK}" \
    --n-patients    "${N_PATIENTS}" \
    --min-biopsies  "${MIN_BIOPSIES}" \
    --gpu           "${GPU}" \
    --wandb-project "${WANDB_PROJECT}"

echo "=== Done: $(date) ==="
