#!/usr/bin/env bash
#SBATCH --job-name=interp
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_interp_f%x.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_interp_f%x.err
#SBATCH --mail-type=NONE

set -uo pipefail
export PYTHONUNBUFFERED=1

FOLD="${FOLD:-0}"
HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
OUT_DIR="${HOME_MIL}/results/mm_abmil_v8"

echo "interp  fold=${FOLD}  job=${SLURM_JOB_ID}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 "${HOME_MIL}/scripts/run_interpretability.py" \
    --fold    "${FOLD}"   \
    --out-dir "${OUT_DIR}"
