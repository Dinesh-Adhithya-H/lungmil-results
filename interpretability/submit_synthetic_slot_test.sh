#!/usr/bin/env bash
# submit_synthetic_slot_test.sh — synthetic cross-modal slot alignment test
# CPU-only: small synthetic dataset, no GPU needed, ~5 min
#
#SBATCH --job-name=syn_slot
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_syn_slot.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_syn_slot.err
#SBATCH --mail-type=NONE

set -uo pipefail
export PYTHONUNBUFFERED=1

HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
LOG_DIR="${HOME_MIL}/results_mm_abmil_v8/slurm_logs"
mkdir -p "${LOG_DIR}"

echo "==============================="
echo " synthetic_slot_test  job=${SLURM_JOB_ID}"
echo " out: ${HOME_MIL}/interpretability/synthetic_slot_test/"
echo "==============================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "Python: $(which python3)"

python3 -u "${HOME_MIL}/interpretability/synthetic_slot_test.py"

echo "Done."
