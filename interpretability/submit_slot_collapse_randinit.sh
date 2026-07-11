#!/usr/bin/env bash
#SBATCH --job-name=slot_collapse_init
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=80G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=01:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_slot_collapse_init.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_slot_collapse_init.err
#SBATCH --mail-type=NONE

set -uo pipefail
export PYTHONUNBUFFERED=1
HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u "${HOME_MIL}/interpretability/slot_collapse_check.py" \
    --split 1 --fold 0 \
    --no-checkpoint \
    --split-set test \
    --n-patients 40 \
    --mods HE BAL CT Clinical \
    --task acr_cls \
    --max-patches 150

echo "Done."
