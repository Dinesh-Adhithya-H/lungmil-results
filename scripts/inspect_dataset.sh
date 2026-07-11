#!/usr/bin/env bash
# inspect_dataset.sh — Fast .pt dataset scan (CPU, ~5 min)
#
#SBATCH --job-name=inspect_data
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs/%j_inspect.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs/%j_inspect.err
#SBATCH --mail-type=NONE

set -uo pipefail
export PYTHONUNBUFFERED=1

HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
mkdir -p "${HOME_MIL}/analysis/nature_paper/logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "Host:   $(hostname)"
echo "Date:   $(date)"
echo "Python: $(which python3)"

python3 -u "${HOME_MIL}/scripts/inspect_dataset.py"

echo "Done: $(date)"
