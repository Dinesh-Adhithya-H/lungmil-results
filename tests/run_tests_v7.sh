#!/usr/bin/env bash
#SBATCH --job-name=test_v7
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/tests/test_v7_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/tests/test_v7_%j.err
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
echo "==== v7 test suite $(date) ===="
cd /home/aih/dinesh.haridoss/chicago_mil
pytest -xvs tests/test_v7.py 2>&1
echo "==== Done $(date) ===="
