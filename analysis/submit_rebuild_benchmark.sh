#!/usr/bin/env bash
#SBATCH --job-name=rebuild_benchmark
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_rebuild_benchmark.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_rebuild_benchmark.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil
export PYTHONPATH="/home/aih/dinesh.haridoss/chicago_mil/src:${PYTHONPATH:-}"

echo "=== rebuild_benchmark job=${SLURM_JOB_ID} $(date) ==="
python -u analysis/rebuild_benchmark_csvs.py
echo "=== DONE $(date) ==="
