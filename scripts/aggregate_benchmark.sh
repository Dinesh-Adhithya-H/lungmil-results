#!/usr/bin/env bash
# aggregate_benchmark.sh — Parse result JSONs → benchmark_summary.csv (CPU, fast)
#
#SBATCH --job-name=agg_bench
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs/%j_agg_bench.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs/%j_agg_bench.err
#SBATCH --mail-type=NONE

set -uo pipefail
export PYTHONUNBUFFERED=1
HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
mkdir -p "${HOME_MIL}/analysis/nature_paper/logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "Host: $(hostname)  Date: $(date)"
python3 -u "${HOME_MIL}/scripts/aggregate_benchmark.py"
echo "Done: $(date)"
