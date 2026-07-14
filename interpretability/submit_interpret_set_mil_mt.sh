#!/usr/bin/env bash
#SBATCH --job-name=interp_set_mil_mt
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/mm_abmil_v8/slurm_logs/%x_%j.out
#SBATCH --error=results/mm_abmil_v8/slurm_logs/%x_%j.err

# Usage:
#   sbatch interpretability/submit_interpret_set_mil_mt.sh [--split N] [--fold N] [--variant mega]
#
# Defaults: --split 0 --fold 1 --variant mega
# fold=1 is the first HP-sweep fold that has a trained model_set_mil_mt_final.pt.
# When fold0 models are available (after all P2 jobs complete) use --fold 0.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda run -n chicago python "${REPO_ROOT}/interpretability/interpret_set_mil_mt.py" "$@"
