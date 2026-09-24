#!/bin/bash
#SBATCH --job-name=merge_unimod_v2
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_merge_unimod_v2.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_merge_unimod_v2.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
cd /home/aih/dinesh.haridoss/chicago_mil

echo "=== merging unimodal ablation v2 ==="
python interpretability/merge_unimodal_ablation_v2.py

echo "=== plotting unimodal ablation v2 ==="
python analysis/plot_unimodal_ablation_v2.py

echo "=== Done ==="
