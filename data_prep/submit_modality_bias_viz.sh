#!/usr/bin/env bash
#SBATCH --job-name=mod_bias_viz
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/visualizations/modality_bias_viz/slurm_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/visualizations/modality_bias_viz/slurm_%j.err
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
mkdir -p /home/aih/dinesh.haridoss/chicago_mil/visualizations/modality_bias_viz
echo "==== modality bias visualizations | $(date) ===="
python3 -u /home/aih/dinesh.haridoss/chicago_mil/data_prep/visualize_modality_bias.py \
    --csv      /home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv \
    --outdir   /home/aih/dinesh.haridoss/chicago_mil/visualizations/modality_bias_viz \
    --max_patients 50
echo "==== Done | $(date) ===="
