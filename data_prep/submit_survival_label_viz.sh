#!/usr/bin/env bash
#SBATCH --job-name=surv_label_viz
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/visualizations/survival_label_viz/slurm_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/visualizations/survival_label_viz/slurm_%j.err
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
mkdir -p /home/aih/dinesh.haridoss/chicago_mil/visualizations/survival_label_viz
echo "==== survival label visualizations | $(date) ===="
python3 -u /home/aih/dinesh.haridoss/chicago_mil/data_prep/visualize_survival_labels.py \
    --csv      /home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv \
    --outdir   /home/aih/dinesh.haridoss/chicago_mil/visualizations/survival_label_viz \
    --max_patients   50 \
    --max_table_rows 120
echo "==== Done | $(date) ===="
