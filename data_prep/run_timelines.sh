#!/usr/bin/env bash
#SBATCH --job-name=viz_timelines
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/patient_plots/slurm_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/patient_plots/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
mkdir -p /lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/patient_plots
echo "==== Patient Timelines $(date) ===="
python3 -u /home/aih/dinesh.haridoss/chicago_mil/data_prep/visualize_patient_timelines.py \
    --csv     /home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv \
    --out_dir /lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/patient_plots
echo "==== Done $(date) ===="
