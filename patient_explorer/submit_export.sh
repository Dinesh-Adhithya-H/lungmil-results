#!/bin/bash
#SBATCH --job-name=explorer_export
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/logs/explorer_export_%j.out

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil/patient_explorer

SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv" \
SAMPLES_DIR="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples" \
RESULTS_DIR="/home/aih/dinesh.haridoss/chicago_mil/results/full_data_middle" \
EMBD_DIR="/home/aih/dinesh.haridoss/chicago/plots/phase2_embeddings/fold_0/fusion" \
python export_data.py 2>&1
