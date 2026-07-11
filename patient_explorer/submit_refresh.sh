#!/bin/bash
#SBATCH --job-name=explorer_refresh
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/logs/explorer_refresh_%j.out

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago

PIP=/home/aih/dinesh.haridoss/miniconda3/envs/chicago/bin/pip
PYTHON=/home/aih/dinesh.haridoss/miniconda3/envs/chicago/bin/python

echo "=== downgrade plotly to 5.x ==="
$PIP install "plotly>=5.20,<6" --quiet
$PYTHON -c "import plotly; print('plotly', plotly.__version__)"

echo "=== re-export clinical features ==="
cd /home/aih/dinesh.haridoss/chicago_mil/patient_explorer
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv" \
SAMPLES_DIR="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples" \
RESULTS_DIR="/home/aih/dinesh.haridoss/chicago_mil/results/full_data_middle" \
EMBD_DIR="/home/aih/dinesh.haridoss/chicago/plots/phase2_embeddings/fold_0/fusion" \
$PYTHON -c "
import sys; sys.path.insert(0, '.')
from export_data import export_clinical, export_splits
import os, pandas as pd
from pathlib import Path
DATA_DIR = Path('data')
splits_df = pd.read_csv(os.environ['SPLITS_CSV'], parse_dates=['anchor_dt'])
splits_df['stem'] = splits_df['file'].str.replace('.pt','',regex=False).str.zfill(5)
export_clinical(splits_df)
" 2>&1

echo "Done"
