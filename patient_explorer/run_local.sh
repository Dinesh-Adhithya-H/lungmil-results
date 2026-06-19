#!/bin/bash
# Launch the app locally (for SSH port-forward access or after data export).
# On the cluster: run this inside an interactive job, then SSH tunnel port 8501.
#
# SSH tunnel from your laptop:
#   ssh -L 8501:localhost:8501 dinesh.haridoss@hpc-submit01.scidom.de

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago

cd "$(dirname "$0")"

# Set data directory (override with env var if needed)
export EXPLORER_DATA="${EXPLORER_DATA:-$(pwd)/data}"

streamlit run app.py \
    --server.port 8501 \
    --server.headless true \
    --server.address 0.0.0.0 \
    2>&1
