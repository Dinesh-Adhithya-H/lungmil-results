#!/bin/bash
#SBATCH --job-name=explorer_install
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/aih/dinesh.haridoss/logs/explorer_install_%j.out

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago

echo "=== trying conda install ==="
conda install -y -n chicago -c conda-forge streamlit plotly pyarrow 2>&1 | tail -20

echo "=== checking import ==="
python -c "import streamlit; print('streamlit', streamlit.__version__)" 2>&1
python -c "import plotly; print('plotly', plotly.__version__)" 2>&1
