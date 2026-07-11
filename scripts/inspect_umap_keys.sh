#!/usr/bin/env bash
#SBATCH --job-name=inspect_umap
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs/%j_umap_keys.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper/logs/%j_umap_keys.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --ntasks=1 --cpus-per-task=2 --mem=32G
#SBATCH --time=00:20:00

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate mil_env

cd /home/aih/dinesh.haridoss/chicago_mil
python scripts/inspect_umap_keys.py
