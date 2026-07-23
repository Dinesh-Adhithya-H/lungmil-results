#!/bin/bash
#SBATCH --job-name=rebuild_uni_bench
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/aih/dinesh.haridoss/logs/rebuild_unimodal_bench_%j.out

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago
cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil
python patient_explorer/rebuild_unimodal_benchmark.py
