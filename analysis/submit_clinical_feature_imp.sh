#!/bin/bash
#SBATCH --job-name=clinical_feat_imp
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/clinical_feat_imp_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/clinical_feat_imp_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --time=0:15:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil

echo "=== Clinical feature importance plots ==="
python3 analysis/plot_clinical_feature_imp.py
echo "=== DONE ==="
