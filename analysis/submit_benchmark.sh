#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/benchmark_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/benchmark_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G
#SBATCH --time=0:45:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil

echo "=== 1. Multimodal benchmark ==="
python3 analysis/plot_benchmark.py

echo "=== 2. Unimodal benchmark (ABMIL per modality) ==="
python3 analysis/plot_unimodal_benchmark.py

echo "=== 3. Modality combo (input combos per biopsy) ==="
python3 analysis/plot_modality_combo_clean.py

echo "=== 4. Unimodal ablation (LongMIL vs ABMIL per modality) ==="
python3 analysis/plot_unimodal_ablation_v2.py

echo "=== DONE ==="
