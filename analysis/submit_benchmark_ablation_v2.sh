#!/bin/bash
#SBATCH --job-name=benchmark_ablation_v2
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/logs/benchmark_ablation_v2_%j.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/logs/benchmark_ablation_v2_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G
#SBATCH --time=0:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil

echo "=== 1. Benchmark v2 (fixed order + linear baselines) ==="
python3 analysis/plot_benchmark_v2.py

echo "=== 2. Unimodal ablation v2 (fixed model order, all models) ==="
python3 analysis/plot_unimodal_ablation_v2.py

echo "=== 3. Modality combo ablation (single mods + all, all models) ==="
python3 analysis/plot_modality_combo_ablation.py

echo "=== DONE ==="
