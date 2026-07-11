#!/usr/bin/env bash
# Submit TCGA GeoMAE pretraining job
# Run FIRST, before submit_tcga.sh

#SBATCH --job-name="tcga_geomae_pretrain"
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=24:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results/tcga_geomae_pretrain/pretrain.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results/tcga_geomae_pretrain/pretrain.err

mkdir -p /home/aih/dinesh.haridoss/chicago_mil/results/tcga_geomae_pretrain

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python3 -u /home/aih/dinesh.haridoss/chicago_mil/tcga_benchmark/pretrain_tcga.py \
    --cancers KIRC BRCA BLCA LGG GBM \
    --out-dir /home/aih/dinesh.haridoss/chicago_mil/results/tcga_geomae_pretrain \
    --n-epochs 200 \
    --hidden-dim 256 \
    --n-layers 3 \
    --knn-k 8 \
    --mask-ratio 0.5 \
    --max-patches 8000 \
    --grad-accum 4

echo "Done."
