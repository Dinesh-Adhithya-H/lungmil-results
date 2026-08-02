#!/usr/bin/env bash
# Re-run longitudinal_mk_no_alibi interp for splits 0-4 (acr_surv + death_surv)
# to generate results_raw.npy caches needed by plot_unified_rep_umap.py
#SBATCH --job-name=longi_npy
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=80G
#SBATCH --gres=gpu:1 --constraint="a100_40gb|a100_80gb|h100_80gb"
#SBATCH --time=04:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_longi_npy.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_longi_npy.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/aih/dinesh.haridoss/chicago_mil
export PYTHONPATH="/home/aih/dinesh.haridoss/chicago_mil/src:${PYTHONPATH:-}"

echo "=== longi_npy job=${SLURM_JOB_ID} $(date) ==="

for SPLIT in 0 1 2 3 4; do
    for TASK in acr_surv death_surv; do
        echo "--- split=$SPLIT task=$TASK ---"
        python -u interpretability/interpret_longitudinal_mk.py \
            --split $SPLIT --fold 0 \
            --variant longitudinal_mk_no_alibi \
            --task $TASK \
            --n-patients 9999 \
            --wandb-project none
    done
done

echo "=== DONE $(date) ==="
