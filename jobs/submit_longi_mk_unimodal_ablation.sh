#!/usr/bin/env bash
#SBATCH --job-name=lmk_uni_abl
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_lmk_uni_abl.out
#SBATCH --error=/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_lmk_uni_abl.err

source ~/.bashrc
conda activate chicago

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/aih/dinesh.haridoss/chicago_mil/src:${PYTHONPATH:-}"

cd /home/aih/dinesh.haridoss/chicago_mil

echo "=== lmk_uni_abl job=${SLURM_JOB_ID} $(date) ==="
python -u scripts/compute_longi_mk_unimodal_ablation.py --splits 0 1 2 3 4
echo "=== DONE $(date) ==="
