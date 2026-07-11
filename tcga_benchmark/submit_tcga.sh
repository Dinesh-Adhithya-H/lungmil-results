#!/usr/bin/env bash
# submit_tcga.sh — submit all TCGA benchmark jobs
#
# Usage:
#   bash submit_tcga.sh              # all cancers × all models × 5 folds
#   bash submit_tcga.sh KIRC         # one cancer only
#   bash submit_tcga.sh KIRC abmil   # one cancer, one model

set -euo pipefail

HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
GEOMAE_CKPT="${HOME_MIL}/results/geomae_pretrain/best_backbone.pt"
LOG_DIR="${HOME_MIL}/results/tcga_benchmark/slurm_logs"
mkdir -p "${LOG_DIR}"

CANCERS="${1:-KIRC BRCA BLCA LGG GBM}"
MODELS_ALL="${2:-abmil transmil dsmil slotmil geomae_slotmil}"
N_FOLDS=5

submit_job() {
    local cancer=$1 model=$2 fold=$3
    local job_name="tcga_${cancer}_${model}_f${fold}"
    local extra=""
    [[ "$model" == "geomae_slotmil" ]] && extra="--geomae-ckpt ${GEOMAE_CKPT}"

    sbatch \
        --job-name="${job_name}" \
        --partition=gpu_p \
        --qos=gpu_normal \
        --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=100G \
        --gres=gpu:1 --constraint="a100_80gb|h100_80gb" \
        --time=06:00:00 \
        --mail-type=NONE \
        --output="${LOG_DIR}/%j_${job_name}.out" \
        --error="${LOG_DIR}/%j_${job_name}.err" \
        --wrap="
source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago
echo 'GPU: '\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
python3 -u ${HOME_MIL}/tcga_benchmark/train_tcga.py \
    --cancer ${cancer} \
    --model  ${model}  \
    --fold   ${fold}   \
    --n-epochs 40      \
    ${extra}
"
    echo "  submitted ${job_name}"
}

total=0
for cancer in $CANCERS; do
    for model in $MODELS_ALL; do
        for fold in $(seq 0 $((N_FOLDS-1))); do
            submit_job "$cancer" "$model" "$fold"
            total=$((total+1))
        done
    done
done

echo ""
echo "Submitted ${total} jobs  (${#CANCERS[@]} cancers × ${#MODELS_ALL[@]} models × ${N_FOLDS} folds)"
