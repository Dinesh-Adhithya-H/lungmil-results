#!/usr/bin/env bash
# Submit all 4 set_mil_mt interpretability jobs (panel D rewrite: all seeds, raw logits, prediction correlation)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

for variant in cls acr_surv clad_surv death_surv; do
    jid=$(sbatch --parsable \
        --job-name="smmt_v4_${variant}" \
        --partition=gpu_p --qos=gpu_normal --gres=gpu:1 \
        --cpus-per-task=8 --mem=64G --time=02:00:00 \
        --output="results/mm_abmil_v8/slurm_logs/%j_smmt_v4_${variant}.out" \
        --error="results/mm_abmil_v8/slurm_logs/%j_smmt_v4_${variant}.err" \
        --mail-type=END,FAIL \
        --mail-user=dinesh.haridoss@helmholtz-munich.de \
        --wrap="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; conda run -n chicago python interpretability/interpret_set_mil_mt.py --split 0 --fold 0 --variant ${variant}")
    echo "Submitted ${variant}: ${jid}"
done
