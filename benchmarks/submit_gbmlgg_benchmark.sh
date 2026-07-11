#!/usr/bin/env bash
# submit_gbmlgg_benchmark.sh  —  submit all 5 folds of GBMLGG benchmark
#
# Usage:
#   bash benchmarks/submit_gbmlgg_benchmark.sh [--methods all] [--cancer gbmlgg]
#
# Each fold runs as an independent job.
# Results land in results_gbmlgg_benchmark/gbmlgg/

set -e

CANCER=${CANCER:-gbmlgg}
METHODS=${METHODS:-all}
LOG_DIR=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs
mkdir -p "$LOG_DIR"

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --cancer)  CANCER="$2";  shift 2 ;;
        --methods) METHODS="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

echo "Submitting GBMLGG benchmark: cancer=$CANCER  methods=$METHODS"

for FOLD in 0 1 2 3 4; do
    JOB=$(sbatch --parsable \
        --job-name="gbmlgg_f${FOLD}" \
        --partition=gpu_p \
        --qos=gpu_normal \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=24:00:00 \
        --output="${LOG_DIR}/%j_gbmlgg_f${FOLD}.out" \
        --error="${LOG_DIR}/%j_gbmlgg_f${FOLD}.err" \
        --wrap="
source \"\$(conda info --base)/etc/profile.d/conda.sh\"
conda activate chicago

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/aih/dinesh.haridoss/chicago_mil/src:\$PYTHONPATH

cd /home/aih/dinesh.haridoss/chicago_mil

echo \"=== GBMLGG Benchmark: cancer=${CANCER}  fold=${FOLD}  methods=${METHODS} ===\"
echo \"Started: \$(date)\"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 benchmarks/train_gbmlgg_benchmark.py \
    --cancer ${CANCER} \
    --fold   ${FOLD} \
    --methods '${METHODS}'

echo \"Finished: \$(date)\"
")
    echo "  fold $FOLD → job $JOB"
done

echo ""
echo "All 5 folds submitted. Monitor with: squeue -u \$USER"
echo "Results: results_gbmlgg_benchmark/${CANCER}/"
