#!/usr/bin/env bash
# Submit TCGA multi-task MIL benchmark jobs.
#
# Usage:
#   bash benchmarks/submit_tcga_multitask.sh               # all cancers, both phases
#   bash benchmarks/submit_tcga_multitask.sh gbmlgg         # one cancer
#   bash benchmarks/submit_tcga_multitask.sh gbmlgg 1       # phase 1 only
#   bash benchmarks/submit_tcga_multitask.sh gbmlgg 2       # phase 2 only (needs splits + P1)
#
# Prerequisites:
#   bash data_prep/submit_make_splits.sh    # creates splits CSVs (wait for completion)
#
# Results:
#   results_tcga_multitask/{cancer}/fold_{0..4}/phase{1,2}/
#   results_tcga_multitask/{cancer}/summary.json
set -euo pipefail

REPO="/home/aih/dinesh.haridoss/chicago_mil"
LOGS="${REPO}/results_mm_abmil_v8/slurm_logs"
mkdir -p "${LOGS}"

CANCER="${1:-all}"
PHASE="${2:-}"

ALL_CANCERS=(gbmlgg blca kirc luad brca)

if [[ "${CANCER}" == "all" ]]; then
    CANCERS=("${ALL_CANCERS[@]}")
else
    CANCERS=("${CANCER}")
fi

PHASE_ARG=""
if [[ -n "${PHASE}" ]]; then
    PHASE_ARG="--phase ${PHASE}"
fi

# Walltime: Phase 1 = 12h (Cox full-batch), Phase 2 = 24h (multi-task)
WALLTIME="24:00:00"
if [[ "${PHASE}" == "1" ]]; then WALLTIME="12:00:00"; fi
if [[ "${PHASE}" == "2" ]]; then WALLTIME="16:00:00"; fi

for C in "${CANCERS[@]}"; do
    SCRIPT_PATH="${REPO}/results_mm_abmil_v8/job_scripts/tcga_multitask_${C}.sh"
    mkdir -p "$(dirname ${SCRIPT_PATH})"

    cat > "${SCRIPT_PATH}" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=tcga_mt_${C}
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=${WALLTIME}
#SBATCH --output=${LOGS}/%j_tcga_mt_${C}.out
#SBATCH --error=${LOGS}/%j_tcga_mt_${C}.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
#SBATCH --signal=B:SIGUSR1@120

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESUBMIT=0
SCRIPT_PATH="\${BASH_SOURCE[0]}"
trap 'RESUBMIT=1; echo "[SIGUSR1] wall-time → resubmitting"; kill \${PY_PID} 2>/dev/null' SIGUSR1

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "=============================="
echo " TCGA multi-task  cancer=${C}  job=\${SLURM_JOB_ID}"
echo " host=\$(hostname)  gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo " started: \$(date)"
echo "=============================="

python3 -u "${REPO}/benchmarks/train_tcga_multitask.py" \\
    --cancer ${C} ${PHASE_ARG} \\
    --folds 0 1 2 3 4 \\
    --save_root "${REPO}/results_tcga_multitask/${C}" &
PY_PID=\$!
wait "\${PY_PID}"; EXIT_CODE=\$?

if [[ "\${RESUBMIT}" -eq 1 ]]; then
    echo "[resubmit] sbatch \${SCRIPT_PATH}"
    sbatch "\${SCRIPT_PATH}"
    exit 0
fi

echo "=============================="
echo " DONE  cancer=${C}  exit=\${EXIT_CODE}  \$(date)"
echo "=============================="
exit \${EXIT_CODE}
HEREDOC

    chmod +x "${SCRIPT_PATH}"
    JOB_ID=$(sbatch --parsable "${SCRIPT_PATH}")
    echo "  ${C}: job ${JOB_ID}  →  ${LOGS}/${JOB_ID}_tcga_mt_${C}.out"
done

echo ""
echo "Monitor: squeue -u \$(whoami) | grep tcga"
echo "Results: ls results_tcga_multitask/"
