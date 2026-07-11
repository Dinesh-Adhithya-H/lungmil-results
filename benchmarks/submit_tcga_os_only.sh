#!/usr/bin/env bash
# Submit OS-only single-task TCGA benchmark (to beat MCAT/MOTCAT/SurvPath).
# Uses the same model/code as multitask but trains on OS only per cancer.
#
# Usage:
#   bash benchmarks/submit_tcga_os_only.sh               # all 5 cancers
#   bash benchmarks/submit_tcga_os_only.sh gbmlgg         # one cancer
set -euo pipefail

REPO="/home/aih/dinesh.haridoss/chicago_mil"
LOGS="${REPO}/results_mm_abmil_v8/slurm_logs"
mkdir -p "${LOGS}"

CANCER="${1:-all}"
ALL_CANCERS=(gbmlgg blca kirc luad brca)

if [[ "${CANCER}" == "all" ]]; then
    CANCERS=("${ALL_CANCERS[@]}")
else
    CANCERS=("${CANCER}")
fi

for C in "${CANCERS[@]}"; do
    SCRIPT_PATH="${REPO}/results_mm_abmil_v8/job_scripts/tcga_os_only_${C}.sh"

    cat > "${SCRIPT_PATH}" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=tcga_os_${C}
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=${LOGS}/%j_tcga_os_${C}.out
#SBATCH --error=${LOGS}/%j_tcga_os_${C}.err
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
echo " TCGA OS-only  cancer=${C}  job=\${SLURM_JOB_ID}"
echo " host=\$(hostname)  gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo " started: \$(date)"
echo "=============================="

python3 -u "${REPO}/benchmarks/train_tcga_multitask.py" \
    --cancer ${C} \
    --folds 0 1 2 3 4 \
    --tasks os \
    --save_root "${REPO}/results_tcga_os_only/${C}" \
    --wandb-project chicago-mil-tcga-os &
PY_PID=\$!
wait "\${PY_PID}"; EXIT_CODE=\$?

if [[ "\${RESUBMIT}" -eq 1 ]]; then
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
    echo "  ${C}: job ${JOB_ID}  →  ${LOGS}/${JOB_ID}_tcga_os_${C}.out"
done

echo ""
echo "Monitor: squeue -u \$(whoami) | grep tcga_os"
echo "Results: ls results_tcga_os_only/"
echo "Wandb:   https://wandb.ai/dineshadhithyaharidoss-helmholtz-munich/chicago-mil-tcga-os"
