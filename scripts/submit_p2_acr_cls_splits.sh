#!/usr/bin/env bash
# Submit set_mil_mt single-task ACR-cls runs for all 5 splits.
# Uses --task cls (maps to ["acr_cls"] only) to test if removing
# survival tasks improves ACR classification BACC.
set -euo pipefail

REPO="/home/aih/dinesh.haridoss/chicago_mil"
LOGS="${REPO}/results_mm_abmil_v8/slurm_logs"
SCRIPTS="${REPO}/results_mm_abmil_v8/job_scripts"
OUT_DIR="${REPO}/results/mm_abmil_v8"

mkdir -p "${LOGS}" "${SCRIPTS}"

for SPLIT in 0 1 2 3 4; do
    SCRIPT="${SCRIPTS}/p2_s${SPLIT}_set_mil_mt_cls.sh"
    cat > "${SCRIPT}" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=p2_s${SPLIT}_smmt_cls
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=220G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=${LOGS}/%j_p2_s${SPLIT}_smmt_cls.out
#SBATCH --error=${LOGS}/%j_p2_s${SPLIT}_smmt_cls.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -uo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPLIT=${SPLIT}
SCRIPT_PATH="\$(realpath "\$0")"

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "=============================="
echo " P2 cls-only  split=\${SPLIT}  variant=set_mil_mt  task=cls  job=\${SLURM_JOB_ID}"
echo " host=\$(hostname)  gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo " started: \$(date)"
echo "=============================="

RESUBMIT=0
PY_PID=""

walltime_handler() {
    echo "[SIGUSR1] Wall-time approaching — resubmitting..."
    RESUBMIT=1
    [[ -n "\${PY_PID}" ]] && kill -SIGTERM "\${PY_PID}" 2>/dev/null || true
    sleep 10
    kill -SIGKILL "\${PY_PID}" 2>/dev/null || true
}
trap walltime_handler SIGUSR1

python3 -u "${REPO}/train_mm_abmil_v8.py" \
    --samples-dir    "/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples" \
    --splits-csv     "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv" \
    --split          \${SPLIT} \
    --out-dir        "${OUT_DIR}" \
    --phase          p2 \
    --p2-variant     set_mil_mt \
    --slot-k         16 \
    --n-cross-layers 1 \
    --task           cls \
    --p2-hp-sweep \
    --p2-all-folds \
    --p2-min-epochs  200 \
    --modal-dropout  0.3 \
    --workers        8 \
    --wandb-project  chicago-mil &
PY_PID=\$!
wait "\${PY_PID}"
EXIT_CODE=\$?
PY_PID=""

if [[ "\${RESUBMIT}" -eq 1 ]]; then
    NEW_JID=\$(sbatch --parsable "\${SCRIPT_PATH}")
    echo "Resubmitted as \${NEW_JID}"
    exit 0
fi
echo "=============================="
echo " DONE  split=\${SPLIT}  exit=\${EXIT_CODE}  \$(date)"
echo "=============================="
exit \${EXIT_CODE}
HEREDOC
    chmod +x "${SCRIPT}"
    JOB_ID=$(sbatch --parsable "${SCRIPT}")
    echo "split${SPLIT}: job ${JOB_ID} → ${LOGS}/%j_p2_s${SPLIT}_smmt_cls.out"
done
