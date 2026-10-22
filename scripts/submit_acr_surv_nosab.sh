#!/usr/bin/env bash
# LongMIL acr_surv training — corrected TTE labels, all 5 splits.
#
# Training: patient-level Cox (default biopsy_batch_size=0).
# Each patient's biopsies are forwarded together; hazards accumulate
# across grad_accum patients before a single Cox backward+step.
#
# TTE labels from compute_tte_first_acr_episode (corrected acr_grade in
# splits CSV — only annotation-confirmed grades used for episode detection).
# Post-episode biopsies (tte=nan) are skipped by the model forward.
#
# HP selection: --p2-all-folds sweeps 4 inner folds,
# then --combined-train trains final model on train+val.
set -euo pipefail

HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="${HOME_MIL}/data/multimodal_splits_nested_cv.csv"
OUT_DIR="${HOME_MIL}/results/mm_abmil_v8_nosab"
LOG_DIR="${OUT_DIR}/slurm_logs"
SCRIPTS_DIR="${OUT_DIR}/job_scripts"
MAIL="dinesh.haridoss@helmholtz-munich.de"

mkdir -p "${LOG_DIR}" "${SCRIPTS_DIR}"

for SPLIT in 0 1 2 3 4; do

JOB="${SCRIPTS_DIR}/acr_surv_s${SPLIT}.sh"
cat > "${JOB}" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=acr_surv_s${SPLIT}
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=220G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=${LOG_DIR}/%j_acr_surv_s${SPLIT}.out
#SBATCH --error=${LOG_DIR}/%j_acr_surv_s${SPLIT}.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL}

set -uo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_PATH="\$(realpath "\$0")"
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

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "=============================="
echo " acr_surv  split=${SPLIT}  variant=longitudinal_mk_no_alibi_v2 (nosab)"
echo " patient-level Cox (biopsy_batch=0)"
echo " host=\$(hostname)  gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo " started: \$(date)"
echo "=============================="

python3 -u ${HOME_MIL}/train_mm_abmil_v8.py \
    --samples-dir    "${SAMPLES}" \
    --splits-csv     "${SPLITS_CSV}" \
    --split          ${SPLIT} \
    --fold           0 \
    --out-dir        "${OUT_DIR}" \
    --phase          p2 \
    --p2-variant     longitudinal_mk_no_alibi_v2 \
    --slot-k         8 \
    --n-cross-layers 0 \
    --task           acr_surv \
    --p2-hp-sweep \
    --p2-all-folds \
    --combined-train \
    --modal-dropout  0.3 \
    --workers        8 &
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
echo " DONE  split=${SPLIT}  exit=\${EXIT_CODE}  \$(date)"
echo "=============================="
exit \${EXIT_CODE}
HEREDOC
chmod +x "${JOB}"
JID=$(sbatch --parsable "${JOB}")
echo "split${SPLIT}  →  job ${JID}"

done

echo ""
echo "All 5 acr_surv jobs submitted. Monitor: squeue -u \$USER"
