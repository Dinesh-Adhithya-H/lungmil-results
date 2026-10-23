#!/usr/bin/env bash
# submit_comparison_acr.sh
#
# Retrains all comparison baselines for acr_cls and acr_surv with corrected
# annotation-only labels (637 samples, NaN'd ghost labels removed).
#
# Methods per split:
#   early, late, middle — p2-all-folds (all 4 tasks; clad/death skip via status file)
#   set_mil_no_sab      — cls then acr_surv separately (p2-all-folds)
#   set_mil_mt_no_sab   — cls then acr_surv separately (p2-all-folds)
#   set_mil_mt          — cls then acr_surv separately (p2-all-folds)
#
# Output: results/mm_abmil_v8/ (same dir as clad/death which are already correct)
# Time limit: 48h with self-resubmit on wall-time.
set -euo pipefail

HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="${HOME_MIL}/data/multimodal_splits_nested_cv.csv"
OUT_DIR="${HOME_MIL}/results/mm_abmil_v8"
LOG_DIR="${OUT_DIR}/slurm_logs"
SCRIPTS_DIR="${OUT_DIR}/job_scripts"
MAIL="dinesh.haridoss@helmholtz-munich.de"

mkdir -p "${LOG_DIR}" "${SCRIPTS_DIR}"

for SPLIT in 0 1 2 3 4; do

JOB="${SCRIPTS_DIR}/comparison_acr_s${SPLIT}.sh"
cat > "${JOB}" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=cmp_acr_s${SPLIT}
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=220G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=${LOG_DIR}/%j_cmp_acr_s${SPLIT}.out
#SBATCH --error=${LOG_DIR}/%j_cmp_acr_s${SPLIT}.err
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
echo " comparison_acr  split=${SPLIT}"
echo " host=\$(hostname)  gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo " started: \$(date)"
echo "=============================="

run_py() {
    python3 -u ${HOME_MIL}/train_mm_abmil_v8.py "\$@" &
    PY_PID=\$!
    wait "\${PY_PID}"
    EXIT_CODE=\$?
    PY_PID=""
    if [[ "\${RESUBMIT}" -eq 1 ]]; then
        echo "Resubmitting..."
        NEW_JID=\$(sbatch --parsable "\${SCRIPT_PATH}")
        echo "Resubmitted as \${NEW_JID}"
        exit 0
    fi
    return "\${EXIT_CODE}"
}

COMMON="--samples-dir ${SAMPLES} --splits-csv ${SPLITS_CSV} --split ${SPLIT} --fold 0 --out-dir ${OUT_DIR} --phase p2 --p2-hp-sweep --p2-all-folds --combined-train --modal-dropout 0.3 --workers 8"

# ── early / late / middle: all 4 tasks (clad/death skip via status file) ──
for VARIANT in early late middle; do
    echo ""
    echo "=============================="
    echo " variant=\${VARIANT}  split=${SPLIT}  \$(date)"
    echo "=============================="
    run_py \${COMMON} --p2-variant "\${VARIANT}" --slot-k 8 --n-cross-layers 0 --task cls || true
done

# ── set_mil_no_sab: cls then acr_surv ──
for TASK in cls acr_surv; do
    echo ""
    echo "=============================="
    echo " variant=set_mil_no_sab  task=\${TASK}  split=${SPLIT}  \$(date)"
    echo "=============================="
    run_py \${COMMON} --p2-variant set_mil_no_sab --slot-k 128 --n-cross-layers 0 --task "\${TASK}" || true
done

# ── set_mil_mt_no_sab: cls then acr_surv ──
for TASK in cls acr_surv; do
    echo ""
    echo "=============================="
    echo " variant=set_mil_mt_no_sab  task=\${TASK}  split=${SPLIT}  \$(date)"
    echo "=============================="
    run_py \${COMMON} --p2-variant set_mil_mt_no_sab --slot-k 128 --n-cross-layers 0 --task "\${TASK}" || true
done

# ── set_mil_mt: cls then acr_surv ──
for TASK in cls acr_surv; do
    echo ""
    echo "=============================="
    echo " variant=set_mil_mt  task=\${TASK}  split=${SPLIT}  \$(date)"
    echo "=============================="
    run_py \${COMMON} --p2-variant set_mil_mt --slot-k 128 --n-cross-layers 0 --task "\${TASK}" || true
done

echo ""
echo "=============================="
echo " DONE  split=${SPLIT}  \$(date)"
echo "=============================="
HEREDOC
chmod +x "${JOB}"
JID=$(sbatch --parsable "${JOB}")
echo "split${SPLIT}  →  job ${JID}"

done

echo ""
echo "All 5 comparison_acr jobs submitted. Monitor: squeue -u \$USER"
