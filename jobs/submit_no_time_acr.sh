#!/usr/bin/env bash
# submit_no_time_acr.sh
#
# Trains the no-time-token ablation (longitudinal_mk_no_alibi_v2_no_time)
# for all 4 tasks (acr_cls, acr_surv, clad_surv, death_surv) with corrected
# annotation-only ACR labels.
#
# Variant: longitudinal_mk_no_alibi_v2_no_time
#   - Same as LongMIL but without time embeddings on seeds
#   - n_cross_layers=0 (nosab), slot_k=8
#   - acr_cls uses biopsy_batch=4, hinge loss
#   - acr_surv uses default patient-level Cox (biopsy_batch=0)
#
# Output: results/mm_abmil_v8_nosab/ (same as main LongMIL nosab)
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

JOB="${SCRIPTS_DIR}/no_time_acr_s${SPLIT}.sh"
cat > "${JOB}" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=notime_s${SPLIT}
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=220G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=${LOG_DIR}/%j_notime_s${SPLIT}.out
#SBATCH --error=${LOG_DIR}/%j_notime_s${SPLIT}.err
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
echo " no_time ablation  split=${SPLIT}"
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

COMMON="--samples-dir ${SAMPLES} --splits-csv ${SPLITS_CSV} --split ${SPLIT} --fold 0 --out-dir ${OUT_DIR} --phase p2 --p2-variant longitudinal_mk_no_alibi_v2_no_time --slot-k 8 --n-cross-layers 0 --p2-hp-sweep --p2-all-folds --combined-train --modal-dropout 0.3 --workers 8"

# acr_cls: biopsy-level accumulation, hinge loss
echo ""
echo "=============================="
echo " no_time  task=cls  split=${SPLIT}  \$(date)"
echo "=============================="
run_py \${COMMON} --task cls --p2-biopsy-batch 4 --p2-cls-loss hinge || true

# acr_surv: patient-level Cox (default biopsy_batch=0)
echo ""
echo "=============================="
echo " no_time  task=acr_surv  split=${SPLIT}  \$(date)"
echo "=============================="
run_py \${COMMON} --task acr_surv || true

# clad_surv
echo ""
echo "=============================="
echo " no_time  task=clad_surv  split=${SPLIT}  \$(date)"
echo "=============================="
run_py \${COMMON} --task clad_surv || true

# death_surv
echo ""
echo "=============================="
echo " no_time  task=death_surv  split=${SPLIT}  \$(date)"
echo "=============================="
run_py \${COMMON} --task death_surv || true

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
echo "All 5 no_time jobs submitted. Monitor: squeue -u \$USER"
