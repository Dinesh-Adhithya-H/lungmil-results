#!/usr/bin/env bash
# =============================================================================
# submit_interpret_clad_best.sh  —  Interpretability for CLAD survival
# =============================================================================
# Runs interpret_mm_abmil.py (--task survival --surv_endpoint clad) for the
# best variant (iterative_r3_k16), split 1, folds 0-3.
# One CPU job per fold.
# =============================================================================
set -euo pipefail

CHICAGO_MIL="/home/aih/dinesh.haridoss/chicago_mil"
INTERP_SCRIPT="${CHICAGO_MIL}/interpretability/interpret_mm_abmil.py"
RESULTS_DIR="${CHICAGO_MIL}/results/clad_survival/runs"
SAMPLES_DIR="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
OUT_BASE="${CHICAGO_MIL}/results/clad_survival/interpretability"
LOG_DIR="${OUT_BASE}/slurm_logs"
JOB_DIR="${OUT_BASE}/job_scripts"

PARTITION="cpu_p"
QOS="cpu_normal"
MAIL="dinesh.haridoss@helmholtz-munich.de"
TIME="08:00:00"
MEM="120G"
CPUS=8

SPLIT=1
VARIANT="iterative_r3_k16"
FOLDS="0 1 2 3"
SPLIT_SET="test"
TOP_K=16
N_SAMPLE_PLOTS=20
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --folds)          FOLDS="$2";          shift 2 ;;
        --variant)        VARIANT="$2";        shift 2 ;;
        --n_sample_plots) N_SAMPLE_PLOTS="$2"; shift 2 ;;
        --dry_run)        DRY_RUN=1;           shift ;;
        *) echo "[error] Unknown flag: $1"; exit 1 ;;
    esac
done

IFS=' ' read -ra FOLD_ARRAY <<< "$FOLDS"
mkdir -p "${LOG_DIR}" "${JOB_DIR}"

echo "========================================================================"
echo "  submit_interpret_clad_best.sh  —  CLAD survival interpretability"
echo "  Split: ${SPLIT}  Variant: ${VARIANT}  Folds: ${FOLDS}"
echo "  Results : ${RESULTS_DIR}"
echo "  Out     : ${OUT_BASE}"
echo "  Dry run : ${DRY_RUN}"
echo "========================================================================"

N_SUBMITTED=0
N_SKIPPED=0

for fold in "${FOLD_ARRAY[@]}"; do
    CKPT="${RESULTS_DIR}/split${SPLIT}_fold${fold}/phase2/model_${VARIANT}.pt"
    if [[ ! -f "${CKPT}" ]]; then
        echo "  [skip] no ckpt: split${SPLIT}_fold${fold}/${VARIANT}"
        (( N_SKIPPED++ )) || true
        continue
    fi

    JOB_NAME="interp_clad_s${SPLIT}f${fold}"
    OUT_DIR="${OUT_BASE}/split${SPLIT}_fold${fold}_${VARIANT}"
    JOB_SCRIPT="${JOB_DIR}/${JOB_NAME}.sh"
    mkdir -p "${OUT_DIR}"

    cat > "${JOB_SCRIPT}" << EOSCRIPT
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${MAIL}
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
export PYTHONPATH="${CHICAGO_MIL}:\${PYTHONPATH:-}"
python -c "import torch; print(f'PyTorch {torch.__version__}')"
echo "==== Interp CLAD [${VARIANT}] split=${SPLIT} fold=${fold} | \$(date) ===="
python3 -u "${INTERP_SCRIPT}" \
    --version       v6 \
    --results_dir   "${RESULTS_DIR}" \
    --split         ${SPLIT} \
    --fold          ${fold} \
    --v6_variant    ${VARIANT} \
    --out_dir       "${OUT_DIR}" \
    --samples_dir   "${SAMPLES_DIR}" \
    --splits_csv    "${SPLITS_CSV}" \
    --split_set     ${SPLIT_SET} \
    --task          survival \
    --surv_endpoint clad \
    --top_k         ${TOP_K} \
    --n_sample_plots ${N_SAMPLE_PLOTS}
echo "==== Done [${VARIANT}] split=${SPLIT} fold=${fold} | \$(date) ===="
EOSCRIPT

    chmod +x "${JOB_SCRIPT}"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "  [dry] ${JOB_NAME}"
    else
        JID=$(sbatch --parsable "${JOB_SCRIPT}")
        echo "  [submit] ${JOB_NAME}  →  JID=${JID}"
        (( N_SUBMITTED++ )) || true
    fi
done

echo ""
echo "  Submitted: ${N_SUBMITTED}   Skipped (no ckpt): ${N_SKIPPED}"
echo "  Outputs  : ${OUT_BASE}/split${SPLIT}_fold<F>_${VARIANT}/"
echo "========================================================================"
