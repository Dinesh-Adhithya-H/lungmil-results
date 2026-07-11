#!/usr/bin/env bash
# =============================================================================
# submit_interpret_v6.sh  —  Interpretability for all v6-milv2 variants
# =============================================================================
# One CPU job per fold × variant; skips if checkpoint missing.
# Uses file-based submission (write script, then sbatch) for reliability.
#
# Usage:
#   bash submit_interpret_v6.sh                          # all folds, all variants
#   bash submit_interpret_v6.sh --folds "0 1"
#   bash submit_interpret_v6.sh --variants "middle iterative_r2_k8"
#   bash submit_interpret_v6.sh --dry_run
# =============================================================================
set -euo pipefail

SCRIPT_DIR="/home/aih/dinesh.haridoss/chicago_mil"
INTERP_SCRIPT="${SCRIPT_DIR}/interpret_mm_abmil.py"
RESULTS_DIR="${SCRIPT_DIR}/results_mm_abmil_v6_milv2"
SAMPLES_DIR="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
OUT_BASE="${SCRIPT_DIR}/interpretability_v6_milv2"
LOG_DIR="${OUT_BASE}/slurm_logs"
JOB_DIR="${OUT_BASE}/job_scripts"

PARTITION="cpu_p"
QOS="cpu_normal"
MAIL="dinesh.haridoss@helmholtz-munich.de"
TIME="08:00:00"
MEM="120G"
CPUS=8

SPLIT=0
FOLDS="0 1 2 3"
ALL_VARIANTS="\
early early_cls late middle middle_cls \
crossattn_k8 crossattn_k8_cls crossattn_k16 crossattn_k16_cls \
crossmodal_k8 crossmodal_k8_cls crossmodal_k16 crossmodal_k16_cls \
iterative_r2_k8 iterative_r2_k8_cls iterative_r2_k16 iterative_r2_k16_cls \
iterative_r3_k8 iterative_r3_k8_cls iterative_r3_k16 iterative_r3_k16_cls"
VARIANTS="$ALL_VARIANTS"

SPLIT_SET="test"
TOP_K=16
N_SAMPLE_PLOTS=20
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split)          SPLIT="$2";          shift 2 ;;
        --folds)          FOLDS="$2";          shift 2 ;;
        --variants)       VARIANTS="$2";       shift 2 ;;
        --split_set)      SPLIT_SET="$2";      shift 2 ;;
        --top_k)          TOP_K="$2";          shift 2 ;;
        --n_sample_plots) N_SAMPLE_PLOTS="$2"; shift 2 ;;
        --dry_run)        DRY_RUN=1;           shift ;;
        *) echo "[error] Unknown flag: $1"; exit 1 ;;
    esac
done

IFS=' ' read -ra FOLD_ARRAY    <<< "$FOLDS"
IFS=' ' read -ra VARIANT_ARRAY <<< "$VARIANTS"
mkdir -p "${LOG_DIR}" "${JOB_DIR}"

echo "========================================================================"
echo "  submit_interpret_v6.sh  —  v6-milv2 interpretability"
echo "========================================================================"
echo "  Split    : ${SPLIT}   Folds : ${FOLDS}"
echo "  Variants : ${#VARIANT_ARRAY[@]}"
echo "  Results  : ${RESULTS_DIR}"
echo "  Samples  : ${SAMPLES_DIR}"
echo "  Out base : ${OUT_BASE}"
echo "  Dry run  : ${DRY_RUN}"
echo "========================================================================"

N_SUBMITTED=0
N_SKIPPED=0

for fold in "${FOLD_ARRAY[@]}"; do
    for variant in "${VARIANT_ARRAY[@]}"; do
        CKPT="${RESULTS_DIR}/split${SPLIT}_fold${fold}/phase2/model_${variant}.pt"
        if [[ ! -f "${CKPT}" ]]; then
            echo "  [skip] no ckpt: fold${fold}/${variant}"
            (( N_SKIPPED++ )) || true
            continue
        fi

        JOB_NAME="iv6_s${SPLIT}f${fold}_${variant}"
        OUT_DIR="${OUT_BASE}/s${SPLIT}f${fold}/${variant}"
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
python -c "import torch; print(f'PyTorch {torch.__version__}')"
echo "==== Interp v6-milv2 [${variant}] split=${SPLIT} fold=${fold} | \$(date) ===="
python3 -u "${INTERP_SCRIPT}" \
    --version v6 \
    --results_dir "${RESULTS_DIR}" \
    --split ${SPLIT} \
    --fold ${fold} \
    --v6_variant ${variant} \
    --out_dir "${OUT_DIR}" \
    --samples_dir "${SAMPLES_DIR}" \
    --splits_csv "${SPLITS_CSV}" \
    --split_set ${SPLIT_SET} \
    --top_k ${TOP_K} \
    --n_sample_plots ${N_SAMPLE_PLOTS}
echo "==== Done [${variant}] split=${SPLIT} fold=${fold} | \$(date) ===="
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
done

echo ""
echo "  Submitted : ${N_SUBMITTED}   Skipped (no ckpt): ${N_SKIPPED}"
echo "  Job scripts: ${JOB_DIR}/"
echo "  Outputs   : ${OUT_BASE}/s${SPLIT}f<F>/<variant>/"
echo "========================================================================"
