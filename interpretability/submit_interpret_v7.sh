#!/usr/bin/env bash
# =============================================================================
# submit_interpret_v7.sh  —  Submit interpretability jobs for v7 (all folds)
# =============================================================================
# Submits one CPU job per split×fold. Each job extracts attention weights,
# gradient attributions, centroid gates, and cross-modal attention, then
# generates all cohort-level and per-sample visualisation plots.
#
# Usage:
#   bash submit_interpret_v7.sh                        # all splits × folds
#   bash submit_interpret_v7.sh --split_set test       # test set only
#   bash submit_interpret_v7.sh --splits "0 1" --folds "0 1"
#   bash submit_interpret_v7.sh --dry_run
# =============================================================================
set -euo pipefail

SCRIPT_DIR="/home/aih/dinesh.haridoss/chicago_mil"
SAMPLES_DIR="/lustre/groups/aih/dinesh.haridoss/mil/dataset_cache_latest_fixed_large/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"

INTERP_SCRIPT="${SCRIPT_DIR}/interpret_mm_abmil.py"
RESULTS_V7="${SCRIPT_DIR}/results_mm_abmil_v7"
RESULTS_V6="${SCRIPT_DIR}/results_mm_abmil_v6"
OUT_BASE="${SCRIPT_DIR}/interpretability_v7"
LOG_DIR="${OUT_BASE}/slurm_logs"

PARTITION="cpu_p"
QOS="cpu_normal"
MAIL="dinesh.haridoss@helmholtz-munich.de"
TIME="06:00:00"; MEM="120G"; CPUS=8

SPLITS="0 1 2 3 4"
FOLDS="0 1 2 3"
TAG="v7_triple"
SPLIT_SET="test"
TOP_K=16
N_SAMPLE_PLOTS=30
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --splits)        SPLITS="$2";       shift 2 ;;
        --folds)         FOLDS="$2";        shift 2 ;;
        --tag)           TAG="$2";          shift 2 ;;
        --split_set)     SPLIT_SET="$2";    shift 2 ;;
        --top_k)         TOP_K="$2";        shift 2 ;;
        --n_sample_plots) N_SAMPLE_PLOTS="$2"; shift 2 ;;
        --out_base)      OUT_BASE="$2"; LOG_DIR="${OUT_BASE}/slurm_logs"; shift 2 ;;
        --dry_run)       DRY_RUN=1;         shift ;;
        *) echo "[error] Unknown flag: $1"; exit 1 ;;
    esac
done

IFS=' ' read -ra SPLIT_ARRAY <<< "$SPLITS"
IFS=' ' read -ra FOLD_ARRAY  <<< "$FOLDS"
mkdir -p "$LOG_DIR"

CONDA_INIT='
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
python -c "import torch; print(f\"PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}\")"
'

submit_job() {
    local name="$1" script="$2"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry] Would submit: ${name}" >&2
        echo "DRY_${name}"; return
    fi
    local jid
    jid=$(sbatch --parsable <<< "$script")
    echo "[submit] ${name}  →  JID=${jid}" >&2
    echo "$jid"
}

echo "================================================================"
echo "  submit_interpret_v7.sh  —  Interpretability for ABMIL v7"
echo "================================================================"
echo "  Script    : ${INTERP_SCRIPT}"
echo "  V7 results: ${RESULTS_V7}"
echo "  V6 phase1 : ${RESULTS_V6}"
echo "  Output    : ${OUT_BASE}"
echo "  Splits    : ${SPLITS}  |  Folds: ${FOLDS}"
echo "  Tag       : ${TAG}  |  Split set: ${SPLIT_SET}"
echo "  Dry run   : ${DRY_RUN}"
echo "================================================================"
echo ""

for split in "${SPLIT_ARRAY[@]}"; do
    for fold in "${FOLD_ARRAY[@]}"; do
        JOB_NAME="interp_v7_s${split}f${fold}"
        OUT_DIR="${OUT_BASE}/split${split}_fold${fold}"

        SCRIPT=$(cat << SCRIPT_END
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
${CONDA_INIT}

echo "==== Interpretability V7 | split=${split} fold=${fold} | \$(date) ===="

python3 -u "${INTERP_SCRIPT}" --version v7 --results_dir "${RESULTS_V7}" --phase1_dir "${RESULTS_V6}" --split ${split} --fold ${fold} --tag ${TAG} --out_dir "${OUT_DIR}" --samples_dir "${SAMPLES_DIR}" --splits_csv "${SPLITS_CSV}" --split_set ${SPLIT_SET} --top_k ${TOP_K} --n_sample_plots ${N_SAMPLE_PLOTS}

echo "==== Done | split=${split} fold=${fold} | \$(date) ===="
SCRIPT_END
        )
        submit_job "$JOB_NAME" "$SCRIPT"
    done
done

N_JOBS=$(( ${#SPLIT_ARRAY[@]} * ${#FOLD_ARRAY[@]} ))
echo ""
echo "  Submitted ${N_JOBS} interpretability jobs."
echo "  Outputs → ${OUT_BASE}/split<S>_fold<F>/"
echo "    centroid_importance.png"
echo "    cross_modal_attention.png"
echo "    modal_contribution.png"
echo "    top_instances_umap.png"
echo "    attention_gradient_scatter.png"
echo "    joint_high_attention.png"
echo "    per_sample/<stem>_attn.png  (${N_SAMPLE_PLOTS} per fold)"
echo "================================================================"
