#!/usr/bin/env bash
# =============================================================================
# submit_tcga_survival.sh  —  TCGA-GBM Survival SLURM submission
# =============================================================================
#
# Submits one job per fold (0..4), then an optional analysis job after all.
#
# Usage
# -----
#   bash submit_tcga_survival.sh                    # all 5 folds, Phase 1+2
#   bash submit_tcga_survival.sh --folds "0 1"      # fold 0 and 1 only
#   bash submit_tcga_survival.sh --phase 1           # Phase 1 only
#   bash submit_tcga_survival.sh --phase 2 \
#       --phase1_dir /path/to/p1                    # reuse existing P1
#   bash submit_tcga_survival.sh --dry_run           # print, don't submit
# =============================================================================
set -euo pipefail

SCRIPT_DIR="/home/aih/dinesh.haridoss/chicago_mil"
SAMPLES_DIR="/lustre/groups/aih/dinesh.haridoss/mil/tcga_cache_gbm/samples"
SPLITS_CSV="${SCRIPT_DIR}/tcga_gbm_splits.csv"

TRAIN_SCRIPT="${SCRIPT_DIR}/train_tcga_survival.py"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_tcga_survival}"
LOG_DIR="${RESULTS_DIR}/slurm_logs"

PARTITION="gpu_p"
QOS="gpu_normal"
MAIL="dinesh.haridoss@helmholtz-munich.de"

TIME="23:00:00"
MEM="200G"
CPUS=8

P2_VARIANTS="early early_cls late middle middle_cls crossattn crossattn_cls crossmodal crossmodal_cls"
P2_SLOT_K="8"
P2_ITER_R="2"

FOLDS="0 1 2 3 4"
PHASE=""
PHASE1_DIR=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --folds)       FOLDS="$2";       shift 2 ;;
        --phase)       PHASE="$2";       shift 2 ;;
        --phase1_dir)  PHASE1_DIR="$2";  shift 2 ;;
        --results_dir) RESULTS_DIR="$2"; LOG_DIR="${RESULTS_DIR}/slurm_logs"; shift 2 ;;
        --dry_run)     DRY_RUN=1;        shift ;;
        *) echo "[error] Unknown flag: $1"; exit 1 ;;
    esac
done

IFS=' ' read -ra FOLD_ARRAY <<< "$FOLDS"
mkdir -p "$LOG_DIR"

CONDA_INIT='
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
python -c "import torch; print(f\"PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}\")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
'

submit_job() {
    local name="$1" dep="$2" script="$3"
    local dep_flag=""
    [[ -n "$dep" ]] && dep_flag="--dependency=${dep} --kill-on-invalid-dep=yes"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry] Would submit: ${name}  dep=${dep:-none}" >&2
        echo "DRY_${name}"; return
    fi
    local jid; jid=$(sbatch --parsable $dep_flag <<< "$script")
    echo "[submit] ${name}  →  JID=${jid}  dep=${dep:-none}" >&2
    echo "$jid"
}

PHASE_ARG=""
[[ -n "$PHASE" ]] && PHASE_ARG="--phase ${PHASE}"
PHASE1_ARG=""
[[ -n "$PHASE1_DIR" ]] && PHASE1_ARG="--phase1_dir ${PHASE1_DIR}"

echo "============================================================"
echo "  submit_tcga_survival.sh  —  TCGA-GBM Survival"
echo "============================================================"
echo "  Train script  : ${TRAIN_SCRIPT}"
echo "  Results dir   : ${RESULTS_DIR}"
echo "  Folds         : ${FOLDS}"
echo "  Phase         : ${PHASE:-1+2}"
echo "  Partition/QOS : ${PARTITION} / ${QOS}"
echo "  Dry run       : ${DRY_RUN}"
echo "============================================================"

declare -a FOLD_JIDS=()

for fold in "${FOLD_ARRAY[@]}"; do
    JOB_NAME="tcga_surv_f${fold}"
    SCRIPT=$(cat << SCRIPT_END
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --gres=gpu:1
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${MAIL}

set -euo pipefail
${CONDA_INIT}

echo "==== TCGA Survival | fold=${fold} | \$(date) ===="
echo "Node: \$(hostname)  JID: \${SLURM_JOB_ID}"

python3 -u "${TRAIN_SCRIPT}" \
    --folds      ${fold} \
    --save_dir   "${RESULTS_DIR}" \
    --samples_dir "${SAMPLES_DIR}" \
    --splits_csv  "${SPLITS_CSV}" \
    --p2_variants ${P2_VARIANTS} \
    --p2_slot_k   ${P2_SLOT_K} \
    --p2_iter_r   ${P2_ITER_R} \
    --p2_slot_iters 3 \
    ${PHASE_ARG} ${PHASE1_ARG}

echo "==== Done fold=${fold} | \$(date) ===="
SCRIPT_END
    )
    JID=$(submit_job "$JOB_NAME" "" "$SCRIPT")
    FOLD_JIDS+=("$JID")
done

echo ""
echo "============================================================"
echo "  Fold JIDs: ${FOLD_JIDS[*]:-none}"
echo ""
if [[ "$DRY_RUN" -eq 0 ]]; then
    echo "  Monitor:"
    echo "    squeue -u \$USER"
    echo ""
    echo "  Live logs:"
    echo "    tail -f ${LOG_DIR}/tcga_surv_f0_*.out"
    echo ""
    echo "  Cancel all:"
    echo "    scancel ${FOLD_JIDS[*]}"
fi
echo "============================================================"
