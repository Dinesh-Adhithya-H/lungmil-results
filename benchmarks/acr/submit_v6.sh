#!/usr/bin/env bash
# =============================================================================
# submit_v6.sh  —  Full v6 pipeline  (Phase 1 → Phase 2 → Analysis)
#
# Stage 0a  Phase 1 teachers  (HE + Clinical)  — 1 job per fold
# Stage 0b  Phase 1 students  (CT  + BAL)      — 1 job per fold, after 0a
# Stage 1   Phase 2 variants  (4 groups)        — 4 jobs per fold, after 0b
# Stage 2   Analysis                            — 1 job, after all P2
#
# Each stage is skipped automatically if its output already exists.
# Teachers and students each get their own 23-hour budget instead of
# competing for one slot (the original timeout root cause).
#
# Usage
# -----
#   bash submit_v6.sh                          # full pipeline, all splits/folds
#   bash submit_v6.sh --splits "0 1"           # restrict splits
#   bash submit_v6.sh --folds "0 1"            # restrict folds
#   bash submit_v6.sh --phase 1                # Phase 1 only
#   bash submit_v6.sh --phase 2                # Phase 2 only (P1 must exist)
#   bash submit_v6.sh --phase 2 \
#       --phase1_dir /path/to/p1/results       # P2 with external P1 dir
#   bash submit_v6.sh --save_dir /custom/path  # override P2 save dir
#   bash submit_v6.sh --dry_run
# =============================================================================
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="/home/aih/dinesh.haridoss/chicago_mil"
SAMPLES_DIR="/lustre/groups/aih/dinesh.haridoss/mil/dataset_cache_latest_fixed_large/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/train_mm_abmil_v6.py}"
ANALYZE_SCRIPT="${ANALYZE_SCRIPT:-${SCRIPT_DIR}/analyze_v6.py}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_mm_abmil_v6}"

# ── SLURM config ──────────────────────────────────────────────────────────────
PARTITION="gpu_p"
QOS="gpu_normal"
MAIL="dinesh.haridoss@helmholtz-munich.de"
P1_TIME="23:00:00"; P1_MEM="200G"; P1_CPUS=8
P2_TIME="24:00:00"; P2_MEM="200G"; P2_CPUS=8
AN_TIME="04:00:00"; AN_MEM="200G"; AN_CPUS=8

# ── Training config ───────────────────────────────────────────────────────────
# P1: cross-attn + CRD + contrastive, early-stop after 8 eval periods (8×25=200
#     epochs) without improvement, hard cap at 300 epochs per modality.
# Set P1_MODE="none" to use task loss only (no auxiliary losses).
P1_MODE="full"
if [[ "$P1_MODE" == "none" ]]; then
    P1_FLAGS="--p1_mode none --p1_epochs 300 --p1_patience 8"
else
    P1_FLAGS="--p1_mode ${P1_MODE} --p1_cross_attn --p1_crd --p1_contrastive --p1_epochs 300 --p1_patience 8"
fi

# P2: 4 groups, early-stop after 5 eval periods (5×20=100 epochs) without improvement.
P2_SLOT_ITERS=3
P2_ITER_R="2 4"
P2_SLOT_K="8 16"
P2_PATIENCE=5

# ── Pipeline control ──────────────────────────────────────────────────────────
SPLITS="0 1 2 3 4"
FOLDS="0 1 2 3"
PHASE=""          # empty=1+2, 1=P1 only, 2=P2 only
PHASE1_DIR=""     # optional: use external P1 dir for P2
SAVE_DIR=""       # optional: override P2 save dir (default = RESULTS_DIR)
DRY_RUN=0

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --splits)      SPLITS="$2";      shift 2 ;;
        --folds)       FOLDS="$2";       shift 2 ;;
        --phase)       PHASE="$2";       shift 2 ;;
        --phase1_dir)  PHASE1_DIR="$2";  shift 2 ;;
        --save_dir)    SAVE_DIR="$2";    shift 2 ;;
        --results_dir) RESULTS_DIR="$2"; shift 2 ;;
        --p1_mode)     P1_MODE="$2";     shift 2 ;;
        --dry_run)     DRY_RUN=1;        shift ;;
        *) echo "[error] Unknown flag: $1"; exit 1 ;;
    esac
done

# Rebuild P1_FLAGS after arg parsing (so --p1_mode CLI arg takes effect)
if [[ "$P1_MODE" == "none" ]]; then
    P1_FLAGS="--p1_mode none --p1_epochs 300 --p1_patience 8"
else
    P1_FLAGS="--p1_mode ${P1_MODE} --p1_cross_attn --p1_crd --p1_contrastive --p1_epochs 300 --p1_patience 8"
fi

[[ -z "$SAVE_DIR" ]] && SAVE_DIR="$RESULTS_DIR"
P2_PHASE1_DIR="${PHASE1_DIR:-${RESULTS_DIR}}"

IFS=' ' read -ra SPLIT_ARRAY <<< "$SPLITS"
IFS=' ' read -ra FOLD_ARRAY  <<< "$FOLDS"

LOG_DIR="${RESULTS_DIR}/slurm_logs"
JOBS_DIR="${RESULTS_DIR}/job_scripts"
mkdir -p "$LOG_DIR" "$JOBS_DIR"

# ── Helper: write script to file, submit, return JID ─────────────────────────
submit_job() {
    local name="$1" dep="$2" sf="$3"
    local dep_flag=""
    [[ -n "$dep" ]] && dep_flag="--dependency=${dep} --kill-on-invalid-dep=yes"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry] Would submit: ${name}  dep=${dep:-none}" >&2
        echo "DRY_${name}"; return
    fi
    local jid
    jid=$(sbatch --parsable $dep_flag "$sf")
    echo "[submit] ${name}  →  JID=${jid}  dep=${dep:-none}" >&2
    echo "$jid"
}

# ── Helper: check if a modality's P1 is done ─────────────────────────────────
p1_done() {
    local dir="$1" mod="$2"
    [[ -f "${dir}/${mod}/best_model.pt" ]] && return 0 || return 1
}

echo "============================================================"
echo "  submit_v6.sh  —  Multimodal ABMIL v6"
echo "============================================================"
echo "  Results dir  : ${RESULTS_DIR}"
echo "  Save dir     : ${SAVE_DIR}"
echo "  Splits       : ${SPLITS}"
echo "  Folds        : ${FOLDS}"
echo "  Phase        : ${PHASE:-1+2}"
echo "  P1 mode      : ${P1_MODE}"
echo "  P1 flags     : ${P1_FLAGS}"
echo "  P2 R×K grid  : R={${P2_ITER_R}}  K={${P2_SLOT_K}}  patience=${P2_PATIENCE}"
echo "  Partition    : ${PARTITION} / ${QOS}"
echo "  Dry run      : ${DRY_RUN}"
echo "============================================================"
echo ""

declare -A P1T_JIDS   # teacher job IDs
declare -A P1S_JIDS   # student job IDs
declare -a P2_JIDS=()

for split in "${SPLIT_ARRAY[@]}"; do
    for fold in "${FOLD_ARRAY[@]}"; do
        TAG="split${split}_fold${fold}"
        P1_DIR="${RESULTS_DIR}/${TAG}/phase1"
        echo "── ${TAG} ──────────────────────────────────────────────────"

        P1T_JID=""
        P1S_JID=""

        # ════════════════════════════════════════════════════════
        # STAGE 0a — P1 teachers (HE + Clinical)
        # ════════════════════════════════════════════════════════
        if [[ -z "$PHASE" || "$PHASE" == "1" ]]; then
            if p1_done "$P1_DIR" HE && p1_done "$P1_DIR" Clinical; then
                echo "  [skip 0a] HE + Clinical already done."
            else
                JOB_NAME="v6_p1t_s${split}f${fold}"
                SF="${JOBS_DIR}/${JOB_NAME}.sh"
                cat > "$SF" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${P1_CPUS}
#SBATCH --mem=${P1_MEM}
#SBATCH --gres=gpu:1
#SBATCH --time=${P1_TIME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${MAIL}
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — aborting'; print(f'PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==== P1 teachers | ${TAG} | \$(date) ===="
python3 -u "${TRAIN_SCRIPT}" --phase 1 --split ${split} --folds ${fold} --samples_dir "${SAMPLES_DIR}" --splits_csv "${SPLITS_CSV}" --save_dir "${RESULTS_DIR}" --p1_teacher_mods HE Clinical ${P1_FLAGS}
echo "==== Done P1 teachers | ${TAG} | \$(date) ===="
HEREDOC
                chmod +x "$SF"
                P1T_JID=$(submit_job "$JOB_NAME" "" "$SF")
            fi
        fi

        # ════════════════════════════════════════════════════════
        # STAGE 0b — P1 students (CT + BAL), depends on 0a
        # ════════════════════════════════════════════════════════
        if [[ -z "$PHASE" || "$PHASE" == "1" ]]; then
            if p1_done "$P1_DIR" CT && p1_done "$P1_DIR" BAL; then
                echo "  [skip 0b] CT + BAL already done."
            else
                JOB_NAME="v6_p1s_s${split}f${fold}"
                SF="${JOBS_DIR}/${JOB_NAME}.sh"
                T_DEP="${P1T_JID:+afterok:${P1T_JID}}"
                cat > "$SF" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${P1_CPUS}
#SBATCH --mem=${P1_MEM}
#SBATCH --gres=gpu:1
#SBATCH --time=${P1_TIME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${MAIL}
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — aborting'; print(f'PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==== P1 students | ${TAG} | \$(date) ===="
python3 -u "${TRAIN_SCRIPT}" --phase 1 --split ${split} --folds ${fold} --samples_dir "${SAMPLES_DIR}" --splits_csv "${SPLITS_CSV}" --save_dir "${RESULTS_DIR}" --p1_student_mods CT BAL ${P1_FLAGS}
echo "==== Done P1 students | ${TAG} | \$(date) ===="
HEREDOC
                chmod +x "$SF"
                P1S_JID=$(submit_job "$JOB_NAME" "$T_DEP" "$SF")
            fi
        fi

        # ════════════════════════════════════════════════════════
        # STAGE 1 — Phase 2 (4 groups), depends on 0b
        # ════════════════════════════════════════════════════════
        if [[ -z "$PHASE" || "$PHASE" == "2" ]]; then
            P2_DEP="${P1S_JID:+afterok:${P1S_JID}}"

            # ── Group A1: early, late ──────────────────────────
            JOB_NAME="v6_p2A1_s${split}f${fold}"
            SF="${JOBS_DIR}/${JOB_NAME}.sh"
            cat > "$SF" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${P2_CPUS}
#SBATCH --mem=${P2_MEM}
#SBATCH --gres=gpu:1
#SBATCH --time=${P2_TIME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${MAIL}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — aborting'; print(f'PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==== P2-A1 | ${TAG} | \$(date) ===="
python3 -u "${TRAIN_SCRIPT}" --phase 2 --split ${split} --folds ${fold} --samples_dir "${SAMPLES_DIR}" --splits_csv "${SPLITS_CSV}" --phase1_dir "${P2_PHASE1_DIR}" --save_dir "${SAVE_DIR}" --p2_slot_iters ${P2_SLOT_ITERS} --p2_slot_k ${P2_SLOT_K} --p2_patience ${P2_PATIENCE} --p2_variants early early_cls late
echo "==== Done P2-A1 | ${TAG} | \$(date) ===="
HEREDOC
            chmod +x "$SF"
            JID=$(submit_job "$JOB_NAME" "$P2_DEP" "$SF")
            P2_JIDS+=("$JID")

            # ── Group A2: middle ───────────────────────────────
            JOB_NAME="v6_p2A2_s${split}f${fold}"
            SF="${JOBS_DIR}/${JOB_NAME}.sh"
            cat > "$SF" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${P2_CPUS}
#SBATCH --mem=${P2_MEM}
#SBATCH --gres=gpu:1
#SBATCH --time=${P2_TIME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${MAIL}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — aborting'; print(f'PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==== P2-A2 | ${TAG} | \$(date) ===="
python3 -u "${TRAIN_SCRIPT}" --phase 2 --split ${split} --folds ${fold} --samples_dir "${SAMPLES_DIR}" --splits_csv "${SPLITS_CSV}" --phase1_dir "${P2_PHASE1_DIR}" --save_dir "${SAVE_DIR}" --p2_slot_iters ${P2_SLOT_ITERS} --p2_slot_k ${P2_SLOT_K} --p2_patience ${P2_PATIENCE} --p2_variants middle middle_cls
echo "==== Done P2-A2 | ${TAG} | \$(date) ===="
HEREDOC
            chmod +x "$SF"
            JID=$(submit_job "$JOB_NAME" "$P2_DEP" "$SF")
            P2_JIDS+=("$JID")

            # ── Group B: crossattn, crossmodal ─────────────────
            # grad_ckpt + 1024 max_patches to avoid OOM on 20 GB GPUs
            JOB_NAME="v6_p2B_s${split}f${fold}"
            SF="${JOBS_DIR}/${JOB_NAME}.sh"
            cat > "$SF" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${P2_CPUS}
#SBATCH --mem=${P2_MEM}
#SBATCH --gres=gpu:1
#SBATCH --time=${P2_TIME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${MAIL}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — aborting'; print(f'PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==== P2-B | ${TAG} | \$(date) ===="
python3 -u "${TRAIN_SCRIPT}" --phase 2 --split ${split} --folds ${fold} --samples_dir "${SAMPLES_DIR}" --splits_csv "${SPLITS_CSV}" --phase1_dir "${P2_PHASE1_DIR}" --save_dir "${SAVE_DIR}" --p2_slot_iters ${P2_SLOT_ITERS} --p2_slot_k ${P2_SLOT_K} --p2_patience ${P2_PATIENCE} --p2_grad_ckpt --p2_max_patches 1024 --p2_variants crossattn crossattn_cls crossmodal crossmodal_cls
echo "==== Done P2-B | ${TAG} | \$(date) ===="
HEREDOC
            chmod +x "$SF"
            JID=$(submit_job "$JOB_NAME" "$P2_DEP" "$SF")
            P2_JIDS+=("$JID")

            # ── Group C: iterative (R×K grid, grad ckpt) ───────
            # grad_ckpt + 1024 max_patches + 512 max_he_block to avoid OOM
            JOB_NAME="v6_p2C_s${split}f${fold}"
            SF="${JOBS_DIR}/${JOB_NAME}.sh"
            cat > "$SF" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${P2_CPUS}
#SBATCH --mem=${P2_MEM}
#SBATCH --gres=gpu:1
#SBATCH --time=${P2_TIME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${MAIL}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — aborting'; print(f'PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==== P2-C | ${TAG} | \$(date) ===="
python3 -u "${TRAIN_SCRIPT}" --phase 2 --split ${split} --folds ${fold} --samples_dir "${SAMPLES_DIR}" --splits_csv "${SPLITS_CSV}" --phase1_dir "${P2_PHASE1_DIR}" --save_dir "${SAVE_DIR}" --p2_slot_iters ${P2_SLOT_ITERS} --p2_slot_k ${P2_SLOT_K} --p2_iter_r ${P2_ITER_R} --p2_grad_ckpt --p2_max_patches 1024 --p2_max_he_block 512 --p2_patience ${P2_PATIENCE} --p2_variants iterative iterative_cls
echo "==== Done P2-C | ${TAG} | \$(date) ===="
HEREDOC
            chmod +x "$SF"
            JID=$(submit_job "$JOB_NAME" "$P2_DEP" "$SF")
            P2_JIDS+=("$JID")
        fi

    done
done

# ════════════════════════════════════════════════════════════
# STAGE 2 — Analysis (after all P2)
# ════════════════════════════════════════════════════════════
AN_JID=""
if [[ -z "$PHASE" || "$PHASE" == "2" ]]; then
    echo ""
    echo "── Stage 2: Analysis ────────────────────────────────────────"
    AN_DEP=""
    if [[ "${#P2_JIDS[@]}" -gt 0 && $DRY_RUN -eq 0 ]]; then
        VALID=()
        for jid in "${P2_JIDS[@]}"; do [[ -n "$jid" ]] && VALID+=("$jid"); done
        [[ "${#VALID[@]}" -gt 0 ]] && AN_DEP="afterok:$(IFS=':'; echo "${VALID[*]}")"
    elif [[ $DRY_RUN -eq 1 ]]; then
        AN_DEP="afterok:ALL_P2_JOBS"
    fi
    JOB_NAME="v6_analyze"
    SF="${JOBS_DIR}/${JOB_NAME}.sh"
    cat > "$SF" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${AN_CPUS}
#SBATCH --mem=${AN_MEM}
#SBATCH --gres=gpu:1
#SBATCH --time=${AN_TIME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL}
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
echo "==== analyze_v6.py | \$(date) ===="
python3 -u "${ANALYZE_SCRIPT}" --results_dir "${SAVE_DIR}" --samples_dir "${SAMPLES_DIR}" --splits_csv "${SPLITS_CSV}" --output_dir "${SAVE_DIR}/analysis" --folds ${FOLDS} --outer_splits ${SPLITS} --tasks all --hidden_dim 256 --dropout 0.4 --n_per_split 20 --min_combo_n 3
echo "==== Analysis done | \$(date) ===="
HEREDOC
    chmod +x "$SF"
    AN_JID=$(submit_job "$JOB_NAME" "$AN_DEP" "$SF")
fi

# ── Summary ───────────────────────────────────────────────────────────────────
N_SF=$(( ${#SPLIT_ARRAY[@]} * ${#FOLD_ARRAY[@]} ))
echo ""
echo "============================================================"
echo "  Job graph"
echo "============================================================"
[[ -z "$PHASE" || "$PHASE" == "1" ]] && printf "  %-30s %s\n" "P1 teacher jobs:" "${N_SF}  (1 per fold)"
[[ -z "$PHASE" || "$PHASE" == "1" ]] && printf "  %-30s %s\n" "P1 student jobs:" "${N_SF}  (1 per fold, after teachers)"
[[ -z "$PHASE" || "$PHASE" == "2" ]] && printf "  %-30s %s\n" "P2 jobs:" "${#P2_JIDS[@]}  (4 groups × ${N_SF} folds)"
[[ -z "$PHASE" || "$PHASE" == "2" ]] && printf "  %-30s %s\n" "Analysis job:" "1"
echo ""
if [[ $DRY_RUN -eq 0 ]]; then
    echo "  Monitor : watch -n 60 squeue -u \$USER"
    echo "  Logs    : ${LOG_DIR}/"
    [[ -n "${AN_JID}" ]] && echo "  Analysis JID: ${AN_JID}"
    echo ""
    ALL_JIDS=("${P2_JIDS[@]:-}" "${AN_JID:-}")
    echo "  Cancel  : scancel ${ALL_JIDS[*]}"
fi
echo "============================================================"
