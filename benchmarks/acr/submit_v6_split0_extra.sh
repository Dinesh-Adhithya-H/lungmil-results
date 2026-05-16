#!/usr/bin/env bash
# Submit v6 Phase 2 — extra variants (crossattn, crossmodal, iterative + _cls)
# P1 checkpoints already exist for all folds — no P1 dependency needed.
# Grid per variant:
#   crossattn / crossmodal / *_cls : slot_k in {8, 16}          → 2 ckpts
#   iterative / iterative_cls       : iter_r in {2,3} × k {8,16} → 4 ckpts

set -euo pipefail

SCRIPT_DIR="/home/aih/dinesh.haridoss/chicago_mil"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_mm_abmil_v6.py"
SAMPLES_DIR="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
SAVE_DIR="${SCRIPT_DIR}/results_mm_abmil_v6_milv2"
LOG_DIR="${SAVE_DIR}/slurm_logs"
JOB_DIR="${SAVE_DIR}/job_scripts"

SPLIT=0
FOLDS=(0 1 2 3)

mkdir -p "${LOG_DIR}" "${JOB_DIR}"

# ── helper: submit one P2 job ─────────────────────────────────────────────────
submit_p2() {
    local FOLD=$1
    local VARIANT=$2
    local EXTRA_ARGS=$3          # e.g. "--p2_slot_k 8 16 --p2_iter_r 2 3"

    local TAG="v6_milv2_s${SPLIT}f${FOLD}"
    local VTAG="${TAG}_p2_${VARIANT}"
    local P2_SCRIPT="${JOB_DIR}/${VTAG}.sh"

    cat > "${P2_SCRIPT}" << EOSCRIPT
#!/usr/bin/env bash
#SBATCH --job-name=${VTAG}
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_40gb|a100_80gb|h100_80gb"
#SBATCH --time=23:00:00
#SBATCH --output=${LOG_DIR}/${VTAG}_%j.out
#SBATCH --error=${LOG_DIR}/${VTAG}_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
python -c "import torch; print(f'PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==== P2-${VARIANT} split=${SPLIT} fold=${FOLD} \$(date) ===="
python3 -u "${TRAIN_SCRIPT}" \\
    --phase 2 \\
    --split ${SPLIT} \\
    --folds ${FOLD} \\
    --samples_dir "${SAMPLES_DIR}" \\
    --splits_csv "${SPLITS_CSV}" \\
    --phase1_dir "${SAVE_DIR}" \\
    --save_dir "${SAVE_DIR}" \\
    --p2_slot_iters 3 \\
    ${EXTRA_ARGS} \\
    --p2_variants ${VARIANT}
echo "==== Done P2-${VARIANT} split=${SPLIT} fold=${FOLD} \$(date) ===="
EOSCRIPT
    chmod +x "${P2_SCRIPT}"
    local JID
    JID=$(sbatch --parsable "${P2_SCRIPT}")
    echo "  Fold ${FOLD} variant=${VARIANT}: JID=${JID}"
}

# ── Submit all extra variants for all folds ───────────────────────────────────
for FOLD in "${FOLDS[@]}"; do
    echo "=== Fold ${FOLD} ==="

    # crossattn / crossattn_cls — grid: slot_k {8, 16}
    submit_p2 "${FOLD}" "crossattn"     "--p2_slot_k 8 16"
    submit_p2 "${FOLD}" "crossattn_cls" "--p2_slot_k 8 16"

    # crossmodal / crossmodal_cls — grid: slot_k {8, 16}
    submit_p2 "${FOLD}" "crossmodal"     "--p2_slot_k 8 16"
    submit_p2 "${FOLD}" "crossmodal_cls" "--p2_slot_k 8 16"

    # iterative / iterative_cls — grid: iter_r {2, 3} × slot_k {8, 16}
    submit_p2 "${FOLD}" "iterative"     "--p2_iter_r 2 3 --p2_slot_k 8 16"
    submit_p2 "${FOLD}" "iterative_cls" "--p2_iter_r 2 3 --p2_slot_k 8 16"
done

echo ""
echo "All extra-variant jobs submitted ($(( ${#FOLDS[@]} * 6 )) total)."
echo "Monitor with: squeue -u \$(whoami)"
