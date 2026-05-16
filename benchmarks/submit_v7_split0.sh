#!/usr/bin/env bash
# Submit v6 training (Phase 1 → Phase 2 per-variant) for all folds of split 0
# Each P2 variant gets its own job so no single job exceeds 23h.

set -euo pipefail

SCRIPT_DIR="/home/aih/dinesh.haridoss/chicago_mil"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_mm_abmil_v7.py"
SAMPLES_DIR="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
SAVE_DIR="${SCRIPT_DIR}/results_mm_abmil_v7_milv2"
LOG_DIR="${SAVE_DIR}/slurm_logs"
JOB_DIR="${SAVE_DIR}/job_scripts"

SPLIT=0
FOLDS=(0 1 2 3)
P2_VARIANTS=(early early_cls late middle middle_cls)

mkdir -p "${LOG_DIR}" "${JOB_DIR}"

for FOLD in "${FOLDS[@]}"; do
    TAG="v6_milv2_s${SPLIT}f${FOLD}"

    # ── Phase 1 job ──────────────────────────────────────────────────────────
    P1_SCRIPT="${JOB_DIR}/${TAG}_p1.sh"
    cat > "${P1_SCRIPT}" << EOSCRIPT
#!/usr/bin/env bash
#SBATCH --job-name=${TAG}_p1
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_40gb|a100_80gb|h100_80gb"
#SBATCH --time=23:00:00
#SBATCH --output=${LOG_DIR}/${TAG}_p1_%j.out
#SBATCH --error=${LOG_DIR}/${TAG}_p1_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
python -c "import torch; print(f'PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==== P1 split=${SPLIT} fold=${FOLD} \$(date) ===="
python3 -u "${TRAIN_SCRIPT}" \\
    --phase 1 \\
    --split ${SPLIT} \\
    --folds ${FOLD} \\
    --samples_dir "${SAMPLES_DIR}" \\
    --splits_csv "${SPLITS_CSV}" \\
    --save_dir "${SAVE_DIR}"
echo "==== Done P1 split=${SPLIT} fold=${FOLD} \$(date) ===="
EOSCRIPT
    chmod +x "${P1_SCRIPT}"
    P1_JID=$(sbatch --parsable "${P1_SCRIPT}")
    echo "Fold ${FOLD}: P1=${P1_JID}"

    # ── One Phase 2 job per variant, chained after P1 ────────────────────────
    for VARIANT in "${P2_VARIANTS[@]}"; do
        VTAG="${TAG}_p2_${VARIANT}"
        P2_SCRIPT="${JOB_DIR}/${VTAG}.sh"
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
    --p2_slot_k 8 16 \\
    --p2_variants ${VARIANT}
echo "==== Done P2-${VARIANT} split=${SPLIT} fold=${FOLD} \$(date) ===="
EOSCRIPT
        chmod +x "${P2_SCRIPT}"
        P2_JID=$(sbatch --parsable --dependency=afterok:${P1_JID} "${P2_SCRIPT}")
        echo "  Fold ${FOLD} variant=${VARIANT}: P2=${P2_JID} (after P1=${P1_JID})"
    done
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$(whoami)"
