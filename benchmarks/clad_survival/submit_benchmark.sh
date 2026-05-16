#!/usr/bin/env bash
# CLAD survival benchmark — split 1, all 4 folds
# Pre-CLAD samples only (clad_days > 0); censored patients use study_end or death_days.
# Post-CLAD samples are excluded — we predict "time until CLAD", not post-CLAD trajectory.
set -euo pipefail

TRAIN="/home/aih/dinesh.haridoss/chicago_mil/train_mm_abmil_v6.py"
SAVE_DIR="/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v6_clad_survival"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
LOGS="$SAVE_DIR/slurm_logs"
SCRIPTS="$SAVE_DIR/job_scripts"

SURV_ENDPOINT="clad"
SPLIT=1   # balanced CLAD event rate across folds

mkdir -p "$LOGS" "$SCRIPTS"

for FOLD in 0 1 2 3; do

  # ── Phase 1 ────────────────────────────────────────────────────────────────
  P1_JOB="clad_p1_s${SPLIT}f${FOLD}"
  P1_SCRIPT="$SCRIPTS/${P1_JOB}.sh"
  cat > "$P1_SCRIPT" << SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${P1_JOB}
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G
#SBATCH --gres=gpu:1 --constraint="a100_40gb|a100_80gb|h100_80gb"
#SBATCH --time=23:00:00
#SBATCH --output=${LOGS}/${P1_JOB}_%j.out
#SBATCH --error=${LOGS}/${P1_JOB}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
python -c "import torch; print(f'PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==== CLAD Survival P1 split=${SPLIT} fold=${FOLD} \$(date) ===="
python3 -u "${TRAIN}" \\
    --task survival \\
    --phase 1 \\
    --split ${SPLIT} \\
    --folds ${FOLD} \\
    --samples_dir "${SAMPLES}" \\
    --splits_csv "${SPLITS}" \\
    --save_dir "${SAVE_DIR}" \\
    --p2_surv_endpoint ${SURV_ENDPOINT}
echo "==== Done P1 split=${SPLIT} fold=${FOLD} \$(date) ===="
SBATCH

  P1_JID=$(sbatch "$P1_SCRIPT" | awk '{print $NF}')
  echo "Submitted P1 JID=$P1_JID  $P1_JOB"

  # ── Phase 2 ────────────────────────────────────────────────────────────────
  for GROUP in early late middle crossattn crossmodal iterative; do
    P2_JOB="clad_p2_s${SPLIT}f${FOLD}_${GROUP}"
    P2_SCRIPT="$SCRIPTS/${P2_JOB}.sh"

    case $GROUP in
      early)      VARIANT_ARGS="--p2_variants early early_cls"; SLOT_ARGS="--p2_slot_k 8 16" ;;
      late)       VARIANT_ARGS="--p2_variants late";             SLOT_ARGS="--p2_slot_k 8 16" ;;
      middle)     VARIANT_ARGS="--p2_variants middle middle_cls"; SLOT_ARGS="--p2_slot_k 8 16" ;;
      crossattn)  VARIANT_ARGS="--p2_variants crossattn";        SLOT_ARGS="--p2_slot_k 8 16" ;;
      crossmodal) VARIANT_ARGS="--p2_variants crossmodal";       SLOT_ARGS="--p2_slot_k 8 16" ;;
      iterative)  VARIANT_ARGS="--p2_variants iterative";        SLOT_ARGS="--p2_slot_k 8 16 --p2_iter_r 2 3" ;;
    esac

    cat > "$P2_SCRIPT" << SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${P2_JOB}
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G
#SBATCH --gres=gpu:1 --constraint="a100_40gb|a100_80gb|h100_80gb"
#SBATCH --time=23:00:00
#SBATCH --dependency=afterok:${P1_JID}
#SBATCH --output=${LOGS}/${P2_JOB}_%j.out
#SBATCH --error=${LOGS}/${P2_JOB}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
python -c "import torch; print(f'PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==== CLAD Survival P2 ${GROUP} split=${SPLIT} fold=${FOLD} \$(date) ===="
python3 -u "${TRAIN}" \\
    --task survival \\
    --phase 2 \\
    --split ${SPLIT} \\
    --folds ${FOLD} \\
    --samples_dir "${SAMPLES}" \\
    --splits_csv "${SPLITS}" \\
    --save_dir "${SAVE_DIR}" \\
    --p2_slot_iters 3 \\
    ${SLOT_ARGS} \\
    ${VARIANT_ARGS} \\
    --p2_surv_endpoint ${SURV_ENDPOINT}
echo "==== Done P2 ${GROUP} split=${SPLIT} fold=${FOLD} \$(date) ===="
SBATCH

    P2_JID=$(sbatch "$P2_SCRIPT" | awk '{print $NF}')
    echo "  Submitted P2 JID=$P2_JID  $P2_JOB  (depends on P1 $P1_JID)"
  done

done
