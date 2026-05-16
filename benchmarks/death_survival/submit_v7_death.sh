#!/usr/bin/env bash
# Death survival — v7 single-task (task=surv, surv_endpoint=death)
# Reuses ACR Phase-1 encoder weights from results_mm_abmil_v7_alt_p1.
# TTE: death_days (biopsy→death); censored = study_end−anchor_dt for alive patients.
set -euo pipefail

TRAIN="/home/aih/dinesh.haridoss/chicago_mil/train_mm_abmil_v7.py"
SAVE_DIR="/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/results_mm_abmil_v7_death"
P1_BASE="/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/results_mm_abmil_v7_alt_p1"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
LOGS="$SAVE_DIR/slurm_logs"
SCRIPTS="$SAVE_DIR/job_scripts"
MAIL="dinesh.haridoss@helmholtz-munich.de"

SPLIT=0
FOLDS="0 1 2 3"
VARIANTS="early late middle crossattn crossmodal iterative"

mkdir -p "$LOGS" "$SCRIPTS"

echo "===== Submitting Death survival (v7) ====="
echo "  Save dir  : $SAVE_DIR"
echo "  P1 base   : $P1_BASE"
echo "  Split     : $SPLIT   Folds: $FOLDS"
echo ""

for FOLD in $FOLDS; do
  for GROUP in $VARIANTS; do

    case $GROUP in
      early)      VARIANT_ARGS="--p2_variants early"; SLOT_ARGS="" ;;
      late)       VARIANT_ARGS="--p2_variants late";  SLOT_ARGS="" ;;
      middle)     VARIANT_ARGS="--p2_variants middle"; SLOT_ARGS="" ;;
      crossattn)  VARIANT_ARGS="--p2_variants crossattn";  SLOT_ARGS="--p2_slot_k 8" ;;
      crossmodal) VARIANT_ARGS="--p2_variants crossmodal"; SLOT_ARGS="--p2_slot_k 8" ;;
      iterative)  VARIANT_ARGS="--p2_variants iterative";  SLOT_ARGS="--p2_slot_k 8 --p2_iter_r 2" ;;
    esac

    JOB="death_v7_s${SPLIT}f${FOLD}_${GROUP}"
    SCRIPT="$SCRIPTS/${JOB}.sh"

    cat > "$SCRIPT" << SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${JOB}
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G
#SBATCH --gres=gpu:1 --constraint="a100_40gb|a100_80gb|h100_80gb"
#SBATCH --time=23:00:00
#SBATCH --output=${LOGS}/${JOB}_%j.out
#SBATCH --error=${LOGS}/${JOB}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL}
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
python -c "import torch; print(f'PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==== Death v7 | ${GROUP} | split=${SPLIT} fold=${FOLD} | \$(date) ===="

python3 -u "${TRAIN}" \\
    --save_dir    "${SAVE_DIR}" \\
    --samples_dir "${SAMPLES}" \\
    --splits_csv  "${SPLITS}" \\
    --split ${SPLIT} --folds ${FOLD} \\
    --task surv \\
    --surv_endpoint death \\
    --p1_base_dir "${P1_BASE}" \\
    --lambda_cox 1.0 \\
    --p2_slot_iters 3 \\
    ${SLOT_ARGS} \\
    ${VARIANT_ARGS}

echo "==== Done Death v7 | ${GROUP} | split=${SPLIT} fold=${FOLD} | \$(date) ===="
SBATCH

    JID=$(sbatch "$SCRIPT" | awk '{print $NF}')
    echo "  Submitted ${JOB} → JID=${JID}"
  done
done

echo ""
echo "Monitor: squeue -u \$USER"
