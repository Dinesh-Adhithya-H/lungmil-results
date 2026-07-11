#!/usr/bin/env bash
# Submit survival interpretability (run after benchmark completes and best variant is known)
# Usage: bash submit_interpret_survival.sh [SPLIT] [ENDPOINT] [VARIANT]
#   SPLIT:    default 1
#   ENDPOINT: default death
#   VARIANT:  default crossattn_k16  (update after seeing analyze_survival results)
set -euo pipefail

CHICAGO_MIL="/home/aih/dinesh.haridoss/chicago_mil"
RESULTS_DIR="$CHICAGO_MIL/results_mm_abmil_v6_survival"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
OUTPUT_BASE="$CHICAGO_MIL/results/death_survival/interpretability"
LOGS="$RESULTS_DIR/slurm_logs"

SPLIT=${1:-1}
ENDPOINT=${2:-death}
VARIANT=${3:-crossattn_k16}

mkdir -p "$OUTPUT_BASE" "$LOGS"

for FOLD in 0 1 2 3; do
  OUTPUT_DIR="$OUTPUT_BASE/split${SPLIT}_fold${FOLD}_${VARIANT}"
  JOB="interp_surv_s${SPLIT}f${FOLD}"
  SCRIPT="$CHICAGO_MIL/interpretability/run_${JOB}.sh"

  cat > "$SCRIPT" << SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${JOB}
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G
#SBATCH --gres=gpu:1 --constraint="a100_40gb|a100_80gb|h100_80gb"
#SBATCH --time=06:00:00
#SBATCH --output=${LOGS}/${JOB}_%j.out
#SBATCH --error=${LOGS}/${JOB}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
echo "==== Interpret survival split=${SPLIT} fold=${FOLD} variant=${VARIANT} \$(date) ===="
python3 -u ${CHICAGO_MIL}/interpretability/interpret_mm_abmil.py \\
    --version       v6 \\
    --results_dir   ${RESULTS_DIR} \\
    --samples_dir   ${SAMPLES} \\
    --splits_csv    ${SPLITS_CSV} \\
    --out_dir       ${OUTPUT_DIR} \\
    --split         ${SPLIT} \\
    --fold          ${FOLD} \\
    --v6_variant    ${VARIANT} \\
    --task          survival \\
    --surv_endpoint ${ENDPOINT} \\
    --split_set     test \\
    --n_sample_plots 30
echo "==== Done \$(date) ===="
SBATCH

  JID=$(sbatch "$SCRIPT" | awk '{print $NF}')
  echo "Submitted $JOB JID=$JID  (variant=${VARIANT})"
done
