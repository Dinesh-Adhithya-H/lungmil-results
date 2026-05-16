#!/usr/bin/env bash
# Submit v7 survival endpoints analysis (ACR / CLAD / Death benchmarks + modality baseline)
set -euo pipefail

CHICAGO_MIL="/home/aih/dinesh.haridoss/chicago_mil"
RESULTS_BASE="/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
OUTPUT_DIR="${RESULTS_BASE}/analysis/v7_surv_endpoints"
LOGS="${OUTPUT_DIR}/slurm_logs"
MAIL="dinesh.haridoss@helmholtz-munich.de"

mkdir -p "$LOGS"

JOB_SCRIPT="${OUTPUT_DIR}/run_analyze_surv_endpoints.sh"

{
  echo "#!/usr/bin/env bash"
  echo "#SBATCH --job-name=analyze_surv_eps"
  echo "#SBATCH --partition=cpu_p"
  echo "#SBATCH --qos=cpu_normal"
  echo "#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G"
  echo "#SBATCH --time=02:00:00"
  echo "#SBATCH --output=${LOGS}/analyze_surv_eps_%j.out"
  echo "#SBATCH --error=${LOGS}/analyze_surv_eps_%j.err"
  echo "#SBATCH --mail-type=END,FAIL"
  echo "#SBATCH --mail-user=${MAIL}"
  echo "set -euo pipefail"
  echo 'source "$(conda info --base)/etc/profile.d/conda.sh"'
  echo "conda activate chicago"
  echo 'echo "==== Analyze surv endpoints | $(date) ===="'
  echo "python3 -u \"${CHICAGO_MIL}/analysis/analyze_surv_endpoints.py\" \\"
  echo "    --results_base \"${RESULTS_BASE}\" \\"
  echo "    --splits_csv   \"${SPLITS_CSV}\" \\"
  echo "    --output_dir   \"${OUTPUT_DIR}\" \\"
  echo "    --folds        0 1 2 3"
  echo 'echo "==== Done | $(date) ===="'
} > "$JOB_SCRIPT"

chmod +x "$JOB_SCRIPT"

JID=$(sbatch "$JOB_SCRIPT" | awk '{print $NF}')
echo "Submitted analyze_surv_endpoints JID=${JID}"
echo "Output: ${OUTPUT_DIR}"
echo "Monitor: squeue -u \$USER"
