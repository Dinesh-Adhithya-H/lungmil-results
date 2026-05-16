#!/usr/bin/env bash
# Submit full endpoint analysis: combined benchmark + UMAP + combo perf for CLAD/Death
set -euo pipefail

CHICAGO_MIL="/home/aih/dinesh.haridoss/chicago_mil"
RESULTS_BASE="/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
OUTPUT_DIR="${RESULTS_BASE}/analysis/v7_endpoints_full"
LOGS="${OUTPUT_DIR}/slurm_logs"
MAIL="dinesh.haridoss@helmholtz-munich.de"

mkdir -p "$LOGS"

JOB_SCRIPT="${OUTPUT_DIR}/run_analyze_endpoints_full.sh"

{
  echo "#!/usr/bin/env bash"
  echo "#SBATCH --job-name=analyze_endpoints"
  echo "#SBATCH --partition=gpu_p"
  echo "#SBATCH --qos=gpu_normal"
  echo "#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G"
  echo "#SBATCH --gres=gpu:1 --constraint=\"a100_40gb|a100_80gb|h100_80gb\""
  echo "#SBATCH --time=12:00:00"
  echo "#SBATCH --output=${LOGS}/analyze_endpoints_%j.out"
  echo "#SBATCH --error=${LOGS}/analyze_endpoints_%j.err"
  echo "#SBATCH --mail-type=END,FAIL"
  echo "#SBATCH --mail-user=${MAIL}"
  echo "set -euo pipefail"
  echo 'source "$(conda info --base)/etc/profile.d/conda.sh"'
  echo "conda activate chicago"
  echo 'python -c "import torch; print(f'"'"'PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}'"'"')"'
  echo 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
  echo 'echo "==== analyze_endpoints_full | $(date) ===="'
  echo "python3 -u \"${CHICAGO_MIL}/analysis/analyze_v7_endpoints_full.py\" \\"
  echo "    --results_base  \"${RESULTS_BASE}\" \\"
  echo "    --splits_csv    \"${SPLITS_CSV}\" \\"
  echo "    --samples_dir   \"${SAMPLES}\" \\"
  echo "    --output_dir    \"${OUTPUT_DIR}\" \\"
  echo "    --chicago_mil_dir \"${CHICAGO_MIL}\" \\"
  echo "    --device        cuda \\"
  echo "    --folds         0 1 2 3 \\"
  echo "    --focus_variant middle \\"
  echo "    --tasks         all"
  echo 'echo "==== Done | $(date) ===="'
} > "$JOB_SCRIPT"

chmod +x "$JOB_SCRIPT"

JID=$(sbatch "$JOB_SCRIPT" | awk '{print $NF}')
echo "Submitted analyze_endpoints_full JID=${JID}"
echo "Output: ${OUTPUT_DIR}"
echo "Monitor: squeue -u \$USER"
