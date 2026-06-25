#!/usr/bin/env bash
# Submit unified MIL analysis (benchmark + UMAP + combo) for all tasks.
# Usage: bash submit_analysis.sh [--tasks acr_cls acr_tte ...] [--skip_umap] [--skip_combo]
set -euo pipefail

CHICAGO_MIL="/home/aih/dinesh.haridoss/chicago_mil"
RESULTS_BASE="/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
OUTPUT_DIR="${RESULTS_BASE}/analysis/v7_analysis"
LOGS="${OUTPUT_DIR}/slurm_logs"
MAIL="dinesh.haridoss@helmholtz-munich.de"

# Forward any extra flags (e.g. --tasks, --skip_umap) directly to run_analysis.py
EXTRA_ARGS="${*}"

mkdir -p "$LOGS"

JOB_SCRIPT=$(mktemp /tmp/submit_analysis_XXXX.sh)
trap "rm -f $JOB_SCRIPT" EXIT

cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=mil_analysis
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G
#SBATCH --gres=gpu:1 --constraint="a100_40gb|a100_80gb|h100_80gb"
#SBATCH --time=12:00:00
#SBATCH --output=${LOGS}/mil_analysis_%j.out
#SBATCH --error=${LOGS}/mil_analysis_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL}
set -euo pipefail

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python -c "import torch; print(f'PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "==== mil_analysis | \$(date) ===="

python3 -u "${CHICAGO_MIL}/analysis/run_analysis.py" \\
    --results_base    "${RESULTS_BASE}" \\
    --splits_csv      "${SPLITS_CSV}" \\
    --samples_dir     "${SAMPLES}" \\
    --output_dir      "${OUTPUT_DIR}" \\
    --chicago_mil_dir "${CHICAGO_MIL}" \\
    --device          cuda \\
    --folds           0 1 2 3 \\
    --splits          0 \\
    --tasks           all \\
    ${EXTRA_ARGS}

echo "==== Done | \$(date) ===="
EOF

chmod +x "$JOB_SCRIPT"

JID=$(sbatch "$JOB_SCRIPT" | awk '{print $NF}')
echo "Submitted mil_analysis  JID=${JID}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Logs:       ${LOGS}/mil_analysis_${JID}.out"
echo "Monitor:    squeue -u \$USER"
