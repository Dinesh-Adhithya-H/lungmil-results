#!/bin/bash
# submit_motcat_lung_benchmark.sh
# ---------------------------------------------------------------------------
# MOTCAT (Chen et al. 2021) on lung transplant data.
#
# Input design:
#   path bag  : HE_cells (N, 1024) — MOTCAT supports one bag input only
#   omic      : Clinical (106-dim float) split into 6 signature groups
#
# Array layout (15 jobs = 3 tasks × 5 splits):
#   0-4:   lung_acr   split 0-4
#   5-9:   lung_clad  split 0-4
#   10-14: lung_death split 0-4
#
# Prerequisite: sbatch benchmarks/prep_lung_benchmark.sh
# Submit:       sbatch benchmarks/submit_motcat_lung_benchmark.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=motcat_lung
#SBATCH --array=0-14
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%A_%a_motcat_lung.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%A_%a_motcat_lung.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -euo pipefail
export PYTHONUNBUFFERED=1
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

MIL_DIR="/lustre/groups/aih/dinesh.haridoss/mil"
MOTCAT_DIR="${MIL_DIR}/MOTCAT"
DATA_DIR="${MIL_DIR}/lung_mcat_data"
RESULTS_BASE="/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_lung_competitors"
WANDB_PROJECT="chicago-mil-lung-competitors"
mkdir -p "${RESULTS_BASE}"

TASKS=(lung_acr lung_clad lung_death)

IDX=$SLURM_ARRAY_TASK_ID
TASK_IDX=$((IDX / 5))
SPLIT=$((IDX % 5))
TASK=${TASKS[$TASK_IDX]}

echo "========================================"
echo "MOTCAT lung benchmark"
echo "Task=${TASK}  Split=${SPLIT}"
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started: $(date)"
echo "========================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python -m pip install -q scikit-survival tensorboardX pot || true

RESULTS_DIR="${RESULTS_BASE}/motcat_${TASK}_split${SPLIT}"
mkdir -p "${RESULTS_DIR}"

# Skip if already done
if [ -f "${RESULTS_DIR}/summary_latest.csv" ] || \
   [ -n "$(find "${RESULTS_DIR}" -name 'summary_latest.csv' 2>/dev/null)" ]; then
    echo "Already done — skipping"
    exit 0
fi

# MOTCAT reads CSV from ./dataset_csv/{task}_all_clean.csv relative to cwd
# and splits from ./splits/5foldcv/{task}/splits_{k}.csv
# Both are symlinked by prep_lung_benchmark.sh
cd "${MOTCAT_DIR}"

python main.py \
    --data_root_dir "${DATA_DIR}" \
    --split_dir "${TASK}" \
    --which_splits "5foldcv" \
    --model_type "motcat" \
    --mode "coattn" \
    --path_input_dim 1024 \
    --fusion concat \
    --apply_sig \
    --k_start ${SPLIT} \
    --k_end $((SPLIT + 1)) \
    --results_dir "${RESULTS_DIR}" \
    --max_epochs 50 \
    --lr 2e-4 \
    --bag_loss nll_surv \
    --reg 1e-5 \
    --drop_out \
    --weighted_sample \
    --ot_impl pot-uot-l2 \
    --ot_reg 0.1 \
    --ot_tau 0.5 \
    --seed 1 \
    --overwrite

echo "-------- logging to wandb --------"
export _WRES="${RESULTS_DIR}" _WTASK="${TASK}" _WSPLIT="${SPLIT}" _WPROJ="${WANDB_PROJECT}"
python3 - <<'PYEOF'
import os, sys
try:
    import wandb, pandas as pd, glob
    results_dir = os.environ["_WRES"]
    task        = os.environ["_WTASK"]
    split       = int(os.environ["_WSPLIT"])
    project     = os.environ["_WPROJ"]
    run = wandb.init(
        project=project,
        name=f"motcat_{task}_split{split}",
        group=f"motcat_{task}",
        config={"task": task, "split": split, "model": "motcat",
                "wsi_input_dim": 1024, "omic": "Clinical106", "path": "HE_only", "max_epochs": 50,
                "ot_impl": "pot-uot-l2"},
        reinit=True,
    )
    csvs = glob.glob(os.path.join(results_dir, "**", "summary_latest.csv"), recursive=True)
    if csvs:
        df = pd.read_csv(csvs[0])
        row = df[df["folds"] == split] if "folds" in df.columns else df.head(1)
        if len(row):
            val_ci = float(row["val_cindex"].values[0])
            run.log({"split": split, "val/cindex": val_ci})
            run.summary["val_cindex"] = val_ci
            print(f"  wandb logged val_cindex={val_ci:.4f}")
    run.finish()
except Exception as e:
    print(f"  [wandb] error: {e}", file=sys.stderr)
PYEOF

echo "========================================"
echo "MOTCAT ${TASK} split${SPLIT} done  $(date)"
echo "========================================"
