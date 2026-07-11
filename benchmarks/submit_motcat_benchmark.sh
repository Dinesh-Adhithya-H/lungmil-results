#!/bin/bash
# MOTCAT benchmark on TCGA cancers with UNI-2 (1536-dim) features.
# Covers 4 cancers (BLCA, BRCA, GBMLGG, LUAD) — KIRC has no MOTCAT splits/csv.
#
# Array layout (20 jobs):
#   0-4:   BLCA   fold 0-4
#   5-9:   BRCA   fold 0-4
#   10-14: GBMLGG fold 0-4
#   15-19: LUAD   fold 0-4
#
#SBATCH --job-name=motcat_bench
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%A_%a_motcat_bench.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%A_%a_motcat_bench.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -euo pipefail
export PYTHONUNBUFFERED=1
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

MIL_DIR="/lustre/groups/aih/dinesh.haridoss/mil"
MOTCAT_DIR="${MIL_DIR}/MOTCAT"
RESULTS_BASE="/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_tcga_competitors"
mkdir -p "${RESULTS_BASE}"

CANCERS=(BLCA BRCA GBMLGG LUAD)

IDX=$SLURM_ARRAY_TASK_ID
CANCER_IDX=$((IDX / 5))
FOLD=$((IDX % 5))

CANCER=${CANCERS[$CANCER_IDX]}
CANCER_LOWER=$(echo "$CANCER" | tr '[:upper:]' '[:lower:]')

echo "========================================"
echo "Model: MOTCAT  Cancer: ${CANCER}  Fold: ${FOLD}"
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started: $(date)"
echo "========================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
python -m pip install -q scikit-survival tensorboardX pot || true

RESULTS_DIR="${RESULTS_BASE}/motcat_${CANCER_LOWER}_fold${FOLD}"
mkdir -p "${RESULTS_DIR}"

cd "${MOTCAT_DIR}"

python main.py \
    --data_root_dir "${MIL_DIR}/uni2_pt/TCGA-${CANCER}" \
    --split_dir "tcga_${CANCER_LOWER}" \
    --which_splits "5foldcv" \
    --model_type "motcat" \
    --mode "coattn" \
    --path_input_dim 1536 \
    --fusion concat \
    --apply_sig \
    --k_start ${FOLD} \
    --k_end $((FOLD + 1)) \
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

EXIT_CODE=$?

# Log val c-index to wandb (post-hoc, from summary CSV)
WANDB_PROJECT="chicago-mil-tcga-competitors"
python3 - <<PYEOF 2>/dev/null || true
import os, sys
try:
    import wandb, pandas as pd
    project = "${WANDB_PROJECT}"
    cancer_lower = "${CANCER_LOWER}"
    fold = ${FOLD}
    results_dir = "${RESULTS_DIR}"

    run = wandb.init(
        project=project,
        name=f"motcat_{cancer_lower}_fold{fold}",
        group=f"motcat_{cancer_lower}",
        config={"cancer": cancer_lower, "fold": fold, "model": "motcat",
                "wsi_input_dim": 1536, "max_epochs": 50, "lr": 2e-4,
                "ot_impl": "pot-uot-l2"},
        reinit=True,
    )
    import glob
    # MOTCAT writes summary inside a subdirectory with exp_code
    csvs = glob.glob(os.path.join(results_dir, "**", "summary_latest.csv"), recursive=True)
    if csvs:
        df = pd.read_csv(csvs[0])
        row = df[df["folds"] == fold]
        if len(row):
            val_ci = float(row["val_cindex"].values[0])
            run.log({"fold": fold, "val/cindex": val_ci})
            run.summary["val_cindex"] = val_ci
            print(f"  wandb logged motcat_{cancer_lower}_fold{fold} val_cindex={val_ci:.4f}")
    run.finish()
except Exception as e:
    print(f"  [wandb] error: {e}", file=sys.stderr)
PYEOF

echo "========================================"
echo "MOTCAT ${CANCER} fold${FOLD} done  exit=${EXIT_CODE}  $(date)"
echo "========================================"
exit ${EXIT_CODE}
