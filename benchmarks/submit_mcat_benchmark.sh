#!/bin/bash
# submit_mcat_benchmark.sh
# ---------------------------------------------------------------------------
# MCAT (Chen et al. 2021) + ABMIL baselines on TCGA with UNI-2 (1536-dim) WSI
# features, using the official MCAT repo on Lustre — NO code changes, just a
# wrapper that runs `python main.py` with our 5 cancers.
#
# Uses existing assets on Lustre:
#   repo    : /lustre/groups/aih/dinesh.haridoss/mil/MCAT
#   WSI feats: /lustre/groups/aih/dinesh.haridoss/mil/uni2_pt/TCGA-<CANCER>/pt_files
#   splits  : MCAT/splits/5foldcv/tcga_<cancer>/
#   csv     : MCAT/dataset_csv/tcga_<cancer>_all_clean.csv
#
# Array layout (50 jobs = 5 cancers x 5 folds x 2 models):
#   0-4:   BLCA   fold 0-4  (MCAT coattn)
#   5-9:   BRCA   fold 0-4  (MCAT coattn)
#   10-14: GBMLGG fold 0-4  (MCAT coattn)
#   15-19: KIRC   fold 0-4  (MCAT coattn)
#   20-24: LUAD   fold 0-4  (MCAT coattn)
#   25-29: BLCA   fold 0-4  (ABMIL path)
#   30-34: BRCA   fold 0-4  (ABMIL path)
#   35-39: GBMLGG fold 0-4  (ABMIL path)
#   40-44: KIRC   fold 0-4  (ABMIL path)
#   45-49: LUAD   fold 0-4  (ABMIL path)
#
# Submit:  sbatch benchmarks/submit_mcat_benchmark.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=mcat_bench
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%A_%a_mcat_bench.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%A_%a_mcat_bench.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -euo pipefail
export PYTHONUNBUFFERED=1
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

MIL_DIR="/lustre/groups/aih/dinesh.haridoss/mil"
MCAT_DIR="${MIL_DIR}/MCAT"
RESULTS_BASE="/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_tcga_competitors"
WANDB_PROJECT="chicago-mil-tcga-competitors"
mkdir -p "${RESULTS_BASE}"

# Our 5 cancers (UCEC removed vs. original sbatch_mcat.sh)
CANCERS=(BLCA BRCA GBMLGG KIRC LUAD)

IDX=$SLURM_ARRAY_TASK_ID
if [ $IDX -lt 25 ]; then
    MODEL_TYPE="mcat"
    MODE="coattn"
    CANCER_IDX=$((IDX / 5))
    FOLD=$((IDX % 5))
else
    MODEL_TYPE="amil"
    MODE="path"
    CANCER_IDX=$(((IDX - 25) / 5))
    FOLD=$(((IDX - 25) % 5))
fi

CANCER=${CANCERS[$CANCER_IDX]}
CANCER_LOWER=$(echo "$CANCER" | tr '[:upper:]' '[:lower:]')

echo "========================================"
echo "MCAT benchmark"
echo "Model: ${MODEL_TYPE}  Mode: ${MODE}  Cancer: ${CANCER}  Fold: ${FOLD}"
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started: $(date)"
echo "========================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python -m pip install -q scikit-survival tensorboardX || true

RESULTS_DIR="${RESULTS_BASE}/${MODEL_TYPE}_${CANCER_LOWER}_fold${FOLD}"
mkdir -p "${RESULTS_DIR}"

cd "${MCAT_DIR}"

# For AMIL (path-only), fusion must be None (no genomic features)
FUSION_ARG="concat"
if [ "${MODEL_TYPE}" == "amil" ]; then
    FUSION_ARG="None"
fi

python main.py \
    --direct_data_dir "${MIL_DIR}/uni2_pt/TCGA-${CANCER}" \
    --direct_csv_path "${MCAT_DIR}/dataset_csv/tcga_${CANCER_LOWER}_all_clean.csv" \
    --split_dir "tcga_${CANCER_LOWER}" \
    --which_splits "5foldcv" \
    --model_type "${MODEL_TYPE}" \
    --mode "${MODE}" \
    --wsi_input_dim 1536 \
    --fusion "${FUSION_ARG}" \
    --apply_sig \
    --k_start ${FOLD} \
    --k_end $((FOLD + 1)) \
    --results_dir "${RESULTS_DIR}" \
    --max_epochs 50 \
    --val_patience 10 \
    --lr 2e-4 \
    --bag_loss nll_surv \
    --reg 1e-5 \
    --drop_out \
    --weighted_sample \
    --seed 1 \
    --overwrite

echo "-------- logging to wandb: ${WANDB_PROJECT} --------"
export _WRES="${RESULTS_DIR}" _WMOD="${MODEL_TYPE}" _WCAN="${CANCER_LOWER}" _WFOLD="${FOLD}" _WPROJ="${WANDB_PROJECT}"
python3 - <<'PYEOF'
import os, wandb, pandas as pd, sys

results_dir  = os.environ["_WRES"]
model_type   = os.environ["_WMOD"]
cancer_lower = os.environ["_WCAN"]
fold         = int(os.environ["_WFOLD"])
project      = os.environ["_WPROJ"]

try:
    run = wandb.init(
        project=project,
        name=f"{model_type}_{cancer_lower}_fold{fold}",
        group=f"{model_type}_{cancer_lower}",
        config={"cancer": cancer_lower, "fold": fold, "model": model_type,
                "mode": "coattn" if model_type == "mcat" else "path",
                "wsi_input_dim": 1536, "max_epochs": 50, "lr": 2e-4},
        reinit=True,
    )
    csv_path = os.path.join(results_dir, "summary_latest.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        row = df[df["folds"] == fold]
        if len(row):
            val_ci = float(row["val_cindex"].values[0])
            run.log({"fold": fold, "val/cindex": val_ci})
            run.summary["val_cindex"] = val_ci
            print(f"  wandb logged val_cindex={val_ci:.4f}")
        else:
            print(f"  [warn] fold {fold} not in {csv_path}")
    else:
        print(f"  [warn] summary not found: {csv_path}")
    run.finish()
except Exception as e:
    print(f"  [wandb] error: {e}", file=sys.stderr)
PYEOF

echo "========================================"
echo "${MODEL_TYPE} ${CANCER} fold${FOLD} done  $(date)"
echo "========================================"
