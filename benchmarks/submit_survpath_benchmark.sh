#!/bin/bash
# submit_survpath_benchmark.sh
# ---------------------------------------------------------------------------
# SurvPath (Jaume et al. 2023) + MCAT(coattn) baselines on TCGA with UNI-2
# (1536-dim) WSI features, using the official SurvPath repo on Lustre —
# NO code changes, just a wrapper that runs `python main.py`.
#
# Supported cancers:
#   Of our 5 target cancers (BLCA BRCA GBMLGG KIRC LUAD), SurvPath only ships
#   splits + xena RNA pathway data for BLCA and BRCA
#   (splits/5foldcv/ contains: blca, brca, coadread, hnsc, stad — GBMLGG/KIRC/
#    LUAD have no SurvPath splits or xena RNA, so they are skipped here).
#
# Uses existing assets on Lustre:
#   repo    : /lustre/groups/aih/dinesh.haridoss/mil/SurvPath
#   WSI feats: /lustre/groups/aih/dinesh.haridoss/mil/uni2_pt/TCGA-<CANCER>/pt_files
#   labels  : SurvPath/datasets_csv/metadata/tcga_<cancer>.csv
#   omics   : SurvPath/datasets_csv/raw_rna_data/xena/<cancer>/
#   splits  : SurvPath/splits/5foldcv/tcga_<cancer>/
#
# Array layout (20 jobs = 2 cancers x 5 folds x 2 models):
#   0-4:   BLCA  fold 0-4  (survpath)
#   5-9:   BRCA  fold 0-4  (survpath)
#   10-14: BLCA  fold 0-4  (coattn / MCAT via SurvPath)
#   15-19: BRCA  fold 0-4  (coattn / MCAT via SurvPath)
#
# Submit:  sbatch benchmarks/submit_survpath_benchmark.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=survpath_bench
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%A_%a_survpath_bench.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%A_%a_survpath_bench.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -euo pipefail
export PYTHONUNBUFFERED=1
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

MIL_DIR="/lustre/groups/aih/dinesh.haridoss/mil"
SURVPATH_DIR="${MIL_DIR}/SurvPath"
RESULTS_BASE="/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_tcga_competitors"
WANDB_PROJECT="chicago-mil-tcga-competitors"
mkdir -p "${RESULTS_BASE}"

# Only cancers of ours that SurvPath supports (have splits + xena RNA)
CANCERS=(BLCA BRCA)

IDX=$SLURM_ARRAY_TASK_ID
if [ $IDX -lt 10 ]; then
    MODALITY="survpath"
    CANCER_IDX=$((IDX / 5))
    FOLD=$((IDX % 5))
else
    MODALITY="coattn"
    CANCER_IDX=$(((IDX - 10) / 5))
    FOLD=$(((IDX - 10) % 5))
fi

CANCER=${CANCERS[$CANCER_IDX]}
CANCER_LOWER=$(echo "$CANCER" | tr '[:upper:]' '[:lower:]')

echo "========================================"
echo "SurvPath benchmark"
echo "Model: ${MODALITY}  Cancer: ${CANCER}  Fold: ${FOLD}"
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started: $(date)"
echo "========================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python -m pip install -q scikit-survival transformers || true

DATA_DIR="${MIL_DIR}/uni2_pt/TCGA-${CANCER}/pt_files"
LABEL_FILE="${SURVPATH_DIR}/datasets_csv/metadata/tcga_${CANCER_LOWER}.csv"
OMICS_DIR="${SURVPATH_DIR}/datasets_csv/raw_rna_data/xena/${CANCER_LOWER}"
RESULTS_DIR="${RESULTS_BASE}/${MODALITY}_${CANCER_LOWER}_fold${FOLD}"
mkdir -p "${RESULTS_DIR}"

cd "${SURVPATH_DIR}"

python main.py \
    --study "tcga_${CANCER_LOWER}" \
    --task survival \
    --modality "${MODALITY}" \
    --wsi_input_dim 1536 \
    --data_root_dir "${DATA_DIR}" \
    --label_file "${LABEL_FILE}" \
    --omics_dir "${OMICS_DIR}" \
    --which_splits 5foldcv \
    --type_of_path xena \
    --label_col "survival_months" \
    --bag_loss nll_surv \
    --fusion concat \
    --n_classes 4 \
    --k 1 \
    --k_start ${FOLD} \
    --k_end $((FOLD + 1)) \
    --results_dir "${RESULTS_DIR}" \
    --max_epochs 50 \
    --val_patience 10 \
    --lr 1e-4 \
    --seed 1 \
    --weighted_sample

echo "-------- logging to wandb: ${WANDB_PROJECT} --------"
export _WRES="${RESULTS_DIR}" _WMOD="${MODALITY}" _WCAN="${CANCER_LOWER}" _WFOLD="${FOLD}" _WPROJ="${WANDB_PROJECT}"
python3 - <<'PYEOF'
import os, wandb, pandas as pd, sys

results_dir  = os.environ["_WRES"]
modality     = os.environ["_WMOD"]
cancer_lower = os.environ["_WCAN"]
fold         = int(os.environ["_WFOLD"])
project      = os.environ["_WPROJ"]

try:
    run = wandb.init(
        project=project,
        name=f"{modality}_{cancer_lower}_fold{fold}",
        group=f"{modality}_{cancer_lower}",
        config={"cancer": cancer_lower, "fold": fold, "model": modality,
                "wsi_input_dim": 1536, "max_epochs": 50, "lr": 1e-4},
        reinit=True,
    )
    # SurvPath writes summary.csv (or summary_partial_<k_start>_<k_end>.csv for partial runs)
    csv_path = os.path.join(results_dir, "summary.csv")
    if not os.path.exists(csv_path):
        import glob
        partials = glob.glob(os.path.join(results_dir, "summary_partial_*.csv"))
        csv_path = partials[0] if partials else csv_path
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        row = df[df["folds"] == fold] if "folds" in df.columns else df
        if len(row):
            metrics = {}
            for col in ["val_cindex", "val_cindex_ipcw", "val_IBS", "val_iauc", "val_loss"]:
                if col in row.columns:
                    metrics[f"val/{col.replace('val_','')}"] = float(row[col].values[0])
            run.log({"fold": fold, **metrics})
            run.summary.update(metrics)
            print(f"  wandb logged: {metrics}")
        else:
            print(f"  [warn] fold {fold} not in {csv_path}")
    else:
        print(f"  [warn] summary not found: {csv_path}")
    run.finish()
except Exception as e:
    print(f"  [wandb] error: {e}", file=sys.stderr)
PYEOF

echo "========================================"
echo "${MODALITY} ${CANCER} fold${FOLD} done  $(date)"
echo "========================================"
