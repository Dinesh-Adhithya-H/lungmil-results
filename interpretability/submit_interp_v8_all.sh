#!/usr/bin/env bash
# submit_interp_v8_all.sh — interpretability for all v8 P2 models, split 0 fold 0
# Usage: bash interpretability/submit_interp_v8_all.sh
set -euo pipefail

REPO="/ictstr01/home/aih/dinesh.haridoss/chicago_mil"
RESULTS="${REPO}/results/mm_abmil_v8"
P2="${RESULTS}/phase2/split0_fold0"
P1_DIR="${RESULTS}/phase1/split0_fold0"
INTERP="${REPO}/interpretability"
OUT_BASE="${INTERP}/v8_interp_s0f0"
LOGS="${RESULTS}/slurm_logs"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
SPLIT=0
FOLD=0

mkdir -p "${LOGS}"

# ── early / late / middle: one job per variant×task ──────────────────────────
# variant → (v6_base, task_flag, surv_endpoint)
declare -A TASK_FLAG=(
  [cls]="acr"
  [acr_surv]="survival surv_endpoint acr"
  [clad_surv]="survival surv_endpoint clad"
  [death_surv]="survival surv_endpoint death"
)

for ARCH in early late middle; do
  for TASK_KEY in cls acr_surv clad_surv death_surv; do
    VDIR="${ARCH}_${TASK_KEY}"
    CKPT="${P2}/${VDIR}/model_${ARCH}_final.pt"
    OUT="${OUT_BASE}/${VDIR}"

    if [ ! -f "${CKPT}" ]; then
      echo "[skip] missing checkpoint: ${CKPT}"
      continue
    fi

    # Build task flags
    if [ "${TASK_KEY}" = "cls" ]; then
      TASK_ARGS="--task acr"
    elif [ "${TASK_KEY}" = "acr_surv" ]; then
      TASK_ARGS="--task survival --surv_endpoint acr"
    elif [ "${TASK_KEY}" = "clad_surv" ]; then
      TASK_ARGS="--task survival --surv_endpoint clad"
    else
      TASK_ARGS="--task survival --surv_endpoint death"
    fi

    JOB="interp_${ARCH}_${TASK_KEY}_s0f0"
    sbatch --job-name="${JOB}" \
           --partition=gpu_p \
           --qos=gpu_normal \
           --gres=gpu:1 \
           --cpus-per-task=8 \
           --mem=80G \
           --time=02:00:00 \
           --output="${LOGS}/%j_${JOB}.out" \
           --error="${LOGS}/%j_${JOB}.err" \
           --mail-type=FAIL \
           --mail-user=dinesh.haridoss@helmholtz-munich.de \
           --wrap="source \$(conda info --base)/etc/profile.d/conda.sh && conda activate chicago && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
python3 -u ${INTERP}/interpret_mm_abmil.py \
  --version v6 \
  --results_dir ${RESULTS} \
  --split ${SPLIT} --fold ${FOLD} \
  --v6_variant ${ARCH} \
  --ckpt ${CKPT} \
  --p1_dir ${P1_DIR} \
  --samples_dir ${SAMPLES} \
  --splits_csv ${SPLITS_CSV} \
  --split_set test \
  --n_sample_plots 20 \
  --out_dir ${OUT} \
  ${TASK_ARGS}"
    echo "Submitted ${JOB}"
  done
done

# ── set_mil_mt: one job per task ──────────────────────────────────────────────
for TASK_KEY in cls acr_surv clad_surv death_surv; do
  VDIR="set_mil_mt_${TASK_KEY}"
  CKPT="${P2}/${VDIR}/model_set_mil_mt_final.pt"

  if [ ! -f "${CKPT}" ]; then
    echo "[skip] missing checkpoint: ${CKPT}"
    continue
  fi

  JOB="interp_smmt_${TASK_KEY}_s0f0"
  sbatch --job-name="${JOB}" \
         --partition=gpu_p \
         --qos=gpu_normal \
         --gres=gpu:1 \
         --cpus-per-task=8 \
         --mem=80G \
         --time=02:00:00 \
         --output="${LOGS}/%j_${JOB}.out" \
         --error="${LOGS}/%j_${JOB}.err" \
         --mail-type=FAIL \
         --mail-user=dinesh.haridoss@helmholtz-munich.de \
         --wrap="source \$(conda info --base)/etc/profile.d/conda.sh && conda activate chicago && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
python3 -u ${INTERP}/interpret_set_mil_mt.py \
  --split ${SPLIT} \
  --fold ${FOLD} \
  --variant ${TASK_KEY} \
  --out-dir ${OUT_BASE}/set_mil_mt_${TASK_KEY}"
  echo "Submitted ${JOB}"
done

echo ""
echo "All interpretability jobs submitted → outputs under ${OUT_BASE}"
