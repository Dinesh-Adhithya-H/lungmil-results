#!/usr/bin/env bash
# Submit LongMIL no-PMA ablation (mean pool instead of PMA) on full cohort, all 5 splits.
# Uses longitudinal_mk_no_alibi_v2_no_pma variant; otherwise matches nosab config.
set -euo pipefail

HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
OUT_DIR="${HOME_MIL}/results/mm_abmil_v8_no_pma"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
LOG_DIR="${OUT_DIR}/slurm_logs"
SCRIPTS_DIR="${OUT_DIR}/job_scripts"
MAIL="dinesh.haridoss@helmholtz-munich.de"

mkdir -p "${LOG_DIR}" "${SCRIPTS_DIR}"

for SPLIT in 0 1 2 3 4; do
JOB="${SCRIPTS_DIR}/split${SPLIT}.sh"
cat > "${JOB}" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=nopma_s${SPLIT}
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=220G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=10:00:00
#SBATCH --output=${LOG_DIR}/%j_split${SPLIT}.out
#SBATCH --error=${LOG_DIR}/%j_split${SPLIT}.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL}

set -uo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
cd ${HOME_MIL}

TASKS=(cls acr_surv clad_surv death_surv)
TASK_LRS=(1e-05 1e-04 1e-04 5e-05)

for i in "\${!TASKS[@]}"; do
  TASK="\${TASKS[\$i]}"; LR="\${TASK_LRS[\$i]}"
  RESULT="${OUT_DIR}/metrics_split${SPLIT}_fold0_longitudinal_mk_no_alibi_v2_no_pma_\${TASK}.json"
  if [[ -f "\${RESULT}" ]]; then echo "[skip] split${SPLIT} \${TASK}"; continue; fi
  echo "--- split${SPLIT} \${TASK} lr=\${LR} \$(date) ---"
  python3 -u train_mm_abmil_v8.py \
    --samples-dir ${SAMPLES} \
    --splits-csv  ${SPLITS_CSV} \
    --split       ${SPLIT} \
    --fold        0 \
    --out-dir     ${OUT_DIR} \
    --phase       p2 \
    --p2-variant  longitudinal_mk_no_alibi_v2_no_pma \
    --slot-k      8 \
    --n-cross-layers 0 \
    --task        "\${TASK}" \
    --p2-lr       "\${LR}" \
    --p2-wd       1e-4 \
    --combined-train \
    --modal-dropout 0.3 \
    --workers     8
done
echo "split${SPLIT} ALL DONE: \$(date)"
HEREDOC
chmod +x "${JOB}"
JID=$(sbatch --parsable "${JOB}")
echo "split${SPLIT}  →  job ${JID}"
done

echo ""
echo "All 5 splits submitted. Monitor: squeue -u \$USER"
