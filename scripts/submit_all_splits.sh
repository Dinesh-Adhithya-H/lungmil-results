#!/usr/bin/env bash
# submit_all_splits.sh
# Generates and submits one GPU job per split (split0–split4).
# Each job runs 4 folds × (early/late/middle/slot) × tasks with per-fold HP sweep.
# Self-resubmits on wall-time to survive across allocations.
set -euo pipefail

HOME_MIL="/home/aih/dinesh.haridoss/chicago_mil"
TRAIN="${HOME_MIL}/train_mm_abmil_v8.py"
SAMPLES="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
OUT_DIR="${HOME_MIL}/results/mm_abmil_v8"
LOG_DIR="${HOME_MIL}/results_mm_abmil_v8/slurm_logs"
SCRIPTS_DIR="${HOME_MIL}/results_mm_abmil_v8/job_scripts"
MAIL="dinesh.haridoss@helmholtz-munich.de"

mkdir -p "${LOG_DIR}" "${SCRIPTS_DIR}"

for SPLIT in 0 1 2 3 4; do

JOB="${SCRIPTS_DIR}/split${SPLIT}_all_folds.sh"

cat > "${JOB}" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=v8_split${SPLIT}
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=220G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=${LOG_DIR}/%j_split${SPLIT}.out
#SBATCH --error=${LOG_DIR}/%j_split${SPLIT}.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL}

set -uo pipefail
export PYTHONUNBUFFERED=1

SPLIT=${SPLIT}
HOME_MIL="${HOME_MIL}"
TRAIN="${TRAIN}"
SAMPLES="${SAMPLES}"
SPLITS_CSV="${SPLITS_CSV}"
OUT_DIR="${OUT_DIR}"
LOG_DIR="${LOG_DIR}"
SCRIPT_PATH="\$(realpath "\$0")"

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "======================================"
echo " v8 split=\${SPLIT}  job=\${SLURM_JOB_ID}"
echo " host=\$(hostname)  gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo " started: \$(date)"
echo "======================================"

RESUBMIT=0
PY_PID=""

walltime_handler() {
    echo "[SIGUSR1] Wall-time approaching — stopping and resubmitting..."
    RESUBMIT=1
    [[ -n "\${PY_PID}" ]] && kill -SIGTERM "\${PY_PID}" 2>/dev/null || true
    sleep 10
    kill -SIGKILL "\${PY_PID}" 2>/dev/null || true
}
trap walltime_handler SIGUSR1

run_one() {
    local fold=\$1 variant=\$2 task=\$3
    local tag="split\${SPLIT}_fold\${fold}"
    local result_file="\${OUT_DIR}/metrics_\${tag}_\${variant}_\${task}.json"

    if [[ -f "\${result_file}" ]]; then
        echo "  [skip] \${tag} \${variant} \${task} — already done"
        return 0
    fi

    echo ""
    echo "--- \${tag}  variant=\${variant}  task=\${task}  \$(date) ---"

    python3 -u "\${TRAIN}" \\
        --samples-dir    "\${SAMPLES}"    \\
        --splits-csv     "\${SPLITS_CSV}" \\
        --split          "\${SPLIT}"      \\
        --fold           "\${fold}"       \\
        --out-dir        "\${OUT_DIR}"    \\
        --phase          both             \\
        --p2-variant     "\${variant}"    \\
        --slot-k         128              \\
        --n-cross-layers 4               \\
        --task           "\${task}"       \\
        --p1-tasks       acr             \\
        --p2-hp-sweep                    \\
        --combined-train                 \\
        --modal-dropout  0.3             \\
        --workers        8               \\
        &
    PY_PID=\$!
    wait "\${PY_PID}"
    local exit_code=\$?
    PY_PID=""

    if [[ "\${RESUBMIT}" -eq 1 ]]; then
        return 1
    fi
    if [[ "\${exit_code}" -ne 0 ]]; then
        echo "  [warn] python exited \${exit_code} for \${tag} \${variant} \${task}"
    fi
    return 0
}

# Variant × task matrix
# early/late/middle run all 4 clinical tasks; slot runs mega only
FOLDS=(0 1 2 3)
BASE_VARIANTS=(early late middle)
BASE_TASKS=(cls acr_surv clad_surv death_surv)

for fold in "\${FOLDS[@]}"; do
    for variant in "\${BASE_VARIANTS[@]}"; do
        for task in "\${BASE_TASKS[@]}"; do
            run_one "\${fold}" "\${variant}" "\${task}" || {
                if [[ "\${RESUBMIT}" -eq 1 ]]; then
                    echo "Resubmitting split=\${SPLIT}..."
                    NEW_JID=\$(sbatch --parsable "\${SCRIPT_PATH}")
                    echo "New job: \${NEW_JID}"
                    exit 0
                fi
            }
        done
    done
    # Slot-MIL (mega task)
    run_one "\${fold}" slot mega || {
        if [[ "\${RESUBMIT}" -eq 1 ]]; then
            echo "Resubmitting split=\${SPLIT}..."
            NEW_JID=\$(sbatch --parsable "\${SCRIPT_PATH}")
            echo "New job: \${NEW_JID}"
            exit 0
        fi
    }
done

echo ""
echo "======================================"
echo " split=\${SPLIT} ALL DONE: \$(date)"
echo "======================================"
HEREDOC

chmod +x "${JOB}"
JID=$(sbatch --parsable "${JOB}")
echo "split${SPLIT}  →  job ${JID}  (${JOB})"

done

echo ""
echo "All 5 splits submitted. Monitor: squeue -u \$USER"
