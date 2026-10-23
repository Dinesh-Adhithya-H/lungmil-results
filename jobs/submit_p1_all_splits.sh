#!/usr/bin/env bash
# submit_p1_all_splits.sh — 5 unimodal ABMIL jobs, one per split.
# Each job: HP sweep on inner folds (1-3), then combined training on fold 0.
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

JOB="${SCRIPTS_DIR}/p1_split${SPLIT}.sh"

cat > "${JOB}" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=p1_s${SPLIT}
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=220G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=${LOG_DIR}/%j_p1_s${SPLIT}.out
#SBATCH --error=${LOG_DIR}/%j_p1_s${SPLIT}.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL}

set -uo pipefail
export PYTHONUNBUFFERED=1

SPLIT=${SPLIT}
SCRIPT_PATH="\$(realpath "\$0")"

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "=============================="
echo " Unimodal ABMIL  split=\${SPLIT}  job=\${SLURM_JOB_ID}"
echo " host=\$(hostname)  gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo " started: \$(date)"
echo "=============================="

RESUBMIT=0
PY_PID=""

walltime_handler() {
    echo "[SIGUSR1] Wall-time approaching — resubmitting..."
    RESUBMIT=1
    [[ -n "\${PY_PID}" ]] && kill -SIGTERM "\${PY_PID}" 2>/dev/null || true
    sleep 10
    kill -SIGKILL "\${PY_PID}" 2>/dev/null || true
}
trap walltime_handler SIGUSR1

run_p1_fold() {
    local fold=\$1
    local extra_flags="\${@:2}"
    local done_file="${OUT_DIR}/phase1/split\${SPLIT}_fold\${fold}/.p1_done_all"

    if [[ -f "\${done_file}" ]]; then
        echo "  [skip] p1 split\${SPLIT} fold\${fold} — already done"
        return 0
    fi

    echo ""
    echo "--- p1 split\${SPLIT} fold\${fold}  \$(date) ---"

    python3 -u "${TRAIN}" \\
        --samples-dir  "${SAMPLES}"    \\
        --splits-csv   "${SPLITS_CSV}" \\
        --split        \${SPLIT}       \\
        --fold         \${fold}        \\
        --out-dir      "${OUT_DIR}"    \\
        --phase        p1              \\
        --p1-tasks     all             \\
        --workers      8               \\
        \${extra_flags} &
    PY_PID=\$!
    wait "\${PY_PID}"
    local exit_code=\$?
    PY_PID=""

    if [[ "\${RESUBMIT}" -eq 1 ]]; then return 1; fi
    if [[ "\${exit_code}" -eq 0 ]]; then
        touch "\${done_file}"
    else
        echo "  [warn] python exited \${exit_code} for split\${SPLIT} fold\${fold}"
    fi
    return 0
}

# Inner folds: HP sweep
for fold in 1 2 3; do
    run_p1_fold \${fold} --hp-sweep || {
        if [[ "\${RESUBMIT}" -eq 1 ]]; then
            NEW_JID=\$(sbatch --parsable "\${SCRIPT_PATH}")
            echo "Resubmitted as \${NEW_JID}"
            exit 0
        fi
    }
done

# Fold 0: HP sweep on fold 0 val, then aggregate HP from all 4 folds, then combined train
run_p1_fold 0 --hp-sweep --p1-global-hp --p1-combined-tag combined || {
    if [[ "\${RESUBMIT}" -eq 1 ]]; then
        NEW_JID=\$(sbatch --parsable "\${SCRIPT_PATH}")
        echo "Resubmitted as \${NEW_JID}"
        exit 0
    fi
}

echo ""
echo "=============================="
echo " Unimodal ABMIL split=\${SPLIT} ALL DONE: \$(date)"
echo "=============================="
HEREDOC

chmod +x "${JOB}"
JID=$(sbatch --parsable "${JOB}")
echo "p1 split${SPLIT} → job ${JID}"

done

echo ""
echo "All 5 unimodal jobs submitted."
