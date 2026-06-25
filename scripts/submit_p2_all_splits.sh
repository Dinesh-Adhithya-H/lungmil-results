#!/usr/bin/env bash
# submit_p2_all_splits.sh — 20 parallel jobs (5 splits × 4 variants).
# Each job handles one variant across all 4 folds: HP sweep (1-3) then combined (0).
# Skip logic via metrics JSON means already-done runs are never repeated.
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
for VARIANT in early late middle mario_kempes; do

# mario_kempes runs a single mega task; others run 4 tasks
if [[ "${VARIANT}" == "mario_kempes" ]]; then
    TASKS="mega"
else
    TASKS="cls acr_surv clad_surv death_surv"
fi

JOB="${SCRIPTS_DIR}/p2_s${SPLIT}_${VARIANT}.sh"

cat > "${JOB}" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=p2_s${SPLIT}_${VARIANT}
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=220G
#SBATCH --gres=gpu:1 --constraint="a100_80gb|h100_80gb"
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=${LOG_DIR}/%j_p2_s${SPLIT}_${VARIANT}.out
#SBATCH --error=${LOG_DIR}/%j_p2_s${SPLIT}_${VARIANT}.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${MAIL}

set -uo pipefail
export PYTHONUNBUFFERED=1

SPLIT=${SPLIT}
VARIANT=${VARIANT}
TASKS="${TASKS}"
SCRIPT_PATH="\$(realpath "\$0")"

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

echo "=============================="
echo " P2  split=\${SPLIT}  variant=\${VARIANT}  job=\${SLURM_JOB_ID}"
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

run_one() {
    local fold=\$1 task=\$2
    local tag="split\${SPLIT}_fold\${fold}"
    local result_file="${OUT_DIR}/metrics_\${tag}_\${VARIANT}_\${task}.json"

    if [[ -f "\${result_file}" ]]; then
        echo "  [skip] \${tag} \${VARIANT} \${task} — already done"
        return 0
    fi

    echo ""
    echo "--- \${tag}  variant=\${VARIANT}  task=\${task}  \$(date) ---"

    # All folds run HP sweep on their own val set.
    # Fold 0: additionally aggregates HP across all 4 folds (0-3) via --global-hp,
    # then trains on train+val combined with that globally-selected HP.
    if [[ \${fold} -eq 0 ]]; then
        HP_FLAGS="--p2-hp-sweep --global-hp --combined-train"
    else
        HP_FLAGS="--p2-hp-sweep --combined-train"
    fi

    python3 -u "${TRAIN}" \\
        --samples-dir    "${SAMPLES}"    \\
        --splits-csv     "${SPLITS_CSV}" \\
        --split          \${SPLIT}       \\
        --fold           \${fold}        \\
        --out-dir        "${OUT_DIR}"    \\
        --phase          p2              \\
        --p2-variant     "\${VARIANT}"   \\
        --slot-k         16              \\
        --n-cross-layers 1              \\
        --task           "\${task}"      \\
        \${HP_FLAGS}                     \\
        --modal-dropout  0.3            \\
        --workers        8              \\
        &
    PY_PID=\$!
    wait "\${PY_PID}"
    local exit_code=\$?
    PY_PID=""

    if [[ "\${RESUBMIT}" -eq 1 ]]; then return 1; fi
    if [[ "\${exit_code}" -ne 0 ]]; then
        echo "  [warn] python exited \${exit_code} for \${tag} \${VARIANT} \${task}"
    fi
    return 0
}

# HP sweep: folds 1-3
for fold in 1 2 3; do
    for task in \${TASKS}; do
        run_one \${fold} \${task} || {
            if [[ "\${RESUBMIT}" -eq 1 ]]; then
                NEW_JID=\$(sbatch --parsable "\${SCRIPT_PATH}")
                echo "Resubmitted as \${NEW_JID}"
                exit 0
            fi
        }
    done
done

# Combined training: fold 0
for task in \${TASKS}; do
    run_one 0 \${task} || {
        if [[ "\${RESUBMIT}" -eq 1 ]]; then
            NEW_JID=\$(sbatch --parsable "\${SCRIPT_PATH}")
            echo "Resubmitted as \${NEW_JID}"
            exit 0
        fi
    }
done

echo ""
echo "=============================="
echo " P2 split=\${SPLIT} variant=\${VARIANT} ALL DONE: \$(date)"
echo "=============================="
HEREDOC

chmod +x "${JOB}"
JID=$(sbatch --parsable "${JOB}")
echo "p2 split${SPLIT} ${VARIANT} → job ${JID}"

done
done

echo ""
echo "All 20 P2 jobs submitted."
