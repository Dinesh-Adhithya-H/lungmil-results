#!/usr/bin/env bash
# =============================================================================
# submit_v7.sh  —  Triple-stream multimodal ABMIL v7 SLURM submission
# =============================================================================
#
# Submits Phase 2 training jobs for train_mm_abmil_v7.py.
# Phase 1 checkpoints are reused from an existing v6 results directory.
#
# Job layout:
#   Stage 0  (parallel)   One training job per split × fold
#                         Uses v6 Phase 1 encoders, adds centroid + count streams
#   Stage 1  (single)     Analysis / aggregation job after all training completes
#
# Missing-modality handling:
#   • Never zero-fills — streams are skipped/ignored when data absent
#   • If only 1 annotation modality has centroids → no centroid cross-attn,
#     just that modality's representation passed through
#   • If counts from 2 of 3 modalities present → they interact cross-modally;
#     missing modality simply not included in the attention stack
#   • Annotation stream dropped entirely with prob annot_dropout (default 0.2)
#
# Usage
# -----
#   bash submit_v7.sh --phase1_dir results_mm_abmil_v6       # required
#   bash submit_v7.sh --phase1_dir results_mm_abmil_v6 \
#       --splits "0 1" --folds "0 1"                         # subset
#   bash submit_v7.sh --phase1_dir ... --use_cls              # CLS-token pooling
#   bash submit_v7.sh --phase1_dir ... --annot_dropout 0.3   # higher annot dropout
#   bash submit_v7.sh --phase1_dir ... --dry_run              # print, don't submit
#
# =============================================================================
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="/home/aih/dinesh.haridoss/chicago_mil"
SAMPLES_DIR="/lustre/groups/aih/dinesh.haridoss/mil/dataset_cache_latest_fixed_large/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/train_mm_abmil_v7.py}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_mm_abmil_v7}"
LOG_DIR="${RESULTS_DIR}/slurm_logs"

# ── SLURM config ──────────────────────────────────────────────────────────────
PARTITION="gpu_p"
QOS="gpu_normal"
MAIL="dinesh.haridoss@helmholtz-munich.de"

TR_TIME="23:00:00";  TR_MEM="200G";  TR_CPUS=8
AN_TIME="04:00:00";  AN_MEM="200G";  AN_CPUS=8

# ── Training hyper-params ─────────────────────────────────────────────────────
MODAL_DROPOUT="0.2"
ANNOT_DROPOUT="0.2"
USE_CLS=""              # set to "--use_cls" to enable CLS-token pooling

# ── Pipeline control ──────────────────────────────────────────────────────────
SPLITS="0 1 2 3 4"
FOLDS="0 1 2 3"
PHASE1_DIR=""           # REQUIRED: path to v6 results containing Phase 1 ckpts
DRY_RUN=0

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase1_dir)    PHASE1_DIR="$2";      shift 2 ;;
        --splits)        SPLITS="$2";           shift 2 ;;
        --folds)         FOLDS="$2";            shift 2 ;;
        --results_dir)   RESULTS_DIR="$2"; LOG_DIR="${RESULTS_DIR}/slurm_logs"; shift 2 ;;
        --use_cls)       USE_CLS="--use_cls";   shift ;;
        --modal_dropout) MODAL_DROPOUT="$2";    shift 2 ;;
        --annot_dropout) ANNOT_DROPOUT="$2";    shift 2 ;;
        --dry_run)       DRY_RUN=1;             shift ;;
        *) echo "[error] Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ -z "$PHASE1_DIR" ]]; then
    echo "[error] --phase1_dir is required (path to v6 Phase 1 results)"
    exit 1
fi

IFS=' ' read -ra SPLIT_ARRAY <<< "$SPLITS"
IFS=' ' read -ra FOLD_ARRAY  <<< "$FOLDS"

mkdir -p "$LOG_DIR"

# ── Conda activation ──────────────────────────────────────────────────────────
CONDA_INIT='
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago
python -c "import torch; print(f\"PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}\")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
'

# ── Helpers ───────────────────────────────────────────────────────────────────
split_fold_tag() { echo "split${1}_fold${2}"; }

submit_job() {
    local name="$1" dep="$2" script="$3"
    local dep_flag=""
    [[ -n "$dep" ]] && dep_flag="--dependency=${dep} --kill-on-invalid-dep=yes"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry] Would submit: ${name}  dep=${dep:-none}" >&2
        echo "DRY_${name}"; return
    fi

    local jid
    jid=$(sbatch --parsable $dep_flag <<< "$script")
    echo "[submit] ${name}  →  JID=${jid}  dep=${dep:-none}" >&2
    echo "$jid"
}

# ── Header ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  submit_v7.sh  —  Multimodal ABMIL v7 (Triple-stream)"
echo "============================================================"
echo "  Train script   : ${TRAIN_SCRIPT}"
echo "  Phase 1 dir    : ${PHASE1_DIR}"
echo "  Results dir    : ${RESULTS_DIR}"
echo "  Splits         : ${SPLITS}"
echo "  Folds          : ${FOLDS}"
echo "  Annot dropout  : ${ANNOT_DROPOUT}"
echo "  Modal dropout  : ${MODAL_DROPOUT}"
echo "  CLS pooling    : ${USE_CLS:-off}"
echo "  Partition/QOS  : ${PARTITION} / ${QOS}"
echo "  Dry run        : ${DRY_RUN}"
echo "============================================================"
echo ""
echo "  Missing-modality policy: skip / ignore — never zero-fill."
echo "  Counts/centroids from partial modality sets interact only"
echo "  with each other (cross-attn requires >=2); otherwise each"
echo "  modality is used individually without padding."
echo "============================================================"
echo ""

# ═════════════════════════════════════════════════════════════════
# STAGE 0  —  One training job per split × fold
# ═════════════════════════════════════════════════════════════════
echo "── Stage 0: V7 Training ────────────────────────────────────"

declare -A TR_JIDS

for split in "${SPLIT_ARRAY[@]}"; do
    for fold in "${FOLD_ARRAY[@]}"; do
        TAG=$(split_fold_tag "$split" "$fold")
        JOB_NAME="v7_s${split}f${fold}"

        SCRIPT=$(cat << SCRIPT_END
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${TR_CPUS}
#SBATCH --mem=${TR_MEM}
#SBATCH --gres=gpu:1
#SBATCH --time=${TR_TIME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${MAIL}

set -euo pipefail
${CONDA_INIT}

echo "==== V7 Triple-stream | split=${split} fold=${fold} | \$(date) ===="
echo "Node: \$(hostname)  JID: \${SLURM_JOB_ID}"
echo "Phase 1 dir: ${PHASE1_DIR}"
echo ""
echo "Missing-modality policy: skip / ignore (no zero-filling)."
echo "  Bag stream   : modalities present in sample only"
echo "  Centroid stream: only mods with valid centroids+counts interact"
echo "  Count stream : only mods with valid counts embedded; if >=2,"
echo "                 cross-attn applied; if 1, used directly; if 0, skipped."

python3 -u "${TRAIN_SCRIPT}" --phase1_dir "${PHASE1_DIR}" --save_dir "${RESULTS_DIR}" --samples_dir "${SAMPLES_DIR}" --splits_csv "${SPLITS_CSV}" --split ${split} --folds ${fold} --modal_dropout ${MODAL_DROPOUT} --annot_dropout ${ANNOT_DROPOUT} ${USE_CLS}

echo "==== Done V7 | split=${split} fold=${fold} | \$(date) ===="
SCRIPT_END
        )

        TR_JIDS[$TAG]=$(submit_job "$JOB_NAME" "" "$SCRIPT")
    done
done
echo ""

# ═════════════════════════════════════════════════════════════════
# STAGE 1  —  Analysis (after all training jobs complete)
# ═════════════════════════════════════════════════════════════════
echo "── Stage 1: Analysis ───────────────────────────────────────"

AN_DEP=""
if [[ "$DRY_RUN" -eq 0 ]]; then
    VALID=()
    for TAG in "${!TR_JIDS[@]}"; do
        jid="${TR_JIDS[$TAG]:-}"
        [[ -n "$jid" ]] && VALID+=("$jid")
    done
    [[ "${#VALID[@]}" -gt 0 ]] && AN_DEP="afterok:$(IFS=':'; echo "${VALID[*]}")"
elif [[ "$DRY_RUN" -eq 1 ]]; then
    AN_DEP="afterok:ALL_V7_JOBS"
fi

FOLDS_STR="$FOLDS"

AN_SCRIPT=$(cat << SCRIPT_END
#!/usr/bin/env bash
#SBATCH --job-name=v7_analyze
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${AN_CPUS}
#SBATCH --mem=${AN_MEM}
#SBATCH --gres=gpu:0
#SBATCH --time=${AN_TIME}
#SBATCH --output=${LOG_DIR}/v7_analyze_%j.out
#SBATCH --error=${LOG_DIR}/v7_analyze_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL}

set -euo pipefail
${CONDA_INIT}

echo "==== V7 Analysis | \$(date) ===="
echo "Node: \$(hostname)  JID: \${SLURM_JOB_ID}"

# Aggregate per-fold metrics into a cross-validation summary
python3 -u - <<'PYEOF'
import json, sys, pathlib, numpy as np

results_dir = pathlib.Path("${RESULTS_DIR}")
splits      = [int(s) for s in "${SPLITS}".split()]
folds       = [int(f) for f in "${FOLDS_STR}".split()]

tag = "v7_triple${USE_CLS:+_cls}"

rows = []
for split in splits:
    for fold in folds:
        fold_tag = f"split{split}_fold{fold}"
        mf = results_dir / fold_tag / f"metrics_{tag}.json"
        if not mf.exists():
            print(f"  [warn] missing {mf}")
            continue
        with open(mf) as fh:
            m = json.load(fh)
        row = {"split": split, "fold": fold, "fold_tag": fold_tag}
        for sn in ("train", "val", "test"):
            sm = m.get(sn, {})
            for k in ("auc", "auprc", "bacc", "mcc", "sens", "spec"):
                row[f"{sn}_{k}"] = sm.get(k, float("nan"))
        rows.append(row)

if not rows:
    print("[error] No result files found. Check job logs.")
    sys.exit(1)

# Summary table
def _fmt(vals):
    v = [x for x in vals if not np.isnan(x)]
    if not v: return " N/A "
    return f"{np.mean(v):.4f}±{np.std(v):.4f}"

print(f"\n{'='*70}")
print(f"  V7 Triple-stream — Cross-validation summary")
print(f"  n_folds={len(rows)}")
print(f"{'='*70}")
metrics = ["auc", "auprc", "bacc", "mcc", "sens", "spec"]
for sn in ("val", "test"):
    print(f"\n  [{sn.upper()}]")
    for k in metrics:
        vals = [r[f"{sn}_{k}"] for r in rows]
        print(f"    {k:6s}: {_fmt(vals)}")

out = results_dir / "cv_summary_v7.json"
with open(out, "w") as fh:
    json.dump(rows, fh, indent=2)
print(f"\n  Full per-fold results written to: {out}")
PYEOF

echo "==== V7 Analysis done | \$(date) ===="
SCRIPT_END
)

AN_JID=$(submit_job "v7_analyze" "$AN_DEP" "$AN_SCRIPT")

# ═════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════
N_TR=$(( ${#SPLIT_ARRAY[@]} * ${#FOLD_ARRAY[@]} ))

echo ""
echo "============================================================"
echo "  Job graph"
echo "============================================================"
echo "  Training jobs : ${N_TR}  (1 per split×fold)"
echo "  Analysis job  : 1  (after all training)"
echo "  Total         : $(( N_TR + 1 ))"
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "  [DRY RUN] No jobs submitted."
else
    echo "  Training JIDs:"
    for split in "${SPLIT_ARRAY[@]}"; do
        for fold in "${FOLD_ARRAY[@]}"; do
            TAG=$(split_fold_tag "$split" "$fold")
            echo "    s${split}f${fold} → ${TR_JIDS[$TAG]:-—}"
        done
    done
    echo ""
    echo "  Analysis JID : ${AN_JID:-—}"
    echo ""
    echo "  Monitor:"
    echo "    squeue -u \$USER"
    echo "    watch -n 60 squeue -u \$USER"
    echo ""
    echo "  Live logs:"
    echo "    tail -f ${LOG_DIR}/v7_s0f0_*.out"
    echo "    tail -f ${LOG_DIR}/v7_analyze_*.out"
    echo ""
    echo "  Cancel everything:"
    ALL_JIDS=()
    for TAG in "${!TR_JIDS[@]}"; do
        [[ -n "${TR_JIDS[$TAG]:-}" ]] && ALL_JIDS+=("${TR_JIDS[$TAG]}")
    done
    [[ -n "${AN_JID:-}" ]] && ALL_JIDS+=("$AN_JID")
    echo "    scancel ${ALL_JIDS[*]}"
fi
echo "============================================================"
