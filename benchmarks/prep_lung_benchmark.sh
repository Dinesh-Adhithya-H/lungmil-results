#!/bin/bash
# prep_lung_benchmark.sh
# ---------------------------------------------------------------------------
# Prepare lung transplant data for MCAT / MOTCAT benchmarks:
#   1. Extract HE patch tensors → benchmarks/lung_mcat_data/pt_files/{stem}.pt
#   2. Write per-task CSV (case_id, slide_id, survival_months, censorship, oncotree_code)
#   3. Write 5 split files per task (splits_0..4.csv) with train=train+val, val=test
#   4. Symlink CSV files into MCAT and MOTCAT dataset directories
#
# Run: sbatch benchmarks/prep_lung_benchmark.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=prep_lung_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_prep_lung_bench.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_prep_lung_bench.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO="/ictstr01/home/aih/dinesh.haridoss/chicago_mil"
MIL_DIR="/lustre/groups/aih/dinesh.haridoss/mil"
SAMPLES_DIR="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
OUT_BASE="${MIL_DIR}/lung_mcat_data"
LOGS="${REPO}/results_mm_abmil_v8/slurm_logs"

mkdir -p "${LOGS}" "${OUT_BASE}/pt_files"

echo "========================================"
echo "Prep lung benchmark data"
echo "Host: $(hostname)  Started: $(date)"
echo "========================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u - <<'PYEOF'
import os, sys, torch
import pandas as pd
import numpy as np
from pathlib import Path

SAMPLES_DIR = Path("/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples")
SPLITS_CSV  = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
OUT_BASE    = Path("/lustre/groups/aih/dinesh.haridoss/mil/lung_mcat_data")
PT_OUT      = OUT_BASE / "pt_files"
PT_OUT.mkdir(parents=True, exist_ok=True)

# ── Load splits CSV ────────────────────────────────────────────────────────
df = pd.read_csv(SPLITS_CSV)

# strip .pt suffix → stem
df["stem"] = df["file"].str.replace(".pt", "", regex=False)

# Include all patients with any modality available
df_valid = df[df["has_HE"] | df["has_CT"] | df["has_BAL"] | df["has_Clinical"]].copy()
print(f"Total patients: {len(df)}, any modality present: {len(df_valid)}")

TARGET_DIM = 1024  # MCAT wsi_input_dim

def pad_to(t, dim):
    """Zero-pad tensor (N, d) → (N, dim) along feature axis."""
    if t.shape[1] == dim:
        return t
    pad = torch.zeros(t.shape[0], dim - t.shape[1], dtype=t.dtype)
    return torch.cat([t, pad], dim=1)

# ── Step 1: Extract all modalities, zero-pad to 1024, concat per patient ──
# HE_cells    (N, 1024) — as-is
# CT_cells    (N, 1024) — as-is
# BAL_cells   (N,   10) — zero-pad → (N, 1024)
# clinical    (106, 491) — stored as clinical_onehot, zero-pad → (106, 1024)
print("\n--- Extracting all-modality patch tensors ---")
extracted = 0
skipped = 0
for _, row in df_valid.iterrows():
    stem = row["stem"]
    src  = SAMPLES_DIR / f"{stem}.pt"
    dst  = PT_OUT / f"{stem}.pt"
    if dst.exists():
        extracted += 1
        continue
    if not src.exists():
        print(f"  [warn] missing src: {src}")
        skipped += 1
        continue
    data = torch.load(src, map_location="cpu", weights_only=False)
    inp  = data.get("inputs", {})
    parts = []
    for key in ("HE_cells", "CT_cells", "BAL_cells"):
        t = inp.get(key, None)
        if t is not None and t.ndim == 2 and t.shape[0] > 0:
            parts.append(pad_to(t.float(), TARGET_DIM))
    # Clinical: stored as clinical_onehot at top level or in inputs
    clin = data.get("clinical_onehot", inp.get("clinical_onehot", None))
    if clin is not None and clin.ndim == 2 and clin.shape[0] > 0:
        parts.append(pad_to(clin.float(), TARGET_DIM))
    if not parts:
        print(f"  [warn] no features in {stem}.pt")
        skipped += 1
        continue
    combined = torch.cat(parts, dim=0)  # (N_total, 1024)
    torch.save(combined, dst)
    extracted += 1
    if extracted % 200 == 0:
        print(f"  extracted {extracted}/{len(df_valid)}  shape={combined.shape}")

print(f"Extraction done: extracted={extracted}, skipped={skipped}")

# ── Step 2: Build per-task CSVs ────────────────────────────────────────────
# MCAT expects: case_id, slide_id, survival_months, censorship, oncotree_code
# censorship: 0=event occurred (uncensored), 1=censored  (opposite of status)

TASKS = {
    "lung_acr":   ("acr_days",   "acr_status"),
    "lung_clad":  ("clad_days",  "clad_status"),
    "lung_death": ("death_days", "death_status"),
}

for task_name, (days_col, status_col) in TASKS.items():
    sub = df_valid[["stem", days_col, status_col]].dropna(subset=[days_col, status_col]).copy()
    sub["case_id"]         = sub["stem"]
    sub["slide_id"]        = sub["stem"]
    sub["survival_months"] = (sub[days_col] / 30.4375).round(4)
    sub["censorship"]      = (1 - sub[status_col]).astype(int)
    sub["oncotree_code"]   = "LUNG"

    # Drop rows with non-positive survival time (MCAT can't bin them)
    sub = sub[sub["survival_months"] > 0]

    out_csv = OUT_BASE / f"{task_name}_all_clean.csv"
    sub[["case_id", "slide_id", "survival_months", "censorship", "oncotree_code"]].to_csv(
        out_csv, index=False)
    print(f"\nCSV [{task_name}]: {len(sub)} patients → {out_csv}")

# ── Step 3: Build split files ──────────────────────────────────────────────
# For each split s: train = split{s}_fold0 train+val, val = split{s}_fold0 test
# Patients not in the HE-present filtered set are excluded.

he_stems = set(df_valid["stem"])

for task_name, (days_col, status_col) in TASKS.items():
    # Build valid_stems directly from df (no CSV re-read, avoids leading-zero loss)
    sub_task = df_valid[["stem", days_col, status_col]].dropna(subset=[days_col, status_col]).copy()
    sub_task["survival_months"] = (sub_task[days_col] / 30.4375)
    sub_task = sub_task[sub_task["survival_months"] > 0]
    valid_stems = set(sub_task["stem"].astype(str))

    splits_dir = OUT_BASE / "splits" / task_name
    splits_dir.mkdir(parents=True, exist_ok=True)

    for s in range(5):
        fold_col = f"split{s}_fold0"
        sub = df[df["stem"].isin(valid_stems)][["stem", fold_col]].dropna()
        train_val = sub[sub[fold_col].isin(["train", "val"])]["stem"].values
        test      = sub[sub[fold_col] == "test"]["stem"].values

        n_train = len(train_val)
        n_test  = len(test)
        n_max   = max(n_train, n_test)

        # Pad shorter column with NaN
        train_col = list(train_val) + [np.nan] * (n_max - n_train)
        val_col   = list(test)      + [np.nan] * (n_max - n_test)

        split_df = pd.DataFrame({"train": train_col, "val": val_col})
        out_path = splits_dir / f"splits_{s}.csv"
        split_df.to_csv(out_path, index=True)
        print(f"  splits [{task_name}] split{s}: train+val={n_train}, test={n_test} → {out_path}")

print("\n--- Data preparation complete ---")
PYEOF

echo ""
echo "========================================"
echo "Creating symlinks in MCAT and MOTCAT repos"
echo "========================================"

MCAT_DIR="${MIL_DIR}/MCAT"
MOTCAT_DIR="${MIL_DIR}/MOTCAT"
OUT_BASE="${MIL_DIR}/lung_mcat_data"

for TASK in lung_acr lung_clad lung_death; do
    CSV_SRC="${OUT_BASE}/${TASK}_all_clean.csv"

    # MCAT symlink
    MCAT_DST="${MCAT_DIR}/dataset_csv/${TASK}_all_clean.csv"
    ln -sf "${CSV_SRC}" "${MCAT_DST}" && echo "  MCAT: ${TASK}_all_clean.csv → ${CSV_SRC}"

    # MOTCAT symlink
    MOTCAT_DST="${MOTCAT_DIR}/dataset_csv/${TASK}_all_clean.csv"
    ln -sf "${CSV_SRC}" "${MOTCAT_DST}" && echo "  MOTCAT: ${TASK}_all_clean.csv → ${CSV_SRC}"

    # MCAT split dir symlink
    MCAT_SPLITS="${MCAT_DIR}/splits/5foldcv/${TASK}"
    ln -sfn "${OUT_BASE}/splits/${TASK}" "${MCAT_SPLITS}" && echo "  MCAT splits: ${TASK} → ${OUT_BASE}/splits/${TASK}"

    # MOTCAT split dir symlink
    MOTCAT_SPLITS="${MOTCAT_DIR}/splits/5foldcv/${TASK}"
    ln -sfn "${OUT_BASE}/splits/${TASK}" "${MOTCAT_SPLITS}" && echo "  MOTCAT splits: ${TASK} → ${OUT_BASE}/splits/${TASK}"
done

echo ""
echo "========================================"
echo "DONE  $(date)"
echo "========================================"
