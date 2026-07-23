#!/bin/bash
# prep_lung_benchmark.sh
# ---------------------------------------------------------------------------
# Prepare lung transplant data for MCAT / MOTCAT benchmarks.
#
# MCAT/MOTCAT support exactly one bag input + one tabular omic input.
# They cannot take multiple separate multi-instance modalities.
#
# Input design:
#   pt_files/{stem}.pt : HE_cells (N, 1024) — one bag per anchor
#                        Missing HE → zeros(1, 1024) placeholder.
#   CSV omic cols      : inputs.Clinical (106-dim float) as Clinical_0_rnaseq..Clinical_105_rnaseq
#                        Total 106 omic features + 1 dummy (sacrificed to reordering)
#   Signatures CSV     : 6 groups of ~17-18 features for MCAT coattn mode
#
# Per-task CSVs: lung_acr, lung_clad, lung_death
# 5 split files per task: splits_0.csv (train+val → train, test → val for fair eval)
# Symlinks wired into MCAT/MOTCAT dirs automatically.
#
# Run: sbatch benchmarks/prep_lung_benchmark.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=prep_lung_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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

mkdir -p "${REPO}/results_mm_abmil_v8/slurm_logs"

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
MCAT_DIR    = Path("/lustre/groups/aih/dinesh.haridoss/mil/MCAT")
MOTCAT_DIR  = Path("/lustre/groups/aih/dinesh.haridoss/mil/MOTCAT")

OUT_BASE.mkdir(parents=True, exist_ok=True)
(OUT_BASE / "pt_files").mkdir(exist_ok=True)

# -----------------------------------------------------------------------
# 1. Load splits CSV
# -----------------------------------------------------------------------
splits_df = pd.read_csv(SPLITS_CSV, low_memory=False)

# Strip .pt suffix from file column to get stem
splits_df["stem"] = splits_df["file"].str.replace(".pt", "", regex=False)

print(f"Splits CSV: {len(splits_df)} rows, {splits_df['stem'].nunique()} unique stems")
print(f"Columns: {list(splits_df.columns[:10])}")

all_stems = splits_df["stem"].unique().tolist()

# -----------------------------------------------------------------------
# 2. Build pt_files: HE_cells only (1024-dim, MCAT's path branch)
# -----------------------------------------------------------------------
print("\n--- Building pt_files ---")
no_he = 0; missing = 0
for stem in all_stems:
    pt_out = OUT_BASE / "pt_files" / f"{stem}.pt"
    if pt_out.exists():
        continue
    src_path = SAMPLES_DIR / f"{stem}.pt"
    if not src_path.exists():
        torch.save(torch.zeros(1, 1024), pt_out)
        missing += 1
        continue
    d = torch.load(src_path, map_location="cpu")
    he = d["inputs"].get("HE_cells")
    if he is not None and he.shape[0] > 0:
        torch.save(he.float(), pt_out)
    else:
        torch.save(torch.zeros(1, 1024), pt_out)
        no_he += 1

print(f"  Done. no_HE={no_he}, src_missing={missing}")

# -----------------------------------------------------------------------
# 3. Load clinical features per stem (BAL is now in the path bag)
# -----------------------------------------------------------------------
print("\n--- Loading clinical features per stem ---")
clinical_rows = {}
for stem in all_stems:
    src_path = SAMPLES_DIR / f"{stem}.pt"
    if not src_path.exists():
        clinical_rows[stem] = np.zeros(106, dtype=np.float32)
        continue
    d = torch.load(src_path, map_location="cpu")
    inputs = d["inputs"]
    cli = inputs.get("Clinical")
    if cli is not None:
        clinical_rows[stem] = cli.float().numpy()            # (106,)
    else:
        clinical_rows[stem] = np.zeros(106, dtype=np.float32)

print(f"  Loaded {len(clinical_rows)} stems. Feature dim=106")

# Column names with _rnaseq suffix for MCAT signature matching
OMIC_COLS = [f"Clinical_{i}_rnaseq" for i in range(106)]
# One extra dummy column that will be consumed by MCAT's internal reordering logic
OMIC_COLS_FULL = OMIC_COLS + ["Clinical_dummy_rnaseq"]

# -----------------------------------------------------------------------
# 4. Build per-task CSVs
# -----------------------------------------------------------------------
# MCAT CSV format (first 9 original cols = metadata; after MCAT's internal
# reordering, first 12 cols become metadata including label, disc_label,
# and the sacrificed dummy column):
#   case_id, slide_id, site, is_female, oncotree_code, age,
#   survival_months, censorship, train,
#   Clinical_0_rnaseq, ..., Clinical_105_rnaseq,
#   Clinical_dummy_rnaseq

TASK_CFG = {
    "lung_acr": {
        "time_col":  "acr_days",
        "status_col": "acr_status",
        "drop_nan": True,   # many anchors have NaN acr_days — exclude them
    },
    "lung_clad": {
        "time_col":  "clad_days",
        "status_col": "clad_status",
        "drop_nan": False,
    },
    "lung_death": {
        "time_col":  "death_days",
        "status_col": "death_status",
        "drop_nan": False,
    },
}

task_csvs = {}
for task, cfg in TASK_CFG.items():
    print(f"\n--- Building CSV for {task} ---")
    df = splits_df.copy()
    if cfg["drop_nan"]:
        df = df[df[cfg["time_col"]].notna()].copy()
    df = df.reset_index(drop=True)
    # survival_months: convert days to months
    df["survival_months"] = df[cfg["time_col"]] / 30.0
    # censorship: MCAT convention: 0=event(uncensored), 1=censored
    df["censorship"] = (1 - df[cfg["status_col"]]).astype(int)
    # Build omic feature matrix
    omic_matrix = np.stack([clinical_rows.get(s, np.zeros(106)) for s in df["stem"]], axis=0)
    omic_df = pd.DataFrame(omic_matrix, columns=OMIC_COLS)
    omic_df["Clinical_dummy_rnaseq"] = 0.0
    meta = pd.DataFrame({
        "case_id":         df["stem"].values,
        "slide_id":        df["stem"].values,
        "site":            "LUNG",
        "is_female":       0,
        "oncotree_code":   "LUNG",
        "age":             0,
        "survival_months": df["survival_months"].values,
        "censorship":      df["censorship"].values,
        "train":           0,
    })
    csv_df = pd.concat([meta.reset_index(drop=True), omic_df.reset_index(drop=True)], axis=1)
    out_csv = OUT_BASE / f"{task}_all_clean.csv"
    csv_df.to_csv(out_csv, index=False)
    task_csvs[task] = (csv_df, df)
    print(f"  Saved {len(csv_df)} rows → {out_csv}")
    print(f"  censorship=0 (event): {(csv_df['censorship']==0).sum()}, =1 (censored): {(csv_df['censorship']==1).sum()}")

# -----------------------------------------------------------------------
# 5. Build splits files per task per split
# -----------------------------------------------------------------------
# For each split s (0-4), create ONE splits file: splits_0.csv
# Using fold0: train+val as train, test as val (fair test-set eval)
print("\n--- Building splits ---")
for task, cfg in TASK_CFG.items():
    csv_df, df_task = task_csvs[task]
    for s in range(5):
        fold_col = f"split{s}_fold0"
        split_dir = OUT_BASE / "splits" / "5foldcv" / task
        split_dir.mkdir(parents=True, exist_ok=True)
        # Get assignments for rows in task CSV
        assignments = df_task[fold_col].values if fold_col in df_task.columns else None
        if assignments is None:
            print(f"  [WARN] {fold_col} not found in df_task for {task}")
            continue
        train_stems = df_task.loc[
            df_task[fold_col].isin(["train", "val"]), "stem"
        ].values
        test_stems = df_task.loc[
            df_task[fold_col] == "test", "stem"
        ].values
        max_len = max(len(train_stems), len(test_stems))
        train_col = list(train_stems) + [np.nan] * (max_len - len(train_stems))
        val_col   = list(test_stems)  + [np.nan] * (max_len - len(test_stems))
        split_csv = pd.DataFrame({"train": train_col, "val": val_col})
        out_path = split_dir / f"splits_{s}.csv"
        split_csv.to_csv(out_path, index=False)
    print(f"  {task}: created splits_0..4 in {OUT_BASE}/splits/5foldcv/{task}/")

# -----------------------------------------------------------------------
# 6. Build signatures CSV (6 groups of ~19 features each from 116 total)
# -----------------------------------------------------------------------
print("\n--- Building signatures CSV ---")
# Base names (without _rnaseq suffix) — Clinical only (BAL is now in path bag)
base_names = [f"Clinical_{i}" for i in range(106)]
# Split 106 features into 6 groups (~17-18 each)
group_size = 106 // 6  # = 17
groups = []
for g in range(6):
    start = g * group_size
    end = start + group_size if g < 5 else 106
    groups.append(base_names[start:end])

max_group_len = max(len(g) for g in groups)
sig_dict = {}
for g_idx, g in enumerate(groups):
    padded = g + [np.nan] * (max_group_len - len(g))
    sig_dict[f"Signature_{g_idx+1}"] = padded

sig_df = pd.DataFrame(sig_dict)
print(f"  Signature groups: {[len(g) for g in groups]}")
print(f"  Sig CSV shape: {sig_df.shape}")

# Write to both MCAT and MOTCAT dirs
for repo_dir in [MCAT_DIR, MOTCAT_DIR]:
    sig_out_dir = repo_dir / "datasets_csv_sig"
    sig_out_dir.mkdir(exist_ok=True)
    sig_out = sig_out_dir / "signatures.csv"
    # Back up existing if needed
    if sig_out.exists():
        import shutil
        shutil.copy(sig_out, sig_out_dir / "signatures_tcga_backup.csv")
        print(f"  Backed up existing signatures → {sig_out_dir}/signatures_tcga_backup.csv")
    sig_df.to_csv(sig_out, index=False)
    print(f"  Written → {sig_out}")

# -----------------------------------------------------------------------
# 7. Wire symlinks into MCAT / MOTCAT dirs
# -----------------------------------------------------------------------
print("\n--- Wiring symlinks ---")
for task in TASK_CFG:
    src_csv = OUT_BASE / f"{task}_all_clean.csv"
    src_splits = OUT_BASE / "splits" / "5foldcv" / task

    # MCAT: only dataset_csv + splits (MCAT uses --direct_csv_path so no csv symlink needed)
    mcat_splits_dst = MCAT_DIR / "splits" / "5foldcv" / task
    mcat_splits_dst.parent.mkdir(parents=True, exist_ok=True)
    if mcat_splits_dst.is_symlink() or mcat_splits_dst.exists():
        if mcat_splits_dst.is_symlink():
            mcat_splits_dst.unlink()
        else:
            import shutil; shutil.rmtree(mcat_splits_dst)
    mcat_splits_dst.symlink_to(src_splits)

    # MOTCAT: dataset_csv + splits (MOTCAT reads CSV from ./dataset_csv/{task}_all_clean.csv)
    motcat_csv_dst = MOTCAT_DIR / "dataset_csv" / f"{task}_all_clean.csv"
    motcat_csv_dst.parent.mkdir(exist_ok=True)
    if motcat_csv_dst.is_symlink() or motcat_csv_dst.exists():
        motcat_csv_dst.unlink()
    motcat_csv_dst.symlink_to(src_csv)

    motcat_splits_dst = MOTCAT_DIR / "splits" / "5foldcv" / task
    motcat_splits_dst.parent.mkdir(parents=True, exist_ok=True)
    if motcat_splits_dst.is_symlink() or motcat_splits_dst.exists():
        if motcat_splits_dst.is_symlink():
            motcat_splits_dst.unlink()
        else:
            import shutil; shutil.rmtree(motcat_splits_dst)
    motcat_splits_dst.symlink_to(src_splits)

    print(f"  {task}: MCAT splits ✓  MOTCAT csv+splits ✓")

print("\n======================================")
print("Prep complete.")
print(f"  pt_files: {len(list((OUT_BASE / 'pt_files').glob('*.pt')))} files")
for task in TASK_CFG:
    n = sum(1 for _ in open(OUT_BASE / f"{task}_all_clean.csv")) - 1
    print(f"  {task}_all_clean.csv: {n} rows")
print(f"  Signatures: 6 groups, {max_group_len} features each")
print("======================================")
PYEOF

echo "========================================"
echo "Done  $(date)"
echo "========================================"
