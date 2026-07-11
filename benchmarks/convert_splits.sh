#!/usr/bin/env bash
# convert_splits.sh — overwrite MCAT/MOTCAT/SurvPath split CSVs with our standardised splits
# so all competitor methods train/evaluate on exactly the same patients as our method.
#
# For fold i:
#   train = patients where fold_i == 'train'  (our train set)
#   val   = patients where fold_i == 'test'   (our held-out test — what every method reports on)
#   (our 'val' set is dropped here; competitors use early stopping on their own schedule)
#
#SBATCH --job-name=convert_splits
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_convert_splits.out

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 - << 'PYEOF'
import os
import pandas as pd

OUR_SPLITS = "/home/aih/dinesh.haridoss/chicago_mil/data/tcga_splits"
MIL_DIR    = "/lustre/groups/aih/dinesh.haridoss/mil"

# (cancer, dir_suffix_in_repo)
TARGETS = {
    "mcat": {
        "repo": f"{MIL_DIR}/MCAT",
        "cancers": {
            "blca":   "tcga_blca",
            "brca":   "tcga_brca",
            "gbmlgg": "tcga_gbmlgg",
            "kirc":   "tcga_kirc",
            "luad":   "tcga_luad",
        },
    },
    "motcat": {
        "repo": f"{MIL_DIR}/MOTCAT",
        "cancers": {
            "blca":   "tcga_blca",
            "brca":   "tcga_brca",
            "gbmlgg": "tcga_gbmlgg",
            "luad":   "tcga_luad",
        },
    },
    "survpath": {
        "repo": f"{MIL_DIR}/SurvPath",
        "cancers": {
            "blca":   "tcga_blca",
            "brca":   "tcga_brca",
        },
    },
}

for method, cfg in TARGETS.items():
    for cancer, dir_name in cfg["cancers"].items():
        src = f"{OUR_SPLITS}/{cancer}.csv"
        if not os.path.exists(src):
            print(f"  [skip] no our-splits for {cancer}")
            continue
        df = pd.read_csv(src)
        split_dir = os.path.join(cfg["repo"], "splits", "5foldcv", dir_name)
        os.makedirs(split_dir, exist_ok=True)

        for fold in range(5):
            col = f"fold_{fold}"
            train_ids = df.loc[df[col] == "train", "identifier"].tolist()
            test_ids  = df.loc[df[col] == "test",  "identifier"].tolist()

            # Build long-form CSV: each row has one train ID and one val (=test) ID
            max_len = max(len(train_ids), len(test_ids))
            train_col = train_ids + [None] * (max_len - len(train_ids))
            val_col   = test_ids  + [None] * (max_len - len(test_ids))
            out = pd.DataFrame({"train": train_col, "val": val_col})
            out.index.name = ""  # numeric index, no name
            out_path = os.path.join(split_dir, f"splits_{fold}.csv")
            out.to_csv(out_path)
            print(f"  {method}/{dir_name}/splits_{fold}.csv  "
                  f"train={len(train_ids)}  val(test)={len(test_ids)}")

print("\nDone. All split files overwritten with standardised folds.")
PYEOF
