#!/bin/bash
#SBATCH --job-name=wandb_long_upload
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_wandb_long.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_wandb_long.err
#SBATCH --partition=cpu_p
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --qos=cpu_normal

# Upload already-generated longitudinal figures to W&B (no GPU needed).
# Loads the saved PNGs from the interpretability output dir and logs them.

set -euo pipefail

SPLIT=${1:-0}
FOLD=${2:-0}
PROJECT=${3:-chicago-mil-interpretability}

REPO=/ictstr01/home/aih/dinesh.haridoss/chicago_mil
cd "${REPO}"

export PYTHONUNBUFFERED=1

conda run -n chicago python - <<PYEOF
import sys, torch
sys.path.insert(0, "src")
from pathlib import Path
from interpretability.interpret_longitudinal_mk import (
    load_model, log_to_wandb, OUT_ROOT, TASK_LABELS
)

split, fold, project = int("${SPLIT}"), int("${FOLD}"), "${PROJECT}"
out_dir = OUT_ROOT / f"split{split}_fold{fold}"
tasks   = ["acr_cls", "acr_surv", "clad", "death"]

device = torch.device("cpu")
model, _ = load_model(split, fold, device)

# log_to_wandb only needs model + out_dir for image upload
# Pass empty all_extractions — scalars/tables will be minimal, images will be full
log_to_wandb(model, [], tasks, out_dir, split, fold, project)
print("Done.")
PYEOF
