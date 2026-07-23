"""
Quick W&B upload for already-generated longitudinal figures.
Usage: python interpretability/upload_longitudinal_wandb.py [--split 0] [--fold 0]
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
from interpretability.interpret_longitudinal_mk import (
    load_model, log_to_wandb, OUT_ROOT
)

parser = argparse.ArgumentParser()
parser.add_argument("--split",   type=int, default=0)
parser.add_argument("--fold",    type=int, default=0)
parser.add_argument("--project", default="chicago-mil-interpretability")
args = parser.parse_args()

device = torch.device("cpu")
tasks  = ["acr_cls", "acr_surv", "clad", "death"]
out_dir = OUT_ROOT / f"split{args.split}_fold{args.fold}"

print(f"[upload] Loading model for figures in {out_dir} ...")
model, _ = load_model(args.split, args.fold, device)

png_count = len(list(out_dir.glob("*.png")))
print(f"[upload] Found {png_count} PNGs to upload")

log_to_wandb(model, [], tasks, out_dir, args.split, args.fold, args.project)
print("[upload] Done.")
