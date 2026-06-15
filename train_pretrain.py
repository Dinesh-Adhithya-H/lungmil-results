#!/usr/bin/env python3
"""
GeoMAE pretraining entry point.

Self-supervised spatial denoising pretraining on HE + CT + Clinical data.
No labels used — pure reconstruction of masked patches.

After pretraining, backbone weights are saved and can be loaded into
MultimodalSlotMIL for downstream multitask MIL.

Usage:
  python train_pretrain.py \\
      --samples-dir /path/to/mil_v2/samples \\
      --out-dir results/geomae_pretrain \\
      --n-epochs 300 --hidden-dim 256
"""

import sys as _sys, pathlib as _pl
_src = _pl.Path(__file__).parent / "src"
if _src.exists() and str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from mil.training.pretrain_trainer import run_pretrain, PT_LR, PT_WEIGHT_DECAY, PT_EPOCHS


DEFAULT_SAMPLES = "/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
DEFAULT_OUT     = Path(__file__).parent / "results" / "geomae_pretrain"


def _parser():
    p = argparse.ArgumentParser(description="GeoMAE spatial pretraining")
    p.add_argument("--samples-dir",   default=DEFAULT_SAMPLES)
    p.add_argument("--out-dir",       default=str(DEFAULT_OUT))
    p.add_argument("--n-epochs",      type=int,   default=PT_EPOCHS)
    p.add_argument("--lr",            type=float, default=PT_LR)
    p.add_argument("--weight-decay",  type=float, default=PT_WEIGHT_DECAY)
    p.add_argument("--hidden-dim",    type=int,   default=256)
    p.add_argument("--n-layers",      type=int,   default=5)      # v2: 5 layers
    p.add_argument("--n-heads",       type=int,   default=4)
    p.add_argument("--n-slots",       type=int,   default=8)
    p.add_argument("--knn-k",         type=int,   default=32,    # v2: k=32 no disconnected
                   help="K for spatial KNN graph")
    p.add_argument("--he-mask-ratio", type=float, default=0.5)
    p.add_argument("--ct-mask-ratio", type=float, default=0.5)
    p.add_argument("--grad-accum",    type=int,   default=4)
    p.add_argument("--val-frac",      type=float, default=0.1)
    p.add_argument("--splits-csv",    default="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv",
                   help="If provided, use only train-split stems for pretraining")
    p.add_argument("--split",         type=int,   default=1)
    p.add_argument("--seed",          type=int,   default=42)
    return p


def main():
    args = _parser().parse_args()
    out  = Path(args.out_dir)

    print("=" * 60)
    print("  GeoMAE — Geometry-Aware Masked Autoencoder Pretraining")
    print("=" * 60)
    print(f"  samples:    {args.samples_dir}")
    print(f"  out_dir:    {out}")
    print(f"  n_epochs:   {args.n_epochs}")
    print(f"  hidden_dim: {args.hidden_dim}  n_layers: {args.n_layers}")
    print(f"  knn_k:      {args.knn_k}  n_slots: {args.n_slots}")
    print(f"  mask_ratio: HE={args.he_mask_ratio}  CT={args.ct_mask_ratio}")
    print(f"  lr:         {args.lr:.0e}  wd: {args.weight_decay:.0e}")
    print()

    best_ckpt = run_pretrain(
        samples_dir   = args.samples_dir,
        save_dir      = out,
        n_epochs      = args.n_epochs,
        lr            = args.lr,
        weight_decay  = args.weight_decay,
        hidden_dim    = args.hidden_dim,
        n_layers      = args.n_layers,
        n_heads       = args.n_heads,
        n_slots       = args.n_slots,
        knn_k         = args.knn_k,
        he_mask_ratio = args.he_mask_ratio,
        ct_mask_ratio = args.ct_mask_ratio,
        grad_accum    = args.grad_accum,
        val_frac      = args.val_frac,
        splits_csv    = getattr(args, 'splits_csv', None),
        split         = getattr(args, 'split', 1),
        seed          = args.seed,
    )

    print(f"\nBest backbone saved → {best_ckpt}")
    print("Load with:")
    print("  from mil.models.pretrain import GeoMAE")
    print("  weights = torch.load('best_backbone.pt')")
    print("  # then inject he_encoder / ct_encoder into MultimodalSlotMIL")


if __name__ == "__main__":
    main()
