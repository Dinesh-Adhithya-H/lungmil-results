#!/usr/bin/env python3
"""
TCGA survival benchmark — WSI-only.

Runs 5-fold CV for one (cancer, model) combination.
Results written to results/tcga_benchmark/{cancer}/{model}/fold{k}/result.json

Usage:
  python train_tcga.py --cancer KIRC --model abmil --fold 0
  python train_tcga.py --cancer KIRC --model geomae_slotmil --fold 0 \
      --geomae-ckpt results/geomae_pretrain/best_backbone.pt
"""
import sys, argparse, json, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch

from tcga_benchmark.data.dataset import (
    ALL_CANCERS, make_splits, load_records, preload_bags)
from tcga_benchmark.models.baselines import build_model, MODELS
from tcga_benchmark.training.trainer import train_and_eval

OUT_BASE = Path(__file__).parent / "results" / "tcga_benchmark"
GEOMAE_DEFAULT = str(
    Path(__file__).parent / "results" / "geomae_pretrain" / "best_backbone.pt")


def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--cancer",      required=True, choices=ALL_CANCERS)
    p.add_argument("--model",       required=True, choices=list(MODELS))
    p.add_argument("--fold",        type=int, required=True)
    p.add_argument("--n-folds",     type=int, default=5)
    p.add_argument("--n-epochs",    type=int, default=40)
    p.add_argument("--hidden",      type=int, default=256)
    p.add_argument("--n-slots",     type=int, default=8)
    p.add_argument("--geomae-ckpt", default=GEOMAE_DEFAULT)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--workers",     type=int, default=8)
    return p


def main():
    args  = _parser().parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  TCGA Benchmark: {args.cancer}  model={args.model}  fold={args.fold}")
    print(f"{'='*60}\n")

    # ── Splits ─────────────────────────────────────────────────────────────────
    splits    = make_splits(args.cancer, args.n_folds, args.seed)
    fold_info = splits[args.fold]

    train_recs = load_records(args.cancer, fold_info["train"])
    val_recs   = load_records(args.cancer, fold_info["val"])
    test_recs  = load_records(args.cancer, fold_info["test"])

    print(f"  train={len(train_recs)}  val={len(val_recs)}  test={len(test_recs)}")
    ev_tr = sum(r["os_event"] for r in train_recs)
    ev_te = sum(r["os_event"] for r in test_recs)
    print(f"  train events={int(ev_tr)}  test events={int(ev_te)}")

    # ── Bags ───────────────────────────────────────────────────────────────────
    all_recs = train_recs + val_recs + test_recs
    bag_cache = preload_bags(args.cancer, all_recs, n_workers=args.workers)

    # ── Model factory ──────────────────────────────────────────────────────────
    geomae_ckpt = args.geomae_ckpt if Path(args.geomae_ckpt).exists() else None
    if args.model == "geomae_slotmil" and geomae_ckpt is None:
        print(f"  WARNING: geomae_ckpt not found at {args.geomae_ckpt}")
        print(f"  GeoMAE-SlotMIL will use random-init backbone (no pretraining)")

    def model_factory():
        kw = dict(hidden=args.hidden)
        if args.model in ("slotmil", "geomae_slotmil"):
            kw["n_slots"] = args.n_slots
        if args.model == "geomae_slotmil":
            kw["geomae_ckpt"] = geomae_ckpt
        return build_model(args.model, **kw)

    # ── Train ──────────────────────────────────────────────────────────────────
    save_dir = (OUT_BASE / args.cancer / args.model /
                f"fold{args.fold}")
    result = train_and_eval(
        model_factory=model_factory,
        train_recs=train_recs,
        val_recs=val_recs,
        test_recs=test_recs,
        bag_cache=bag_cache,
        save_dir=save_dir,
        n_epochs=args.n_epochs,
        seed=args.seed,
    )

    print(f"\n  Result: {result}")
    print(f"  Saved  → {save_dir}/result.json")


if __name__ == "__main__":
    main()
