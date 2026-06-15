#!/usr/bin/env python3
"""
train_mm_abmil_v8.py  ·  Multimodal MIL v8

v8 design principles
---------------------
Phase 1 — Per-modality pretraining (one modality at a time, independently)
  Purpose : produce task-predictive tokens for Phase 2 fusion. That is ALL.
  Loss    : hinge (ACR cls) + Cox (ACR TTE).  No CLR, KD, CRD, or cross-modal.
  Output  : one best_model.pt per modality per fold.

Phase 2 — Multimodal fusion (MultimodalSlotMIL = slot attn + CrossModalTransformer)
  Input   : Phase 1 tokens (K ABMIL slots per modality)
  Fusion  : K*M slots → 4-layer CrossModalTransformer → MultiTaskHead
  Tasks   : acr_cls (hinge) + acr_surv (Cox) + clad (Cox) + death (Cox)
  Loss    : L_hinge + λ₁·L_cox_acr + λ₂·L_cox_clad + λ₃·L_cox_death

Hyperparameter selection — per fold
  1. HP sweep on val set (train set only, not touching test)
  2. Pick best HP (metric = BACC for cls, C-index for survival)
  3. Retrain on train+val with selected HP
  4. Evaluate ONCE on test

Repo structure
  All outputs go under results/ inside this repo.
  Never scatter to /lustre or home_backup.

Usage
-----
  # Phase 1 + 2 for one fold (typical single job):
  python train_mm_abmil_v8.py \\
      --samples-dir  /path/to/samples \\
      --splits-csv   /path/to/splits.csv \\
      --split 1 --fold 0 \\
      --phase both \\
      --p2-variant slot --slot-k 8 --task mega \\
      --hp-sweep --p2-hp-sweep

  # Phase 1 only (run first for all folds in parallel):
  python train_mm_abmil_v8.py --phase p1 --split 1 --fold 0 ...

  # Phase 2 only (depends on Phase 1 outputs):
  python train_mm_abmil_v8.py --phase p2 --split 1 --fold 0 ...
"""

import sys as _sys, pathlib as _pl
_src = _pl.Path(__file__).parent / "src"
if _src.exists() and str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from mil.data.loader import preload_bags
from mil.data.splits import build_splits_multitask
from mil.data.registry import MODALITIES, _feat_dim
from mil.models.builders import build_model_v8
from mil.models.phase1 import SingleModalMIL
from mil.models.phase2 import _load_p1_encoder
from mil.training.phase1_trainer import (
    run_phase1_modality,
    run_phase1_hp_sweep,
    P1_LR, P1_WEIGHT_DECAY, P1_EPOCHS,
    HP_LR_GRID, HP_WD_GRID,
)
from mil.training.phase2_trainer import (
    run_phase2_variant,
    run_phase2_hp_sweep,
    run_single_modal_eval,
    P2_LR, P2_WEIGHT_DECAY,
    P2_HP_LR_GRID, P2_HP_WD_GRID,
)


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_OUT   = Path(__file__).parent / "results" / "mm_abmil_v8"
P1_PATIENCE   = 8   # eval periods (× P1_EVAL_EVERY=25 epochs = 200 ep effective)
P2_PATIENCE   = 20  # 20 × eval_every=20 = 400 epochs without improvement before stopping
COX_LAMBDA    = 0.5   # weight for ACR Cox loss in Phase 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seeds(s: int = 42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _tag(split: int, fold: int) -> str:
    return f"split{split}_fold{fold}"


def _p1_dir(out: Path, split: int, fold: int, mod: str, task: str = "acr") -> Path:
    return out / "phase1" / _tag(split, fold) / task / mod


def _p2_dir(out: Path, split: int, fold: int, variant: str, task: str,
             alternating: bool = False, tag: str = "") -> Path:
    suffix = "_alt" if alternating else ""
    suffix += f"_{tag}" if tag else ""
    return out / "phase2" / _tag(split, fold) / f"{variant}_{task}{suffix}"


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multimodal MIL v8")
    p.add_argument("--samples-dir",  required=True)
    p.add_argument("--splits-csv",   required=True)
    p.add_argument("--split",        type=int, default=1)
    p.add_argument("--fold",         type=int, default=0)
    p.add_argument("--out-dir",      default=str(DEFAULT_OUT))
    p.add_argument("--phase",        choices=["p1", "p2", "both"], default="both")
    p.add_argument("--p2-variant",     default="slot",
                   help="Phase 2 fusion variant: slot (recommended) / early / late / middle")
    p.add_argument("--slot-k",         type=int, default=8,
                   help="Number of slot tokens per modality (HP; sweep [4, 8, 16])")
    p.add_argument("--n-cross-layers", type=int, default=4,
                   help="CrossModalTransformer depth (HP)")
    p.add_argument("--modal-dropout",  type=float, default=0.3,
                   help="Per-modality drop probability during training (default 0.3)")
    p.add_argument("--max-he-patches", type=int, default=99999,
                   help="Max HE patches per patient per forward pass (default: all patches)")
    p.add_argument("--task",           default="mega",
                   help="Task group: cls/surv/both/both_alt/clad_surv/death_surv/mega")
    p.add_argument("--hp-sweep",       action="store_true",
                   help="Run per-fold Phase 1 HP search on val set before final training")
    p.add_argument("--p2-hp-sweep",    action="store_true",
                   help="Run per-fold Phase 2 HP search on val set before final training")
    p.add_argument("--p2-lr",  type=float, default=None,
                   help="Fixed P2 learning rate (overrides HP sweep and default)")
    p.add_argument("--p2-wd",  type=float, default=None,
                   help="Fixed P2 weight decay (overrides HP sweep and default)")
    p.add_argument("--alternating",   action="store_true",
                   help="Use alternating task training in Phase 2 (one task per epoch, stratified)")
    p.add_argument("--geomae-ckpt",   default="",
                   help="Path to GeoMAE best_backbone.pt checkpoint. "
                        "When set, replaces HE/CT linear backbone with pretrained "
                        "SpatialDenoisingEncoder. Also enables 'recon' task in "
                        "alternating sampling (use --task geomae_alt).")
    p.add_argument("--geomae-frozen", action="store_true",
                   help="Keep GeoMAE backbone frozen during Phase 2 (stage-1 fine-tuning)")
    p.add_argument("--p2-tag",        default="",
                   help="Extra tag appended to Phase 2 output dir (e.g. 'nop1' for no-Phase1 ablation)")
    p.add_argument("--p1-tasks",      default="acr",
                   help="Comma-separated Phase 1 tasks to train: acr,clad,death or 'all'")
    p.add_argument("--p1-epochs",      type=int, default=P1_EPOCHS)
    p.add_argument("--p2-frozen-p1",   action="store_true",
                   help="Keep Phase 1 encoders frozen during Phase 2 (default: fine-tune)")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--workers",      type=int, default=8)
    return p


# ── Phase 1 runner ────────────────────────────────────────────────────────────

def run_phase1(args, splits_dict, bag_cache, device, out: Path):
    """Train (or load) Phase 1 for all modalities, for each requested task."""
    split, fold = args.split, args.fold
    train_recs  = splits_dict["train"]
    val_recs    = splits_dict["val"]
    test_recs   = splits_dict["test"]
    trainval_recs = train_recs + val_recs

    raw_tasks = args.p1_tasks.lower().split(",")
    if "all" in raw_tasks:
        raw_tasks = ["acr", "acr_surv", "clad", "death"]
    p1_task_list = [t.strip() for t in raw_tasks]

    # Maps task name → (phase1_task_type, surv_endpoint)
    _task_spec = {
        "acr":     ("acr",      "acr"),    # ACR classification (hinge)
        "acr_surv":("survival", "acr"),    # ACR TTE (Cox, landmark to first episode)
        "clad":    ("survival", "clad"),   # CLAD TTE (Cox)
        "death":   ("survival", "death"),  # Death TTE (Cox, all timepoints)
    }

    p1_paths: Dict[str, Path] = {}   # ACR paths for Phase 2 loading

    for p1_task in p1_task_list:
        task_type, endpoint = _task_spec.get(p1_task, ("acr", "acr"))
        print(f"\n{'='*60}")
        print(f"  Phase 1  task={p1_task}  endpoint={endpoint}")
        print(f"{'='*60}")

        for mod in MODALITIES:
            save_dir = _p1_dir(out, split, fold, mod, p1_task)

            if args.hp_sweep:
                sweep_dir = save_dir / "hp_sweep"
                lr, wd = run_phase1_hp_sweep(
                    mod_name=mod, device=device, bag_cache=bag_cache,
                    train_recs=train_recs, val_recs=val_recs,
                    save_dir=sweep_dir, task=task_type,
                    surv_endpoint=endpoint,
                    lr_grid=HP_LR_GRID, wd_grid=HP_WD_GRID,
                )
            else:
                lr, wd = P1_LR, P1_WEIGHT_DECAY

            final_dir = save_dir / "final"
            ckpt = run_phase1_modality(
                mod_name=mod, fold=fold, device=device, bag_cache=bag_cache,
                train_recs=trainval_recs, val_recs=val_recs, test_recs=test_recs,
                save_dir=final_dir,
                n_epochs=args.p1_epochs, patience=P1_PATIENCE,
                task=task_type, surv_endpoint=endpoint,
                lr=lr, weight_decay=wd,
            )
            if p1_task == "acr":
                p1_paths[mod] = ckpt
            print(f"  [P1-{p1_task}] {mod} → {ckpt}")

    return p1_paths


# ── Phase 2 runner ────────────────────────────────────────────────────────────

def run_phase2(args, splits_dict, bag_cache, device, out: Path,
               p1_paths: Optional[Dict[str, Path]] = None):
    """Build fusion model, optionally load Phase 1 weights, run Phase 2 training."""
    split, fold = args.split, args.fold
    train_recs  = splits_dict["train"]
    val_recs    = splits_dict["val"]
    test_recs   = splits_dict["test"]

    def _make_model():
        return build_model_v8(
            variant=args.p2_variant,
            slot_k=args.slot_k,
            n_cross_layers=args.n_cross_layers,
            task=args.task,
            modal_dropout=getattr(args, "modal_dropout", 0.3),
            max_he_patches=getattr(args, "max_he_patches", 4096),
        )

    model = _make_model()

    # Load GeoMAE backbone if checkpoint provided (replaces HE/CT linear backbone)
    if getattr(args, "geomae_ckpt", ""):
        from mil.models.builders import load_geomae_weights
        load_geomae_weights(model, args.geomae_ckpt,
                            trainable=not getattr(args, "geomae_frozen", False))

    # Load Phase 1 encoder weights if available
    if p1_paths:
        for mod in MODALITIES:
            ckpt_path = (p1_paths.get(mod) or
                         _p1_dir(out, split, fold, mod) / "final" / "best_model.pt")
            if ckpt_path and ckpt_path.exists():
                try:
                    enc = _load_p1_encoder(
                        ckpt_path.parent.parent,
                        mod,
                        trainable=not args.p2_frozen_p1,
                    )
                    model.encoders[mod].load_state_dict(
                        enc.state_dict(), strict=False)
                    status = "frozen" if args.p2_frozen_p1 else "trainable"
                    print(f"  [P2] loaded P1 encoder: {mod} ({status})")
                except Exception as e:
                    print(f"  [P2] warn: could not load P1 encoder {mod}: {e}")
    model = model.to(device)

    save_dir = _p2_dir(out, split, fold, args.p2_variant, args.task,
                       alternating=args.alternating, tag=args.p2_tag)

    # Per-fold Phase 2 HP sweep on val set
    # Allow explicit lr/wd override (for global HP selection across folds)
    p2_lr = getattr(args, 'p2_lr', None) or P2_LR
    p2_wd = getattr(args, 'p2_wd', None) or P2_WEIGHT_DECAY
    if args.p2_hp_sweep:
        sweep_dir = save_dir / "hp_sweep"
        p2_lr, p2_wd = run_phase2_hp_sweep(
            model_factory=_make_model,
            records_train=train_recs,
            records_val=val_recs,
            device=device,
            bag_cache=bag_cache,
            save_dir=sweep_dir,
            task=args.task,
            lr_grid=P2_HP_LR_GRID,
            wd_grid=P2_HP_WD_GRID,
            alternating=args.alternating,
        )
        print(f"  [P2] HP sweep selected: lr={p2_lr}  wd={p2_wd}")

    # Retrain on train+val with selected HP, evaluate on test
    trainval_recs = train_recs + val_recs
    metrics  = run_phase2_variant(
        model=model, variant=args.p2_variant, fold=fold,
        device=device, bag_cache=bag_cache,
        train_recs=trainval_recs, val_recs=val_recs, test_recs=test_recs,
        save_dir=save_dir,
        cox_lambda=COX_LAMBDA,
        task=args.task,
        patience=P2_PATIENCE,
        lr=p2_lr, weight_decay=p2_wd,
        alternating=args.alternating,
    )

    # ── Single-modal baselines on SAME test set (fair comparison) ─────────────
    # Evaluate each P1 checkpoint (if it exists) on the identical test_recs so
    # multimodal vs unimodal numbers share the same denominator.
    majority_labels = [r.get("label") for r in test_recs if r.get("label") in (0,1)]
    majority_label  = int(sum(majority_labels) / max(len(majority_labels), 1) >= 0.5)

    # Maps p1_task name → surv_endpoint for run_single_modal_eval
    _p1_surv_ep = {"acr": "acr", "acr_surv": "acr", "clad": "clad", "death": "death"}

    p1_baselines: Dict[str, dict] = {}
    for mod in MODALITIES:
        for p1_task in ["acr", "acr_surv", "clad", "death"]:
            ckpt = _p1_dir(out, split, fold, mod, p1_task) / "final" / "best_model.pt"
            if not ckpt.exists():
                continue
            try:
                result = run_single_modal_eval(
                    p1_ckpt_path=ckpt,
                    mod_name=mod,
                    all_test_records=test_recs,
                    device=device,
                    bag_cache=bag_cache,
                    majority_label=majority_label,
                    surv_endpoint=_p1_surv_ep[p1_task],
                )
                p1_baselines[f"{mod}_{p1_task}"] = result
                bacc = result.get("bacc"); ci = result.get("c_index")
                val  = f"BACC={bacc:.3f}" if bacc else f"C-idx={ci:.3f}" if ci else "?"
                print(f"  [unimodal] {mod}/{p1_task} → {val}")
            except Exception as e:
                print(f"  [unimodal] {mod}/{p1_task} skip: {e}")

    if p1_baselines:
        metrics["unimodal_baselines"] = p1_baselines

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = _build_parser().parse_args()
    out    = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tag = _tag(args.split, args.fold)
    print(f"\n{'='*60}")
    print(f"  v8  {tag}  phase={args.phase}  variant={args.p2_variant}  task={args.task}")
    print(f"  slot_k={args.slot_k}  n_cross_layers={args.n_cross_layers}")
    print(f"  hp_sweep={args.hp_sweep}  p2_hp_sweep={args.p2_hp_sweep}  alternating={args.alternating}")
    print(f"  p1_tasks={args.p1_tasks}  p2_frozen_p1={args.p2_frozen_p1}")
    print(f"  out → {out}")
    print(f"{'='*60}\n")

    # ── Load splits ───────────────────────────────────────────────────────────
    splits_dict = build_splits_multitask(
        samples_dir=args.samples_dir,
        splits_csv=args.splits_csv,
        fold=args.fold,
        split=args.split,
    )
    all_recs  = (splits_dict["train"] + splits_dict["val"] + splits_dict["test"])
    all_stems = [r["stem"] for r in all_recs]

    # ── Preload bags ──────────────────────────────────────────────────────────
    bag_cache = preload_bags(all_stems, args.samples_dir, n_workers=args.workers)

    # ── Run phases ────────────────────────────────────────────────────────────
    p1_paths = None
    if args.phase in ("p1", "both"):
        p1_paths = run_phase1(args, splits_dict, bag_cache, device, out)

    if args.phase in ("p2", "both"):
        metrics = run_phase2(args, splits_dict, bag_cache, device, out,
                             p1_paths=p1_paths)
        summary_path = out / f"metrics_{tag}_{args.p2_variant}_{args.task}.json"
        with open(summary_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Results → {summary_path}")


if __name__ == "__main__":
    main()
