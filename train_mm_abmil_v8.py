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
      --p2-variant mario_kempes --slot-k 16 --task mega \\
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
from mil.data.splits import build_splits_multitask, build_splits_longitudinal
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
    run_phase2_final,
    run_phase2_hp_sweep,
    run_single_modal_eval,
    run_longitudinal_hp_sweep,
    run_longitudinal_final,
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


def _aggregate_global_hp(out: Path, split: int, folds: range,
                          variant: str, task: str,
                          alternating: bool = False, tag: str = "") -> tuple:
    """Read per-fold HP sweep JSONs, average val_bacc per (lr,wd) combo, return best."""
    fname = "hp_sweep_p2_alt.json" if alternating else "hp_sweep_p2.json"
    combo_scores: dict = {}  # (lr,wd) -> list of scores
    n_found = 0
    for f in folds:
        sweep_json = _p2_dir(out, split, f, variant, task, alternating, tag) / "hp_sweep" / fname
        if not sweep_json.exists():
            print(f"  [global-HP] fold {f} sweep not found at {sweep_json} — skipping")
            continue
        with open(sweep_json) as fh:
            data = json.load(fh)
        for row in data.get("grid", []):
            key = (row["lr"], row["wd"])
            score = row.get("val_bacc", row.get("val_cidx", row.get("val_cox", row.get("val_loss", None))))
            if score is not None:
                combo_scores.setdefault(key, []).append(score)
        n_found += 1
    if not combo_scores:
        raise RuntimeError(f"[global-HP] No HP sweep results found for split={split}")
    # Higher val_bacc = better (we always saved as val_bacc even for survival tasks)
    avg_scores = {k: sum(v)/len(v) for k, v in combo_scores.items()}
    best_key   = max(avg_scores, key=lambda k: avg_scores[k])
    best_lr, best_wd = best_key
    print(f"  [global-HP] aggregated {n_found} folds  best: lr={best_lr}  wd={best_wd}"
          f"  avg_val_bacc={avg_scores[best_key]:.4f}")
    for (lr, wd), sc in sorted(avg_scores.items()):
        mark = "  ← best" if (lr, wd) == best_key else ""
        print(f"    lr={lr}  wd={wd}  avg={sc:.4f}  n={len(combo_scores[(lr,wd)])}{mark}")
    return best_lr, best_wd


def _aggregate_best_epoch(out: Path, split: int, folds: range,
                           variant: str, task: str,
                           alternating: bool = False, tag: str = "") -> int:
    """Read per-fold final training status JSONs, return cross-validated best epoch.

    Used to set n_epochs for combined training (train+val) without a val set:
    the per-fold best_epoch is a cross-validated estimate of when to stop.
    A 10% buffer is added since combined data is ~25% larger → slightly longer convergence.

    Falls back to hp_sweep JSON stopped_ep when no status file exists (e.g. folds
    that only ran HP sweeps without combined training, such as mario_kempes folds 1-3).
    """
    from mil.training.phase2_trainer import P2_FINAL_EPOCHS
    best_epochs = []
    for f in folds:
        save_d = _p2_dir(out, split, f, variant, task, alternating, tag)
        vtag   = tag or variant
        status_path = save_d / f"status_{vtag}_final.json"
        if not status_path.exists():
            status_path = save_d / f"status_{vtag}.json"
        ep = None
        if status_path.exists():
            with open(status_path) as fh:
                st = json.load(fh)
            ep = st.get("best_epoch")
            if ep:
                print(f"  [cv-epoch] fold {f}: best_epoch={ep}")
        if not ep:
            # Fall back: read stopped_ep of the best HP combo from hp_sweep JSON
            fname = "hp_sweep_p2_alt.json" if alternating else "hp_sweep_p2.json"
            sweep_path = save_d / "hp_sweep" / fname
            if sweep_path.exists():
                with open(sweep_path) as fh:
                    sw = json.load(fh)
                best_lr_sw = sw.get("best_lr")
                best_wd_sw = sw.get("best_wd")
                for row in sw.get("grid", []):
                    if row.get("lr") == best_lr_sw and row.get("wd") == best_wd_sw:
                        ep = row.get("stopped_ep")
                        if ep:
                            print(f"  [cv-epoch] fold {f}: best_epoch={ep} (from hp_sweep)")
                        break
        if not ep:
            print(f"  [cv-epoch] fold {f}: no epoch info found — skipping")
            continue
        best_epochs.append(ep)
    if not best_epochs:
        print(f"  [cv-epoch] No fold results available; using P2_FINAL_EPOCHS={P2_FINAL_EPOCHS}")
        return P2_FINAL_EPOCHS
    avg_ep  = sum(best_epochs) / len(best_epochs)
    # 10% buffer for larger combined dataset, rounded to nearest eval_every (20)
    n_ep    = max(20, round(avg_ep * 1.10 / 20) * 20)
    print(f"  [cv-epoch] avg best_epoch={avg_ep:.1f}  → combined n_epochs={n_ep} (+10% buffer)")
    return n_ep


def _aggregate_global_hp_p1(out: Path, split: int, folds: range,
                             mod: str, task: str) -> tuple:
    """Aggregate Phase 1 HP sweep results across folds, return best (lr, wd)."""
    combo_scores: dict = {}
    n_found = 0
    for f in folds:
        sweep_json = _p1_dir(out, split, f, mod, task) / "hp_sweep" / "hp_sweep.json"
        if not sweep_json.exists():
            print(f"  [p1-global-HP] fold {f} sweep not found — skipping")
            continue
        with open(sweep_json) as fh:
            data = json.load(fh)
        for row in data.get("grid", []):
            key = (row["lr"], row["wd"])
            score = row.get("val_metric", row.get("val_bacc"))
            if score is not None:
                combo_scores.setdefault(key, []).append(score)
        n_found += 1
    if not combo_scores:
        raise RuntimeError(f"[p1-global-HP] No sweep results for {mod}/{task}")
    avg_scores = {k: sum(v) / len(v) for k, v in combo_scores.items()}
    best_key   = max(avg_scores, key=lambda k: avg_scores[k])
    best_lr, best_wd = best_key
    print(f"  [p1-global-HP] {mod}/{task}: aggregated {n_found} folds  "
          f"best: lr={best_lr}  wd={best_wd}  avg={avg_scores[best_key]:.4f}")
    return best_lr, best_wd


def _aggregate_best_epoch_p1(out: Path, split: int, folds: range,
                              mod: str, task: str) -> int:
    """Aggregate Phase 1 per-fold best_epoch, add 10% buffer for combined training."""
    from mil.training.phase1_trainer import P1_EPOCHS
    best_epochs = []
    for f in folds:
        status_path = _p1_dir(out, split, f, mod, task) / "final" / "status.json"
        if not status_path.exists():
            print(f"  [p1-cv-epoch] fold {f} status not found — skipping")
            continue
        with open(status_path) as fh:
            st = json.load(fh)
        ep = st.get("best_epoch")
        if ep:
            best_epochs.append(ep)
            print(f"  [p1-cv-epoch] {mod}/{task} fold {f}: best_epoch={ep}")
    if not best_epochs:
        print(f"  [p1-cv-epoch] No fold results; using P1_EPOCHS={P1_EPOCHS}")
        return P1_EPOCHS
    avg_ep = sum(best_epochs) / len(best_epochs)
    n_ep   = max(25, round(avg_ep * 1.10 / 25) * 25)
    print(f"  [p1-cv-epoch] {mod}/{task}: avg={avg_ep:.1f} → combined n_epochs={n_ep}")
    return n_ep


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multimodal MIL v8")
    p.add_argument("--samples-dir",  required=True)
    p.add_argument("--splits-csv",   required=True)
    p.add_argument("--split",        type=int, default=1)
    p.add_argument("--fold",         type=int, default=0)
    p.add_argument("--out-dir",      default=str(DEFAULT_OUT))
    p.add_argument("--phase",        choices=["p1", "p2", "both"], default="both")
    p.add_argument("--p2-variant",     default="mario_kempes",
                   help="Phase 2 fusion variant: mario_kempes (recommended) / early / late / middle")
    p.add_argument("--slot-k",         type=int, default=16,
                   help="Number of seed tokens per modality (PMA, default 16)")
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
    p.add_argument("--global-hp",     action="store_true",
                   help="After per-fold HP sweeps complete, pick best HP averaged across all folds "
                        "instead of using only the current fold's sweep result")
    p.add_argument("--hp-source-tag", default=None,
                   help="Tag of per-fold training dirs to read HP sweeps and best_epoch from. "
                        "Use when --p2-tag differs from the per-fold tag, e.g. "
                        "--p2-tag shared_combined --hp-source-tag shared")
    p.add_argument("--p2-all-folds",   action="store_true",
                   help="Load bags once and run HP sweeps for all 4 folds in one process. "
                        "Eliminates 4x repeated disk I/O. Fold 0 also runs --global-hp and "
                        "--combined-train. Overrides --fold (all folds are run).")
    p.add_argument("--combined-train", action="store_true",
                   help="Train final model on train+val combined (all folds share same test set). "
                        "Requires --global-hp or --p2-lr/--p2-wd. Trains for full P2_FINAL_EPOCHS "
                        "with no early stopping.")
    p.add_argument("--p1-tasks",      default="acr",
                   help="Comma-separated Phase 1 tasks to train: acr,clad,death or 'all'")
    p.add_argument("--p1-epochs",      type=int, default=P1_EPOCHS)
    p.add_argument("--p1-global-hp",   action="store_true",
                   help="Select Phase 1 HP by averaging val_metric across all 4 folds "
                        "(requires per-fold HP sweeps to exist)")
    p.add_argument("--p1-combined-tag", default="",
                   help="Suffix for combined Phase 1 save dir, e.g. 'combined'. "
                        "Saves to {p1_dir}/final_{tag}/ to avoid overwriting per-fold models.")
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

            # Global HP: average val_metric across all folds, overrides per-fold sweep
            if getattr(args, 'p1_global_hp', False):
                lr, wd = _aggregate_global_hp_p1(out, split, range(4), mod, p1_task)

            # Choose save dir — use combined tag to avoid overwriting per-fold model
            p1_combined_tag = getattr(args, 'p1_combined_tag', '')
            final_dir_name  = f"final_{p1_combined_tag}" if p1_combined_tag else "final"
            final_dir       = save_dir / final_dir_name

            # For combined training: cross-validated epoch count
            n_epochs = args.p1_epochs
            if p1_combined_tag:
                n_epochs = _aggregate_best_epoch_p1(out, split, range(4), mod, p1_task)

            ckpt = run_phase1_modality(
                mod_name=mod, fold=fold, device=device, bag_cache=bag_cache,
                train_recs=trainval_recs, val_recs=val_recs, test_recs=test_recs,
                save_dir=final_dir,
                n_epochs=n_epochs, patience=P1_PATIENCE,
                task=task_type, surv_endpoint=endpoint,
                lr=lr, weight_decay=wd,
            )
            if p1_task == "acr":
                p1_paths[mod] = ckpt
            print(f"  [P1-{p1_task}] {mod} → {ckpt}")

    return p1_paths


# ── Phase 2 runner ────────────────────────────────────────────────────────────

def run_phase2_longitudinal(args, bag_cache, device, out: Path,
                            fold: Optional[int] = None):
    """Separate Phase 2 path for longitudinal_mk variant."""
    split = args.split
    fold  = fold if fold is not None else args.fold

    long_splits = build_splits_longitudinal(
        samples_dir=args.samples_dir,
        splits_csv=args.splits_csv,
        fold=fold,
        split=split,
    )
    patient_train = long_splits["train"]
    patient_val   = long_splits["val"]
    patient_test  = long_splits["test"]

    def _make_model():
        return build_model_v8(
            variant="longitudinal_mk",
            slot_k=args.slot_k,
            n_cross_layers=args.n_cross_layers,
            task=args.task,
            modal_dropout=getattr(args, "modal_dropout", 0.3),
            max_he_patches=getattr(args, "max_he_patches", 4096),
        )

    model    = _make_model().to(device)
    save_dir = _p2_dir(out, split, fold, "longitudinal_mk", args.task, tag=args.p2_tag)
    p2_lr    = getattr(args, "p2_lr", None) or P2_LR
    p2_wd    = getattr(args, "p2_wd", None) or P2_WEIGHT_DECAY

    if args.p2_hp_sweep:
        sweep_dir = save_dir / "hp_sweep"
        p2_lr, p2_wd = run_longitudinal_hp_sweep(
            model_factory=_make_model,
            patient_train=patient_train,
            patient_val=patient_val,
            device=device,
            bag_cache=bag_cache,
            save_dir=sweep_dir,
            lr_grid=P2_HP_LR_GRID,
            wd_grid=P2_HP_WD_GRID,
        )
        print(f"  [LMK] HP sweep selected: lr={p2_lr}  wd={p2_wd}")

    _hp_tag = getattr(args, "hp_source_tag", None) or args.p2_tag
    if getattr(args, "global_hp", False):
        p2_lr, p2_wd = _aggregate_global_hp(
            out, split, range(4), "longitudinal_mk", args.task, tag=_hp_tag)
        print(f"  [LMK] global HP selected: lr={p2_lr}  wd={p2_wd}")

    use_combined = getattr(args, "combined_train", False)
    combined_n_epochs = None
    if use_combined:
        combined_n_epochs = _aggregate_best_epoch(
            out, split, range(4), "longitudinal_mk", args.task, tag=_hp_tag)

    final_kwargs = {}
    if combined_n_epochs is not None:
        final_kwargs["n_epochs"] = combined_n_epochs

    metrics = run_longitudinal_final(
        model=model,
        variant="longitudinal_mk",
        fold=fold,
        device=device,
        bag_cache=bag_cache,
        patient_train=patient_train,
        patient_val=patient_val,
        patient_test=patient_test,
        save_dir=save_dir,
        lr=p2_lr,
        weight_decay=p2_wd,
        combined_train=use_combined,
        **final_kwargs,
    )
    return metrics


def run_phase2(args, splits_dict, bag_cache, device, out: Path,
               p1_paths: Optional[Dict[str, Path]] = None,
               fold: Optional[int] = None):
    """Build fusion model, optionally load Phase 1 weights, run Phase 2 training."""
    split = args.split
    fold  = fold if fold is not None else args.fold
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

    # Global HP: aggregate across all folds, overrides per-fold sweep result
    # hp_source_tag: use per-fold training dirs (may differ from combined save tag)
    _hp_tag = getattr(args, 'hp_source_tag', None) or args.p2_tag
    if getattr(args, 'global_hp', False):
        p2_lr, p2_wd = _aggregate_global_hp(
            out, split, range(4), args.p2_variant, args.task,
            alternating=args.alternating, tag=_hp_tag)
        print(f"  [P2] global HP selected: lr={p2_lr}  wd={p2_wd}")

    # Combined train: merge train+val → more training data (all folds share same test set)
    use_combined = getattr(args, 'combined_train', False)
    final_train_recs = train_recs + val_recs if use_combined else train_recs
    combined_n_epochs = None
    if use_combined:
        print(f"  [P2] combined_train=True: {len(train_recs)} train + {len(val_recs)} val "
              f"= {len(final_train_recs)} total training records")
        # Cross-validated epoch estimate from per-fold final training (anti-overfitting)
        combined_n_epochs = _aggregate_best_epoch(
            out, split, range(4), args.p2_variant, args.task,
            alternating=args.alternating, tag=_hp_tag)

    # Retrain on train+val combined with selected HP — fixed epochs, no val leakage
    final_kwargs = {}
    if combined_n_epochs is not None:
        final_kwargs["n_epochs"] = combined_n_epochs
    metrics = run_phase2_final(
        model=model, variant=args.p2_variant, fold=fold,
        device=device, bag_cache=bag_cache,
        train_recs=final_train_recs, val_recs=val_recs, test_recs=test_recs,
        save_dir=save_dir,
        task=args.task,
        lr=p2_lr, weight_decay=p2_wd,
        combined_train=use_combined,
        alternating=args.alternating,
        **final_kwargs,
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

    # ── --p2-all-folds: load bags once, iterate over all folds (and tasks) ──────
    if getattr(args, "p2_all_folds", False) and args.phase in ("p2", "both"):
        # For multi-task variants (early/late/middle) iterate over all tasks with
        # one bag load. mega variants (mario_kempes, longitudinal_mk) use args.task directly.
        _MULTI_TASKS = {
            "early":  ["cls", "acr_surv", "clad_surv", "death_surv"],
            "late":   ["cls", "acr_surv", "clad_surv", "death_surv"],
            "middle": ["cls", "acr_surv", "clad_surv", "death_surv"],
        }
        _task_list = _MULTI_TASKS.get(args.p2_variant, [args.task])

        # Collect all stems across all 4 folds (union = full split).
        all_stems: set = set()
        for _f in range(4):
            _sd = build_splits_multitask(args.samples_dir, args.splits_csv,
                                         fold=_f, split=args.split)
            for _r in _sd["train"] + _sd["val"] + _sd["test"]:
                all_stems.add(_r["stem"])
        if args.p2_variant == "longitudinal_mk":
            for _f in range(4):
                _ls = build_splits_longitudinal(args.samples_dir, args.splits_csv,
                                                fold=_f, split=args.split)
                for _pats in _ls.values():
                    for _pat in _pats:
                        all_stems.update(_pat["stems"])

        print(f"\n  [all-folds] Loading {len(all_stems)} unique stems once for all folds/tasks ...")
        bag_cache = preload_bags(list(all_stems), args.samples_dir, n_workers=args.workers)

        _orig_task = args.task
        for _task in _task_list:
            args.task = _task
            print(f"\n{'='*60}\n  [all-folds] task={_task}\n{'='*60}")

            # Folds 1-3: HP sweep only (no combined training)
            for _f in [1, 2, 3]:
                print(f"\n{'='*60}\n  [all-folds] task={_task}  fold={_f}  (HP sweep only)\n{'='*60}")
                _sd = build_splits_multitask(args.samples_dir, args.splits_csv,
                                             fold=_f, split=args.split)
                if args.p2_variant == "longitudinal_mk":
                    run_phase2_longitudinal(args, bag_cache, device, out, fold=_f)
                else:
                    run_phase2(args, _sd, bag_cache, device, out, fold=_f)

            # Fold 0: HP sweep + aggregate global HP + combined train
            print(f"\n{'='*60}\n  [all-folds] task={_task}  fold=0  (HP sweep + global HP + combined train)\n{'='*60}")
            _sd0 = build_splits_multitask(args.samples_dir, args.splits_csv,
                                          fold=0, split=args.split)
            _orig_global_hp      = args.global_hp
            _orig_combined_train = args.combined_train
            args.global_hp       = True
            args.combined_train  = True
            if args.p2_variant == "longitudinal_mk":
                metrics = run_phase2_longitudinal(args, bag_cache, device, out, fold=0)
            else:
                metrics = run_phase2(args, _sd0, bag_cache, device, out, fold=0)
            args.global_hp      = _orig_global_hp
            args.combined_train = _orig_combined_train

            summary_path = out / f"metrics_{_tag(args.split, 0)}_{args.p2_variant}_{args.task}.json"
            with open(summary_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\n  Results → {summary_path}")

        args.task = _orig_task
        return

    # ── Single-fold path (original behaviour) ─────────────────────────────────
    # ── Load splits ───────────────────────────────────────────────────────────
    splits_dict = build_splits_multitask(
        samples_dir=args.samples_dir,
        splits_csv=args.splits_csv,
        fold=args.fold,
        split=args.split,
    )
    all_recs  = (splits_dict["train"] + splits_dict["val"] + splits_dict["test"])
    all_stems = list({r["stem"] for r in all_recs})

    # For longitudinal_mk: stems come from patient-level splits (same stems, but ensure all included)
    if args.p2_variant == "longitudinal_mk" and args.phase in ("p2", "both"):
        _long_splits = build_splits_longitudinal(
            samples_dir=args.samples_dir,
            splits_csv=args.splits_csv,
            fold=args.fold,
            split=args.split,
        )
        _long_stems = {s for pat_list in _long_splits.values()
                       for pat in pat_list for s in pat["stems"]}
        all_stems = list(set(all_stems) | _long_stems)

    # ── Preload bags ──────────────────────────────────────────────────────────
    bag_cache = preload_bags(all_stems, args.samples_dir, n_workers=args.workers)

    # ── Run phases ────────────────────────────────────────────────────────────
    p1_paths = None
    if args.phase in ("p1", "both"):
        p1_paths = run_phase1(args, splits_dict, bag_cache, device, out)

    if args.phase in ("p2", "both"):
        if args.p2_variant == "longitudinal_mk":
            metrics = run_phase2_longitudinal(args, bag_cache, device, out)
        else:
            metrics = run_phase2(args, splits_dict, bag_cache, device, out,
                                 p1_paths=p1_paths)
        summary_path = out / f"metrics_{tag}_{args.p2_variant}_{args.task}.json"
        with open(summary_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Results → {summary_path}")


if __name__ == "__main__":
    main()
