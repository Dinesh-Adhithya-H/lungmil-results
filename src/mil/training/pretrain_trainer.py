"""
GeoMAE pretraining trainer.

Trains the spatial denoising autoencoder (GeoMAE) on all patients
without using any labels — pure self-supervised reconstruction.

Pipeline per patient per epoch:
  1. Load HE patches + 2D pixel coords → spatial KNN graph → contiguous mask
  2. Load CT patches + 3D voxel coords → spatial KNN graph → block mask
  3. Load Clinical features → category-block mask
  4. Forward through GeoMAE (spatial denoising + slot cross-modal)
  5. Loss = λ_HE * L_recon_HE + λ_CT * L_recon_CT + λ_clin * L_clin
  6. Backward + optimizer step

Checkpoints saved every eval_every epochs.
Best model selected by lowest total reconstruction loss on val patients.
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from mil.models.pretrain import GeoMAE, build_knn_graph


# ── Constants ─────────────────────────────────────────────────────────────────
PT_LR           = 1e-4
PT_WEIGHT_DECAY = 1e-4
PT_EPOCHS       = 300
PT_EVAL_EVERY   = 10
PT_GRAD_ACCUM   = 4
PT_MAX_HE_PAT   = 2000    # reduced from 4000 — avoids OOM, more patients per epoch
PT_MAX_CT_PAT   = 256


def _gc():
    import ctypes
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except: pass
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_patient(pt_path: str, device: torch.device, max_he: int, max_ct: int
                  ) -> Optional[Dict[str, torch.Tensor]]:
    """Load one patient's spatial data from .pt file."""
    try:
        d   = torch.load(pt_path, map_location="cpu", weights_only=False)
        inp = d.get("inputs", {})
        isc = d.get("instance_spatial_coords", {})

        out = {}

        he_feats  = inp.get("HE_cells")
        he_coords = isc.get("HE_cells")
        if he_feats is not None and he_coords is not None:
            he_feats  = he_feats.float()
            he_coords = he_coords.float()
            # Drop NaN patches
            valid = ~torch.isnan(he_feats).any(dim=1)
            he_feats  = he_feats[valid]; he_coords = he_coords[valid]
            if len(he_feats) > max_he:
                idx = torch.randperm(len(he_feats))[:max_he]
                he_feats = he_feats[idx]; he_coords = he_coords[idx]
            if len(he_feats) >= 20:
                out["he_feats"]  = he_feats.to(device)
                out["he_coords"] = he_coords.to(device)

        ct_feats  = inp.get("CT_cells")
        ct_coords = isc.get("CT_cells")
        if ct_feats is not None and ct_coords is not None:
            ct_feats  = ct_feats.float()
            ct_coords = ct_coords.float()
            valid = ~torch.isnan(ct_feats).any(dim=1)
            ct_feats  = ct_feats[valid]; ct_coords = ct_coords[valid]
            if len(ct_feats) > max_ct:
                idx = torch.randperm(len(ct_feats))[:max_ct]
                ct_feats = ct_feats[idx]; ct_coords = ct_coords[idx]
            if len(ct_feats) >= 10:
                out["ct_feats"]  = ct_feats.to(device)
                out["ct_coords"] = ct_coords.to(device)

        clin = inp.get("Clinical")
        if clin is not None and not torch.isnan(clin).any():
            out["clin_feats"] = clin.float().to(device)

        return out if out else None
    except Exception:
        return None


# ── Pretraining loop ──────────────────────────────────────────────────────────

def pretrain_epoch(model: GeoMAE, pt_files: List[str], optimizer,
                   device: torch.device, scaler, grad_accum: int,
                   max_he: int, max_ct: int) -> Dict[str, float]:
    """One full pass over all patients."""
    model.train()
    random.shuffle(pt_files)

    total_loss = total_he = total_ct = total_clin = 0.0
    n_patients = 0; accum_step = 0
    pending_loss: Optional[torch.Tensor] = None
    optimizer.zero_grad()

    for pt_path in pt_files:
        data = _load_patient(pt_path, device, max_he, max_ct)
        if data is None:
            continue

        try:
            use_amp = scaler is not None
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(
                    he_feats  = data.get("he_feats"),
                    he_coords = data.get("he_coords"),
                    ct_feats  = data.get("ct_feats"),
                    ct_coords = data.get("ct_coords"),
                    clin_feats= data.get("clin_feats"),
                )

            loss = out["loss"] / grad_accum
            if not (loss.requires_grad and torch.isfinite(loss)):
                continue

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            pending_loss = loss
            total_loss  += loss.item() * grad_accum
            if out.get("loss_he")   is not None: total_he   += out["loss_he"].item()
            if out.get("loss_ct")   is not None: total_ct   += out["loss_ct"].item()
            if out.get("loss_clin") is not None: total_clin += out["loss_clin"].item()
            n_patients += 1; accum_step += 1

        except torch.cuda.OutOfMemoryError:
            _gc(); optimizer.zero_grad(); pending_loss = None; accum_step = 0
            continue
        except Exception as e:
            print(f"  [pretrain] skip {pt_path}: {type(e).__name__}: {e}", flush=True)
            continue

        if accum_step >= grad_accum:
            if scaler:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
            pending_loss = None; accum_step = 0
            _gc()

    # Final flush
    if accum_step > 0 and pending_loss is not None:
        if scaler:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()

    n = max(n_patients, 1)
    return {"loss": total_loss/n, "he": total_he/n,
            "ct": total_ct/n, "clin": total_clin/n,
            "n_patients": n_patients}


@torch.no_grad()
def eval_epoch(model: GeoMAE, pt_files: List[str], device: torch.device,
               max_he: int, max_ct: int) -> float:
    """Compute average reconstruction loss on a held-out set."""
    model.eval()
    total_loss = 0.0; n = 0

    for pt_path in random.sample(pt_files, min(50, len(pt_files))):
        data = _load_patient(pt_path, device, max_he, max_ct)
        if data is None:
            continue
        try:
            out = model(
                he_feats  = data.get("he_feats"),
                he_coords = data.get("he_coords"),
                ct_feats  = data.get("ct_feats"),
                ct_coords = data.get("ct_coords"),
                clin_feats= data.get("clin_feats"),
            )
            if torch.isfinite(out["loss"]):
                total_loss += out["loss"].item(); n += 1
        except Exception:
            continue
    return total_loss / max(n, 1)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_pretrain(
    samples_dir: str,
    save_dir: Path,
    n_epochs:      int   = PT_EPOCHS,
    eval_every:    int   = PT_EVAL_EVERY,
    lr:            float = PT_LR,
    weight_decay:  float = PT_WEIGHT_DECAY,
    grad_accum:    int   = PT_GRAD_ACCUM,
    hidden_dim:    int   = 256,
    n_layers:      int   = 5,
    n_heads:       int   = 4,
    n_slots:       int   = 8,
    knn_k:         int   = 32,
    he_mask_ratio: float = 0.5,
    ct_mask_ratio: float = 0.5,
    val_frac:      float = 0.1,
    splits_csv:    Optional[str] = None,
    split:         int   = 1,
    seed:          int   = 42,
) -> Path:
    """
    Full GeoMAE pretraining run.
    Returns path to best backbone checkpoint.
    """
    import os
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Collect patient .pt files — use only train-split stems if splits_csv given
    all_files = sorted([str(p) for p in Path(samples_dir).glob("*.pt")])

    if splits_csv and Path(splits_csv).exists():
        import pandas as pd
        df = pd.read_csv(splits_csv)
        fold_col = f"split{split}_fold0"   # use fold0 train as pretraining set
        if fold_col in df.columns:
            train_stems = set(
                Path(str(r)).stem for r in
                df[df[fold_col] == "train"]["file"].dropna())
            # Also include fold 1-3 train stems for maximum pretraining data
            for fold in [1, 2, 3]:
                fc = f"split{split}_fold{fold}"
                if fc in df.columns:
                    train_stems |= set(
                        Path(str(r)).stem for r in
                        df[df[fc] == "train"]["file"].dropna())
            all_files = [f for f in all_files
                         if Path(f).stem in train_stems]
            print(f"Using train-split only: {len(all_files)} files "
                  f"(excluded val/test to prevent data leakage)")
        else:
            print(f"Warning: {fold_col} not found in splits CSV — using all files")

    random.shuffle(all_files)
    n_val = max(1, int(len(all_files) * val_frac))
    val_files   = all_files[:n_val]
    train_files = all_files[n_val:]
    print(f"Pretraining patients: {len(train_files)} train  |  {len(val_files)} val")

    # Build model
    model = GeoMAE(
        he_feat_dim   = 1024,
        ct_feat_dim   = 1024,
        n_clin_feats  = 102,
        hidden_dim    = hidden_dim,
        n_layers      = n_layers,
        n_heads       = n_heads,
        n_slots       = n_slots,
        knn_k         = knn_k,
        he_mask_ratio = he_mask_ratio,
        ct_mask_ratio = ct_mask_ratio,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"GeoMAE params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr*0.01)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Resume from checkpoint if exists
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    best_path   = save_dir / "best_backbone.pt"
    status_path = save_dir / "pretrain_status.json"
    start_epoch = 0
    best_val    = float("inf")

    existing_ckpts = sorted(ckpt_dir.glob("ep*.pt"))
    if existing_ckpts:
        ckpt = torch.load(existing_ckpts[-1], map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        if scaler and ckpt.get("scaler"): scaler.load_state_dict(ckpt["scaler"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]; best_val = ckpt.get("best_val", float("inf"))
        print(f"Resumed from ep {start_epoch}  best_val={best_val:.4f}")

    history = {"train_loss": [], "val_loss": [], "lr": []}

    print(f"\nPretraining GeoMAE for {n_epochs} epochs...")
    for epoch in range(start_epoch, n_epochs):
        train_stats = pretrain_epoch(
            model, train_files, optimizer, device, scaler, grad_accum,
            PT_MAX_HE_PAT, PT_MAX_CT_PAT)
        scheduler.step()
        tl = train_stats["loss"]
        history["train_loss"].append(tl)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        log = (f"ep {epoch+1:4d}  loss={tl:.4f}  "
               f"he={train_stats['he']:.3f}  ct={train_stats['ct']:.3f}  "
               f"clin={train_stats['clin']:.3f}  "
               f"lr={optimizer.param_groups[0]['lr']:.1e}  "
               f"n={train_stats['n_patients']}")

        if (epoch + 1) % eval_every == 0:
            val_loss = eval_epoch(model, val_files, device,
                                  PT_MAX_HE_PAT, PT_MAX_CT_PAT)
            history["val_loss"].append(val_loss)
            log += f"  val={val_loss:.4f}"

            # Save checkpoint
            torch.save({
                "epoch": epoch + 1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "scheduler": scheduler.state_dict(),
                "best_val": best_val,
            }, ckpt_dir / f"ep{epoch+1:04d}.pt")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.get_backbone_weights(), best_path)
                log += "  [best]"

            _gc()

        print(log, flush=True)

    # Save final status
    with open(status_path, "w") as f:
        json.dump({"completed": True, "best_val": best_val,
                   "n_epochs": n_epochs}, f, indent=2)
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f)

    print(f"\nPretraining done. Best backbone → {best_path}")
    return best_path
