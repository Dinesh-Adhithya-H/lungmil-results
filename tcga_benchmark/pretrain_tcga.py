#!/usr/bin/env python3
"""
GeoMAE pretraining on TCGA WSI patches.

Same BFS-flood spatial contiguous masking + denoising objective as lung
transplant pretraining — but trained on cancer-specific tissue from TCGA H5 files.

After pretraining, the backbone is used in GeoMAE-SlotMIL for survival prediction
with alternating supervised (Cox) / reconstruction (recon) epochs.

Usage:
  python pretrain_tcga.py \\
      --cancers KIRC BRCA BLCA LGG GBM \\
      --out-dir results/tcga_geomae_pretrain \\
      --n-epochs 200
"""
import sys, argparse, random, json, math
from pathlib import Path
from typing import List, Optional, Dict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

from mil.models.pretrain import (
    GeoMAE, SpatialDenoisingEncoder,
    build_knn_graph, bfs_distances, contiguous_region_mask)

# ── Config ────────────────────────────────────────────────────────────────────
H5_DIRS = {
    "KIRC": "/lustre/groups/aih/dinesh.haridoss/mil/TCGA-KIRC",
    "BRCA": "/lustre/groups/aih/dinesh.haridoss/mil/TCGA-BRCA",
    "BLCA": "/lustre/groups/aih/dinesh.haridoss/mil/TCGA-BLCA",
    "LGG":  "/lustre/groups/aih/dinesh.haridoss/mil/TCGA-LGG",
    "GBM":  "/lustre/groups/aih/dinesh.haridoss/mil/TCGA-GBM",
}

PT_LR           = 1e-4
PT_WEIGHT_DECAY = 1e-4
PT_EVAL_EVERY   = 10
PT_GRAD_ACCUM   = 4
MAX_PATCHES     = 8000   # subsample per patient per epoch


def _gc():
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ── Data loading from H5 ──────────────────────────────────────────────────────

def _load_h5(h5_path: Path, max_patches: int, device: torch.device
             ) -> Optional[Dict[str, torch.Tensor]]:
    """Load features + spatial coords from one TCGA H5 file."""
    try:
        with h5py.File(h5_path, "r") as f:
            feats  = torch.from_numpy(f["features"][0]).float()    # (N, 1536)
            coords = torch.from_numpy(f["coords_patching"][:]).float()  # (N, 2)
    except Exception:
        return None

    # Drop NaN
    ok = ~torch.isnan(feats).any(1)
    feats = feats[ok]; coords = coords[ok]
    if len(feats) < 20:
        return None

    # Subsample
    if len(feats) > max_patches:
        idx = torch.randperm(len(feats))[:max_patches]
        feats = feats[idx]; coords = coords[idx]

    return {"feats": feats.to(device), "coords": coords.to(device)}


def collect_h5_files(cancers: List[str]) -> List[Path]:
    files = []
    for cancer in cancers:
        h5_dir = Path(H5_DIRS.get(cancer, ""))
        if h5_dir.exists():
            cancer_files = sorted(h5_dir.glob("*.h5"))
            files.extend(cancer_files)
            print(f"  {cancer}: {len(cancer_files)} H5 files")
    print(f"  Total: {len(files)} files")
    return files


# ── Training ──────────────────────────────────────────────────────────────────

def pretrain_epoch(encoder: SpatialDenoisingEncoder,
                   h5_files: List[Path],
                   optimizer, device: torch.device,
                   scaler, grad_accum: int,
                   mask_ratio: float,
                   max_patches: int) -> Dict[str, float]:
    encoder.train()
    random.shuffle(h5_files)

    total_loss = 0.0; n = 0; accum = 0
    optimizer.zero_grad()

    for h5_path in h5_files:
        data = _load_h5(h5_path, max_patches, device)
        if data is None:
            continue
        feats  = data["feats"]
        coords = data["coords"]

        try:
            use_amp = (scaler is not None)
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Build KNN graph
                ei, ew = build_knn_graph(coords, encoder.knn_k)
                # BFS contiguous masking
                visible = contiguous_region_mask(coords, mask_ratio, ei)
                # Forward
                out  = encoder(feats, coords, visible_mask=visible,
                               precomputed_graph=(ei, ew))
                # Loss: weighted MSE on masked patches
                masked = ~out["visible_mask"]
                if not masked.any():
                    continue
                eps_pred = out["noise_pred"][masked]
                eps_true = out["noise_true"][masked]
                d_w = (out["distances"][masked].float() /
                       max(out["distances"].max().item(), 1))
                loss = (d_w * F.mse_loss(eps_pred, eps_true,
                                         reduction="none").mean(-1)).mean()
                loss = loss / grad_accum

            if not (loss.requires_grad and torch.isfinite(loss)):
                continue

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * grad_accum
            n += 1; accum += 1

        except torch.cuda.OutOfMemoryError:
            _gc(); optimizer.zero_grad(); accum = 0
            continue
        except Exception:
            continue

        if accum >= grad_accum:
            if scaler:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(); accum = 0; _gc()

    # Final flush
    if accum > 0:
        if scaler:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()

    return {"loss": total_loss / max(n, 1), "n_patients": n}


@torch.no_grad()
def eval_epoch(encoder: SpatialDenoisingEncoder,
               h5_files: List[Path],
               device: torch.device,
               mask_ratio: float, max_patches: int,
               n_eval: int = 50) -> float:
    encoder.eval()
    total = 0.0; n = 0
    for h5_path in random.sample(h5_files, min(n_eval, len(h5_files))):
        data = _load_h5(h5_path, max_patches, device)
        if data is None:
            continue
        try:
            feats  = data["feats"]; coords = data["coords"]
            ei, ew = build_knn_graph(coords, encoder.knn_k)
            visible = contiguous_region_mask(coords, mask_ratio, ei)
            out     = encoder(feats, coords, visible_mask=visible,
                              precomputed_graph=(ei, ew))
            masked  = ~out["visible_mask"]
            if masked.any():
                loss = F.mse_loss(out["noise_pred"][masked],
                                  out["noise_true"][masked])
                if torch.isfinite(loss):
                    total += loss.item(); n += 1
        except Exception:
            continue
    return total / max(n, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--cancers",     nargs="+", default=list(H5_DIRS.keys()))
    p.add_argument("--out-dir",     default="results/tcga_geomae_pretrain")
    p.add_argument("--n-epochs",    type=int,   default=200)
    p.add_argument("--lr",          type=float, default=PT_LR)
    p.add_argument("--weight-decay",type=float, default=PT_WEIGHT_DECAY)
    p.add_argument("--hidden-dim",  type=int,   default=256)
    p.add_argument("--n-layers",    type=int,   default=3)
    p.add_argument("--n-heads",     type=int,   default=4)
    p.add_argument("--knn-k",       type=int,   default=8)
    p.add_argument("--mask-ratio",  type=float, default=0.5)
    p.add_argument("--max-patches", type=int,   default=MAX_PATCHES)
    p.add_argument("--grad-accum",  type=int,   default=PT_GRAD_ACCUM)
    p.add_argument("--val-frac",    type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=42)
    return p


def main():
    args = _parser().parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nCollecting H5 files for: {args.cancers}")
    all_files = collect_h5_files(args.cancers)
    random.shuffle(all_files)
    n_val      = max(1, int(len(all_files) * args.val_frac))
    val_files  = all_files[:n_val]
    train_files= all_files[n_val:]
    print(f"Train: {len(train_files)}  Val: {len(val_files)}")

    # Build encoder (WSI-only, feat_dim=1536)
    encoder = SpatialDenoisingEncoder(
        feat_dim   = 1536,
        hidden_dim = args.hidden_dim,
        n_layers   = args.n_layers,
        n_heads    = args.n_heads,
        knn_k      = args.knn_k,
        max_dist   = 32,
    ).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder params: {n_params:,}")

    optimizer = torch.optim.AdamW(encoder.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.n_epochs, eta_min=args.lr * 0.01)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Resume
    ckpt_dir  = out_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    best_path = out_dir / "best_backbone.pt"
    start_ep  = 0; best_val = float("inf")
    existing  = sorted(ckpt_dir.glob("ep*.pt"))
    if existing:
        ckpt = torch.load(existing[-1], map_location="cpu", weights_only=False)
        encoder.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler and ckpt.get("scaler"): scaler.load_state_dict(ckpt["scaler"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_ep = ckpt["epoch"]; best_val = ckpt.get("best_val", float("inf"))
        print(f"Resumed from ep {start_ep}  best_val={best_val:.4f}")

    print(f"\nPretraining for {args.n_epochs} epochs...")
    for epoch in range(start_ep, args.n_epochs):
        stats = pretrain_epoch(encoder, train_files, optimizer, device, scaler,
                               args.grad_accum, args.mask_ratio, args.max_patches)
        scheduler.step()

        log = (f"ep {epoch+1:4d}  loss={stats['loss']:.4f}  "
               f"lr={optimizer.param_groups[0]['lr']:.1e}  "
               f"n={stats['n_patients']}")

        if (epoch + 1) % PT_EVAL_EVERY == 0:
            val_loss = eval_epoch(encoder, val_files, device,
                                  args.mask_ratio, args.max_patches)
            log += f"  val={val_loss:.4f}"

            torch.save({
                "epoch": epoch+1, "model": encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "scheduler": scheduler.state_dict(),
                "best_val": best_val,
            }, ckpt_dir / f"ep{epoch+1:04d}.pt")

            if val_loss < best_val:
                best_val = val_loss
                # Save as backbone weights dict (compatible with load_geomae_weights)
                torch.save({"he_encoder": encoder.state_dict()}, best_path)
                log += "  [best]"
            _gc()

        print(log, flush=True)

    print(f"\nPretraining done. Best backbone → {best_path}")
    with open(out_dir / "pretrain_status.json", "w") as f:
        json.dump({"completed": True, "best_val": best_val,
                   "n_epochs": args.n_epochs,
                   "cancers": args.cancers}, f, indent=2)


if __name__ == "__main__":
    main()
