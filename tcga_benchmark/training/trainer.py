"""
TCGA survival training loop.

Full-batch Cox-Breslow loss (same approach as lung transplant pipeline).
HP sweep on val → retrain on train+val → evaluate on test.
"""
import json, math, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index

# ── Cox loss (Breslow) ────────────────────────────────────────────────────────

def cox_loss(hazards: torch.Tensor, times: torch.Tensor,
             events: torch.Tensor) -> Optional[torch.Tensor]:
    """Full-batch Cox-Breslow loss. Returns None if no events."""
    if events.sum() == 0:
        return None
    # Sort by descending time
    order    = torch.argsort(times, descending=True)
    h, e     = hazards[order], events[order]
    log_risk = torch.logcumsumexp(h, dim=0)
    loss     = -(h - log_risk)[e.bool()].mean()
    return loss


def c_index(hazards: List[float], times: List[float],
            events: List[float]) -> float:
    if sum(events) == 0:
        return 0.5
    try:
        return float(concordance_index(times, [-h for h in hazards], events))
    except Exception:
        return 0.5


# ── Training epoch ────────────────────────────────────────────────────────────

def train_epoch(model: nn.Module, records: List[dict],
                optimizer, device: torch.device,
                bag_cache: dict, scaler,
                grad_accum: int = 4) -> float:
    model.train()
    random.shuffle(records)

    haz_buf, t_buf, e_buf = [], [], []
    accum_step = 0
    optimizer.zero_grad()

    for rec in records:
        bags = bag_cache.get(rec["stem"])
        if bags is None:
            continue
        try:
            use_amp = (device.type == "cuda")
            with torch.amp.autocast("cuda", enabled=use_amp):
                h = model(bags, device)
            if not (h.requires_grad and torch.isfinite(h)):
                continue
            haz_buf.append(h)
            t_buf.append(rec["os_time"])
            e_buf.append(rec["os_event"])
            accum_step += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            haz_buf.clear(); t_buf.clear(); e_buf.clear(); accum_step = 0
            continue
        except Exception:
            continue

        if accum_step >= grad_accum:
            _flush(haz_buf, t_buf, e_buf, optimizer, scaler, device)
            haz_buf.clear(); t_buf.clear(); e_buf.clear(); accum_step = 0

    if haz_buf:
        _flush(haz_buf, t_buf, e_buf, optimizer, scaler, device)

    return 0.0


def _flush(haz_buf, t_buf, e_buf, optimizer, scaler, device):
    if not haz_buf:
        return
    h = torch.stack(haz_buf)
    t = torch.tensor(t_buf, device=device)
    e = torch.tensor(e_buf, device=device)
    loss = cox_loss(h, t, e)
    if loss is None or not torch.isfinite(loss):
        optimizer.zero_grad(); return
    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [p for p in optimizer.param_groups[0]["params"]], 1.0)
        scaler.step(optimizer); scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in optimizer.param_groups[0]["params"]], 1.0)
        optimizer.step()
    optimizer.zero_grad()


@torch.no_grad()
def evaluate(model: nn.Module, records: List[dict],
             device: torch.device, bag_cache: dict) -> float:
    model.eval()
    hazards, times, events = [], [], []
    for rec in records:
        bags = bag_cache.get(rec["stem"])
        if bags is None:
            continue
        try:
            h = model(bags, device)
            if torch.isfinite(h):
                hazards.append(h.item())
                times.append(rec["os_time"])
                events.append(rec["os_event"])
        except Exception:
            continue
    return c_index(hazards, times, events)


# ── HP sweep + final training ─────────────────────────────────────────────────

LR_GRID = [1e-4, 5e-5, 1e-5]
WD_GRID  = [1e-3, 1e-4]
N_EPOCHS_SWEEP  = 20
N_EPOCHS_FINAL  = 40
EVAL_EVERY      = 5
PATIENCE        = 8
GRAD_ACCUM      = 8

# Alternating task weights: recon 20%, surv 80%
RECON_PROB = 0.20


# ── Reconstruction epoch (for GeoMAE-SlotMIL alternating training) ────────────

def recon_epoch(model: nn.Module, records: List[dict],
                optimizer, device: torch.device,
                bag_cache: dict, scaler,
                mask_ratio: float = 0.5,
                grad_accum: int = GRAD_ACCUM) -> float:
    """
    One reconstruction epoch — applies BFS masking to WSI patches and
    minimises the spatial denoising loss via the GeoMAE backbone.
    Skips silently for models without a GeoMAE backbone.
    """
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2] / "src"))
    try:
        from mil.models.encoders import GeoMAESpatialBackbone
    except ImportError:
        return 0.0

    # Only runs for GeoMAESlotMIL
    backbone = getattr(model, "backbone", None)
    if not isinstance(backbone, GeoMAESpatialBackbone):
        return 0.0

    model.train()
    random.shuffle(records)
    total = 0.0; n = 0; accum = 0
    optimizer.zero_grad()

    for rec in records:
        bags = bag_cache.get(rec["stem"])
        if bags is None: continue
        wsi    = bags.get("WSI")
        coords = bags.get("WSI_coords")
        if wsi is None or coords is None: continue

        try:
            use_amp = (device.type == "cuda")
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = backbone.forward_recon(
                    wsi.to(device), coords.to(device), mask_ratio)
            if loss.requires_grad and torch.isfinite(loss):
                loss = loss / grad_accum
                if scaler: scaler.scale(loss).backward()
                else: loss.backward()
                total += loss.item() * grad_accum
                n += 1; accum += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); optimizer.zero_grad(); accum = 0
            continue
        except Exception:
            continue

        if accum >= grad_accum:
            if scaler:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(); accum = 0

    if accum > 0:
        if scaler:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()

    return total / max(n, 1)


def train_and_eval(
    model_factory,            # () → fresh model on CPU
    train_recs:  List[dict],
    val_recs:    List[dict],
    test_recs:   List[dict],
    bag_cache:   dict,
    save_dir:    Path,
    n_epochs:    int = N_EPOCHS_FINAL,
    lr_grid:     list = LR_GRID,
    wd_grid:     list = WD_GRID,
    seed:        int  = 42,
) -> dict:
    """
    1. HP sweep (lr × wd) on val C-index.
    2. Retrain best HP on train+val.
    3. Evaluate on test.
    Returns {val_cidx, test_cidx, best_lr, best_wd, best_ep}.
    """
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resume if done
    result_path = save_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    # ── HP sweep ──────────────────────────────────────────────────────────────
    best_val, best_lr, best_wd = -1.0, lr_grid[0], wd_grid[0]
    for lr in lr_grid:
        for wd in wd_grid:
            model = model_factory().to(device)
            opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            scal  = torch.amp.GradScaler("cuda") if device.type=="cuda" else None
            for ep in range(N_EPOCHS_SWEEP):
                train_epoch(model, train_recs, opt, device, bag_cache,
                            scal, GRAD_ACCUM)
            val_ci = evaluate(model, val_recs, device, bag_cache)
            print(f"  [HP] lr={lr:.0e} wd={wd:.0e}  val_cidx={val_ci:.4f}")
            if val_ci > best_val:
                best_val, best_lr, best_wd = val_ci, lr, wd
            del model

    print(f"  → best HP: lr={best_lr:.0e}  wd={best_wd:.0e}  val={best_val:.4f}")

    # ── Retrain on train+val ──────────────────────────────────────────────────
    trainval_recs = train_recs + val_recs
    model = model_factory().to(device)
    opt   = torch.optim.Adam(model.parameters(),
                             lr=best_lr, weight_decay=best_wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs, eta_min=best_lr*0.01)
    scal  = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val2, best_ep, no_imp = -1.0, 0, 0
    best_state = None

    for ep in range(1, n_epochs + 1):
        # Alternating: RECON_PROB chance of reconstruction epoch (GeoMAE only)
        if random.random() < RECON_PROB:
            recon_epoch(model, trainval_recs, opt, device, bag_cache, scal)
        else:
            train_epoch(model, trainval_recs, opt, device, bag_cache, scal, GRAD_ACCUM)
        sched.step()
        if ep % EVAL_EVERY == 0:
            val_ci = evaluate(model, val_recs, device, bag_cache)
            print(f"  ep {ep:3d}  val_cidx={val_ci:.4f}")
            if val_ci > best_val2:
                best_val2 = val_ci; best_ep = ep; no_imp = 0
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                no_imp += 1
            if no_imp >= PATIENCE:
                print(f"  Early stop ep={ep}")
                break

    # Load best checkpoint and evaluate on test
    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()
    test_ci = evaluate(model, test_recs, device, bag_cache)
    print(f"  → test_cidx={test_ci:.4f}  (best val ep={best_ep}  val={best_val2:.4f})")

    result = {
        "val_cidx":  round(best_val2, 4),
        "test_cidx": round(test_ci,   4),
        "best_lr":   best_lr,
        "best_wd":   best_wd,
        "best_ep":   best_ep,
    }
    result_path.write_text(json.dumps(result, indent=2))

    torch.save({"model": best_state, "result": result},
               save_dir / "best_model.pt")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
