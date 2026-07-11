"""
Phase 1 training loops and runner (v8 design).

v8 design — Phase 1 purpose
-----------------------------
Phase 1 trains each modality encoder independently.
Loss: hinge classification (ACR) and/or Cox-Breslow survival. Nothing else.
No CLR, KD, CRD, or cross-modal objectives — these optimise a different
objective and can hurt the task-predictive quality of Phase 2 inputs.

Functions exported
------------------
p1_train_one_epoch
p1_train_one_epoch_survival
p1_evaluate
p1_evaluate_survival
run_phase1_modality
run_phase1_hp_sweep
"""

import json
import math
import random
from itertools import product as iproduct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None
    _WANDB_AVAILABLE = False

from mil.models.phase1 import SingleModalMIL
from mil.data.registry import MODALITIES, _feat_dim, _pres_col
from mil.training.losses import (
    hinge_loss, compute_class_weights,
    cox_breslow_loss, c_index,
)
from mil.training.metrics import compute_metrics

# ── Type aliases ──────────────────────────────────────────────────────────────
BagCache = Dict[str, Dict[str, Optional[torch.Tensor]]]

# ── Phase 1 hyperparameter defaults ──────────────────────────────────────────
HIDDEN_DIM      = 256
DROPOUT         = 0.4

P1_LR           = 1e-5
P1_WEIGHT_DECAY = 1e-3
P1_EPOCHS       = 150
P1_EVAL_EVERY   = 25
P1_GRAD_ACCUM   = 4

# HP sweep search space (used by run_phase1_hp_sweep)
HP_LR_GRID      = [1e-5, 5e-5, 1e-4]
HP_WD_GRID      = [1e-4, 1e-3, 1e-2, 1e-1]
HP_SWEEP_EPOCHS = 100   # shorter run per HP candidate; pick best on val BACC


# ── Internal helpers ──────────────────────────────────────────────────────────

def _malloc_trim():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _gc():
    _malloc_trim()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_status(path: Path, completed: bool, **kwargs) -> None:
    data = {"completed": completed, **kwargs}
    tmp  = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _read_status(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _is_completed(save_dir: Path) -> bool:
    s = _read_status(save_dir / "status.json")
    return s is not None and s.get("completed", False)


def _find_resume_epoch(ckpt_dir: Path) -> int:
    if not ckpt_dir.exists():
        return 0
    epochs = []
    for cp in ckpt_dir.glob("ep*.pt"):
        try:
            epochs.append(int(cp.stem[2:]))
        except ValueError:
            pass
    return max(epochs) if epochs else 0


def _load_checkpoint(ckpt_dir: Path, epoch: int) -> Optional[dict]:
    path = ckpt_dir / f"ep{epoch:04d}.pt"
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [warn] failed to load {path}: {e}")
        return None


# ── Phase 1 classification training epoch ────────────────────────────────────

def p1_train_one_epoch(
    model: SingleModalMIL,
    records: List[dict],
    mod_name: str,
    optimizer: torch.optim.Optimizer,
    cw: Tuple[float, float],
    device: torch.device,
    bag_cache: BagCache,
    scaler: Optional[torch.amp.GradScaler],
    grad_accum: int,
    use_spatial: bool = False,
) -> float:
    """Hinge classification loss only. Returns mean loss."""
    model.train()
    random.shuffle(records)
    use_spatial_for_mod = use_spatial and mod_name == "HE"

    total_loss = 0.0; n_steps = 0; accum_step = 0
    grad_accumulated = False
    optimizer.zero_grad()

    for rec in records:
        label = rec.get("label")
        if label is None:
            continue
        bag = bag_cache.get(rec["stem"], {}).get(mod_name)
        if bag is None:
            continue

        bag_dev    = bag.to(device, non_blocking=True)
        target     = torch.tensor([label], dtype=torch.float32, device=device)
        he_coords  = bag_cache.get(rec["stem"], {}).get("HE_coords") if use_spatial_for_mod else None
        use_amp    = scaler is not None

        with torch.amp.autocast("cuda", enabled=use_amp):
            logit = model(bag_dev, coords=he_coords)
            loss  = hinge_loss(logit.unsqueeze(0), target, cw) / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * grad_accum
        n_steps    += 1
        accum_step += 1
        grad_accumulated = True

        if accum_step == grad_accum:
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            accum_step = 0; grad_accumulated = False

        del bag_dev, target
        if n_steps % 200 == 0:
            _gc()

    if accum_step > 0 and grad_accumulated:
        if scaler is not None:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(n_steps, 1)


# ── Phase 1 survival training epoch ──────────────────────────────────────────

def p1_train_one_epoch_survival(
    model: SingleModalMIL,
    records: List[dict],
    mod_name: str,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    bag_cache: BagCache,
    scaler: Optional[torch.amp.GradScaler],
    grad_accum: int,
    surv_endpoint: str = "clad",
    use_spatial: bool = False,
) -> float:
    """Cox-Breslow loss only. Returns mean loss."""
    model.train()
    random.shuffle(records)
    use_spatial_for_mod = use_spatial and mod_name == "HE"
    use_amp = scaler is not None

    cox_buffer: list = []
    total_loss = 0.0; n_steps = 0; accum_step = 0
    optimizer.zero_grad()

    def _step():
        nonlocal total_loss, n_steps
        L = cox_breslow_loss(cox_buffer)
        if L is not None and L.requires_grad:
            if scaler:
                scaler.scale(L).backward()
                scaler.step(optimizer); scaler.update()
            else:
                L.backward(); optimizer.step()
            total_loss += L.item(); n_steps += 1
        optimizer.zero_grad()
        cox_buffer.clear()
        _gc()

    for rec in records:
        # ACR uses non-standard column names
        _t_key = "tte_next_acr"  if surv_endpoint == "acr" else f"{surv_endpoint}_time"
        _e_key = "event_next_acr" if surv_endpoint == "acr" else f"{surv_endpoint}_event"
        surv_t = rec.get(_t_key, float("nan"))
        surv_e = rec.get(_e_key, float("nan"))
        if not isinstance(surv_t, float) or math.isnan(surv_t):
            continue
        bag = bag_cache.get(rec["stem"], {}).get(mod_name)
        if bag is None:
            continue

        bag_dev   = bag.to(device, non_blocking=True)
        he_coords = bag_cache.get(rec["stem"], {}).get("HE_coords") if use_spatial_for_mod else None

        with torch.amp.autocast("cuda", enabled=use_amp):
            _, extras = model(bag_dev, return_extras=True, coords=he_coords)
        hazard = extras.get("hazard")
        if hazard is None:
            del bag_dev; continue

        cox_buffer.append((hazard.float(), surv_t, surv_e))
        accum_step += 1
        del bag_dev

        if accum_step == grad_accum:
            _step(); accum_step = 0

    if accum_step > 0:
        _step()

    return total_loss / max(n_steps, 1)


# ── Phase 1 evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def p1_evaluate(model, records, mod_name, device, bag_cache,
                use_spatial=False, cw=None):
    """Returns (probs, labels) or (probs, labels, val_loss) if cw is given."""
    model.eval()
    probs, labels, losses = [], [], []
    use_amp = (device.type == "cuda")
    use_spatial_for_mod = use_spatial and mod_name == "HE"
    for rec in records:
        label = rec.get("label")
        if label is None:
            continue
        bag = bag_cache.get(rec["stem"], {}).get(mod_name)
        if bag is None:
            continue
        bag_dev   = bag.to(device, non_blocking=True)
        he_coords = bag_cache.get(rec["stem"], {}).get("HE_coords") if use_spatial_for_mod else None
        with torch.amp.autocast("cuda", enabled=use_amp):
            logit = model(bag_dev, coords=he_coords)
        probs.append(torch.sigmoid(logit.float()).item())
        labels.append(label)
        if cw is not None:
            ta = logit.new_tensor([label])
            losses.append(hinge_loss(logit.unsqueeze(0), ta, cw).item())
        del bag_dev
    if cw is not None:
        return (np.array(probs), np.array(labels),
                float(np.mean(losses)) if losses else 0.0)
    return np.array(probs), np.array(labels)


@torch.no_grad()
def p1_evaluate_survival(model, records, mod_name, device, bag_cache,
                          surv_endpoint="clad", use_spatial=False):
    """Returns (c_index, mean_cox_loss)."""
    model.eval()
    hazards, times, events, cox_buf = [], [], [], []
    use_amp = (device.type == "cuda")
    use_spatial_for_mod = use_spatial and mod_name == "HE"
    for rec in records:
        # ACR uses non-standard column names
        _t_key = "tte_next_acr"  if surv_endpoint == "acr" else f"{surv_endpoint}_time"
        _e_key = "event_next_acr" if surv_endpoint == "acr" else f"{surv_endpoint}_event"
        surv_t = rec.get(_t_key, float("nan"))
        surv_e = rec.get(_e_key, float("nan"))
        if not isinstance(surv_t, float) or math.isnan(surv_t):
            continue
        bag = bag_cache.get(rec["stem"], {}).get(mod_name)
        if bag is None:
            continue
        bag_dev   = bag.to(device, non_blocking=True)
        he_coords = bag_cache.get(rec["stem"], {}).get("HE_coords") if use_spatial_for_mod else None
        with torch.amp.autocast("cuda", enabled=use_amp):
            with torch.enable_grad():
                _, extras = model(bag_dev, return_extras=True, coords=he_coords)
        hazard = extras.get("hazard")
        if hazard is None:
            del bag_dev; continue
        hazards.append(hazard.float().item())
        times.append(surv_t); events.append(surv_e)
        cox_buf.append((hazard.detach().float(), surv_t, surv_e))
        del bag_dev
    ci     = c_index(hazards, times, events) if len(hazards) >= 2 and sum(events) > 0 else 0.5
    val_cox = cox_breslow_loss(cox_buf)
    return ci, float(val_cox.item()) if val_cox is not None else 0.0


# ── Per-fold HP sweep ─────────────────────────────────────────────────────────

def run_phase1_hp_sweep(
    mod_name: str,
    device: torch.device,
    bag_cache: BagCache,
    train_recs: List[dict],
    val_recs: List[dict],
    save_dir: Path,
    task: str = "acr",
    surv_endpoint: str = "clad",
    use_spatial: bool = False,
    lr_grid: List[float] = HP_LR_GRID,
    wd_grid: List[float] = HP_WD_GRID,
    sweep_epochs: int = HP_SWEEP_EPOCHS,
    eval_every: int = 25,
) -> Tuple[float, float]:
    """
    Grid search over (lr, weight_decay) on this fold's val set.

    Returns (best_lr, best_wd). The caller should retrain on train+val
    with these hyperparameters before evaluating on test.
    """
    from sklearn.metrics import balanced_accuracy_score

    save_dir.mkdir(parents=True, exist_ok=True)
    result_path = save_dir / "hp_sweep.json"
    if result_path.exists():
        with open(result_path) as f:
            res = json.load(f)
        print(f"  [{mod_name}] HP sweep already done: "
              f"lr={res['best_lr']}  wd={res['best_wd']}  val_bacc={res['best_val_bacc']:.4f}")
        return res["best_lr"], res["best_wd"]

    pc = _pres_col(mod_name)
    tr = [r for r in train_recs if r.get(pc) and (task == "survival" or r.get("label") is not None)]
    vl = [r for r in val_recs   if r.get(pc) and (task == "survival" or r.get("label") is not None)]

    cw = compute_class_weights(tr) if task == "acr" else (1.0, 1.0)
    feat_dim = _feat_dim(mod_name)

    best_metric = -1.0
    best_lr, best_wd = lr_grid[0], wd_grid[0]
    results = []

    for lr, wd in iproduct(lr_grid, wd_grid):
        print(f"  [{mod_name}] HP sweep lr={lr:.0e}  wd={wd:.0e}", end="  ", flush=True)
        model = SingleModalMIL(feat_dim, HIDDEN_DIM, DROPOUT,
                               use_spatial=(use_spatial and mod_name == "HE")).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
        best_ep_metric = -1.0

        for ep in range(sweep_epochs):
            if task == "acr":
                p1_train_one_epoch(model, tr, mod_name, opt, cw, device, bag_cache,
                                   scaler, P1_GRAD_ACCUM, use_spatial)
            else:
                p1_train_one_epoch_survival(model, tr, mod_name, opt, device, bag_cache,
                                            scaler, P1_GRAD_ACCUM, surv_endpoint, use_spatial)
            if (ep + 1) % eval_every == 0:
                if task == "acr":
                    vp, vl_ = p1_evaluate(model, vl, mod_name, device, bag_cache, use_spatial)
                    m = compute_metrics(vl_, vp)
                    metric = m["bacc"]
                else:
                    metric, _ = p1_evaluate_survival(model, vl, mod_name, device, bag_cache,
                                                     surv_endpoint, use_spatial)
                best_ep_metric = max(best_ep_metric, metric)
                _gc()

        print(f"val_metric={best_ep_metric:.4f}")
        results.append({"lr": lr, "wd": wd, "val_metric": best_ep_metric})
        if best_ep_metric > best_metric:
            best_metric = best_ep_metric
            best_lr, best_wd = lr, wd

        del model, opt, scaler
        _gc()

    res = {"best_lr": best_lr, "best_wd": best_wd, "best_val_bacc": best_metric,
           "grid": results}
    with open(result_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  [{mod_name}] HP sweep done: lr={best_lr}  wd={best_wd}  "
          f"val_bacc={best_metric:.4f}")
    return best_lr, best_wd


# ── Phase 1 modality runner ───────────────────────────────────────────────────

def run_phase1_modality(
    mod_name: str,
    fold: int,
    device: torch.device,
    bag_cache: BagCache,
    train_recs: List[dict],
    val_recs: List[dict],
    test_recs: List[dict],
    save_dir: Path,
    use_spatial: bool = False,
    n_epochs: int = P1_EPOCHS,
    patience: int = 0,
    task: str = "acr",
    surv_endpoint: str = "clad",
    lr: float = P1_LR,
    weight_decay: float = P1_WEIGHT_DECAY,
    wandb_project: str = "chicago-mil",
    split: int = -1,
) -> Path:
    """
    Train Phase 1 for one modality.

    v8: task loss only (hinge or Cox). No CLR/KD/CRD.
    Saves best_model.pt (by val BACC or C-index) and metrics.json.
    """
    from sklearn.metrics import average_precision_score
    from mil.training.metrics import _plot_training_curves

    print(f"\n  {'─'*60}")
    print(f"  Phase 1 v8 — {mod_name}  fold={fold}  task={task}  "
          f"lr={lr:.0e}  wd={weight_decay:.0e}")

    pc = _pres_col(mod_name)
    tr = [r for r in train_recs if r.get(pc) and (task == "survival" or r.get("label") is not None)]
    vl = [r for r in val_recs   if r.get(pc) and (task == "survival" or r.get("label") is not None)]
    te = [r for r in test_recs  if r.get(pc) and (task == "survival" or r.get("label") is not None)]
    print(f"  Present: train={len(tr)}  val={len(vl)}  test={len(te)}")

    save_dir.mkdir(parents=True, exist_ok=True)

    if len(tr) == 0:
        dummy = SingleModalMIL(_feat_dim(mod_name), HIDDEN_DIM, DROPOUT,
                               use_spatial=(use_spatial and mod_name == "HE"))
        torch.save(dummy.state_dict(), save_dir / "best_model.pt")
        _write_status(save_dir / "status.json", completed=True,
                      best_epoch=0, best_metric=0.0, note="dummy_no_data")
        return save_dir / "best_model.pt"

    if _is_completed(save_dir):
        st = _read_status(save_dir / "status.json")
        print(f"  [{mod_name}] Already complete "
              f"(ep={st.get('best_epoch')}  metric={st.get('best_metric',0):.4f}). Skipping.")
        return save_dir / "best_model.pt"

    _wb = None
    if wandb_project and _WANDB_AVAILABLE:
        try:
            _wb = _wandb.init(
                project=wandb_project,
                name=f"p1_s{split}f{fold}_{mod_name}_{task}",
                group=f"phase1_split{split}",
                config={
                    "phase": 1, "modality": mod_name, "fold": fold, "split": split,
                    "task": task, "surv_endpoint": surv_endpoint,
                    "lr": lr, "weight_decay": weight_decay, "n_epochs": n_epochs,
                },
                reinit=True,
            )
        except Exception as _we:
            print(f"  [wandb] init failed: {_we} — continuing without wandb")
            _wb = None

    use_spatial_for_mod = use_spatial and mod_name == "HE"
    cw = compute_class_weights(tr) if task == "acr" else (1.0, 1.0)
    feat_dim = _feat_dim(mod_name)

    model  = SingleModalMIL(feat_dim, HIDDEN_DIM, DROPOUT,
                             use_spatial=use_spatial_for_mod).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    ckpt_dir = save_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params={n_params:,}  task={task}  AMP={scaler is not None}  "
          f"patience={patience or 'off'}  eval_every={P1_EVAL_EVERY}")

    # Resume from checkpoint if available
    resume_ep = _find_resume_epoch(ckpt_dir)
    hist: Dict[str, list] = {k: [] for k in
                              ["train_loss", "val_loss", "val_metric"]}
    if resume_ep > 0:
        ckpt = _load_checkpoint(ckpt_dir, resume_ep)
        if ckpt is not None:
            model.load_state_dict(ckpt["model"], strict=False)
            opt.load_state_dict(ckpt["optimizer"])
            if scaler and ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            for k in hist:
                if k in ckpt.get("history", {}):
                    hist[k] = ckpt["history"][k]
            print(f"  [{mod_name}] Resumed from ep {resume_ep}")

    best_metric  = max(hist["val_metric"]) if hist["val_metric"] else -1.0
    best_ep      = 0
    no_improve   = 0
    start_ep     = resume_ep

    for ep in range(start_ep, n_epochs):
        if task == "acr":
            train_loss = p1_train_one_epoch(
                model, tr, mod_name, opt, cw, device, bag_cache,
                scaler, P1_GRAD_ACCUM, use_spatial)
        else:
            train_loss = p1_train_one_epoch_survival(
                model, tr, mod_name, opt, device, bag_cache,
                scaler, P1_GRAD_ACCUM, surv_endpoint, use_spatial)
        hist["train_loss"].append(train_loss)
        _gc()

        if (ep + 1) % P1_EVAL_EVERY == 0:
            if task == "acr":
                vp, vl_, val_loss = p1_evaluate(model, vl, mod_name, device, bag_cache,
                                                 use_spatial, cw)
                vm = compute_metrics(vl_, vp)
                metric = vm["bacc"]
                tag_str = (f"Lt={train_loss:.4f}/{val_loss:.4f}  "
                           f"auc={vm['auc']:.3f}  bacc={metric:.3f}")
            else:
                metric, val_loss = p1_evaluate_survival(
                    model, vl, mod_name, device, bag_cache,
                    surv_endpoint, use_spatial)
                tag_str = f"cox={train_loss:.4f}/{val_loss:.4f}  cidx={metric:.3f}"

            hist["val_loss"].append(val_loss)
            hist["val_metric"].append(metric)

            torch.save({
                "epoch": ep + 1, "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "history": hist,
            }, ckpt_dir / f"ep{ep+1:04d}.pt")

            improved = metric > best_metric
            if improved:
                best_metric = metric; best_ep = ep + 1; no_improve = 0
                torch.save(model.state_dict(), save_dir / "best_model.pt")
                ckpt_tag = "[ckpt*]"
            else:
                no_improve += 1; ckpt_tag = "[ckpt]"

            # Test eval — logging only, not used for early stopping
            if task == "acr":
                te_p, te_l_, _ = p1_evaluate(model, test_recs, mod_name, device,
                                              bag_cache, use_spatial, cw)
                te_vm  = compute_metrics(te_l_, te_p)
                te_str = f"  [test] bacc={te_vm['bacc']:.3f}  auc={te_vm['auc']:.3f}"
            else:
                te_ci, _ = p1_evaluate_survival(model, test_recs, mod_name, device,
                                                bag_cache, surv_endpoint, use_spatial)
                te_str = f"  [test] ci={te_ci:.3f}"

            print(f"  [{mod_name}] ep {ep+1:3d}  {tag_str}  {ckpt_tag}"
                  + (f"  no_improve={no_improve}/{patience}" if patience > 0 else "")
                  + te_str)

            if _wb is not None:
                _log: dict = {"epoch": ep + 1, "train/loss": train_loss}
                if task == "acr":
                    _log.update({"val/bacc": metric, "val/auc": vm["auc"],
                                 "test/bacc": te_vm["bacc"], "test/auc": te_vm["auc"]})
                else:
                    _log.update({"val/ci": metric, "test/ci": te_ci})
                try:
                    _wb.log(_log)
                except Exception:
                    pass

            _gc()

            if patience > 0 and no_improve >= patience:
                print(f"  [{mod_name}] Early stop "
                      f"(best_ep={best_ep}  best={best_metric:.4f})")
                break

        elif (ep + 1) % 10 == 0 or ep == 0:
            print(f"  [{mod_name}] ep {ep+1:3d}  train_loss={train_loss:.4f}")

    # ── Load best model and write final metrics ───────────────────────────────
    if (save_dir / "best_model.pt").exists() and best_ep > 0:
        state = torch.load(save_dir / "best_model.pt",
                           map_location="cpu", weights_only=False)
        state = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state, strict=False); model.to(device); del state
    else:
        # Fall back to checkpoint scan
        ckpts = sorted(ckpt_dir.glob("ep*.pt"))
        if ckpts:
            best_ck_metric, best_ck = -1.0, ckpts[-1]
            for cp in ckpts:
                try:
                    d = torch.load(cp, map_location="cpu", weights_only=False)
                    vm_list = d.get("history", {}).get("val_metric", [])
                    m = max(vm_list) if vm_list else -1.0
                    if m > best_ck_metric:
                        best_ck_metric, best_ck = m, cp
                    del d
                except Exception:
                    pass
            d = torch.load(best_ck, map_location="cpu", weights_only=False)
            state = d["model"] if isinstance(d, dict) else d
            model.load_state_dict(state, strict=False); model.to(device)
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            best_metric = best_ck_metric
            del d, state

    _write_status(save_dir / "status.json", completed=True,
                  best_epoch=best_ep, best_metric=round(best_metric, 4),
                  last_epoch=n_epochs)

    # Final split metrics
    metrics: dict = {}
    for sn, recs in [("train", tr), ("val", vl), ("test", te)]:
        if task == "acr":
            p, l = p1_evaluate(model, recs, mod_name, device, bag_cache, use_spatial)
            m = compute_metrics(l, p)
            m["auprc"] = (average_precision_score(l, p)
                          if len(np.unique(l)) > 1 else 0.0)
            metrics[sn] = {**m, "probs": p.tolist(), "labels": l.tolist()}
            print(f"  [{mod_name}] {sn:5s}  AUC={m['auc']:.4f}  "
                  f"BAcc={m['bacc']:.4f}  MCC={m.get('mcc', 0):.4f}")
        else:
            ci, _ = p1_evaluate_survival(model, recs, mod_name, device, bag_cache,
                                          surv_endpoint, use_spatial)
            metrics[sn] = {"c_index": ci}
            print(f"  [{mod_name}] {sn:5s}  C-index={ci:.4f}")

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(save_dir / "history.json", "w") as f:
        json.dump(hist, f)
    _plot_training_curves(
        {"train_loss": hist["train_loss"],
         "val_loss":   hist["val_loss"],
         "val_bacc":   hist["val_metric"]},
        save_dir / "plots", tag=mod_name)
    if _wb is not None:
        summary: dict = {}
        for sn in ("train", "val", "test"):
            m = metrics.get(sn, {})
            if task == "acr":
                summary[f"{sn}/bacc"] = m.get("bacc", float("nan"))
                summary[f"{sn}/auc"]  = m.get("auc",  float("nan"))
            else:
                summary[f"{sn}/ci"] = m.get("c_index", float("nan"))
        try:
            _wb.summary.update(summary)
            _wb.finish()
        except Exception:
            pass

    del model, opt, scaler
    _gc()
    return save_dir / "best_model.pt"
