"""
Phase 2 training loops and runner (v8 design).

Phase 2 fuses per-modality tokens from Phase 1 into a joint multimodal prediction.
Phase 1 encoders can be (a) randomly initialised, (b) loaded frozen, or (c) fine-tuned.

v8 improvements
---------------
- Multitask training loop: hinge (ACR cls) + Cox(ACR surv) + Cox(CLAD) + Cox(death)
  simultaneously from MultiTaskHead dict output.
- More epochs: 500 for slot variants (iterative/crossmodal/self_attn), 300 for simple.
- Cosine LR with linear warmup (10% of total epochs).
- Per-fold HP sweep (LR × weight_decay) on val set.
- Fair evaluation: all test records included regardless of missing modalities;
  missing-modality records receive majority-class prediction for BACC computation.
- Single-modality baseline via `run_single_modal_eval` (same test records as multimodal).

Fusion variants (see ``mil.models.builders.build_model_v8``):
  slot   — MultimodalSlotMIL (slot_k slots/mod, n_cross_layers, 3 slot iters)
  early  — all modality patches concatenated, single ABMIL pool  [ablation]
  late   — per-modality ABMIL pools, combined  [ablation]
  middle — cross-modal transformer over modality summaries  [ablation]

Functions exported
------------------
p2_train_one_epoch
p2_train_one_epoch_multitask
p2_train_one_epoch_alternating
p2_evaluate
p2_evaluate_fair
run_phase2_hp_sweep
run_phase2_variant
run_single_modal_eval
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from mil.data.registry import MODALITIES
from mil.training.losses import (
    hinge_loss, compute_class_weights,
    cox_breslow_loss, c_index,
    batch_supcon_loss,
)
from mil.training.metrics import compute_metrics

# ── Type aliases ──────────────────────────────────────────────────────────────
BagCache = Dict[str, Dict[str, Optional[torch.Tensor]]]

# ── Phase 2 hyperparameter defaults ──────────────────────────────────────────
P2_LR             = 5e-5
P2_WEIGHT_DECAY   = 1e-3
P2_EPOCHS         = 600    # early/late/middle — enough for cosine to converge
P2_EPOCHS_SLOT    = 1000   # slot needs more iterations (K slots × cross-attn × 4 tasks)
P2_EVAL_EVERY     = 20
P2_GRAD_ACCUM     = 8      # larger effective batch improves stability
P2_MODAL_DROPOUT  = 0.3    # 0.3 → trains with each single modality ~24% of epochs
P2_WARMUP_FRAC    = 0.10   # 10% warmup → cosine decay over remaining 90%

# HP sweep search space
P2_HP_LR_GRID     = [1e-4, 5e-5, 1e-5]
P2_HP_WD_GRID     = [1e-3, 1e-4]
P2_HP_SWEEP_EPOCHS = 150   # longer sweep for more reliable HP selection

# Task Cox lambdas for multitask loss
P2_COX_LAMBDA_ACR   = 0.5   # ACR survival Cox weight
P2_COX_LAMBDA_CLAD  = 0.3   # CLAD Cox weight
P2_COX_LAMBDA_DEATH = 0.2   # Death Cox weight

# CLR defaults (kept for backward compat, not used in v8)
P1_CLR_TAU    = 0.07
P1_CLR_LAMBDA = 0.1

# Slot variants — need more epochs
_SLOT_VARIANTS = {"slot"}

_ALTERNATING_TASKS  = ["acr_cls", "acr_surv", "clad", "death"]
_SURV_SPEC_ALT      = {
    "acr_surv": ("tte_next_acr",  "event_next_acr"),
    "clad":     ("clad_time",      "clad_event"),
    "death":    ("death_time",     "death_event"),
}
_TASK_STRAT_FIELD   = {
    "acr_cls":  "label",
    "acr_surv": "event_next_acr",
    "clad":     "clad_event",
    "death":    "death_event",
}


# ── Internal helpers (mirrors phase1_trainer helpers) ─────────────────────────

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


def _is_completed(save_dir: Path, tag: str = "status") -> bool:
    s = _read_status(save_dir / f"{tag}.json")
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


def _load_checkpoint(ckpt_dir: Path, epoch: int):
    path = ckpt_dir / f"ep{epoch:04d}.pt"
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None


    # ── per-task survival keys and default Cox weights ────────────────────────
_SURV_KEYS = {
    "acr_surv": ("tte_next_acr",  "event_next_acr"),
    "clad":     ("clad_time",      "clad_event"),
    "death":    ("death_time",     "death_event"),
}
_DEFAULT_COX_LAMBDAS = {
    "acr_surv": P2_COX_LAMBDA_ACR,
    "clad":     P2_COX_LAMBDA_CLAD,
    "death":    P2_COX_LAMBDA_DEATH,
}
# Alternating task sampling distribution (un-normalised; gets normalised at use)
DEFAULT_TASK_WEIGHTS = {
    "acr_cls":  0.20,
    "acr_surv": 0.20,
    "clad":     0.25,
    "death":    0.25,
}
# With GeoMAE reconstruction regularisation — recon keeps backbone from forgetting
GEOMAE_TASK_WEIGHTS = {
    "recon":    0.15,
    "acr_cls":  0.18,
    "acr_surv": 0.18,
    "clad":     0.22,
    "death":    0.22,
}
GEOMAE_RECON_MASK_RATIO = 0.50   # same as pretraining


def _parse_model_output(result) -> dict:
    """Normalise any model output to a task-keyed dict of (tensor, rep) tuples."""
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple) and len(result) >= 2:
        return {"acr_cls": (result[0], None), "acr_surv": (result[1], None)}
    return {"acr_cls": (result, None)}


def _stratified_record_order(records: List[dict], task_name: str) -> List[dict]:
    """Interleave positive/negative records so each grad-accum window is class-balanced."""
    field = _TASK_STRAT_FIELD.get(task_name)
    if field is None:
        out = list(records); random.shuffle(out); return out
    pos    = [r for r in records if r.get(field) == 1]
    neg    = [r for r in records if r.get(field) == 0]
    others = [r for r in records if r.get(field) not in (0, 1)]
    random.shuffle(pos); random.shuffle(neg); random.shuffle(others)
    result: List[dict] = []
    i = j = 0
    while i < len(pos) or j < len(neg):
        if i < len(pos): result.append(pos[i]); i += 1
        if j < len(neg): result.append(neg[j]); j += 1
    result.extend(others)
    return result


# ── Full-batch Cox epoch (survival-only tasks) ────────────────────────────────

def p2_cox_full_epoch(
    model, records, task_name: str, optimizer, device, bag_cache, scaler,
    cox_lambda: float = 1.0,
) -> float:
    """
    Full-batch Cox for a single survival endpoint.

    WHY: mini-batch Cox (every grad_accum steps) fails when the event rate is
    low (~10-15% for CLAD/Death). Most windows are all-censored → cox_breslow
    returns None → zero gradient for the majority of records.

    FIX: one forward pass over ALL survival records, collect every hazard
    tensor (keeping computation graphs alive), compute Cox on the full risk set
    once, single backward. N≈300 patients × small hazard tensor is trivial RAM.
    """
    model.train()
    t_key, e_key = _SURV_KEYS[task_name]
    use_amp = scaler is not None
    bags_buf = {m: None for m in MODALITIES}
    bags_buf["HE_coords"] = None
    OOM_SKIP = 3; oom_counts: dict = {}

    # Shuffle so ordering doesn't bias the risk set tie-breaking
    ordered = list(records); random.shuffle(ordered)

    cox_buf: list = []   # (hazard_tensor_with_grad, t, e)

    for rec in ordered:
        if oom_counts.get(rec["stem"], 0) >= OOM_SKIP:
            continue
        t_v = rec.get(t_key, float("nan"))
        e_v = rec.get(e_key, float("nan"))
        if not (isinstance(t_v, float) and not math.isnan(t_v) and t_v >= 0):
            continue   # no survival data for this record, skip

        entry = bag_cache.get(rec["stem"], {})
        for m in MODALITIES:
            bags_buf[m] = entry.get(m)
        bags_buf["HE_coords"] = entry.get("HE_coords")
        if all(bags_buf.get(m) is None for m in MODALITIES):
            continue

        try:
            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model(bags_buf, device)
            task_out = _parse_model_output(result)
            haz = task_out.get(task_name, (None,))[0]
            if not isinstance(haz, torch.Tensor):
                continue
            e_s = float(e_v) if (e_v is not None and
                                  not math.isnan(float(e_v))) else 0.0
            cox_buf.append((haz.float(), float(t_v), e_s))

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # If OOM mid-accumulation, free graphs and restart without this bag
            cox_buf.clear()
            oom_counts[rec["stem"]] = oom_counts.get(rec["stem"], 0) + 1
            if oom_counts[rec["stem"]] >= OOM_SKIP:
                print(f"  [OOM-cox] {rec['stem']} permanently skipped", flush=True)
            continue

    if not cox_buf:
        return 0.0

    n_events = sum(1 for _, _, e in cox_buf if e > 0)
    if n_events == 0:
        # No events in training set for this endpoint this fold — no signal
        for h, _, _ in cox_buf:
            del h
        return 0.0

    optimizer.zero_grad()
    L_cox = cox_breslow_loss(cox_buf)
    if L_cox is None or not L_cox.requires_grad:
        return 0.0

    loss_val = L_cox.item()
    if scaler:
        scaler.scale(L_cox * cox_lambda).backward()
        scaler.step(optimizer); scaler.update()
    else:
        (L_cox * cox_lambda).backward()
        optimizer.step()
    optimizer.zero_grad()
    _gc()
    return loss_val


# ── Phase 2 training epoch ────────────────────────────────────────────────────

def p2_train_one_epoch(model, records, optimizer, cw, device, bag_cache,
                       scaler, grad_accum,
                       use_contrastive=False,
                       clr_tau=P1_CLR_TAU, clr_lambda=P1_CLR_LAMBDA,
                       recon_lambda=0.0,
                       cox_lambda=0.0, surv_endpoint='clad',
                       task='acr'):
    model.train()
    random.shuffle(records)
    total_loss = 0.0; n = 0
    accum_step = 0; grad_accumulated = False
    batch_buffer: List[Tuple[torch.Tensor, int, str, str]] = []
    cox_buffer: list = []
    pending_loss_ref: List[Optional[torch.Tensor]] = [None]
    optimizer.zero_grad()

    has_enc  = hasattr(model, "encoders")
    has_proj = hasattr(model, "proj_heads")

    # Reuse a single bags dict every iteration — avoids creating a new dict
    # object per sample (important when Python GC is unreliable).
    bags: dict = {m: None for m in MODALITIES}
    bags["HE_coords"] = None

    # Per-bag OOM counter — bags that OOM this many times in one epoch are
    # permanently skipped for the rest of the epoch to prevent infinite loops.
    OOM_SKIP_THRESHOLD = 3
    oom_per_bag: dict = {}

    for rec in records:
        # Skip bags that repeatedly OOM this epoch
        if oom_per_bag.get(rec["stem"], 0) >= OOM_SKIP_THRESHOLD:
            continue
        # Refill in-place from the pre-loaded CPU cache
        entry = bag_cache.get(rec["stem"], {})
        for m in MODALITIES:
            bags[m] = entry.get(m)
        bags["HE_coords"] = entry.get("HE_coords")
        if all(bags.get(m) is None for m in MODALITIES): continue

        target = torch.tensor([rec["label"]], dtype=torch.float32, device=device)
        _surv_key_map_hp = {
            "acr":   ("tte_next_acr",  "event_next_acr"),
            "clad":  ("clad_time",      "clad_event"),
            "death": ("death_time",     "death_event"),
        }
        surv_time_key, surv_event_key = _surv_key_map_hp.get(
            surv_endpoint, (f"{surv_endpoint}_time", f"{surv_endpoint}_event"))
        surv_t = rec.get(surv_time_key, float("nan"))
        has_surv_data = (isinstance(surv_t, float) and not math.isnan(surv_t))

        try:
            # Contrastive reps — before main forward so graph is alive through encoders
            if use_contrastive and has_enc and has_proj:
                for mod, enc in model.encoders.items():
                    bag = bags.get(mod)
                    if bag is None: continue
                    bag_dev = bag.to(device, non_blocking=True)
                    crds = bags.get("HE_coords") if mod == "HE" else None
                    rep, _, _ = enc(bag_dev, coords=crds)
                    pz = model.proj_heads[mod](rep)
                    batch_buffer.append((pz, rec["label"], rec["stem"], mod))

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                result = model(bags, device)
                if isinstance(result, dict):
                    logit  = result.get("acr_cls", (None,))[0]
                    hazard = result.get("acr_surv", (None,))[0]
                    L_recon = None
                elif isinstance(result, tuple) and len(result) >= 3:
                    logit, hazard = result[0], result[1]; L_recon = None
                elif isinstance(result, tuple) and len(result) == 2:
                    logit, L_recon = result; hazard = None
                else:
                    logit = result; L_recon = None; hazard = None
                if not isinstance(logit, torch.Tensor) or logit.grad_fn is None:
                    continue

            # ── Accumulate losses (no backward per-record) ───────────────
            if task == 'acr':
                with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                    loss = hinge_loss(logit.unsqueeze(0), target, cw) / grad_accum
                    if L_recon is not None and recon_lambda > 0:
                        loss = loss + recon_lambda * L_recon / grad_accum
                pending_loss_ref[0] = (loss if pending_loss_ref[0] is None
                                       else pending_loss_ref[0] + loss)
                if hazard is not None and cox_lambda > 0 and has_surv_data:
                    cox_buffer.append((hazard.float(), rec[surv_time_key], rec[surv_event_key]))
                total_loss += loss.item() * grad_accum
                grad_accumulated = True
                n += 1; accum_step += 1

            else:  # survival
                if hazard is not None and has_surv_data:
                    cox_buffer.append((hazard.float(), rec[surv_time_key], rec[surv_event_key]))
                grad_accumulated = True
                n += 1; accum_step += 1

            if accum_step == grad_accum:
                combined = pending_loss_ref[0]
                eff_cox_lambda = cox_lambda if task == 'acr' else 1.0
                if cox_buffer and eff_cox_lambda > 0:
                    L_cox = cox_breslow_loss(cox_buffer)
                    if L_cox is not None and L_cox.requires_grad:
                        term = L_cox * eff_cox_lambda
                        combined = term if combined is None else combined + term
                        if task == 'survival':
                            total_loss += L_cox.item()
                cox_buffer.clear()
                if combined is not None and combined.requires_grad:
                    if scaler is not None:
                        scaler.scale(combined).backward()
                        scaler.step(optimizer); scaler.update()
                    else:
                        combined.backward(); optimizer.step()
                optimizer.zero_grad()
                batch_buffer.clear()
                pending_loss_ref[0] = None
                accum_step = 0; grad_accumulated = False
                _gc()

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            batch_buffer.clear()
            cox_buffer.clear()
            pending_loss_ref[0] = None
            accum_step = 0
            grad_accumulated = False
            oom_per_bag[rec["stem"]] = oom_per_bag.get(rec["stem"], 0) + 1
            count = oom_per_bag[rec["stem"]]
            if count >= OOM_SKIP_THRESHOLD:
                print(f"  [OOM-p2] {rec['stem']} hit OOM {count}× — permanently skipping for this epoch", flush=True)
            else:
                print(f"  [OOM-p2] skipped {rec['stem']} ({count}/{OOM_SKIP_THRESHOLD}) — cache cleared", flush=True)

    if accum_step > 0 and grad_accumulated:
        combined = pending_loss_ref[0]
        eff_cox_lambda = cox_lambda if task == 'acr' else 1.0
        if cox_buffer and eff_cox_lambda > 0:
            L_cox = cox_breslow_loss(cox_buffer)
            if L_cox is not None and L_cox.requires_grad:
                term = L_cox * eff_cox_lambda
                combined = term if combined is None else combined + term
                if task == 'survival':
                    total_loss += L_cox.item()
        cox_buffer.clear()
        if combined is not None and combined.requires_grad:
            if scaler is not None:
                scaler.scale(combined).backward()
                scaler.step(optimizer); scaler.update()
            else:
                combined.backward(); optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(n, 1)


# ── Phase 2 evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def p2_evaluate(model, records, device, bag_cache, cw=None,
                surv_endpoint='clad', task='acr'):
    """Single-pass eval.
    ACR mode:      returns (probs, labels, val_loss, ci, [], [], [])
    Survival mode: returns (probs, labels, val_loss, ci, hazards, times, events)
      where hazards/times/events are parallel lists for samples with valid survival data.
    """
    model.eval(); probs, labels, losses = [], [], []
    hazard_list, surv_times, surv_events = [], [], []
    use_amp = (device.type == "cuda")
    for rec in records:
        bags = {m: bag_cache.get(rec["stem"], {}).get(m) for m in MODALITIES}
        bags["HE_coords"] = bag_cache.get(rec["stem"], {}).get("HE_coords")
        if all(bags.get(m) is None for m in MODALITIES): continue
        try:
            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model(bags, device)
            # Handle dict output from MultiTaskHead
            if isinstance(result, dict):
                logit  = result.get("acr_cls", (None,))[0]
                # map surv_endpoint → dict key
                _haz_key = {"acr": "acr_surv", "clad": "clad",
                            "death": "death"}.get(surv_endpoint, surv_endpoint)
                hazard = result.get(_haz_key, (None,))[0]
                if logit is None and hazard is None: continue
                if logit is None:
                    # survival-only model — use hazard as proxy logit for probs
                    logit = hazard
            elif isinstance(result, tuple) and len(result) >= 3:
                logit, hazard = result[0], result[1]   # DualGatedPool: (logit, hazard, ...)
            elif isinstance(result, tuple):
                logit = result[0]; hazard = None
            else:
                logit = result; hazard = None
            # ── Classification (BACC): only records with ACR label ──────
            label = rec.get("label")
            if label is not None and isinstance(logit, torch.Tensor):
                probs.append(torch.sigmoid(logit.float()).item())
                labels.append(label)
                if cw is not None:
                    ta = logit.new_tensor([label])
                    losses.append(hinge_loss(logit.unsqueeze(0), ta, cw).item())

            # ── Survival (C-index): ALL records with valid TTE ────────
            # Do NOT require ACR label — CLAD/Death records may have none.
            # ACR endpoint uses non-standard column names.
            _surv_key_map = {
                "acr":   ("tte_next_acr",  "event_next_acr"),
                "clad":  ("clad_time",      "clad_event"),
                "death": ("death_time",     "death_event"),
            }
            _t_key, _e_key = _surv_key_map.get(
                surv_endpoint,
                (f"{surv_endpoint}_time", f"{surv_endpoint}_event"))
            if hazard is not None and isinstance(hazard, torch.Tensor):
                t_val = rec.get(_t_key, float("nan"))
                e_val = rec.get(_e_key, float("nan"))
                if (isinstance(t_val, float) and not math.isnan(t_val) and t_val >= 0):
                    e_safe = float(e_val) if (e_val is not None and
                                              not math.isnan(float(e_val))) else 0.0
                    hazard_list.append(hazard.float().item())
                    surv_times.append(float(t_val))
                    surv_events.append(e_safe)

            if isinstance(logit, torch.Tensor):
                del logit
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [OOM-eval] skipped {rec['stem']} — cache cleared", flush=True)
    ci = None
    if len(hazard_list) >= 2 and sum(surv_events) > 0:
        ci = c_index(hazard_list, surv_times, surv_events)

    # Survival-only tasks: primary metric = C-index, loss = Cox
    _surv_tasks = {"clad_surv", "death_surv", "surv", "acr_surv"}
    if task in _surv_tasks:
        cox_buf = [(torch.tensor(h), t, e)
                   for h, t, e in zip(hazard_list, surv_times, surv_events)]
        cox_l   = cox_breslow_loss(cox_buf)
        val_loss = float(cox_l.item()) if cox_l is not None else 0.0
        return np.array(probs), np.array(labels), val_loss, ci, hazard_list, surv_times, surv_events

    # Classification tasks: primary metric = BACC
    val_loss = float(np.mean(losses)) if losses else 0.0
    return np.array(probs), np.array(labels), val_loss, ci, hazard_list, surv_times, surv_events


# ── Multitask training epoch ──────────────────────────────────────────────────

def p2_train_one_epoch_multitask(
    model, records, optimizer, cw, device, bag_cache, scaler, grad_accum,
    cox_lambda_acr:   float = P2_COX_LAMBDA_ACR,
    cox_lambda_clad:  float = P2_COX_LAMBDA_CLAD,
    cox_lambda_death: float = P2_COX_LAMBDA_DEATH,
) -> dict:
    """
    Multitask epoch for MultiTaskHead models (dict output).

    Applies simultaneously:
      - Hinge loss for records with label (acr_cls)
      - Cox-Breslow for records with tte_next_acr (acr_surv)
      - Cox-Breslow for records with clad_time (clad)
      - Cox-Breslow for records with death_time (death)

    All losses accumulate across grad_accum samples before an optimizer step.
    Cox losses flush at each step boundary.
    """
    model.train()
    random.shuffle(records)
    use_amp = (scaler is not None)
    bags_buf = {m: None for m in MODALITIES}
    bags_buf["HE_coords"] = None
    OOM_SKIP = 3
    oom_counts: dict = {}

    total_hinge = 0.0; total_cox = 0.0; n_hinge = 0; n_cox_steps = 0
    accum_step = 0

    # Accumulate losses as LIVE tensors — do NOT backward per-record.
    # All graphs stay alive until _flush() calls a single backward.
    # (Per-record backward + retain_graph frees earlier records' graphs.)
    pending_loss: Optional[torch.Tensor] = None
    cox_bufs: Dict[str, list] = {"acr_surv": [], "clad": [], "death": []}
    optimizer.zero_grad()

    # Task → (tte_key, event_key, lambda)
    surv_spec = {
        "acr_surv": ("tte_next_acr",  "event_next_acr", cox_lambda_acr),
        "clad":     ("clad_time",      "clad_event",     cox_lambda_clad),
        "death":    ("death_time",     "death_event",    cox_lambda_death),
    }

    def _flush():
        nonlocal pending_loss, n_cox_steps
        combined = pending_loss
        for tk, buf in cox_bufs.items():
            if not buf: continue
            lam = surv_spec[tk][2]
            L_cox = cox_breslow_loss(buf)
            if L_cox is not None and L_cox.requires_grad:
                term = L_cox * lam
                combined = term if combined is None else combined + term
                total_cox_ref[0] += L_cox.item()
                n_cox_steps += 1
            buf.clear()
        if combined is not None and combined.requires_grad:
            if scaler:
                scaler.scale(combined).backward()
                scaler.step(optimizer); scaler.update()
            else:
                combined.backward()
                optimizer.step()
        optimizer.zero_grad()
        pending_loss = None
        _gc()

    total_cox_ref = [0.0]

    for rec in records:
        if oom_counts.get(rec["stem"], 0) >= OOM_SKIP:
            continue
        entry = bag_cache.get(rec["stem"], {})
        for m in MODALITIES:
            bags_buf[m] = entry.get(m)
        bags_buf["HE_coords"] = entry.get("HE_coords")
        if all(bags_buf.get(m) is None for m in MODALITIES):
            continue

        try:
            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model(bags_buf, device)

            if isinstance(result, dict):
                task_out = result
            elif isinstance(result, tuple) and len(result) >= 2:
                task_out = {"acr_cls": (result[0], None)}
                if result[1] is not None:
                    task_out["acr_surv"] = (result[1], None)
            else:
                task_out = {"acr_cls": (result, None)}

            # ── Accumulate hinge loss (no backward yet) ───────────────
            label = rec.get("label")
            if label is not None and "acr_cls" in task_out:
                logit_val = task_out["acr_cls"][0]
                if isinstance(logit_val, torch.Tensor) and logit_val.grad_fn is not None:
                    target = torch.tensor([float(label)], device=device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        L_h = hinge_loss(logit_val.unsqueeze(0), target, cw) / grad_accum
                    pending_loss = L_h if pending_loss is None else pending_loss + L_h
                    total_hinge += L_h.item() * grad_accum
                    n_hinge += 1

            # ── Accumulate hazard tensors (graph stays alive) ─────────
            for tk, (tk_key, ev_key, _) in surv_spec.items():
                if tk not in task_out: continue
                hazard_val = task_out[tk][0]
                if not isinstance(hazard_val, torch.Tensor): continue
                t_val = rec.get(tk_key, float("nan"))
                e_val = rec.get(ev_key, float("nan"))
                if isinstance(t_val, float) and not math.isnan(t_val) and t_val >= 0:
                    e_safe = (float(e_val) if (e_val is not None and
                              not math.isnan(float(e_val))) else 0.0)
                    cox_bufs[tk].append((hazard_val.float(), float(t_val), e_safe))

            accum_step += 1

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            for buf in cox_bufs.values(): buf.clear()
            pending_loss = None; accum_step = 0
            oom_counts[rec["stem"]] = oom_counts.get(rec["stem"], 0) + 1
            if oom_counts[rec["stem"]] >= OOM_SKIP:
                print(f"  [OOM-p2mt] {rec['stem']} permanently skipped", flush=True)
            continue

        if accum_step == grad_accum:
            _flush(); accum_step = 0

    if accum_step > 0:
        _flush()

    return {
        "hinge": total_hinge / max(n_hinge, 1),
        "cox":   total_cox_ref[0] / max(n_cox_steps, 1),
    }


def p2_train_one_epoch_alternating(
    model, records, task_name: str, optimizer, cw, device, bag_cache,
    scaler, grad_accum, cox_lambda: float = 1.0,
) -> float:
    """
    Train one epoch on a SINGLE task with stratified record ordering.

    task_name ∈ {'acr_cls', 'acr_surv', 'clad', 'death'}.
    Called by run_phase2_variant when alternating=True; the caller randomly
    picks task_name for each epoch.
    """
    model.train()
    ordered  = _stratified_record_order(records, task_name)
    use_amp  = scaler is not None
    bags_buf = {m: None for m in MODALITIES}
    bags_buf["HE_coords"] = None
    OOM_SKIP = 3; oom_counts: dict = {}

    is_cls  = (task_name == "acr_cls")
    is_surv = task_name in _SURV_SPEC_ALT

    total_loss = 0.0; n = 0; accum_step = 0
    pending_loss: Optional[torch.Tensor] = None
    cox_buffer: list = []
    optimizer.zero_grad()

    def _flush_step():
        nonlocal pending_loss
        combined = pending_loss
        if is_surv and cox_buffer:
            L_cox = cox_breslow_loss(cox_buffer)
            if L_cox is not None and L_cox.requires_grad:
                term = L_cox * cox_lambda
                combined = term if combined is None else combined + term
                total_loss_ref[0] += L_cox.item()
            cox_buffer.clear()
        if combined is not None and combined.requires_grad:
            if scaler:
                scaler.scale(combined).backward()
                scaler.step(optimizer); scaler.update()
            else:
                combined.backward(); optimizer.step()
        optimizer.zero_grad(); pending_loss = None; _gc()

    total_loss_ref = [0.0]

    for rec in ordered:
        if oom_counts.get(rec["stem"], 0) >= OOM_SKIP:
            continue
        entry = bag_cache.get(rec["stem"], {})
        for m in MODALITIES:
            bags_buf[m] = entry.get(m)
        bags_buf["HE_coords"] = entry.get("HE_coords")
        if all(bags_buf.get(m) is None for m in MODALITIES):
            continue

        try:
            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model(bags_buf, device)

            if isinstance(result, dict):
                task_out = result
            elif isinstance(result, tuple) and len(result) >= 2:
                task_out = {"acr_cls": (result[0], None),
                            "acr_surv": (result[1], None)}
            else:
                task_out = {"acr_cls": (result, None)}

            if is_cls:
                label = rec.get("label")
                if label is None or "acr_cls" not in task_out:
                    continue
                logit_v = task_out["acr_cls"][0]
                if not isinstance(logit_v, torch.Tensor) or logit_v.grad_fn is None:
                    continue
                target = torch.tensor([float(label)], device=device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    L = hinge_loss(logit_v.unsqueeze(0), target, cw) / grad_accum
                pending_loss = L if pending_loss is None else pending_loss + L
                total_loss += L.item() * grad_accum

            elif is_surv:
                t_key, e_key = _SURV_SPEC_ALT[task_name]
                if task_name not in task_out:
                    continue
                haz_v = task_out[task_name][0]
                if not isinstance(haz_v, torch.Tensor):
                    continue
                t_val = rec.get(t_key, float("nan"))
                e_val = rec.get(e_key, float("nan"))
                if isinstance(t_val, float) and not math.isnan(t_val) and t_val >= 0:
                    e_safe = float(e_val) if (e_val is not None and not math.isnan(float(e_val))) else 0.0
                    cox_buffer.append((haz_v.float(), float(t_val), e_safe))

            n += 1; accum_step += 1

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); optimizer.zero_grad()
            cox_buffer.clear(); pending_loss = None; accum_step = 0
            oom_counts[rec["stem"]] = oom_counts.get(rec["stem"], 0) + 1
            if oom_counts[rec["stem"]] >= OOM_SKIP:
                print(f"  [OOM-alt] {rec['stem']} permanently skipped", flush=True)
            continue

        if accum_step == grad_accum:
            _flush_step(); accum_step = 0

    if accum_step > 0:
        _flush_step()

    return (total_loss + total_loss_ref[0]) / max(n, 1)


# ── GeoMAE reconstruction regularisation epoch ───────────────────────────────

def p2_recon_epoch(
    model, records, optimizer, device, bag_cache,
    scaler, grad_accum,
    mask_ratio: float = GEOMAE_RECON_MASK_RATIO,
) -> dict:
    """
    One reconstruction epoch using GeoMAE-backbone encoders.

    For each record: if HE or CT bags exist AND the corresponding encoder is a
    GeoMAESpatialBackbone, apply BFS-flood masking and compute the denoising
    reconstruction loss.  Backprop keeps the spatial encoder from forgetting
    its pretraining objective while fine-tuning on MIL tasks.

    Records without spatial modalities (BAL-only, Clinical-only) are skipped.
    """
    from mil.models.encoders import GeoMAESpatialBackbone

    model.train()
    random.shuffle(records)

    total_loss = 0.0
    n_steps = 0
    accum_step = 0
    optimizer.zero_grad()

    SPATIAL_MODS = ["HE", "CT"]

    for rec in records:
        losses_per_rec = []

        for mod in SPATIAL_MODS:
            enc = model.encoders.get(mod) if hasattr(model, "encoders") else None
            if not isinstance(enc, GeoMAESpatialBackbone):
                continue                   # not a GeoMAE backbone — skip

            bag = bag_cache.get(rec["stem"], {}).get(mod)
            if bag is None:
                continue

            coords_key = "HE_coords" if mod == "HE" else "CT_coords"
            coords = bag_cache.get(rec["stem"], {}).get(coords_key)
            if coords is None:
                continue                   # no coords → can't do spatial masking

            bag    = bag.to(device, non_blocking=True)
            coords = coords.to(device, non_blocking=True)

            try:
                use_amp = (device.type == "cuda")
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = enc.forward_recon(bag, coords, mask_ratio=mask_ratio)

                if loss.requires_grad and torch.isfinite(loss):
                    losses_per_rec.append(loss / grad_accum)
            except torch.cuda.OutOfMemoryError:
                _gc(); optimizer.zero_grad(); accum_step = 0
                break
            except Exception:
                continue

        if not losses_per_rec:
            continue

        combined = sum(losses_per_rec) / len(losses_per_rec)
        if scaler:
            scaler.scale(combined).backward()
        else:
            combined.backward()

        total_loss  += combined.item() * grad_accum
        n_steps     += 1
        accum_step  += 1

        if accum_step >= grad_accum:
            if scaler:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
            accum_step = 0
            _gc()

    # Final flush
    if accum_step > 0:
        if scaler:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()

    n = max(n_steps, 1)
    return {"loss": total_loss / n, "n_records": n_steps, "task": "recon"}


# ── Unified Phase 2 training epoch ───────────────────────────────────────────

def p2_train_epoch(
    model, records, optimizer, cw, device, bag_cache,
    scaler, grad_accum,
    mode: str = "simultaneous",
    task_weights: Optional[Dict[str, float]] = None,
    cox_lambdas:  Optional[Dict[str, float]] = None,
) -> dict:
    """
    Single unified training epoch for all Phase 2 modes.

    mode
    ----
    'simultaneous'  : all available task losses added each step
    'alternating'   : sample one task per epoch from task_weights distribution
    'acr_cls'       : hinge loss only
    'acr_surv'      : Cox ACR only
    'clad'          : Cox CLAD only
    'death'         : Cox Death only

    Key design: losses accumulate as live tensors; ONE backward per flush.
    No per-record backward, no retain_graph — avoids double-backward errors.

    Survival-only modes ('acr_surv', 'clad', 'death') use FULL-BATCH Cox:
    all hazard tensors collected across the epoch, one backward at the end.
    This fixes the all-censored-batch problem from low event rates (~10-15%).
    """
    import random as _random

    tw   = task_weights or DEFAULT_TASK_WEIGHTS
    cl   = cox_lambdas  or _DEFAULT_COX_LAMBDAS

    # For alternating: pick ONE task for this whole epoch
    if mode == "alternating":
        tasks   = list(tw.keys())
        weights = [tw[t] for t in tasks]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        active_task = _random.choices(tasks, weights=weights, k=1)[0]
    else:
        active_task = mode   # 'simultaneous' or a specific task name

    # Reconstruction task → GeoMAE backbone denoising epoch
    if active_task == "recon":
        stats = p2_recon_epoch(model, records, optimizer, device, bag_cache,
                               scaler, grad_accum=4)
        return {"mode": "recon", "recon_loss": stats["loss"],
                "n_records": stats["n_records"], "hinge": 0.0, "cox": 0.0}

    # Pure survival tasks → full-batch Cox (fixes all-censored mini-batch problem)
    _PURE_SURV = {"acr_surv", "clad", "death"}
    if active_task in _PURE_SURV:
        cox_loss = p2_cox_full_epoch(
            model, records, active_task, optimizer, device, bag_cache, scaler,
            cox_lambda=cl.get(active_task, 1.0),
        )
        return {"mode": active_task, "hinge": 0.0, "cox": cox_loss}

    # Stratify record order around the active task
    strat_task = active_task if active_task != "simultaneous" else "acr_cls"
    ordered    = _stratified_record_order(records, strat_task)

    model.train()
    use_amp  = scaler is not None
    bags_buf = {m: None for m in MODALITIES}
    bags_buf["HE_coords"] = None
    OOM_SKIP  = 3; oom_counts: dict = {}

    total_hinge = 0.0; total_cox = 0.0
    n_hinge = 0; n_cox_steps = 0
    accum_step = 0

    # Pending tensor loss — accumulate WITHOUT calling backward until flush
    pending: Optional[torch.Tensor] = None
    cox_bufs: Dict[str, list] = {k: [] for k in _SURV_KEYS}
    optimizer.zero_grad()

    def _should_train(task: str) -> bool:
        if active_task == "simultaneous": return True
        return task == active_task

    def _flush():
        nonlocal pending, n_cox_steps
        combined = pending
        for tk, buf in cox_bufs.items():
            if not buf or not _should_train(tk): continue
            lam   = cl.get(tk, 1.0)
            L_cox = cox_breslow_loss(buf)
            if L_cox is not None and L_cox.requires_grad:
                term     = L_cox * lam
                combined = term if combined is None else combined + term
                total_cox_ref[0] += L_cox.item()
                n_cox_steps += 1
            buf.clear()
        if combined is not None and combined.requires_grad:
            if scaler:
                scaler.scale(combined).backward()
                scaler.step(optimizer); scaler.update()
            else:
                combined.backward(); optimizer.step()
        optimizer.zero_grad()
        pending = None
        _gc()

    total_cox_ref = [0.0]

    for rec in ordered:
        if oom_counts.get(rec["stem"], 0) >= OOM_SKIP:
            continue
        entry = bag_cache.get(rec["stem"], {})
        for m in MODALITIES:
            bags_buf[m] = entry.get(m)
        bags_buf["HE_coords"] = entry.get("HE_coords")
        if all(bags_buf.get(m) is None for m in MODALITIES):
            continue

        try:
            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model(bags_buf, device)
            task_out = _parse_model_output(result)

            # ── Hinge (acr_cls) ───────────────────────────────────────
            if _should_train("acr_cls"):
                label = rec.get("label")
                logit = task_out.get("acr_cls", (None,))[0]
                if (label is not None and isinstance(logit, torch.Tensor)
                        and logit.grad_fn is not None):
                    target = torch.tensor([float(label)], device=device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        L_h = hinge_loss(logit.unsqueeze(0), target, cw) / grad_accum
                    pending = L_h if pending is None else pending + L_h
                    total_hinge += L_h.item() * grad_accum
                    n_hinge += 1

            # ── Cox losses ────────────────────────────────────────────
            for tk, (tk_key, ev_key) in _SURV_KEYS.items():
                if not _should_train(tk): continue
                haz = task_out.get(tk, (None,))[0]
                if not isinstance(haz, torch.Tensor): continue
                t_v = rec.get(tk_key, float("nan"))
                e_v = rec.get(ev_key, float("nan"))
                if isinstance(t_v, float) and not math.isnan(t_v) and t_v >= 0:
                    e_s = float(e_v) if (e_v is not None
                                         and not math.isnan(float(e_v))) else 0.0
                    cox_bufs[tk].append((haz.float(), float(t_v), e_s))

            accum_step += 1

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); optimizer.zero_grad()
            for buf in cox_bufs.values(): buf.clear()
            pending = None; accum_step = 0
            oom_counts[rec["stem"]] = oom_counts.get(rec["stem"], 0) + 1
            if oom_counts[rec["stem"]] >= OOM_SKIP:
                print(f"  [OOM] {rec['stem']} permanently skipped", flush=True)
            continue

        if accum_step == grad_accum:
            _flush(); accum_step = 0

    if accum_step > 0:
        _flush()

    # For simultaneous mode: CLAD and Death used mini-batch Cox above, which
    # suffers from all-censored windows. Supplement with a full-batch Cox pass
    # for low-event-rate endpoints to ensure every patient contributes.
    extra_cox = 0.0
    if active_task == "simultaneous":
        for tk in ("clad", "death"):
            extra_cox += p2_cox_full_epoch(
                model, records, tk, optimizer, device, bag_cache, scaler,
                cox_lambda=cl.get(tk, 1.0),
            )

    return {
        "mode":  active_task,
        "hinge": total_hinge / max(n_hinge, 1),
        "cox":   (total_cox_ref[0] / max(n_cox_steps, 1)) + extra_cox,
    }


# ── Fair evaluation (same test set for all models) ────────────────────────────

@torch.no_grad()
def p2_evaluate_fair(model, all_test_records, device, bag_cache,
                     cw=None, majority_label: int = 0):
    """
    Evaluate on the SAME set of records regardless of modality availability.

    Records where NO modality is present receive a fixed 'no-data' prediction
    (majority_label → probability 0.5 or 0.0) so the denominator is constant
    across single-modal and multimodal models.

    Returns (probs, labels) over all records.
    """
    model.eval()
    probs, labels = [], []
    use_amp = (device.type == "cuda")

    for rec in all_test_records:
        label = rec.get("label")
        if label is None:
            continue

        bags = {m: bag_cache.get(rec["stem"], {}).get(m) for m in MODALITIES}
        bags["HE_coords"] = bag_cache.get(rec["stem"], {}).get("HE_coords")
        has_data = any(bags.get(m) is not None for m in MODALITIES)

        if not has_data:
            # No modality available: predict majority-class probability
            prob = float(majority_label)
            probs.append(prob); labels.append(label)
            continue

        try:
            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model(bags, device)
            if isinstance(result, dict):
                logit = result.get("acr_cls", (None,))[0]
            elif isinstance(result, tuple):
                logit = result[0]
            else:
                logit = result
            if logit is None or not isinstance(logit, torch.Tensor):
                probs.append(float(majority_label)); labels.append(label)
                continue
            probs.append(torch.sigmoid(logit.float()).item())
            labels.append(label)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            probs.append(float(majority_label)); labels.append(label)

    return np.array(probs), np.array(labels)


# ── Phase 2 HP sweep ──────────────────────────────────────────────────────────

def run_phase2_hp_sweep(
    model_factory,       # callable () → fresh model on CPU
    records_train: List[dict],
    records_val:   List[dict],
    device: torch.device,
    bag_cache: BagCache,
    save_dir: Path,
    task: str = "mega",
    lr_grid: List[float] = P2_HP_LR_GRID,
    wd_grid: List[float] = P2_HP_WD_GRID,
    alternating: bool = False,
    sweep_epochs: int = P2_HP_SWEEP_EPOCHS,
    eval_every: int = P2_EVAL_EVERY,
) -> Tuple[float, float]:
    """
    Grid-search (lr, weight_decay) on val BACC for Phase 2.

    Returns (best_lr, best_wd).
    Caller should rebuild model and retrain on train+val with these HP.
    """
    from itertools import product as iproduct
    save_dir.mkdir(parents=True, exist_ok=True)
    result_path = save_dir / ("hp_sweep_p2_alt.json" if alternating else "hp_sweep_p2.json")
    if result_path.exists():
        with open(result_path) as f:
            res = json.load(f)
        print(f"  [P2-HP] Already done: lr={res['best_lr']}  "
              f"wd={res['best_wd']}  val_bacc={res['best_val_bacc']:.4f}")
        return res["best_lr"], res["best_wd"]

    cw  = compute_class_weights(records_train)
    _tm = {"cls":       "acr_cls",
           "acr_surv":  "acr_surv",
           "surv":      "acr_surv",
           "clad_surv": "clad",
           "death_surv":"death",
           "mega":      "simultaneous",
           "both":      "simultaneous"}
    train_mode = ("alternating" if alternating else _tm.get(task, "simultaneous"))

    _surv_ep_map = {"acr_surv": "acr", "surv": "acr",
                    "clad_surv": "clad", "death_surv": "death"}
    _is_surv_only = task in _surv_ep_map
    _hp_surv_ep   = _surv_ep_map.get(task, "acr")

    # Survival: minimize Cox loss (lower = better). Cls: maximize BACC.
    best_metric = float("inf") if _is_surv_only else -1.0
    best_lr, best_wd = lr_grid[0], wd_grid[0]
    results = []

    for lr, wd in iproduct(lr_grid, wd_grid):
        print(f"  [P2-HP] lr={lr:.0e}  wd={wd:.0e}", end="  ", flush=True)
        model = model_factory().to(device)
        opt   = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
        best_ep_metric = -1.0

        best_ep_metric = float("inf") if _is_surv_only else -1.0
        for ep in range(sweep_epochs):
            p2_train_epoch(model, records_train, opt, cw, device, bag_cache,
                           scaler, P2_GRAD_ACCUM, mode=train_mode)
            if (ep + 1) % eval_every == 0:
                vp, vl, val_loss_ep, ci, *_ = p2_evaluate(
                    model, records_val, device, bag_cache,
                    cw=(cw if not _is_surv_only else None),
                    surv_endpoint=_hp_surv_ep, task=task)
                if _is_surv_only:
                    # Use Cox loss (lower = better) — more stable than C-index on small val
                    metric = float(val_loss_ep) if val_loss_ep > 0.0 else float("inf")
                    best_ep_metric = min(best_ep_metric, metric)
                else:
                    metric = compute_metrics(vl, vp)["bacc"]
                    best_ep_metric = max(best_ep_metric, metric)
                _gc()

        metric_label = "val_cox" if _is_surv_only else "val_bacc"
        print(f"{metric_label}={best_ep_metric:.4f}")
        results.append({"lr": lr, "wd": wd, metric_label: best_ep_metric})
        # For survival: lower Cox loss wins; for cls: higher BACC wins
        if _is_surv_only:
            if best_ep_metric < best_metric:
                best_metric, best_lr, best_wd = best_ep_metric, lr, wd
        else:
            if best_ep_metric > best_metric:
                best_metric, best_lr, best_wd = best_ep_metric, lr, wd
        del model, opt, scaler; _gc()

    res = {"best_lr": best_lr, "best_wd": best_wd,
           "best_val_bacc": best_metric, "grid": results}
    with open(result_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  [P2-HP] done: lr={best_lr}  wd={best_wd}  val_bacc={best_metric:.4f}")
    return best_lr, best_wd


# ── LR schedule helper ────────────────────────────────────────────────────────

def _flat_cosine_scheduler(optimizer, n_epochs: int,
                            warmup_frac: float = P2_WARMUP_FRAC,
                            flat_frac: float = 0.70):
    """Warmup → flat → cosine tail.
    Default: 10% warmup, 70% flat, 20% cosine decay.
    Keeps full LR for most of training; decays only at the end.
    """
    warmup_eps  = max(1, int(n_epochs * warmup_frac))
    decay_start = warmup_eps + int(n_epochs * flat_frac)

    def _lr_lambda(ep: int) -> float:
        if ep < warmup_eps:
            return (ep + 1) / warmup_eps
        if ep < decay_start:
            return 1.0
        prog = (ep - decay_start) / max(n_epochs - decay_start, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# ── Single-modality baseline eval (fair: same test records) ──────────────────

@torch.no_grad()
@torch.no_grad()
def evaluate_unimodal_ablation(
    model: nn.Module,
    records: List[dict],
    device: torch.device,
    bag_cache: BagCache,
    surv_endpoint: str = "acr",
    task: str = "acr_cls",
) -> Dict[str, dict]:
    """
    Evaluate the trained multimodal model with only ONE modality at a time.

    For each modality M: set all other modality bags to None → forward pass →
    record BACC / C-index.  This tests whether the multimodal model can fall
    back gracefully when only one data source is available at inference.

    Returns dict: { "HE": {"bacc": ..., "c_index": ...}, "BAL": {...}, ... }
    """
    _surv_key_map = {
        "acr":   ("tte_next_acr",  "event_next_acr"),
        "clad":  ("clad_time",      "clad_event"),
        "death": ("death_time",     "death_event"),
    }
    t_key, e_key = _surv_key_map.get(surv_endpoint, ("tte_next_acr", "event_next_acr"))
    _is_surv = task in {"acr_surv", "clad", "clad_surv", "death", "death_surv", "surv"}

    use_amp = (device.type == "cuda")
    results: Dict[str, dict] = {}

    for active_mod in MODALITIES:
        probs, labels_cls = [], []
        hazards, surv_times, surv_events = [], [], []

        for rec in records:
            # Build bags with only active_mod; all others set to None
            bags = {m: None for m in MODALITIES}
            bags["HE_coords"] = None
            b = bag_cache.get(rec["stem"], {}).get(active_mod)
            if b is None:
                continue                    # skip records missing this modality
            bags[active_mod] = b.to(device, non_blocking=True)
            if active_mod == "HE":
                bags["HE_coords"] = bag_cache.get(rec["stem"], {}).get("HE_coords")

            try:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    result = model(bags, device)

                if isinstance(result, dict):
                    logit  = result.get("acr_cls", (None,))[0]
                    _hk    = {"acr": "acr_surv", "clad": "clad", "death": "death"}.get(
                        surv_endpoint, surv_endpoint)
                    hazard = result.get(_hk, (None,))[0]
                elif isinstance(result, tuple) and len(result) >= 3:
                    logit, hazard = result[0], result[1]
                elif isinstance(result, tuple):
                    logit = result[0]; hazard = None
                else:
                    logit = result; hazard = None

                label = rec.get("label")
                if label is not None and isinstance(logit, torch.Tensor):
                    probs.append(torch.sigmoid(logit.float()).item())
                    labels_cls.append(label)

                t_val = rec.get(t_key, float("nan"))
                e_val = rec.get(e_key, float("nan"))
                haz_src = hazard if hazard is not None else (logit if _is_surv else None)
                if (haz_src is not None and isinstance(haz_src, torch.Tensor)
                        and isinstance(t_val, float) and not math.isnan(t_val) and t_val >= 0):
                    e_safe = float(e_val) if (e_val is not None
                                              and not math.isnan(float(e_val))) else 0.0
                    hazards.append(haz_src.float().item())
                    surv_times.append(float(t_val))
                    surv_events.append(e_safe)
            except Exception:
                continue

        out: dict = {"n": len(probs) + len(hazards)}
        if not _is_surv and probs and labels_cls:
            try:
                m = compute_metrics(np.array(labels_cls), np.array(probs))
                out["bacc"] = m.get("bacc")
                out["auc"]  = m.get("auc")
            except Exception:
                pass
        if len(hazards) >= 2 and sum(surv_events) > 0:
            out["c_index"] = c_index(hazards, surv_times, surv_events)
        results[active_mod] = out

    return results


def run_single_modal_eval(
    p1_ckpt_path: Path,
    mod_name: str,
    all_test_records: List[dict],
    device: torch.device,
    bag_cache: BagCache,
    majority_label: int = 0,
    surv_endpoint: str = "acr",   # acr | clad | death
) -> dict:
    """
    Evaluate a Phase 1 single-modality model on the SAME test records used
    for multimodal evaluation (fair comparison).

    For classification (surv_endpoint='acr' with label available): BACC.
    For survival (surv_endpoint in clad/death/acr): C-index on hazard head.
    Records where the modality is missing get majority-class prediction.
    """
    from mil.models.phase1 import SingleModalMIL
    from mil.data.registry import _feat_dim

    _surv_keys = {
        "acr":   ("tte_next_acr",  "event_next_acr"),
        "clad":  ("clad_time",      "clad_event"),
        "death": ("death_time",     "death_event"),
    }
    t_key, e_key = _surv_keys.get(surv_endpoint, ("tte_next_acr", "event_next_acr"))

    state = torch.load(p1_ckpt_path, map_location="cpu", weights_only=False)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    feat_dim = _feat_dim(mod_name)
    w = state.get("encoder.backbone.0.weight")
    if w is not None:
        feat_dim = int(w.shape[1])
    model = SingleModalMIL(feat_dim, use_spatial=(mod_name == "HE")).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    del state

    probs, labels_cls = [], []
    hazards, surv_times, surv_events = [], [], []
    use_amp = (device.type == "cuda")

    for rec in all_test_records:
        bag = bag_cache.get(rec["stem"], {}).get(mod_name)
        use_amp_ = use_amp and bag is not None
        bag_dev  = bag.to(device, non_blocking=True) if bag is not None else None

        # Classification output
        label = rec.get("label")
        if label is not None:
            if bag_dev is not None:
                with torch.amp.autocast("cuda", enabled=use_amp_):
                    logit, extras = model(bag_dev, return_extras=True)
                probs.append(torch.sigmoid(logit.float()).item())
                # also collect hazard from extras for survival
                haz = extras.get("hazard")
            else:
                logit = None
                probs.append(float(majority_label))
                haz = None
            labels_cls.append(label)
        else:
            if bag_dev is not None:
                with torch.amp.autocast("cuda", enabled=use_amp_):
                    _, extras = model(bag_dev, return_extras=True)
                haz = extras.get("hazard")
            else:
                haz = None

        # Survival output
        t_val = rec.get(t_key, float("nan"))
        e_val = rec.get(e_key, float("nan"))
        if (isinstance(t_val, float) and not math.isnan(t_val) and t_val >= 0
                and haz is not None):
            e_safe = float(e_val) if (e_val is not None
                                       and not math.isnan(float(e_val))) else 0.0
            hazards.append(haz.float().item())
            surv_times.append(float(t_val))
            surv_events.append(e_safe)

        if bag_dev is not None:
            del bag_dev

    del model; _gc()

    result: dict = {}
    # Classification metrics
    if probs and labels_cls:
        probs_np  = np.array(probs)
        labels_np = np.array(labels_cls)
        try:
            m = compute_metrics(labels_np, probs_np)
            result.update(m)
        except Exception:
            pass

    # Survival C-index
    if len(hazards) >= 2 and sum(surv_events) > 0:
        result["c_index"] = c_index(hazards, surv_times, surv_events)

    return result


# ── Phase 2 final trainer (train+val, fixed epochs, no val leakage) ──────────

def run_phase2_final(
    model: nn.Module,
    variant: str,
    fold: int,
    device: torch.device,
    bag_cache: "BagCache",
    train_recs: List[dict],
    val_recs: List[dict],
    test_recs: List[dict],
    save_dir: Path,
    tag: Optional[str] = None,
    lr: float = P2_LR,
    weight_decay: float = P2_WEIGHT_DECAY,
    n_epochs: int = P2_HP_SWEEP_EPOCHS,  # same as HP sweep — honest epoch count
    warmup_frac: float = P2_WARMUP_FRAC,
    task: str = "mega",
) -> dict:
    """
    Final Phase 2 training using the proper nested CV protocol:

      1. HP sweep already selected best (lr, wd) on val set  (on train only).
      2. THIS function retrains on train+val combined for n_epochs fixed epochs
         — no held-out val, no early stopping, no val leakage.
      3. Saves the final (last-epoch) model and evaluates once on test.

    n_epochs defaults to P2_HP_SWEEP_EPOCHS (150) — the epoch count used in
    the HP sweep. That is the honest estimate of how long to train before
    memorisation kicks in. Using more epochs with no stopping signal risks
    overfitting.
    """
    vtag     = tag or variant
    is_mega  = task in ("mega", "both", "both_alt")
    _surv_ep = "acr"

    print(f"\n  {'='*60}")
    print(f"  Phase 2 FINAL [{vtag}]  fold={fold}  task={task}  epochs={n_epochs}")
    print(f"  lr={lr:.0e}  wd={weight_decay:.0e}  train+val combined (no early stopping)")
    print(f"  {'='*60}")
    save_dir.mkdir(parents=True, exist_ok=True)

    status_path = save_dir / f"status_{vtag}_final.json"
    if _is_completed(save_dir, tag=f"status_{vtag}_final"):
        st = _read_status(status_path)
        print(f"  [{vtag}_final] Already completed (ep={st.get('last_epoch')}). Skipping.")
        mf = save_dir / f"metrics_{vtag}_final.json"
        if mf.exists():
            with open(mf) as f: return json.load(f)
        return {}

    # Combine train+val — no held-out set during training
    all_train = train_recs + val_recs
    print(f"  Combined train+val: {len(all_train)} records  "
          f"(train={len(train_recs)}, val={len(val_recs)})")

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable={n_tr:,}")

    cw        = compute_class_weights(all_train)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay)
    scheduler = _flat_cosine_scheduler(optimizer, n_epochs, warmup_frac)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print(f"  cw=(neg={cw[0]:.3f}, pos={cw[1]:.3f})")

    for epoch in range(n_epochs):
        loss_d = p2_train_epoch(
            model, all_train, optimizer, cw, device, bag_cache,
            scaler, P2_GRAD_ACCUM, mode="simultaneous")
        tl = loss_d["hinge"] + loss_d["cox"]
        scheduler.step()
        if (epoch + 1) % 50 == 0 or epoch == n_epochs - 1:
            print(f"  [{vtag}_final] ep {epoch+1:3d}/{n_epochs}  train_loss={tl:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")
        _gc()

    # Save final model (last epoch — no checkpoint selection bias)
    torch.save(model.state_dict(), save_dir / f"model_{vtag}_final.pt")
    _write_status(status_path, completed=True, last_epoch=n_epochs,
                  lr=lr, weight_decay=weight_decay)
    print(f"\n  [{vtag}_final] Training complete. Evaluating on test set...")

    # Evaluate on test only (train/val used for training — can't give unbiased estimate)
    all_metrics: dict = {}
    for sn, recs in [("test", test_recs)]:
        p, l, _, ci, h_list, t_list, e_list = p2_evaluate(
            model, recs, device, bag_cache, surv_endpoint=_surv_ep, task=task)
        m = compute_metrics(l, p)
        m["auprc"] = average_precision_score(l, p) if len(np.unique(l)) > 1 else 0.0
        if ci is not None:
            m["c_index"] = ci
        if is_mega:
            for ep_name in ("clad", "death"):
                _, _, _, ci_ep, *_ = p2_evaluate(model, recs, device, bag_cache,
                                                  surv_endpoint=ep_name, task=task)
                if ci_ep is not None:
                    m[f"{ep_name}_c_index"] = ci_ep
        all_metrics[sn] = {**m, "probs": p.tolist(), "labels": l.tolist()}
        ci_str = f"  C-idx(ACR)={ci:.4f}" if ci is not None else ""
        if is_mega:
            ci_str += (f"  C-idx(CLAD)={m.get('clad_c_index', float('nan')):.4f}"
                       f"  C-idx(Death)={m.get('death_c_index', float('nan')):.4f}")
        print(f"  [{vtag}_final] {sn:5s}  AUC={m['auc']:.4f}  BAcc={m['bacc']:.4f}"
              f"  MCC={m.get('mcc',0):.4f}{ci_str}")

    # Unimodal ablation
    try:
        model.eval()
        uni_abl = evaluate_unimodal_ablation(
            model, test_recs, device, bag_cache, surv_endpoint=_surv_ep, task=task)
        all_metrics["unimodal_ablation"] = uni_abl
    except Exception as e:
        print(f"  [{vtag}_final] unimodal ablation failed: {e}")

    with open(save_dir / f"metrics_{vtag}_final.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    del model, optimizer, scaler; _gc()
    return all_metrics


# ── Phase 2 runner ────────────────────────────────────────────────────────────

def run_phase2_variant(model: nn.Module, variant: str, fold: int,
                       device: torch.device, bag_cache: BagCache,
                       train_recs: List[dict], val_recs: List[dict],
                       test_recs: List[dict], save_dir: Path,
                       tag: Optional[str] = None,
                       patience: int = 10,
                       cox_lambda: float = 0.0,
                       surv_endpoint: str = 'clad',
                       task: str = 'acr',
                       lr: float = P2_LR,
                       weight_decay: float = P2_WEIGHT_DECAY,
                       n_epochs: Optional[int] = None,
                       warmup_frac: float = P2_WARMUP_FRAC,
                       alternating: bool = False,
                       # legacy kwargs accepted and ignored
                       use_contrastive: bool = False,
                       clr_lambda: float = P1_CLR_LAMBDA,
                       clr_tau: float = P1_CLR_TAU,
                       recon_lambda: float = 0.0,
                       ) -> dict:
    vtag    = tag or variant
    is_slot = any(s in variant for s in _SLOT_VARIANTS)
    total_eps = n_epochs or (P2_EPOCHS_SLOT if is_slot else P2_EPOCHS)

    # Map --task → (training mode, survival endpoint for eval, is_surv_only)
    _task_map = {
        "cls":        ("acr_cls",       "acr",   False),  # BACC
        "acr_surv":   ("acr_surv",      "acr",   True),   # C-index, ACR TTE
        "surv":       ("acr_surv",      "acr",   True),   # alias
        "clad_surv":  ("clad",          "clad",  True),   # C-index, CLAD TTE
        "death_surv": ("death",         "death", True),   # C-index, Death TTE
        "mega":       ("simultaneous",  "acr",   False),  # BACC (primary) + C-index logged
        "both":       ("simultaneous",  "acr",   False),
        "both_alt":   ("simultaneous",  "acr",   False),
    }
    _default_train_mode, _surv_ep, _is_surv_only = _task_map.get(
        task, ("simultaneous", surv_endpoint, False))
    if alternating:
        _default_train_mode = "alternating"
    is_mega = (task in ("mega", "both", "both_alt"))

    print(f"\n  {'='*60}")
    print(f"  Phase 2 v8 [{vtag}]  fold={fold}  task={task}  epochs={total_eps}")
    print(f"  lr={lr:.0e}  wd={weight_decay:.0e}  warmup={warmup_frac:.0%}")
    print(f"  {'='*60}")
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = save_dir / f"ckpts_{vtag}"; ckpt_dir.mkdir(exist_ok=True)

    status_path = save_dir / f"status_{vtag}.json"
    if _is_completed(save_dir, tag=f"status_{vtag}"):
        st = _read_status(status_path)
        print(f"  [{vtag}] Already completed "
              f"(best_ep={st.get('best_epoch')}  "
              f"best_bacc={st.get('best_bacc', 0):.4f}). Skipping.")
        mf = save_dir / f"metrics_{vtag}.json"
        if mf.exists():
            with open(mf) as f: return json.load(f)
        return {}

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_fr = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable={n_tr:,}  Frozen={n_fr:,}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay)
    scheduler = _flat_cosine_scheduler(optimizer, total_eps, warmup_frac)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    cw     = compute_class_weights(train_recs)
    print(f"  cw=(neg={cw[0]:.3f}, pos={cw[1]:.3f})  "
          f"modal_dropout={P2_MODAL_DROPOUT}  grad_accum={P2_GRAD_ACCUM}")

    history = {k: [] for k in
               ["train_loss","val_loss","val_auc","val_bacc","val_mcc",
                "lr"]}

    resume_epoch = _find_resume_epoch(ckpt_dir); start_epoch = 0
    if resume_epoch >= total_eps:
        print(f"  [{vtag}] Already complete (ep={resume_epoch}). Rescanning.")
        start_epoch = total_eps
    elif resume_epoch > 0:
        ckpt = _load_checkpoint(ckpt_dir, resume_epoch)
        if ckpt is not None:
            model.load_state_dict(
                ckpt["model"] if isinstance(ckpt, dict) else ckpt, strict=False)
            if isinstance(ckpt, dict) and "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if isinstance(ckpt, dict) and scaler and ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            if isinstance(ckpt, dict) and "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if isinstance(ckpt, dict) and "history" in ckpt:
                history = ckpt["history"]
                # back-fill lr key if not in older checkpoint
                if "lr" not in history:
                    history["lr"] = [lr] * len(history.get("train_loss", []))
            model.to(device); start_epoch = resume_epoch
            print(f"  [{vtag}] Resumed from ep {resume_epoch}")

    # Survival: track val Cox loss (lower = better). Cls: val BACC (higher = better).
    if _is_surv_only:
        _p2_best_metric: float = min(history["val_loss"]) if history.get("val_loss") else float("inf")
    else:
        _p2_best_metric: float = max(history["val_bacc"]) if history["val_bacc"] else -1.0
    _p2_best_ep:   int   = 0
    _p2_no_improve: int  = 0

    for epoch in range(start_epoch, total_eps):
        loss_d = p2_train_epoch(
            model, train_recs, optimizer, cw, device, bag_cache,
            scaler, P2_GRAD_ACCUM,
            mode=_default_train_mode,
            task_weights=(GEOMAE_TASK_WEIGHTS if task == "geomae_alt"
                          else DEFAULT_TASK_WEIGHTS if alternating
                          else None),
        )
        tl = loss_d["hinge"] + loss_d["cox"]
        scheduler.step()
        history["train_loss"].append(tl)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        _gc()

        if (epoch + 1) % P2_EVAL_EVERY == 0:
            vl_p, vl_l, val_loss, ci, *_ = p2_evaluate(
                model, val_recs, device, bag_cache,
                cw=(cw if not _is_surv_only else None),
                surv_endpoint=_surv_ep, task=task)
            model.train()

            if _is_surv_only:
                cidx_val = float(ci) if ci is not None else 0.0
                val_loss = float(val_loss) if val_loss is not None else float("inf")
                history["val_loss"].append(val_loss)
                history["val_auc"].append(cidx_val)
                history["val_bacc"].append(cidx_val)
                history["val_mcc"].append(0.0)
                metric_str = f"cox_loss={val_loss:.4f}  cidx={cidx_val:.3f}"
                # Checkpoint: lower Cox loss is better (more principled than val C-idx)
                improved = (val_loss > 0.0 and val_loss < _p2_best_metric)
            else:
                vm = compute_metrics(vl_l, vl_p)
                primary_metric = vm["bacc"]
                history["val_loss"].append(val_loss)
                history["val_auc"].append(vm["auc"])
                history["val_bacc"].append(vm["bacc"])
                history["val_mcc"].append(vm.get("mcc", 0.0))
                ci_str = f"  C-idx={ci:.3f}" if ci is not None else ""
                metric_str = f"auc={vm['auc']:.3f}  bacc={vm['bacc']:.3f}{ci_str}"
                improved = (primary_metric > _p2_best_metric)

            torch.save({
                "epoch": epoch+1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "scheduler": scheduler.state_dict(),
                "history": history,
            }, ckpt_dir / f"ep{epoch+1:04d}.pt")

            if improved:
                _p2_best_metric = val_loss if _is_surv_only else primary_metric
                _p2_best_ep     = epoch + 1
                _p2_no_improve  = 0
                torch.save(model.state_dict(), save_dir / f"model_{vtag}.pt")
                ckpt_tag = "[ckpt*]"
            else:
                _p2_no_improve += 1
                ckpt_tag = "[ckpt]"
            improve_str = (f"  no_improve={_p2_no_improve}/{patience}"
                           if patience > 0 else "")
            print(f"  [{vtag}] ep {epoch+1:3d}  loss={tl:.4f}/{val_loss:.4f}  "
                  f"{metric_str}  {ckpt_tag}{improve_str}")
            _gc()
            if patience > 0 and _p2_no_improve >= patience:
                print(f"  [{vtag}] Early stop: {_p2_no_improve} eval periods "
                      f"without improvement (patience={patience}).")
                break
        elif (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{vtag}] ep {epoch+1:3d}  train_loss={tl:.4f}")

    # ── Finalise best model ────────────────────────────────────────
    ckpts = sorted(ckpt_dir.glob("ep*.pt"))
    if not ckpts and not (save_dir / f"model_{vtag}.pt").exists():
        print(f"  [{vtag}] [warn] no checkpoints."); return {}

    best_model_path = save_dir / f"model_{vtag}.pt"
    if best_model_path.exists() and _p2_best_ep > 0:
        print(f"\n  [{vtag}] Using inline best "
              f"(ep={_p2_best_ep}  metric={_p2_best_metric:.4f})")
        state = torch.load(best_model_path, map_location="cpu", weights_only=False)
        state = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state, strict=False); model.to(device); del state
        best_ep, best_bacc = _p2_best_ep, _p2_best_metric
    elif ckpts:
        print(f"\n  [{vtag}] Fast-scanning {len(ckpts)} checkpoint histories ...")
        best_bacc, best_ep, best_path = -1.0, 0, ckpts[-1]
        for cp in ckpts:
            try:
                data = torch.load(cp, map_location="cpu", weights_only=False)
                hist_b = data.get("history", {}).get("val_bacc", [])
                b = max(hist_b) if hist_b else -1.0
                if b > best_bacc:
                    best_bacc, best_ep, best_path = b, int(cp.stem[2:]), cp
                del data
            except Exception: pass
        print(f"  [{vtag}] best ep≈{best_ep}  metric={best_bacc:.4f}")
        data  = torch.load(best_path, map_location="cpu", weights_only=False)
        state = data["model"] if isinstance(data, dict) else data
        model.load_state_dict(state, strict=False); model.to(device); del data, state
        torch.save(model.state_dict(), best_model_path)
    else:
        best_ep, best_bacc = 0, 0.0

    _write_status(status_path, completed=True,
                  best_epoch=best_ep, best_bacc=round(best_bacc, 4),
                  last_epoch=_p2_best_ep or total_eps,
                  lr=lr, weight_decay=weight_decay)

    all_metrics: dict = {}
    for sn, recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        p, l, _, ci, h_list, t_list, e_list = p2_evaluate(model, recs, device, bag_cache,
                                                            surv_endpoint=_surv_ep, task=task)
        if _is_surv_only:
            all_metrics[sn] = {
                "c_index": ci or 0.0,
                "probs":   h_list,   # actual hazard scores (not sigmoid logit)
                "labels":  e_list,   # event flags
                "times":   t_list,   # survival times
            }
            print(f"  [{vtag}] {sn:5s}  C-index={ci:.4f}" if ci else
                  f"  [{vtag}] {sn:5s}  C-index=N/A")
        else:
            m = compute_metrics(l, p)
            m["auprc"] = average_precision_score(l, p) if len(np.unique(l)) > 1 else 0.0
            if ci is not None:
                m["c_index"] = ci  # ACR surv C-index
            # For mega task: also evaluate CLAD and death survival C-indices
            if is_mega:
                for ep_name in ("clad", "death"):
                    _, _, _, ci_ep, *_ = p2_evaluate(model, recs, device, bag_cache,
                                                      surv_endpoint=ep_name, task=task)
                    if ci_ep is not None:
                        m[f"{ep_name}_c_index"] = ci_ep
            all_metrics[sn] = {**m, "probs": p.tolist(), "labels": l.tolist()}
            ci_str = f"  C-idx(ACR)={ci:.4f}" if ci is not None else ""
            if is_mega:
                ci_str += (f"  C-idx(CLAD)={m.get('clad_c_index', float('nan')):.4f}"
                           f"  C-idx(Death)={m.get('death_c_index', float('nan')):.4f}")
            print(f"  [{vtag}] {sn:5s}  AUC={m['auc']:.4f}  AUPRC={m['auprc']:.4f}  "
                  f"BAcc={m['bacc']:.4f}  MCC={m.get('mcc',0):.4f}  "
                  f"Sens={m['sens']:.4f}  Spec={m['spec']:.4f}{ci_str}")

    # ── Unimodal ablation: multimodal model with only 1 modality at test time ──
    print(f"  [{vtag}] Running unimodal ablation (multimodal model, 1 modality at a time)...")
    try:
        model.eval()
        uni_abl = evaluate_unimodal_ablation(
            model, test_recs, device, bag_cache,
            surv_endpoint=_surv_ep, task=task)
        all_metrics["unimodal_ablation"] = uni_abl
        for mod, res in uni_abl.items():
            bacc = res.get("bacc"); ci_v = res.get("c_index")
            val  = f"BACC={bacc:.3f}" if bacc is not None else \
                   f"C-idx={ci_v:.3f}" if ci_v is not None else "n/a"
            print(f"  [{vtag}]   {mod:10s} only → {val}  (n={res.get('n',0)})")
    except Exception as e:
        print(f"  [{vtag}] unimodal ablation failed: {e}")

    with open(save_dir / f"metrics_{vtag}.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(save_dir / f"history_{vtag}.json", "w") as f:
        json.dump(history, f)
    del model, optimizer, scaler; _gc()
    return all_metrics
