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
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from mil.data.registry import MODALITIES
from mil.training.losses import (
    hinge_loss, bce_loss, compute_class_weights,
    cox_breslow_loss, c_index,
    batch_supcon_loss,
)
from mil.training.metrics import compute_metrics

# ── Type aliases ──────────────────────────────────────────────────────────────
BagCache = Dict[str, Dict[str, Optional[torch.Tensor]]]

# ── Phase 2 hyperparameter defaults ──────────────────────────────────────────
P2_LR             = 5e-5
P2_WEIGHT_DECAY   = 1e-3
P2_EPOCHS         = 1000   # early/late/middle — extended after ceiling hits at 600
P2_EPOCHS_SLOT    = 1000   # slot needs more iterations (K slots × cross-attn × 4 tasks)
P2_EVAL_EVERY     = 5
P2_GRAD_ACCUM     = 32     # effective batch size (patients per backward step)
P2_MODAL_DROPOUT  = 0.3    # 0.3 → trains with each single modality ~24% of epochs
P2_WARMUP_FRAC    = 0.10   # 10% warmup → cosine decay over remaining 90%

# Slot diversity regularisation — penalises cosine similarity between slot pairs
SLOT_DIV_WEIGHT   = 0.0    # disabled: orthogonal init already separates slots; penalty prevented routing learning


def _slot_div_loss(shared_slots: torch.Tensor) -> torch.Tensor:
    """
    Diversity regularisation on shared slot init tokens.

    Computes mean squared cosine similarity between all pairs of slots and
    returns SLOT_DIV_WEIGHT * that value.  Gradients push slots toward
    orthogonality so each slot specialises to a distinct region.

    shared_slots: (K, H)  — model.shared_slots parameter
    """
    s   = F.normalize(shared_slots.float(), dim=-1)            # (K, H) unit sphere
    sim = s @ s.T                                               # (K, K) cosine sims
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
    return SLOT_DIV_WEIGHT * (sim * (1 - eye)).pow(2).mean()

# HP sweep search space
P2_HP_LR_GRID     = [1e-4, 5e-5, 1e-5]
P2_HP_WD_GRID     = [1e-3, 1e-4]
P2_HP_SWEEP_EPOCHS  = 150  # HP selection sweep
P2_HP_EVAL_EVERY    = 5    # eval interval inside HP sweep (finer than final training)
P2_HP_PATIENCE      = 10   # early-stop if no improvement for this many evals in sweep
P2_FINAL_EPOCHS     = 150  # max epochs for run_phase2_final

# Task Cox lambdas for multitask loss
P2_COX_LAMBDA_ACR   = 0.5   # ACR survival Cox weight
P2_COX_LAMBDA_CLAD  = 0.3   # CLAD Cox weight
P2_COX_LAMBDA_DEATH = 0.2   # Death Cox weight

# BCE scale: Cox losses are ~4× larger than BCE; multiply BCE to match gradient magnitude
P2_BCE_LOSS_SCALE   = 4.0

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


def _slot_collapse_stats(model) -> str:
    """
    Collapse diagnostics from slot Gaussian params (no forward pass needed).
    slot_mu_std : std of slot_mu — near 0=all slots start at same point
    slot_sigma  : mean exp(log_sigma) — spread of slot init distribution
    """
    if not hasattr(model, "slot_mu"):
        return ""
    with torch.no_grad():
        mu_std    = model.slot_mu.std().item()
        sigma_mean = model.slot_log_sigma.exp().mean().item()
    return f"  slot_mu_std={mu_std:.3f}  slot_sigma={sigma_mean:.3f}"


def _routing_entropy_stats(model, bag_cache, records, device) -> str:
    """
    Run one patient through slot attention to compute per-patch routing entropy.

    Uses the pre-renorm competitive softmax (scores.softmax over K slots) — this
    IS the per-patch distribution over slots. Entropy near log(K) = random routing;
    near 0 = sharp single-slot assignment.

    Also reports per-task ABMIL slot importance: run slots through each task's gate
    and show top-5 slots by attention weight + entropy of the distribution.
    """
    if not hasattr(model, "slot_attns") or not hasattr(model, "slot_mu"):
        return ""
    rec = next((r for r in records if any(bag_cache.get(r.get("stem"), {}).get(m) is not None
                                           for m in MODALITIES)), None)
    if rec is None:
        return ""
    entry    = bag_cache.get(rec["stem"], {})
    K        = model.n_slots
    log_K    = math.log(K)
    parts    = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            # Use slot_mu as deterministic init for diagnostics (no randomness)
            slots_init = model.slot_mu.float()  # (K, H) — per-slot learned means
            mod_slots = []
            for mod, slot_attn in model.slot_attns.items():
                t = entry.get(mod)
                if t is None:
                    continue
                t = t.to(device)
                h = model.encoders[mod].encode_patches(t)          # (N, H)
                N, H = h.shape
                nh = slot_attn.n_heads
                dk = slot_attn.d_k

                # Replicate the competitive softmax to get the true per-patch distribution
                h_norm  = slot_attn.norm_in(F.normalize(h.float(), dim=-1))
                k_feat  = slot_attn.to_k(h_norm)                   # (N, H)
                q_feat  = slot_attn.to_q(slot_attn.norm_q(slots_init))  # (K, H)
                q_h = q_feat.view(K, nh, dk).permute(1, 0, 2)     # (nh, K, dk)
                k_h = k_feat.view(N, nh, dk).permute(1, 0, 2)     # (nh, N, dk)
                scores  = torch.bmm(q_h, k_h.transpose(1, 2)) * slot_attn.scale  # (nh, K, N)
                # competitive softmax over K slots: per-patch distribution over slots
                attn_patch = scores.softmax(dim=1).mean(0)         # (K, N) avg over heads
                H_patch = -(attn_patch * (attn_patch + 1e-8).log()).sum(dim=0)   # (N,)
                parts.append(f"{mod}={H_patch.mean().item():.2f}/{log_K:.2f}")

                # run slot attn fully to get slot representations for ABMIL
                slots_out = slot_attn(h, slots_init)               # (K, H)
                mod_slots.append(slots_out)

            routing_str = "  routing_entr=" + " ".join(parts) if parts else ""

            # Per-task ABMIL: which slots get highest attention weight
            abmil_str = ""
            if mod_slots and hasattr(model, "abmil_V") and hasattr(model, "abmil_w"):
                slots_agg = torch.stack(mod_slots).mean(0).float()  # (K, H)
                for tname in model.task_names:
                    gate  = model.abmil_V[tname](slots_agg) * model.abmil_U[tname](slots_agg)
                    alpha = torch.softmax(model.abmil_w[tname](gate), dim=0).squeeze()  # (K,)
                    entr  = -(alpha * (alpha + 1e-8).log()).sum().item()
                    top5  = alpha.topk(5).indices.tolist()
                    abmil_str += f"\n      [{tname}] abmil_entr={entr:.2f}/{log_K:.2f} top_slots={top5}"

    finally:
        if was_training:
            model.train()
    return routing_str + abmil_str


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
# Inverse-frequency weights so all tasks contribute equally:
# observed freq: cls≈0.37, acr_surv≈0.64, clad≈0.66, death≈0.73
# inv-freq norm to sum=1: 1/f / sum(1/f)
DEFAULT_TASK_WEIGHTS = {
    "acr_cls":  0.25,
    "acr_surv": 0.25,
    "clad":     0.25,
    "death":    0.25,
}

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
    try:
        if scaler:
            scaler.scale(L_cox * cox_lambda).backward()
            scaler.step(optimizer); scaler.update()
        else:
            (L_cox * cox_lambda).backward()
            optimizer.step()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  [OOM-cox-backward] {len(cox_buf)} records — skipping Cox update this step",
              flush=True)
        try:
            optimizer.zero_grad()
        except Exception:
            pass
        _gc()
        return 0.0
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
                    loss = bce_loss(logit.unsqueeze(0), target, cw) * P2_BCE_LOSS_SCALE
                    if L_recon is not None and recon_lambda > 0:
                        loss = loss + recon_lambda * L_recon
                pending_loss_ref[0] = (loss if pending_loss_ref[0] is None
                                       else pending_loss_ref[0] + loss)
                if hazard is not None and cox_lambda > 0 and has_surv_data:
                    cox_buffer.append((hazard.float(), rec[surv_time_key], rec[surv_event_key]))
                total_loss += loss.item()
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
                if hasattr(model, "shared_slots"):
                    L_div = _slot_div_loss(model.shared_slots)
                    combined = L_div if combined is None else combined + L_div
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
        if hasattr(model, "shared_slots"):
            L_div = _slot_div_loss(model.shared_slots)
            combined = L_div if combined is None else combined + L_div
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
                    losses.append(bce_loss(logit.unsqueeze(0), ta, cw).item())

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

    total_bce = 0.0; total_cox = 0.0; n_bce = 0; n_cox_steps = 0
    accum_step = 0
    # batch distribution counters (reset each epoch)
    batch_stats = {
        "n_cls": 0, "n_cls_pos": 0,
        "acr_surv_event": 0, "acr_surv_cens": 0,
        "clad_event": 0,     "clad_cens": 0,
        "death_event": 0,    "death_cens": 0,
    }

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
                        L_h = bce_loss(logit_val.unsqueeze(0), target, cw) * P2_BCE_LOSS_SCALE
                    pending_loss = L_h if pending_loss is None else pending_loss + L_h
                    total_bce += L_h.item()
                    n_bce += 1
                    batch_stats["n_cls"] += 1
                    batch_stats["n_cls_pos"] += int(label)

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
                    # track event/censoring counts
                    _tk_short = tk.replace("_surv", "").replace("acr_surv", "acr_surv")
                    if e_safe > 0:
                        batch_stats[f"{_tk_short}_event"] = batch_stats.get(f"{_tk_short}_event", 0) + 1
                    else:
                        batch_stats[f"{_tk_short}_cens"] = batch_stats.get(f"{_tk_short}_cens", 0) + 1

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
        "bce":       total_bce / max(n_bce, 1),
        "cox":         total_cox_ref[0] / max(n_cox_steps, 1),
        "batch_stats": batch_stats,
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
        # Slot diversity: penalise inter-slot cosine similarity (once per step)
        if hasattr(model, "shared_slots"):
            L_div = _slot_div_loss(model.shared_slots)
            combined = L_div if combined is None else combined + L_div
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
                    L = bce_loss(logit_v.unsqueeze(0), target, cw) * P2_BCE_LOSS_SCALE
                pending_loss = L if pending_loss is None else pending_loss + L
                total_loss += L.item()

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

    # Build task sampling distribution for per-batch alternating
    _alt_tasks   = list(tw.keys())
    _alt_weights = [tw[t] for t in _alt_tasks]
    _alt_total   = sum(_alt_weights)
    _alt_weights = [w / _alt_total for w in _alt_weights]

    def _sample_task() -> str:
        return _random.choices(_alt_tasks, weights=_alt_weights, k=1)[0]

    if mode == "alternating":
        # Per-batch: start with a sampled task; _flush() resamples for next batch
        active_task = _sample_task()
    else:
        active_task = mode   # 'simultaneous' or a specific task name

    # Stratify by acr_cls for simultaneous; by active task for alternating (first batch task)
    strat_task = "acr_cls" if active_task == "simultaneous" else active_task
    ordered    = _stratified_record_order(records, strat_task)

    model.train()
    use_amp  = scaler is not None
    bags_buf = {m: None for m in MODALITIES}
    bags_buf["HE_coords"] = None
    OOM_SKIP  = 3; oom_counts: dict = {}

    total_bce = 0.0
    n_bce = 0; n_cox_steps = 0
    accum_step = 0
    per_cox: Dict[str, float] = {k: 0.0 for k in _SURV_KEYS}
    per_cox_n: Dict[str, int]  = {k: 0   for k in _SURV_KEYS}
    batch_stats = {
        "n_cls": 0, "n_cls_pos": 0,
        "acr_surv_event": 0, "acr_surv_cens": 0,
        "clad_event": 0,     "clad_cens": 0,
        "death_event": 0,    "death_cens": 0,
    }
    train_probs: List[float] = []
    train_labels: List[int]  = []

    # Pending tensor loss — accumulate WITHOUT calling backward until flush
    pending: Optional[torch.Tensor] = None
    cox_bufs: Dict[str, list] = {k: [] for k in _SURV_KEYS}
    optimizer.zero_grad()

    def _should_train(task: str) -> bool:
        if active_task == "simultaneous": return True
        return task == active_task

    def _flush():
        nonlocal pending, n_cox_steps, active_task
        combined = pending
        for tk, buf in cox_bufs.items():
            if not buf or not _should_train(tk): continue
            lam   = tw.get(tk, cl.get(tk, 1.0))
            L_cox = cox_breslow_loss(buf)
            if L_cox is not None and L_cox.requires_grad:
                term     = L_cox * lam
                combined = term if combined is None else combined + term
                per_cox[tk]   += L_cox.item()
                per_cox_n[tk] += 1
                n_cox_steps   += 1
            buf.clear()
        if combined is not None and combined.requires_grad:
            if scaler:
                scaler.scale(combined).backward()
                scaler.step(optimizer); scaler.update()
            else:
                combined.backward(); optimizer.step()
        optimizer.zero_grad()
        pending = None
        # Per-batch alternating: resample task for the next batch
        if mode == "alternating":
            active_task = _sample_task()
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
                        L_h = bce_loss(logit.unsqueeze(0), target, cw) * P2_BCE_LOSS_SCALE * tw.get("acr_cls", 1.0)
                    pending = L_h if pending is None else pending + L_h
                    total_bce += L_h.item()
                    n_bce += 1
                    batch_stats["n_cls"] += 1
                    batch_stats["n_cls_pos"] += int(label)
                    train_probs.append(torch.sigmoid(logit.detach().float()).item())
                    train_labels.append(int(label))

            # ── Cox losses ────────────────────────────────────────────
            _surv_stat_keys = {
                "acr_surv": ("acr_surv_event", "acr_surv_cens"),
                "clad":     ("clad_event",     "clad_cens"),
                "death":    ("death_event",    "death_cens"),
            }
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
                    ev_k, ce_k = _surv_stat_keys.get(tk, (None, None))
                    if ev_k:
                        if e_s > 0: batch_stats[ev_k] += 1
                        else:       batch_stats[ce_k] += 1

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
    # (Alternating mode uses per-batch Cox for all survival tasks — no extra pass.)
    extra_cox = 0.0
    if mode == "simultaneous":
        for tk in ("clad", "death"):
            extra_cox += p2_cox_full_epoch(
                model, records, tk, optimizer, device, bag_cache, scaler,
                cox_lambda=tw.get(tk, cl.get(tk, 1.0)),
            )

    train_bacc = compute_metrics(
        np.array(train_labels), np.array(train_probs))["bacc"] if train_labels else 0.5

    return {
        "mode":    mode if mode == "alternating" else active_task,
        "bce":   total_bce / max(n_bce, 1),
        "cox_acr": per_cox["acr_surv"] / max(per_cox_n["acr_surv"], 1),
        "cox_clad":  per_cox["clad"]   / max(per_cox_n["clad"],   1),
        "cox_death": per_cox["death"]  / max(per_cox_n["death"],  1),
        "cox_extra": extra_cox,
        "batch_stats": batch_stats,
        "train_bacc": train_bacc,
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
    eval_every: int = P2_HP_EVAL_EVERY,
    hp_patience: int = P2_HP_PATIENCE,
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
        print(f"  [P2-HP] lr={lr:.0e}  wd={wd:.0e}", flush=True)
        model = model_factory().to(device)
        opt   = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        best_ep_metric = float("inf") if _is_surv_only else -1.0
        no_improve = 0
        stopped_ep = sweep_epochs
        for ep in range(sweep_epochs):
            if hasattr(model, 'routing_temperature'):
                model.routing_temperature = max(0.1, 1.0 - ep / max(sweep_epochs - 1, 1) * 0.9)
            tr = p2_train_epoch(model, records_train, opt, cw, device, bag_cache,
                                scaler, P2_GRAD_ACCUM, mode=train_mode)
            if (ep + 1) % eval_every == 0:
                # primary eval (cls BACC or survival Cox)
                vp, vl, val_loss_ep, ci_primary, *_ = p2_evaluate(
                    model, records_val, device, bag_cache,
                    cw=(cw if not _is_surv_only else None),
                    surv_endpoint=_hp_surv_ep, task=task)
                # C-indices for all 3 survival endpoints (mega only)
                ci_acr = ci_clad = ci_death = None
                if task == "mega":
                    for ep_name in ("acr", "clad", "death"):
                        _, _, _, _ci, *_ = p2_evaluate(
                            model, records_val, device, bag_cache,
                            cw=None, surv_endpoint=ep_name, task=f"{ep_name}_surv")
                        if ep_name == "acr":   ci_acr   = _ci
                        elif ep_name == "clad": ci_clad  = _ci
                        else:                   ci_death = _ci

                val_bacc = compute_metrics(vl, vp)["bacc"] if not _is_surv_only else 0.5
                if _is_surv_only:
                    metric = float(val_loss_ep) if val_loss_ep > 0.0 else float("inf")
                    improved = metric < best_ep_metric
                    best_ep_metric = min(best_ep_metric, metric)
                elif task == "mega":
                    # Combined metric: mean of BACC + all available C-indices
                    parts = [val_bacc]
                    for _ci in (ci_acr, ci_clad, ci_death):
                        if _ci is not None:
                            parts.append(_ci)
                    metric = sum(parts) / len(parts)
                    improved = metric > best_ep_metric
                    best_ep_metric = max(best_ep_metric, metric)
                else:
                    metric = val_bacc
                    improved = metric > best_ep_metric
                    best_ep_metric = max(best_ep_metric, metric)
                no_improve = 0 if improved else no_improve + 1
                metric_label = "val_cox" if _is_surv_only else ("combined" if task == "mega" else "val_bacc")
                stop_flag = f"  [stop in {hp_patience - no_improve}]" if no_improve > 0 else ""

                # batch distribution
                bs = tr.get("batch_stats", {})
                active_str = f"[{tr.get('mode','?')}]" if alternating else ""
                print(
                    f"  ep {ep+1:3d}/{sweep_epochs}{active_str}"
                    f"  bce={tr.get('bce', 0):.4f}"
                    f"  cox_acr={tr.get('cox_acr', 0):.4f}"
                    f"  cox_clad={tr.get('cox_clad', 0):.4f}"
                    f"  cox_death={tr.get('cox_death', 0):.4f}"
                    f"  | train_bacc={tr.get('train_bacc', 0.5):.4f}"
                    f"  val_bacc={val_bacc:.4f}  {metric_label}={metric:.4f}  best={best_ep_metric:.4f}"
                    + (f"  ci_acr={ci_acr:.3f}" if ci_acr is not None else "")
                    + (f"  ci_clad={ci_clad:.3f}" if ci_clad is not None else "")
                    + (f"  ci_death={ci_death:.3f}" if ci_death is not None else "")
                    + f"{stop_flag}"
                    + f"\n    batch: cls={bs.get('n_cls',0)}(pos={bs.get('n_cls_pos',0)})"
                    f"  acr={bs.get('acr_surv_event',0)}ev/{bs.get('acr_surv_cens',0)}ce"
                    f"  clad={bs.get('clad_event',0)}ev/{bs.get('clad_cens',0)}ce"
                    f"  death={bs.get('death_event',0)}ev/{bs.get('death_cens',0)}ce"
                    + _slot_collapse_stats(model)
                    + (f"  temp={model.routing_temperature:.3f}" if hasattr(model, 'routing_temperature') else "")
                    + (_routing_entropy_stats(model, bag_cache, records_train, device)
                       if (ep + 1) % (eval_every * 2) == 0 else ""),
                    flush=True)
                _gc()
                if no_improve >= hp_patience:
                    stopped_ep = ep + 1
                    print(f"  [HP early-stop] no improvement for {hp_patience} evals — stopping combo", flush=True)
                    break

        metric_label = "val_cox" if _is_surv_only else ("combined" if task == "mega" else "val_bacc")
        print(f"  [P2-HP] lr={lr:.0e}  wd={wd:.0e}  DONE  {metric_label}={best_ep_metric:.4f}"
              f"  (stopped ep {stopped_ep}/{sweep_epochs})", flush=True)
        results.append({"lr": lr, "wd": wd, metric_label: best_ep_metric,
                        "stopped_ep": stopped_ep})
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


# ── Phase 2 final trainer (train only, val early-stopping, test eval) ──────────

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
    n_epochs: int = P2_EPOCHS,
    warmup_frac: float = P2_WARMUP_FRAC,
    task: str = "mega",
    patience: int = 20,
    eval_every: int = P2_EVAL_EVERY,
    grad_accum: int = 32,
    combined_train: bool = False,
    alternating: bool = False,
) -> dict:
    """
    Final Phase 2 training with val-monitored early stopping.

    Protocol
    --------
      1. HP sweep already selected best (lr, wd) on val (train split only).
      2. THIS function trains on train_recs ONLY — val_recs remain independent.
      3. Evaluates val every eval_every epochs; saves best-val checkpoint.
      4. Early stopping if val BACC (or Cox loss for surv) does not improve
         for `patience` consecutive eval periods.
      5. Loads best-val checkpoint and evaluates ONCE on test.

    Val is independent throughout (never in training data) so early stopping
    is an honest overfitting signal.  Max 150 epochs guards against runaway.

    If combined_train=True: train_recs already contains train+val merged by caller.
    No early stopping is applied — model trains for full n_epochs and final
    checkpoint is used (all folds share the same test set so val is not independent).
    """
    vtag     = tag or variant
    is_mega  = task in ("mega", "both", "both_alt")
    _task_surv_ep_map = {
        "acr_surv": "acr", "surv": "acr",
        "clad_surv": "clad", "death_surv": "death",
    }
    _surv_ep = _task_surv_ep_map.get(task, "acr")
    is_surv_only = task in ("acr_surv", "surv", "clad_surv", "death_surv")

    print(f"\n  {'='*60}")
    print(f"  Phase 2 FINAL [{vtag}]  fold={fold}  task={task}  max_epochs={n_epochs}")
    print(f"  lr={lr:.0e}  wd={weight_decay:.0e}  patience={patience}  "
          f"eval_every={eval_every}")
    if combined_train:
        print(f"  train={len(train_recs)} (train+val combined)  "
              f"val={len(val_recs)} (monitoring only, no early-stop)")
    else:
        print(f"  train={len(train_recs)}  val={len(val_recs)} (independent, for early-stop)")
    print(f"  {'='*60}")
    save_dir.mkdir(parents=True, exist_ok=True)

    status_path = save_dir / f"status_{vtag}_final.json"
    if _is_completed(save_dir, tag=f"status_{vtag}_final"):
        st = _read_status(status_path)
        print(f"  [{vtag}_final] Already completed "
              f"(best_ep={st.get('best_epoch')}). Skipping.")
        mf = save_dir / f"metrics_{vtag}_final.json"
        if mf.exists():
            with open(mf) as f: return json.load(f)
        return {}

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable={n_tr:,}")

    cw        = compute_class_weights(train_recs)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay)
    scheduler = _flat_cosine_scheduler(optimizer, n_epochs, warmup_frac)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_metric  = -1.0  # always maximize (C-index for surv, BACC for cls)
    best_epoch   = 0
    no_improve   = 0
    start_epoch  = 0
    ckpt_dir     = save_dir / f"ckpts_{vtag}_final"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_every   = 10   # save resume checkpoint every N epochs

    # ── Resume from last periodic checkpoint if available ─────────────────
    resume_ckpts = sorted(ckpt_dir.glob("ep_*.pt"),
                          key=lambda p: int(p.stem.split("_")[1]))
    if resume_ckpts:
        rc = resume_ckpts[-1]
        try:
            state = torch.load(rc, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            if scaler and state.get("scaler"):
                scaler.load_state_dict(state["scaler"])
            start_epoch  = state["epoch"]
            best_metric  = state.get("best_metric", best_metric)
            best_epoch   = state.get("best_epoch", 0)
            no_improve   = state.get("no_improve", 0)
            print(f"  [{vtag}_final] Resumed from ep {start_epoch} "
                  f"(best_ep={best_epoch}  best_metric={best_metric:.4f}  "
                  f"no_improve={no_improve})")
        except Exception as e:
            print(f"  [{vtag}_final] Could not resume from {rc.name}: {e}. Starting fresh.")
            start_epoch = 0

    train_mode = "alternating" if alternating else "simultaneous"
    print(f"  cw=(neg={cw[0]:.3f}, pos={cw[1]:.3f})  grad_accum={grad_accum}  mode={train_mode}")

    for epoch in range(start_epoch, n_epochs):
        if hasattr(model, 'routing_temperature'):
            model.routing_temperature = max(0.1, 1.0 - epoch / max(n_epochs - 1, 1) * 0.9)
        model.train()
        loss_d = p2_train_epoch(
            model, train_recs, optimizer, cw, device, bag_cache,
            scaler, grad_accum, mode=train_mode)
        scheduler.step()

        if (epoch + 1) % eval_every == 0 or epoch == n_epochs - 1:
            ci_vals: dict = {}
            if is_surv_only:
                # Survival-only task: evaluate the task-specific endpoint, use C-index
                _, _, _, ci_task, *_ = p2_evaluate(
                    model, val_recs, device, bag_cache,
                    surv_endpoint=_surv_ep, task=task)
                val_bacc = float(ci_task) if ci_task is not None else 0.0
                val_auc  = val_bacc
                ci_vals[_surv_ep] = val_bacc
                metric = val_bacc
            else:
                # Cls or mega: BACC (+ all C-indices for mega)
                vp, vl, val_loss, ci_acr, *_ = p2_evaluate(
                    model, val_recs, device, bag_cache,
                    surv_endpoint="acr", task=task)
                val_m    = compute_metrics(vl, vp) if len(np.unique(vl)) > 1 else {}
                val_bacc = val_m.get("bacc", 0.0)
                val_auc  = val_m.get("auc",  0.0)
                if is_mega:
                    for ep_name in ("acr", "clad", "death"):
                        _, _, _, ci_ep, *_ = p2_evaluate(
                            model, val_recs, device, bag_cache,
                            surv_endpoint=ep_name, task=task)
                        if ci_ep is not None:
                            ci_vals[ep_name] = ci_ep
                ci_list = list(ci_vals.values())
                metric  = (val_bacc + sum(ci_list)) / (1 + len(ci_list))

            improved = metric > best_metric

            # Test evaluation — logging only, never used for HP/early-stop decisions
            te_p, te_l, _, te_ci_acr, *_ = p2_evaluate(
                model, test_recs, device, bag_cache, surv_endpoint=_surv_ep, task=task)
            te_m_metrics = compute_metrics(te_l, te_p) if len(np.unique(te_l)) > 1 else {}
            te_bacc = te_m_metrics.get("bacc", float("nan"))
            te_auc  = te_m_metrics.get("auc",  float("nan"))
            te_ci: dict = {}
            if is_mega:
                for ep_name in ("acr", "clad", "death"):
                    _, _, _, ci_ep, *_ = p2_evaluate(
                        model, test_recs, device, bag_cache,
                        surv_endpoint=ep_name, task=task)
                    if ci_ep is not None:
                        te_ci[ep_name] = ci_ep
            if te_ci_acr is not None and not is_mega:
                te_ci[_surv_ep] = float(te_ci_acr) if te_ci_acr is not None else float("nan")

            bs    = loss_d.get("batch_stats", {})
            ci_str = "  ".join(f"ci_{k}={v:.4f}" for k, v in ci_vals.items())
            te_ci_str = "  ".join(f"te_ci_{k}={v:.4f}" for k, v in te_ci.items())
            active_tag = f"[{loss_d.get('mode','?')}]" if alternating else ""
            print(f"  [{vtag}_final] ep {epoch+1:3d}/{n_epochs}{active_tag}"
                  f"  bce={loss_d.get('bce',0):.4f}"
                  f"  cox_acr={loss_d.get('cox_acr',0):.4f}"
                  f"  cox_clad={loss_d.get('cox_clad',0):.4f}"
                  f"  cox_death={loss_d.get('cox_death',0):.4f}"
                  f"  lr={optimizer.param_groups[0]['lr']:.2e}"
                  f"\n    batch: cls={bs.get('n_cls',0)}(pos={bs.get('n_cls_pos',0)})"
                  f"  acr={bs.get('acr_surv_event',0)}ev/{bs.get('acr_surv_cens',0)}ce"
                  f"  clad={bs.get('clad_event',0)}ev/{bs.get('clad_cens',0)}ce"
                  f"  death={bs.get('death_event',0)}ev/{bs.get('death_cens',0)}ce"
                  + _slot_collapse_stats(model)
                  + f"\n    val_bacc={val_bacc:.4f}  val_auc={val_auc:.4f}"
                  + (f"  {ci_str}" if ci_str else "")
                  + f"  combined={metric:.4f}"
                  + ("  *best*" if improved else f"  (no_improve={no_improve+1}/{patience})")
                  + f"\n    [test] te_bacc={te_bacc:.4f}  te_auc={te_auc:.4f}"
                  + (f"  {te_ci_str}" if te_ci_str else ""))

            if improved:
                best_metric = metric
                best_epoch  = epoch + 1
                no_improve  = 0
                if not combined_train:
                    torch.save(model.state_dict(), ckpt_dir / "best_val.pt")
            else:
                no_improve += 1
                if not combined_train and no_improve >= patience:
                    print(f"  [{vtag}_final] Early stop at ep {epoch+1} "
                          f"(best_ep={best_epoch}  combined={best_metric:.4f})")
                    _gc()
                    break

        # Periodic checkpoint every ckpt_every epochs (keeps only the last one)
        if (epoch + 1) % ckpt_every == 0:
            ckpt_path = ckpt_dir / f"ep_{epoch+1:04d}.pt"
            torch.save({
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "scaler":      scaler.state_dict() if scaler else None,
                "epoch":       epoch + 1,
                "best_metric": best_metric,
                "best_epoch":  best_epoch,
                "no_improve":  no_improve,
            }, ckpt_path)
            # Delete previous periodic checkpoints to save disk
            for old in ckpt_dir.glob("ep_*.pt"):
                if old != ckpt_path:
                    old.unlink(missing_ok=True)
        _gc()

    # Load best-val checkpoint (skip in combined_train mode — use last epoch)
    best_ckpt = ckpt_dir / "best_val.pt"
    if combined_train:
        print(f"  [{vtag}_final] combined_train: using last epoch weights.")
    elif best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device,
                                         weights_only=True))
        print(f"  [{vtag}_final] Loaded best-val checkpoint (ep={best_epoch})")
    else:
        print(f"  [{vtag}_final] No best-val checkpoint found; using last epoch.")

    torch.save(model.state_dict(), save_dir / f"model_{vtag}_final.pt")
    _write_status(status_path, completed=True, best_epoch=best_epoch,
                  best_val_metric=best_metric, lr=lr, weight_decay=weight_decay)
    print(f"\n  [{vtag}_final] Training complete. Evaluating on test set...")

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
                       patience: int = 2,
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

    # Always maximize: C-index for survival tasks, BACC for cls tasks.
    _p2_best_metric: float = max(history["val_bacc"]) if history.get("val_bacc") else -1.0
    _p2_best_ep:   int   = 0
    _p2_no_improve: int  = 0

    for epoch in range(start_epoch, total_eps):
        if hasattr(model, 'routing_temperature'):
            model.routing_temperature = max(0.1, 1.0 - epoch / max(total_eps - 1, 1) * 0.9)
        loss_d = p2_train_epoch(
            model, train_recs, optimizer, cw, device, bag_cache,
            scaler, P2_GRAD_ACCUM,
            mode=_default_train_mode,
            task_weights=(DEFAULT_TASK_WEIGHTS if alternating else None),
        )
        tl = loss_d["bce"] + loss_d["cox"]
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
                # Checkpoint: higher C-index is better (directly optimizes the metric)
                improved = (cidx_val > _p2_best_metric)
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
                _p2_best_metric = cidx_val if _is_surv_only else primary_metric
                _p2_best_ep     = epoch + 1
                _p2_no_improve  = 0
                torch.save(model.state_dict(), save_dir / f"model_{vtag}.pt")
                ckpt_tag = "[ckpt*]"
            else:
                _p2_no_improve += 1
                ckpt_tag = "[ckpt]"
            # Test eval — logging only, not used for early stopping
            te_p, te_l, _, te_ci, *_ = p2_evaluate(
                model, test_recs, device, bag_cache,
                surv_endpoint=_surv_ep, task=task)
            model.train()
            if _is_surv_only:
                te_str = f"  [test] ci={float(te_ci):.3f}" if te_ci is not None else ""
            else:
                te_vm  = compute_metrics(te_l, te_p) if len(np.unique(te_l)) > 1 else {}
                te_ci_s = f"  ci={te_ci:.3f}" if te_ci is not None else ""
                te_str = f"  [test] bacc={te_vm.get('bacc', float('nan')):.3f}  auc={te_vm.get('auc', float('nan')):.3f}{te_ci_s}"

            improve_str = (f"  no_improve={_p2_no_improve}/{patience}"
                           if patience > 0 else "")
            print(f"  [{vtag}] ep {epoch+1:3d}  loss={tl:.4f}/{val_loss:.4f}  "
                  f"{metric_str}  {ckpt_tag}{improve_str}{te_str}")
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


# ── Longitudinal training/eval/hp-sweep/final ─────────────────────────────────

def p2_train_longitudinal_epoch(
    model, patient_records, optimizer, cw, device, bag_cache, scaler, grad_accum,
    cox_lambda_acr:   float = P2_COX_LAMBDA_ACR,
    cox_lambda_clad:  float = P2_COX_LAMBDA_CLAD,
    cox_lambda_death: float = P2_COX_LAMBDA_DEATH,
) -> dict:
    """Training epoch for LongitudinalMIL. One forward pass per patient."""
    model.train()
    random.shuffle(patient_records)
    use_amp  = (scaler is not None)
    OOM_SKIP = 3
    oom_counts: dict = {}

    total_bce = 0.0; n_bce = 0; n_steps = 0
    pending_loss: Optional[torch.Tensor] = None
    cox_bufs: Dict[str, list] = {"acr_surv": [], "clad": [], "death": []}
    accum_step = 0
    optimizer.zero_grad()

    surv_lam = {"acr_surv": cox_lambda_acr, "clad": cox_lambda_clad, "death": cox_lambda_death}

    def _flush():
        nonlocal pending_loss, n_steps
        combined = pending_loss
        for tk, buf in cox_bufs.items():
            if not buf:
                continue
            L_cox = cox_breslow_loss(buf)
            if L_cox is not None and L_cox.requires_grad:
                combined = L_cox * surv_lam[tk] if combined is None else combined + L_cox * surv_lam[tk]
            buf.clear()
        if combined is not None and combined.requires_grad:
            if scaler:
                scaler.scale(combined).backward(); scaler.step(optimizer); scaler.update()
            else:
                combined.backward(); optimizer.step()
        optimizer.zero_grad(); pending_loss = None; n_steps += 1; _gc()

    for pat in patient_records:
        pid = pat["patient_id"]
        if oom_counts.get(pid, 0) >= OOM_SKIP:
            continue

        stems   = pat["stems"]
        days    = pat["days"]
        records = pat["records"]
        bags_list = [{m: bag_cache.get(s, {}).get(m) for m in MODALITIES} for s in stems]

        if all(all(b.get(m) is None for m in MODALITIES) for b in bags_list):
            continue

        try:
            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model({"bags_list": bags_list, "days": days, "records": records}, device)

            if isinstance(result, torch.Tensor):
                continue

            # ACR surv (patient-level)
            acr_out = result.get("acr_surv")
            if acr_out is not None and isinstance(acr_out, tuple) and len(acr_out) == 4:
                hazard, _, acr_t, acr_e = acr_out
                if isinstance(hazard, torch.Tensor) and not math.isnan(acr_t):
                    cox_bufs["acr_surv"].append((hazard.float(), acr_t, acr_e))

            # CLAD + Death (per-biopsy gap-time)
            for tk in ("clad", "death"):
                biopsy_hazards = result.get(tk, [])
                if not isinstance(biopsy_hazards, list):
                    continue  # degenerate: forward returned 0-d tensor
                for hazard, t_val, e_val in biopsy_hazards:
                    if isinstance(hazard, torch.Tensor):
                        cox_bufs[tk].append((hazard.float(), t_val, e_val))

            # ACR cls (per labeled biopsy)
            cls_out = result.get("acr_cls", [])
            if not isinstance(cls_out, list):
                cls_out = []
            for logit, label in cls_out:
                if isinstance(logit, torch.Tensor):
                    target = logit.new_tensor([float(label)])
                    loss   = bce_loss(logit.unsqueeze(0), target, cw) * P2_BCE_LOSS_SCALE
                    pending_loss = loss if pending_loss is None else pending_loss + loss
                    total_bce += loss.item(); n_bce += 1

            accum_step += 1
            if accum_step >= grad_accum:
                _flush(); accum_step = 0

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            oom_counts[pid] = oom_counts.get(pid, 0) + 1
            pending_loss = None
            for buf in cox_bufs.values(): buf.clear()
            accum_step = 0; optimizer.zero_grad()
            if oom_counts[pid] >= OOM_SKIP:
                print(f"  [OOM-longitudinal] patient {pid} permanently skipped", flush=True)

    if accum_step > 0:
        _flush()

    return {"bce": total_bce / max(n_bce, 1), "n_steps": n_steps}


@torch.no_grad()
def p2_evaluate_longitudinal(model, patient_records, device, bag_cache, cw=None) -> dict:
    """Evaluate LongitudinalMIL. Returns dict with metrics for all 4 tasks + val_score."""
    model.eval()
    use_amp = (device.type == "cuda")

    cls_probs: List[float] = []; cls_labels: List[int] = []
    surv_data: Dict[str, dict] = {
        "acr_surv": {"h": [], "t": [], "e": []},
        "clad":     {"h": [], "t": [], "e": []},
        "death":    {"h": [], "t": [], "e": []},
    }

    for pat in patient_records:
        stems   = pat["stems"]; days = pat["days"]; records = pat["records"]
        bags_list = [{m: bag_cache.get(s, {}).get(m) for m in MODALITIES} for s in stems]
        if all(all(b.get(m) is None for m in MODALITIES) for b in bags_list):
            continue
        try:
            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model({"bags_list": bags_list, "days": days, "records": records}, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); continue

        if isinstance(result, torch.Tensor):
            continue

        acr_out = result.get("acr_surv")
        if acr_out is not None and isinstance(acr_out, tuple) and len(acr_out) == 4:
            hazard, _, acr_t, acr_e = acr_out
            if isinstance(hazard, torch.Tensor) and not math.isnan(acr_t):
                surv_data["acr_surv"]["h"].append(hazard.float().item())
                surv_data["acr_surv"]["t"].append(acr_t)
                surv_data["acr_surv"]["e"].append(acr_e)

        for tk in ("clad", "death"):
            biopsy_hazards = result.get(tk, [])
            if not isinstance(biopsy_hazards, list):
                continue  # degenerate: forward returned 0-d tensor
            for hazard, t_val, e_val in biopsy_hazards:
                if isinstance(hazard, torch.Tensor):
                    surv_data[tk]["h"].append(hazard.float().item())
                    surv_data[tk]["t"].append(t_val)
                    surv_data[tk]["e"].append(e_val)

        cls_out = result.get("acr_cls", [])
        if not isinstance(cls_out, list):
            cls_out = []
        for logit, label in cls_out:
            if isinstance(logit, torch.Tensor):
                cls_probs.append(torch.sigmoid(logit.float()).item())
                cls_labels.append(label)

    metrics: dict = {}

    if cls_probs and cls_labels:
        m = compute_metrics(np.array(cls_labels), np.array(cls_probs))
        metrics["acr_cls"] = m
    else:
        metrics["acr_cls"] = {"bacc": 0.5, "auc": 0.5}

    for tk, sd in surv_data.items():
        if len(sd["h"]) >= 2 and sum(sd["e"]) > 0:
            ci = c_index(sd["h"], sd["t"], sd["e"])
            metrics[tk] = {"c_index": ci if ci is not None else 0.5}
        else:
            metrics[tk] = {"c_index": 0.5}

    bacc     = metrics["acr_cls"].get("bacc", 0.5)
    ci_acr   = metrics["acr_surv"].get("c_index", 0.5)
    ci_clad  = metrics["clad"].get("c_index", 0.5)
    ci_death = metrics["death"].get("c_index", 0.5)
    metrics["val_score"] = 0.5 * bacc + 0.5 * float(np.mean([ci_acr, ci_clad, ci_death]))
    return metrics


def run_longitudinal_hp_sweep(
    model_factory,
    patient_train: List[dict],
    patient_val:   List[dict],
    device: torch.device,
    bag_cache: BagCache,
    save_dir: Path,
    lr_grid: List[float] = P2_HP_LR_GRID,
    wd_grid: List[float] = P2_HP_WD_GRID,
    sweep_epochs: int = P2_HP_SWEEP_EPOCHS,
    eval_every:   int = P2_HP_EVAL_EVERY,
    hp_patience:  int = P2_HP_PATIENCE,
) -> Tuple[float, float]:
    """HP sweep for LongitudinalMIL — same grid as mario_kempes."""
    from itertools import product as iproduct
    save_dir.mkdir(parents=True, exist_ok=True)
    result_path = save_dir / "hp_sweep_p2.json"
    if result_path.exists():
        with open(result_path) as f:
            res = json.load(f)
        print(f"  [LMK-HP] Already done: lr={res['best_lr']}  wd={res['best_wd']}  "
              f"val_score={res['best_val_bacc']:.4f}")
        return res["best_lr"], res["best_wd"]

    # Derive class weights from flattened biopsy records
    flat_train = [r for pat in patient_train for r in pat["records"]]
    cw = compute_class_weights(flat_train)

    best_metric = -1.0
    best_lr, best_wd = lr_grid[0], wd_grid[0]
    results = []

    for lr, wd in iproduct(lr_grid, wd_grid):
        print(f"  [LMK-HP] lr={lr:.0e}  wd={wd:.0e}", flush=True)
        model = model_factory().to(device)
        opt   = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=lr, weight_decay=wd)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
        best_ep_metric = -1.0; no_improve = 0; stopped_ep = sweep_epochs

        for ep in range(sweep_epochs):
            p2_train_longitudinal_epoch(model, patient_train, opt, cw, device,
                                        bag_cache, scaler, P2_GRAD_ACCUM)
            if (ep + 1) % eval_every == 0:
                vm = p2_evaluate_longitudinal(model, patient_val, device, bag_cache, cw)
                metric = vm["val_score"]
                improved = metric > best_ep_metric
                best_ep_metric = max(best_ep_metric, metric)
                no_improve = 0 if improved else no_improve + 1
                print(f"  ep {ep+1:3d}/{sweep_epochs}  val_score={metric:.4f}  "
                      f"best={best_ep_metric:.4f}", flush=True)
                _gc()
                if no_improve >= hp_patience:
                    stopped_ep = ep + 1
                    print(f"  [LMK-HP] early stop", flush=True); break

        print(f"  [LMK-HP] lr={lr:.0e}  wd={wd:.0e}  DONE  val_score={best_ep_metric:.4f}")
        results.append({"lr": lr, "wd": wd, "val_bacc": best_ep_metric,
                        "stopped_ep": stopped_ep})
        if best_ep_metric > best_metric:
            best_metric, best_lr, best_wd = best_ep_metric, lr, wd
        del model, opt, scaler; _gc()

    res = {"best_lr": best_lr, "best_wd": best_wd, "best_val_bacc": best_metric,
           "grid": results}
    with open(result_path, "w") as f:
        json.dump(res, f, indent=2)
    return best_lr, best_wd


def run_longitudinal_final(
    model: nn.Module,
    variant: str,
    fold: int,
    device: torch.device,
    bag_cache: BagCache,
    patient_train: List[dict],
    patient_val:   List[dict],
    patient_test:  List[dict],
    save_dir: Path,
    lr: float = P2_LR,
    weight_decay: float = P2_WEIGHT_DECAY,
    n_epochs: int = P2_FINAL_EPOCHS,
    eval_every: int = P2_EVAL_EVERY,
    patience: int = 20,
    grad_accum: int = 32,
    combined_train: bool = False,
) -> dict:
    """Final training + test eval for LongitudinalMIL."""
    vtag = variant
    save_dir.mkdir(parents=True, exist_ok=True)
    status_path = save_dir / f"status_{vtag}_final.json"

    if _is_completed(save_dir, tag=f"status_{vtag}_final"):
        st = _read_status(status_path)
        print(f"  [LMK_final] Already completed (best_ep={st.get('best_epoch')}). Skipping.")
        mf = save_dir / f"metrics_{vtag}_final.json"
        if mf.exists():
            with open(mf) as f: return json.load(f)
        return {}

    flat_train = [r for pat in patient_train for r in pat["records"]]
    cw = compute_class_weights(flat_train)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=lr, weight_decay=weight_decay)
    scheduler = _flat_cosine_scheduler(optimizer, n_epochs)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_metric = -1.0; best_epoch = 0; no_improve = 0
    ckpt_dir = save_dir / f"ckpts_{vtag}_final"; ckpt_dir.mkdir(exist_ok=True)

    train_pats = patient_train + patient_val if combined_train else patient_train
    if combined_train:
        print(f"  [LMK_final] combined_train: {len(train_pats)} patients")

    for epoch in range(n_epochs):
        loss_d = p2_train_longitudinal_epoch(model, train_pats, optimizer, cw,
                                             device, bag_cache, scaler, grad_accum)
        scheduler.step()

        if (epoch + 1) % eval_every == 0 or epoch == n_epochs - 1:
            vm = p2_evaluate_longitudinal(model, patient_val, device, bag_cache, cw)
            metric = vm["val_score"]
            improved = metric > best_metric
            print(f"  [LMK_final] ep {epoch+1:3d}/{n_epochs}  bce={loss_d['bce']:.4f}"
                  f"  val_score={metric:.4f}  best={best_metric:.4f}"
                  + ("  *best*" if improved else f"  (no_improve={no_improve+1}/{patience})"),
                  flush=True)
            if improved:
                best_metric = metric; best_epoch = epoch + 1; no_improve = 0
                if not combined_train:
                    torch.save(model.state_dict(), ckpt_dir / "best_val.pt")
            else:
                no_improve += 1
                if not combined_train and no_improve >= patience:
                    print(f"  [LMK_final] Early stop at ep {epoch+1}"); _gc(); break

        if (epoch + 1) % 10 == 0:
            ckpt_path = ckpt_dir / f"ep_{epoch+1:04d}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch+1,
                        "best_metric": best_metric, "best_epoch": best_epoch,
                        "no_improve": no_improve}, ckpt_path)
            for old in ckpt_dir.glob("ep_*.pt"):
                if old != ckpt_path: old.unlink(missing_ok=True)
        _gc()

    best_ckpt = ckpt_dir / "best_val.pt"
    if not combined_train and best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
        print(f"  [LMK_final] Loaded best-val checkpoint (ep={best_epoch})")

    torch.save(model.state_dict(), save_dir / f"model_{vtag}_final.pt")
    _write_status(status_path, completed=True, best_epoch=best_epoch,
                  best_val_metric=best_metric, lr=lr, weight_decay=weight_decay)

    print(f"\n  [LMK_final] Evaluating on test set ({len(patient_test)} patients)...")
    tm = p2_evaluate_longitudinal(model, patient_test, device, bag_cache)
    all_metrics: dict = {"test": tm}
    print(f"  [LMK_final] test  val_score={tm['val_score']:.4f}"
          f"  bacc={tm['acr_cls'].get('bacc', 0):.4f}"
          f"  ci_acr={tm['acr_surv'].get('c_index', 0):.4f}"
          f"  ci_clad={tm['clad'].get('c_index', 0):.4f}"
          f"  ci_death={tm['death'].get('c_index', 0):.4f}")

    with open(save_dir / f"metrics_{vtag}_final.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    del model, optimizer, scaler; _gc()
    return all_metrics
