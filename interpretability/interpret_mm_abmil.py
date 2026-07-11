"""
interpret_mm_abmil.py  ·  Interpretability suite for Multimodal ABMIL (v6 all variants + v7)
==============================================================================================

Supports every v6 Phase-2 variant and v7 triple-stream, dispatching
to the appropriate extractor automatically based on model class type.

Extraction schema (all models produce what they can)
-----------------------------------------------------
  alpha_{mod}        (N_m,)        ABMIL attention over instances / effective patch importance
  h_{mod}            (N_m, H)      backbone embeddings
  grad_imp_{mod}     (N_m,)        |∂logit/∂h_i| × α_i  [GradCAM-style]
  modal_contrib      {mod: float}  ||r_bag_m|| fraction

  xmodal_attn        (M, M)        cross-modal transformer attention (Middle / CrossAttn / Slot / Iterative / v7)
  mod_order          list[str]

  slot_assign_{mod}  (K, N_m)      slot competition weights A[k,i] — which patches belong to which slot
                                   (CrossAttn / Slot / Iterative)
  slot_importance_{mod} (N_m,)     patch importance derived from slots: Σ_k α_slot[k] × A[k,i]

  bidir_attn_{ma}_{mb} (N_a,)      mean attention from mb→ma patches  (CrossAttn only)

  self_attn_{mod}_r{r} (N,)        per-patch self-attn energy per iter block  (Iterative only)
  cross_attn_{mod}_r{r} (N,)       per-patch cross-attn energy per iter block (Iterative only)

  modal_logits       {mod: float}  per-modality logit                  (Late only)
  modal_weights      {mod: float}  learned modality weight              (Late only)

  cent_gate_{mod}    (K_m,)        centroid count-gate strength         (v7 only)
  cent_xattn         (Ktot, Ktot)  centroid cross-modal attention       (v7 only)

Usage
-----
  python3 interpret_mm_abmil.py \\
      --version v6 \\
      --results_dir results_mm_abmil_v6 \\
      --split 0 --fold 0 \\
      --v6_variant iterative_r4_k16_cls \\
      --out_dir interpretability/v6_s0f0/iterative_r4_k16_cls

  python3 interpret_mm_abmil.py \\
      --version v7 \\
      --results_dir results_mm_abmil_v7 \\
      --phase1_dir  results_mm_abmil_v6 \\
      --split 0 --fold 0 --tag v7_triple \\
      --out_dir interpretability/v7_s0f0
"""

from __future__ import annotations
import argparse, gc, json, sys, warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from train_mm_abmil_v6 import (
    MODALITIES, MODALITY_REGISTRY, HIDDEN_DIM,
    SAMPLES_DIR, SPLITS_CSV,
    set_seeds, _gc as _gc6, BagCache, preload_bags, build_splits,
    build_splits_survival,
    update_presence_from_cache, acr_label,
    _load_p1_encoder, CrossModalTransformer,
    EarlyFusionMIL, LateFusionMIL, MiddleFusionMIL,
    CrossAttnFusionMIL, SlotCrossModalMIL, IterativeXModalMIL,
    BidirPatchCrossAttn, IterativeSlotAttn,
    build_p2_model, _pool,
    DEVICE, SEED, FOLDS as _FOLDS,
    P2_N_HEADS, P2_N_CROSS_LAYERS, P2_ATTN_DROPOUT,
)
try:
    from train_mm_abmil_v7 import (
        ANNOT_REGISTRY, ANNOT_MODS, AnnotCache,
        build_v7_model, preload_annotations,
    )
    HAS_V7 = True
except ModuleNotFoundError:
    HAS_V7 = False
    ANNOT_MODS = []
    ANNOT_REGISTRY = {}
    AnnotCache = build_v7_model = preload_annotations = None

try:
    from umap import UMAP as UMAPTransform
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[warn] umap-learn not installed — UMAP plot skipped")

LABEL_NAMES = {0: "No rejection", 1: "Rejection"}
MOD_COLORS  = {"HE": "#4e79a7", "BAL": "#f28e2b", "CT": "#59a14f", "Clinical": "#e15759"}
SLOT_CMAP   = "plasma"
MOD_TO_CLUSTER_KEY = {"HE": "HE_cells", "BAL": "BAL_cells", "CT": "CT_cells"}
# Clinical is also a bag of F feature tokens — feature names serve as cluster labels
CLUSTER_MODS = list(MOD_TO_CLUSTER_KEY.keys()) + ["Clinical"]


# ══════════════════════════════════════════════════════════════════
# ATTENTION CAPTURE UTILITIES
# ══════════════════════════════════════════════════════════════════

class _AttnCapture:
    def __init__(self):
        self.weights: List[torch.Tensor] = []
        self._handles = []

    def hook(self, module: nn.Module):
        h = module.register_forward_hook(self._cb)
        self._handles.append(h)

    def _cb(self, module, inp, output):
        if isinstance(output, (tuple, list)) and len(output) >= 2 and output[1] is not None:
            self.weights.append(output[1].detach().cpu())

    def last(self) -> Optional[torch.Tensor]:
        return self.weights[-1] if self.weights else None

    def remove(self):
        for h in self._handles: h.remove()
        self._handles.clear()


@contextmanager
def capture_mha(modules: List[nn.Module]):
    cap = _AttnCapture()
    for m in modules:
        for sub in m.modules():
            if isinstance(sub, nn.MultiheadAttention):
                cap.hook(sub)
    try:
        yield cap
    finally:
        cap.remove()


# ══════════════════════════════════════════════════════════════════
# SHARED LOW-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════════

def _abmil_forward(enc, h: torch.Tensor):
    """
    Run gated-attention ABMIL on pre-computed h. Returns (rep, alpha).
    Works on both GatedAttentionEncoder (has att_drop) and model-level
    ABMIL heads (EarlyFusion, CrossAttn, Slot, Iterative — no att_drop).
    In eval mode att_drop is a no-op anyway, so skipping it is correct.
    """
    gate  = enc.att_V(h) * enc.att_U(h)
    drop  = getattr(enc, "att_drop", None)
    gated = drop(gate) if drop is not None else gate
    raw   = enc.att_w(gated)
    alpha = F.softmax(raw, dim=0)       # (N, 1)
    rep   = (alpha * h).sum(0)          # (H,)
    return rep, alpha.squeeze(1)        # (H,), (N,)


def _slot_forward_capture(module: IterativeSlotAttn,
                          h: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate IterativeSlotAttn.forward in eval mode, capturing
    the final-round slot assignment matrix A (K, N) and raw logits (K, N).
    Returns (slots, attn_KN, logits_KN) — attn_KN is post-softmax.
    """
    slots = module.slot_mu.squeeze(0).detach().clone()  # (K, H) deterministic eval
    h_norm = module.norm_input(h)
    k = module.proj_k(h_norm)
    v = module.proj_v(h_norm)
    last_attn = None
    last_logits = None
    for _ in range(module.n_iters):
        slots_prev = slots
        q      = module.proj_q(module.norm_slots(slots))
        logits = torch.matmul(q, k.T) * module.scale      # (K, N)
        attn   = F.softmax(logits, dim=0)                  # (K, N)
        last_attn   = attn.detach().cpu()
        last_logits = logits.detach().cpu()
        attn_norm = attn / (attn.sum(dim=1, keepdim=True) + 1e-6)
        updates = attn_norm @ v                             # (K, H)
        slots = module.gru(
            updates.view(-1, updates.shape[-1]),
            slots_prev.view(-1, slots_prev.shape[-1]),
        ).view(slots.shape)
        slots = module.norm_mlp(slots + module.mlp(slots))
    return slots, last_attn, last_logits   # (K,H) differentiable; (K,N) post; (K,N) pre


def _bidir_capture(module: BidirPatchCrossAttn,
                   h_a: torch.Tensor, h_b: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run BidirPatchCrossAttn capturing per-direction attn weights.
    Returns h_a_enr, h_b_enr, attn_a2b (N_a, N_b), attn_b2a (N_b, N_a).
    """
    Qa = module.Wq_a(h_a); Kb = module.Wk_b(h_b); Vb = module.Wv_b(h_b)
    out_a2b, attn_a2b = module._attn(Qa, Kb, Vb)    # (N_a, H), (N_a, N_b)
    out_a2b = module.Wo_a(out_a2b)
    h_a_res = h_a + out_a2b
    h_a_enr = module.norm_a2(h_a_res + module.ffn_a(module.norm_a(h_a_res)))

    Qb = module.Wq_b(h_b); Ka = module.Wk_a(h_a); Va = module.Wv_a(h_a)
    out_b2a, attn_b2a = module._attn(Qb, Ka, Va)    # (N_b, H), (N_b, N_a)
    out_b2a = module.Wo_b(out_b2a)
    h_b_res = h_b + out_b2a
    h_b_enr = module.norm_b2(h_b_res + module.ffn_b(module.norm_b(h_b_res)))

    return h_a_enr, h_b_enr, attn_a2b.detach().cpu(), attn_b2a.detach().cpu()


def _pool_capture(model, tokens: torch.Tensor, device):
    """
    Pool a token sequence (T, H) using the model's pooling mechanism.
    Returns (rep (H,), alpha (T,) or None).
    """
    if hasattr(model, "use_cls") and model.use_cls:
        cls = model.cls_token.to(device)
        seq = torch.cat([cls, tokens], dim=0).unsqueeze(0)  # (1, T+1, H)
        with capture_mha([model.cls_attn]) as cap:
            out, _ = model.cls_attn(seq, seq, seq)
        out   = model.cls_norm(seq + out)
        rep   = out.squeeze(0)[0]        # CLS position
        # CLS attention to other tokens (excluding self)
        attn_w = cap.last()              # (1, T+1, T+1) or None
        alpha  = attn_w[0, 0, 1:].numpy() if attn_w is not None else None
    else:
        gate  = model.att_V(tokens) * model.att_U(tokens)
        raw   = model.att_w(gate)
        alpha = F.softmax(raw, dim=0).squeeze(1).detach().cpu().numpy()  # (T,)
        rep   = (F.softmax(raw, dim=0) * tokens).sum(0)
    return rep, alpha


def _grad_attribution(h_store: Dict[str, torch.Tensor], logit: torch.Tensor,
                       alpha_store: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute gradient attribution for each modality after logit.backward()."""
    result = {}
    logit.backward()
    for mod, h in h_store.items():
        if h.grad is None: continue
        alpha = torch.from_numpy(alpha_store[mod])       # (N,)
        grad_norm = h.grad.norm(dim=-1).cpu()            # (N,)
        result[mod] = (alpha * grad_norm).numpy()
    return result


def _risk_attribution(model, h_store: Dict[str, torch.Tensor], rep: torch.Tensor,
                      alpha_store: Dict[str, np.ndarray]) -> Tuple[Optional[float], Dict[str, np.ndarray]]:
    """Compute hazard score and gradient attribution w.r.t. hazard head.
    Returns (hazard_score, {mod: risk_attr_array}).
    Only runs if model has a hazard_head attribute.
    """
    if not hasattr(model, 'hazard_head'):
        return None, {}
    # Zero any existing grads
    for h in h_store.values():
        if h.grad is not None:
            h.grad.zero_()
    hazard = model.hazard_head(rep.detach().requires_grad_(False))
    # We need grad w.r.t. h_store — recompute rep from h_store is complex;
    # instead backprop hazard scalar w.r.t. h_store tensors that still have grad_fn
    # Since rep may not have grad_fn after detach, we use a simple approach:
    # just apply hazard_head to rep that has grad (if available)
    hazard_score = hazard.squeeze().item()
    risk_attrs = {}
    # Attempt gradient-based risk attribution through h tensors that require grad
    try:
        hazard_grad = model.hazard_head(rep)
        if hazard_grad.requires_grad:
            hazard_grad.squeeze().backward(retain_graph=True)
            for mod, h in h_store.items():
                if h.grad is not None:
                    alpha = alpha_store.get(mod)
                    gn = h.grad.norm(dim=-1).cpu().numpy()
                    if alpha is not None:
                        risk_attrs[f"risk_attr_{mod}"] = alpha * gn
                    else:
                        risk_attrs[f"risk_attr_{mod}"] = gn
    except Exception:
        pass
    # Zero grads after attribution
    for h in h_store.values():
        if h.grad is not None:
            h.grad.zero_()
    return hazard_score, risk_attrs


def _xfmr_last_attn(xfmr: CrossModalTransformer,
                    x: torch.Tensor) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    """Run CrossModalTransformer, capturing last layer's attention. Returns (output, attn_NxN)."""
    mha_list = [L["attn"] for L in xfmr.layers]
    with capture_mha(mha_list) as cap:
        out = xfmr(x)
    attn = cap.last()
    attn_np = attn.squeeze(0).numpy() if attn is not None else None
    return out, attn_np


def _backbone_h(enc, t: torch.Tensor, requires_grad: bool = True) -> torch.Tensor:
    """Compute backbone features with optional grad tracking."""
    with torch.no_grad():
        h = enc.encode_patches(t)
    if requires_grad:
        h = h.detach().requires_grad_(True)
    return h


# ══════════════════════════════════════════════════════════════════
# MODEL-SPECIFIC EXTRACTORS
# ══════════════════════════════════════════════════════════════════

@torch.enable_grad()
def _extract_early(model: EarlyFusionMIL, bags, device, label) -> Optional[dict]:
    """
    EarlyFusion: all patches concatenated → shared ABMIL.
    alpha is over ALL patches jointly; we split back by modality.
    """
    result: dict = {"label": label}
    h_parts, mod_labels, h_store = [], [], {}

    for mod, enc in model.encoders.items():
        t = bags.get(mod)
        if t is None: continue
        h = _backbone_h(enc, t.to(device))
        h_parts.append(h); h_store[mod] = h
        mod_labels.extend([mod] * h.shape[0])

    if not h_parts: return None
    H_all = torch.cat(h_parts, dim=0)   # (N_all, H)
    result["mods_present"]   = list(h_store.keys())
    result["patch_mod_labels"] = mod_labels  # which modality each patch came from

    if model.use_cls:
        rep   = _pool(True, H_all, model.cls_token, model.cls_attn,
                      model.cls_norm, None, None, None, device)
        alpha = torch.ones(H_all.shape[0], device=device) / H_all.shape[0]
    else:
        rep, alpha = _abmil_forward(model, H_all)  # (H,), (N_all,)
    result["alpha_joint"] = alpha.detach().cpu().numpy()

    # Split alpha back by modality
    offset = 0
    alpha_store_mod = {}
    for mod, h in h_store.items():
        n = h.shape[0]
        a = alpha[offset:offset + n].detach().cpu()
        result[f"alpha_{mod}"] = a.numpy()
        result[f"h_{mod}"]     = h.detach().cpu().numpy()
        alpha_store_mod[mod]   = a.numpy()
        offset += n

    logit = model.head(rep).squeeze()
    result["logit"] = logit.item()
    result["prob"]  = torch.sigmoid(logit).item()

    # Gradient attribution on H_all, split back (skip if no grad_fn, e.g. CLS path)
    if logit.requires_grad:
        logit.backward(retain_graph=hasattr(model, 'hazard_head'))
        offset = 0
        alpha_np = result["alpha_joint"]
        for mod, h in h_store.items():
            n = h.shape[0]
            if h.grad is not None:
                gn = h.grad.norm(dim=-1).cpu().numpy()
                result[f"grad_imp_{mod}"] = (alpha_np[offset:offset + n] * gn)
            offset += n

    # Risk attribution via hazard head
    if hasattr(model, 'hazard_head'):
        hazard_score, risk_attrs = _risk_attribution(model, h_store, rep, alpha_store_mod)
        if hazard_score is not None:
            result["hazard_score"] = hazard_score
        result.update(risk_attrs)

    norms = {m: h_store[m].norm(dim=-1).mean().item() for m in h_store}
    total = sum(norms.values()) + 1e-8
    result["modal_contrib"] = {m: norms[m] / total for m in norms}
    result["xmodal_attn"]   = None   # not applicable for early fusion
    result["mod_order"]     = list(h_store.keys())
    return result


@torch.enable_grad()
def _extract_late(model: LateFusionMIL, bags, device, label) -> Optional[dict]:
    """
    LateFusion: per-modality ABMIL → per-modality logit → learned weighted sum.
    """
    result: dict = {"label": label}
    h_store, alpha_store, rep_store, logit_store = {}, {}, {}, {}

    for mod, enc in model.encoders.items():
        t = bags.get(mod)
        if t is None: continue
        h = _backbone_h(enc, t.to(device))
        rep, alpha = _abmil_forward(enc, h)
        h_store[mod]     = h
        alpha_store[mod] = alpha.detach().cpu().numpy()
        rep_store[mod]   = rep
        result[f"alpha_{mod}"] = alpha_store[mod]
        result[f"h_{mod}"]     = h.detach().cpu().numpy()

    if not h_store: return None
    result["mods_present"] = list(h_store.keys())

    # Per-modality logits
    indices = [model.mod_index[m] for m in h_store]
    modal_logits  = {m: model.heads[m](rep_store[m]).squeeze() for m in h_store}
    logit_vals    = torch.stack(list(modal_logits.values()))
    weights_raw   = F.softmax(model.log_weights[
        torch.tensor(indices, device=device)], dim=0)

    logit = (weights_raw * logit_vals).sum()
    result["logit"]        = logit.item()
    result["prob"]         = torch.sigmoid(logit).item()
    result["modal_logits"] = {m: modal_logits[m].item() for m in modal_logits}
    result["modal_weights"]= {m: weights_raw[i].item()
                               for i, m in enumerate(h_store)}
    result["modal_contrib"]= result["modal_weights"]  # for consistency
    result["xmodal_attn"]  = None
    result["mod_order"]    = list(h_store.keys())

    # Hazard score (mean over per-modality hazard estimates)
    if hasattr(model, 'hazard_head'):
        try:
            h_vals = [model.hazard_head(rep_store[m]).squeeze() for m in rep_store]
            hazard_score = torch.stack(h_vals).mean().item()
            result["hazard_score"] = hazard_score
        except Exception:
            pass

    result.update(_grad_attribution(h_store, logit, alpha_store))
    return result


@torch.enable_grad()
def _extract_middle(model: MiddleFusionMIL, bags, device, label) -> Optional[dict]:
    """
    MiddleFusion: per-modality ABMIL → cross-modal transformer → pool → head.
    """
    result: dict = {"label": label}
    r_bag, h_store, alpha_store = {}, {}, {}

    for mod, enc in model.encoders.items():
        t = bags.get(mod)
        if t is None: continue
        h = _backbone_h(enc, t.to(device))
        rep, alpha = _abmil_forward(enc, h)
        h_store[mod]           = h
        alpha_store[mod]       = alpha.detach().cpu().numpy()
        result[f"alpha_{mod}"] = alpha_store[mod]
        result[f"h_{mod}"]     = h.detach().cpu().numpy()
        r_bag[mod]             = rep

    if not r_bag: return None
    result["mods_present"] = list(r_bag.keys())
    mod_order = list(r_bag.keys())

    tokens = list(r_bag.values())
    if hasattr(model, "cls_token") and model.cls_token is not None:
        tokens = [model.cls_token.squeeze(0).to(device)] + tokens
    x = torch.stack(tokens, dim=0).unsqueeze(0)   # (1, M[+cls], H)

    xmodal_attn = None
    if len(r_bag) >= 2:
        mha_list = [L["attn"] for L in model.transformer]
        with capture_mha(mha_list) as cap:
            for L in model.transformer:
                a, _ = L["attn"](x, x, x)
                x    = L["ffn"](L["norm"](x + a))
        aw = cap.last()
        if aw is not None:
            offset = 1 if (hasattr(model, "cls_token") and model.cls_token is not None) else 0
            xmodal_attn = aw.squeeze(0)[offset:, offset:].detach().numpy()
    else:
        for L in model.transformer:
            a, _ = L["attn"](x, x, x)
            x    = L["ffn"](L["norm"](x + a))

    result["xmodal_attn"] = xmodal_attn
    result["mod_order"]   = mod_order

    is_cls = hasattr(model, "cls_token") and model.cls_token is not None
    r_final = x.squeeze(0)[0] if is_cls else x.squeeze(0).mean(0)
    logit   = model.head(r_final).squeeze()
    result["logit"] = logit.item()
    result["prob"]  = torch.sigmoid(logit).item()

    # Hazard score
    if hasattr(model, 'hazard_head'):
        try:
            result["hazard_score"] = model.hazard_head(r_final.detach()).squeeze().item()
        except Exception:
            pass

    norms = {m: r_bag[m].norm().item() for m in r_bag}
    result["modal_contrib"] = {m: norms[m] / (sum(norms.values()) + 1e-8) for m in norms}

    logit.backward(retain_graph=hasattr(model, 'hazard_head'))
    for mod, h in h_store.items():
        if h.grad is not None:
            gn = h.grad.norm(dim=-1).cpu().numpy()
            result[f"grad_imp_{mod}"] = alpha_store[mod] * gn

    # Risk attribution
    if hasattr(model, 'hazard_head'):
        _, risk_attrs = _risk_attribution(model, h_store, r_final.detach().requires_grad_(False),
                                          alpha_store)
        result.update(risk_attrs)
    return result


@torch.enable_grad()
def _extract_slot_base(model, bags, device, label,
                       has_bidir: bool = False) -> Optional[dict]:
    """
    Shared extractor for SlotCrossModalMIL and CrossAttnFusionMIL.
    has_bidir=True adds bidir patch cross-attention extraction.
    """
    result: dict = {"label": label}
    h_store: Dict[str, torch.Tensor] = {}

    # Stage 1: backbone features
    for mod, enc in model.encoders.items():
        t = bags.get(mod)
        if t is None: continue
        h = _backbone_h(enc, t.to(device))
        h_store[mod] = h
        result[f"h_{mod}"] = h.detach().cpu().numpy()

    if not h_store: return None
    result["mods_present"] = list(h_store.keys())
    mod_order = list(h_store.keys())

    # Stage 2 (CrossAttn only): bidir patch cross-attention
    enriched_h = dict(h_store)
    if has_bidir and len(h_store) >= 2:
        mods = list(h_store.keys())
        for i, m_a in enumerate(mods):
            for m_b in mods[i + 1:]:
                h_a = enriched_h[m_a]
                h_b_cap = enriched_h[m_b]
                if h_b_cap.shape[0] > model.max_patches_bidir:
                    h_b_cap = h_b_cap[:model.max_patches_bidir]
                h_a_enr, h_b_enr, attn_a2b, attn_b2a = _bidir_capture(
                    model.bidir_cross, h_a, h_b_cap)
                enriched_h[m_a] = h_a_enr
                if h_b_cap.shape[0] == enriched_h[m_b].shape[0]:
                    enriched_h[m_b] = h_b_enr
                # alpha_a2b: mean attention from mb patches → ma patches  (N_a,)
                result[f"bidir_attn_{m_a}_{m_b}"] = attn_b2a.mean(0).numpy()  # (N_a,)
                result[f"bidir_attn_{m_b}_{m_a}"] = attn_a2b.mean(0).numpy()  # (N_b,)

    # Stage 3: slot attention per modality (capture assignment + run model)
    slot_dict, slot_assigns, h_after = {}, {}, {}
    for mod, h in enriched_h.items():
        slots, attn_KN, logits_KN = _slot_forward_capture(model.slot_attn, h)
        slot_dict[mod]   = slots          # (K, H)
        slot_assigns[mod] = attn_KN.numpy() if attn_KN is not None else None  # (K, N)
        h_after[mod]     = h
        result[f"slot_assign_{mod}"] = slot_assigns[mod]
        result[f"slot_logits_{mod}"] = logits_KN.numpy() if logits_KN is not None else None

    # Also compute alpha from ABMIL per modality (on enriched features before slots)
    for mod, h in enriched_h.items():
        _, alpha = _abmil_forward(model, h) if (not model.use_cls) else (None, None)
        if alpha is not None:
            result[f"alpha_{mod}"] = alpha.detach().cpu().numpy()
        else:
            # No modality-level ABMIL — use slot assignment as proxy
            if slot_assigns[mod] is not None:
                result[f"alpha_{mod}"] = slot_assigns[mod].sum(0)  # (N,) sum over slots

    # Stage 4: cross-modal transformer over concatenated slots
    slices, pos = [], 0
    for mod in mod_order:
        k = slot_dict[mod].shape[0]
        slices.append((mod, pos, pos + k)); pos += k

    all_slots = torch.cat([slot_dict[m] for m in mod_order], dim=0).unsqueeze(0)
    xmodal_attn = None
    if len(mod_order) >= 2:
        all_slots, attn_np = _xfmr_last_attn(model.cross_xfmr, all_slots)
        xmodal_attn = attn_np   # (K*M, K*M)
    all_slots = all_slots.squeeze(0)   # (K*M, H)

    result["xmodal_attn"]  = xmodal_attn   # full slot-level attention matrix
    result["mod_order"]    = mod_order
    result["slot_slices"]  = [(m, s, e) for m, s, e in slices]

    # Stage 5: pool over all slots
    rep, alpha_slots = _pool_capture(model, all_slots, device)

    # Slot-level importance → per-modality patch importance
    if alpha_slots is not None:
        for mod, s, e in slices:
            a_slot = alpha_slots[s:e]          # (K,) importance of each slot
            A      = slot_assigns[mod]         # (K, N)
            if A is not None:
                patch_imp = (a_slot[:, None] * A).sum(0)   # (N,)
                result[f"slot_importance_{mod}"] = patch_imp
                # Overwrite alpha with slot-derived importance for downstream viz
                result[f"alpha_{mod}"] = patch_imp

    logit = model.head(rep).squeeze()
    result["logit"] = logit.item()
    result["prob"]  = torch.sigmoid(logit).item()

    # Hazard score
    if hasattr(model, 'hazard_head'):
        try:
            result["hazard_score"] = model.hazard_head(rep.detach()).squeeze().item()
        except Exception:
            pass

    norms = {m: slot_dict[m].norm(dim=-1).mean().item() for m in mod_order}
    result["modal_contrib"] = {m: norms[m] / (sum(norms.values()) + 1e-8) for m in norms}

    # Gradient attribution on original backbone features
    alpha_store_slot = {mod: result.get(f"alpha_{mod}", np.ones(h_store[mod].shape[0]) / h_store[mod].shape[0])
                        for mod in h_store}
    logit.backward(retain_graph=hasattr(model, 'hazard_head'))
    for mod in h_store:
        h = h_store[mod]
        if h.grad is not None:
            alpha = result.get(f"alpha_{mod}", np.ones(h.shape[0]) / h.shape[0])
            gn = h.grad.norm(dim=-1).cpu().numpy()
            result[f"grad_imp_{mod}"] = alpha * gn

    # Risk attribution
    if hasattr(model, 'hazard_head'):
        _, risk_attrs = _risk_attribution(model, h_store, rep.detach().requires_grad_(False),
                                          alpha_store_slot)
        result.update(risk_attrs)
    return result


@torch.enable_grad()
def _extract_iterative(model: IterativeXModalMIL, bags, device, label) -> Optional[dict]:
    """
    IterativeXModalMIL: R blocks of (self-attn + cross-attn at patch level)
    → slot attention → cross-modal transformer → pool → head.
    Captures per-block self- and cross-attn energy per patch.
    """
    result: dict = {"label": label}
    h_store: Dict[str, torch.Tensor] = {}

    for mod, enc in model.encoders.items():
        t = bags.get(mod)
        if t is None: continue
        t = t.to(device)
        if mod == "HE" and t.shape[0] > model.max_he_patches:
            t = t[:model.max_he_patches]
        h = _backbone_h(enc, t)
        h_store[mod] = h
        result[f"h_{mod}"] = h.detach().cpu().numpy()

    if not h_store: return None
    result["mods_present"] = list(h_store.keys())
    mod_order = list(h_store.keys())

    # Iterative self-attn + cross-attn blocks (capture energy)
    h_dict = {m: h_store[m] for m in mod_order}
    for r in range(model.n_iter_blocks):
        SA = model.self_attn_blocks[r]
        CA = model.cross_attn_blocks[r]

        # Within-modal self-attn — capture per-patch energy (row sum of attn)
        h_self = {}
        for mod, h in h_dict.items():
            x = h.unsqueeze(0)
            with capture_mha([SA["attn"]]) as cap:
                a, _ = SA["attn"](x, x, x)
                h_s  = SA["ffn"](SA["norm"](x + a)).squeeze(0)
            aw = cap.last()
            if aw is not None:
                aw_np = aw.squeeze(0).numpy()   # (N, N)
                result[f"self_attn_{mod}_r{r}"] = aw_np.mean(0)
                # For last block only: save capped full matrix for C×C co-attention
                if r == model.n_iter_blocks - 1:
                    N_cap = min(aw_np.shape[0], 256)
                    result[f"self_attn_matrix_{mod}"] = aw_np[:N_cap, :N_cap]
            h_self[mod] = h_s

        if len(h_self) < 2:
            h_dict = h_self; continue

        # Cross-modal cross-attn
        h_cross = {}
        for mod, h in h_self.items():
            others = torch.cat([v for k, v in h_self.items() if k != mod], dim=0)
            q = h.unsqueeze(0)
            kv = others.unsqueeze(0)
            with capture_mha([CA["attn"]]) as cap:
                a, _ = CA["attn"](q, kv, kv)
                h_c  = CA["ffn"](CA["norm"](q + a)).squeeze(0)
            aw = cap.last()
            if aw is not None:
                # Cross-attn: (1, N_q, N_kv) → per-query mean attention received from others
                result[f"cross_attn_{mod}_r{r}"] = aw.squeeze(0).mean(1).numpy()
            h_cross[mod] = h_c
        h_dict = h_cross

    # Per-modality ABMIL proxy before slots
    for mod, h in h_dict.items():
        result[f"alpha_{mod}"] = F.softmax(
            model.att_w(model.att_V(h) * model.att_U(h)), dim=0
        ).squeeze(1).detach().cpu().numpy() if not model.use_cls else np.ones(h.shape[0]) / h.shape[0]

    # Slot attention
    slot_dict, slot_assigns = {}, {}
    slices, pos = [], 0
    for mod, h in h_dict.items():
        slots, attn_KN, logits_KN = _slot_forward_capture(model.slot_attn, h)
        slot_dict[mod]   = slots
        slot_assigns[mod] = attn_KN.numpy() if attn_KN is not None else None
        result[f"slot_assign_{mod}"] = slot_assigns[mod]
        result[f"slot_logits_{mod}"] = logits_KN.numpy() if logits_KN is not None else None
        k = slots.shape[0]
        slices.append((mod, pos, pos + k)); pos += k

    all_slots = torch.cat([slot_dict[m] for m in mod_order], dim=0).unsqueeze(0)
    xmodal_attn = None
    if len(mod_order) >= 2:
        all_slots, attn_np = _xfmr_last_attn(model.cross_xfmr, all_slots)
        xmodal_attn = attn_np
    all_slots = all_slots.squeeze(0)

    result["xmodal_attn"]  = xmodal_attn
    result["mod_order"]    = mod_order
    result["slot_slices"]  = [(m, s, e) for m, s, e in slices]

    rep, alpha_slots = _pool_capture(model, all_slots, device)

    # Slot-weighted patch importance
    if alpha_slots is not None:
        for mod, s, e in slices:
            a_slot = alpha_slots[s:e]
            A      = slot_assigns[mod]
            if A is not None:
                patch_imp = (a_slot[:, None] * A).sum(0)
                result[f"slot_importance_{mod}"] = patch_imp
                result[f"alpha_{mod}"]           = patch_imp

    logit = model.head(rep).squeeze()
    result["logit"] = logit.item()
    result["prob"]  = torch.sigmoid(logit).item()

    # Hazard score
    if hasattr(model, 'hazard_head'):
        try:
            result["hazard_score"] = model.hazard_head(rep.detach()).squeeze().item()
        except Exception:
            pass

    norms = {m: slot_dict[m].norm(dim=-1).mean().item() for m in mod_order}
    result["modal_contrib"] = {m: norms[m] / (sum(norms.values()) + 1e-8) for m in norms}

    alpha_store_iter = {mod: result.get(f"alpha_{mod}", np.ones(h_store[mod].shape[0]) / h_store[mod].shape[0])
                        for mod in h_store}
    logit.backward(retain_graph=hasattr(model, 'hazard_head'))
    for mod in h_store:
        h = h_store[mod]
        if h.grad is not None:
            alpha = result.get(f"alpha_{mod}", np.ones(h.shape[0]) / h.shape[0])
            gn = h.grad.norm(dim=-1).cpu().numpy()
            result[f"grad_imp_{mod}"] = alpha * gn

    # Risk attribution
    if hasattr(model, 'hazard_head'):
        _, risk_attrs = _risk_attribution(model, h_store, rep.detach().requires_grad_(False),
                                          alpha_store_iter)
        result.update(risk_attrs)
    return result


@torch.no_grad()
def _clinical_cluster_crossattn(
    clinical_h: torch.Tensor,
    cent_tok_all: Dict[str, torch.Tensor],
    annot: Dict,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Scaled dot-product cross-attention: Q=clinical, KV=cluster centroids.

    clinical_h   : (F, H) — backbone-projected clinical feature rows.
                   Row f = one clinical feature embedded from clinical_onehot.
    cent_tok_all : {mod: (K_mod, H)} — cluster centroid tokens (from Stream B).

    Returns
    -------
    attn_FxK   : (F, K_total) numpy — softmax attention weights.
                 attn[f, k] = "how much does clinical feature f attend to cluster k?"
                 Aggregating over rows → (K,) per-cluster clinical attention score.
                 Aggregating over cols → (F,) per-feature cluster attention score.
    col_labels : list[K_total] — "mod/cluster_name" labels for columns.
    """
    if not cent_tok_all or clinical_h.shape[0] == 0:
        return None, []

    # Concatenate cluster tokens across all modalities
    all_clust_tensors = [cent_tok_all[m] for m in cent_tok_all]
    all_clust = torch.cat(all_clust_tensors, dim=0)          # (K_total, H)
    K_total   = all_clust.shape[0]
    H         = clinical_h.shape[1]

    # Scaled dot-product: (F, H) @ (H, K_total) → (F, K_total)
    scale  = H ** -0.5
    logits = torch.mm(clinical_h.float(), all_clust.float().T) * scale
    # Two views: rows softmaxed (clinical feature → cluster distribution)
    #            cols softmaxed (cluster → clinical feature distribution)
    attn_rows = torch.softmax(logits, dim=1).cpu().numpy()   # (F, K_total)

    # Build column labels: "mod/cluster_name" or "mod/k"
    col_labels: List[str] = []
    for mod in cent_tok_all:
        names = (annot.get(mod) or {}).get("cluster_names")
        K_m = cent_tok_all[mod].shape[0]
        if names:
            col_labels.extend([f"{mod}/{names[k]}" if k < len(names) else f"{mod}/{k}"
                                for k in range(K_m)])
        else:
            col_labels.extend([f"{mod}/{k}" for k in range(K_m)])

    return attn_rows, col_labels


# ══════════════════════════════════════════════════════════════════
# CLUSTER LABEL HELPERS
# ══════════════════════════════════════════════════════════════════

def _load_cluster_labels(stem: str, samples_dir) -> Dict[str, List[str]]:
    """Load per-instance cluster type strings from the .pt sample file."""
    pt = Path(samples_dir) / f"{stem}.pt"
    if not pt.exists():
        return {}
    try:
        data = torch.load(pt, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    raw = data.get("cluster_labels", {})
    out = {}
    for mod, key in MOD_TO_CLUSTER_KEY.items():
        labs = raw.get(key)
        if labs is not None:
            out[mod] = labs if isinstance(labs, list) else list(labs)
    return out


def _load_clinical_token_labels(stem: str, samples_dir) -> Optional[List[str]]:
    """
    Returns per-token vocab labels for the clinical bag of this patient,
    e.g. ["age_q3", "FEV1_q0", "BMI_nan", ...] — one label per feature token.

    Uses clinical_token_ids (which bin each feature fell into) + clinical_vocab
    (the id→label mapping) stored in the .pt file, so we can distinguish
    high-value (q3) vs low-value (q0) tokens in downstream attention plots.
    """
    pt = Path(samples_dir) / f"{stem}.pt"
    if not pt.exists():
        return None
    try:
        data = torch.load(pt, map_location="cpu", weights_only=False)
    except Exception:
        return None
    token_ids = data.get("clinical_token_ids")
    vocab     = data.get("clinical_vocab")
    if token_ids is None or vocab is None:
        return None
    id_to_label = {entry["id"]: entry["label"] for entry in vocab}
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return [id_to_label.get(int(tid), f"unk_{tid}") for tid in token_ids]


def _aggregate_by_cluster(values: np.ndarray, labels: List[str],
                           all_types: Optional[List[str]] = None
                           ) -> Tuple[np.ndarray, List[str]]:
    """
    Average `values` (N,) per unique cluster type.
    Returns (C,) means and list of C cluster type names.
    """
    n = min(len(values), len(labels))
    groups: Dict[str, List[float]] = {}
    for i in range(n):
        groups.setdefault(labels[i], []).append(float(values[i]))
    types = all_types if all_types is not None else sorted(groups.keys())
    means = np.array([np.mean(groups[t]) if t in groups else 0.0 for t in types],
                     dtype=np.float32)
    return means, types


def _slot_cluster_matrix(slot_KN: np.ndarray, labels: List[str],
                          all_types: Optional[List[str]] = None
                          ) -> Tuple[np.ndarray, List[str]]:
    """
    Compute K×C matrix: mean slot weight per (slot, cluster_type) pair.

    slot_KN  : (K, N) slot assignment (post- or pre-softmax)
    labels   : N cluster type strings
    Returns  : (K, C) array, list of C cluster type names
    """
    K = slot_KN.shape[0]
    n = min(slot_KN.shape[1], len(labels))
    A = slot_KN[:, :n]
    labs = labels[:n]
    ctypes = all_types if all_types is not None else sorted(set(labs))
    C = len(ctypes)
    idx_map = {ct: i for i, ct in enumerate(ctypes)}
    result = np.zeros((K, C), dtype=np.float32)
    counts = np.zeros(C, dtype=np.int32)
    for j, ct in enumerate(labs):
        ci = idx_map.get(ct)
        if ci is not None:
            result[:, ci] += A[:, j]
            counts[ci] += 1
    mask = counts > 0
    result[:, mask] /= counts[mask]
    return result, ctypes


@torch.enable_grad()
def _extract_v7(model, bags, annot, device, label) -> Optional[dict]:
    """v7 TripleStreamFusionMIL extractor (Stream A + B + C)."""
    result: dict = {"label": label}
    r_bag, h_store, alpha_store = {}, {}, {}

    for mod, enc in model.encoders.items():
        t = bags.get(mod)
        if t is None: continue
        h = _backbone_h(enc, t.to(device))
        rep, alpha = _abmil_forward(enc, h)
        h_store[mod]           = h
        alpha_store[mod]       = alpha.detach().cpu().numpy()
        result[f"alpha_{mod}"] = alpha_store[mod]
        result[f"h_{mod}"]     = h.detach().cpu().numpy()
        r_bag[mod]             = rep

    if not r_bag: return None
    result["mods_present"] = list(r_bag.keys())
    mod_order = list(r_bag.keys())

    bag_tokens = torch.stack(list(r_bag.values()), dim=0)
    xmodal_attn = None
    if len(r_bag) >= 2:
        bag_tokens_out, attn_np = _xfmr_last_attn(
            model.patient_xfmr, bag_tokens.unsqueeze(0))
        bag_tokens_out = bag_tokens_out.squeeze(0)
        xmodal_attn    = attn_np
    else:
        bag_tokens_out = bag_tokens

    result["xmodal_attn"] = xmodal_attn
    result["mod_order"]   = mod_order

    r_bag_patient = model._pool_tokens(bag_tokens_out, device)

    # Stream B: centroids
    cent_tok_all, gate_outputs, slices, pos = {}, {}, [], 0
    for mod in ANNOT_MODS:
        if mod not in r_bag: continue
        a = annot.get(mod)
        if a is None: continue
        c = a["centroids"].to(device); n_cnt = a["counts"].to(device)
        enc_c = model.cent_encoders[mod]
        gate_cap = []
        hdl = enc_c.count_gate.register_forward_hook(
            lambda m, inp, out: gate_cap.append(torch.sigmoid(out).detach().cpu()))
        tok = enc_c(c, n_cnt)
        hdl.remove()
        if gate_cap:
            result[f"cent_gate_{mod}"] = gate_cap[0].mean(-1).numpy()
        cent_tok_all[mod] = tok
        k = tok.shape[0]; slices.append((mod, pos, pos + k)); pos += k

    cent_xattn = None; r_cent_summary = None
    if cent_tok_all:
        all_tok = torch.cat(list(cent_tok_all.values()), dim=0).unsqueeze(0)
        if len(cent_tok_all) >= 2:
            all_tok, attn_np = _xfmr_last_attn(model.cent_xfmr, all_tok)
            cent_xattn = attn_np
        all_tok = all_tok.squeeze(0)
        r_cent_per = {m: all_tok[s:e].mean(0) for m, s, e in slices}
        r_cent_summary = model._xfmr_pool(r_cent_per, model.cent_summary_xfmr,
                                           model.cent_summary_norm, model.cent_summary_ffn)
    result["cent_xattn"]  = cent_xattn
    result["cent_slices"] = [(m, s, e) for m, s, e in slices]

    # Stream C: counts (cluster_count_onehot → pooled K*4 repr)
    count_reps = {}
    for mod in ANNOT_MODS:
        if mod not in r_bag: continue
        a = annot.get(mod)
        if a is None: continue
        n_max = ANNOT_REGISTRY[mod][2]; n_bins = 4
        expected_dim = n_max * n_bins
        coh = a.get("count_onehot")
        if coh is not None:
            n_rep = coh.to(device).sum(0)
            if n_rep.shape[0] < expected_dim:
                n_rep = F.pad(n_rep, (0, expected_dim - n_rep.shape[0]))
            elif n_rep.shape[0] > expected_dim:
                n_rep = n_rep[:expected_dim]
        else:
            n = a["counts"].to(device)
            if n.shape[0] < n_max: n = F.pad(n, (0, n_max - n.shape[0]))
            n_rep = n.unsqueeze(1).expand(-1, n_bins).reshape(-1)
        count_reps[mod] = model.count_embedders[mod](n_rep)
    r_count_summary = model._xfmr_pool(
        count_reps, model.count_xfmr, model.count_norm, model.count_ffn) if count_reps else None

    parts = [x for x in (r_cent_summary, r_count_summary) if x is not None]
    if len(parts) == 2:
        r_final = model.final_fuse_3(torch.cat([r_bag_patient, parts[0], parts[1]]))
    elif len(parts) == 1:
        r_final = model.final_fuse_2(torch.cat([r_bag_patient, parts[0]]))
    else:
        r_final = r_bag_patient

    logit = model.head(r_final).squeeze()
    result["logit"] = logit.item()
    result["prob"]  = torch.sigmoid(logit).item()

    norms = {m: r_bag[m].norm().item() for m in r_bag}
    result["modal_contrib"] = {m: norms[m] / (sum(norms.values()) + 1e-8) for m in norms}

    logit.backward()
    for mod, h in h_store.items():
        if h.grad is not None:
            gn = h.grad.norm(dim=-1).cpu().numpy()
            result[f"grad_imp_{mod}"] = alpha_store[mod] * gn

    # ── Clinical × Cluster cross-attention (interpretability) ─────────────────
    # h_store["Clinical"] = (F, H) — backbone-projected clinical feature rows.
    # Each row f = one clinical feature as an instance (from clinical_onehot).
    # cent_tok_all[mod]   = (K_mod, H) — backbone-projected cluster centroid tokens.
    # Scaled dot-product Q=clinical(F,H), KV=clusters(K_total,H) → (F, K_total).
    # Gives: "for each clinical feature, how much does it attend to each cluster?"
    if cent_tok_all and "Clinical" in h_store:
        clinical_h = h_store["Clinical"]          # (F, H) — detached for attn
        attn_FxK, col_labels = _clinical_cluster_crossattn(
            clinical_h.detach(), cent_tok_all, annot)
        if attn_FxK is not None:
            result["clin_clust_attn"]      = attn_FxK   # (F, K_total) numpy
            result["clin_clust_col_names"] = col_labels  # list[K_total]

    return result


# ══════════════════════════════════════════════════════════════════
# DISPATCHER
# ══════════════════════════════════════════════════════════════════

def extract_sample(model, bags, annot, device, label) -> Optional[dict]:
    model.eval()
    if isinstance(model, EarlyFusionMIL):
        return _extract_early(model, bags, device, label)
    elif isinstance(model, LateFusionMIL):
        return _extract_late(model, bags, device, label)
    elif isinstance(model, MiddleFusionMIL):
        return _extract_middle(model, bags, device, label)
    elif isinstance(model, CrossAttnFusionMIL):
        return _extract_slot_base(model, bags, device, label, has_bidir=True)
    elif isinstance(model, SlotCrossModalMIL):
        return _extract_slot_base(model, bags, device, label, has_bidir=False)
    elif isinstance(model, IterativeXModalMIL):
        return _extract_iterative(model, bags, device, label)
    else:
        # v7 TripleStreamFusionMIL (import at runtime to avoid circular issues)
        return _extract_v7(model, bags, annot, device, label)


# ══════════════════════════════════════════════════════════════════
# COHORT RUNNER
# ══════════════════════════════════════════════════════════════════

def run_extraction(model, records, bag_cache, annot_cache, device, out_dir,
                   version="v7", samples_dir=None):
    npy_dir = out_dir / "npy"; npy_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, rec in enumerate(records):
        stem  = rec["stem"]
        entry = bag_cache.get(stem, {})
        bags  = {m: entry.get(m) for m in MODALITIES}
        annot = annot_cache.get(stem, {m: None for m in ANNOT_MODS})
        if all(v is None for v in bags.values()): continue

        try:
            r = extract_sample(model, bags, annot, device, rec["label"])
        except Exception as e:
            print(f"  [warn] {stem}: {e}")
            r = None

        if r is None: continue
        r["stem"] = stem
        # Attach survival data from the record
        r["surv_clad_time"]  = rec.get("clad_time",  float("nan"))
        r["surv_clad_event"] = rec.get("clad_event", float("nan"))
        r["surv_death_time"] = rec.get("death_time",  float("nan"))
        r["surv_death_event"]= rec.get("death_event", float("nan"))
        # Attach clinical feature names and cluster names from bag_cache metadata
        cfn = entry.get("_clinical_feature_names")
        if cfn is not None:
            r["clinical_feature_names"] = cfn
        for mod in ANNOT_MODS:
            cn = entry.get(f"_{mod}_cluster_names")
            if cn is not None:
                r[f"{mod}_cluster_names"] = cn

        # Attach per-instance cluster labels from .pt file (truncated to processed N)
        if samples_dir is not None:
            cl_map = _load_cluster_labels(stem, samples_dir)
            for mod, labels in cl_map.items():
                ref = r.get(f"alpha_{mod}")
                if ref is None:
                    ref_h = r.get(f"h_{mod}")
                    n = len(ref_h) if ref_h is not None else len(labels)
                else:
                    n = len(ref)
                r[f"cluster_labels_{mod}"] = labels[:n]

        # Clinical: use vocab labels (e.g. "FEV1_q3") so plots distinguish
        # high vs low values, not just feature names.
        alpha_clin = r.get("alpha_Clinical")
        if alpha_clin is not None:
            clin_labels = (
                _load_clinical_token_labels(stem, samples_dir)
                if samples_dir is not None else None
            )
            if clin_labels is not None:
                r["cluster_labels_Clinical"] = clin_labels[:len(alpha_clin)]
            else:
                cfn = r.get("clinical_feature_names")
                if cfn is not None:
                    r["cluster_labels_Clinical"] = list(cfn)[:len(alpha_clin)]

        np_data = {k: v for k, v in r.items() if isinstance(v, np.ndarray)}
        meta    = {k: v for k, v in r.items()
                   if not isinstance(v, np.ndarray) and k != "patch_mod_labels"}
        if np_data:
            np.savez_compressed(npy_dir / f"{stem}.npz", **np_data)
        try:
            with open(npy_dir / f"{stem}_meta.json", "w") as f:
                json.dump(meta, f, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
        except Exception:
            pass

        results.append(r)
        if (i + 1) % 20 == 0:
            print(f"  extracted {i+1}/{len(records)}", flush=True)
        _gc6()
        gc.collect()  # free retained autograd graphs (retain_graph=True from attribution)

    print(f"  Saved {len(results)} samples → {npy_dir}")
    return results


# ══════════════════════════════════════════════════════════════════
# VISUALIZATION — PER SAMPLE
# ══════════════════════════════════════════════════════════════════

def _model_type_label(r):
    """Infer a brief model description from result keys."""
    if "cent_gate_HE" in r: return "v7 TripleStream"
    if "slot_assign_HE" in r and "bidir_attn_HE_BAL" in r: return "CrossAttnFusion"
    if "slot_assign_HE" in r and any(k.startswith("self_attn_") for k in r): return "IterativeXModal"
    if "slot_assign_HE" in r: return "SlotCrossModal"
    if "modal_logits" in r: return "LateFusion"
    if "alpha_joint" in r: return "EarlyFusion"
    return "MiddleFusion"


def _active_slot_mask(A: np.ndarray, slot_thr: Optional[float],
                       min_active: int = 3) -> np.ndarray:
    """Boolean mask (K,): True for slots with meaningful attention.
    slot_thr=None → auto = 2/K (twice the uniform baseline)."""
    K, N = A.shape
    thr = slot_thr if slot_thr is not None else 2.0 / max(K, 1)
    mask = A.max(axis=1) > thr
    # always keep at least min_active (the most active ones)
    if mask.sum() < min_active:
        top = np.argsort(A.max(axis=1))[-min_active:]
        mask = np.zeros(K, dtype=bool)
        mask[top] = True
    return mask


def _sig_instance_cols(A_active: np.ndarray, alpha: Optional[np.ndarray],
                        max_inst: int = 48) -> np.ndarray:
    """Column indices of instances that are strongly owned by ≥1 active slot,
    intersected with high-alpha ranking. Returns sorted indices."""
    K, N = A_active.shape
    inst_thr = 1.5 / max(K, 1)
    owned = np.where(A_active.max(axis=0) > inst_thr)[0]
    if alpha is not None and len(alpha) == N:
        top_alpha = np.argsort(alpha)[-max_inst:]
        cols = np.intersect1d(owned, top_alpha)
        if len(cols) == 0:         # fall back: just top-alpha
            cols = np.argsort(alpha)[-min(max_inst, N):]
    else:
        cols = owned
    # cap and sort by alpha if available
    if alpha is not None and len(alpha) == N and len(cols) > max_inst:
        cols = cols[np.argsort(alpha[cols])[-max_inst:]]
    return np.sort(cols)


def plot_sample(r: dict, out_path: Path, top_k: int = 16,
                slot_thr: Optional[float] = None,
                min_active_slots: int = 3,
                max_inst_shown: int = 48,
                mark_cell_thr: Optional[float] = None):
    """
    slot_thr: active-slot threshold (None → auto 2/K).
    min_active_slots: always keep at least this many slots.
    max_inst_shown: max instances shown on slot heatmap column axis.
    mark_cell_thr: grey-out heatmap cells below this value (None → skip).
    """
    mods      = r.get("mods_present", [])
    has_xattn = r.get("xmodal_attn") is not None
    has_cent  = any(f"cent_gate_{m}" in r for m in ANNOT_MODS)
    has_slots = any(f"slot_assign_{m}" in r for m in mods)
    has_bidir = any(k.startswith("bidir_attn_") for k in r)

    n_rows = 1 + int(has_xattn) + int(has_cent) + int(has_slots)
    fig = plt.figure(figsize=(max(14, 4 * len(mods)), 4 * n_rows))
    gs  = gridspec.GridSpec(n_rows, max(len(mods), 2), figure=fig,
                            hspace=0.5, wspace=0.35)
    row = 0

    # ── Row 0: per-modality attention bars ───────────────────────
    hazard_score = r.get("hazard_score")
    for ci, mod in enumerate(mods):
        ax = fig.add_subplot(gs[row, ci])
        alpha    = r.get(f"alpha_{mod}")
        grad_imp = r.get(f"grad_imp_{mod}")
        risk_imp = r.get(f"risk_attr_{mod}")
        bidir    = r.get(f"bidir_attn_HE_{mod}") if mod != "HE" else r.get("bidir_attn_HE_BAL")
        if alpha is None: continue
        n   = len(alpha)
        idx = np.argsort(alpha)[-top_k:][::-1]
        x   = np.arange(len(idx))

        color = MOD_COLORS.get(mod, "steelblue")
        label_str = "slot imp" if f"slot_importance_{mod}" in r else "ABMIL α"
        ax.bar(x, alpha[idx], color=color, alpha=0.75, label=label_str)

        if grad_imp is not None:
            ax2 = ax.twinx()
            ax2.plot(x, grad_imp[idx], "o--", color="crimson", ms=4, lw=1.2)
            ax2.set_ylabel("Grad×α", fontsize=7, color="crimson")
            ax2.tick_params(axis="y", colors="crimson", labelsize=6)

        ax.set_title(f"{mod}  (N={n})", fontsize=9)
        ax.set_xlabel("Instance rank", fontsize=7)
        ax.set_ylabel(label_str, fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(idx + 1, rotation=45, fontsize=5)
        ax.legend(fontsize=7)

        # Show risk attribution if available (overlay as step line)
        if risk_imp is not None and len(risk_imp) >= len(idx):
            ax3 = ax.twinx() if grad_imp is None else ax.twinx()
            ax3.plot(x, risk_imp[idx], "s:", color="darkorange", ms=3, lw=1.0, alpha=0.8, label="risk attr")
            ax3.set_ylabel("Risk attr", fontsize=6, color="darkorange")
            ax3.tick_params(axis="y", colors="darkorange", labelsize=5)
            ax3.spines["right"].set_position(("outward", 40))

        # Hazard score annotation on first modality panel
        if ci == 0 and hazard_score is not None:
            ax.text(0.98, 0.98, f"hazard={hazard_score:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7,
                    bbox=dict(facecolor="lightyellow", edgecolor="gray", alpha=0.8, boxstyle="round,pad=0.2"))
    row += 1

    # ── Row 1: cross-modal attention (slot-level or modality-level) ─
    if has_xattn:
        xa = r["xmodal_attn"]
        mo = r.get("mod_order", mods)
        ax = fig.add_subplot(gs[row, :])

        if "slot_slices" in r:
            # Slot-level: filter to active slots only across all modalities
            slices = r["slot_slices"]
            all_A = [r.get(f"slot_assign_{m}") for _, _, m in slices
                     if r.get(f"slot_assign_{m}") is not None]
            if all_A:
                active_per_mod = []
                offset = 0
                global_active = []
                for mod_name, s, e in slices:
                    A_m = r.get(f"slot_assign_{mod_name}")
                    if A_m is None:
                        offset += (e - s); continue
                    local_mask = _active_slot_mask(A_m, slot_thr, min_active_slots)
                    global_active.extend([s + i for i, ok in enumerate(local_mask) if ok])
                    offset += (e - s)
                global_active = np.array(global_active)
                if len(global_active) > 1:
                    xa_sub = xa[np.ix_(global_active, global_active)]
                    im = ax.imshow(xa_sub, cmap="Blues", vmin=0)
                    plt.colorbar(im, ax=ax, fraction=0.02)
                    ax.set_title(
                        f"Slot cross-modal attention — active slots only "
                        f"({len(global_active)}/{xa.shape[0]} slots)", fontsize=9)
                else:
                    im = ax.imshow(xa, cmap="Blues", vmin=0)
                    plt.colorbar(im, ax=ax, fraction=0.02)
                    ax.set_title("Slot cross-modal attention", fontsize=9)
            else:
                im = ax.imshow(xa, cmap="Blues", vmin=0)
                plt.colorbar(im, ax=ax, fraction=0.02)
                ax.set_title("Slot cross-modal attention", fontsize=9)
            ax.set_xlabel("Key slots"); ax.set_ylabel("Query slots")
        else:
            im = ax.imshow(xa, cmap="Blues", vmin=0)
            ax.set_xticks(range(len(mo))); ax.set_xticklabels(mo, fontsize=8)
            ax.set_yticks(range(len(mo))); ax.set_yticklabels(mo, fontsize=8)
            for i in range(len(mo)):
                for j in range(len(mo)):
                    ax.text(j, i, f"{xa[i,j]:.2f}", ha="center", va="center", fontsize=7)
            ax.set_title("Cross-modal attention (query → key)", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.03)
        row += 1

    # ── Row 2: slot assignment heatmap — active slots × significant instances ─
    if has_slots:
        for ci, mod in enumerate(mods):
            A = r.get(f"slot_assign_{mod}")
            if A is None: continue
            K, N = A.shape
            alpha = r.get(f"alpha_{mod}")

            # Layer 1 — active slot rows
            slot_mask = _active_slot_mask(A, slot_thr, min_active_slots)
            active_idx = np.where(slot_mask)[0]
            A_active = A[active_idx, :]

            # Layer 1 — significant instance columns
            cols = _sig_instance_cols(A_active, alpha, max_inst_shown)
            if len(cols) == 0:
                cols = np.arange(min(max_inst_shown, N))
            A_show = A_active[:, cols]

            # Layer 2 — cell significance mask (white-out weak cells)
            if mark_cell_thr is not None:
                A_masked = np.where(A_show >= mark_cell_thr, A_show, np.nan)
            else:
                A_masked = A_show

            ax = fig.add_subplot(gs[row, ci])
            cmap = plt.get_cmap(SLOT_CMAP).copy()
            cmap.set_bad(color="whitesmoke")
            im = ax.imshow(A_masked, aspect="auto", cmap=cmap, vmin=0,
                           interpolation="nearest")
            ax.set_yticks(range(len(active_idx)))
            ax.set_yticklabels([f"S{k}" for k in active_idx], fontsize=6)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(cols + 1, rotation=60, fontsize=5, ha="right")
            ax.set_xlabel("Instance (significant)", fontsize=7)
            ax.set_ylabel("Active slot", fontsize=7)
            cell_note = f"  |  cells<{mark_cell_thr:.3f} masked" if mark_cell_thr else ""
            ax.set_title(
                f"{mod}  active {len(active_idx)}/{K} slots × {len(cols)} sig. instances{cell_note}",
                fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.05)
        row += 1

    # ── Row 3: centroid gate (v7) ────────────────────────────────
    if has_cent:
        ax = fig.add_subplot(gs[row, :])
        offset = 0
        for mod in ANNOT_MODS:
            gates = r.get(f"cent_gate_{mod}")
            if gates is None: continue
            x = np.arange(offset, offset + len(gates))
            ax.bar(x, gates, color=MOD_COLORS.get(mod, "gray"), alpha=0.8, label=mod)
            offset += len(gates) + 1
        ax.set_title("Centroid count-gate (v7 Stream B)", fontsize=9)
        ax.set_ylabel("mean σ(gate)"); ax.legend(fontsize=8)

    # ── Row: clinical × cluster cross-attention ──────────────────
    cca = r.get("clin_clust_attn")
    if cca is not None:
        ax_cca = fig.add_subplot(gs[row, :])
        feat_names = r.get("clinical_feature_names")
        col_names  = r.get("clin_clust_col_names")
        F_s, K_s   = cca.shape
        # Show top-20 features by max attention
        top_f = np.argsort(-cca.max(axis=1))[:min(20, F_s)]
        top_k = np.argsort(-cca.max(axis=0))[:min(40, K_s)]
        sub   = cca[np.ix_(top_f, top_k)]
        im = ax_cca.imshow(sub, aspect="auto", cmap="YlOrRd", vmin=0,
                           interpolation="nearest")
        fn = [feat_names[i] if feat_names and i < len(feat_names) else str(i)
              for i in top_f]
        cn = [col_names[i] if col_names and i < len(col_names) else str(i)
              for i in top_k]
        ax_cca.set_yticks(range(len(top_f))); ax_cca.set_yticklabels(fn, fontsize=5)
        ax_cca.set_xticks(range(len(top_k))); ax_cca.set_xticklabels(cn, rotation=70,
                           ha="right", fontsize=5)
        ax_cca.set_title("Clinical feature → Cluster attention  (Q=clinical, KV=cluster tokens)",
                         fontsize=8)
        plt.colorbar(im, ax=ax_cca, fraction=0.02, shrink=0.7)

    prob  = r.get("prob", 0.5); label = r.get("label", -1); stem = r.get("stem", "")
    mtype = _model_type_label(r)
    hazard_score = r.get("hazard_score")
    hazard_str = f"  hazard={hazard_score:.3f}" if hazard_score is not None else ""
    fig.suptitle(f"{stem}  [{mtype}]  label={LABEL_NAMES.get(label, label)}  prob={prob:.3f}{hazard_str}",
                 fontsize=10, fontweight="bold")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# VISUALIZATION — COHORT
# ══════════════════════════════════════════════════════════════════

def plot_cross_modal_attention(results, out_path: Path):
    attn_by_label = {0: [], 1: []}; mod_order = None
    for r in results:
        xa = r.get("xmodal_attn"); mo = r.get("mod_order")
        if xa is None or mo is None: continue
        lab = r.get("label", -1)
        if lab not in (0, 1): continue
        if mod_order is None or (len(mo) >= len(mod_order)): mod_order = mo
        # Only accumulate modality-level matrices (not full slot matrices)
        if xa.shape[0] == len(mo):
            attn_by_label[lab].append(xa)

    if not any(attn_by_label.values()) or mod_order is None: return
    M = len(mod_order)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, lab in zip(axes, [0, 1]):
        vals = [v for v in attn_by_label[lab] if v.shape[0] == M]
        if not vals: ax.set_visible(False); continue
        mean_a = np.stack(vals).mean(0)
        im = ax.imshow(mean_a, cmap="Blues", vmin=0)
        ax.set_xticks(range(M)); ax.set_xticklabels(mod_order, fontsize=9)
        ax.set_yticks(range(M)); ax.set_yticklabels(mod_order, fontsize=9)
        ax.set_title(f"{LABEL_NAMES.get(lab, lab)} (n={len(vals)})", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        for i in range(M):
            for j in range(M):
                ax.text(j, i, f"{mean_a[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.suptitle("Mean cross-modal attention (patient transformer)", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out_path}")


def plot_modal_contributions(results, out_path: Path):
    contrib = {0: {m: [] for m in MODALITIES}, 1: {m: [] for m in MODALITIES}}
    for r in results:
        lab = r.get("label", -1)
        if lab not in (0, 1): continue
        for mod, v in r.get("modal_contrib", {}).items():
            contrib[lab][mod].append(v)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(MODALITIES)); w = 0.35
    for lab, color, off in [(0, "#4e79a7", -w/2), (1, "#e15759", w/2)]:
        means = [np.mean(contrib[lab][m]) if contrib[lab][m] else 0 for m in MODALITIES]
        ses   = [np.std(contrib[lab][m]) / max(len(contrib[lab][m])**.5, 1) for m in MODALITIES]
        ax.bar(x + off, means, w, color=color, alpha=0.8, label=LABEL_NAMES[lab])
        ax.errorbar(x + off, means, ses, fmt="none", color="black", capsize=3, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(MODALITIES)
    ax.set_ylabel("Contribution fraction"); ax.set_title("Modality contribution to patient rep")
    ax.legend(); fig.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out_path}")


def plot_centroid_importance(results, out_path: Path):
    gate_by = {m: {0: [], 1: []} for m in ANNOT_MODS}
    for r in results:
        lab = r.get("label", -1)
        if lab not in (0, 1): continue
        for mod in ANNOT_MODS:
            g = r.get(f"cent_gate_{mod}")
            if g is not None: gate_by[mod][lab].append(g)

    mods = [m for m in ANNOT_MODS if any(gate_by[m].values())]
    if not mods: return
    fig, axes = plt.subplots(1, len(mods), figsize=(6 * len(mods), 4))
    if len(mods) == 1: axes = [axes]
    for ax, mod in zip(axes, mods):
        n_k = ANNOT_REGISTRY[mod][2]; x = np.arange(n_k); w = 0.35
        for lab, color, name in [(0, "#4e79a7", "No rejection"), (1, "#e15759", "Rejection")]:
            vals = gate_by[mod][lab]
            if not vals: continue
            arr = np.stack([v[:n_k] if len(v) >= n_k else np.pad(v, (0, n_k - len(v))) for v in vals])
            off = -w/2 if lab == 0 else w/2
            ax.bar(x + off, arr.mean(0), w, color=color, alpha=0.8, label=name)
            ax.errorbar(x + off, arr.mean(0), arr.std(0)/(len(arr)**.5),
                        fmt="none", color="black", capsize=2, lw=0.8)
        ax.set_title(f"{mod} centroid gate"); ax.set_xlabel("Cluster"); ax.set_ylabel("σ(gate)")
        ax.legend(fontsize=8); ax.set_xticks(x); ax.set_xticklabels(x, fontsize=5)
    fig.suptitle("Which centroid clusters gate-up per label?", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out_path}")


def plot_clinical_cluster_attn(results: List[dict], out_dir: Path,
                               top_k_feat: int = 30, top_k_clust: int = 40):
    """
    Cohort-level clinical × cluster cross-attention visualisation.

    Produces per-label mean (F, K) attention heatmaps + discriminative summaries:
      (a) Mean attention heatmap: top_k_feat clinical features × top_k_clust clusters.
          Rows = clinical features sorted by |pos-neg| difference.
          Cols = clusters sorted by |pos-neg| difference.
      (b) Per-cluster bar chart: mean attention from ALL clinical features → each cluster,
          split pos/neg.  "Which clusters does the clinical profile attend to?"
      (c) Per-clinical-feature bar chart: mean attention from each feature → ALL clusters,
          split pos/neg.  "Which features are most discriminative by cluster response?"
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Align samples: collect (attn_FxK, feat_names, col_names, label)
    by_label: Dict[int, List[np.ndarray]] = {0: [], 1: []}
    feat_names_ref: Optional[List[str]] = None
    col_names_ref:  Optional[List[str]] = None

    for r in results:
        attn = r.get("clin_clust_attn")
        lab  = r.get("label", -1)
        if attn is None or lab not in (0, 1): continue
        by_label[lab].append(attn)
        if feat_names_ref is None:
            feat_names_ref = r.get("clinical_feature_names")
        if col_names_ref is None:
            col_names_ref  = r.get("clin_clust_col_names")

    n0 = len(by_label[0]); n1 = len(by_label[1])
    if n0 + n1 == 0:
        print("  [clin_clust_attn] no data — skipping"); return

    # Align shapes: use the most common (F, K) shape
    all_mats = by_label[0] + by_label[1]
    shapes   = [m.shape for m in all_mats]
    F_ref, K_ref = max(set(shapes), key=shapes.count)

    def _align(mats):
        aligned = []
        for m in mats:
            if m.shape == (F_ref, K_ref):
                aligned.append(m)
            else:
                f = min(m.shape[0], F_ref); k = min(m.shape[1], K_ref)
                tmp = np.zeros((F_ref, K_ref), dtype=np.float32)
                tmp[:f, :k] = m[:f, :k]
                aligned.append(tmp)
        return aligned

    mats0 = _align(by_label[0]); mats1 = _align(by_label[1])
    mean0 = np.stack(mats0).mean(0) if mats0 else np.zeros((F_ref, K_ref))
    mean1 = np.stack(mats1).mean(0) if mats1 else np.zeros((F_ref, K_ref))

    feat_names = (feat_names_ref or [f"F{f}" for f in range(F_ref)])[:F_ref]
    col_names  = (col_names_ref  or [f"C{k}" for k in range(K_ref)])[:K_ref]

    diff = mean1 - mean0  # (F, K)  pos > neg → positive

    # ── (a) Heatmap: discriminative features × clusters ───────────
    feat_importance  = np.abs(diff).mean(axis=1)  # (F,) mean |diff| across clusters
    clust_importance = np.abs(diff).mean(axis=0)  # (K,) mean |diff| across features

    top_f_idx = np.argsort(-feat_importance)[:top_k_feat]
    top_k_idx = np.argsort(-clust_importance)[:top_k_clust]

    top_f_names = [feat_names[i] if i < len(feat_names) else f"F{i}" for i in top_f_idx]
    top_k_names = [col_names[i]  if i < len(col_names)  else f"C{i}" for i in top_k_idx]

    for suffix, mat, title_suf in [
        ("neg", mean0[np.ix_(top_f_idx, top_k_idx)], f"Neg A0 (n={n0})"),
        ("pos", mean1[np.ix_(top_f_idx, top_k_idx)], f"Pos A1/A2 (n={n1})"),
        ("diff", diff[np.ix_(top_f_idx, top_k_idx)],  "Pos − Neg"),
    ]:
        fig_h = max(6, len(top_f_idx) * 0.22 + 1.5)
        fig_w = max(8, len(top_k_idx) * 0.25 + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        vmax = np.abs(mat).max() + 1e-8
        cmap = "RdBu_r" if suffix == "diff" else "YlOrRd"
        vmin = -vmax if suffix == "diff" else 0
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks(range(len(top_k_idx)))
        ax.set_xticklabels(top_k_names, rotation=70, ha="right", fontsize=6)
        ax.set_yticks(range(len(top_f_idx)))
        ax.set_yticklabels(top_f_names, fontsize=6)
        ax.set_xlabel("Cluster (mod/name)", fontsize=9)
        ax.set_ylabel("Clinical feature", fontsize=9)
        ax.set_title(f"Clinical → Cluster attention — {title_suf}\n"
                     f"Top {len(top_f_idx)} features × top {len(top_k_idx)} clusters "
                     f"(sorted by |pos−neg|)", fontsize=9)
        plt.colorbar(im, ax=ax, label="attention weight", shrink=0.6, fraction=0.02)
        fig.tight_layout()
        p = out_dir / f"clin_clust_attn_{suffix}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  → {p}")

    # ── (b) Per-cluster summary bar: Σ_f attn[f,k] per label ─────
    clust_score0 = mean0.mean(axis=0)  # (K,)
    clust_score1 = mean1.mean(axis=0)
    diff_clust   = clust_score1 - clust_score0
    sort_k       = np.argsort(-np.abs(diff_clust))[:top_k_clust]

    fig, ax = plt.subplots(figsize=(max(10, len(sort_k) * 0.38 + 2), 5))
    x = np.arange(len(sort_k)); w = 0.38
    ax.bar(x - w/2, clust_score0[sort_k], w, color="#5c9be0", alpha=0.85, label="Neg A0")
    ax.bar(x + w/2, clust_score1[sort_k], w, color="#e05c5c", alpha=0.85, label="Pos A1/A2")
    ax.set_xticks(x)
    ax.set_xticklabels([col_names[i] if i < len(col_names) else str(i) for i in sort_k],
                       rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("Mean clinical attention → cluster", fontsize=9)
    ax.set_title("Per-cluster: total clinical attention (Σ_features)\n"
                 "Sorted by |pos−neg| difference", fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    p = out_dir / "clin_clust_per_cluster.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")

    # ── (c) Per-feature summary bar: Σ_k attn[f,k] per label ────
    feat_score0 = mean0.mean(axis=1)   # (F,)
    feat_score1 = mean1.mean(axis=1)
    diff_feat   = feat_score1 - feat_score0
    sort_f      = np.argsort(-np.abs(diff_feat))[:top_k_feat]

    fig, ax = plt.subplots(figsize=(max(10, len(sort_f) * 0.38 + 2), 5))
    x = np.arange(len(sort_f)); w = 0.38
    ax.bar(x - w/2, feat_score0[sort_f], w, color="#5c9be0", alpha=0.85, label="Neg A0")
    ax.bar(x + w/2, feat_score1[sort_f], w, color="#e05c5c", alpha=0.85, label="Pos A1/A2")
    ax.set_xticks(x)
    ax.set_xticklabels([feat_names[i] if i < len(feat_names) else str(i) for i in sort_f],
                       rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Mean cluster attention ← feature (Σ_clusters)", fontsize=9)
    ax.set_title("Per-clinical-feature: total cluster attention\n"
                 "Sorted by |pos−neg| difference", fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    p = out_dir / "clin_clust_per_feature.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")

    print(f"  Clinical×Cluster attention plots → {out_dir}")


def plot_slot_assignment_summary(results, out_path: Path, mod: str = "HE",
                                  slot_thr: Optional[float] = None,
                                  min_active_slots: int = 3):
    """
    Cohort-level: mean slot assignment per slot, split by label.
    Active slots (max attention > slot_thr) shown in full colour.
    Dead slots shown faded. Significance stars from Mann-Whitney U test.
    """
    by_label = {0: [], 1: []}
    for r in results:
        lab = r.get("label", -1)
        if lab not in (0, 1): continue
        A = r.get(f"slot_assign_{mod}")
        if A is None: continue
        by_label[lab].append(A.sum(1) / (A.sum() + 1e-8))  # (K,) usage fraction

    if not any(by_label.values()): return
    K_vals = [v.shape[0] for v in (by_label[0] + by_label[1]) if hasattr(v, 'shape')]
    if not K_vals: return
    K = K_vals[0]

    arr0 = np.stack([v for v in by_label[0] if len(v) == K]) if by_label[0] else None
    arr1 = np.stack([v for v in by_label[1] if len(v) == K]) if by_label[1] else None

    # Layer 1 — active slot mask across cohort
    # Use mean per-slot max-attention over all patients to decide activity
    all_A_raw = []
    for r in results:
        A = r.get(f"slot_assign_{mod}")
        if A is not None and A.shape[0] == K:
            all_A_raw.append(A.max(axis=1))  # (K,) per patient
    if all_A_raw:
        cohort_max = np.stack(all_A_raw).mean(0)  # (K,) mean peak attention
        thr = slot_thr if slot_thr is not None else 2.0 / max(K, 1)
        active_mask = cohort_max > thr
        if active_mask.sum() < min_active_slots:
            top = np.argsort(cohort_max)[-min_active_slots:]
            active_mask = np.zeros(K, dtype=bool); active_mask[top] = True
    else:
        active_mask = np.ones(K, dtype=bool)

    # Layer 3 — Mann-Whitney U significance per slot
    try:
        from scipy.stats import mannwhitneyu
        pvals = np.ones(K)
        if arr0 is not None and arr1 is not None:
            for k in range(K):
                _, pvals[k] = mannwhitneyu(arr0[:, k], arr1[:, k], alternative="two-sided")
        has_stats = True
    except ImportError:
        pvals = np.ones(K)
        has_stats = False

    def _sig_star(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    x = np.arange(K); w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, K * 0.55 + 2), 5))

    for lab, color, off, name in [(0, "#4e79a7", -w/2, "No rejection"),
                                   (1, "#e15759", w/2,  "Rejection")]:
        arr = arr0 if lab == 0 else arr1
        if arr is None: continue
        means = arr.mean(0)
        sems  = arr.std(0) / max(len(arr) ** 0.5, 1)
        # faded colour for dead slots
        bar_colors = [color if active_mask[k] else "#cccccc" for k in range(K)]
        ax.bar(x + off, means, w, color=bar_colors, alpha=0.85, label=name,
               edgecolor="white", linewidth=0.4)
        ax.errorbar(x + off, means, sems, fmt="none", color="#444", capsize=2, lw=0.8)

    # Significance annotations above the bar pairs
    y_max = ax.get_ylim()[1]
    for k in range(K):
        star = _sig_star(pvals[k])
        if not star: continue
        bar_top = max(
            (arr0[:, k].mean() if arr0 is not None else 0),
            (arr1[:, k].mean() if arr1 is not None else 0),
        )
        ax.text(k, bar_top + y_max * 0.03, star, ha="center", va="bottom",
                fontsize=9, color="#222", fontweight="bold")

    # Legend patch for active/dead
    import matplotlib.patches as mpatches
    active_patch = mpatches.Patch(color="#888888", label="Active slot")
    dead_patch   = mpatches.Patch(color="#cccccc", label="Dead slot (faded)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [active_patch, dead_patch], labels + ["Active slot", "Dead slot"],
              fontsize=8, ncol=2)

    n_active = active_mask.sum()
    sig_slots = [k for k in range(K) if _sig_star(pvals[k])]
    stat_note = f"  |  sig. slots: {sig_slots}" if sig_slots else ""
    ax.set_xlabel("Slot index", fontsize=10)
    ax.set_ylabel("Mean slot usage fraction", fontsize=10)
    ax.set_title(
        f"{mod} — slot activity by label  "
        f"(active: {n_active}/{K}{stat_note})",
        fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{k}{'*' if active_mask[k] else ''}" for k in range(K)],
                        fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out_path}")


def plot_iterative_attn_profile(results, out_path: Path, mod: str = "HE"):
    """
    IterativeXModal: per-block mean self- and cross-attn energy, by label.
    """
    n_blocks = max(
        (int(k.split("_r")[-1]) + 1
         for r in results
         for k in r if k.startswith(f"self_attn_{mod}_r")),
        default=0)
    if n_blocks == 0: return

    fig, axes = plt.subplots(1, n_blocks, figsize=(5 * n_blocks, 4), squeeze=False)
    for r_idx in range(n_blocks):
        ax = axes[0][r_idx]
        for lab, color, name in [(0, "#4e79a7", "No rej"), (1, "#e15759", "Rejection")]:
            sa_vals = [r[f"self_attn_{mod}_r{r_idx}"].mean()
                       for r in results
                       if r.get("label") == lab and f"self_attn_{mod}_r{r_idx}" in r]
            ca_vals = [r[f"cross_attn_{mod}_r{r_idx}"].mean()
                       for r in results
                       if r.get("label") == lab and f"cross_attn_{mod}_r{r_idx}" in r]
            if sa_vals: ax.scatter([r_idx - 0.1] * len(sa_vals), sa_vals,
                                   c=color, s=15, alpha=0.5, marker="o", label=f"{name} self")
            if ca_vals: ax.scatter([r_idx + 0.1] * len(ca_vals), ca_vals,
                                   c=color, s=15, alpha=0.5, marker="^", label=f"{name} cross")
        ax.set_title(f"Block {r_idx}"); ax.set_xlabel("Block"); ax.set_ylabel("Mean attn energy")
        ax.legend(fontsize=7)
    fig.suptitle(f"{mod} — self vs cross attention energy per iterative block",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out_path}")


def plot_top_instances_umap(results, out_path: Path, mod: str = "HE",
                            top_k: int = 8, max_pts: int = 4000):
    if not HAS_UMAP: return
    all_h, all_alpha, all_grad, all_label = [], [], [], []
    for r in results:
        h = r.get(f"h_{mod}"); alpha = r.get(f"alpha_{mod}"); grad = r.get(f"grad_imp_{mod}")
        if h is None or alpha is None: continue
        idx = np.argsort(alpha)[-top_k:]
        all_h.append(h[idx]); all_alpha.append(alpha[idx])
        all_grad.append(grad[idx] if grad is not None else np.zeros(len(idx)))
        all_label.extend([r.get("label", -1)] * len(idx))

    if not all_h: return
    H = np.concatenate(all_h); A = np.concatenate(all_alpha)
    G = np.concatenate(all_grad); L = np.array(all_label)
    if len(H) > max_pts:
        idx = np.random.choice(len(H), max_pts, replace=False)
        H, A, G, L = H[idx], A[idx], G[idx], L[idx]
    print(f"  UMAP {len(H)} {mod} instances ...", flush=True)
    emb = UMAPTransform(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1).fit_transform(H)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (lab, c, name) in zip([axes[0]] * 3,
                                   [(0, "#4e79a7", "No rejection"),
                                    (1, "#e15759", "Rejection"),
                                    (-1, "gray", "?")]):
        m = (L == lab)
        if m.any(): axes[0].scatter(emb[m, 0], emb[m, 1], c=c, s=6, alpha=0.5, label=name)
    axes[0].legend(markerscale=3, fontsize=8); axes[0].set_title(f"{mod} — by label")
    sc1 = axes[1].scatter(emb[:, 0], emb[:, 1], c=A, cmap="magma", s=6, alpha=0.6)
    plt.colorbar(sc1, ax=axes[1], label="instance importance α")
    axes[1].set_title(f"{mod} — importance")
    sc2 = axes[2].scatter(emb[:, 0], emb[:, 1], c=G, cmap="viridis", s=6, alpha=0.6)
    plt.colorbar(sc2, ax=axes[2], label="Grad×α")
    axes[2].set_title(f"{mod} — gradient attribution")
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"UMAP top-{top_k} {mod} instances per patient", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out_path}")


def plot_joint_high_attention(results, out_path: Path, mod: str = "HE", thr: float = 0.9):
    fracs = {0: [], 1: []}
    for r in results:
        lab = r.get("label", -1)
        if lab not in (0, 1): continue
        a = r.get(f"alpha_{mod}"); g = r.get(f"grad_imp_{mod}")
        if a is None or g is None: continue
        fracs[lab].append(float(((a >= np.quantile(a, thr)) & (g >= np.quantile(g, thr))).mean()))
    fig, ax = plt.subplots(figsize=(5, 4))
    for lab, color, name in [(0, "#4e79a7", "No rejection"), (1, "#e15759", "Rejection")]:
        vals = fracs[lab]
        if not vals: continue
        ax.boxplot(vals, positions=[lab], patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.7),
                   medianprops=dict(color="black", lw=2), widths=0.4)
    ax.set_xticks([0, 1]); ax.set_xticklabels([LABEL_NAMES[0], LABEL_NAMES[1]])
    ax.set_ylabel(f"Jointly high-α & high-grad fraction")
    ax.set_title(f"{mod} — joint importance (top {int(thr*100)}%)")
    fig.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════
# SURVIVAL ANALYSIS PLOTS
# ══════════════════════════════════════════════════════════════════

def plot_kaplan_meier(results: List[dict], out_path: Path,
                      endpoint: str = 'clad', n_groups: int = 2):
    """
    Kaplan-Meier curves split by predicted hazard score.
    Patients are split at the median hazard into low/high risk groups.
    Uses lifelines if available, else manual step-function KM.

    Parameters
    ----------
    results    : list of result dicts (each with 'hazard_score', 'surv_{ep}_time',
                 'surv_{ep}_event' fields)
    out_path   : output PNG path
    endpoint   : 'clad' or 'death'
    n_groups   : number of risk groups (default 2 = low/high by median)
    """
    time_key  = f"surv_{endpoint}_time"
    event_key = f"surv_{endpoint}_event"

    # Collect records with valid survival + hazard data
    valid = []
    for r in results:
        h  = r.get("hazard_score")
        t  = r.get(time_key)
        e  = r.get(event_key)
        if h is None or t is None or e is None: continue
        try:
            t_f = float(t); e_f = float(e)
        except (TypeError, ValueError):
            continue
        if t_f != t_f or e_f != e_f: continue  # NaN check
        valid.append((h, t_f, e_f))

    if len(valid) < 4:
        print(f"  [KM] Not enough valid records for endpoint={endpoint} (n={len(valid)})")
        return

    hazards = np.array([v[0] for v in valid])
    times   = np.array([v[1] for v in valid])
    events  = np.array([v[2] for v in valid])

    # Split at median hazard
    median_h = np.median(hazards)
    groups   = (hazards >= median_h).astype(int)  # 0=low, 1=high risk

    fig, ax = plt.subplots(figsize=(8, 5))

    colors     = ["#4e79a7", "#e15759"]
    grp_labels = ["Low risk (hazard < median)", "High risk (hazard >= median)"]

    try:
        from lifelines import KaplanMeierFitter
        at_risk_tables = []
        for grp_idx, (color, lab) in enumerate(zip(colors, grp_labels)):
            mask = groups == grp_idx
            if mask.sum() < 2: continue
            kmf = KaplanMeierFitter()
            kmf.fit(times[mask], event_observed=events[mask], label=lab)
            kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2)
            at_risk_tables.append((lab, mask.sum()))
        ax.set_xlabel(f"Time (days) — {endpoint.upper()}", fontsize=11)
        ax.set_ylabel("Survival probability", fontsize=11)
    except ImportError:
        # Manual KM step function
        def _km_step(t_arr, e_arr):
            """Returns (times, survival) for KM step function."""
            order = np.argsort(t_arr)
            t_s = t_arr[order]; e_s = e_arr[order]
            t_uniq = np.unique(t_s)
            surv = 1.0
            km_t = [0]; km_s = [1.0]
            n = len(t_s)
            at_risk = n
            for tu in t_uniq:
                d = int(e_s[t_s == tu].sum())   # events at time tu
                n_at = int((t_s >= tu).sum())   # at risk just before tu
                if n_at > 0 and d > 0:
                    surv = surv * (1.0 - d / n_at)
                km_t.append(tu); km_s.append(surv)
                at_risk -= int((t_s == tu).sum())
            return np.array(km_t), np.array(km_s)

        for grp_idx, (color, lab) in enumerate(zip(colors, grp_labels)):
            mask = groups == grp_idx
            if mask.sum() < 2: continue
            km_t, km_s = _km_step(times[mask], events[mask])
            ax.step(km_t, km_s, where="post", color=color, linewidth=2, label=lab)
        ax.set_xlabel(f"Time (days) — {endpoint.upper()}", fontsize=11)
        ax.set_ylabel("Survival probability", fontsize=11)

    # Number at risk table below plot
    n_low  = int((groups == 0).sum())
    n_high = int((groups == 1).sum())
    ev_low  = int(events[groups == 0].sum())
    ev_high = int(events[groups == 1].sum())
    ax.text(0.02, 0.04,
            f"Low risk n={n_low} (events={ev_low})   High risk n={n_high} (events={ev_high})",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(facecolor="lightyellow", alpha=0.7, edgecolor="gray", boxstyle="round"))

    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.set_title(f"Kaplan-Meier — {endpoint.upper()} by predicted hazard\n"
                 f"Split at median hazard = {median_h:.3f}  (n={len(valid)} patients)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out_path}")


def plot_risk_stratification(results: List[dict], out_path: Path,
                             endpoint: str = 'clad'):
    """
    Two-panel cohort-level risk stratification plot:
      Panel 1: Box/violin plot of hazard scores by classification label (rejection vs no-rejection)
      Panel 2: Scatter plot of hazard score vs time-to-event, coloured by event/censored status
    """
    hazards, labels, times, events = [], [], [], []
    time_key  = f"surv_{endpoint}_time"
    event_key = f"surv_{endpoint}_event"

    for r in results:
        h  = r.get("hazard_score")
        lb = r.get("label", -1)
        if h is None: continue
        hazards.append(float(h)); labels.append(int(lb))
        t_val = r.get(time_key)
        e_val = r.get(event_key)
        try:
            t_f = float(t_val) if t_val is not None else float("nan")
            e_f = float(e_val) if e_val is not None else float("nan")
        except (TypeError, ValueError):
            t_f = e_f = float("nan")
        times.append(t_f); events.append(e_f)

    if len(hazards) < 2:
        print(f"  [risk_strat] Not enough records (n={len(hazards)})")
        return

    hazards = np.array(hazards)
    labels  = np.array(labels)
    times   = np.array(times)
    events  = np.array(events)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: violin/box by classification label
    ax1 = axes[0]
    for lab_idx, (lab_name, color) in enumerate([(0, "#4e79a7"), (1, "#e15759")]):
        mask = labels == lab_idx
        if mask.sum() == 0: continue
        parts = ax1.violinplot(hazards[mask], positions=[lab_idx], widths=0.7, showmedians=True)
        for pc in parts.get("bodies", []):
            pc.set_facecolor(color); pc.set_alpha(0.6)
        for part_name in ("cmedians", "cmaxes", "cmins", "cbars"):
            p = parts.get(part_name)
            if p is not None: p.set_edgecolor(color)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([LABEL_NAMES.get(0, "No rejection"), LABEL_NAMES.get(1, "Rejection")],
                        fontsize=10)
    ax1.set_ylabel("Predicted hazard score", fontsize=10)
    ax1.set_title("Risk by rejection label", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: scatter hazard vs time, coloured by event status
    ax2 = axes[1]
    valid_surv = ~np.isnan(times) & ~np.isnan(events)
    if valid_surv.sum() >= 2:
        ev_mask = valid_surv & (events == 1)
        cen_mask = valid_surv & (events != 1)
        if ev_mask.sum() > 0:
            ax2.scatter(times[ev_mask], hazards[ev_mask],
                        c="#e15759", s=30, alpha=0.7, label=f"Event (n={ev_mask.sum()})",
                        edgecolors="none", marker="o")
        if cen_mask.sum() > 0:
            ax2.scatter(times[cen_mask], hazards[cen_mask],
                        c="#4e79a7", s=20, alpha=0.5, label=f"Censored (n={cen_mask.sum()})",
                        edgecolors="none", marker="^")
        ax2.set_xlabel(f"Time to {endpoint.upper()} (days)", fontsize=10)
        ax2.set_ylabel("Predicted hazard score", fontsize=10)
        ax2.set_title(f"Hazard vs time-to-event ({endpoint.upper()})", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, f"No survival data for endpoint={endpoint}",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11)
        ax2.set_visible(True)

    fig.suptitle(f"Risk stratification — predicted hazard  (n={len(hazards)} patients)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════
# CLUSTER-LEVEL INTERPRETABILITY PLOTS
# ══════════════════════════════════════════════════════════════════

def plot_abmil_by_cluster(results: List[dict], out_dir: Path):
    """
    ABMIL attention averaged per cluster/cell type, split ACR vs Non-ACR.
    Bar chart: which cell types receive highest attention in each group?
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for mod in CLUSTER_MODS:
        by_ctype: Dict[str, Dict[int, List[float]]] = {}
        all_ctypes: set = set()

        for r in results:
            lab = r.get("label", -1)
            if lab not in (0, 1): continue
            alpha = r.get(f"alpha_{mod}")
            cl    = r.get(f"cluster_labels_{mod}")
            if alpha is None or cl is None: continue
            n = min(len(alpha), len(cl))
            for i in range(n):
                ct = cl[i]
                all_ctypes.add(ct)
                by_ctype.setdefault(ct, {0: [], 1: []})[lab].append(float(alpha[i]))

        if not by_ctype: continue
        ctypes = sorted(all_ctypes)
        x = np.arange(len(ctypes)); w = 0.38

        fig, ax = plt.subplots(figsize=(max(10, len(ctypes) * 0.55 + 2), 5))
        for lab, color, off, name in [(0, "#5c9be0", -w/2, "Non-ACR"),
                                      (1, "#e05c5c",  w/2, "ACR")]:
            means = [np.mean(by_ctype[ct][lab]) if by_ctype.get(ct, {}).get(lab) else 0.0
                     for ct in ctypes]
            sems  = [np.std(by_ctype[ct][lab]) / max(len(by_ctype[ct][lab])**.5, 1)
                     if by_ctype.get(ct, {}).get(lab) else 0.0 for ct in ctypes]
            ax.bar(x + off, means, w, color=color, alpha=0.85, label=name)
            ax.errorbar(x + off, means, sems, fmt="none", color="black", capsize=3, lw=0.8)

        ax.set_xticks(x); ax.set_xticklabels(ctypes, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean ABMIL attention per cluster type", fontsize=10)
        ax.set_title(f"{mod} — ABMIL attention by cluster/cell type (ACR vs Non-ACR)", fontsize=10)
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"abmil_by_cluster_{mod}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  → {p}")


def plot_slot_cluster_contribution(results: List[dict], out_dir: Path,
                                    slot_thr: Optional[float] = None,
                                    min_active_slots: int = 3,
                                    mark_cell_thr: Optional[float] = None):
    """
    K×C slot-cluster contribution heatmaps (post- and pre-softmax).
    Layer 1: only active slots shown as rows.
    Layer 2: cells below mark_cell_thr greyed out.
    Layer 3: significance stars per slot (Mann-Whitney across labels).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from scipy.stats import mannwhitneyu
        _has_stats = True
    except ImportError:
        _has_stats = False

    def _sig_star(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    for mod in CLUSTER_MODS:
        all_ctypes: Optional[List[str]] = None
        for r in results:
            A = r.get(f"slot_assign_{mod}"); cl = r.get(f"cluster_labels_{mod}")
            if A is None or cl is None: continue
            n = min(A.shape[1], len(cl))
            ctypes = sorted(set(cl[:n]))
            if all_ctypes is None or len(ctypes) > len(all_ctypes):
                all_ctypes = ctypes
        if all_ctypes is None: continue

        kc_post: Dict[int, List[np.ndarray]] = {0: [], 1: []}
        kc_pre:  Dict[int, List[np.ndarray]] = {0: [], 1: []}
        K_ref = None

        # Also collect raw A matrices to build cohort active-slot mask
        all_A_max: List[np.ndarray] = []

        for r in results:
            lab = r.get("label", -1)
            A_post = r.get(f"slot_assign_{mod}")
            A_pre  = r.get(f"slot_logits_{mod}")
            cl     = r.get(f"cluster_labels_{mod}")
            if A_post is None or cl is None: continue
            if K_ref is None: K_ref = A_post.shape[0]
            all_A_max.append(A_post.max(axis=1))   # (K,)
            if lab not in (0, 1): continue
            kc, _ = _slot_cluster_matrix(A_post, cl, all_types=all_ctypes)
            kc_post[lab].append(kc)
            if A_pre is not None:
                kc2, _ = _slot_cluster_matrix(A_pre, cl, all_types=all_ctypes)
                kc_pre[lab].append(kc2)

        if K_ref is None or (not kc_post[0] and not kc_post[1]): continue
        C = len(all_ctypes)

        # Layer 1 — cohort-level active slot mask
        if all_A_max:
            cohort_max = np.stack(all_A_max).mean(0)
            thr = slot_thr if slot_thr is not None else 2.0 / max(K_ref, 1)
            active_mask = cohort_max > thr
            if active_mask.sum() < min_active_slots:
                top = np.argsort(cohort_max)[-min_active_slots:]
                active_mask = np.zeros(K_ref, dtype=bool); active_mask[top] = True
        else:
            active_mask = np.ones(K_ref, dtype=bool)
        active_idx = np.where(active_mask)[0]

        # Layer 3 — per-slot significance (Mann-Whitney on slot usage fraction)
        pvals = np.ones(K_ref)
        if _has_stats and kc_post[0] and kc_post[1]:
            m0_stack = np.stack([m for m in kc_post[0] if m.shape == (K_ref, C)])
            m1_stack = np.stack([m for m in kc_post[1] if m.shape == (K_ref, C)])
            for k in range(K_ref):
                # test on mean cluster contribution per slot (collapsed over C)
                v0 = m0_stack[:, k, :].mean(1)
                v1 = m1_stack[:, k, :].mean(1)
                try:
                    _, pvals[k] = mannwhitneyu(v0, v1, alternative="two-sided")
                except Exception:
                    pass

        for prefix, kc_dict, title_pf in [
            ("post", kc_post, "Post-softmax (slot competition)"),
            ("pre",  kc_pre,  "Pre-softmax (raw logits)"),
        ]:
            m0 = [m for m in kc_dict[0] if m.shape == (K_ref, C)]
            m1 = [m for m in kc_dict[1] if m.shape == (K_ref, C)]
            if not m0 and not m1: continue

            panels = []
            if m0: panels.append((np.stack(m0).mean(0), "Non-ACR"))
            if m1: panels.append((np.stack(m1).mean(0), "ACR"))
            if m0 and m1:
                panels.append((np.stack(m1).mean(0) - np.stack(m0).mean(0), "ACR − Non-ACR"))

            n_active = len(active_idx)
            fig, axes = plt.subplots(1, len(panels),
                                     figsize=(7 * len(panels), max(4, n_active * 0.5 + 2)),
                                     squeeze=False)
            for ax, (mat_full, ptitle) in zip(axes[0], panels):
                is_diff = "−" in ptitle
                # Layer 1 — filter to active slot rows only
                mat = mat_full[active_idx, :]
                vmax = np.abs(mat).max() + 1e-8
                cmap_name = "RdBu_r" if is_diff else "YlOrRd"
                vmin = -vmax if is_diff else 0

                # Layer 2 — cell significance mask
                if mark_cell_thr is not None and not is_diff:
                    mat_plot = np.where(mat >= mark_cell_thr, mat, np.nan)
                    cmap_obj = plt.get_cmap(cmap_name).copy()
                    cmap_obj.set_bad(color="whitesmoke")
                else:
                    mat_plot = mat
                    cmap_obj = cmap_name

                im = ax.imshow(mat_plot, aspect="auto", cmap=cmap_obj,
                               vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_xticks(range(C))
                ax.set_xticklabels(all_ctypes, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(n_active))
                # Layer 3 — star annotation on y-tick labels
                ylabels = []
                for k in active_idx:
                    star = _sig_star(pvals[k])
                    ylabels.append(f"S{k} {star}" if star else f"S{k}")
                ax.set_yticklabels(ylabels, fontsize=7)
                ax.set_xlabel("Cluster / Feature type", fontsize=9)
                ax.set_ylabel("Active slot", fontsize=9)
                ax.set_title(ptitle, fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046, label="mean weight")

            cell_note = f"  |  cells<{mark_cell_thr} masked" if mark_cell_thr else ""
            fig.suptitle(
                f"{mod} — Slot × Cluster  [{title_pf}]\n"
                f"Active: {n_active}/{K_ref} slots × C={C} types{cell_note}",
                fontsize=11, fontweight="bold")
            fig.tight_layout()
            p = out_dir / f"slot_cluster_{mod}_{prefix}softmax.png"
            fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"  → {p}")


def plot_self_attn_coattn(results: List[dict], out_dir: Path):
    """
    C×C cluster co-attention for IterativeXModal self-attention (last block).
    Shows which cell types attend to which other cell types within each modality.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for mod in CLUSTER_MODS:
        all_ctypes: Optional[List[str]] = None
        for r in results:
            cl = r.get(f"cluster_labels_{mod}")
            mat = r.get(f"self_attn_matrix_{mod}")
            if cl is None or mat is None: continue
            n = min(mat.shape[0], len(cl))
            ctypes = sorted(set(cl[:n]))
            if all_ctypes is None or len(ctypes) > len(all_ctypes):
                all_ctypes = ctypes
        if all_ctypes is None: continue

        C = len(all_ctypes)
        cc_by: Dict[int, List[np.ndarray]] = {0: [], 1: []}

        for r in results:
            lab = r.get("label", -1)
            if lab not in (0, 1): continue
            cl  = r.get(f"cluster_labels_{mod}")
            mat = r.get(f"self_attn_matrix_{mod}")
            if cl is None or mat is None: continue
            N_cap = mat.shape[0]
            n = min(N_cap, len(cl))
            labs_n = cl[:n]
            idx_map = {ct: i for i, ct in enumerate(all_ctypes)}
            cc = np.zeros((C, C), dtype=np.float32)
            cnt = np.zeros((C, C), dtype=np.int32)
            for qi in range(n):
                ci = idx_map.get(labs_n[qi])
                if ci is None: continue
                for kj in range(n):
                    cj = idx_map.get(labs_n[kj])
                    if cj is None: continue
                    cc[ci, cj] += mat[qi, kj]
                    cnt[ci, cj] += 1
            mask = cnt > 0
            cc[mask] /= cnt[mask]
            cc_by[lab].append(cc)

        m0 = cc_by[0]; m1 = cc_by[1]
        panels = []
        if m0: panels.append((np.stack(m0).mean(0), "Non-ACR"))
        if m1: panels.append((np.stack(m1).mean(0), "ACR"))
        if m0 and m1: panels.append((np.stack(m1).mean(0) - np.stack(m0).mean(0), "ACR − Non-ACR"))
        if not panels: continue

        fig, axes = plt.subplots(1, len(panels),
                                 figsize=(5 * len(panels), max(4, C * 0.4 + 2)),
                                 squeeze=False)
        for ax, (mat, ptitle) in zip(axes[0], panels):
            is_diff = "−" in ptitle
            vmax = np.abs(mat).max() + 1e-8
            cmap = "RdBu_r" if is_diff else "YlOrRd"
            vmin = -vmax if is_diff else 0
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(C)); ax.set_xticklabels(all_ctypes, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(C)); ax.set_yticklabels(all_ctypes, fontsize=7)
            ax.set_xlabel("Key cluster / feature", fontsize=9); ax.set_ylabel("Query cluster / feature", fontsize=9)
            ax.set_title(ptitle, fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle(f"{mod} — Self-attention cluster co-attention (last iterative block)\n"
                     f"C={C}: which cell types attend to which cell types?",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        p = out_dir / f"self_attn_coattn_{mod}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  → {p}")


def plot_cross_modal_cluster_attn(results: List[dict], out_dir: Path):
    """
    Effective C_a × C_b cluster-to-cluster cross-modal attention.
    For each modality pair: slot_cluster_a.T @ xmodal_attn_ab @ slot_cluster_b.
    Shows which cell type in modality A attends to which cell type in modality B.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mods = CLUSTER_MODS

    for ma in mods:
        for mb in mods:
            if ma == mb: continue
            all_ctypes_a = all_ctypes_b = None
            cc_by: Dict[int, List[np.ndarray]] = {0: [], 1: []}

            for r in results:
                slices = r.get("slot_slices"); xa = r.get("xmodal_attn")
                A_a = r.get(f"slot_assign_{ma}"); A_b = r.get(f"slot_assign_{mb}")
                cl_a = r.get(f"cluster_labels_{ma}"); cl_b = r.get(f"cluster_labels_{mb}")
                if any(v is None for v in [slices, xa, A_a, A_b, cl_a, cl_b]): continue
                sa_e = {m: (s, e) for m, s, e in slices}
                if ma not in sa_e or mb not in sa_e: continue
                sa, ea = sa_e[ma]; sb, eb = sa_e[mb]
                xa_ab = xa[sa:ea, sb:eb]   # (K_a, K_b)
                if all_ctypes_a is None:
                    _, all_ctypes_a = _slot_cluster_matrix(A_a, cl_a)
                    _, all_ctypes_b = _slot_cluster_matrix(A_b, cl_b)

            if all_ctypes_a is None: continue

            for r in results:
                lab = r.get("label", -1)
                if lab not in (0, 1): continue
                slices = r.get("slot_slices"); xa = r.get("xmodal_attn")
                A_a = r.get(f"slot_assign_{ma}"); A_b = r.get(f"slot_assign_{mb}")
                cl_a = r.get(f"cluster_labels_{ma}"); cl_b = r.get(f"cluster_labels_{mb}")
                if any(v is None for v in [slices, xa, A_a, A_b, cl_a, cl_b]): continue
                sa_e = {m: (s, e) for m, s, e in slices}
                if ma not in sa_e or mb not in sa_e: continue
                sa, ea = sa_e[ma]; sb, eb = sa_e[mb]
                xa_ab = xa[sa:ea, sb:eb]
                kc_a, _ = _slot_cluster_matrix(A_a, cl_a, all_types=all_ctypes_a)
                kc_b, _ = _slot_cluster_matrix(A_b, cl_b, all_types=all_ctypes_b)
                # (C_a, K_a) @ (K_a, K_b) @ (K_b, C_b) → (C_a, C_b)
                cc = kc_a.T @ xa_ab @ kc_b
                cc_by[lab].append(cc)

            C_a, C_b = len(all_ctypes_a), len(all_ctypes_b)
            m0, m1 = cc_by[0], cc_by[1]
            panels = []
            if m0: panels.append((np.stack(m0).mean(0), "Non-ACR"))
            if m1: panels.append((np.stack(m1).mean(0), "ACR"))
            if m0 and m1: panels.append((np.stack(m1).mean(0) - np.stack(m0).mean(0), "ACR − Non-ACR"))
            if not panels: continue

            fig, axes = plt.subplots(1, len(panels),
                                     figsize=(6 * len(panels), max(4, C_a * 0.4 + 2)),
                                     squeeze=False)
            for ax, (mat, ptitle) in zip(axes[0], panels):
                is_diff = "−" in ptitle
                vmax = np.abs(mat).max() + 1e-8
                cmap = "RdBu_r" if is_diff else "YlOrRd"
                vmin = -vmax if is_diff else 0
                im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks(range(C_b)); ax.set_xticklabels(all_ctypes_b, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(C_a)); ax.set_yticklabels(all_ctypes_a, fontsize=7)
                ax.set_xlabel(f"{mb} cluster / feature", fontsize=9)
                ax.set_ylabel(f"{ma} cluster / feature", fontsize=9)
                ax.set_title(ptitle, fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046)
            fig.suptitle(f"Cluster-to-cluster cross-modal attention: {ma} → {mb}\n"
                         f"= slot_cluster_{ma}ᵀ · xmodal_attn · slot_cluster_{mb}",
                         fontsize=11, fontweight="bold")
            fig.tight_layout()
            p = out_dir / f"cluster_xmodal_{ma}_{mb}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"  → {p}")


def plot_cross_modal_connection(results: List[dict], out_dir: Path,
                                mod_a: str = "HE", mod_b: str = "BAL"):
    """
    Three-panel connection figure:
      Left  : mod_a slot × cluster contribution (K_a × C_a)
      Center: cross-modal attention (K_a × K_b) — the learned connection
      Right : mod_b cluster × slot contribution (C_b × K_b)

    Reads: which mod_a cell type → which slot → cross-modal → which mod_b slot → which mod_b cell type.
    Shown separately for ACR, Non-ACR, and their difference.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    kc_a_by: Dict[int, List[np.ndarray]] = {0: [], 1: []}
    kc_b_by: Dict[int, List[np.ndarray]] = {0: [], 1: []}
    xa_by:   Dict[int, List[np.ndarray]] = {0: [], 1: []}
    ctypes_a = ctypes_b = None

    # First pass: build common cluster type lists
    for r in results:
        A_a = r.get(f"slot_assign_{mod_a}"); cl_a = r.get(f"cluster_labels_{mod_a}")
        A_b = r.get(f"slot_assign_{mod_b}"); cl_b = r.get(f"cluster_labels_{mod_b}")
        if any(v is None for v in [A_a, A_b, cl_a, cl_b]): continue
        if ctypes_a is None:
            _, ctypes_a = _slot_cluster_matrix(A_a, cl_a)
            _, ctypes_b = _slot_cluster_matrix(A_b, cl_b)
        break
    if ctypes_a is None: return

    for r in results:
        lab = r.get("label", -1)
        if lab not in (0, 1): continue
        slices = r.get("slot_slices"); xa = r.get("xmodal_attn")
        A_a = r.get(f"slot_assign_{mod_a}"); cl_a = r.get(f"cluster_labels_{mod_a}")
        A_b = r.get(f"slot_assign_{mod_b}"); cl_b = r.get(f"cluster_labels_{mod_b}")
        if any(v is None for v in [slices, xa, A_a, A_b, cl_a, cl_b]): continue
        sa_e = {m: (s, e) for m, s, e in slices}
        if mod_a not in sa_e or mod_b not in sa_e: continue
        sa, ea = sa_e[mod_a]; sb, eb = sa_e[mod_b]
        xa_ab = xa[sa:ea, sb:eb]
        kc_a, _ = _slot_cluster_matrix(A_a, cl_a, all_types=ctypes_a)
        kc_b, _ = _slot_cluster_matrix(A_b, cl_b, all_types=ctypes_b)
        kc_a_by[lab].append(kc_a); kc_b_by[lab].append(kc_b); xa_by[lab].append(xa_ab)

    if not kc_a_by[0] and not kc_a_by[1]: return
    all_data = kc_a_by[0] + kc_a_by[1]
    K_a = all_data[0].shape[0]; C_a = len(ctypes_a)
    K_b_all = kc_b_by[0] + kc_b_by[1]
    K_b = K_b_all[0].shape[0]; C_b = len(ctypes_b)

    def _mean_or_none(lst, K, C):
        filtered = [m for m in lst if m.shape == (K, C)]
        return np.stack(filtered).mean(0) if filtered else None

    for lab_id, name in [(0, "non_acr"), (1, "acr"), (2, "diff")]:
        if lab_id == 2:
            m_a0 = _mean_or_none(kc_a_by[0], K_a, C_a)
            m_a1 = _mean_or_none(kc_a_by[1], K_a, C_a)
            m_b0 = _mean_or_none(kc_b_by[0], K_b, C_b)
            m_b1 = _mean_or_none(kc_b_by[1], K_b, C_b)
            xa0  = _mean_or_none(xa_by[0], K_a, K_b)
            xa1  = _mean_or_none(xa_by[1], K_a, K_b)
            if any(v is None for v in [m_a0, m_a1, m_b0, m_b1, xa0]): continue
            kc_a_plot = m_a1 - m_a0
            kc_b_plot = m_b1 - m_b0
            xa_plot   = (xa0 + (xa1 if xa1 is not None else xa0)) / 2
            title_str = f"ACR − Non-ACR"
            cmap_kc = "RdBu_r"
        else:
            m_a = _mean_or_none(kc_a_by[lab_id], K_a, C_a)
            m_b = _mean_or_none(kc_b_by[lab_id], K_b, C_b)
            xa  = _mean_or_none(xa_by[lab_id], K_a, K_b)
            if any(v is None for v in [m_a, m_b, xa]): continue
            kc_a_plot = m_a; kc_b_plot = m_b; xa_plot = xa
            title_str = "Non-ACR" if lab_id == 0 else "ACR"
            cmap_kc = "YlOrRd"

        fig_w = max(14, C_a * 0.4 + K_a * 0.6 + C_b * 0.4 + 3)
        fig_h = max(5, max(K_a, K_b) * 0.6)
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs  = gridspec.GridSpec(1, 3, figure=fig,
                                width_ratios=[C_a, max(K_a, K_b), C_b], wspace=0.45)

        # Left: mod_a slot × cluster
        ax_l = fig.add_subplot(gs[0])
        vmax = np.abs(kc_a_plot).max() + 1e-8
        im_l = ax_l.imshow(kc_a_plot, aspect="auto", cmap=cmap_kc,
                           vmin=-vmax if cmap_kc == "RdBu_r" else 0, vmax=vmax)
        ax_l.set_xticks(range(C_a)); ax_l.set_xticklabels(ctypes_a, rotation=45, ha="right", fontsize=6)
        ax_l.set_yticks(range(K_a)); ax_l.set_yticklabels([f"S{k}" for k in range(K_a)], fontsize=7)
        ax_l.set_title(f"{mod_a}\n slot ← cluster", fontsize=9)
        ax_l.set_ylabel("Slot", fontsize=8); plt.colorbar(im_l, ax=ax_l, fraction=0.08)

        # Center: cross-modal attention K_a × K_b
        ax_c = fig.add_subplot(gs[1])
        im_c = ax_c.imshow(xa_plot, aspect="auto", cmap="Blues", vmin=0)
        ax_c.set_xticks(range(K_b)); ax_c.set_xticklabels([f"{k}" for k in range(K_b)], fontsize=7)
        ax_c.set_yticks(range(K_a)); ax_c.set_yticklabels([f"{k}" for k in range(K_a)], fontsize=7)
        ax_c.set_xlabel(f"{mod_b} slots", fontsize=8); ax_c.set_ylabel(f"{mod_a} slots", fontsize=8)
        ax_c.set_title(f"Cross-modal\nattn", fontsize=9)
        plt.colorbar(im_c, ax=ax_c, fraction=0.08)

        # Right: mod_b cluster × slot (transposed so rows=clusters, cols=slots)
        ax_r = fig.add_subplot(gs[2])
        vmax_r = np.abs(kc_b_plot).max() + 1e-8
        im_r = ax_r.imshow(kc_b_plot.T, aspect="auto", cmap=cmap_kc,
                           vmin=-vmax_r if cmap_kc == "RdBu_r" else 0, vmax=vmax_r)
        ax_r.set_yticks(range(C_b)); ax_r.set_yticklabels(ctypes_b, fontsize=6)
        ax_r.set_xticks(range(K_b)); ax_r.set_xticklabels([f"S{k}" for k in range(K_b)], fontsize=7)
        ax_r.set_title(f"{mod_b}\n cluster ← slot", fontsize=9)
        ax_r.set_xlabel("Slot", fontsize=8); plt.colorbar(im_r, ax=ax_r, fraction=0.08)

        fig.suptitle(f"{mod_a} cells → slots ─→ cross-modal ─→ slots → {mod_b} cells  [{title_str}]\n"
                     f"Indirect: which {mod_a} cell types interact with which {mod_b} cell types?",
                     fontsize=11, fontweight="bold")
        p = out_dir / f"cross_modal_connection_{mod_a}_{mod_b}_{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  → {p}")


def plot_modality_classifier_weights(results: List[dict], out_path: Path):
    """
    Late-fusion only: modal_weights show learned classifier importance per modality.
    Bar chart per label shows which modality drives the decision for ACR vs Non-ACR.
    """
    by_lab: Dict[int, Dict[str, List[float]]] = {0: {}, 1: {}}
    for r in results:
        lab = r.get("label", -1)
        if lab not in (0, 1): continue
        mw = r.get("modal_weights")
        if mw is None: continue
        for mod, w in mw.items():
            by_lab[lab].setdefault(mod, []).append(float(w))
    if not any(any(v.values()) for v in by_lab.values()): return

    all_mods = sorted(set(m for v in by_lab.values() for m in v))
    x = np.arange(len(all_mods)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, len(all_mods) * 1.2), 4))
    for lab, color, off, name in [(0, "#5c9be0", -w/2, "Non-ACR"), (1, "#e05c5c", w/2, "ACR")]:
        means = [np.mean(by_lab[lab].get(m, [0])) for m in all_mods]
        sems  = [np.std(by_lab[lab].get(m, [0])) / max(len(by_lab[lab].get(m, [0]))**.5, 1)
                 for m in all_mods]
        ax.bar(x + off, means, w, color=color, alpha=0.85, label=name)
        ax.errorbar(x + off, means, sems, fmt="none", color="black", capsize=3, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(all_mods, fontsize=10)
    ax.set_ylabel("Learned modality weight (softmax)", fontsize=10)
    ax.set_title("Classifier modality importance (Late fusion)\nACR vs Non-ACR", fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════

def _parse_v6_variant(variant: str):
    """
    Parse a checkpoint variant name into (base_variant, slot_k, iter_r).
    e.g. 'iterative_r4_k16_cls' -> ('iterative_cls', 16, 4)
         'crossattn_k8'          -> ('crossattn',      8, None)
         'middle_cls'            -> ('middle_cls',   None, None)
    """
    import re
    km = re.search(r'_k(\d+)', variant)
    rm = re.search(r'_r(\d+)', variant)
    slot_k = int(km.group(1)) if km else None
    iter_r = int(rm.group(1)) if rm else None
    use_cls = variant.endswith("_cls")

    for prefix in ("crossattn", "crossmodal", "iterative", "early", "late", "middle"):
        if variant.startswith(prefix):
            base = prefix + ("_cls" if use_cls else "")
            return base, slot_k, iter_r

    return variant, slot_k, iter_r   # fallback: pass as-is


def load_v6_model(results_dir: Path, fold_tag: str, variant: str) -> nn.Module:
    ckpt  = results_dir / fold_tag / "phase2" / f"model_{variant}.pt"
    assert ckpt.exists(), f"Missing: {ckpt}"
    p1    = results_dir / fold_tag / "phase1"

    base, slot_k, iter_r = _parse_v6_variant(variant)
    kwargs = {}
    if slot_k is not None: kwargs["slot_k"] = slot_k
    if iter_r is not None: kwargs["iter_r"] = iter_r

    model = build_p2_model(variant=base, p1_dir=p1, **kwargs)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state: state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    # hazard_head keys will be missing in old checkpoints — that is fine;
    # interpretability code guards with hasattr(model, 'hazard_head')
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:5]}")
    model.to(DEVICE).eval()
    print(f"  Loaded v6 [{variant}] (base={base} slot_k={slot_k} iter_r={iter_r}) from {ckpt}")
    return model


def load_v7_model(results_dir: Path, fold_tag: str, tag: str, phase1_dir: Path) -> nn.Module:
    ckpt  = results_dir / fold_tag / f"model_{tag}.pt"
    assert ckpt.exists(), f"Missing: {ckpt}"
    model = build_v7_model(p1_dir=phase1_dir / fold_tag / "phase1", use_cls=("cls" in tag))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state: state = state["model"]
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    print(f"  Loaded v7 [{tag}] from {ckpt}")
    return model


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Interpretability for ABMIL v6/v7 — all variants")
    p.add_argument("--version",        choices=["v6", "v7"], default="v7")
    p.add_argument("--results_dir",    required=True)
    p.add_argument("--phase1_dir",     default=None, help="v7 only")
    p.add_argument("--split",          type=int, required=True)
    p.add_argument("--fold",           type=int, required=True)
    p.add_argument("--tag",            default="v7_triple", help="v7 model tag")
    p.add_argument("--v6_variant",     default="middle",
                   help="v6 variant: early|late|middle|crossattn|crossmodal|iterative + suffixes")
    p.add_argument("--out_dir",        required=True)
    p.add_argument("--samples_dir",    default=SAMPLES_DIR)
    p.add_argument("--splits_csv",     default=SPLITS_CSV)
    p.add_argument("--top_k",          type=int, default=16)
    p.add_argument("--n_sample_plots", type=int, default=20)
    p.add_argument("--umap_mod",       default="HE")
    p.add_argument("--split_set",      default="test",
                   choices=["train", "val", "test", "all"])
    p.add_argument("--task",           default="acr", choices=["acr", "survival"],
                   help="Task type: acr=classification, survival=Cox hazard")
    p.add_argument("--surv_endpoint",  default="death", choices=["death", "clad"],
                   help="Survival endpoint used during training (only used when --task survival)")
    return p.parse_args()


def main():
    args     = parse_args()
    set_seeds(SEED)
    out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_sample").mkdir(exist_ok=True)

    fold_tag = f"split{args.split}_fold{args.fold}"
    variant  = args.v6_variant if args.version == "v6" else args.tag
    print(f"\n{'='*65}")
    print(f"  Interpretability  {args.version.upper()}  [{variant}]  {fold_tag}")
    print(f"  Task: {args.task}" + (f"  endpoint: {args.surv_endpoint}" if args.task == "survival" else ""))
    print(f"{'='*65}\n")

    if args.version == "v7":
        assert args.phase1_dir, "--phase1_dir required for v7"
        model = load_v7_model(Path(args.results_dir), fold_tag, args.tag, Path(args.phase1_dir))
    else:
        model = load_v6_model(Path(args.results_dir), fold_tag, args.v6_variant)

    import pandas as pd
    df_csv = pd.read_csv(args.splits_csv)
    if args.task == "survival":
        # Survival: include all patients with valid endpoint data (no ACR grade filter)
        ep = args.surv_endpoint
        valid_stems = list({Path(str(r["file"])).stem for _, r in df_csv.iterrows()
                            if pd.notna(r.get(f"{ep}_status"))})
        train_r, val_r, test_r = build_splits_survival(
            args.samples_dir, args.splits_csv, args.fold,
            split=args.split, endpoint=ep)
    else:
        valid_stems = list({Path(str(r["file"])).stem for _, r in df_csv.iterrows()
                            if acr_label(r.get("acr_grade")) is not None})
        train_r, val_r, test_r = build_splits(args.samples_dir, args.splits_csv,
                                               args.fold, split=args.split)

    records = ({"all": train_r + val_r + test_r, "train": train_r,
                "val": val_r, "test": test_r})[args.split_set]

    # Only preload bags for the records we'll actually process — avoids OOM
    # when the full cohort is large (e.g. death survival has 4000+ patients).
    valid_stems = list({r["stem"] for r in records})

    print(f"  Records ({args.split_set}): {len(records)}")
    print("  Preloading bags ...")
    bag_cache = preload_bags(valid_stems, args.samples_dir); _gc6()

    if args.version == "v7":
        print("  Preloading annotations ...")
        annot_cache = preload_annotations(valid_stems, args.samples_dir); _gc6()
    else:
        annot_cache = {s: {m: None for m in ANNOT_MODS} for s in valid_stems}

    update_presence_from_cache(records, bag_cache)

    print("\n  Extracting ...")
    results = run_extraction(model, records, bag_cache, annot_cache, DEVICE, out_dir,
                             samples_dir=args.samples_dir)
    print(f"  {len(results)} samples extracted.\n")

    rng     = np.random.default_rng(42)
    sampled = rng.choice(len(results), min(args.n_sample_plots, len(results)), replace=False)
    print("  Per-sample plots ...")
    for idx in sampled:
        r = results[idx]
        plot_sample(r, out_dir / "per_sample" / f"{r.get('stem', idx)}_attn.png",
                    top_k=args.top_k)

    print("  Cohort plots ...")
    plot_cross_modal_attention(results, out_dir / "cross_modal_attention.png")
    plot_modal_contributions(results, out_dir / "modal_contribution.png")
    plot_centroid_importance(results, out_dir / "centroid_importance.png")
    plot_modality_classifier_weights(results, out_dir / "modality_classifier_weights.png")
    # Survival analysis plots (only plotted if hazard_score is present in results)
    if any(r.get("hazard_score") is not None for r in results):
        print("  Survival analysis plots ...")
        plot_kaplan_meier(results, out_dir / "kaplan_meier_clad.png", endpoint="clad")
        plot_kaplan_meier(results, out_dir / "kaplan_meier_death.png", endpoint="death")
        plot_risk_stratification(results, out_dir / "risk_stratification_clad.png", endpoint="clad")
        plot_risk_stratification(results, out_dir / "risk_stratification_death.png", endpoint="death")
    plot_top_instances_umap(results, out_dir / "top_instances_umap.png",
                            mod=args.umap_mod, top_k=args.top_k)
    plot_joint_high_attention(results, out_dir / "joint_high_attention.png", mod=args.umap_mod)
    plot_slot_assignment_summary(results, out_dir / "slot_assignment_summary.png", mod=args.umap_mod)
    plot_iterative_attn_profile(results, out_dir / "iterative_attn_profile.png", mod=args.umap_mod)
    print("  Clinical × Cluster cross-attention ...")
    plot_clinical_cluster_attn(results, out_dir / "clinical_cluster_attn")
    print("  Cluster-level interpretability ...")
    plot_abmil_by_cluster(results, out_dir / "cluster_abmil")
    plot_slot_cluster_contribution(results, out_dir / "cluster_slot")
    plot_self_attn_coattn(results, out_dir / "cluster_self_attn_coattn")
    plot_cross_modal_cluster_attn(results, out_dir / "cluster_xmodal")
    plot_cross_modal_connection(results, out_dir / "cluster_connection", mod_a="HE",      mod_b="BAL")
    plot_cross_modal_connection(results, out_dir / "cluster_connection", mod_a="HE",      mod_b="CT")
    plot_cross_modal_connection(results, out_dir / "cluster_connection", mod_a="Clinical", mod_b="HE")
    plot_cross_modal_connection(results, out_dir / "cluster_connection", mod_a="Clinical", mod_b="BAL")
    plot_cross_modal_connection(results, out_dir / "cluster_connection", mod_a="Clinical", mod_b="CT")

    summary = {"version": args.version, "variant": variant, "fold_tag": fold_tag,
                "n": len(results),
                "n_pos": sum(r["label"] == 1 for r in results),
                "n_neg": sum(r["label"] == 0 for r in results)}
    with open(out_dir / "summary.json", "w") as f: json.dump(summary, f, indent=2)
    print(f"\n  Done → {out_dir}\n")


if __name__ == "__main__":
    main()
