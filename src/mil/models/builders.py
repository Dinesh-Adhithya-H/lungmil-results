"""
Model factory for the multimodal ABMIL framework (v8 design).

Two-phase design context
------------------------
``build_model_v8`` creates Phase 2 fusion models with fresh random-init encoders.
In the two-phase workflow:
  Phase 1: per-modality encoders trained independently (see phase1_trainer).
  Phase 2: build_model_v8 creates the fusion model; caller loads Phase 1 weights.

v8 architecture
---------------
  slot — SharedSlotMIL (recommended):
    1. Per-modality ModalFFNEncoder: 2-layer FFN (feat_dim → H*2 → H)
    2. K=128 globally shared slot init tokens
    3. Per-modality MHASlotAttn (separate weights) → (K, H) per modality
    4. Mean over present modalities → fair cross-modal aggregation
    5. Per-task gated ABMIL over K shared slots → task representation
    6. Per-task cls/survival heads

  Simpler ablation baselines (early / late / middle) kept for comparison.

Functions exported
------------------
build_model_v8
build_model     (alias)
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch

from .encoders import GatedAttentionEncoder, ModalFFNEncoder, GeoMAESpatialBackbone
from .phase2 import (
    EarlyFusionMIL,
    LateFusionMIL,
    MiddleFusionMIL,
    TaskSpecificSlotMIL,
    SharedSlotMIL,
)
from mil.data.registry import MODALITIES, _feat_dim

# ── Phase 2 hyperparameters ───────────────────────────────────────────────────
HIDDEN_DIM        = 256
DROPOUT           = 0.4
P2_MODAL_DROPOUT  = 0.3    # 0.3 → model trains with each modality alone ~24% of the time
P2_N_HEADS        = 4
P2_N_CROSS_LAYERS = 1      # 1 layer sufficient for small dataset; 4 overfits
P2_ATTN_DROPOUT   = 0.1
P2_MAX_PATCHES    = 2048
P2_MAX_HE_BLOCK   = 99999  # effectively uncapped — use all HE patches (A100/H100 80GB safe up to ~50k)
P2_SLOT_K         = 128    # shared slots across modalities
P2_SLOT_ITERS     = 1      # 1 iter sufficient; more risks overfitting

# Available variants
P2_VARIANTS = ["slot", "early", "late", "middle"]

# Maps task name → list of prediction heads to build
TASK_GROUPS = {
    "cls":        ["acr_cls"],
    "surv":       ["acr_surv"],
    "both":       ["acr_cls", "acr_surv"],
    "both_alt":   ["acr_cls", "acr_surv"],
    "clad_surv":  ["clad"],
    "death_surv": ["death"],
    "mega":       ["acr_cls", "acr_surv", "clad", "death"],
    # GeoMAE-backed alternating: same prediction heads as mega,
    # but training adds "recon" to the multinomial sampling distribution
    "geomae_alt": ["acr_cls", "acr_surv", "clad", "death"],
}


def build_model_v8(
    variant: str = "slot",
    modal_dropout: float = P2_MODAL_DROPOUT,
    slot_k: int = P2_SLOT_K,
    n_cross_layers: int = P2_N_CROSS_LAYERS,
    task: str = "mega",
    max_he_patches: int = P2_MAX_HE_BLOCK,
) -> "nn.Module":
    """
    Build a fresh (randomly initialised) Phase 2 fusion model.

    Parameters
    ----------
    variant       : "slot" (recommended) | "early" | "late" | "middle"
    modal_dropout : probability of dropping a modality during training
    slot_k        : number of slot tokens per modality (hyperparameter; default 8)
    n_cross_layers: unused for slot variant (kept for API compat with early/middle)
    task          : controls which prediction heads are built (see TASK_GROUPS)
                    mega recommended for slot (trains all 4 tasks jointly)
    """
    import torch.nn as nn
    _task_list = TASK_GROUPS.get(task, ["acr_cls", "acr_surv"])
    kw = dict(hidden_dim=HIDDEN_DIM, dropout=DROPOUT, modal_dropout=modal_dropout)

    if variant == "slot":
        # SharedSlotMIL: K=128 globally shared slot tokens, per-modality FFN encoders,
        # per-modality MHASlotAttn (separate weights), mean fairness aggregation,
        # per-task gated ABMIL + per-task heads.
        encoders = {m: ModalFFNEncoder(_feat_dim(m), HIDDEN_DIM, DROPOUT)
                    for m in MODALITIES}
        return SharedSlotMIL(
            encoders,
            hidden_dim=HIDDEN_DIM,
            n_heads=P2_N_HEADS,
            dropout=P2_ATTN_DROPOUT,
            modal_dropout=modal_dropout,
            n_slots=slot_k,
            n_slot_iters=P2_SLOT_ITERS,
            max_he_patches=max_he_patches,
            tasks=_task_list,
        )
    if variant == "early":
        return EarlyFusionMIL(encoders, proj_heads, **kw,
                               max_patches_per_mod=P2_MAX_PATCHES,
                               tasks=_task_list)
    if variant == "late":
        return LateFusionMIL(encoders, proj_heads, **kw, tasks=_task_list)
    if variant == "middle":
        return MiddleFusionMIL(encoders, proj_heads,
                                n_heads=P2_N_HEADS,
                                n_layers=2,
                                dropout=P2_ATTN_DROPOUT,
                                modal_dropout=modal_dropout,
                                hidden_dim=HIDDEN_DIM,
                                tasks=_task_list)
    raise ValueError(f"Unknown variant {variant!r}. Choose from: {P2_VARIANTS}")


build_model = build_model_v8


# ── GeoMAE-initialised MIL model ──────────────────────────────────────────────

def load_geomae_weights(
    model: "torch.nn.Module",
    geomae_ckpt: str,
    trainable: bool = True,
) -> Dict[str, bool]:
    """
    Load pretrained GeoMAE backbone weights into a TaskSpecificSlotMIL model.

    Replaces the simple Linear(1024→256) backbone in HE and CT encoders with
    GeoMAESpatialBackbone (wrapping the pretrained SpatialDenoisingEncoder).
    BAL and Clinical encoders keep their GatedAttentionEncoder backbones.

    Returns dict of {modality: loaded} indicating which encoders were replaced.

    Usage:
        model = build_model_v8(variant="slot", task="mega")
        loaded = load_geomae_weights(model, "results/geomae_pretrain/best_backbone.pt")
        # model.encoders["HE"] and model.encoders["CT"] now use GeoMAE backbone
    """
    from .pretrain import SpatialDenoisingEncoder, ClinicalMaskedEncoder, GeoMAE

    ckpt = torch.load(geomae_ckpt, map_location="cpu", weights_only=False)
    # best_backbone.pt is saved by GeoMAE.get_backbone_weights() →
    # {"he_encoder": state_dict, "ct_encoder": state_dict, "clin_encoder": state_dict}
    if not isinstance(ckpt, dict) or "he_encoder" not in ckpt:
        raise ValueError(
            f"Expected GeoMAE backbone checkpoint with keys "
            f"[he_encoder, ct_encoder, clin_encoder], got: {list(ckpt.keys())[:5]}")

    loaded = {}

    # ── HE encoder ────────────────────────────────────────────────────────────
    if hasattr(model, "encoders") and "HE" in model.encoders:
        he_enc = SpatialDenoisingEncoder(
            feat_dim=1024, hidden_dim=HIDDEN_DIM, n_layers=3, n_heads=4,
            knn_k=8, max_dist=32)
        he_enc.load_state_dict(ckpt["he_encoder"], strict=True)
        if not trainable:
            for p in he_enc.parameters(): p.requires_grad_(False)
        model.encoders["HE"] = GeoMAESpatialBackbone(he_enc)
        loaded["HE"] = True
        print(f"  [GeoMAE] HE encoder loaded  ({'trainable' if trainable else 'frozen'})")

    # ── CT encoder ────────────────────────────────────────────────────────────
    if hasattr(model, "encoders") and "CT" in model.encoders:
        ct_enc = SpatialDenoisingEncoder(
            feat_dim=1024, hidden_dim=HIDDEN_DIM, n_layers=3, n_heads=4,
            knn_k=8, max_dist=32)
        ct_enc.load_state_dict(ckpt["ct_encoder"], strict=True)
        if not trainable:
            for p in ct_enc.parameters(): p.requires_grad_(False)
        model.encoders["CT"] = GeoMAESpatialBackbone(ct_enc)
        loaded["CT"] = True
        print(f"  [GeoMAE] CT encoder loaded  ({'trainable' if trainable else 'frozen'})")

    # ── BAL / Clinical: keep GatedAttentionEncoder (no GeoMAE for these) ──────
    for mod in ["BAL", "Clinical"]:
        loaded[mod] = False

    return loaded


