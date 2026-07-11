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
  slot — SetTransformerMIL (recommended):
    1. Per-modality ModalFFNEncoder: 2-layer FFN (feat_dim → H*2 → H)
    2. Per-modality PMA: K learned seeds cross-attend to N patches → (K, H)
       Standard multi-head attention (softmax over patches), exactly as in
       Lee et al. Set Transformer 2019.
    3. Concatenate all modality seeds → (M*K, H)
    4. SAB (self-attention) for cross-modal seed interaction
    5. Per-task gated ABMIL over M*K tokens → task representation
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

from .encoders import GatedAttentionEncoder, ModalFFNEncoder
from .phase2 import (
    EarlyFusionMIL,
    LateFusionMIL,
    MiddleFusionMIL,
    SetTransformerMIL,
    LongitudinalMIL,
)
from mil.data.registry import MODALITIES, _feat_dim

# ── Phase 2 hyperparameters ───────────────────────────────────────────────────
HIDDEN_DIM        = 256
DROPOUT           = 0.4
P2_MODAL_DROPOUT  = 0.3    # 0.3 → model trains with each modality alone ~24% of the time
P2_N_HEADS        = 1
P2_N_CROSS_LAYERS = 1      # 1 layer sufficient for small dataset; 4 overfits
P2_ATTN_DROPOUT   = 0.1
P2_MAX_PATCHES    = 2048
P2_MAX_HE_BLOCK   = 99999  # effectively uncapped — use all HE patches (A100/H100 80GB safe up to ~50k)
P2_SLOT_K         = 16     # dot products ~1/√128 → softmax uniform)
P2_PMA_LAYERS     = 2      # cross-attention layers inside PMA

# Available variants
P2_VARIANTS = ["set_mil", "set_mil_mt", "early", "late", "middle",
               "longitudinal_mk", "longitudinal_mk_mt"]

# Maps task name → list of prediction heads to build
TASK_GROUPS = {
    "cls":        ["acr_cls"],
    "surv":       ["acr_surv"],
    "both":       ["acr_cls", "acr_surv"],
    "both_alt":   ["acr_cls", "acr_surv"],
    "clad_surv":  ["clad"],
    "death_surv": ["death"],
    "mega":       ["acr_cls", "acr_surv", "clad", "death"],
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
    variant       : "set_mil" (recommended) | "set_mil_mt" | "early" | "late" | "middle"
    modal_dropout : probability of dropping a modality during training
    slot_k        : number of slot tokens per modality (hyperparameter; default 8)
    n_cross_layers: unused for slot variant (kept for API compat with early/middle)
    task          : controls which prediction heads are built (see TASK_GROUPS)
                    mega recommended for slot (trains all 4 tasks jointly)
    """
    import torch.nn as nn
    # GatedAttentionEncoder used by early/late/middle; ModalFFNEncoder used by slot
    encoders   = {m: GatedAttentionEncoder(_feat_dim(m), HIDDEN_DIM, DROPOUT)
                  for m in MODALITIES}
    proj_heads = {}   # unused in v8; kept for API compat
    _task_list = TASK_GROUPS.get(task, ["acr_cls", "acr_surv"])
    kw = dict(hidden_dim=HIDDEN_DIM, dropout=DROPOUT, modal_dropout=modal_dropout)

    if variant in ("set_mil", "set_mil_mt"):
        encoders = {m: ModalFFNEncoder(_feat_dim(m), HIDDEN_DIM, DROPOUT)
                    for m in MODALITIES}
        # HE is rare (~20% of samples) — never drop it during training.
        # Other modalities keep the standard modal_dropout rate.
        he_aware_dropout = {m: (0.0 if m == "HE" else modal_dropout) for m in MODALITIES}
        return SetTransformerMIL(
            encoders,
            hidden_dim=HIDDEN_DIM,
            n_seeds=slot_k,
            n_pma_layers=2,
            n_sab_layers=max(1, n_cross_layers),
            n_heads=P2_N_HEADS,
            dropout=P2_ATTN_DROPOUT,
            modal_dropout=he_aware_dropout,
            max_he_patches=max_he_patches,
            tasks=_task_list,
            use_task_gate=(variant == "set_mil_mt"),
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
    if variant in ("longitudinal_mk", "longitudinal_mk_mt"):
        encoders = {m: ModalFFNEncoder(_feat_dim(m), HIDDEN_DIM, DROPOUT)
                    for m in MODALITIES}
        return LongitudinalMIL(
            encoders,
            hidden_dim=HIDDEN_DIM,
            n_seeds=slot_k,
            n_pma_layers=2,
            n_sab_layers=max(1, n_cross_layers),
            n_heads=P2_N_HEADS,
            dropout=P2_ATTN_DROPOUT,
            modal_dropout=modal_dropout,
            max_he_patches=max_he_patches,
            tasks=_task_list,
            use_task_gate=(variant == "longitudinal_mk_mt"),
        )
    raise ValueError(f"Unknown variant {variant!r}. Choose from: {P2_VARIANTS}")


build_model = build_model_v8
