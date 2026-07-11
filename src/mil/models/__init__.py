"""
Multimodal ABMIL model classes and factory.

Two-phase design
----------------
Phase 1 — per-modality pre-training: each modality's encoder is trained
independently to compress raw patch features into compact summary tokens
that are already predictive before any cross-modal fusion is attempted.
See ``phase1.py`` (SingleModalMIL).

Phase 2 — multimodal fusion: takes the Phase 1 summary tokens from all
available modalities and fuses them.
  SharedSlotMIL  — K=128 competitive slots, recommended
  EarlyFusionMIL / LateFusionMIL / MiddleFusionMIL — ablation baselines

The canonical entry point is ``build_model`` (alias for ``build_model_v8``)
from ``builders.py``.

Example
-------
>>> from mil.models.builders import build_model
>>> model = build_model("set_mil", task="mega")
"""

from .encoders import (
    GatedAttentionEncoder,
    ModalFFNEncoder,
    PositionEncoding2D,
    ProjectionHead,
    FFN,
    CrossModalTransformer,
    PMA,
    SAB,
)
from .phase1 import SingleModalMIL
from .phase2 import (
    EarlyFusionMIL,
    LateFusionMIL,
    MiddleFusionMIL,
    SetTransformerMIL,
    DualGatedPool,
    MultiTaskHead,
    _load_p1_encoder,
    _load_p1_proj_head,
    _abmil_pool,
    _pool,
)
from .builders import (
    build_model,
    build_model_v8,
    TASK_GROUPS,
    P2_VARIANTS,
    HIDDEN_DIM,
    DROPOUT,
)

__all__ = [
    # encoders
    "GatedAttentionEncoder",
    "ModalFFNEncoder",
    "PositionEncoding2D",
    "ProjectionHead",
    "FFN",
    "CrossModalTransformer",
    "PMA",
    "SAB",
    # phase1
    "SingleModalMIL",
    # phase2
    "EarlyFusionMIL",
    "LateFusionMIL",
    "MiddleFusionMIL",
    "SetTransformerMIL",
    "DualGatedPool",
    "MultiTaskHead",
    "_load_p1_encoder",
    "_load_p1_proj_head",
    "_abmil_pool",
    "_pool",
    # factory
    "build_model",
    "build_model_v8",
    "TASK_GROUPS",
    "P2_VARIANTS",
    "HIDDEN_DIM",
    "DROPOUT",
]
