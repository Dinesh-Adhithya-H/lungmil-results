"""Modality registry — single source of truth for feature keys, dims, presence columns."""

from typing import Dict, Tuple

MODALITY_REGISTRY: Dict[str, Tuple[str, int, str]] = {
    "HE":       ("HE_cells",        1024, "has_HE"),
    "BAL":      ("BAL_cells",       10,   "has_BAL"),
    "CT":       ("CT_cells",        1024, "has_CT"),
    # Clinical: 106 tokens × 491-dim one-hot (precomputed in .pt files; vocab=491).
    "Clinical": ("clinical_onehot", 491,  "has_Clinical"),
}

MODALITIES         = list(MODALITY_REGISTRY.keys())
TEACHER_MODALITIES = ["HE", "Clinical"]
STUDENT_MODALITIES = ["CT", "BAL"]


def _feat_key(mod: str) -> str:
    return MODALITY_REGISTRY[mod][0]

def _feat_dim(mod: str) -> int:
    return MODALITY_REGISTRY[mod][1]

def _pres_col(mod: str) -> str:
    return MODALITY_REGISTRY[mod][2]
