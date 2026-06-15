"""
Phase 1 per-modality model: SingleModalMIL.

v8 design — Phase 1 purpose
-----------------------------
Phase 1 trains each modality encoder independently.
Its ONLY job: compress raw patch features into compact ABMIL-pooled summary
tokens that are predictive for the downstream task (ACR classification and/or
survival). These tokens are what Phase 2 fusion receives.

Loss: hinge (ACR cls) and/or Cox-Breslow (survival). Nothing else.
No CLR, KD, CRD, or cross-modal objectives — these would optimise a different
objective and can prevent the encoder from learning what Phase 2 needs.

Classes exported
----------------
SingleModalMIL
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import GatedAttentionEncoder


class SingleModalMIL(nn.Module):
    """
    Phase 1 encoder for one modality.

    Components
    ----------
    encoder    : GatedAttentionEncoder — backbone + gated ABMIL
    head       : linear classifier (logit for hinge loss)
    hazard_head: linear hazard (for Cox loss)

    Forward returns logit (scalar) when return_extras=False.
    When return_extras=True returns (logit, dict) where dict contains:
      r_final : (H,)   ABMIL-pooled summary token
      alpha   : (N,)   attention weights over patches
      h       : (N, H) backbone patch features
      hazard  : ()     hazard scalar for Cox
    """

    def __init__(self, feat_dim: int = 1024, hidden_dim: int = 256,
                 dropout: float = 0.4,
                 use_spatial: bool = False):
        super().__init__()
        self.encoder     = GatedAttentionEncoder(feat_dim, hidden_dim, dropout,
                                                  use_spatial=use_spatial)
        self.head        = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.hazard_head = nn.Linear(hidden_dim, 1, bias=True)
        nn.init.normal_(self.hazard_head.weight, 0.0, 0.01)
        nn.init.zeros_(self.hazard_head.bias)

    def forward(self, x: torch.Tensor,
                return_extras: bool = False,
                coords=None):
        rep, alpha, h = self.encoder(x, coords=coords)
        logit  = self.head(rep).squeeze()
        if not return_extras:
            return logit
        hazard = self.hazard_head(rep).squeeze()
        return logit, {"r_final": rep, "alpha": alpha, "h": h, "hazard": hazard}
