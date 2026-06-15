"""
Encoder building blocks for the multimodal ABMIL MIL framework.

Two-phase design context
------------------------
Phase 1 (per-modality pre-training): these modules are used directly inside
``SingleModalMIL`` (see ``phase1.py``).  The goal of Phase 1 is to train
each modality's backbone + attention encoder independently so that the
resulting representations are already predictive before any cross-modal
fusion is attempted.

Phase 2 (multimodal fusion): the trained encoder weights are loaded into
the Phase 2 fusion models (see ``phase2.py``).  ``ModalFFNEncoder`` is the
per-modality 2-layer FFN used by ``SharedSlotMIL``.  ``GatedAttentionEncoder``
is kept for Phase 1 and ablation baselines.  ``MHASlotAttn`` is the
within-modality slot compression.  ``CrossModalTransformer`` and ``FFN``
are the cross-modal interaction modules.

Classes exported
----------------
GatedAttentionEncoder
ModalFFNEncoder
PositionEncoding2D
ProjectionHead
MHASlotAttn
FFN
CrossModalTransformer
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Phase 1 hyperparameter defaults (referenced in default arguments)
P1_ATTN_N_HEADS = 4
P1_ATTN_DROPOUT = 0.1

# Phase 2 hyperparameter defaults
P2_MAX_HE_BLOCK = 1024


class GatedAttentionEncoder(nn.Module):
    """
    Gated attention MIL encoder.
    backbone: Linear(feat_dim → H) + Tanh + Dropout
    gate:     att_V (tanh) * att_U (sigmoid) → att_w → softmax → weighted sum
    Returns:  rep (H,), alpha (N,), h (N, H)
    """
    def __init__(self, feat_dim: int = 1024, hidden_dim: int = 256,
                 dropout: float = 0.4, use_spatial: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone   = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout))
        self.pos_enc  = PositionEncoding2D(hidden_dim) if use_spatial else None
        self.att_V    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.att_U    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.att_w    = nn.Linear(hidden_dim, 1, bias=False)
        self.att_drop = nn.Dropout(dropout)

    def encode_patches(self, x: torch.Tensor, coords=None) -> torch.Tensor:
        """Backbone → optional 2-D sinusoidal PE → (N, H)."""
        h = self.backbone(x)
        if self.pos_enc is not None and coords is not None:
            h = h + self.pos_enc(coords.to(h.device))
        return h

    def forward(self, x: torch.Tensor,
                coords=None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h     = self.encode_patches(x, coords)                  # (N, H)
        gate  = self.att_V(h) * self.att_U(h)                  # (N, H)
        raw   = self.att_w(self.att_drop(gate))                 # (N, 1)
        alpha = F.softmax(raw, dim=0)                           # (N, 1)
        rep   = (alpha * h).sum(dim=0)                          # (H,)
        return rep, alpha.squeeze(1), h


class ModalFFNEncoder(nn.Module):
    """
    Per-modality 2-layer FFN encoder for SharedSlotMIL.

    Projects raw patch features (e.g. L2-normalized UNI-2 1024-dim) into a
    shared hidden_dim slot space using a wider intermediate layer.  The extra
    width (hidden_dim*2) lets the model learn a modality-specific geometry
    transform before slot attention aligns representations across modalities.

    Tanh preserves both positive and negative directions (no dead units).
    Final L2 normalization keeps patch features on the unit sphere — preserving
    the cosine-similarity geometry of foundation model (UNI-2) embeddings so
    that slot attention computes meaningful dot products.
    Compatible with the encode_patches(x, coords) interface used by all encoders.
    """
    def __init__(self, feat_dim: int = 1024, hidden_dim: int = 256,
                 dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def encode_patches(self, x: torch.Tensor, coords=None) -> torch.Tensor:
        """x: (N, feat_dim) → (N, hidden_dim), L2-normalized (unit sphere)."""
        return F.normalize(self.net(x), dim=-1)


class GeoMAESpatialBackbone(nn.Module):
    """
    Drop-in backbone replacement for GatedAttentionEncoder that uses a
    pretrained SpatialDenoisingEncoder (from GeoMAE) instead of Linear(1024→256).

    Implements the same encode_patches(x, coords) → (N, H) interface so the
    rest of the MIL pipeline (slot attention, cross-modal, task head) is unchanged.

    During a 'recon' training epoch the caller invokes forward_recon() which
    applies BFS-flood masking, adds spatial noise, and returns the reconstruction
    loss — keeping the encoder from forgetting its pretraining objective.

    During normal MIL epochs: all visible (no masking), returns clean (N, H) reps.
    """
    def __init__(self, spatial_encoder: "nn.Module"):
        super().__init__()
        self.encoder = spatial_encoder   # SpatialDenoisingEncoder instance

    def encode_patches(self, x: torch.Tensor,
                       coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """MIL forward: all patches visible, return spatially-contextualised (N, H)."""
        if coords is None:
            # No coords — fall back to projection only (first layer of encoder)
            return self.encoder.proj(x)
        out = self.encoder(x, coords)   # visible_mask=None → all visible
        return out["encoded"]           # (N, hidden_dim)

    def forward_recon(self, x: torch.Tensor,
                      coords: torch.Tensor,
                      mask_ratio: float = 0.5) -> torch.Tensor:
        """
        Reconstruction forward for self-supervised regularisation epochs.
        Applies BFS-flood contiguous masking then computes weighted MSE loss
        on masked patches (same objective as GeoMAE pretraining).
        Returns scalar loss tensor.
        """
        import torch.nn.functional as F
        from mil.models.pretrain import build_knn_graph, contiguous_region_mask
        import random as _random

        N = x.shape[0]
        if N < 20:
            return x.new_zeros(1).squeeze()

        ei, ew       = build_knn_graph(coords, self.encoder.knn_k)
        visible_mask = contiguous_region_mask(coords, mask_ratio, ei)
        out          = self.encoder(x, coords, visible_mask=visible_mask,
                                    precomputed_graph=(ei, ew))

        masked = ~out["visible_mask"]
        if not masked.any():
            return x.new_zeros(1).squeeze()

        eps_pred = out["noise_pred"][masked]
        eps_true = out["noise_true"][masked]
        d_w      = (out["distances"][masked].float() /
                    max(out["distances"].max().item(), 1))
        return (d_w * F.mse_loss(eps_pred, eps_true, reduction="none").mean(-1)).mean()

    def forward(self, x, coords=None):
        """Standard forward — returns (rep, alpha, h) like GatedAttentionEncoder."""
        raise NotImplementedError(
            "Use encode_patches() for MIL or forward_recon() for reconstruction.")


class PositionEncoding2D(nn.Module):
    """
    Fixed 2-D sinusoidal positional encoding for WSI tiles (TransMIL style).

    Tile coordinates (tile_left, tile_top) are in pixels; dividing by
    tile_stride (224 px) converts to grid indices.  hidden_dim // 4 frequency
    bands are applied per axis, yielding hidden_dim values total:
      [sin_col | cos_col | sin_row | cos_row]

    No learnable parameters — encoding is a pure function of coordinates.
    Works for any irregular patch layout (no grid-reshape required).
    """
    def __init__(self, hidden_dim: int, tile_stride: int = 224,
                 temperature: float = 10000.0):
        super().__init__()
        assert hidden_dim % 4 == 0, "hidden_dim must be divisible by 4 for 2-D PE"
        self.tile_stride = tile_stride
        d    = hidden_dim // 4
        freq = temperature ** (-torch.arange(d, dtype=torch.float32) / d)
        self.register_buffer("freq", freq)   # (d,)  — no gradient

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (N, 2) float [tile_left, tile_top] in pixels → (N, hidden_dim)"""
        col  = coords[:, 0].float() / self.tile_stride   # (N,) grid column
        row  = coords[:, 1].float() / self.tile_stride   # (N,) grid row
        freq = self.freq                                  # (d,)
        sin_col = torch.sin(col.unsqueeze(1) * freq)     # (N, d)
        cos_col = torch.cos(col.unsqueeze(1) * freq)
        sin_row = torch.sin(row.unsqueeze(1) * freq)
        cos_row = torch.cos(row.unsqueeze(1) * freq)
        return torch.cat([sin_col, cos_col, sin_row, cos_row], dim=1)  # (N, H)


class ProjectionHead(nn.Module):
    """2-layer MLP projection head for contrastive learning."""
    def __init__(self, hidden_dim: int = 256, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class MHASlotAttn(nn.Module):
    """
    Within-modality non-competitive slot attention.

    Q = K learned slot tokens (nn.Parameter),
    K = V = pretrained backbone patch features h (N, H).

    Standard MHA: softmax over N patches, so every patch can contribute to
    every slot — no competition between slots.  No cold-start: backbone h
    already carries Phase-1 trained representations.

    Per iteration (n_iters rounds):
      slots ← slots + MHA(norm_q(slots), L2_norm(h), L2_norm(h))
      slots ← slots + FFN(norm(slots))

    Keys/values use L2 normalisation (not LayerNorm) so that dot-product attention
    computes cosine similarities — preserving the spherical geometry of foundation
    model (UNI-2) embeddings that enter via ModalFFNEncoder.
    Queries (slot tokens) keep LayerNorm since they are learned, not pre-spherical.
    """
    def __init__(self, hidden_dim: int, n_slots: int = 8,
                 n_iters: int = 3, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_iters   = n_iters
        self.slot_init = nn.Parameter(torch.empty(n_slots, hidden_dim))
        nn.init.normal_(self.slot_init, std=0.02)
        self.norm_q  = nn.LayerNorm(hidden_dim)
        self.mha     = nn.MultiheadAttention(hidden_dim, n_heads,
                                              dropout=dropout, batch_first=True)
        self.mlp     = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim))

    def forward(self, h: torch.Tensor,
                init: "Optional[torch.Tensor]" = None) -> torch.Tensor:
        """
        h:    (N, H) — pretrained backbone patch features
        init: (K, H) — optional external slot init (overrides self.slot_init).
              Pass per-task slot tokens for task-specific routing.
        Returns: (K, H) — slot features
        """
        slots = (init if init is not None else self.slot_init).clone()  # (K, H)
        kv    = F.normalize(h, dim=-1).unsqueeze(0)    # (1, N, H) — L2 norm preserves sphere
        for _ in range(self.n_iters):
            q      = self.norm_q(slots).unsqueeze(0)   # (1, K, H)
            out, _ = self.mha(q, kv, kv)               # (1, K, H); softmax over N
            slots  = slots + out.squeeze(0)
            slots  = slots + self.mlp(slots)
        return slots   # (K, H)


class FFN(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(dim, dim*2), nn.Tanh(),
                                   nn.Dropout(dropout), nn.Linear(dim*2, dim),
                                   nn.Dropout(dropout))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x + self.net(x))


class CrossModalTransformer(nn.Module):
    def __init__(self, dim, n_heads, dropout, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({
            "attn": nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True),
            "norm": nn.LayerNorm(dim),
            "ffn":  FFN(dim, dropout),
        }) for _ in range(n_layers)])

    def forward(self, x):
        for L in self.layers:
            a, _ = L["attn"](x, x, x)
            x = L["ffn"](L["norm"](x + a))
        return x
