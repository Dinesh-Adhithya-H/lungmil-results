"""
Encoder building blocks for the multimodal ABMIL MIL framework.

Classes exported
----------------
GatedAttentionEncoder   — Phase 1 gated-attention MIL encoder
ModalFFNEncoder         — per-modality FFN projector for SetTransformerMIL
PositionEncoding2D      — 2-D sinusoidal PE for H&E tile coordinates
ProjectionHead          — 2-layer MLP projection head (Phase 1 contrastive)
FFN                     — pre-norm FFN residual block
CrossModalTransformer   — self-attention over concatenated modality tokens
PMA                     — Pooling by Multihead Attention (Set Transformer seed compression)
SAB                     — Set Attention Block (cross-modal seed interaction)
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
    Per-modality 2-layer FFN projector for SetTransformerMIL.

    Projects raw patch features into hidden_dim space.  L2 normalization at
    the output preserves cosine-similarity geometry of foundation model embeddings
    so cross-attention dot products are geometrically meaningful.
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


class PMA(nn.Module):
    """Pooling by Multihead Attention (Lee et al., Set Transformer 2019).

    K learned seed vectors (queries) cross-attend to N patch tokens (KV)
    → (K, H) compressed representation per modality.

    Standard softmax over N patches per seed — no GRU, no iterative routing.
    Seeds stay diverse: each seed has a distinct learned query vector and
    receives independent gradients with no shared hidden state.
    """
    def __init__(self, hidden_dim: int, n_seeds: int,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.seeds   = nn.Parameter(torch.empty(1, n_seeds, hidden_dim))
        nn.init.trunc_normal_(self.seeds, std=0.02)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.layers  = nn.ModuleList([nn.ModuleDict({
            "cross": nn.MultiheadAttention(hidden_dim, n_heads,
                                            dropout=dropout, batch_first=True),
            "norm":  nn.LayerNorm(hidden_dim),
            "ffn":   FFN(hidden_dim, dropout),
        }) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, H) → (K, H)"""
        kv = self.norm_in(x).unsqueeze(0)        # (1, N, H)
        s  = self.seeds                           # (1, K, H)
        for L in self.layers:
            a, _ = L["cross"](s, kv, kv)         # q=seeds attend to patches
            s = L["norm"](s + a)
            s = L["ffn"](s)
        return s.squeeze(0)                       # (K, H)


class SAB(nn.Module):
    """Set Attention Block — self-attention over M*K seed tokens.

    After per-modality PMA compression, seeds from all modalities are
    concatenated and passed through SAB so they can exchange information
    across modalities before ABMIL aggregation.
    """
    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads,
                                           dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn  = FFN(hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, H) → (T, H)"""
        x_b = x.unsqueeze(0)                     # (1, T, H)
        a, _ = self.attn(x_b, x_b, x_b)
        return self.ffn(self.norm(x_b + a)).squeeze(0)   # (T, H)


class TemporalSAB(nn.Module):
    """SAB with causal masking and ALiBi temporal attention bias.

    For a longitudinal sequence of biopsy seeds ordered in time:
      attn_logit[q,k] += -|slope_h| * |days_q - days_k| / (days_range + 1)
    Causal mask: seeds from future biopsies (days_k > days_q) → -inf.

    Slopes are learned per attention head, starting at 0.1.
    """
    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1,
                 n_layers: int = 1):
        super().__init__()
        self.n_heads = n_heads
        self.layers  = nn.ModuleList([nn.ModuleDict({
            "attn": nn.MultiheadAttention(hidden_dim, n_heads,
                                           dropout=dropout, batch_first=True),
            "norm": nn.LayerNorm(hidden_dim),
            "ffn":  FFN(hidden_dim, dropout),
        }) for _ in range(n_layers)])
        self.alibi_slopes = nn.Parameter(torch.ones(n_heads) * 0.1)

    def forward(self, x: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        """
        x:    (N, H) — concatenated seeds from all biopsies in temporal order
        days: (N,)   — days post-transplant for each token
        Returns: (N, H) causally contextualized tokens
        """
        N = x.shape[0]
        # delta[q,k] = days_q - days_k  (positive = q is more recent than k)
        delta      = days.unsqueeze(1) - days.unsqueeze(0)   # (N, N)
        days_range = (days.max() - days.min() + 1.0).clamp(min=1.0)
        dist       = delta.abs() / days_range                 # (N, N) in [0, 1]

        # ALiBi bias: closer tokens attend more strongly
        slopes = self.alibi_slopes.abs()                      # (n_heads,)
        alibi  = -slopes.view(-1, 1, 1) * dist.unsqueeze(0)  # (n_heads, N, N)

        # Causal: q cannot attend to future tokens (delta < 0 means k is future)
        causal = (delta < 0).to(x.dtype) * -1e9              # (N, N)
        bias   = (alibi + causal.unsqueeze(0)).to(x.dtype)   # (n_heads, N, N)

        x_b = x.unsqueeze(0)   # (1, N, H)
        for L in self.layers:
            a, _ = L["attn"](x_b, x_b, x_b,
                             attn_mask=bias.view(self.n_heads, N, N))
            x_b = L["ffn"](L["norm"](x_b + a))
        return x_b.squeeze(0)  # (N, H)
