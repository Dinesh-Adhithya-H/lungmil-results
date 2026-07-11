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
PMA                     — Pooling by Multihead Attention (Lee et al., Set Transformer 2019)
SAB                     — Set Attention Block (cross-modal seed interaction)
"""

import math
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


class WNLinear(nn.Module):
    """Linear layer with L2-normalized weight rows (no bias).

    For L2-normalized input x and normalized weight row w_i:
        output[i] = x · w_i = cos(x, w_i)  ∈ [-1, 1]

    Used in PMA projections so that b-cos attention scores are cosine-based
    without requiring explicit renormalization of q and k after projection.
    Initialized orthogonally so rows start maximally spread.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, F.normalize(self.weight, dim=1))


class PMA(nn.Module):
    """Pooling by Multihead Attention with b-cos attention (Lee et al. 2019 + Böhle et al.).

    K seed vectors (queries) cross-attend to N L2-normalized patch tokens (KV).
    Weight-normalized projections preserve hyperspherical geometry:
      k[n, i] = cos(x_n, w_k_i)  — no explicit renorm of k needed.
    Seeds are normalized before each cross-attention for the same property.

    b-cos attention: weights = ReLU(q · k)^b / Σ  — seeds must specialise on
    distinct patch directions → prevents all seeds collapsing to the mean pool.

    b=0: falls back to standard scaled dot-product softmax.
    b=4: default — strong seed specialisation, lowest inter-seed cosine in ablation.
    """
    def __init__(self, hidden_dim: int, n_seeds: int,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
                 b: float = 4.0):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.b        = b
        self.n_heads  = n_heads
        self.head_dim = hidden_dim // n_heads
        self.seeds    = nn.Parameter(torch.empty(n_seeds, hidden_dim))
        nn.init.trunc_normal_(self.seeds, std=0.02)
        # Weight-normalized projections for q and k — value/output unconstrained
        self.proj_q = WNLinear(hidden_dim, hidden_dim)
        self.proj_k = WNLinear(hidden_dim, hidden_dim)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.layers = nn.ModuleList([nn.ModuleDict({
            "norm": nn.LayerNorm(hidden_dim),
            "ffn":  FFN(hidden_dim, dropout),
        }) for _ in range(n_layers)])

    def _attn(self, q: torch.Tensor, k: torch.Tensor,
              v: torch.Tensor, return_w: bool = False,
              return_logits: bool = False):
        """q: (nh, K, hd)  k: (nh, N, hd)  v: (nh, N, hd) → (nh, K, hd) [, (nh,K,N) [, (nh,K,N)]]
        return_logits: also return raw q@k.T dot products (pre-relu/softmax)."""
        dots = q @ k.transpose(-2, -1)           # (nh, K, N) — raw dot products
        if self.b == 0:
            w = F.softmax(dots / math.sqrt(self.head_dim), dim=-1)
            relu_dots = F.relu(dots)
            raw       = relu_dots                # no pow for b=0
        else:
            relu_dots = F.relu(dots)             # (nh, K, N) — after relu
            raw = relu_dots.pow(self.b)          # (nh, K, N) — after relu^b
            # Guard: collapsed seeds (all dot-products ≤ 0) fall back to uniform
            row_sum = raw.sum(-1, keepdim=True)  # (nh, K, 1)
            w = torch.where(
                row_sum > 1e-6,
                raw / (row_sum + 1e-9),
                torch.ones_like(raw) / raw.shape[-1],
            )
        out = w @ v
        # logits tuple: (raw_dots, post_relu, post_relu_pow_b) — all (nh,K,N)
        if return_w and return_logits:
            return out, w, (dots, relu_dots, raw)
        if return_w:
            return out, w
        if return_logits:
            return out, (dots, relu_dots, raw)
        return out

    def forward(self, x: torch.Tensor,
                return_attn: bool = False,
                return_logits: bool = False):
        """x: (N, H) L2-normalized patches → (K, H) [, (K, N) mean attn] [, (K, N) mean logits]"""
        N, H   = x.shape
        K      = self.seeds.shape[0]
        nh, hd = self.n_heads, self.head_dim

        s = self.seeds   # (K, H)
        last_w = None
        last_logits = None
        for L in self.layers:
            s_n = F.normalize(s, dim=-1)
            q = self.proj_q(s_n).view(K, nh, hd).transpose(0, 1)   # (nh, K, hd)
            k = self.proj_k(x).view(N, nh, hd).transpose(0, 1)     # (nh, N, hd)
            v = self.proj_v(x).view(N, nh, hd).transpose(0, 1)     # (nh, N, hd)

            if return_attn or return_logits:
                out, w, (dots, relu_dots, raw_pow) = self._attn(q, k, v, return_w=True, return_logits=True)
                last_w      = w.mean(0)           # (K, N) avg heads — post-norm weights
                last_logits = (dots.mean(0),       # (K, N) raw q·k
                               relu_dots.mean(0),  # (K, N) relu(q·k)
                               raw_pow.mean(0))    # (K, N) relu(q·k)^b
            else:
                out = self._attn(q, k, v)
            out = self.proj_o(out.transpose(0, 1).contiguous().view(K, H))

            s = L["norm"](s + out)
            s = L["ffn"](s)

        if return_attn and return_logits:
            return s, last_w, last_logits   # last_logits = (dots, relu, relu^b) each (K,N)
        if return_attn:
            return s, last_w                # (K, H), (K, N)
        if return_logits:
            return s, last_logits
        return s   # (K, H)


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
