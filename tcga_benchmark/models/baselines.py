"""
WSI-only survival baselines for TCGA benchmark.

bags = {WSI: (N,1536), WSI_coords: (N,2)} → hazard scalar.

Non-spatial (ignore coords):
  ABMIL, TransMIL, SlotMIL

Spatial-aware (use WSI_coords):
  SETMIL        — spatial encoding transformer (torchmil)
  PatchGCN      — graph convolution on spatial KNN graph (torchmil)
  GeoMAE-SlotMIL — ours: GeoMAE spatial encoder + slot attention
"""
import sys, random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmil.models as tm

WSI_DIM = 1536
HIDDEN  = 256


# ── Shared ────────────────────────────────────────────────────────────────────

class CoxHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def _proj(in_dim, hidden, dropout=0.25):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
                         nn.GELU(), nn.Dropout(dropout))


def _build_adj(coords: torch.Tensor, k: int = 8) -> torch.Tensor:
    """KNN adjacency (N,N) from pixel coordinates."""
    d = torch.cdist(coords, coords)
    d.fill_diagonal_(float("inf"))
    _, idx = d.topk(k, largest=False, dim=1)
    adj = torch.zeros(coords.shape[0], coords.shape[0], device=coords.device)
    adj.scatter_(1, idx, 1.0)
    return ((adj + adj.T) > 0).float()


# ── 1. ABMIL ──────────────────────────────────────────────────────────────────

class ABMIL(nn.Module):
    name = "ABMIL"

    def __init__(self, hidden: int = HIDDEN, dropout: float = 0.25):
        super().__init__()
        self.enc = _proj(WSI_DIM, hidden, dropout)
        self.mil = tm.ABMIL(in_shape=(hidden,), att_dim=hidden, gated=True)
        self.cox = CoxHead(hidden)

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        wsi = bags.get("WSI")
        if wsi is None: return torch.tensor(0.0, device=device)
        h   = self.enc(wsi.to(device)).unsqueeze(0)     # (1, N, H)
        # mil returns logit; we need bag rep → use attention-weighted mean
        att = self.mil.att(self.mil.att_V(h) * self.mil.att_U(h))  # (1,N,1)
        att = torch.softmax(att, dim=1)
        rep = (att * h).sum(1).squeeze(0)               # (H,)
        return self.cox(rep)

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        wsi = bags.get("WSI")
        if wsi is None: return torch.tensor(0.0, device=device)
        h   = self.enc(wsi.to(device))                  # (N, H)
        # gated attention pool
        A_v = torch.tanh(self.mil.att_V(h.unsqueeze(0)))    # (1,N,H)
        A_u = torch.sigmoid(self.mil.att_U(h.unsqueeze(0)))
        A   = torch.softmax(self.mil.att(A_v * A_u), dim=1) # (1,N,1)
        rep = (A * h.unsqueeze(0)).sum(1).squeeze(0)
        return self.cox(rep)


# Clean ABMIL without torchmil internals
class ABMIL(nn.Module):
    name = "ABMIL"

    def __init__(self, hidden: int = HIDDEN, dropout: float = 0.25):
        super().__init__()
        self.enc   = _proj(WSI_DIM, hidden, dropout)
        self.att_V = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.att_U = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.att_w = nn.Linear(hidden, 1, bias=False)
        self.cox   = CoxHead(hidden)

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        wsi = bags.get("WSI")
        if wsi is None: return torch.tensor(0.0, device=device)
        h   = self.enc(wsi.to(device))
        A   = torch.softmax(self.att_w(self.att_V(h) * self.att_U(h)), dim=0)
        rep = (A * h).sum(0)
        return self.cox(rep)


# ── 2. TransMIL ───────────────────────────────────────────────────────────────

class TransMIL(nn.Module):
    name = "TransMIL"

    def __init__(self, hidden: int = HIDDEN, dropout: float = 0.1):
        super().__init__()
        self.enc  = _proj(WSI_DIM, hidden, dropout)
        self.cls  = nn.Parameter(torch.randn(1, hidden) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            hidden, 8, hidden*2, dropout, batch_first=True, norm_first=True)
        self.xfmr = nn.TransformerEncoder(enc_layer, 2)
        self.cox  = CoxHead(hidden)

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        wsi = bags.get("WSI")
        if wsi is None: return torch.tensor(0.0, device=device)
        h   = self.enc(wsi.to(device))
        cls = self.cls.to(device)
        h   = torch.cat([cls, h], 0).unsqueeze(0)
        h   = self.xfmr(h).squeeze(0)
        return self.cox(h[0])


# ── 3. SETMIL — spatial encoding transformer (torchmil) ──────────────────────

class SETMIL(nn.Module):
    """SETMIL: spatial coords as sinusoidal PE injected into TransMIL."""
    name = "SETMIL"
    uses_coords = True

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.enc = _proj(WSI_DIM, hidden)
        self.mil = tm.SETMIL(in_shape=(hidden,), att_dim=hidden)
        self.cox = CoxHead(hidden)

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        wsi    = bags.get("WSI")
        coords = bags.get("WSI_coords")
        if wsi is None: return torch.tensor(0.0, device=device)
        h = self.enc(wsi.to(device)).unsqueeze(0)    # (1, N, H)
        try:
            if coords is not None:
                c = coords.to(device).unsqueeze(0)   # (1, N, 2)
                out = self.mil(h, c)
            else:
                out = self.mil(h, torch.zeros(1, h.shape[1], 2, device=device))
            # out is (logit,) or (logit, rep)
            logit = out[0] if isinstance(out, tuple) else out
            # Use final layer norm output as rep for Cox
            rep   = out[1].squeeze(0) if (isinstance(out, tuple)
                                          and len(out) > 1) else h.mean(1).squeeze(0)
        except Exception:
            rep = h.mean(1).squeeze(0)
        return self.cox(rep) if rep.shape == (HIDDEN,) else \
               self.cox(h.mean(1).squeeze(0))


# ── 4. PatchGCN — graph conv on spatial KNN (torchmil) ───────────────────────

class PatchGCN(nn.Module):
    """PatchGCN: 4-layer GCN over spatial KNN graph."""
    name = "PatchGCN"
    uses_coords = True

    def __init__(self, hidden: int = HIDDEN, knn_k: int = 8):
        super().__init__()
        self.knn_k = knn_k
        self.enc   = _proj(WSI_DIM, hidden)
        self.mil   = tm.PatchGCN(in_shape=(hidden,), hidden_dim=hidden,
                                  n_gcn_layers=4, mlp_depth=1)
        self.cox   = CoxHead(hidden)

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        wsi    = bags.get("WSI")
        coords = bags.get("WSI_coords")
        if wsi is None: return torch.tensor(0.0, device=device)
        h = self.enc(wsi.to(device)).unsqueeze(0)   # (1, N, H)
        try:
            if coords is not None:
                adj = _build_adj(coords.to(device), self.knn_k).unsqueeze(0)
                out = self.mil(h, adj)
            else:
                N   = h.shape[1]
                adj = torch.eye(N, device=device).unsqueeze(0)
                out = self.mil(h, adj)
            rep = out[1].squeeze(0) if (isinstance(out, tuple)
                                        and len(out) > 1) else h.mean(1).squeeze(0)
        except Exception:
            rep = h.mean(1).squeeze(0)
        return self.cox(rep)


# ── 5. SlotMIL (ours, no spatial) ────────────────────────────────────────────

class _SlotAttn(nn.Module):
    def __init__(self, hidden, n_slots, n_iters=3, n_heads=4, dropout=0.1):
        super().__init__()
        self.slots   = nn.Parameter(torch.randn(n_slots, hidden) * 0.02)
        self.n_iters = n_iters
        self.attn    = nn.MultiheadAttention(hidden, n_heads,
                                             dropout=dropout, batch_first=True)
        self.norm_s  = nn.LayerNorm(hidden)
        self.norm_x  = nn.LayerNorm(hidden)
        self.ffn     = nn.Sequential(nn.Linear(hidden, hidden*2), nn.GELU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden*2, hidden))
        self.norm_f  = nn.LayerNorm(hidden)

    def forward(self, x):
        S = self.slots.unsqueeze(0)
        X = self.norm_x(x).unsqueeze(0)
        for _ in range(self.n_iters):
            q, _ = self.attn(self.norm_s(S), X, X)
            S = S + q
            S = S + self.ffn(self.norm_f(S))
        return S.squeeze(0)


class SlotMIL(nn.Module):
    name = "SlotMIL"

    def __init__(self, hidden: int = HIDDEN, n_slots: int = 8,
                 n_iters: int = 3, n_cross: int = 1, dropout: float = 0.25):
        super().__init__()
        self.enc   = _proj(WSI_DIM, hidden, dropout)
        self.slots = _SlotAttn(hidden, n_slots, n_iters)
        enc_layer  = nn.TransformerEncoderLayer(
            hidden, 4, hidden*2, 0.1, batch_first=True, norm_first=True)
        self.xfmr  = nn.TransformerEncoder(enc_layer, n_cross)
        self.cox   = CoxHead(hidden)

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        wsi = bags.get("WSI")
        if wsi is None: return torch.tensor(0.0, device=device)
        h = self.enc(wsi.to(device))
        t = self.slots(h).unsqueeze(0)
        t = self.xfmr(t).squeeze(0)
        return self.cox(t.mean(0))


# ── 6. GeoMAE-SlotMIL (ours, spatial encoder) ────────────────────────────────

class GeoMAESlotMIL(nn.Module):
    """
    SlotMIL with pretrained SpatialDenoisingEncoder backbone.
    Encodes (N,1536)+coords → (N,256) with spatial graph context.
    Fine-tuned with alternating Cox survival / reconstruction epochs.
    """
    name = "GeoMAE-SlotMIL"
    uses_coords = True

    def __init__(self, hidden: int = HIDDEN, n_slots: int = 8,
                 n_iters: int = 3, n_cross: int = 1, dropout: float = 0.25,
                 geomae_ckpt: Optional[str] = None,
                 trainable_backbone: bool = True):
        super().__init__()
        sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
        from mil.models.pretrain import SpatialDenoisingEncoder
        from mil.models.encoders import GeoMAESpatialBackbone

        enc = SpatialDenoisingEncoder(
            feat_dim=WSI_DIM, hidden_dim=hidden,
            n_layers=3, n_heads=4, knn_k=8, max_dist=32)

        if geomae_ckpt and Path(geomae_ckpt).exists():
            import torch as _t
            ckpt = _t.load(geomae_ckpt, map_location="cpu", weights_only=False)
            key  = "he_encoder" if "he_encoder" in ckpt else list(ckpt.keys())[0]
            enc.load_state_dict(ckpt[key], strict=True)
            print(f"  [GeoMAE] loaded {key} from {geomae_ckpt}")

        if not trainable_backbone:
            for p in enc.parameters(): p.requires_grad_(False)

        self.backbone = GeoMAESpatialBackbone(enc)

        self.slots = _SlotAttn(hidden, n_slots, n_iters)
        enc_layer  = nn.TransformerEncoderLayer(
            hidden, 4, hidden*2, 0.1, batch_first=True, norm_first=True)
        self.xfmr  = nn.TransformerEncoder(enc_layer, n_cross)
        self.cox   = CoxHead(hidden)

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        wsi    = bags.get("WSI")
        coords = bags.get("WSI_coords")
        if wsi is None: return torch.tensor(0.0, device=device)
        h = self.backbone.encode_patches(
            wsi.to(device),
            coords.to(device) if coords is not None else None)
        t = self.slots(h).unsqueeze(0)
        t = self.xfmr(t).squeeze(0)
        return self.cox(t.mean(0))


# ── Registry ──────────────────────────────────────────────────────────────────

MODELS = {
    "abmil":          ABMIL,
    "transmil":       TransMIL,
    "setmil":         SETMIL,
    "patchgcn":       PatchGCN,
    "slotmil":        SlotMIL,
    "geomae_slotmil": GeoMAESlotMIL,
}
SPATIAL_MODELS = {"setmil", "patchgcn", "geomae_slotmil"}


def build_model(name: str, **kw) -> nn.Module:
    cls = MODELS.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown model {name!r}. Choose: {list(MODELS)}")
    return cls(**kw)
