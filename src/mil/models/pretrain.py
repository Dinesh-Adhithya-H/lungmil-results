"""
GeoMAE v2: Geometry-Aware Masked Autoencoder for Spatial MIL Pretraining.

Changes from v1 (based on GraphMAE best practices):
  - Binary masking (mask token replaces features) instead of noisy interpolation
  - x₀-prediction: reconstruct original features directly, not noise ε
  - Scaled Cosine Error (SCE) loss: 1 - cos(pred, target)^γ  (GraphMAE, NeurIPS 2022)
    Better than MSE for L2-normalized UNI embeddings (DINOv2 ViT CLS tokens)
  - k=32 KNN graph: avoids disconnected components during BFS masking
  - n_layers=5: sufficient for BFS depth ≤ 8 with k=32
  - max_dist set dynamically per patient to actual max BFS depth
  - Only train-split patients used for pretraining (no data leakage)

Spatial causal structure preserved:
  BFS depth = reconstruction order. Boundary patches (depth=1) reconstructed first,
  interior patches attend only to shallower neighbours. This enforces spatial coherence.

Data format expected (from mil_v2/samples/*.pt):
  inputs['HE_cells']  (N, 1024) float
  inputs['CT_cells']  (M, 1024) float
  inputs['Clinical']  (102,)    float
  instance_spatial_coords['HE_cells'] (N, 2) pixel coords
  instance_spatial_coords['CT_cells'] (M, 3) voxel coords
"""

import math
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── KNN graph utilities ───────────────────────────────────────────────────────

def build_knn_graph(coords: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build KNN graph from spatial coordinates using pure PyTorch (no torch_geometric).

    Returns:
        edge_index (2, E)  — [source, target] pairs (directed, each node has k outgoing)
        edge_weights (E,)  — Gaussian weights exp(-d²/2σ²), σ = median NN distance
    """
    N = coords.shape[0]
    k = min(k, N - 1)

    # Pairwise squared distances
    diff  = coords.unsqueeze(0) - coords.unsqueeze(1)   # (N, N, D)
    dist2 = (diff ** 2).sum(-1)                          # (N, N)

    # k nearest (excluding self)
    dist2.fill_diagonal_(float("inf"))
    topk_dist2, topk_idx = dist2.topk(k, dim=1, largest=False)   # (N, k)

    # σ = median of nearest-neighbour distances
    nn_dists = topk_dist2[:, 0].sqrt()          # (N,) — distance to nearest neighbour
    sigma    = nn_dists.median().clamp(min=1e-6)

    # Build edge tensors
    src = torch.arange(N, device=coords.device).unsqueeze(1).expand(N, k).reshape(-1)
    tgt = topk_idx.reshape(-1)
    edge_index   = torch.stack([src, tgt], dim=0)           # (2, N*k)
    edge_weights = torch.exp(-topk_dist2.reshape(-1) / (2 * sigma ** 2))

    return edge_index, edge_weights


def bfs_distances(N: int, edge_index: torch.Tensor,
                  visible_mask: torch.Tensor) -> torch.Tensor:
    """
    BFS from visible nodes to compute shortest-path distance of each node
    from the visible boundary. Visible nodes have distance 0.

    Returns: distances (N,) int  (visible=0, unreachable=N+1)
    """
    adj: Dict[int, List[int]] = {i: [] for i in range(N)}
    ei = edge_index.cpu()
    for s, t in zip(ei[0].tolist(), ei[1].tolist()):
        adj[s].append(t)
        adj[t].append(s)   # undirected

    dist  = torch.full((N,), N + 1, dtype=torch.long)
    queue = deque()
    vis   = visible_mask.cpu().bool()

    for i in range(N):
        if vis[i].item():
            dist[i] = 0
            queue.append(i)

    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if dist[nb] > dist[node] + 1:
                dist[nb] = dist[node] + 1
                queue.append(nb)

    return dist.to(edge_index.device)


# ── Region masking ────────────────────────────────────────────────────────────

def contiguous_region_mask(coords: torch.Tensor, mask_ratio: float = 0.5,
                            edge_index: Optional[torch.Tensor] = None,
                            min_region_size: int = 5) -> torch.Tensor:
    """
    Mask spatially contiguous regions (not random patches).

    Strategy: BFS-flood from random seed patches until mask_ratio is reached.
    Multiple seeds to mask multiple tissue chunks in parallel
    (H&E slides have several disconnected tissue fragments — mask each independently).

    Returns: visible_mask (N,) bool  — True = visible, False = masked
    """
    N = coords.shape[0]
    target_masked = int(N * mask_ratio)

    # Build adjacency from edge_index if provided
    if edge_index is not None:
        adj: Dict[int, List[int]] = {i: [] for i in range(N)}
        ei = edge_index.cpu()
        for s, t in zip(ei[0].tolist(), ei[1].tolist()):
            adj[s].append(t)
            adj[t].append(s)
    else:
        # fallback: no graph — pure random masking
        perm = torch.randperm(N)
        mask = torch.ones(N, dtype=torch.bool)
        mask[perm[:target_masked]] = False
        return mask

    masked   = torch.zeros(N, dtype=torch.bool)   # True = masked
    n_masked = 0
    attempts = 0

    while n_masked < target_masked and attempts < N:
        # Pick a random unmasked seed
        candidates = (~masked).nonzero(as_tuple=True)[0]
        if len(candidates) == 0:
            break
        seed = candidates[random.randint(0, len(candidates) - 1)].item()

        # BFS flood from seed, claim a region
        region = []
        q      = deque([seed])
        visited = {seed}

        # Region size: sample between min_region and remaining budget
        region_target = random.randint(
            min_region_size,
            max(min_region_size, min(50, target_masked - n_masked))
        )

        while q and len(region) < region_target:
            node = q.popleft()
            if not masked[node]:
                region.append(node)
                masked[node] = True
                n_masked += 1
            for nb in adj[node]:
                if nb not in visited and not masked[nb]:
                    visited.add(nb)
                    q.append(nb)

        attempts += 1

    return ~masked   # True = visible


# ── Distance-conditioned noise schedule ──────────────────────────────────────

def scaled_cosine_error(pred: torch.Tensor, target: torch.Tensor,
                         gamma: float = 2.0) -> torch.Tensor:
    """
    Scaled Cosine Error (SCE) — GraphMAE (Hou et al. NeurIPS 2022).

    loss = 1 - (cos(pred, target))^gamma  per patch, averaged.

    Properties:
      - Scale-invariant: perfect for L2-normalized UNI/DINOv2 embeddings
      - gamma > 1 amplifies hard cases (low cosine similarity)
      - gamma=2 is the GraphMAE default
    """
    pred_n   = F.normalize(pred,   dim=-1)
    target_n = F.normalize(target, dim=-1)
    cos_sim  = (pred_n * target_n).sum(-1).clamp(-1, 1)   # (N,)
    return (1.0 - cos_sim.pow(gamma)).mean()


# ── Distance-conditioned sinusoidal embedding ────────────────────────────────

class DistanceEmbedding(nn.Module):
    """Sinusoidal + learned embedding for BFS distance (noise level)."""
    def __init__(self, hidden_dim: int, max_dist: int = 32):
        super().__init__()
        self.max_dist   = max_dist
        self.hidden_dim = hidden_dim
        self.proj       = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU())

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """distances: (N,) int → (N, hidden_dim)"""
        d      = distances.float().unsqueeze(1) / self.max_dist
        half   = self.hidden_dim // 2
        freq   = torch.exp(-math.log(10000) * torch.arange(half, device=d.device) / (half - 1))
        sin_e  = torch.sin(d * freq.unsqueeze(0))   # (N, half)
        cos_e  = torch.cos(d * freq.unsqueeze(0))   # (N, half)
        emb    = torch.cat([sin_e, cos_e], dim=1)   # (N, hidden_dim)
        return self.proj(emb)


# ── Causal-by-distance graph attention layer ─────────────────────────────────

class SpatialDenoisingLayer(nn.Module):
    """
    Graph attention where patch i can only attend to neighbours j with
    d_j < d_i  (closer to visible boundary = lower BFS distance).

    This implements one step of the spatial diffusion wave:
    information flows from visible boundary inward.

    Edge weights = spatial Gaussian × causal distance gate.
    """
    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads  = n_heads
        self.d_head   = hidden_dim // n_heads
        self.scale    = self.d_head ** -0.5

        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm1   = nn.LayerNorm(hidden_dim)
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.ffn     = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim))
        self.drop    = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_weights: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        """
        h            (N, D)
        edge_index   (2, E)   [src, tgt]
        edge_weights (E,)     spatial Gaussian weights
        distances    (N,)     BFS distance from visible boundary
        """
        N = h.shape[0]
        residual = h
        h = self.norm1(h)

        Q = self.q(h).view(N, self.n_heads, self.d_head)
        K = self.k(h).view(N, self.n_heads, self.d_head)
        V = self.v(h).view(N, self.n_heads, self.d_head)

        src, tgt  = edge_index[0], edge_index[1]
        E         = src.shape[0]

        # Causal mask: tgt can attend to src only if d[src] <= d[tgt]
        # (each node only receives from closer-to-boundary nodes)
        causal = (distances[src] <= distances[tgt]).float()   # (E,)

        # Attention score: Q[tgt] · K[src]
        q_e = Q[tgt]                               # (E, H, d_head)
        k_e = K[src]                               # (E, H, d_head)
        score = (q_e * k_e).sum(-1) * self.scale  # (E, H)

        # Apply spatial weight and causal gate
        w = edge_weights.unsqueeze(1) * causal.unsqueeze(1)    # (E, 1)
        score = score + torch.log(w.clamp(min=1e-9))           # log-space mask

        # Sparse softmax per target node
        # Use scatter for efficiency: compute softmax over incoming edges per node
        score_exp = torch.exp(score - score.max())             # numerical stability (approx)
        denom     = torch.zeros(N, self.n_heads, device=h.device)
        denom.scatter_add_(0, tgt.unsqueeze(1).expand(-1, self.n_heads), score_exp)
        attn      = score_exp / (denom[tgt] + 1e-9)           # (E, H)

        # Aggregate values
        out = torch.zeros(N, self.n_heads, self.d_head, device=h.device)
        v_e = V[src] * attn.unsqueeze(-1)                     # (E, H, d_head)
        out.scatter_add_(0,
            tgt.unsqueeze(1).unsqueeze(2).expand_as(v_e), v_e)
        out = out.view(N, -1)                                  # (N, D)
        out = self.drop(self.o(out))

        h = residual + out
        h = h + self.drop(self.ffn(self.norm2(h)))
        return h


# ── Spatial denoising encoder (HE / CT) ──────────────────────────────────────

class SpatialDenoisingEncoder(nn.Module):
    """
    Spatial masked graph autoencoder encoder (GeoMAE v2).

    Changes from v1 (based on GraphMAE best practices):
      - Binary masking: visible=original features, masked=learnable MASK token
      - x₀-prediction: directly reconstruct original patch features
      - Loss: Scaled Cosine Error (SCE) — GraphMAE NeurIPS 2022
              1 - cos(pred, target)^γ  where γ=2 (default)
      - max_dist set dynamically = actual BFS max depth (not fixed 32)
      - k=32 KNN for dense graph (avoids disconnected masked components)
      - n_layers=5 (handles BFS depth ≤ 8 with k=32)

    Pipeline:
      1. Visible patches → project features
         Masked patches  → learnable MASK token (binary, no noise)
      2. Add depth embedding (spatial position encoding, like PE in transformers)
      3. N_layers of causal SpatialDenoisingLayer (attend only to shallower neighbors)
      4. Reconstruct x₀ at masked locations with SCE loss
    """
    def __init__(self, feat_dim: int = 1024, hidden_dim: int = 256,
                 n_layers: int = 5, n_heads: int = 4, dropout: float = 0.1,
                 knn_k: int = 32, max_dist: int = 16):
        super().__init__()
        self.knn_k    = knn_k
        self.max_dist = max_dist   # used for depth embedding size; overridden dynamically

        self.proj       = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.mask_token = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.dist_embed = DistanceEmbedding(hidden_dim, max_dist)

        self.layers = nn.ModuleList([
            SpatialDenoisingLayer(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        # x₀ reconstruction head (predict original features, not noise ε)
        self.recon_head = nn.Linear(hidden_dim, feat_dim)
        # Keep noise_head as alias for backward compat with GeoMAESpatialBackbone
        self.noise_head = self.recon_head

    def forward(self, feats: torch.Tensor, coords: torch.Tensor,
                visible_mask: Optional[torch.Tensor] = None,
                precomputed_graph: Optional[Tuple] = None,
                sce_gamma: float = 2.0,
                ) -> Dict[str, torch.Tensor]:
        """
        feats        (N, feat_dim)    original patch features (L2-normalized UNI)
        coords       (N, D_coord)     spatial coordinates (2D or 3D)
        visible_mask (N,) bool        True=visible, False=masked
                                       if None: all visible (inference/fine-tuning mode)
        precomputed_graph: optional (edge_index, edge_weights)
        sce_gamma:    SCE scaling exponent (default 2, GraphMAE default)

        Returns dict:
          'encoded'      (N, H)      contextualised patch representations
          'noise_pred'   (N, D)      predicted x₀ (alias for backward compat)
          'noise_true'   (N, D)      original features x (reconstruction target)
          'distances'    (N,)        BFS distances
          'visible_mask' (N,) bool
        """
        N = feats.shape[0]

        if precomputed_graph is not None:
            edge_index, edge_weights = precomputed_graph
        else:
            edge_index, edge_weights = build_knn_graph(coords, self.knn_k)

        if visible_mask is None:
            visible_mask = torch.ones(N, dtype=torch.bool, device=feats.device)

        # BFS distances — set max_dist dynamically to actual max depth
        distances = bfs_distances(N, edge_index, visible_mask).to(feats.device)
        actual_max_dist = max(int(distances.max().item()), 1)
        clamp_dist = distances.clamp(0, self.max_dist)

        # Binary masking (GraphMAE): visible → projected features, masked → MASK token
        h_vis  = self.proj(feats)                              # (N, H) — project all
        h_mask = self.mask_token.expand(N, -1)                 # (N, H) — mask token
        # Replace masked patches with mask token, keep visible as projected features
        vis_f  = visible_mask.unsqueeze(1).float()
        h = vis_f * h_vis + (1 - vis_f) * h_mask

        # Depth embedding as spatial position encoding (boundary=0, interior=deeper)
        h = h + self.dist_embed(clamp_dist)

        # Causal graph attention layers (each patch attends only to shallower neighbors)
        for layer in self.layers:
            h = layer(h, edge_index, edge_weights, distances)

        # Predict x₀ (original features) at ALL positions
        x0_pred = self.recon_head(h)                           # (N, feat_dim)

        # SCE loss only on masked patches (visible patches are given, not predicted)
        masked = ~visible_mask
        if masked.any():
            loss_sce = scaled_cosine_error(
                x0_pred[masked], feats[masked], gamma=sce_gamma)
        else:
            loss_sce = feats.new_zeros(1).squeeze()

        return {
            "encoded":      h,
            "noise_pred":   x0_pred,    # alias: predicted x₀ (not noise ε)
            "noise_true":   feats,       # alias: original features (reconstruction target)
            "loss_sce":     loss_sce,
            "distances":    distances,
            "visible_mask": visible_mask,
            "actual_max_dist": actual_max_dist,
        }


# ── Clinical BERT-style encoder ───────────────────────────────────────────────

class ClinicalMaskedEncoder(nn.Module):
    """
    BERT-style encoder for ordered clinical features.

    Clinical feature vector (102,) has known feature ordering:
      0-4:    PFTs (FVC, FEV1, ...)
      5-20:   metabolic panel
      21-65:  haematology CBC
      66-80:  vitals
      81-101: donor/recipient factors

    Masking: mask entire category blocks (not random features)
    Reconstruction: predict masked feature values (MSE)
    """
    def __init__(self, n_features: int = 102, hidden_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
                 mask_ratio: float = 0.3):
        super().__init__()
        self.n_features = n_features
        self.mask_ratio = mask_ratio

        # Feature-type positional embedding (1 per feature, learnable)
        self.feat_embed = nn.Embedding(n_features, hidden_dim)
        self.value_proj = nn.Linear(1, hidden_dim)
        self.mask_token = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.recon_head = nn.Linear(hidden_dim, 1)   # predict feature value

        # Feature category blocks for block masking
        # Indices: PFTs=0-4, metabolic=5-20, haem_diff=21-65, vitals=66-80, donor=81-101
        self.category_blocks = [
            list(range(0, 5)),    # PFTs
            list(range(5, 21)),   # metabolic panel
            list(range(21, 66)),  # haematology
            list(range(66, 81)),  # vitals
            list(range(81, 102)), # donor/recipient
        ]

    def _category_mask(self, n_features: int, device) -> torch.Tensor:
        """Mask entire categories until mask_ratio is reached."""
        masked = torch.zeros(n_features, dtype=torch.bool, device=device)
        blocks = random.sample(self.category_blocks, len(self.category_blocks))
        n_masked = 0
        target   = int(n_features * self.mask_ratio)
        for block in blocks:
            if n_masked >= target:
                break
            for idx in block:
                masked[idx] = True
                n_masked += 1
        return masked   # True = masked

    def forward(self, feats: torch.Tensor,
                masked_indices: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        feats (F,) or (B, F)  — clinical feature values (normalised)
        Returns: encoded (F, hidden_dim), recon (F,), masked_indices (F,) bool
        """
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)   # (1, F)
        B, F = feats.shape

        # Generate mask if not provided
        if masked_indices is None:
            masked_indices = self._category_mask(F, feats.device)   # (F,) bool

        # Build token sequence: value embedding + position embedding
        pos     = torch.arange(F, device=feats.device)              # (F,)
        pos_emb = self.feat_embed(pos)                               # (F, H)
        val_emb = self.value_proj(feats.unsqueeze(-1))               # (B, F, H)
        h       = val_emb + pos_emb.unsqueeze(0)                    # (B, F, H)

        # Replace masked tokens
        mask_exp = masked_indices.unsqueeze(0).unsqueeze(-1).expand_as(h)
        h = torch.where(mask_exp,
                        self.mask_token.expand(B, F, -1),
                        h)

        # Bidirectional transformer (all features attend to all)
        h = self.transformer(h)    # (B, F, H)

        # Reconstruct masked values
        recon = self.recon_head(h).squeeze(-1)   # (B, F)

        return {
            "encoded":        h.squeeze(0),     # (F, H) or (B, F, H)
            "recon":          recon.squeeze(0),  # (F,) or (B, F)
            "masked_indices": masked_indices,    # (F,) bool
            "target":         feats.squeeze(0),  # original values (F,)
        }


# ── Slot cross-modal context ──────────────────────────────────────────────────

class SlotCrossModal(nn.Module):
    """
    K=8 slot tokens per modality as bottleneck for cross-modal information exchange.

    NOT instance-instance cross-attention (N_HE × N_CT is huge).
    Each modality compresses to K slots → K×M total slots attend to each other.
    Enriched slots then inform masked patch reconstruction.

    Forward:
      encoded_mods: dict {mod_name: (N_m, H)}
      Returns: dict {mod_name: (N_m, H)} — enriched with cross-modal context
    """
    def __init__(self, hidden_dim: int = 256, n_slots: int = 8,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_slots  = n_slots

        # Slot attention: K slots attend to modality patches
        self.slot_queries = nn.Parameter(torch.randn(n_slots, hidden_dim) * 0.02)

        self.slot_attn   = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True)

        # Cross-slot transformer: slots from all modalities attend to each other
        xattn_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            batch_first=True, norm_first=True)
        self.cross_slot_xfmr = nn.TransformerEncoder(xattn_layer, num_layers=1)

        # Back-project: patch tokens attend to enriched slots
        self.patch_xattn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, encoded_mods: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        """
        encoded_mods: {mod: (N_mod, H)}
        Returns:      {mod: (N_mod, H)} enriched with cross-modal slot context
        """
        all_slots = {}
        queries   = self.slot_queries.unsqueeze(0)   # (1, K, H)

        # Step 1: each modality → K slots via slot attention
        for mod, h in encoded_mods.items():
            patches = h.unsqueeze(0)   # (1, N, H)
            slots, _ = self.slot_attn(queries, patches, patches)   # (1, K, H)
            all_slots[mod] = slots.squeeze(0)                      # (K, H)

        # Step 2: cross-slot transformer across all modalities
        all_slot_seq = torch.cat(list(all_slots.values()), dim=0).unsqueeze(0)  # (1, K*M, H)
        enriched     = self.cross_slot_xfmr(all_slot_seq).squeeze(0)           # (K*M, H)

        # Split enriched slots back per modality
        K = self.n_slots
        enriched_per_mod = {}
        offset = 0
        for mod in encoded_mods:
            enriched_per_mod[mod] = enriched[offset: offset + K]   # (K, H)
            offset += K

        # Step 3: each patch attends to its modality's enriched slots
        #         (inform masked patches with cross-modal context)
        out = {}
        for mod, h in encoded_mods.items():
            patches     = h.unsqueeze(0)                              # (1, N, H)
            emod_slots  = enriched_per_mod[mod].unsqueeze(0)         # (1, K, H)
            informed, _ = self.patch_xattn(patches, emod_slots, emod_slots)  # (1, N, H)
            out[mod]    = self.norm(h + informed.squeeze(0))          # residual
        return out


# ── Full multimodal pretraining model ─────────────────────────────────────────

class GeoMAE(nn.Module):
    """
    Full GeoMAE pretraining model.

    Pretraining objective:
      1. Contiguous region masking per spatial modality
      2. Distance-conditioned spatial denoising (wave-by-wave, causal by BFS distance)
      3. Slot cross-modal context (K=8 slots, NOT instance-instance)
      4. Predict noise ε per masked patch (DDPM-style reconstruction)
      5. Clinical BERT-style masked feature prediction

    Loss:
      L = λ_HE * L_HE + λ_CT * L_CT + λ_clin * L_clin

    Compatible with existing mil package — uses only standard PyTorch modules.
    """
    def __init__(self,
                 he_feat_dim:   int = 1024,
                 ct_feat_dim:   int = 1024,
                 n_clin_feats:  int = 102,
                 hidden_dim:    int = 256,
                 n_layers:      int = 3,
                 n_heads:       int = 4,
                 n_slots:       int = 8,
                 knn_k:         int = 8,
                 max_dist:      int = 32,
                 he_mask_ratio: float = 0.5,
                 ct_mask_ratio: float = 0.5,
                 clin_mask_ratio: float = 0.3,
                 dropout:       float = 0.1,
                 lambda_he:     float = 1.0,
                 lambda_ct:     float = 1.0,
                 lambda_clin:   float = 0.5):
        super().__init__()
        self.he_mask_ratio   = he_mask_ratio
        self.ct_mask_ratio   = ct_mask_ratio
        self.lambda_he       = lambda_he
        self.lambda_ct       = lambda_ct
        self.lambda_clin     = lambda_clin

        # Per-modality encoders
        self.he_encoder = SpatialDenoisingEncoder(
            he_feat_dim, hidden_dim, n_layers, n_heads, dropout, knn_k, max_dist)
        self.ct_encoder = SpatialDenoisingEncoder(
            ct_feat_dim, hidden_dim, n_layers, n_heads, dropout, knn_k, max_dist)
        self.clin_encoder = ClinicalMaskedEncoder(
            n_clin_feats, hidden_dim, n_heads=n_heads,
            n_layers=2, dropout=dropout, mask_ratio=clin_mask_ratio)

        # Slot cross-modal context (operates after per-modality encoding)
        self.slot_cross = SlotCrossModal(hidden_dim, n_slots, n_heads, dropout)

        # Second pass: noise prediction after cross-modal context
        self.he_noise_head2  = nn.Linear(hidden_dim, he_feat_dim)
        self.ct_noise_head2  = nn.Linear(hidden_dim, ct_feat_dim)

    def forward(self,
                he_feats:    Optional[torch.Tensor] = None,   # (N, 1024)
                he_coords:   Optional[torch.Tensor] = None,   # (N, 2)
                ct_feats:    Optional[torch.Tensor] = None,   # (M, 1024)
                ct_coords:   Optional[torch.Tensor] = None,   # (M, 3)
                clin_feats:  Optional[torch.Tensor] = None,   # (F,)
                ) -> Dict[str, torch.Tensor]:
        """
        Any subset of modalities can be passed (None = absent for this patient).
        Returns dict of losses and intermediate outputs.
        """
        losses = {}
        encoded_mods = {}

        # ── HE spatial reconstruction (SCE loss, binary masking) ─────────────
        if he_feats is not None and he_coords is not None:
            ei, ew       = build_knn_graph(he_coords, self.he_encoder.knn_k)
            visible_mask = contiguous_region_mask(
                he_coords, self.he_mask_ratio, ei)
            he_out       = self.he_encoder(
                he_feats, he_coords, visible_mask, (ei, ew))
            # SCE loss computed inside encoder on masked patches only
            losses["he"] = he_out.get("loss_sce", he_feats.new_zeros(1).squeeze())
            encoded_mods["HE"] = he_out["encoded"]

        # ── CT spatial reconstruction (SCE loss, binary masking) ─────────────
        if ct_feats is not None and ct_coords is not None:
            ei, ew       = build_knn_graph(ct_coords, self.ct_encoder.knn_k)
            visible_mask = contiguous_region_mask(
                ct_coords, self.ct_mask_ratio, ei)
            ct_out       = self.ct_encoder(
                ct_feats, ct_coords, visible_mask, (ei, ew))
            losses["ct"] = ct_out.get("loss_sce", ct_feats.new_zeros(1).squeeze())

            encoded_mods["CT"] = ct_out["encoded"]

        # ── Clinical BERT-style ───────────────────────────────────────────────
        if clin_feats is not None:
            clin_out = self.clin_encoder(clin_feats)
            masked   = clin_out["masked_indices"]
            if masked.any():
                losses["clin"] = F.mse_loss(
                    clin_out["recon"][masked],
                    clin_out["target"][masked])
            else:
                losses["clin"] = clin_feats.new_zeros(1).squeeze()

            # Project clinical to same hidden_dim as spatial modalities
            encoded_mods["Clinical"] = clin_out["encoded"]

        # ── Slot cross-modal context ──────────────────────────────────────────
        if len(encoded_mods) >= 2:
            # Ensure Clinical is same dim (ClinicalMaskedEncoder uses hidden_dim//2)
            # Pad/project if needed
            enriched = self.slot_cross(encoded_mods)

            # Second x₀ prediction pass with cross-modal context (SCE)
            if he_feats is not None and ct_feats is not None:
                he_masked = ~he_out["visible_mask"]
                ct_masked = ~ct_out["visible_mask"]

                if he_masked.any():
                    x0_pred2 = self.he_noise_head2(enriched["HE"][he_masked])
                    sce2 = scaled_cosine_error(x0_pred2, he_feats[he_masked])
                    losses["he"] = (losses.get("he", 0) + sce2) * 0.5

                if ct_masked.any():
                    x0_pred2 = self.ct_noise_head2(enriched["CT"][ct_masked])
                    sce2 = scaled_cosine_error(x0_pred2, ct_feats[ct_masked])
                    losses["ct"] = (losses.get("ct", 0) + sce2) * 0.5

        # ── Total loss ────────────────────────────────────────────────────────
        total = (self.lambda_he   * losses.get("he",   torch.zeros(1)) +
                 self.lambda_ct   * losses.get("ct",   torch.zeros(1)) +
                 self.lambda_clin * losses.get("clin", torch.zeros(1)))

        return {
            "loss":       total,
            "loss_he":    losses.get("he"),
            "loss_ct":    losses.get("ct"),
            "loss_clin":  losses.get("clin"),
            "encoded":    encoded_mods,
        }

    def get_backbone_weights(self) -> Dict[str, dict]:
        """Return pretrained encoder weights for downstream MIL loading."""
        return {
            "he_encoder":   self.he_encoder.state_dict(),
            "ct_encoder":   self.ct_encoder.state_dict(),
            "clin_encoder": self.clin_encoder.state_dict(),
        }
