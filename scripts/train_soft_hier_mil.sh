#!/usr/bin/env bash
#SBATCH --job-name=soft_hier_mil
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G
#SBATCH --gres=gpu:1 --constraint="a100_40gb|a100_80gb|h100_80gb"
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/soft_hier_mil/train_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/soft_hier_mil/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

mkdir -p /lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/soft_hier_mil

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
"""
Soft Hierarchical Spatial MIL vs existing 2-level Spatial ABMIL.

SoftHierarchicalMIL:
  - Flattens all patches from all DBSCAN clusters (ignores hard clustering)
  - Projects patch features: Linear(1024→256) + LayerNorm + GELU
  - L=4 rounds of cross-attention with growing Gaussian spatial bias:
      Level 1: sigma=500px   (immediate neighbours, ~4 tile widths)
      Level 2: sigma=2000px  (local region)
      Level 3: sigma=8000px  (tissue compartment)
      Level 4: sigma=inf     (global, no spatial mask)
  - Each level: MultiHeadAttn(Q=K=V=h, bias=log_G) + residual + FFN
  - GatedAttentionPool over final patch representations → slide rep
  - Linear(256→1) → sigmoid → ACR probability

Gaussian bias added to attention logits (log-domain):
  log_G_ij = -||pos_i - pos_j||² / (2σ²)
  attn_w_ij ∝ exp(Q_i·K_j/sqrt(d) + log_G_ij)

Training: StratifiedGroupKFold(n_splits=5), all 5 folds.
Comparison: retrain existing SpatialABMIL2Level on same splits for fair comparison.
"""
import math, json, time, random
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.spatial import cKDTree

FEAT_DIR = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/slide_cluster_feats")
OUTDIR   = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/soft_hier_mil")
OUTDIR.mkdir(exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
IN_DIM       = 1024
HIDDEN       = 256
N_HEADS      = 4
DROPOUT      = 0.25
ATTN_DROPOUT = 0.1
SIGMAS       = (500.0, 2000.0, 8000.0, None)  # None = global (no spatial mask)
MAX_N          = 1024    # max patches per slide (subsample if more)
TARGET_NBHD_SIZE = 64   # SLIC: target patches per neighbourhood
MIN_NBHD_SIZE    = 10   # SLIC: merge clusters smaller than this into nearest
N_SPLITS       = 5
JOINT_EPOCHS   = 250     # was 150 — longer joint training
PATIENCE       = 30      # max no-improve checks before early stop
PATIENCE_EVERY = 5       # evaluate val every N epochs (was 10)
EVAL_SEED      = 42      # fixed seed for deterministic eval subsampling
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
GRAD_ACCUM     = 8
SEED           = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

def set_seeds(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if device.type == "cuda": torch.cuda.manual_seed_all(s)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL A: Soft Hierarchical Spatial MIL (new)
# ══════════════════════════════════════════════════════════════════════════════

class SpatialCrossAttnBlock(nn.Module):
    """
    One level of the hierarchy.
    Applies multi-head self-attention with additive Gaussian log-bias,
    then a two-layer FFN. Both have residual connections and LayerNorm.

    The Gaussian bias: log_G_ij = -||pos_i - pos_j||^2 / (2*sigma^2)
    Added to QK^T / sqrt(d) before softmax, so attention decays with distance.
    sigma=None means no spatial mask (global attention).

    Pairs whose Gaussian value < exp(LOG_THRESH) are masked to -inf,
    making their softmax weight exactly zero without contributing to gradients.
    This corresponds to dist > sigma * sqrt(-2 * LOG_THRESH).
    """
    LOG_THRESH = -4.0   # exp(-4) ≈ 1.8% — pairs below this are hard-masked

    def __init__(self, dim, n_heads, attn_dropout=0.1, ffn_dropout=0.1, sigma=None):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, n_heads, dropout=attn_dropout,
                                           batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(ffn_dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(ffn_dropout))
        self.sigma = sigma
        # precompute: max dist² beyond which we hard-mask
        if sigma is not None:
            self._max_dist2 = -2.0 * sigma ** 2 * self.LOG_THRESH  # scalar

    def _gaussian_bias(self, coords):
        """coords: (N, 2) float tensor on same device. Returns (N, N) additive log-bias."""
        if self.sigma is None:
            return None
        diff  = coords.unsqueeze(0) - coords.unsqueeze(1)   # (N, N, 2)
        dist2 = (diff ** 2).sum(-1)                          # (N, N)
        bias  = -dist2 / (2.0 * self.sigma ** 2)             # (N, N)
        bias[dist2 > self._max_dist2] = float('-inf')        # hard mask far pairs
        return bias

    def forward(self, x, coords):
        """x: (1, N, dim), coords: (N, 2)  →  (1, N, dim)"""
        bias = self._gaussian_bias(coords)   # (N, N) or None
        out, _ = self.attn(x, x, x, attn_mask=bias)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


class GatedPool(nn.Module):
    """Gated attention pooling: (N, d) → (d,) + attention weights (N,)."""
    def __init__(self, dim):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(dim, dim // 2), nn.Sigmoid())
        self.w = nn.Linear(dim // 2, 1, bias=False)

    def forward(self, h):
        a = self.w(self.V(h) * self.U(h))          # (N, 1)
        a = torch.softmax(a, dim=0)
        return (a * h).sum(0), a.squeeze(-1)        # (dim,), (N,)


class SoftHierarchicalMIL(nn.Module):
    """
    Soft Hierarchical Spatial MIL.

    Forward:
      feats  (N, 1024)   — patch feature vectors
      coords (N, 2)      — spatial coordinates in pixel space

    Returns: logit (scalar), attention_weights (N,)
    """
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, dropout=DROPOUT,
                 n_heads=N_HEADS, sigmas=SIGMAS, attn_dropout=ATTN_DROPOUT):
        super().__init__()
        self.proj   = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(dropout))
        self.levels = nn.ModuleList([
            SpatialCrossAttnBlock(hidden, n_heads,
                                  attn_dropout=attn_dropout,
                                  ffn_dropout=dropout,
                                  sigma=s)
            for s in sigmas
        ])
        self.pool = GatedPool(hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, feats, coords):
        h = self.proj(feats).unsqueeze(0)     # (1, N, hidden)
        for level in self.levels:
            h = level(h, coords)              # (1, N, hidden)
        h = h.squeeze(0)                      # (N, hidden)
        rep, attn = self.pool(h)              # (hidden,), (N,)
        return self.head(rep).squeeze(), attn


# ══════════════════════════════════════════════════════════════════════════════
# MODEL B: Existing 2-level Spatial ABMIL (baseline) + N-level generalisation
# ══════════════════════════════════════════════════════════════════════════════

class GatedAttentionPool(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(in_dim, hidden), nn.Sigmoid())
        self.w = nn.Linear(hidden, 1, bias=False)
    def forward(self, h):
        a = self.w(self.V(h) * self.U(h))
        a = torch.softmax(a, dim=1)
        return (a * h).sum(dim=1), a.squeeze(-1)

class SpatialABMIL2Level(nn.Module):
    def __init__(self, in_dim=1024, hidden=256, dropout=0.25):
        super().__init__()
        self.patch_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.patch_attn = GatedAttentionPool(hidden, hidden // 2)
        self.nbhd_proj  = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.nbhd_attn  = GatedAttentionPool(hidden, hidden // 2)
        self.head       = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, clusters):
        nbhd_reps, patch_attns = [], []
        for c in clusters:
            h = self.patch_proj(c.unsqueeze(0))
            rep, pa = self.patch_attn(h)
            nbhd_reps.append(rep); patch_attns.append(pa.squeeze(0))
        H = self.nbhd_proj(torch.stack(nbhd_reps, dim=1))
        slide_rep, na = self.nbhd_attn(H)
        return self.head(slide_rep).squeeze(-1), na.squeeze(0), patch_attns


def build_slic_multilevel(all_coords: np.ndarray, fine_labels: np.ndarray,
                           coarser_sizes: list) -> list:
    """
    Build coarser SLIC groupings from fine SLIC labels.

    all_coords:    (N, 2)  original patch pixel coordinates
    fine_labels:   (N,)    fine SLIC cluster assignments (0..K_fine-1)
    coarser_sizes: list of target patch counts per super-cluster at each coarser level
                   e.g. [512] → 3-level,  [512, 2048] → 4-level

    Returns:
        group_ids: list of lists, group_ids[l][k] = super-cluster id of cluster k at level l+1
                   len(group_ids) == len(coarser_sizes)
    """
    K_fine = int(fine_labels.max()) + 1
    N      = len(all_coords)
    # Centroid of each fine cluster in pixel space
    centroids = np.array([all_coords[fine_labels == ci].mean(0)
                          for ci in range(K_fine)], dtype=np.float64)

    group_ids = []
    current_centroids = centroids   # (K_current, 2)

    for target_patch_size in coarser_sizes:
        K_current = len(current_centroids)
        # How many super-clusters we want at this level
        K_target  = max(2, int(np.ceil(N / target_patch_size)))
        # SLIC target: K_current centroids → K_target groups
        # → target_size for SLIC on centroids = K_current / K_target (fine-clusters per super)
        fine_per_super = max(2, int(np.ceil(K_current / K_target)))

        super_labels = spatial_slic(current_centroids,
                                     target_size=fine_per_super,
                                     min_size=1,    # never merge at centroid level
                                     max_iter=15)
        group_ids.append(super_labels.tolist())

        # Centroids of super-clusters for next level
        K_super = int(super_labels.max()) + 1
        current_centroids = np.array([
            current_centroids[super_labels == g].mean(0)
            for g in range(K_super)
        ], dtype=np.float64)

    return group_ids


class SpatialABMILNLevel(nn.Module):
    """
    N-level hierarchical spatial ABMIL.

    Architecture (for n_levels=3):
      patches → GatedPool → fine_cluster_reps   (level 0→1)
      fine_cluster_reps → group by l1_ids → GatedPool → super_reps  (level 1→2)
      super_reps → GatedPool → slide_rep  (level 2→slide)
      head(slide_rep) → logit

    Data: feats_list (K fine clusters) + group_ids (L-1 lists of group assignments)
    built by build_multilevel().
    """
    def __init__(self, n_levels=3, in_dim=1024, hidden=256, dropout=0.25):
        super().__init__()
        assert n_levels >= 2
        self.n_levels   = n_levels
        self.patch_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        # n_levels pools: patch→L1, L1→L2, ..., L(n-1)→slide
        self.pools = nn.ModuleList([
            GatedAttentionPool(hidden, hidden // 2) for _ in range(n_levels)
        ])
        self.norms = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout))
            for _ in range(n_levels)
        ])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def _pool_groups(self, reps, ids, pool, norm, device):
        """reps: list of (hidden,) tensors, ids: list of int group labels."""
        n_groups = max(ids) + 1
        group_reps, group_attns = [], []
        for g in range(n_groups):
            members = [reps[k] for k, gid in enumerate(ids) if gid == g]
            if not members:
                members = [torch.zeros(reps[0].shape, device=device)]
            H = norm(torch.stack(members, dim=0).unsqueeze(0))  # (1, M, hidden)
            rep, attn = pool(H)
            group_reps.append(rep.squeeze(0))
            group_attns.append(attn.squeeze(0))
        return group_reps, group_attns

    def forward(self, feats_list, group_ids):
        """
        feats_list : list of K tensors (N_k, 1024)
        group_ids  : list of (n_levels-1) lists of int, built by build_multilevel()
        Returns: logit, top_level_attns (list), all_level_attns (list of lists)
        """
        device = feats_list[0].device

        # Level 0→1: pool patches within each fine cluster
        cluster_reps = []
        for feats in feats_list:
            h = self.patch_proj(feats).unsqueeze(0)        # (1, N_k, hidden)
            h = self.norms[0](h)
            rep, _ = self.pools[0](h)
            cluster_reps.append(rep.squeeze(0))            # (hidden,)

        all_attns = []
        current_reps = cluster_reps
        for lvl, ids in enumerate(group_ids):
            current_reps, attns = self._pool_groups(
                current_reps, ids, self.pools[lvl + 1], self.norms[lvl + 1], device)
            all_attns.append(attns)

        # Final pool to slide rep
        H = self.norms[-1](torch.stack(current_reps, dim=0).unsqueeze(0))  # (1, G, hidden)
        slide_rep, top_attn = self.pools[-1](H)
        all_attns.append([top_attn.squeeze(0)])

        return self.head(slide_rep.squeeze(0)).squeeze(), top_attn.squeeze(0), all_attns


# ══════════════════════════════════════════════════════════════════════════════
# MODEL C: Point Transformer-style KNN Spatial MIL
# ══════════════════════════════════════════════════════════════════════════════
#
# Efficient O(N·K) attention — no N×N matrix ever built.
# Based on Point Transformer (Zhao et al. ICCV 2021) adapted for WSI patches.
#
# For each patch i, only its K nearest neighbours are attended to:
#   q_i  = W_q(x_i)                              (d_h,)
#   k_j  = W_k(x_j) + pos_enc(p_j - p_i)        (K, d_h)  ← position-aware
#   v_j  = W_v(x_j) + pos_enc(p_j - p_i)        (K, d_h)  ← position-aware
#   a_ij = softmax( (q_i * k_j).sum(-1) / √d )  (K,)
#   out_i = Σ_j a_ij * v_j                       (d_h,)
#
# KNN built with scipy cKDTree: O(N log N), precomputed once per slide.
# Receptive field grows automatically with depth (K-hop message passing):
#   layer 1 → K^1 patches,  layer 2 → K^2,  layer 3 → K^3,  layer 4 → K^4
#
# ABMIL readout at every layer → multi-scale interpretability:
#   layer 1 attn: which patches are predictive at instance/patch scale
#   layer 2 attn: which local regions are predictive
#   layer 3 attn: which tissue neighbourhoods are predictive
#   layer 4 attn: slide-level importance


def build_knn_idx(coords_np, k):
    """
    coords_np: (N, 2) numpy array of pixel coordinates.
    Returns: knn_idx (N, K) int64 — indices of K nearest neighbours for each patch.
    Uses scipy cKDTree: O(N log N), much faster than O(N²) brute-force.
    """
    tree = cKDTree(coords_np)
    # query k+1 to exclude self (distance 0), then drop it
    _, idx = tree.query(coords_np, k=k + 1)
    return idx[:, 1:]   # (N, K) — exclude self


class PointTransformerBlock(nn.Module):
    """
    Point Transformer attention over K spatial neighbours.
    O(N·K) memory and compute.

    Position encoding: relative displacement (Δx, Δy) → learned embedding
    added to both keys and values, making attention spatially aware without
    building any N×N structure.
    """
    def __init__(self, dim, k=8, dropout=0.1):
        super().__init__()
        self.k    = k
        self.dim  = dim
        self.scale = dim ** -0.5
        self.W_q  = nn.Linear(dim, dim, bias=False)
        self.W_k  = nn.Linear(dim, dim, bias=False)
        self.W_v  = nn.Linear(dim, dim, bias=False)
        self.W_o  = nn.Linear(dim, dim, bias=False)
        # relative position encoder: (Δx, Δy) → dim
        self.pos_enc = nn.Sequential(
            nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm1   = nn.LayerNorm(dim)
        self.norm2   = nn.LayerNorm(dim)
        self.ff      = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout))
        self.drop    = nn.Dropout(dropout)

    def forward(self, x, coords, knn_idx):
        """
        x:       (N, dim)
        coords:  (N, 2)   float, pixel coordinates
        knn_idx: (N, K)   long, precomputed K nearest neighbour indices

        Returns: (N, dim)
        """
        N, K = knn_idx.shape

        # ── relative position encoding ──────────────────────────────────────
        # coords of K neighbours for each patch: (N, K, 2)
        nbr_coords = coords[knn_idx]                          # (N, K, 2)
        rel_pos    = nbr_coords - coords.unsqueeze(1)         # (N, K, 2)
        pos_bias   = self.pos_enc(rel_pos)                    # (N, K, dim)

        # ── QKV projections ─────────────────────────────────────────────────
        q = self.W_q(x)                                       # (N, dim)
        k = self.W_k(x)[knn_idx] + pos_bias                  # (N, K, dim)
        v = self.W_v(x)[knn_idx] + pos_bias                  # (N, K, dim)

        # ── attention over K neighbours only ────────────────────────────────
        # (N, 1, dim) × (N, dim, K) → (N, 1, K) → (N, K)
        attn = (q.unsqueeze(1) * k).sum(-1) * self.scale     # (N, K)
        attn = torch.softmax(attn, dim=-1)                    # (N, K)

        # ── aggregate ───────────────────────────────────────────────────────
        out  = (attn.unsqueeze(-1) * v).sum(1)               # (N, dim)
        out  = self.drop(self.W_o(out))
        x    = self.norm1(x + out)
        x    = self.norm2(x + self.ff(x))
        return x


class SpatialKNNMIL(nn.Module):
    """
    Stacked Point Transformer blocks with ABMIL readout at each layer.

    KNN graph built once per slide with cKDTree (O(N log N)).
    Each block: O(N·K) attention — scales linearly, not quadratically.

    Depth creates hierarchy automatically (K-hop message passing):
      layer 1 → instance/patch context   (K^1 neighbours)
      layer 2 → local region context     (K^2 neighbours)
      layer 3 → tissue neighbourhood     (K^3 neighbours)
      layer 4 → slide-level context      (K^4 neighbours)

    ABMIL (GatedPool) after each layer:
      → separate prediction at each spatial scale
      → attention weights = importance map at that scale
    Final logit = softmax-weighted sum of layer logits (weights learned).
    """
    def __init__(self, in_dim=1024, hidden=256, dropout=0.25, k=8, n_layers=4):
        super().__init__()
        self.k        = k
        self.n_layers = n_layers
        self.proj     = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([
            PointTransformerBlock(hidden, k=k, dropout=dropout)
            for _ in range(n_layers)
        ])
        # independent ABMIL head per layer
        self.pools  = nn.ModuleList([GatedPool(hidden)          for _ in range(n_layers)])
        self.heads  = nn.ModuleList([
            nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
            for _ in range(n_layers)
        ])
        # learned ensemble weights across layers
        self.layer_w = nn.Parameter(torch.zeros(n_layers))   # softmax → equal init

    def forward(self, feats, coords):
        """
        feats:  (N, 1024)
        coords: (N, 2)    pixel coordinates, float
        Returns:
          final_logit  — scalar
          layer_logits — list of n_layers scalars
          layer_attns  — list of n_layers (N,) tensors (ABMIL weights)
        """
        # build KNN once per slide — O(N log N), small N
        knn_idx = torch.from_numpy(
            build_knn_idx(coords.cpu().numpy(), self.k)
        ).long().to(feats.device)                             # (N, K)

        x = self.proj(feats)                                  # (N, hidden)

        layer_logits, layer_attns = [], []
        for block, pool, head in zip(self.blocks, self.pools, self.heads):
            x = block(x, coords, knn_idx)                    # (N, hidden)
            rep, attn = pool(x)                               # (hidden,), (N,)
            layer_logits.append(head(rep).squeeze())
            layer_attns.append(attn)

        w = torch.softmax(self.layer_w, dim=0)
        final_logit = sum(w[i] * layer_logits[i] for i in range(self.n_layers))
        return final_logit, layer_logits, layer_attns


# ══════════════════════════════════════════════════════════════════════════════
# MODEL D: Masked KNN-MIL  (spatial MAE + multi-scale classifier)
# ══════════════════════════════════════════════════════════════════════════════
#
# Two joint objectives:
#   1. Masked patch reconstruction — KNN cross-attn predicts masked patch
#      embeddings from visible spatial neighbours only (local constraint).
#      Forces the model to learn spatially coherent tissue representations.
#   2. Multi-scale classification — ABMIL at each layer predicts disease state.
#
# [MASK] token replaces masked patches; model must fill them in from K neighbours.
# At inference (eval): no masking, pure classification path.
#
# Loss = cls_loss + λ_recon * recon_loss   (λ=0.5 by default)

MASK_RATIO      = 0.30  # fraction of patches masked during training
LAMBDA_RECON    = 0.3   # weight of reconstruction loss in joint phase (was 0.5)
PRETRAIN_EPOCHS = 100   # epochs of recon-only pretraining (was 30)

def p_recon_schedule(joint_epoch, total_joint):
    """Anneal recon task probability 0.5 → 0.1 over joint training."""
    return max(0.1, 0.5 - 0.4 * joint_epoch / max(total_joint - 1, 1))

def cosine_recon_loss(pred, target):
    """Cosine reconstruction loss on L2-normalised features. Range 0–2."""
    return (1.0 - F.cosine_similarity(
        F.normalize(pred.float(), dim=-1),
        F.normalize(target.float(), dim=-1)
    )).mean()

class MaskedKNNMIL(nn.Module):
    """
    Masked Spatial KNN-MIL.

    Training: randomly mask MASK_RATIO patches → replace with learned [MASK] token
              → KNN message passing reconstructs masked patch embeddings from
                 visible spatial neighbours → reconstruction loss (MSE on raw feats).
    Both train and eval: ABMIL at each layer → multi-scale disease prediction.

    The spatial constraint (KNN, not global attn) means the model *must* learn
    local tissue context to reconstruct — it cannot just copy from far away.
    """
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, dropout=DROPOUT,
                 k=8, n_layers=4, mask_ratio=MASK_RATIO):
        super().__init__()
        self.k          = k
        self.n_layers   = n_layers
        self.mask_ratio = mask_ratio
        self.in_dim     = in_dim

        # Learnable [MASK] token — same dim as projected features
        self.mask_token = nn.Parameter(torch.zeros(1, hidden))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(dropout))

        self.blocks = nn.ModuleList([
            PointTransformerBlock(hidden, k=k, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.pools   = nn.ModuleList([GatedPool(hidden) for _ in range(n_layers)])
        self.heads   = nn.ModuleList([
            nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
            for _ in range(n_layers)
        ])
        self.layer_w = nn.Parameter(torch.zeros(n_layers))

        # Reconstruction head: hidden → in_dim (predict normalised patch features)
        self.recon_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, in_dim))
        # Set True during joint phase so encoder only receives CLS gradient
        self._stop_grad_recon = False

    def forward(self, feats, coords):
        """
        feats:  (N, in_dim)   raw patch features
        coords: (N, 2)        pixel coordinates

        Training returns: (final_logit, layer_logits, layer_attns, recon_pred, recon_target)
        Eval returns:     (final_logit, layer_logits, layer_attns, None, None)
        """
        N = feats.shape[0]
        knn_idx = torch.from_numpy(
            build_knn_idx(coords.cpu().numpy(), self.k)
        ).long().to(feats.device)

        x = self.proj(feats)   # (N, hidden)

        # ── Masking (train only) ────────────────────────────────────────────
        mask = None
        if self.training and self.mask_ratio > 0:
            n_mask = max(1, int(N * self.mask_ratio))
            mask_idx = torch.randperm(N, device=feats.device)[:n_mask]
            mask = torch.zeros(N, dtype=torch.bool, device=feats.device)
            mask[mask_idx] = True
            x = x.clone()
            x[mask] = self.mask_token.expand(mask.sum(), -1)

        # ── KNN message passing ─────────────────────────────────────────────
        layer_logits, layer_attns = [], []
        for block, pool, head in zip(self.blocks, self.pools, self.heads):
            x = block(x, coords, knn_idx)
            rep, attn = pool(x)
            layer_logits.append(head(rep).squeeze())
            layer_attns.append(attn)

        # ── Final classification logit ──────────────────────────────────────
        w = torch.softmax(self.layer_w, dim=0)
        final_logit = sum(w[i] * layer_logits[i] for i in range(self.n_layers))

        # ── Reconstruction at masked positions ──────────────────────────────
        if mask is not None and mask.sum() > 0:
            # stop-grad on encoder output during joint phase → decoder trains
            # separately, encoder only receives CLS gradient
            _enc_out = x[mask].detach() if self._stop_grad_recon else x[mask]
            recon_pred   = self.recon_head(_enc_out)          # (n_mask, in_dim)
            recon_target = F.normalize(feats[mask].detach(), dim=-1)  # L2-norm UNI sphere
        else:
            recon_pred = recon_target = None

        return final_logit, layer_logits, layer_attns, recon_pred, recon_target


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_slide(path, max_n=MAX_N, seed=None):
    """
    Returns feats (N, 1024), coords (N, 2), clusters (list of dicts),
    label (int), slide (str), patient (str).
    Flattens across DBSCAN clusters; subsamples if N > max_n.
    """
    d = torch.load(path, map_location='cpu', weights_only=False)
    feats_list  = [c['feats']  for c in d['clusters']]
    coords_list = [c['coords'] for c in d['clusters']]
    all_feats   = torch.cat(feats_list,  dim=0)   # (N_total, 1024)
    all_coords  = torch.cat(coords_list, dim=0)   # (N_total, 2)

    if all_feats.shape[0] > max_n:
        rng = torch.Generator()
        if seed is not None: rng.manual_seed(seed)
        idx = torch.randperm(all_feats.shape[0], generator=rng)[:max_n]
        all_feats  = all_feats[idx]
        all_coords = all_coords[idx]

    return (all_feats, all_coords, d['clusters'],
            d['label'], d['slide'], d['patient'])


class HierDataset(Dataset):
    """Returns flattened (feats, coords, label, slide, patient)."""
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        feats, coords, _, label, slide, patient = load_slide(self.paths[i])
        return feats, coords.float(), int(label), slide, patient

class ABMILDataset(Dataset):
    """Returns SLIC-clustered format for 2-level ABMIL."""
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        clusters, label, slide, patient = load_slic_slide(self.paths[i])
        return clusters, label, slide, patient

def hier_collate(b):  return b[0]
def abmil_collate(b): return b[0]


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def hinge_loss(logit, label, cw):
    """Hinge loss with class weighting."""
    y  = 2.0 * float(label) - 1.0
    w  = cw[int(label)]
    return w * F.relu(1.0 - y * logit)

def compute_cw(labels):
    n  = len(labels)
    n1 = sum(labels); n0 = n - n1
    return [n / (2 * max(n0, 1)), n / (2 * max(n1, 1))]


def spatial_slic(coords: np.ndarray,
                 target_size: int = TARGET_NBHD_SIZE,
                 min_size:    int = MIN_NBHD_SIZE,
                 max_iter:    int = 10) -> np.ndarray:
    """
    SLIC-like superpatch clustering on 2D patch coordinates.

    Algorithm:
      1. Place K = ceil(N / target_size) seeds on a regular grid over the
         tissue bounding box, snap each seed to its nearest actual patch.
      2. Iteratively: assign each patch to its nearest seed, then recompute
         each seed as the centroid of its assigned patches.
      3. Merge clusters smaller than min_size into the nearest larger cluster.

    Returns labels (N,) re-indexed 0..K'-1.
    """
    N = len(coords)
    if N == 0:
        return np.zeros(0, dtype=np.int32)
    if N <= target_size:
        return np.zeros(N, dtype=np.int32)

    # Normalise coords to [0, 1] so grid spacing is scale-invariant
    xy   = coords.astype(np.float64)
    mins = xy.min(0); span = max((xy.max(0) - mins).max(), 1.0)
    xy_n = (xy - mins) / span        # (N, 2)

    K = max(2, int(np.ceil(N / target_size)))
    S = 1.0 / np.sqrt(K)             # grid spacing (normalised)

    # ── Seed initialisation: regular grid → snap to nearest patch ──────────
    gx = np.arange(S / 2, 1.0, S)
    gy = np.arange(S / 2, 1.0, S)
    grid = np.array([[x, y] for y in gy for x in gx], dtype=np.float64)
    snap_tree  = cKDTree(xy_n)
    _, snap_idx = snap_tree.query(grid, k=1)
    seeds = xy_n[snap_idx]            # snap to actual patch positions
    seeds = np.unique(seeds, axis=0)  # drop duplicates (sparse tissue edges)
    K     = len(seeds)

    # ── Iterative assign-update ─────────────────────────────────────────────
    labels = np.zeros(N, dtype=np.int32)
    for _ in range(max_iter):
        seed_tree = cKDTree(seeds)
        _, new_labels = seed_tree.query(xy_n, k=1)
        new_labels = new_labels.astype(np.int32)

        seeds_new = np.zeros_like(seeds)
        counts    = np.zeros(K, dtype=np.int64)
        for i in range(N):
            lbl = new_labels[i]
            seeds_new[lbl] += xy_n[i]
            counts[lbl]    += 1

        valid = counts > 0
        seeds_new[valid]  /= counts[valid, np.newaxis]
        seeds_new[~valid]  = seeds[~valid]  # keep old position if no patches

        if np.allclose(seeds, seeds_new, atol=1e-7):
            labels = new_labels; break
        seeds  = seeds_new
        labels = new_labels

    # ── Merge tiny clusters into nearest bigger cluster ─────────────────────
    unique, cnts = np.unique(labels, return_counts=True)
    small_ids    = set(unique[cnts < min_size])
    if small_ids:
        big_ids   = [u for u in unique if u not in small_ids]
        if big_ids:
            big_seeds = seeds[big_ids]
            big_tree  = cKDTree(big_seeds)
            for s in small_ids:
                mask     = labels == s
                center   = xy_n[mask].mean(0)
                _, nn    = big_tree.query([center], k=1)
                labels[mask] = big_ids[nn[0]]

    # ── Re-index to compact 0..K'-1 ────────────────────────────────────────
    uniq = np.unique(labels)
    remap = np.empty(labels.max() + 1, dtype=np.int32)
    for new_i, old in enumerate(uniq):
        remap[old] = new_i
    return remap[labels]


def load_slic_slide(path, target_size=TARGET_NBHD_SIZE, min_size=MIN_NBHD_SIZE,
                    max_patches_per_cluster=512):
    """
    Load a .pt slide file, discard DBSCAN cluster structure, and re-cluster
    all patches using spatial SLIC for balanced, spatially coherent neighbourhoods.

    Returns: clusters (list of (N_c, in_dim) tensors), label (int), slide (str), patient (str)
    """
    d = torch.load(path, map_location='cpu', weights_only=False)

    # Concatenate all patches across existing DBSCAN clusters
    feats_list  = [c['feats']  for c in d['clusters']]
    coords_list = [c['coords'] for c in d['clusters']]
    all_feats  = torch.cat(feats_list,  dim=0).float()   # (N_total, in_dim)
    all_coords = torch.cat(coords_list, dim=0).float()   # (N_total, 2)

    # SLIC re-clustering on spatial coordinates
    labels = spatial_slic(all_coords.numpy(), target_size=target_size,
                          min_size=min_size)
    n_clusters = int(labels.max()) + 1

    # Group features by SLIC label → list of tensors
    clusters = []
    for ci in range(n_clusters):
        idx = np.where(labels == ci)[0]
        f   = all_feats[idx]
        if f.shape[0] > max_patches_per_cluster:
            f = f[torch.randperm(f.shape[0])[:max_patches_per_cluster]]
        clusters.append(f)

    return clusters, int(d['label']), d['slide'], d['patient']

def load_slic_multilevel(path, nbhd_sizes=(64, 512), max_patches=512):
    """
    Load a .pt slide and build a balanced multi-level SLIC hierarchy.

    nbhd_sizes: (fine_target, coarse_target, ...) — target patch count per cluster
                at each level, from finest to coarsest.
                e.g. (64, 512)   → 3-level (patches → fine → coarse → slide)
                     (64, 256, 1024) → 4-level

    Returns: feats_list (list of feature tensors per fine cluster),
             group_ids  (list of group-id lists for coarser levels),
             label (int)
    """
    d = torch.load(path, map_location='cpu', weights_only=False)

    # Concatenate all patches (ignoring DBSCAN cluster structure entirely)
    all_feats  = torch.cat([c['feats']  for c in d['clusters']], dim=0).float()
    all_coords = torch.cat([c['coords'] for c in d['clusters']], dim=0).float()

    # Fine SLIC clusters (level 1)
    fine_labels = spatial_slic(all_coords.numpy(), target_size=nbhd_sizes[0])
    K_fine      = int(fine_labels.max()) + 1

    feats_list = []
    for ci in range(K_fine):
        idx = np.where(fine_labels == ci)[0]
        f   = all_feats[idx]
        if f.shape[0] > max_patches:
            f = f[torch.randperm(f.shape[0])[:max_patches]]
        feats_list.append(f)

    # Coarser SLIC levels
    coarser_sizes = list(nbhd_sizes[1:])
    group_ids = build_slic_multilevel(all_coords.numpy(), fine_labels, coarser_sizes)

    return feats_list, group_ids, int(d['label'])


@torch.no_grad()
def evaluate(model, paths, model_type, device, cw, nbhd_sizes=None, seed=None):
    model.eval()
    probs, truths = [], []
    for i, path in enumerate(paths):
        slide_seed = (seed + i) if seed is not None else None
        if model_type in ('hier', 'knnmil', 'maskedknn'):
            feats, coords, _, label, _, _ = load_slide(path, seed=slide_seed)
            feats  = feats.to(device)
            coords = coords.float().to(device)
            logit, *_ = model(feats, coords)
        elif model_type == 'multilevel':
            feats_list, group_ids, label = load_slic_multilevel(
                path, nbhd_sizes=nbhd_sizes or (64, 512))
            feats_list = [f.to(device) for f in feats_list]
            logit, _, _ = model(feats_list, group_ids)
        else:
            # SLIC-balanced neighbourhood clustering (replaces raw DBSCAN clusters)
            clusters, label, _, _ = load_slic_slide(path)
            clusters = [f.to(device) for f in clusters]
            logit, _, _ = model(clusters)

        probs.append(torch.sigmoid(logit).item())
        truths.append(label)

    auc  = roc_auc_score(truths, probs) if len(set(truths)) > 1 else 0.5
    preds = [1 if p > 0.5 else 0 for p in probs]
    bacc = balanced_accuracy_score(truths, preds)
    return auc, bacc, probs, truths


@torch.no_grad()
def evaluate_with_uncertainty(model, paths, device, T=50, seed=None):
    """MC Dropout uncertainty for MaskedKNNMIL.

    Returns per-slide dict with:
      prob_mean, prob_std  — slide-level prediction + epistemic uncertainty
      attn_mean, attn_std  — (N, n_layers) per-patch attention mean/std over T passes
      coords               — (N, 2) patch coordinates for spatial maps
      label                — ground truth
    """
    results = []
    for i, path in enumerate(paths):
        slide_seed = (seed + i) if seed is not None else None
        feats, coords, _, label, slide, _ = load_slide(path, seed=slide_seed)
        feats_d  = feats.to(device)
        coords_d = coords.float().to(device)
        N = feats_d.shape[0]
        n_layers = model.n_layers

        # enable dropout for MC passes
        model.train()
        probs_mc   = []
        attns_mc   = []  # list of (N, n_layers) arrays
        for _ in range(T):
            out   = model(feats_d, coords_d)
            logit = out[0]
            probs_mc.append(torch.sigmoid(logit).item())
            # layer_attns: list of n_layers tensors, each shape (N,)
            layer_attns = out[2]
            stacked = torch.stack([a.detach().cpu() for a in layer_attns], dim=1)  # (N, n_layers)
            attns_mc.append(stacked.numpy())
        model.eval()

        probs_arr = np.array(probs_mc)           # (T,)
        attns_arr = np.stack(attns_mc, axis=0)   # (T, N, n_layers)

        results.append({
            'slide':      slide,
            'label':      label,
            'prob_mean':  probs_arr.mean(),
            'prob_std':   probs_arr.std(),
            'attn_mean':  attns_arr.mean(axis=0),   # (N, n_layers)
            'attn_std':   attns_arr.std(axis=0),    # (N, n_layers) — spatial uncertainty
            'coords':     coords.numpy(),            # (N, 2)
        })
    return results


def train_fold(model, model_type, tr_paths, va_paths, fold, tag, cw, scaler,
               nbhd_sizes=None):
    """Train one fold. Returns best val balanced-accuracy and model.

    For maskedknn:
      Phase 1 (PRETRAIN_EPOCHS): recon-only (no CLS), stop-grad OFF.
      Phase 2 (JOINT_EPOCHS):    per-slide coin-toss — choose CLS or RECON,
                                  p(recon) annealed 0.5→0.1, stop-grad ON.
      Val check every PATIENCE_EVERY epochs; PATIENCE no-improve checks before stop.
    For all other models: JOINT_EPOCHS epochs, val every PATIENCE_EVERY, same stop.
    """
    is_masked = (model_type == 'maskedknn')
    pretrain_total = PRETRAIN_EPOCHS if is_masked else 0
    total_epochs   = pretrain_total + JOINT_EPOCHS

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=LR * 0.01)

    best_bacc  = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(total_epochs):
        pretrain_phase = is_masked and (epoch < pretrain_total)
        joint_epoch    = max(0, epoch - pretrain_total)  # 0-indexed into joint phase

        # ── set stop-grad flag on model ────────────────────────────────────
        if is_masked:
            model._stop_grad_recon = not pretrain_phase  # ON during joint phase

        model.train()
        random.shuffle(tr_paths)
        optimizer.zero_grad()
        loss_sum = 0.0; n_steps = 0; accum = 0

        for path in tr_paths:
            try:
                if model_type in ('hier', 'knnmil', 'maskedknn'):
                    feats, coords, _, label, _, _ = load_slide(path, seed=epoch)
                    feats  = feats.to(device)
                    coords = coords.float().to(device)
                    with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                        out   = model(feats, coords)
                        logit = out[0]
                        if is_masked and out[3] is not None:
                            recon_loss = cosine_recon_loss(out[3], out[4])
                            if pretrain_phase:
                                loss = recon_loss / GRAD_ACCUM
                            else:
                                # coin-toss: choose CLS or RECON for this slide
                                p_r = p_recon_schedule(joint_epoch, JOINT_EPOCHS)
                                if random.random() < p_r:
                                    loss = (LAMBDA_RECON * recon_loss) / GRAD_ACCUM
                                else:
                                    loss = hinge_loss(logit, label, cw) / GRAD_ACCUM
                        else:
                            loss = hinge_loss(logit, label, cw) / GRAD_ACCUM
                elif model_type == 'multilevel':
                    feats_list, group_ids, label = load_slic_multilevel(
                        path, nbhd_sizes=nbhd_sizes or (64, 512))
                    feats_list = [f.to(device) for f in feats_list]
                    with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                        logit, _, _ = model(feats_list, group_ids)
                        loss = hinge_loss(logit, label, cw) / GRAD_ACCUM
                else:
                    # SLIC-balanced neighbourhood clustering
                    clusters, label, _, _ = load_slic_slide(path)
                    clusters = [f.to(device) for f in clusters]
                    with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                        logit, _, _ = model(clusters)
                        loss = hinge_loss(logit, label, cw) / GRAD_ACCUM

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_sum += loss.item() * GRAD_ACCUM
                accum += 1

                if accum == GRAD_ACCUM:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer); scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad()
                    n_steps += 1; accum = 0

            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(); accum = 0
                torch.cuda.empty_cache()
                print(f"  [OOM] {path.name} — skip", flush=True)

        if accum > 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(); n_steps += 1

        scheduler.step()

        # ── logging + val check ────────────────────────────────────────────
        log_this = ((epoch + 1) % PATIENCE_EVERY == 0)
        if log_this:
            mean_loss = loss_sum / max(n_steps, 1)
            if pretrain_phase:
                print(f"  [{tag}] fold={fold} ep={epoch+1:3d}  [PRETRAIN recon-only]"
                      f"  loss={mean_loss:.4f}", flush=True)
            else:
                val_auc, val_bacc, _, _ = evaluate(
                    model, va_paths, model_type, device, cw,
                    nbhd_sizes=nbhd_sizes, seed=EVAL_SEED)
                p_r = p_recon_schedule(joint_epoch, JOINT_EPOCHS)
                print(f"  [{tag}] fold={fold} ep={epoch+1:3d}  "
                      f"loss={mean_loss:.4f}  val_auc={val_auc:.4f}"
                      f"  val_bacc={val_bacc:.4f}  p_r={p_r:.2f}", flush=True)
                if val_bacc > best_bacc:
                    best_bacc  = val_bacc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= PATIENCE:
                        print(f"  [{tag}] fold={fold} early stop at ep={epoch+1}"
                              f" ({PATIENCE} checks × {PATIENCE_EVERY} ep = "
                              f"{PATIENCE*PATIENCE_EVERY} ep patience)", flush=True)
                        break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_bacc, model


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: load data, cross-validate both models
# ══════════════════════════════════════════════════════════════════════════════
set_seeds()

all_paths = sorted(FEAT_DIR.glob("*.pt"))
labels, patients = [], []
for p in all_paths:
    d = torch.load(p, map_location='cpu', weights_only=False)
    labels.append(d['label']); patients.append(d['patient'])
labels  = np.array(labels)
patients = np.array(patients)
print(f"Total slides: {len(all_paths)}  ACR+={labels.sum()}  ACR-={len(labels)-labels.sum()}",
      flush=True)

cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
splits = list(cv.split(all_paths, labels, patients))

# SLIC neighbourhood sizes for multi-level variants:
#   3-level: patches → fine (~64 patches) → coarse (~512 patches) → slide
#   4-level: patches → fine (~64) → medium (~256) → coarse (~1024) → slide
NBHD_3LEVEL = (64, 512)
NBHD_4LEVEL = (64, 256, 1024)

# KNN MIL: fixed K=8 neighbours, 4 layers
# receptive field grows automatically: layer l covers ~K^l patches
KNN_K       = 8
KNN_NLAYERS = 4

# ── Per-job model/fold selection (set via env vars MODEL and FOLD) ────────────
import os as _os
RUN_MODEL = _os.environ.get("MODEL", "all")   # e.g. "maskedknn", "knnmil", "hier", "all"
RUN_FOLD  = int(_os.environ.get("FOLD", "-1")) # -1 = all folds

results = {'hier': [], 'abmil': [], 'abmil3': [], 'abmil4': [], 'knnmil': [], 'maskedknn': []}

fold_iter = [(fold, s) for fold, s in enumerate(splits)
             if RUN_FOLD == -1 or fold == RUN_FOLD]

for fold, (tr_idx, te_idx) in fold_iter:
    print(f"\n{'='*60}", flush=True)
    print(f"FOLD {fold}", flush=True)
    print(f"{'='*60}", flush=True)

    tr_paths = [all_paths[i] for i in tr_idx]
    te_paths = [all_paths[i] for i in te_idx]

    # 80/20 train/val within training set (by patient)
    tr_labels   = labels[tr_idx]
    tr_patients = patients[tr_idx]
    inner_cv    = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    inner_tr_idx, inner_va_idx = next(inner_cv.split(tr_paths, tr_labels, tr_patients))
    va_paths  = [tr_paths[i] for i in inner_va_idx]
    tr_paths2 = [tr_paths[i] for i in inner_tr_idx]

    tr_labels2 = [labels[tr_idx[i]] for i in inner_tr_idx]
    cw = compute_cw(tr_labels2)
    print(f"  train={len(tr_paths2)}  val={len(va_paths)}  test={len(te_paths)}  "
          f"cw=[{cw[0]:.2f},{cw[1]:.2f}]", flush=True)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    def _run_model(tag, build_fn, model_type, label, nbhd_sizes=None):
        if RUN_MODEL not in ('all', tag): return
        print(f"\n  --- {label} ---", flush=True)
        set_seeds(SEED + fold)
        m = build_fn().to(device)
        print(f"  Params: {sum(p.numel() for p in m.parameters()):,}", flush=True)
        _, m = train_fold(m, model_type, tr_paths2, va_paths, fold, tag.upper(), cw, scaler,
                          nbhd_sizes=nbhd_sizes)
        auc, bacc, probs, truths = evaluate(m, te_paths, model_type, device, cw,
                                            nbhd_sizes=nbhd_sizes)
        print(f"  [{tag.upper()}] fold={fold}  TEST AUC={auc:.4f}  BACC={bacc:.4f}", flush=True)
        torch.save(m.state_dict(), OUTDIR / f"{tag}_fold{fold}.pt")
        results[tag].append({'fold': fold, 'auc': auc, 'bacc': bacc,
                             'probs': probs, 'truths': truths})
        # Save per-fold JSON immediately so partial results are readable
        import json as _json
        _json.dump({'model': tag, 'fold': fold, 'auc': auc, 'bacc': bacc},
                   open(OUTDIR / f"result_{tag}_fold{fold}.json", "w"))

    _run_model('hier', SoftHierarchicalMIL, 'hier',
               f'SoftHier (Gaussian σ={SIGMAS})')

    _run_model('abmil', SpatialABMIL2Level, 'abmil',
               'Spatial ABMIL 2-level (baseline)')

    _run_model('knnmil',
               lambda: SpatialKNNMIL(k=KNN_K, n_layers=KNN_NLAYERS), 'knnmil',
               f'KNN-MIL (K={KNN_K}, {KNN_NLAYERS}L, RF ~{KNN_K}→{KNN_K**4})')

    _run_model('maskedknn',
               lambda: MaskedKNNMIL(k=KNN_K, n_layers=KNN_NLAYERS), 'maskedknn',
               f'Masked KNN-MIL (K={KNN_K}, {KNN_NLAYERS}L, mask={MASK_RATIO:.0%}, λ={LAMBDA_RECON})')

    _run_model('abmil3',
               lambda: SpatialABMILNLevel(n_levels=3), 'multilevel',
               f'Spatial ABMIL 3-level (SLIC {NBHD_3LEVEL})',
               nbhd_sizes=NBHD_3LEVEL)

    _run_model('abmil4',
               lambda: SpatialABMILNLevel(n_levels=4), 'multilevel',
               f'Spatial ABMIL 4-level (SLIC {NBHD_4LEVEL})',
               nbhd_sizes=NBHD_4LEVEL)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY + PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print("FINAL COMPARISON", flush=True)
print(f"{'='*60}", flush=True)

ALL_MODELS = [
    ('hier',      'SoftHier (Gaussian)',              '#E53030'),
    ('knnmil',    f'KNN-MIL (K={KNN_K}, {KNN_NLAYERS}L)',    '#9B30FF'),
    ('maskedknn', f'Masked KNN-MIL (mask={MASK_RATIO:.0%})', '#FF69B4'),
    ('abmil',     'Spatial ABMIL 2-level',            '#4477CC'),
    ('abmil3',    'Spatial ABMIL 3-level',            '#22AA44'),
    ('abmil4',    'Spatial ABMIL 4-level',            '#FF8800'),
]
for tag, name, _ in ALL_MODELS:
    if not results[tag]: continue
    aucs  = [r['auc']  for r in results[tag]]
    baccs = [r['bacc'] for r in results[tag]]
    print(f"\n  {name}", flush=True)
    for r in results[tag]:
        print(f"    fold {r['fold']}: AUC={r['auc']:.4f}  BACC={r['bacc']:.4f}", flush=True)
    print(f"    MEAN AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}", flush=True)
    print(f"    MEAN BACC= {np.mean(baccs):.4f} ± {np.std(baccs):.4f}", flush=True)

json.dump(results, open(OUTDIR / "results.json", "w"),
          default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x,
          indent=2)

# ── Figure 1: AUC comparison per fold + mean ─────────────────────────────────
from sklearn.metrics import roc_curve
fig, axes = plt.subplots(1, 3, figsize=(19, 5))
fig.suptitle("Spatial ABMIL: 2-level vs 3-level vs 4-level vs Soft Hierarchical", fontsize=12)

# Panel 1: AUC per fold
ax = axes[0]
folds = [r['fold'] for r in results['abmil']]
n_models = len([t for t, *_ in ALL_MODELS if results[t]])
w = 0.7 / n_models
offsets = np.linspace(-(n_models-1)*w/2, (n_models-1)*w/2, n_models)
for (tag, name, color), offset in zip(
        [(t,n,c) for t,n,c in ALL_MODELS if results[t]], offsets):
    aucs = [r['auc'] for r in results[tag]]
    x = np.arange(len(aucs))
    ax.bar(x + offset, aucs, w, label=f'{name} (μ={np.mean(aucs):.3f})',
           color=color, alpha=0.82, edgecolor='white')
ax.set_xticks(np.arange(len(folds))); ax.set_xticklabels([f'Fold {f}' for f in folds])
ax.set_ylabel("Test AUC"); ax.set_ylim(0.4, 1.0)
ax.legend(fontsize=7); ax.set_title("AUC per fold")
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Panel 2: BAcc per fold
ax = axes[1]
for (tag, name, color), offset in zip(
        [(t,n,c) for t,n,c in ALL_MODELS if results[t]], offsets):
    baccs = [r['bacc'] for r in results[tag]]
    x = np.arange(len(baccs))
    ax.bar(x + offset, baccs, w, label=f'{name} (μ={np.mean(baccs):.3f})',
           color=color, alpha=0.82, edgecolor='white')
ax.set_xticks(np.arange(len(folds))); ax.set_xticklabels([f'Fold {f}' for f in folds])
ax.set_ylabel("Test BAcc"); ax.set_ylim(0.4, 1.0)
ax.legend(fontsize=7); ax.set_title("BAcc per fold")
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Panel 3: OOF ROC
ax2 = axes[2]
for tag, name, color in ALL_MODELS:
    if not results[tag]: continue
    all_probs  = sum([r['probs']  for r in results[tag]], [])
    all_truths = sum([r['truths'] for r in results[tag]], [])
    fpr, tpr, _ = roc_curve(all_truths, all_probs)
    oof_auc = roc_auc_score(all_truths, all_probs)
    ax2.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (OOF={oof_auc:.3f})')
ax2.plot([0,1],[0,1],'k--',linewidth=0.8)
ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
ax2.set_title("OOF ROC curve")
ax2.legend(fontsize=7)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(OUTDIR / "comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved comparison.png", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL ATTENTION VISUALISATION (SoftHierarchicalMIL)
# Same 4-panel layout as spatial_scatter_plots:
#   Tissue type | Patch attn (from gated pool) | — | Combined (same here, single level)
# For SoftHier the gated pool produces one attention weight per patch directly.
# We show: tissue | attn score scatter — for 10 ACR+, 10 ACR-, up to 5 wrong preds.
# ══════════════════════════════════════════════════════════════════════════════
import anndata as ad

VIS_DIR = OUTDIR / "attn_plots"
VIS_DIR.mkdir(exist_ok=True)

TISSUE_COLORS = {
    "Alveolar":                                       "#4CAF50",
    "Alveolar with empty spaces":                     "#2196F3",
    "Bronchial":                                      "#9C27B0",
    "Cartilage":                                      "#00BCD4",
    "Alveolar with hemorrhage and inflammation":      "#FF9800",
    "Lymphocytoplasmic inflammation":                 "#F44336",
    "Unknown":                                        "#AAAAAA",
}
DRAW_ORDER = list(TISSUE_COLORS.keys())

import matplotlib.patches as mpatches

if not results['hier']:
    print(f"\nAll outputs in: {OUTDIR}", flush=True)
    print("Skipping tissue-type visualization (hier not in this run).", flush=True)
    import sys; sys.exit(0)

print("\nLoading h5ad for tissue types...", flush=True)
adata   = ad.read_h5ad("/lustre/groups/aih/dinesh.haridoss/datasets/adata_v3.h5ad", backed='r')
obs     = adata.obs[['slide_name_clean','chunk_left','chunk_top','tissue_type']].copy()
obs['chunk_left'] = obs['chunk_left'].astype(np.float32)
obs['chunk_top']  = obs['chunk_top'].astype(np.float32)

# Collect all test-set slides with predictions
slide_records = []
for fold_res in results['hier']:
    fold = fold_res['fold']
    _, te_idx = splits[fold]
    te_paths = [all_paths[i] for i in te_idx]
    model = SoftHierarchicalMIL().to(device)
    model.load_state_dict(torch.load(OUTDIR / f"soft_hier_fold{fold}.pt",
                                     map_location=device))
    model.eval()
    with torch.no_grad():
        for path, prob, truth in zip(te_paths, fold_res['probs'], fold_res['truths']):
            feats, coords, _, label, slide, patient = load_slide(path)
            feats  = feats.to(device); coords = coords.float().to(device)
            _, attn = model(feats, coords)
            slide_records.append({
                'fold': fold, 'slide': slide, 'label': truth, 'prob': prob,
                'coords': coords.cpu().numpy(),   # (N, 2)
                'attn':   attn.cpu().numpy(),     # (N,)
            })

slide_df2 = pd.DataFrame([{k: v for k, v in r.items()
                            if k not in ('coords','attn')} for r in slide_records])
result_map2 = {r['slide']: r for r in slide_records}

# Select slides
pos = slide_df2[slide_df2['label']==1].sort_values('prob', ascending=False)
neg = slide_df2[slide_df2['label']==0].sort_values('prob', ascending=True)
fp  = slide_df2[(slide_df2['label']==0)&(slide_df2['prob']>0.5)].sort_values('prob', ascending=False)
fn  = slide_df2[(slide_df2['label']==1)&(slide_df2['prob']<0.5)].sort_values('prob', ascending=True)
selected2 = {
    'ACR_positive': pos.head(10)['slide'].tolist(),
    'ACR_negative': neg.head(10)['slide'].tolist(),
    'FalsePositive': fp.head(5)['slide'].tolist(),
    'FalseNegative': fn.head(5)['slide'].tolist(),
}

# Build tissue maps for needed slides
all_needed2 = set(sum(selected2.values(), []))
obs_needed2 = obs[obs['slide_name_clean'].isin(all_needed2)]
slide_tissue2 = {}
for slide, grp in obs_needed2.groupby('slide_name_clean'):
    arr = grp[['chunk_left','chunk_top','tissue_type']].values
    slide_tissue2[slide] = {(round(float(r[0])),round(float(r[1]))): str(r[2]) for r in arr}

def _minmax(a):
    a = np.array(a, dtype=float)
    lo, hi = a.min(), a.max()
    return (a-lo)/(hi-lo) if hi>lo else np.zeros_like(a)

def make_vis_figure(slide_names, category, slides_per_row=5):
    n = len(slide_names);
    if n == 0: return
    n_rows = int(np.ceil(n / slides_per_row))
    # 2 panels per slide: tissue | attn
    fig_w = slides_per_row * 6
    fig_h = n_rows * 3.5 + 1.2
    fig, axes = plt.subplots(n_rows, slides_per_row*2,
                              figsize=(fig_w, fig_h),
                              gridspec_kw={'wspace':0.04,'hspace':0.40})
    axes = np.array(axes).reshape(n_rows, slides_per_row*2)

    for i, slide in enumerate(slide_names):
        row = i // slides_per_row
        c0  = (i % slides_per_row) * 2
        ax_t, ax_a = axes[row, c0], axes[row, c0+1]

        if slide not in result_map2:
            ax_t.set_visible(False); ax_a.set_visible(False); continue

        res  = result_map2[slide]
        tmap = slide_tissue2.get(slide, {})
        coords = res['coords']          # (N, 2)
        attn   = res['attn']            # (N,)
        xs = coords[:, 0]; ys = -coords[:, 1]

        # Tissue panel
        tissues = [tmap.get((round(float(x)),round(float(y))),'Unknown')
                   for x,y in zip(xs,-ys)]
        for tt in DRAW_ORDER:
            mask = np.array([t==tt for t in tissues])
            if mask.any():
                ax_t.scatter(xs[mask], ys[mask], s=1, c=TISSUE_COLORS[tt],
                             alpha=0.85, linewidths=0, rasterized=True)

        # Attention panel
        anorm = _minmax(attn)
        order = np.argsort(anorm)
        ax_a.scatter(xs[order], ys[order], s=1, c=anorm[order],
                     cmap='RdYlGn_r', vmin=0, vmax=1,
                     alpha=0.9, linewidths=0, rasterized=True)

        for ax in [ax_t, ax_a]:
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect('equal'); ax.spines[:].set_visible(False)

        lbl  = "ACR+" if res['label']==1 else "ACR-"
        ax_t.set_title(f"{lbl} p={res['prob']:.2f}\n{slide.split('/')[-1][:24]}",
                       fontsize=5.5, pad=2)
        ax_a.set_title(f"Attn [{attn.min():.4f},{attn.max():.4f}]", fontsize=5.5, pad=2)

    # Hide extras
    for i in range(n, n_rows*slides_per_row):
        row = i//slides_per_row; c0=(i%slides_per_row)*2
        for dc in [0,1]:
            axes[row, c0+dc].set_visible(False)

    legend_items = [mpatches.Patch(color=c, label=t) for t,c in TISSUE_COLORS.items()]
    fig.legend(handles=legend_items, loc='lower left', ncol=4, fontsize=6,
               frameon=False, bbox_to_anchor=(0.01,0.01))
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(0,1))
    sm.set_array([])
    cax = fig.add_axes([0.82, 0.02, 0.15, 0.02])
    cb  = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('Attn (raw min-max)', fontsize=6); cb.ax.tick_params(labelsize=5)
    plt.suptitle(f'SoftHier MIL — {category}  |  Tissue type · Patch attention',
                 fontsize=10, y=0.97)
    fig.savefig(VIS_DIR / f"{category}.png", dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {category}.png", flush=True)

make_vis_figure(selected2['ACR_positive'],  'ACR_positive_top10',      slides_per_row=5)
make_vis_figure(selected2['ACR_negative'],  'ACR_negative_top10',      slides_per_row=5)
make_vis_figure(selected2['FalsePositive'], 'WrongPred_FalsePositive', slides_per_row=5)
make_vis_figure(selected2['FalseNegative'], 'WrongPred_FalseNegative', slides_per_row=5)

print(f"\nAll outputs in: {OUTDIR}", flush=True)
PYEOF
