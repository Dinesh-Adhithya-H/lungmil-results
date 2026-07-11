#!/usr/bin/env bash
#SBATCH --job-name=benchmark_mil
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G
#SBATCH --gres=gpu:1 --constraint="a100_40gb|a100_80gb|h100_80gb"
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/benchmark/%j_%x.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/benchmark/%j_%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

mkdir -p /lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/benchmark

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

# Fix broken conda pip using the bootstrap script (pre-downloaded to ~/get-pip.py).
# This fully replaces the corrupted pip vendor modules in the chicago env.
python3 ~/get-pip.py --force-reinstall --quiet \
    && echo "pip bootstrap OK" \
    || echo "WARNING: pip bootstrap failed — continuing without pip"

# Install torchmil, tensordict, torch-geometric
python3 -m pip install --quiet torchmil tensordict \
    && echo "torchmil OK" \
    || echo "WARNING: torchmil install failed — torchmil models will be skipped"

python3 -m pip install --quiet torch-geometric \
    && echo "torch-geometric OK" \
    || echo "WARNING: torch-geometric not installed — PatchGCN uses GraphSAGE fallback"

python3 -m pip install --quiet pyg-lib torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html \
    && echo "pyg sparse kernels OK" \
    || echo "WARNING: pyg sparse kernels failed"

python3 -u << 'PYEOF'
"""
Fair MIL Benchmark  +  MaskedKNNMIL Ablations.

Published baselines (via torchmil):
  abmil      — ABMIL             Ilse et al.  ICML 2018
  clam       — CLAM-SB           Lu et al.    Nature BME 2021
  dsmil      — DSMIL             Li et al.    CVPR 2021
  transmil   — TransMIL          Shao et al.  NeurIPS 2021
  gtp        — GTP               Zheng et al. NeurIPS 2023  (graph, spatial-aware)

Our own models:
  patchgcn   — PatchGCN          Chen et al.  MICCAI 2021  (our impl — no PyG needed)
  knnmil     — SpatialKNNMIL     ours (no masking)
  maskedknn  — MaskedKNNMIL      ours (full model)

Ablations of MaskedKNNMIL (each disables one novel component):
  abl_nopretrain   skip recon-only Phase 1
  abl_mse          MSE recon loss instead of cosine
  abl_nosg         no stop-gradient on decoder
  abl_noannealing  fixed p_recon=0.5
  abl_additive     additive CLS+λ·RECON every slide (no coin-toss)
  abl_1scale       single ABMIL head (n_layers=1)

All methods: same UNI features, same 5-fold StratifiedGroupKFold splits, same
MAX_N=1024, same LR/WD, same hinge loss, same seeded evaluation.

Dispatch: MODEL env var (default "all"), FOLD env var (default -1 = all folds).
"""
import dataclasses, json, math, os, random
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# ── torchmil imports (graceful fallback if install failed) ────────────────────
try:
    import torchmil
    import torchmil.models as tm
    from tensordict import TensorDict
    TORCHMIL_OK = True
    print(f"torchmil {getattr(torchmil, '__version__', 'unknown')} loaded", flush=True)
except Exception as _e:
    TORCHMIL_OK = False
    print(f"WARNING: torchmil not available ({_e}) — torchmil models will be skipped",
          flush=True)

FEAT_DIR = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/slide_cluster_feats")
OUTDIR   = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/benchmark")
OUTDIR.mkdir(exist_ok=True)

# ── Hyperparameters (match train_soft_hier_mil.sh exactly) ────────────────────
IN_DIM         = 1024
HIDDEN         = 256
DROPOUT        = 0.25
MAX_N          = 1024
N_SPLITS       = 5
JOINT_EPOCHS   = 250
PATIENCE       = 30
PATIENCE_EVERY = 5
EVAL_SEED      = 42
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
GRAD_ACCUM     = 8
SEED           = 42
KNN_K          = 8
KNN_NLAYERS    = 4
MASK_RATIO     = 0.30
LAMBDA_RECON   = 0.3
PRETRAIN_EPOCHS= 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
if TORCHMIL_OK:
    print(f"torchmil loaded (version: {getattr(torchmil, '__version__', 'unknown')})", flush=True)

def set_seeds(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if device.type == "cuda": torch.cuda.manual_seed_all(s)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def build_knn_idx(coords_np, k):
    _, idx = cKDTree(coords_np).query(coords_np, k=k + 1)
    return idx[:, 1:]  # (N, K) exclude self

def load_slide(path, max_n=MAX_N, seed=None):
    d = torch.load(path, map_location='cpu', weights_only=False)
    feats  = torch.cat([c['feats']  for c in d['clusters']], 0)
    coords = torch.cat([c['coords'] for c in d['clusters']], 0)
    if feats.shape[0] > max_n:
        rng = torch.Generator()
        if seed is not None: rng.manual_seed(seed)
        idx = torch.randperm(feats.shape[0], generator=rng)[:max_n]
        feats = feats[idx]; coords = coords[idx]
    return feats, coords, d['label'], d['slide'], d['patient']

class GatedPool(nn.Module):
    """Gated attention pooling (N,d) → (d,), (N,)."""
    def __init__(self, dim):
        super().__init__()
        h = dim // 2
        self.V = nn.Sequential(nn.Linear(dim, h), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(dim, h), nn.Sigmoid())
        self.w = nn.Linear(h, 1, bias=False)
    def forward(self, h):
        a = torch.softmax(self.w(self.V(h) * self.U(h)), dim=0)
        return (a * h).sum(0), a.squeeze(-1)

class PointTransformerBlock(nn.Module):
    """O(N·K) spatial attention with relative position encoding."""
    def __init__(self, dim, k=8, dropout=0.1):
        super().__init__()
        self.scale   = dim ** -0.5
        self.W_q     = nn.Linear(dim, dim, bias=False)
        self.W_k     = nn.Linear(dim, dim, bias=False)
        self.W_v     = nn.Linear(dim, dim, bias=False)
        self.W_o     = nn.Linear(dim, dim, bias=False)
        self.pos_enc = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm1   = nn.LayerNorm(dim)
        self.norm2   = nn.LayerNorm(dim)
        self.ff      = nn.Sequential(
            nn.Linear(dim, dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim*2, dim), nn.Dropout(dropout))
        self.drop    = nn.Dropout(dropout)
    def forward(self, x, coords, knn_idx):
        rel     = coords[knn_idx] - coords.unsqueeze(1)
        bias    = self.pos_enc(rel)
        q       = self.W_q(x)
        k_      = self.W_k(x)[knn_idx] + bias
        v       = self.W_v(x)[knn_idx] + bias
        attn    = torch.softmax((q.unsqueeze(1) * k_).sum(-1) * self.scale, dim=-1)
        out     = self.drop(self.W_o((attn.unsqueeze(-1) * v).sum(1)))
        x       = self.norm1(x + out)
        return  self.norm2(x + self.ff(x))

def hinge_loss(logit, label, cw):
    y = 2.0 * float(label) - 1.0
    return cw[int(label)] * F.relu(1.0 - y * logit)

def compute_cw(labels):
    n = len(labels); n1 = sum(labels); n0 = n - n1
    return [n / (2*max(n0,1)), n / (2*max(n1,1))]

def cosine_recon_loss(pred, target):
    return (1.0 - F.cosine_similarity(
        F.normalize(pred.float(), dim=-1),
        F.normalize(target.float(), dim=-1))).mean()

def p_recon_schedule(joint_epoch, total_joint):
    return max(0.1, 0.5 - 0.4 * joint_epoch / max(total_joint - 1, 1))


# ══════════════════════════════════════════════════════════════════════════════
# TORCHMIL WRAPPER
# Converts our (feats: Tensor, coords: Tensor) → logit interface
# into torchmil's TensorDict bag interface.
# ══════════════════════════════════════════════════════════════════════════════

class TorchMILAdapter(nn.Module):
    """Thin adapter: torchmil model → our (feats, coords) → (logit, inst_loss) API."""
    def __init__(self, tmil_model, pass_coords=False):
        super().__init__()
        self.model       = tmil_model
        self.pass_coords = pass_coords

    def _extract_logit(self, out):
        if isinstance(out, torch.Tensor):
            return out.squeeze()
        for key in ('logit', 'logits', 'bag_logit', 'bag_logits'):
            if key in out.keys():
                return out[key].squeeze()
        raise ValueError(f"Cannot find logit in output keys: {list(out.keys())}")

    def _extract_inst_loss(self, out):
        if not isinstance(out, torch.Tensor):
            for key in ('inst_loss', 'instance_loss', 'aux_loss'):
                if key in out.keys():
                    return out[key]
        return None

    def forward(self, feats, coords=None):
        N   = feats.shape[0]
        d   = {'features': feats}
        if self.pass_coords and coords is not None:
            d['coords'] = coords.float()
        bag = TensorDict(d, batch_size=[N])
        out = self.model(bag)
        return self._extract_logit(out), self._extract_inst_loss(out)


def make_abmil():
    if not TORCHMIL_OK: raise ImportError("torchmil not available")
    return TorchMILAdapter(tm.ABMIL(in_shape=(IN_DIM,), n_classes=1))

def make_clam():
    if not TORCHMIL_OK: raise ImportError("torchmil not available")
    return TorchMILAdapter(tm.CLAM(in_shape=(IN_DIM,), n_classes=1))

def make_dsmil():
    if not TORCHMIL_OK: raise ImportError("torchmil not available")
    return TorchMILAdapter(tm.DSMIL(in_shape=(IN_DIM,), n_classes=1))

def make_transmil():
    if not TORCHMIL_OK: raise ImportError("torchmil not available")
    return TorchMILAdapter(tm.TransMIL(in_shape=(IN_DIM,), n_classes=1),
                           pass_coords=True)

def make_gtp():
    if not TORCHMIL_OK: raise ImportError("torchmil not available")
    return TorchMILAdapter(tm.GTP(in_shape=(IN_DIM,), n_classes=1),
                           pass_coords=True)


# ══════════════════════════════════════════════════════════════════════════════
# OUR MODELS
# ══════════════════════════════════════════════════════════════════════════════

try:
    from torch_geometric.nn import GATv2Conv as _GATv2Conv
    PYGEOM_OK = True
    print("torch_geometric available — PatchGCN using GATv2Conv", flush=True)
except ImportError:
    PYGEOM_OK = False
    print("torch_geometric not available — PatchGCN using GraphSAGE-mean fallback", flush=True)


class PatchGCN(nn.Module):
    """Chen et al. MICCAI 2021 — spatial k-NN graph MIL.

    Uses GATv2Conv (Brody et al. ICLR 2022) when torch_geometric is available,
    otherwise falls back to manual GraphSAGE-mean (same spatial idea, no PyG dep).
    Both variants: K=8 spatial neighbours, 4 layers, residual + LayerNorm, global mean pool.
    """
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, k=KNN_K,
                 n_layers=4, heads=4, dropout=DROPOUT):
        super().__init__()
        self.k      = k
        self.use_pyg = PYGEOM_OK
        self.proj   = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        if self.use_pyg:
            self.convs = nn.ModuleList([
                _GATv2Conv(hidden, hidden // heads, heads=heads,
                           dropout=dropout, concat=True, add_self_loops=True)
                for _ in range(n_layers)])
        else:
            self.convs = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden),
                              nn.GELU(), nn.Dropout(dropout))
                for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop  = nn.Dropout(dropout)
        self.head  = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    @staticmethod
    def _edge_index(coords_np, k, device):
        knn_idx = build_knn_idx(coords_np, k)
        N = knn_idx.shape[0]
        src = np.repeat(np.arange(N), k)
        dst = knn_idx.reshape(-1)
        ei  = torch.tensor(np.stack([np.concatenate([src, dst]),
                                      np.concatenate([dst, src])], 0),
                           dtype=torch.long, device=device)
        return ei

    def forward(self, feats, coords):
        h = self.proj(feats)
        if self.use_pyg:
            ei = self._edge_index(coords.cpu().numpy(), self.k, feats.device)
            for conv, norm in zip(self.convs, self.norms):
                h = norm(h + self.drop(conv(h, ei)))
        else:
            knn_idx = torch.from_numpy(
                build_knn_idx(coords.cpu().numpy(), self.k)).long().to(feats.device)
            for conv, norm in zip(self.convs, self.norms):
                nbr = h[knn_idx].mean(1)
                h   = norm(h + conv(torch.cat([h, nbr], -1)))
        return self.head(h.mean(0)).squeeze(), None


class SpatialKNNMIL(nn.Module):
    """PointTransformer MIL, no masking."""
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, dropout=DROPOUT,
                 k=KNN_K, n_layers=KNN_NLAYERS):
        super().__init__()
        self.k = k; self.n_layers = n_layers
        self.proj    = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.blocks  = nn.ModuleList([PointTransformerBlock(hidden, k=k, dropout=dropout)
                                      for _ in range(n_layers)])
        self.pools   = nn.ModuleList([GatedPool(hidden) for _ in range(n_layers)])
        self.heads   = nn.ModuleList([
            nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
            for _ in range(n_layers)])
        self.layer_w = nn.Parameter(torch.zeros(n_layers))
    def forward(self, feats, coords):
        knn_idx = torch.from_numpy(
            build_knn_idx(coords.cpu().numpy(), self.k)).long().to(feats.device)
        x = self.proj(feats)
        logits, attns = [], []
        for blk, pool, head in zip(self.blocks, self.pools, self.heads):
            x = blk(x, coords, knn_idx)
            rep, attn = pool(x)
            logits.append(head(rep).squeeze()); attns.append(attn)
        w = torch.softmax(self.layer_w, dim=0)
        return sum(w[i]*logits[i] for i in range(self.n_layers)), logits, attns


class MaskedKNNMIL(nn.Module):
    """Masked spatial KNN-MIL with ablation flags.
    recon_loss: 'cosine' (default) | 'mse'
    """
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, dropout=DROPOUT,
                 k=KNN_K, n_layers=KNN_NLAYERS, mask_ratio=MASK_RATIO,
                 recon_loss='cosine'):
        super().__init__()
        self.k = k; self.n_layers = n_layers; self.mask_ratio = mask_ratio
        self.recon_loss_type = recon_loss
        self.mask_token = nn.Parameter(torch.zeros(1, hidden))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.proj    = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.blocks  = nn.ModuleList([PointTransformerBlock(hidden, k=k, dropout=dropout)
                                      for _ in range(n_layers)])
        self.pools   = nn.ModuleList([GatedPool(hidden) for _ in range(n_layers)])
        self.heads   = nn.ModuleList([
            nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
            for _ in range(n_layers)])
        self.layer_w    = nn.Parameter(torch.zeros(n_layers))
        self.recon_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, in_dim))
        self._stop_grad_recon = False

    def forward(self, feats, coords):
        N = feats.shape[0]
        knn_idx = torch.from_numpy(
            build_knn_idx(coords.cpu().numpy(), self.k)).long().to(feats.device)
        x = self.proj(feats)
        mask = None
        if self.training and self.mask_ratio > 0:
            n_mask = max(1, int(N * self.mask_ratio))
            mask   = torch.zeros(N, dtype=torch.bool, device=feats.device)
            mask[torch.randperm(N, device=feats.device)[:n_mask]] = True
            x = x.clone()
            x[mask] = self.mask_token.expand(mask.sum(), -1)
        logits, attns = [], []
        for blk, pool, head in zip(self.blocks, self.pools, self.heads):
            x = blk(x, coords, knn_idx)
            rep, attn = pool(x)
            logits.append(head(rep).squeeze()); attns.append(attn)
        w = torch.softmax(self.layer_w, dim=0)
        final = sum(w[i]*logits[i] for i in range(self.n_layers))
        if mask is not None and mask.sum() > 0:
            enc_out = x[mask].detach() if self._stop_grad_recon else x[mask]
            recon_pred   = self.recon_head(enc_out)
            recon_target = F.normalize(feats[mask].detach(), dim=-1)
        else:
            recon_pred = recon_target = None
        return final, logits, attns, recon_pred, recon_target

    def recon_loss(self, pred, target):
        if self.recon_loss_type == 'mse':
            return F.mse_loss(pred.float(), target.float())
        return cosine_recon_loss(pred, target)


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class AblCfg:
    pretrain_epochs: int  = PRETRAIN_EPOCHS
    use_stopgrad:    bool = True
    use_annealing:   bool = True
    use_altern:      bool = True    # False → additive CLS+λ·RECON


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, paths, cw, seed=None):
    model.eval()
    probs, truths = [], []
    for i, path in enumerate(paths):
        s = (seed + i) if seed is not None else None
        feats, coords, label, _, _ = load_slide(path, seed=s)
        feats = feats.to(device); coords = coords.float().to(device)
        out   = model(feats, coords)
        logit = out[0] if isinstance(out, (tuple, list)) else out
        probs.append(torch.sigmoid(logit).item())
        truths.append(int(label))
    auc  = roc_auc_score(truths, probs) if len(set(truths)) > 1 else 0.5
    bacc = balanced_accuracy_score(truths, [1 if p > 0.5 else 0 for p in probs])
    return auc, bacc, probs, truths


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN FOLD
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(model, model_type, tr_paths, va_paths, fold, tag, cw, scaler,
               abl: AblCfg = None):
    if abl is None: abl = AblCfg()
    is_masked      = (model_type == 'maskedknn')
    pretrain_total = abl.pretrain_epochs if is_masked else 0
    total_epochs   = pretrain_total + JOINT_EPOCHS

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs, eta_min=LR*0.01)
    best_bacc, best_state, no_improve = 0.0, None, 0

    for epoch in range(total_epochs):
        pretrain_phase = is_masked and (epoch < pretrain_total)
        joint_epoch    = max(0, epoch - pretrain_total)
        if is_masked:
            model._stop_grad_recon = abl.use_stopgrad and (not pretrain_phase)

        model.train()
        random.shuffle(tr_paths)
        opt.zero_grad()
        loss_sum = 0.0; n_steps = 0; accum = 0

        for path in tr_paths:
            try:
                feats, coords, label, _, _ = load_slide(path, seed=epoch)
                feats  = feats.to(device)
                coords = coords.float().to(device)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    out   = model(feats, coords)
                    logit = out[0] if isinstance(out, (tuple, list)) else out

                    if model_type == 'torchmil_clam':
                        inst_loss = out[1] if isinstance(out, tuple) else None
                        cls_loss  = hinge_loss(logit, label, cw)
                        raw_loss  = (cls_loss + 0.3 * inst_loss
                                     if inst_loss is not None else cls_loss)
                    elif is_masked and isinstance(out, tuple) and out[3] is not None:
                        recon = model.recon_loss(out[3], out[4])
                        if pretrain_phase:
                            raw_loss = recon
                        elif not abl.use_altern:
                            # additive: both terms every slide
                            raw_loss = hinge_loss(logit, label, cw) + LAMBDA_RECON * recon
                        else:
                            p_r = (0.5 if not abl.use_annealing
                                   else p_recon_schedule(joint_epoch, JOINT_EPOCHS))
                            raw_loss = ((LAMBDA_RECON * recon) if random.random() < p_r
                                        else hinge_loss(logit, label, cw))
                    else:
                        raw_loss = hinge_loss(logit, label, cw)

                    loss = raw_loss / GRAD_ACCUM

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_sum += loss.item() * GRAD_ACCUM
                accum += 1
                if accum == GRAD_ACCUM:
                    if scaler:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(opt); scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                    opt.zero_grad(); n_steps += 1; accum = 0

            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(); accum = 0; torch.cuda.empty_cache()
                print(f"  [OOM] {path.name}", flush=True)

        if accum > 0:
            if scaler:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            opt.zero_grad(); n_steps += 1
        sch.step()

        if (epoch + 1) % PATIENCE_EVERY == 0:
            mean_loss = loss_sum / max(n_steps, 1)
            if pretrain_phase:
                print(f"  [{tag}] f{fold} ep={epoch+1:3d} [PRETRAIN] loss={mean_loss:.4f}",
                      flush=True)
            else:
                val_auc, val_bacc, _, _ = evaluate(model, va_paths, cw, seed=EVAL_SEED)
                print(f"  [{tag}] f{fold} ep={epoch+1:3d} loss={mean_loss:.4f}"
                      f" val_auc={val_auc:.4f} val_bacc={val_bacc:.4f}", flush=True)
                if val_bacc > best_bacc:
                    best_bacc  = val_bacc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= PATIENCE:
                        print(f"  [{tag}] f{fold} early stop ep={epoch+1}", flush=True)
                        break

    if best_state: model.load_state_dict(best_state)
    return best_bacc, model


# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# (tag, display_label, hex_color, build_fn, model_type, abl_cfg_or_None)
# ══════════════════════════════════════════════════════════════════════════════

REGISTRY = [
    # ── torchmil published baselines ─────────────────────────────────────────
    ('abmil',    'ABMIL (Ilse 2018)',         '#4477CC', make_abmil,   'torchmil',       None),
    ('clam',     'CLAM-SB (Lu 2021)',         '#22AA44', make_clam,    'torchmil_clam',  None),
    ('dsmil',    'DSMIL (Li 2021)',           '#FF8800', make_dsmil,   'torchmil',       None),
    ('transmil', 'TransMIL (Shao 2021)',      '#AA2244', make_transmil,'torchmil',       None),
    ('gtp',      'GTP (Zheng 2023)',          '#884400', make_gtp,     'torchmil',       None),
    # ── our spatial models ────────────────────────────────────────────────────
    ('patchgcn', 'PatchGCN (Chen 2021)',      '#665500', lambda: PatchGCN(),        'patchgcn', None),
    ('knnmil',   'SpatialKNNMIL (ours)',      '#9B30FF', lambda: SpatialKNNMIL(),  'knnmil',   None),
    ('maskedknn','MaskedKNNMIL (ours)',       '#FF69B4', lambda: MaskedKNNMIL(),   'maskedknn',None),
    # ── ablations ─────────────────────────────────────────────────────────────
    ('abl_nopretrain', 'w/o pretrain',        '#BBBBBB',
     lambda: MaskedKNNMIL(),            'maskedknn', AblCfg(pretrain_epochs=0)),
    ('abl_mse',        'MSE recon',           '#CCAAAA',
     lambda: MaskedKNNMIL(recon_loss='mse'), 'maskedknn', None),
    ('abl_nosg',       'no stop-grad',        '#AACCAA',
     lambda: MaskedKNNMIL(),            'maskedknn', AblCfg(use_stopgrad=False)),
    ('abl_noannealing','no annealing',        '#AAAACC',
     lambda: MaskedKNNMIL(),            'maskedknn', AblCfg(use_annealing=False)),
    ('abl_additive',   'additive loss',       '#CCCCAA',
     lambda: MaskedKNNMIL(),            'maskedknn', AblCfg(use_altern=False)),
    ('abl_1scale',     '1-scale ABMIL',       '#AACCCC',
     lambda: MaskedKNNMIL(n_layers=1), 'maskedknn', None),
]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
set_seeds()

RUN_MODEL = os.environ.get("MODEL", "all")
RUN_FOLD  = int(os.environ.get("FOLD", "-1"))

all_paths = sorted(FEAT_DIR.glob("*.pt"))
labels, patients = [], []
for p in all_paths:
    d = torch.load(p, map_location='cpu', weights_only=False)
    labels.append(d['label']); patients.append(d['patient'])
labels   = np.array(labels)
patients = np.array(patients)
print(f"Total slides: {len(all_paths)}  ACR+={labels.sum()}  "
      f"ACR-={len(labels)-labels.sum()}", flush=True)

cv     = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
splits = list(cv.split(all_paths, labels, patients))

results = {tag: [] for tag, *_ in REGISTRY}

fold_iter = [(f, s) for f, s in enumerate(splits) if RUN_FOLD == -1 or f == RUN_FOLD]

for fold, (tr_idx, te_idx) in fold_iter:
    print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}", flush=True)
    tr_paths = [all_paths[i] for i in tr_idx]
    te_paths = [all_paths[i] for i in te_idx]
    inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    inner_tr, inner_va = next(inner_cv.split(tr_paths, labels[tr_idx], patients[tr_idx]))
    va_paths  = [tr_paths[i] for i in inner_va]
    tr_paths2 = [tr_paths[i] for i in inner_tr]
    cw        = compute_cw([labels[tr_idx[i]] for i in inner_tr])
    print(f"  train={len(tr_paths2)} val={len(va_paths)} test={len(te_paths)}"
          f"  cw=[{cw[0]:.2f},{cw[1]:.2f}]", flush=True)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for tag, label, _, build_fn, model_type, abl_cfg in REGISTRY:
        if RUN_MODEL not in ('all', tag): continue
        print(f"\n  --- {label} ---", flush=True)
        set_seeds(SEED + fold)
        try:
            m = build_fn().to(device)
        except Exception as e:
            print(f"  [SKIP] {tag}: build failed — {e}", flush=True)
            continue
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  Params: {n_params:,}", flush=True)
        _, m = train_fold(m, model_type, tr_paths2, va_paths, fold, tag.upper(), cw,
                          scaler, abl=abl_cfg)
        auc, bacc, probs, truths = evaluate(m, te_paths, cw, seed=EVAL_SEED)
        print(f"  [{tag.upper()}] fold={fold}  TEST AUC={auc:.4f}  BACC={bacc:.4f}", flush=True)
        torch.save(m.state_dict(), OUTDIR / f"{tag}_fold{fold}.pt")
        results[tag].append({'fold': fold, 'auc': auc, 'bacc': bacc,
                              'probs': probs, 'truths': truths, 'n_params': n_params})
        json.dump({'model': tag, 'fold': fold, 'auc': auc, 'bacc': bacc},
                  open(OUTDIR / f"result_{tag}_fold{fold}.json", 'w'))


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
def _stats(tag):
    rs = results[tag]
    if not rs: return None, None, None, None
    aucs  = [r['auc']  for r in rs]
    baccs = [r['bacc'] for r in rs]
    return np.mean(aucs), np.std(aucs), np.mean(baccs), np.std(baccs)

print(f"\n{'='*72}", flush=True)
print(f"{'Model':<30} {'AUC':>12} {'BACC':>12} {'#Params':>10}", flush=True)
print(f"{'─'*72}", flush=True)
SECTIONS = {
    'abmil':           '── Published baselines (torchmil) ──',
    'patchgcn':        '── Spatial models (our impl) ──',
    'abl_nopretrain':  '── Ablations of MaskedKNNMIL ──',
}
for tag, lbl, *_ in REGISTRY:
    if not results[tag]: continue
    if tag in SECTIONS:
        print(f"\n  {SECTIONS[tag]}", flush=True)
    a_m, a_s, b_m, b_s = _stats(tag)
    n = results[tag][0].get('n_params', 0)
    print(f"  {lbl:<28} {a_m:.3f}±{a_s:.3f}  {b_m:.3f}±{b_s:.3f}  {n/1e3:>7.1f}K",
          flush=True)
print(f"{'='*72}", flush=True)

json.dump(results, open(OUTDIR / "results_all.json", 'w'),
          default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x,
          indent=2)

# ── Figure: AUC bars (benchmark) + delta bars (ablations) ────────────────────
def _auc_stats(tag):
    rs = results[tag]
    if not rs: return 0.0, 0.0
    v = [r['auc'] for r in rs]
    return float(np.mean(v)), float(np.std(v))

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('MIL Benchmark — ACR Classification (UNI features, 5-fold CV)', fontsize=12)
colors = {t: c for t, _, c, *_ in REGISTRY}

# panel 1: benchmark models
bench = ['abmil','clam','dsmil','transmil','gtp','patchgcn','knnmil','maskedknn']
bench = [t for t in bench if results[t]]
x = np.arange(len(bench))
means = [_auc_stats(t)[0] for t in bench]
stds  = [_auc_stats(t)[1] for t in bench]
xlbls = [next(l for tt,l,*_ in REGISTRY if tt==t) for t in bench]
ax = axes[0]
ax.bar(x, means, yerr=stds, capsize=4, color=[colors[t] for t in bench],
       alpha=0.85, edgecolor='white', width=0.65)
ax.axhline(0.5, color='grey', lw=0.8, ls='--', label='chance')
ax.set_xticks(x); ax.set_xticklabels(xlbls, rotation=35, ha='right', fontsize=7)
ax.set_ylabel('Test AUC'); ax.set_ylim(0.4, 1.0)
ax.set_title('Benchmark: published (torchmil) + ours')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# panel 2: ablation deltas
full_auc = _auc_stats('maskedknn')[0]
abls = [t for t,*_ in REGISTRY if t.startswith('abl_') and results[t]]
deltas = [_auc_stats(t)[0] - full_auc for t in abls]
astds  = [_auc_stats(t)[1] for t in abls]
albls  = [next(l for tt,l,*_ in REGISTRY if tt==t).replace('w/o ','−').replace('no ','−')
           for t in abls]
ax2 = axes[1]
ax2.barh(np.arange(len(abls)), deltas, xerr=astds, capsize=3,
         color=[colors[t] for t in abls], alpha=0.85, edgecolor='white')
ax2.axvline(0, color='black', lw=1.0)
ax2.set_yticks(np.arange(len(abls))); ax2.set_yticklabels(albls, fontsize=8)
ax2.set_xlabel('ΔAUC vs full MaskedKNNMIL  (negative = component helps)')
ax2.set_title('Ablations')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(OUTDIR / 'benchmark_summary.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("Saved benchmark_summary.pdf", flush=True)
PYEOF
