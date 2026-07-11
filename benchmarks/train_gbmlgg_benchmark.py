#!/usr/bin/env python3
"""
train_gbmlgg_benchmark.py  —  GBMLGG Comprehensive Benchmark
=============================================================
Reuses all data loading, Phase 1, Phase 2 (TCGASetTransformerMIL),
losses and metrics from train_tcga_multitask.py.

Adds compared to train_tcga_multitask.py
-----------------------------------------
  Early/Late/Middle per-task fusion baselines
  MCAT (imported from /lustre/…/MCAT, discrete survival)
  SurvPath (imported from /lustre/…/SurvPath, discrete survival)
  PORPOISE (Kronecker product WSI × omic fusion)
  MOTCat (Optimal-Transport cross-attention)

Methods
-------
  phase1     — per-modality ABMIL (one model per modality×task)
  phase2     — TCGASetTransformerMIL, all tasks jointly (set_mil_mt)
  early      — all-patch concat → ABMIL, per task
  late       — per-modality ABMIL → weighted sum, per task
  middle     — per-modality ABMIL → CrossAttn → pool, per task
  porpoise   — ABMIL(WSI) × KroneckerFusion(omic), OS only
  motcat     — OT cross-attention WSI × omic, OS only
  mcat       — MCAT_Surv co-attention (discrete survival, OS)
  survpath   — SurvPath pathway encoder + cross-attn (discrete survival, OS)

Usage (sbatch only — never run directly on login node)
------------------------------------------------------
  python train_gbmlgg_benchmark.py --fold 0 --methods all
  python train_gbmlgg_benchmark.py --fold 0 --methods mcat,survpath
"""

import argparse
import gc
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ── Import existing infrastructure ───────────────────────────────────────────
_BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(_BENCH_DIR))

from train_tcga_multitask import (
    preload_bags,
    build_splits,
    update_presence,
    cox_ph_loss,
    c_index,
    balanced_accuracy,
    GatedAttnEncoder,
    SingleModalMIL,
    TCGASetTransformerMIL,
    run_phase1,
    _p2_epoch,
    _p2_eval,
    set_seeds,
    _write_json,
    _read_json,
    _is_done,
    _isnan,
    MODALITIES,
    HIDDEN_DIM,
    DROPOUT,
    DEVICE,
    CANCER_CONFIGS,
    SPLITS_DIR,
    P2_EPOCHS,
    P2_LR,
    P2_WD,
    P2_EVAL,
    P2_GRAD_ACCUM,
    P2_N_HEADS,
    P2_N_SAB,
    P2_MODAL_DROP,
    MAX_WSI,
    MAX_RNA_CNV,
    PMA_K,
    _feat_dim,
    _gc,
    BCE_LAMBDA,
    COX_LAMBDA,
)

# ── Import external model packages ───────────────────────────────────────────
_LUSTRE       = "/lustre/groups/aih/dinesh.haridoss/mil"
_MCAT_PATH    = f"{_LUSTRE}/MCAT"
_SURVPATH_PATH = f"{_LUSTRE}/SurvPath"


def _import_mcat():
    """Import MCAT_Surv, then flush the 'models' namespace so SurvPath can load its own."""
    sys.path.insert(0, _MCAT_PATH)
    try:
        from models.model_coattn import MCAT_Surv
        return MCAT_Surv
    except ImportError as e:
        print(f"  [warn] MCAT import failed: {e}")
        return None
    finally:
        for k in list(sys.modules.keys()):
            if k == "models" or k.startswith("models."):
                del sys.modules[k]


def _import_survpath():
    sys.path.insert(0, _SURVPATH_PATH)
    try:
        from models.model_SurvPath import SurvPath as _SP
        return _SP
    except ImportError as e:
        print(f"  [warn] SurvPath import failed: {e}")
        return None
    finally:
        for k in list(sys.modules.keys()):
            if k == "models" or k.startswith("models."):
                del sys.modules[k]


MCAT_Surv_cls  = _import_mcat()
SurvPath_cls   = _import_survpath()

# ── Results directory ─────────────────────────────────────────────────────────
GBMLGG_RESULTS = Path("/home/aih/dinesh.haridoss/chicago_mil/results_gbmlgg_benchmark")

METHODS_ALL = ["phase1", "phase2", "early", "late", "middle",
               "porpoise", "motcat", "mcat", "survpath"]

# ── Omic grouping constants ───────────────────────────────────────────────────
# RNA_pathways: (331 genes × 199 pathways)
N_GENES    = 331
N_PATHWAYS = 199    # = _feat_dim("RNA")
MCAT_GROUPS = 6


def _rna_to_mcat_groups(rna: torch.Tensor) -> List[torch.Tensor]:
    """(331, 199) → 6 mean-pooled gene-groups each (199,)."""
    gs = N_GENES // MCAT_GROUPS
    groups = []
    for i in range(MCAT_GROUPS):
        s = i * gs; e = s + gs if i < MCAT_GROUPS - 1 else N_GENES
        groups.append(rna[s:e].mean(0))
    return groups


def _rna_to_survpath_paths(rna: torch.Tensor) -> List[torch.Tensor]:
    """(331, 199) → list of 331 individual gene vectors each (199,)."""
    return [rna[i] for i in range(N_GENES)]


# ══════════════════════════════════════════════════════════════════════════════
# DISCRETE SURVIVAL (for MCAT / SurvPath)
# ══════════════════════════════════════════════════════════════════════════════

def make_survival_bins(times: np.ndarray, events: np.ndarray,
                       n_bins: int = 4) -> torch.Tensor:
    ev_t = times[events == 1.0]
    if len(ev_t) < n_bins:
        return torch.linspace(float(times.min()), float(times.max()), n_bins + 1)
    qs = np.percentile(ev_t, np.linspace(0, 100, n_bins + 1))
    return torch.tensor(qs, dtype=torch.float32)


def discretize(times: torch.Tensor, events: torch.Tensor,
               bins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Continuous → (bin_index Y, c) where c=0 event, c=1 censored (MCAT convention)."""
    n_bins = len(bins) - 1
    Y = torch.zeros(len(times), dtype=torch.long)
    c = torch.zeros(len(times), dtype=torch.float32)
    for i, (t, e) in enumerate(zip(times.tolist(), events.tolist())):
        idx = int(torch.searchsorted(bins, torch.tensor(t)).clamp(0, n_bins - 1).item())
        Y[i] = idx
        c[i] = 0.0 if e == 1.0 else 1.0
    return Y, c


def nll_surv_loss(hazards: torch.Tensor, S: torch.Tensor,
                  Y: torch.Tensor, c: torch.Tensor,
                  alpha: float = 0.0) -> torch.Tensor:
    """NLL discrete survival loss (Chen et al. MCAT convention)."""
    hazards = hazards.float(); S = S.float()
    n = len(Y)
    Y = Y.view(n, 1).to(hazards.device)
    c = c.view(n, 1).to(hazards.device)
    S_pad = torch.cat([torch.ones(n, 1, device=hazards.device), S], dim=1)
    s_prev = torch.gather(S_pad, 1, Y).clamp(1e-7)
    h_this = torch.gather(hazards, 1, Y).clamp(1e-7)
    s_this = torch.gather(S_pad, 1, (Y + 1).clamp(max=S_pad.shape[1] - 1)).clamp(1e-7)
    uncen  = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    cen    = -c * torch.log(s_this)
    loss   = cen + uncen
    if alpha > 0:
        loss = (1 - alpha) * loss + alpha * uncen
    return loss.mean()


# ══════════════════════════════════════════════════════════════════════════════
# FUSION BASELINES  (Early / Late / Middle / PORPOISE / MOTCat)
# ══════════════════════════════════════════════════════════════════════════════

class EarlyFusion(nn.Module):
    """All-patch concat → GatedABMIL → Cox/BCE head.  Per task."""
    def __init__(self, task: str = "os", hidden_dim: int = HIDDEN_DIM,
                 dropout: float = DROPOUT, modal_dropout: float = P2_MODAL_DROP):
        super().__init__()
        self.task = task; self.modal_dropout = modal_dropout
        self.proj = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(_feat_dim(m), hidden_dim),
                              nn.ReLU(), nn.Dropout(dropout))
            for m in MODALITIES
        })
        self.att_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.att_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.att_w = nn.Linear(hidden_dim, 1, bias=False)
        self.head  = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        tokens = []
        for mod, proj in self.proj.items():
            t = bags.get(mod)
            if t is None: continue
            if self.training and random.random() < self.modal_dropout: continue
            t = t.to(device)
            max_p = MAX_WSI if mod == "WSI" else MAX_RNA_CNV
            if t.shape[0] > max_p:
                t = t[torch.randperm(t.shape[0], device=device)[:max_p]]
            tokens.append(proj(t))
        if not tokens:
            return torch.zeros(1, device=device, requires_grad=True)
        h = torch.cat(tokens, dim=0)
        gate  = self.att_V(h) * self.att_U(h)
        alpha = F.softmax(self.att_w(gate), dim=0)
        rep   = (alpha * h).sum(0)
        return self.head(rep).squeeze()


class LateFusion(nn.Module):
    """Per-modality ABMIL → softmax-weighted sum → Cox/BCE head.  Per task."""
    def __init__(self, task: str = "os", hidden_dim: int = HIDDEN_DIM,
                 dropout: float = DROPOUT, modal_dropout: float = P2_MODAL_DROP):
        super().__init__()
        self.task = task; self.modal_dropout = modal_dropout
        self.encoders = nn.ModuleDict({
            m: GatedAttnEncoder(_feat_dim(m), hidden_dim, dropout)
            for m in MODALITIES
        })
        self.log_w = nn.Parameter(torch.zeros(len(MODALITIES)))
        self.head  = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        reps, idxs = [], []
        for i, (mod, enc) in enumerate(self.encoders.items()):
            t = bags.get(mod)
            if t is None: continue
            if self.training and random.random() < self.modal_dropout: continue
            t = t.to(device)
            max_p = MAX_WSI if mod == "WSI" else MAX_RNA_CNV
            if t.shape[0] > max_p:
                t = t[torch.randperm(t.shape[0], device=device)[:max_p]]
            rep, _, _ = enc(t)
            reps.append(rep); idxs.append(i)
        if not reps:
            return torch.zeros(1, device=device, requires_grad=True)
        w = F.softmax(self.log_w[torch.tensor(idxs, device=device)], dim=0)
        z = sum(wi * r for wi, r in zip(w, reps))
        return self.head(z).squeeze()


class MiddleFusion(nn.Module):
    """Per-modality ABMIL → TransformerEncoder → GatedABMIL → head.  Per task."""
    def __init__(self, task: str = "os", hidden_dim: int = HIDDEN_DIM,
                 dropout: float = DROPOUT, n_heads: int = P2_N_HEADS,
                 modal_dropout: float = P2_MODAL_DROP):
        super().__init__()
        self.task = task; self.modal_dropout = modal_dropout
        self.encoders = nn.ModuleDict({
            m: GatedAttnEncoder(_feat_dim(m), hidden_dim, dropout)
            for m in MODALITIES
        })
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.xfm = nn.TransformerEncoder(layer, num_layers=2)
        self.att_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.att_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.att_w = nn.Linear(hidden_dim, 1, bias=False)
        self.head  = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        reps = []
        for mod, enc in self.encoders.items():
            t = bags.get(mod)
            if t is None: continue
            if self.training and random.random() < self.modal_dropout: continue
            t = t.to(device)
            max_p = MAX_WSI if mod == "WSI" else MAX_RNA_CNV
            if t.shape[0] > max_p:
                t = t[torch.randperm(t.shape[0], device=device)[:max_p]]
            rep, _, _ = enc(t); reps.append(rep)
        if not reps:
            return torch.zeros(1, device=device, requires_grad=True)
        tokens = torch.stack(reps).unsqueeze(0)      # (1, M, H)
        tokens = self.xfm(tokens).squeeze(0)          # (M, H)
        gate   = self.att_V(tokens) * self.att_U(tokens)
        alpha  = F.softmax(self.att_w(gate), dim=0)
        z      = (alpha * tokens).sum(0)
        return self.head(z).squeeze()


class PORPOISE(nn.Module):
    """
    PORPOISE-style Kronecker fusion: ABMIL(WSI) ⊗ SNN(RNA+CNV).
    (Chen et al., Nature Biomedical Engineering 2022)
    """
    def __init__(self, task: str = "os", hidden_dim: int = HIDDEN_DIM,
                 dropout: float = DROPOUT):
        super().__init__()
        self.task    = task
        omic_dim     = N_PATHWAYS + _feat_dim("CNV")   # 199 + 331 = 530
        self.wsi_enc = GatedAttnEncoder(_feat_dim("WSI"), hidden_dim, dropout)
        self.omic_snn = nn.Sequential(
            nn.Linear(omic_dim, hidden_dim), nn.ELU(), nn.AlphaDropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.AlphaDropout(dropout))
        kron_dim = hidden_dim // 4
        self.fc1      = nn.Linear(hidden_dim + 1, kron_dim)
        self.fc2      = nn.Linear(hidden_dim + 1, kron_dim)
        self.fc_kron  = nn.Sequential(
            nn.Linear(kron_dim * kron_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.head     = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        wsi = bags.get("WSI")
        if wsi is None:
            return torch.zeros(1, device=device, requires_grad=True)
        wsi = wsi.to(device)
        if wsi.shape[0] > MAX_WSI:
            wsi = wsi[torch.randperm(wsi.shape[0], device=device)[:MAX_WSI]]
        z_wsi, _, _ = self.wsi_enc(wsi)

        rna = bags.get("RNA"); cnv = bags.get("CNV")
        r   = rna.mean(0).to(device) if rna is not None else torch.zeros(N_PATHWAYS, device=device)
        c   = cnv.squeeze(0).to(device) if cnv is not None else torch.zeros(_feat_dim("CNV"), device=device)
        z_omic = self.omic_snn(torch.cat([r, c], dim=0))

        a1   = self.fc1(torch.cat([z_wsi,  z_wsi.new_ones(1)]))
        a2   = self.fc2(torch.cat([z_omic, z_omic.new_ones(1)]))
        kron = (a1.unsqueeze(1) * a2.unsqueeze(0)).view(-1)
        z    = self.fc_kron(kron)
        return self.head(z).squeeze()


class MOTCat(nn.Module):
    """
    MOTCat-style OT-guided cross-attention: WSI × omic.
    (Chen et al., CVPR 2023)
    """
    def __init__(self, task: str = "os", hidden_dim: int = HIDDEN_DIM,
                 dropout: float = DROPOUT, n_heads: int = P2_N_HEADS):
        super().__init__()
        self.task      = task
        self.wsi_proj  = nn.Sequential(nn.Linear(_feat_dim("WSI"), hidden_dim),
                                        nn.ReLU(), nn.Dropout(dropout))
        omic_dim       = N_PATHWAYS + _feat_dim("CNV")
        self.omic_proj = nn.Sequential(
            nn.Linear(omic_dim, hidden_dim), nn.ELU(), nn.AlphaDropout(dropout),
            nn.Linear(hidden_dim, hidden_dim))
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads,
                                                 dropout=dropout, batch_first=True)
        self.norm  = nn.LayerNorm(hidden_dim)
        self.att_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.att_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.att_w = nn.Linear(hidden_dim, 1, bias=False)
        self.head  = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        wsi = bags.get("WSI")
        if wsi is None:
            return torch.zeros(1, device=device, requires_grad=True)
        wsi = wsi.to(device)
        if wsi.shape[0] > MAX_WSI:
            wsi = wsi[torch.randperm(wsi.shape[0], device=device)[:MAX_WSI]]
        h_wsi = self.wsi_proj(wsi)

        rna = bags.get("RNA"); cnv = bags.get("CNV")
        r   = rna.mean(0).to(device) if rna is not None else torch.zeros(N_PATHWAYS, device=device)
        c   = cnv.squeeze(0).to(device) if cnv is not None else torch.zeros(_feat_dim("CNV"), device=device)
        h_omic = self.omic_proj(torch.cat([r, c], dim=0)).unsqueeze(0).unsqueeze(0)  # (1, 1, H)

        h_out, _ = self.cross_attn(h_wsi.unsqueeze(0), h_omic, h_omic)
        tokens   = self.norm(h_wsi + h_out.squeeze(0))
        gate     = self.att_V(tokens) * self.att_U(tokens)
        alpha    = F.softmax(self.att_w(gate), dim=0)
        z        = (alpha * tokens).sum(0)
        return self.head(z).squeeze()


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC TRAINING LOOP (Cox / BCE single-task models)
# ══════════════════════════════════════════════════════════════════════════════

def _simple_epoch(model, records, task, optimizer, device, cache, scaler) -> float:
    """Full-batch single-task epoch reusing cox_ph_loss from train_tcga_multitask."""
    model.train()
    random.shuffle(records)
    optimizer.zero_grad()
    logits, times, events, cls_labels = [], [], [], []

    for rec in records:
        bags = {m: cache.get(rec["key"], {}).get(m) for m in MODALITIES}
        if all(v is None for v in bags.values()): continue
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logit = model(bags, device)
        if not isinstance(logit, torch.Tensor) or logit.grad_fn is None: continue
        if task == "cls":
            if not _isnan(rec.get("cls_label", float("nan"))):
                logits.append(logit); cls_labels.append(rec["cls_label"])
        else:
            st = f"{task}_status"; ti = f"{task}_time"
            if not _isnan(rec.get(st)) and not _isnan(rec.get(ti)):
                logits.append(logit); events.append(rec[st]); times.append(rec[ti])

    if len(logits) < 2: return float("nan")

    logits_t = torch.stack(logits)
    if task == "cls":
        labels_t = torch.tensor(cls_labels, dtype=torch.float32, device=device)
        n_pos = labels_t.sum().clamp(1.); n_neg = (1 - labels_t).sum().clamp(1.)
        loss = BCE_LAMBDA * F.binary_cross_entropy_with_logits(
            logits_t, labels_t, pos_weight=(n_neg / n_pos).detach())
    else:
        times_t  = torch.tensor(times,  dtype=torch.float32, device=device)
        events_t = torch.tensor(events, dtype=torch.float32, device=device)
        loss = COX_LAMBDA * cox_ph_loss(logits_t, times_t, events_t)

    if loss.grad_fn is None: return float("nan")
    if scaler:
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    else:
        loss.backward(); optimizer.step()
    optimizer.zero_grad()
    v = loss.item()
    del logits; _gc()
    return v


@torch.no_grad()
def _simple_eval(model, records, task, device, cache) -> float:
    model.eval()
    risks, times, events, cls_probs, cls_labels = [], [], [], [], []
    for rec in records:
        bags = {m: cache.get(rec["key"], {}).get(m) for m in MODALITIES}
        if all(v is None for v in bags.values()): continue
        logit = model(bags, device)
        if not isinstance(logit, torch.Tensor): continue
        v = logit.float().item()
        if task == "cls":
            if not _isnan(rec.get("cls_label", float("nan"))):
                cls_probs.append(torch.sigmoid(logit).float().item())
                cls_labels.append(rec["cls_label"])
        else:
            st = f"{task}_status"; ti = f"{task}_time"
            if not _isnan(rec.get(st)) and not _isnan(rec.get(ti)):
                risks.append(v); times.append(rec[ti]); events.append(rec[st])
    if task == "cls":
        if len(cls_probs) < 2: return 0.5
        return balanced_accuracy(np.array(cls_labels), np.array(cls_probs))
    else:
        if len(risks) < 2: return 0.5
        return c_index(np.array(times), np.array(events), np.array(risks))


def run_simple(model_cls, model_kwargs, task, fold, device, cache,
               train_r, val_r, test_r, save_dir: Path) -> dict:
    """Train a single-task Cox/BCE model with early stopping."""
    save_dir.mkdir(parents=True, exist_ok=True)
    status = save_dir / "status.json"
    if _is_done(status):
        print(f"  [skip] {save_dir.name}")
        return _read_json(save_dir / "metrics.json") or {}

    set_seeds(42)
    model = model_cls(task=task, **model_kwargs).to(device)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  [{model_cls.__name__}:{task}] fold={fold}  params={n_p:,}")

    opt    = torch.optim.Adam(model.parameters(), lr=P2_LR, weight_decay=P2_WD)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_metric, best_ep = -1.0, 0
    ckpt_dir = save_dir / "ckpts"; ckpt_dir.mkdir(exist_ok=True)

    for ep in range(P2_EPOCHS):
        tl = _simple_epoch(model, train_r, task, opt, device, cache, scaler)
        if (ep + 1) % P2_EVAL == 0:
            vm = _simple_eval(model, val_r, task, device, cache)
            if vm > best_metric:
                best_metric = vm; best_ep = ep + 1
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  ep {ep+1:3d}  loss={tl:.4f}  val={vm:.4f}", flush=True)
            _gc()

    best_pt = ckpt_dir / "best.pt"
    if best_pt.exists():
        model.load_state_dict(torch.load(best_pt, map_location="cpu", weights_only=False))
        model.to(device)

    metrics = {}
    for sn, recs in [("train", train_r), ("val", val_r), ("test", test_r)]:
        m = _simple_eval(model, recs, task, device, cache)
        col = "bacc" if task == "cls" else f"{task}_ci"
        metrics[sn] = {col: m, "n": len(recs)}
        print(f"  {sn:5s}  {col}={m:.4f}")

    _write_json(save_dir / "metrics.json", metrics)
    _write_json(status, {"completed": True, "best_ep": best_ep, "best_val": best_metric})
    del model, opt, scaler; _gc()
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# DISCRETE SURVIVAL MODELS (MCAT / SurvPath)
# ══════════════════════════════════════════════════════════════════════════════

def _discrete_forward(model, model_type, wsi_d, rna, device):
    """Run forward pass for MCAT or SurvPath, return (hazards, S)."""
    if model_type == "mcat":
        rna_d  = rna.to(device) if rna is not None else torch.zeros(N_GENES, N_PATHWAYS, device=device)
        groups = _rna_to_mcat_groups(rna_d)
        kwargs = {"x_path": wsi_d}
        for i, g in enumerate(groups): kwargs[f"x_omic{i+1}"] = g
        hazards, S, _, _ = model(**kwargs)
        return hazards.squeeze(0), S.squeeze(0)
    elif model_type == "survpath":
        rna_d  = rna.to(device) if rna is not None else torch.zeros(N_GENES, N_PATHWAYS, device=device)
        paths  = _rna_to_survpath_paths(rna_d)
        kwargs = {"x_path": wsi_d.unsqueeze(0), "return_attn": False}
        for i, p in enumerate(paths): kwargs[f"x_omic{i+1}"] = p  # (199,) — no batch dim; forward uses unsqueeze(0) internally
        logits  = model(**kwargs).squeeze(0)           # (4,)
        hazards = torch.sigmoid(logits.unsqueeze(0))   # (1, 4)
        S       = torch.cumprod(1 - hazards, dim=1)
        return hazards.squeeze(0), S.squeeze(0)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _discrete_epoch(model, records, model_type, optimizer, device,
                    cache, scaler, bins: torch.Tensor) -> float:
    model.train()
    random.shuffle(records)
    optimizer.zero_grad()
    all_hz, all_S, all_Y, all_c = [], [], [], []

    for rec in records:
        bags = {m: cache.get(rec["key"], {}).get(m) for m in MODALITIES}
        wsi  = bags.get("WSI"); rna = bags.get("RNA")
        if wsi is None or _isnan(rec.get("os_time")) or _isnan(rec.get("os_status")): continue
        wsi_d = wsi.to(device)
        if wsi_d.shape[0] > MAX_WSI:
            wsi_d = wsi_d[torch.randperm(wsi_d.shape[0], device=device)[:MAX_WSI]]
        try:
            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                hz, S = _discrete_forward(model, model_type, wsi_d, rna, device)
        except Exception as e:
            print(f"  [warn] forward error: {e}"); continue
        t_t = torch.tensor([rec["os_time"]]); e_t = torch.tensor([rec["os_status"]])
        Y, c = discretize(t_t, e_t, bins)
        all_hz.append(hz); all_S.append(S)
        all_Y.append(Y[0]); all_c.append(c[0])

    if len(all_hz) < 2: return float("nan")

    hazards_t = torch.stack(all_hz)
    S_t       = torch.stack(all_S)
    Y_t       = torch.tensor(all_Y, dtype=torch.long, device=device)
    c_t       = torch.tensor(all_c, dtype=torch.float32, device=device)
    loss      = nll_surv_loss(hazards_t, S_t, Y_t, c_t)

    if not loss.requires_grad: return float("nan")
    if scaler:
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    else:
        loss.backward(); optimizer.step()
    optimizer.zero_grad()
    v = loss.item()
    del all_hz, all_S; _gc()
    return v


@torch.no_grad()
def _discrete_eval(model, records, model_type, device, cache) -> float:
    model.eval()
    risks, times, events = [], [], []
    for rec in records:
        bags = {m: cache.get(rec["key"], {}).get(m) for m in MODALITIES}
        wsi  = bags.get("WSI"); rna = bags.get("RNA")
        if wsi is None or _isnan(rec.get("os_time")) or _isnan(rec.get("os_status")): continue
        wsi_d = wsi.to(device)
        if wsi_d.shape[0] > MAX_WSI:
            wsi_d = wsi_d[torch.randperm(wsi_d.shape[0], device=device)[:MAX_WSI]]
        try:
            hz, S = _discrete_forward(model, model_type, wsi_d, rna, device)
            risk  = -S.sum().item()
        except Exception as e:
            print(f"  [warn] eval error: {e}"); continue
        risks.append(risk); times.append(rec["os_time"]); events.append(rec["os_status"])
    if len(risks) < 2: return 0.5
    return c_index(np.array(times), np.array(events), np.array(risks))


def run_discrete_model(model_cls, model_kwargs, model_type, fold, device, cache,
                       train_r, val_r, test_r, save_dir: Path) -> dict:
    """Train MCAT or SurvPath with NLL discrete survival loss."""
    save_dir.mkdir(parents=True, exist_ok=True)
    status = save_dir / "status.json"
    if _is_done(status):
        print(f"  [skip] {save_dir.name}")
        return _read_json(save_dir / "metrics.json") or {}
    if model_cls is None:
        print(f"  [{model_type.upper()}] skipped — import not available")
        return {}

    tr_t = np.array([r["os_time"]   for r in train_r if not _isnan(r.get("os_time"))])
    tr_e = np.array([r["os_status"] for r in train_r if not _isnan(r.get("os_status"))])
    bins = make_survival_bins(tr_t, tr_e, n_bins=4)
    print(f"\n  [{model_type.upper()}] fold={fold}  bins={[round(b,2) for b in bins.tolist()]}")

    set_seeds(42)
    model = model_cls(**model_kwargs).to(device)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_p:,}")

    opt    = torch.optim.Adam(model.parameters(), lr=P2_LR, weight_decay=P2_WD)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    best_ci, best_ep = -1.0, 0
    ckpt_dir = save_dir / "ckpts"; ckpt_dir.mkdir(exist_ok=True)

    for ep in range(P2_EPOCHS):
        tl = _discrete_epoch(model, train_r, model_type, opt, device, cache, scaler, bins)
        if (ep + 1) % P2_EVAL == 0:
            ci = _discrete_eval(model, val_r, model_type, device, cache)
            if ci > best_ci:
                best_ci = ci; best_ep = ep + 1
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  ep {ep+1:3d}  loss={tl:.4f}  val_ci={ci:.4f}", flush=True)
            _gc()

    best_pt = ckpt_dir / "best.pt"
    if best_pt.exists():
        model.load_state_dict(torch.load(best_pt, map_location="cpu", weights_only=False))
        model.to(device)

    metrics = {}
    for sn, recs in [("train", train_r), ("val", val_r), ("test", test_r)]:
        ci = _discrete_eval(model, recs, model_type, device, cache)
        metrics[sn] = {"os_ci": ci, "n": len(recs)}
        print(f"  {sn:5s}  os_ci={ci:.4f}")

    _write_json(save_dir / "metrics.json", metrics)
    _write_json(status, {"completed": True, "best_ep": best_ep, "best_val": best_ci})
    del model, opt, scaler; _gc()
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# FOLD RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_fold(cancer: str, fold: int, methods: List[str],
             device, bag_cache, save_root: Path) -> dict:
    set_seeds(42)
    train_r, val_r, test_r = build_splits(cancer, fold)
    update_presence(train_r + val_r + test_r, bag_cache)

    cfg      = CANCER_CONFIGS[cancer]
    all_tasks = (["cls"] if cfg["cls_task"] else []) + ["os", "dss", "pfi"]
    results  = {}
    fold_dir = save_root / f"fold_{fold}"

    # ── Phase 1: per-modality ABMIL ──────────────────────────────────────────
    if "phase1" in methods:
        for mod in MODALITIES:
            p1_dir = fold_dir / "phase1" / mod
            run_phase1(mod, fold, device, bag_cache, train_r, val_r, test_r, p1_dir)
            m = _read_json(p1_dir / "metrics.json") or {}
            results[f"phase1_{mod.lower()}"] = m.get("test", m)

    # ── Phase 2: TCGASetTransformerMIL (set_mil_mt) ─────────────────────
    if "phase2" in methods:
        from train_tcga_multitask import run_phase2, load_p1_encoder
        p1_dir = fold_dir / "phase1"
        p2_dir = fold_dir / "phase2"
        # Ensure P1 encoders exist (needed for phase2's load_p1_encoder)
        for mod in MODALITIES:
            mp1 = p1_dir / mod
            if not _is_done(mp1 / "status.json"):
                run_phase1(mod, fold, device, bag_cache, train_r, val_r, test_r, mp1)
        m = run_phase2(cancer, fold, device, bag_cache, train_r, val_r, test_r,
                       p2_dir, p1_dir, all_tasks)
        results["phase2"] = m.get("test", m)

    # ── Early / Late / Middle per-task ────────────────────────────────────────
    for variant_cls, variant_name in [
        (EarlyFusion,  "early"),
        (LateFusion,   "late"),
        (MiddleFusion, "middle"),
    ]:
        if variant_name not in methods: continue
        for task in all_tasks:
            run_dir = fold_dir / variant_name / task
            m = run_simple(variant_cls, {}, task, fold, device, bag_cache,
                           train_r, val_r, test_r, run_dir)
            results[f"{variant_name}_{task}"] = m.get("test", m)

    # ── PORPOISE (OS only, Cox) ───────────────────────────────────────────────
    if "porpoise" in methods:
        m = run_simple(PORPOISE, {}, "os", fold, device, bag_cache,
                       train_r, val_r, test_r, fold_dir / "porpoise")
        results["porpoise"] = m.get("test", m)

    # ── MOTCat (OS only, Cox) ─────────────────────────────────────────────────
    if "motcat" in methods:
        m = run_simple(MOTCat, {}, "os", fold, device, bag_cache,
                       train_r, val_r, test_r, fold_dir / "motcat")
        results["motcat"] = m.get("test", m)

    # ── MCAT (OS only, discrete NLL) ─────────────────────────────────────────
    if "mcat" in methods:
        mcat_kw = dict(
            fusion="concat",
            omic_sizes=[N_PATHWAYS] * MCAT_GROUPS,  # 6 groups of 199
            n_classes=4,
            model_size_wsi="small", model_size_omic="small",
            dropout=0.25, wsi_input_dim=_feat_dim("WSI"))
        m = run_discrete_model(MCAT_Surv_cls, mcat_kw, "mcat", fold, device,
                               bag_cache, train_r, val_r, test_r, fold_dir / "mcat")
        results["mcat"] = m.get("test", m)

    # ── SurvPath (OS only, discrete NLL) ─────────────────────────────────────
    if "survpath" in methods:
        sp_kw = dict(
            omic_sizes=[N_PATHWAYS] * N_GENES,  # 331 gene-groups of size 199
            wsi_embedding_dim=_feat_dim("WSI"),
            dropout=0.1, num_classes=4, wsi_projection_dim=256)
        m = run_discrete_model(SurvPath_cls, sp_kw, "survpath", fold, device,
                               bag_cache, train_r, val_r, test_r, fold_dir / "survpath")
        results["survpath"] = m.get("test", m)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  GBMLGG {cancer}  fold={fold}  TEST RESULTS")
    print(f"{'─'*70}")
    for key, m in results.items():
        if isinstance(m, dict):
            kv = "  ".join(f"{k}={v:.4f}" for k, v in m.items()
                           if isinstance(v, float))
        else:
            kv = str(m)
        print(f"  {key:<35} {kv}")

    _write_json(save_root / f"fold_{fold}_summary.json", results)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="GBMLGG comprehensive benchmark")
    ap.add_argument("--cancer",    default="gbmlgg", choices=list(CANCER_CONFIGS.keys()))
    ap.add_argument("--fold",      type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--methods",   default="all",
                    help=f"Comma-sep from {METHODS_ALL} or 'all'")
    ap.add_argument("--save_root", default=None)
    args = ap.parse_args()

    if args.methods.strip().lower() == "all":
        methods = METHODS_ALL
    else:
        methods = [m.strip() for m in args.methods.split(",")]
        unknown = [m for m in methods if m not in METHODS_ALL]
        if unknown:
            ap.error(f"Unknown methods: {unknown}")

    save_root = Path(args.save_root) if args.save_root else GBMLGG_RESULTS / args.cancer
    save_root.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    splits_csv = SPLITS_DIR / f"{args.cancer}.csv"
    assert splits_csv.exists(), (
        f"Splits CSV not found: {splits_csv}\n"
        f"Run: sbatch data_prep/submit_make_splits.sh --cancer {args.cancer}")

    df       = pd.read_csv(splits_csv)
    load_rec = [{"key": str(r["key"]), "cancer": str(r["cancer"]),
                 "idx": int(r["idx"])} for _, r in df.iterrows()]

    print(f"\n{'#'*70}")
    print(f"#  GBMLGG Benchmark  |  cancer={args.cancer}  fold={args.fold}")
    print(f"#  Methods : {methods}")
    print(f"#  Device  : {DEVICE}")
    print(f"{'#'*70}\n")

    print(f"  Preloading {len(load_rec)} bags …")
    bag_cache = preload_bags(load_rec)
    _gc()

    run_fold(args.cancer, args.fold, methods, DEVICE, bag_cache, save_root)

    # Aggregate if all 5 folds done
    all_fold = {}
    for f in range(5):
        fp = save_root / f"fold_{f}_summary.json"
        if fp.exists():
            all_fold[f] = _read_json(fp) or {}
    if len(all_fold) == 5:
        print(f"\n{'='*70}")
        print(f"  AGGREGATE RESULTS (5-fold mean ± std)")
        print(f"{'='*70}")
        all_methods = set()
        all_metrics = set()
        for fs in all_fold.values():
            all_methods.update(fs.keys())
            for v in fs.values():
                if isinstance(v, dict):
                    all_metrics.update(v.keys())
        for meth in sorted(all_methods):
            for metric in sorted(all_metrics):
                vals = [fs[meth].get(metric) for fs in all_fold.values()
                        if isinstance(fs.get(meth), dict) and
                        isinstance(fs[meth].get(metric), float)]
                if vals:
                    print(f"  {meth:<25} {metric:<12} "
                          f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")
        _write_json(save_root / "aggregate_summary.json",
                    {"per_fold": {str(k): v for k, v in all_fold.items()}})


if __name__ == "__main__":
    main()
