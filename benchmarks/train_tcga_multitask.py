#!/usr/bin/env python3
"""
train_tcga_multitask.py
Multi-task MIL benchmark on TCGA: joint classification + survival.

Architecture  (TCGASetTransformerMIL — "set_mil" applied to TCGA)
──────────────────────────────────────────────────────────────────────
  Phase 1  Per-modality ABMIL backbone trained on OS (Cox PH loss).
           One model per (modality, fold).

  Phase 2  TCGASetTransformerMIL:
             • PMA (K=8 learned slot queries cross-attend to N patches)
               applied independently per modality
             • SAB (self-attention over M × K = 24 slots) — cross-modal fusion
             • Per-task gated ABMIL over 24 slots → scalar head
           Tasks:
             gbmlgg → cls (grade: GBM=1 vs LGG=0), OS, DSS, PFI
             blca / kirc / luad / brca → OS, DSS, PFI (no cls)
           Training: joint loss per epoch (Cox for survival + BCE for cls)

Modalities (TCGA)
─────────────────
  WSI  — WSI_patches  (N_patches × 1536)
  RNA  — RNA_pathways (331 × 199)
  CNV  — CNV_pathways (331,) → unsqueezed to (1 × 331)

Usage
─────
  # Step 1: build splits (via sbatch, once per cancer type)
  python3 data_prep/make_tcga_multitask_splits.py --cancer gbmlgg

  # Step 2: Phase 1 (per-modality backbone)
  python3 benchmarks/train_tcga_multitask.py --cancer gbmlgg --phase 1

  # Step 3: Phase 2 (multitask fusion)
  python3 benchmarks/train_tcga_multitask.py --cancer gbmlgg --phase 2

  # Or both in one job:
  python3 benchmarks/train_tcga_multitask.py --cancer gbmlgg
"""

import argparse
import gc
import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None
    _WANDB_AVAILABLE = False

try:
    import ctypes
    _libc = ctypes.CDLL("libc.so.6")
    def _malloc_trim(): _libc.malloc_trim(0)
except Exception:
    def _malloc_trim(): pass

def _gc():
    gc.collect(); _malloc_trim()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

LUSTRE    = "/lustre/groups/aih/dinesh.haridoss/mil"
SPLITS_DIR = Path("/home/aih/dinesh.haridoss/chicago_mil/data/tcga_splits")
SAVE_ROOT  = Path("/home/aih/dinesh.haridoss/chicago_mil/results_tcga_multitask")
FOLDS      = [0, 1, 2, 3, 4]
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED       = 42

CACHE_MAP = {
    "gbm":  f"{LUSTRE}/tcga_cache_gbm/samples",
    "lgg":  f"{LUSTRE}/tcga_cache_lgg/samples",
    "blca": f"{LUSTRE}/tcga_cache_blca/samples",
    "brca": f"{LUSTRE}/tcga_cache_brca/samples",
    "kirc": f"{LUSTRE}/tcga_cache_kirc/samples",
    "luad": f"{LUSTRE}/tcga_cache_luad/samples",
}

CANCER_CONFIGS = {
    "gbmlgg": {"cancers": ["gbm", "lgg"], "cls_task": True},
    "blca":   {"cancers": ["blca"],        "cls_task": False},
    "brca":   {"cancers": ["brca"],        "cls_task": False},
    "kirc":   {"cancers": ["kirc"],        "cls_task": False},
    "luad":   {"cancers": ["luad"],        "cls_task": False},
}

MODALITY_REGISTRY = {
    "WSI": ("WSI_patches",  1536),
    "RNA": ("RNA_pathways",  199),
    "CNV": ("CNV_pathways",  331),
}
MODALITIES = ["WSI", "RNA", "CNV"]

def _feat_key(mod): return MODALITY_REGISTRY[mod][0]
def _feat_dim(mod): return MODALITY_REGISTRY[mod][1]

SURV_ENDPOINTS = ["OS", "DSS", "PFI"]

HIDDEN_DIM     = 256
DROPOUT        = 0.3
PMA_K          = 8
P2_N_HEADS     = 4
P2_N_SAB       = 2
P2_MODAL_DROP  = 0.3
MAX_WSI        = 4096
MAX_RNA_CNV    = 512

# Phase 1
P1_LR     = 1e-4
P1_WD     = 1e-3
P1_EPOCHS = 300
P1_EVAL   = 25

# Phase 2
P2_LR         = 1e-4
P2_WD         = 1e-3
P2_EPOCHS     = 300
P2_EVAL       = 25
P2_GRAD_ACCUM = 8
COX_LAMBDA    = 1.0
BCE_LAMBDA    = 1.0

BagCache = Dict[str, Dict[str, Optional[torch.Tensor]]]


# ══════════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════════

def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _write_json(path: Path, obj: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f: json.dump(obj, f, indent=2)
    tmp.replace(path)

def _read_json(path: Path) -> Optional[dict]:
    if not path.exists(): return None
    try:
        with open(path) as f: return json.load(f)
    except Exception: return None

def _is_done(path: Path) -> bool:
    d = _read_json(path)
    return d is not None and d.get("completed", False)

def _find_resume(ckpt_dir: Path) -> int:
    if not ckpt_dir.exists(): return 0
    eps = []
    for cp in ckpt_dir.glob("ep*.pt"):
        try: eps.append(int(cp.stem[2:]))
        except ValueError: pass
    return max(eps) if eps else 0


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def preload_bags(records: list) -> BagCache:
    """Load .pt files; key = record["key"], path = CACHE_MAP[record["cancer"]]/{idx:05d}.pt"""
    cache: BagCache = {}
    n_total = len(records)
    count = {m: 0 for m in MODALITIES}

    for i, rec in enumerate(records):
        key      = rec["key"]
        cancer   = rec["cancer"]
        idx      = int(rec["idx"])
        pt_path  = Path(CACHE_MAP[cancer]) / f"{idx:05d}.pt"
        entry    = {m: None for m in MODALITIES}

        if pt_path.exists():
            try:
                d = torch.load(pt_path, map_location="cpu", weights_only=False)
                inp = d.get("inputs", {})
                for mod in MODALITIES:
                    t = inp.get(_feat_key(mod))
                    if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0:
                        continue
                    if t.dtype == torch.float16: t = t.float()
                    if t.dim() == 1: t = t.unsqueeze(0)  # CNV: (331,) → (1,331)
                    entry[mod] = t
                    count[mod] += 1
                del d, inp
            except Exception as e:
                print(f"  [warn] {pt_path.name}: {e}")

        cache[key] = entry

        if (i + 1) % 100 == 0:
            mb = sum(v.numel()*4/1e6 for e in cache.values() for v in e.values() if v is not None)
            print(f"    preload {i+1}/{n_total}  "
                  f"WSI={count['WSI']}  RNA={count['RNA']}  CNV={count['CNV']}  "
                  f"RAM={mb:.0f}MB", flush=True)
            _gc()

    mb = sum(v.numel()*4/1e6 for e in cache.values() for v in e.values() if v is not None)
    for mod in MODALITIES:
        print(f"  {mod:5s}: n={count[mod]}")
    print(f"  Total RAM: {mb:.0f} MB")
    return cache


def build_splits(cancer: str, fold: int) -> Tuple[list, list, list]:
    import pandas as pd
    csv_path = SPLITS_DIR / f"{cancer}.csv"
    assert csv_path.exists(), f"Splits CSV not found: {csv_path}\nRun: python3 data_prep/make_tcga_multitask_splits.py --cancer {cancer}"
    df = pd.read_csv(csv_path)
    col = f"fold_{fold}"
    assert col in df.columns, f"Column {col!r} not in {csv_path}"

    splits: Dict[str, list] = {"train": [], "val": [], "test": []}
    for _, row in df.iterrows():
        sp = str(row[col])
        if sp not in splits: continue
        rec = {
            "key":         str(row["key"]),
            "cancer":      str(row["cancer"]),
            "idx":         int(row["idx"]),
            "identifier":  str(row["identifier"]),
            "cls_label":   float(row["cls_label"]) if not _isnan(row["cls_label"]) else float("nan"),
            "os_status":   float(row["os_status"])   if not _isnan(row["os_status"])   else 0.0,
            "os_time":     float(row["os_time"])     if not _isnan(row["os_time"])     else float("nan"),
            "dss_status":  float(row["dss_status"])  if not _isnan(row["dss_status"])  else 0.0,
            "dss_time":    float(row["dss_time"])    if not _isnan(row["dss_time"])    else float("nan"),
            "pfi_status":  float(row["pfi_status"])  if not _isnan(row["pfi_status"])  else 0.0,
            "pfi_time":    float(row["pfi_time"])    if not _isnan(row["pfi_time"])    else float("nan"),
            "has_WSI": False, "has_RNA": False, "has_CNV": False,
        }
        splits[sp].append(rec)

    for sp, recs in splits.items():
        ev = sum(r["os_status"] == 1.0 for r in recs)
        cls_n = sum(not _isnan(r["cls_label"]) for r in recs)
        print(f"  [fold_{fold}] {sp:5s}  n={len(recs)}  OS_events={ev}  cls_labeled={cls_n}")
    return splits["train"], splits["val"], splits["test"]

def _isnan(v) -> bool:
    try: return v != v or str(v) in ("nan", "NaN", "")
    except Exception: return True

def update_presence(records: list, bag_cache: BagCache):
    for rec in records:
        entry = bag_cache.get(rec["key"], {})
        for mod in MODALITIES:
            rec[f"has_{mod}"] = entry.get(mod) is not None


# ══════════════════════════════════════════════════════════════════
# LOSS & METRICS
# ══════════════════════════════════════════════════════════════════

def cox_ph_loss(logits: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    # logcumsumexp_backward is not implemented for float16; cast to float32
    logits = logits.float(); times = times.float(); events = events.float()
    order        = torch.argsort(times, descending=True)
    h_sorted     = logits[order]
    e_sorted     = events[order]
    log_risk_set = torch.logcumsumexp(h_sorted, dim=0)
    log_lik      = (h_sorted - log_risk_set) * e_sorted
    n_events     = e_sorted.sum().clamp(min=1.0)
    return -log_lik.sum() / n_events


def c_index(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> float:
    concordant = 0.0; comparable = 0.0
    for i in range(len(times)):
        if events[i] == 0: continue
        for j in range(len(times)):
            if times[j] <= times[i]: continue
            comparable += 1.0
            if risks[i] > risks[j]:   concordant += 1.0
            elif risks[i] == risks[j]: concordant += 0.5
    return concordant / max(comparable, 1.0)


def balanced_accuracy(labels: np.ndarray, probs: np.ndarray) -> float:
    preds = (probs >= 0.5).astype(int)
    classes = np.unique(labels)
    if len(classes) < 2: return float("nan")
    recall_per_class = [
        np.mean(preds[labels == c] == c) for c in classes
    ]
    return float(np.mean(recall_per_class))


# ══════════════════════════════════════════════════════════════════
# MODELS — PHASE 1
# ══════════════════════════════════════════════════════════════════

class GatedAttnEncoder(nn.Module):
    """ABMIL encoder: (N, feat_dim) → rep (H,)."""
    def __init__(self, feat_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.att_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.att_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.att_w = nn.Linear(hidden_dim, 1, bias=False)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h     = self.backbone(x)
        gate  = self.att_V(h) * self.att_U(h)
        raw   = self.att_w(self.drop(gate))
        alpha = F.softmax(raw, dim=0)
        rep   = (alpha * h).sum(dim=0)
        return rep, alpha.squeeze(1), h


class SingleModalMIL(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.encoder = GatedAttnEncoder(feat_dim, hidden_dim, dropout)
        self.head    = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rep, _, _ = self.encoder(x)
        return self.head(rep).squeeze()


# ══════════════════════════════════════════════════════════════════
# MODELS — PHASE 2 (TCGASetTransformerMIL)
# ══════════════════════════════════════════════════════════════════

class _FFN(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Linear(dim, dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim*2, dim), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x + self.net(x))


class PMA(nn.Module):
    """Pooling by Multihead Attention.  K learned queries cross-attend to N patches."""
    def __init__(self, dim: int, K: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(K, dim) * 0.02)
        self.attn  = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm  = nn.LayerNorm(dim)
        self.ffn   = _FFN(dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q   = self.slots.unsqueeze(0)   # (1, K, dim)
        kv  = x.unsqueeze(0)            # (1, N, dim)
        out, _ = self.attn(q, kv, kv)
        return self.ffn(self.norm(out.squeeze(0)))  # (K, dim)


class SAB(nn.Module):
    """Self-Attention Block over M*K slots (cross-modal)."""
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn   = _FFN(dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = x.unsqueeze(0)
        a, _ = self.attn(xb, xb, xb)
        return self.ffn(self.norm1(x + a.squeeze(0)))


class TCGASetTransformerMIL(nn.Module):
    """
    Multi-task SetTransformerMIL for TCGA.

    Forward flow:
      1. Per-modality backbone (Linear→H) + PMA → K slots per modality
      2. Concat M modalities → M*K slots
      3. SAB (×n_sab) for cross-modal fusion
      4. Per-task gated ABMIL over M*K slots → scalar head
    """
    def __init__(
        self,
        encoders: Dict[str, GatedAttnEncoder],
        tasks: List[str],
        hidden_dim: int = 256,
        pma_k: int = 8,
        n_heads: int = 4,
        n_sab: int = 2,
        dropout: float = 0.1,
        modal_dropout: float = 0.3,
        max_wsi: int = 4096,
        max_other: int = 512,
    ):
        super().__init__()
        self.encoders      = nn.ModuleDict(encoders)
        self.tasks         = tasks
        self.modal_dropout = modal_dropout
        self.max_wsi       = max_wsi
        self.max_other     = max_other
        self.hidden_dim    = hidden_dim

        # PMA per modality
        self.pmas = nn.ModuleDict({m: PMA(hidden_dim, pma_k, n_heads, dropout)
                                   for m in encoders})
        # SAB stack
        self.sab  = nn.Sequential(*[SAB(hidden_dim, n_heads, dropout)
                                    for _ in range(n_sab)])
        # Per-task ABMIL + scalar head
        self.task_att_V = nn.ModuleDict({t: nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh()) for t in tasks})
        self.task_att_U = nn.ModuleDict({t: nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()) for t in tasks})
        self.task_att_w = nn.ModuleDict({t: nn.Linear(hidden_dim, 1, bias=False) for t in tasks})
        self.task_drop  = nn.Dropout(dropout)
        # Classification tasks output logit → BCE; survival tasks output risk → Cox
        self.task_heads = nn.ModuleDict({t: nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)) for t in tasks})

    def forward(self, bags: Dict[str, Optional[torch.Tensor]], device: torch.device) -> Dict[str, torch.Tensor]:
        slot_list = []
        for mod, enc in self.encoders.items():
            t = bags.get(mod)
            if t is None: continue
            if self.training and random.random() < self.modal_dropout: continue
            t = t.to(device, non_blocking=True)
            # Patch truncation
            max_p = self.max_wsi if mod == "WSI" else self.max_other
            if t.shape[0] > max_p:
                t = t[torch.randperm(t.shape[0], device=device)[:max_p]]
            h = enc.backbone(t)           # (N, H)
            slots = self.pmas[mod](h)     # (K, H)
            slot_list.append(slots)

        if not slot_list:
            return {t: torch.tensor(0.0, device=device, requires_grad=True)
                    for t in self.tasks}

        all_slots = torch.cat(slot_list, dim=0)   # (M*K, H)
        # SAB cross-modal fusion
        for sab_layer in self.sab:
            all_slots = sab_layer(all_slots)

        out = {}
        for task in self.tasks:
            gate = self.task_att_V[task](all_slots) * self.task_att_U[task](all_slots)
            alpha = F.softmax(self.task_att_w[task](self.task_drop(gate)), dim=0)
            rep   = (alpha * all_slots).sum(0)
            out[task] = self.task_heads[task](rep).squeeze()
        return out


# ══════════════════════════════════════════════════════════════════
# PHASE 1 TRAINING
# ══════════════════════════════════════════════════════════════════

def _p1_epoch(model, records, mod, optimizer, device, cache, scaler) -> float:
    model.train()
    random.shuffle(records)
    optimizer.zero_grad()

    logits, times, events = [], [], []
    for rec in records:
        if not rec.get(f"has_{mod}"): continue
        bag = cache.get(rec["key"], {}).get(mod)
        if bag is None: continue
        if _isnan(rec["os_time"]) or _isnan(rec["os_status"]): continue
        bag_d = bag.to(device, non_blocking=True)
        max_p = MAX_WSI if mod == "WSI" else MAX_RNA_CNV
        if bag_d.shape[0] > max_p:
            bag_d = bag_d[torch.randperm(bag_d.shape[0], device=device)[:max_p]]
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits.append(model(bag_d))
        times.append(rec["os_time"]); events.append(rec["os_status"])

    if len(logits) < 2: return float("nan")
    L = cox_ph_loss(torch.stack(logits),
                    torch.tensor(times,  dtype=torch.float32, device=device),
                    torch.tensor(events, dtype=torch.float32, device=device))
    if scaler:
        scaler.scale(L).backward(); scaler.step(optimizer); scaler.update()
    else:
        L.backward(); optimizer.step()
    del logits; _gc()
    return L.item()


@torch.no_grad()
def _p1_eval(model, records, mod, device, cache):
    model.eval()
    risks, times, events = [], [], []
    for rec in records:
        if not rec.get(f"has_{mod}"): continue
        bag = cache.get(rec["key"], {}).get(mod)
        if bag is None or _isnan(rec["os_time"]) or _isnan(rec["os_status"]): continue
        bag_d = bag.to(device)
        max_p = MAX_WSI if mod == "WSI" else MAX_RNA_CNV
        if bag_d.shape[0] > max_p:
            bag_d = bag_d[torch.randperm(bag_d.shape[0], device=device)[:max_p]]
        risks.append(model(bag_d).float().item())
        times.append(rec["os_time"]); events.append(rec["os_status"])
        del bag_d
    if len(risks) < 2: return np.array([]), np.array([]), np.array([]), 0.5
    r = np.array(risks); t = np.array(times); e = np.array(events)
    return r, t, e, c_index(t, e, r)


def run_phase1(mod: str, fold: int, device, cache, train_r, val_r, test_r, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    status_path = save_dir / "status.json"

    tr = [r for r in train_r if r.get(f"has_{mod}") and not _isnan(r["os_time"])]
    vl = [r for r in val_r   if r.get(f"has_{mod}") and not _isnan(r["os_time"])]
    te = [r for r in test_r  if r.get(f"has_{mod}") and not _isnan(r["os_time"])]
    print(f"  [P1:{mod}] fold={fold}  train={len(tr)}  val={len(vl)}  test={len(te)}")

    if len(tr) == 0:
        dummy = SingleModalMIL(_feat_dim(mod), HIDDEN_DIM, DROPOUT)
        torch.save(dummy.state_dict(), save_dir / "best_model.pt")
        _write_json(status_path, {"completed": True, "note": "dummy", "best_cindex": 0.5})
        return

    if _is_done(status_path):
        st = _read_json(status_path)
        print(f"  [P1:{mod}] already done  ci={st.get('best_cindex',0):.4f}")
        return

    ckpt_dir = save_dir / "ckpts"; ckpt_dir.mkdir(exist_ok=True)
    model     = SingleModalMIL(_feat_dim(mod), HIDDEN_DIM, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=P1_LR, weight_decay=P1_WD)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    resume = _find_resume(ckpt_dir)
    history = {"train_loss": [], "val_cindex": []}
    if 0 < resume < P1_EPOCHS:
        ckpt = torch.load(ckpt_dir / f"ep{resume:04d}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        if scaler and ckpt.get("scaler"): scaler.load_state_dict(ckpt["scaler"])
        if "history" in ckpt: history = ckpt["history"]
        print(f"  Resumed from ep {resume}")

    for ep in range(resume, P1_EPOCHS):
        tl = _p1_epoch(model, tr, mod, optimizer, device, cache, scaler)
        history["train_loss"].append(tl); _gc()

        if (ep + 1) % P1_EVAL == 0:
            _, _, _, val_ci = _p1_eval(model, vl, mod, device, cache)
            history["val_cindex"].append(val_ci)
            torch.save({
                "epoch": ep+1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "history": history,
            }, ckpt_dir / f"ep{ep+1:04d}.pt")
            print(f"  [P1:{mod}] ep {ep+1:3d}  loss={tl:.4f}  val_ci={val_ci:.4f}", flush=True)
            _gc()
        elif (ep + 1) % 50 == 0:
            print(f"  [P1:{mod}] ep {ep+1:3d}  loss={tl:.4f}")

    # Best checkpoint by val C-index
    ckpts = sorted(ckpt_dir.glob("ep*.pt"))
    if not ckpts:
        torch.save(model.state_dict(), save_dir / "best_model.pt")
        _write_json(status_path, {"completed": True, "best_cindex": 0.5})
        return

    print(f"  [P1:{mod}] rescanning {len(ckpts)} ckpts ...")
    best_ci, best_path = -1.0, ckpts[-1]
    for cp in ckpts:
        d = torch.load(cp, map_location="cpu", weights_only=False)
        model.load_state_dict(d.get("model", d)); model.to(device); del d
        _, _, _, ci = _p1_eval(model, vl, mod, device, cache)
        print(f"    ep {int(cp.stem[2:]):4d}  ci={ci:.4f}")
        if ci > best_ci: best_ci, best_path = ci, cp

    d = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(d.get("model", d)); model.to(device); del d
    torch.save(model.state_dict(), save_dir / "best_model.pt")
    _write_json(status_path, {"completed": True, "best_cindex": round(best_ci, 4)})

    metrics = {}
    for sn, recs in [("train", tr), ("val", vl), ("test", te)]:
        r, t, e, ci = _p1_eval(model, recs, mod, device, cache)
        metrics[sn] = {"cindex": ci, "n": len(r)}
        print(f"  [P1:{mod}] {sn:5s}  ci={ci:.4f}  n={len(r)}")
    _write_json(save_dir / "metrics.json", metrics)
    del model, optimizer, scaler; _gc()


def load_p1_encoder(p1_dir: Path, mod: str) -> GatedAttnEncoder:
    ckpt = p1_dir / mod / "best_model.pt"
    assert ckpt.exists(), f"Missing P1 ckpt: {ckpt}"
    base  = SingleModalMIL(_feat_dim(mod), HIDDEN_DIM, DROPOUT)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    base.load_state_dict(state, strict=False); del state
    return base.encoder


# ══════════════════════════════════════════════════════════════════
# PHASE 2 TRAINING
# ══════════════════════════════════════════════════════════════════

def _get_surv_records(records, endpoint):
    st_col = f"{endpoint.lower()}_status"
    ti_col = f"{endpoint.lower()}_time"
    return [(r, r[st_col], r[ti_col])
            for r in records
            if not _isnan(r.get(st_col, float("nan"))) and not _isnan(r.get(ti_col, float("nan")))]


def _p2_epoch(model, records, tasks, optimizer, device, cache, scaler, grad_accum=P2_GRAD_ACCUM) -> float:
    model.train()
    random.shuffle(records)
    optimizer.zero_grad()

    task_logits: Dict[str, List] = {t: [] for t in tasks}
    task_times:  Dict[str, List] = {t: [] for t in tasks}
    task_events: Dict[str, List] = {t: [] for t in tasks}
    task_cls_logits: List = []
    task_cls_labels: List = []

    for step, rec in enumerate(records):
        bags = {m: cache.get(rec["key"], {}).get(m) for m in MODALITIES}
        if all(v is None for v in bags.values()): continue

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            out = model(bags, device)

        for task in tasks:
            logit = out.get(task)
            if logit is None or not isinstance(logit, torch.Tensor): continue
            if logit.grad_fn is None: continue

            if task == "cls":
                if not _isnan(rec.get("cls_label", float("nan"))):
                    task_cls_logits.append(logit)
                    task_cls_labels.append(rec["cls_label"])
            else:
                ep   = task.split("_")[0].upper()  # OS → OS, DSS → DSS, PFI → PFI
                st   = f"{ep.lower()}_status"
                ti   = f"{ep.lower()}_time"
                if not _isnan(rec.get(st, float("nan"))) and not _isnan(rec.get(ti, float("nan"))):
                    task_logits[task].append(logit)
                    task_events[task].append(rec[st])
                    task_times[task].append(rec[ti])

        # Gradient accumulation — compute + step every grad_accum samples
        # (forward all then backward once at end for Cox correctness)

    # Compute total loss over all accumulated logits
    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0

    for task in tasks:
        if task == "cls":
            if len(task_cls_logits) >= 2:
                logits_t = torch.stack(task_cls_logits)
                labels_t = torch.tensor(task_cls_labels, dtype=torch.float32, device=device)
                n_pos = labels_t.sum().clamp(min=1.0)
                n_neg = (1.0 - labels_t).sum().clamp(min=1.0)
                pos_weight = (n_neg / n_pos).detach()
                bce = F.binary_cross_entropy_with_logits(logits_t, labels_t, pos_weight=pos_weight)
                total_loss = total_loss + BCE_LAMBDA * bce
                n_terms += 1
        else:
            if len(task_logits[task]) >= 2:
                logits_t = torch.stack(task_logits[task])
                times_t  = torch.tensor(task_times[task],  dtype=torch.float32, device=device)
                events_t = torch.tensor(task_events[task], dtype=torch.float32, device=device)
                try:
                    cox = cox_ph_loss(logits_t, times_t, events_t)
                    total_loss = total_loss + COX_LAMBDA * cox
                    n_terms += 1
                except Exception as e:
                    print(f"  [warn] cox loss {task}: {e}")

    if n_terms == 0 or total_loss.grad_fn is None:
        del task_logits, task_cls_logits; _gc()
        return float("nan")

    try:
        if scaler:
            scaler.scale(total_loss / n_terms).backward()
            scaler.step(optimizer); scaler.update()
        else:
            (total_loss / n_terms).backward()
            optimizer.step()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); optimizer.zero_grad(); _gc()
        print("  [OOM] skipping backward this epoch", flush=True)
        return float("nan")

    optimizer.zero_grad()
    loss_val = (total_loss / n_terms).item()
    del task_logits, task_cls_logits; _gc()
    return loss_val


@torch.no_grad()
def _p2_eval(model, records, tasks, device, cache) -> Dict[str, float]:
    model.eval()
    task_risks:  Dict[str, List] = {t: [] for t in tasks}
    task_times:  Dict[str, List] = {t: [] for t in tasks}
    task_events: Dict[str, List] = {t: [] for t in tasks}
    task_cls_probs:  List = []
    task_cls_labels: List = []

    for rec in records:
        bags = {m: cache.get(rec["key"], {}).get(m) for m in MODALITIES}
        if all(v is None for v in bags.values()): continue
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(bags, device)
        for task in tasks:
            logit = out.get(task)
            if logit is None: continue
            v = logit.float().item()
            if task == "cls":
                if not _isnan(rec.get("cls_label", float("nan"))):
                    task_cls_probs.append(torch.sigmoid(logit).float().item())
                    task_cls_labels.append(rec["cls_label"])
            else:
                ep = task.split("_")[0].upper()
                st = f"{ep.lower()}_status"; ti = f"{ep.lower()}_time"
                if not _isnan(rec.get(st, float("nan"))) and not _isnan(rec.get(ti, float("nan"))):
                    task_risks[task].append(v)
                    task_events[task].append(rec[st])
                    task_times[task].append(rec[ti])

    metrics: Dict[str, float] = {}
    for task in tasks:
        if task == "cls":
            if len(task_cls_probs) >= 2:
                metrics["cls_bacc"] = balanced_accuracy(
                    np.array(task_cls_labels), np.array(task_cls_probs))
        else:
            if len(task_risks[task]) >= 2:
                metrics[f"{task}_ci"] = c_index(
                    np.array(task_times[task]),
                    np.array(task_events[task]),
                    np.array(task_risks[task]))
    # Primary val metric: mean C-index (+ BACC if cls available)
    ci_vals = [v for k, v in metrics.items() if k.endswith("_ci")]
    bacc = metrics.get("cls_bacc", None)
    if ci_vals and bacc is not None:
        metrics["val_primary"] = 0.5 * np.mean(ci_vals) + 0.5 * bacc
    elif ci_vals:
        metrics["val_primary"] = float(np.mean(ci_vals))
    elif bacc is not None:
        metrics["val_primary"] = bacc
    else:
        metrics["val_primary"] = 0.5
    return metrics


def run_phase2(cancer: str, fold: int, device, cache, train_r, val_r, test_r,
               save_dir: Path, p1_dir: Path, tasks: List[str],
               wandb_project: str = ""):
    save_dir.mkdir(parents=True, exist_ok=True)
    status_path = save_dir / "status.json"

    _wb = None
    if wandb_project and _WANDB_AVAILABLE:
        try:
            _wb = _wandb.init(
                project=wandb_project,
                name=f"tcga_{cancer}_f{fold}",
                group=cancer,
                config={
                    "cancer": cancer, "fold": fold, "tasks": tasks,
                    "pma_k": PMA_K, "n_sab": P2_N_SAB,
                    "lr": P2_LR, "epochs": P2_EPOCHS,
                },
                reinit=True,
            )
        except Exception as _we:
            print(f"  [wandb] init failed: {_we}")
            _wb = None

    if _is_done(status_path):
        st = _read_json(status_path)
        print(f"  [P2:{cancer} fold={fold}] already done  primary={st.get('best_primary',0):.4f}")
        if _wb is not None:
            try: _wb.finish()
            except Exception: pass
        return _read_json(save_dir / "metrics.json") or {}

    print(f"\n  [P2:{cancer} fold={fold}] tasks={tasks}")

    # Load P1 encoders (frozen initially, can fine-tune all)
    try:
        encoders = {m: load_p1_encoder(p1_dir, m) for m in MODALITIES}
    except AssertionError as e:
        print(f"  [P2] ERROR: {e}")
        return {}

    model = TCGASetTransformerMIL(
        encoders=encoders,
        tasks=tasks,
        hidden_dim=HIDDEN_DIM,
        pma_k=PMA_K,
        n_heads=P2_N_HEADS,
        n_sab=P2_N_SAB,
        dropout=DROPOUT,
        modal_dropout=P2_MODAL_DROP,
        max_wsi=MAX_WSI,
        max_other=MAX_RNA_CNV,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=P2_LR, weight_decay=P2_WD)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=P2_EPOCHS, eta_min=1e-6)

    ckpt_dir = save_dir / "ckpts"; ckpt_dir.mkdir(exist_ok=True)
    history  = {"train_loss": [], "val_primary": []}

    resume = _find_resume(ckpt_dir)
    if 0 < resume < P2_EPOCHS:
        ckpt = torch.load(ckpt_dir / f"ep{resume:04d}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt))
        if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        if scaler and ckpt.get("scaler"): scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
        if "history" in ckpt: history = ckpt["history"]
        print(f"  Resumed from ep {resume}")

    best_primary, best_ep, best_path = -1.0, 0, None

    for ep in range(resume, P2_EPOCHS):
        tl = _p2_epoch(model, train_r, tasks, optimizer, device, cache, scaler)
        history["train_loss"].append(tl)
        scheduler.step()
        _gc()

        if (ep + 1) % P2_EVAL == 0:
            vm = _p2_eval(model, val_r, tasks, device, cache)
            prim = vm.get("val_primary", 0.5)
            history["val_primary"].append(prim)

            cp_path = ckpt_dir / f"ep{ep+1:04d}.pt"
            torch.save({
                "epoch": ep+1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "scheduler": scheduler.state_dict(),
                "history": history, "val_metrics": vm,
            }, cp_path)

            mstr = "  ".join(f"{k}={v:.4f}" for k,v in vm.items())
            print(f"  [P2] ep {ep+1:3d}  loss={tl:.4f}  {mstr}", flush=True)

            if _wb is not None:
                try:
                    _wb.log({"epoch": ep+1, "train/loss": tl,
                             **{f"val/{k}": v for k, v in vm.items()}})
                except Exception: pass

            if prim > best_primary:
                best_primary, best_ep, best_path = prim, ep+1, cp_path
            _gc()
        elif (ep + 1) % 50 == 0:
            print(f"  [P2] ep {ep+1:3d}  loss={tl:.4f}")

    if best_path is None:
        best_path = sorted(ckpt_dir.glob("ep*.pt"))[-1] if list(ckpt_dir.glob("ep*.pt")) else None

    if best_path is not None:
        d = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(d.get("model", d)); model.to(device); del d

    torch.save(model.state_dict(), save_dir / "best_model.pt")
    _write_json(status_path, {
        "completed": True, "best_epoch": best_ep,
        "best_primary": round(best_primary, 4)
    })

    all_metrics: dict = {}
    for sn, recs in [("train", train_r), ("val", val_r), ("test", test_r)]:
        m = _p2_eval(model, recs, tasks, device, cache)
        all_metrics[sn] = m
        mstr = "  ".join(f"{k}={v:.4f}" for k,v in m.items())
        print(f"  [P2] {sn:5s}  {mstr}")

    _write_json(save_dir / "metrics.json", all_metrics)

    if _wb is not None:
        try:
            _test = all_metrics.get("test", {})
            _wb.summary.update({f"final/test_{k}": v for k, v in _test.items()})
            _wb.summary["final/best_epoch"] = best_ep
            _wb.finish()
        except Exception: pass

    del model, optimizer, scaler; _gc()
    return all_metrics


# ══════════════════════════════════════════════════════════════════
# FOLD RUNNER
# ══════════════════════════════════════════════════════════════════

def run_fold(cancer: str, fold: int, phase: Optional[int], device, bag_cache: BagCache,
             save_root: Path, tasks: List[str], wandb_project: str = "") -> dict:
    set_seeds(SEED)
    fold_dir = save_root / f"fold_{fold}"

    train_r, val_r, test_r = build_splits(cancer, fold)
    update_presence(train_r + val_r + test_r, bag_cache)

    p1_dir = fold_dir / "phase1"
    p2_dir = fold_dir / "phase2"

    if phase in (1, None):
        for mod in MODALITIES:
            run_phase1(mod, fold, device, bag_cache, train_r, val_r, test_r,
                       p1_dir / mod)

    if phase in (2, None):
        return run_phase2(cancer, fold, device, bag_cache, train_r, val_r, test_r,
                          p2_dir, p1_dir, tasks, wandb_project=wandb_project)
    return {}


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="TCGA Multi-task SetTransformerMIL")
    p.add_argument("--cancer",  required=True, choices=list(CANCER_CONFIGS.keys()))
    p.add_argument("--phase",   type=int, default=None, help="1=P1 only  2=P2 only  omit=both")
    p.add_argument("--folds",   nargs="+", type=int, default=FOLDS)
    p.add_argument("--save_root", type=str, default=None)
    p.add_argument("--no_cls",  action="store_true", help="Skip classification task even if available")
    p.add_argument("--tasks",   nargs="+", default=None,
                   help="Restrict to these tasks, e.g. --tasks os  (default: all tasks)")
    p.add_argument("--wandb-project", default="", dest="wandb_project",
                   help="Weights & Biases project name for live metric tracking")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = CANCER_CONFIGS[args.cancer]

    tasks = []
    if cfg["cls_task"] and not args.no_cls:
        tasks.append("cls")
    for ep in SURV_ENDPOINTS:
        tasks.append(ep.lower())  # os, dss, pfi
    if args.tasks:
        tasks = [t for t in tasks if t in args.tasks]

    save_root = Path(args.save_root) if args.save_root else SAVE_ROOT / args.cancer
    save_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*65}")
    print(f"#  TCGA Multi-task MIL  —  {args.cancer.upper()}")
    print(f"#  Tasks   : {tasks}")
    print(f"#  Folds   : {args.folds}")
    print(f"#  Phase   : {args.phase or '1→2'}")
    print(f"#  Device  : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"#  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"#  VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"{'#'*65}\n")

    import pandas as pd
    splits_csv = SPLITS_DIR / f"{args.cancer}.csv"
    assert splits_csv.exists(), (
        f"Splits not found: {splits_csv}\n"
        f"Run first: sbatch data_prep/submit_make_splits.sh --cancer {args.cancer}")
    df = pd.read_csv(splits_csv)
    all_keys = df["key"].tolist()
    # Reconstruct cancer/idx from key for preload
    records_for_load = []
    for _, row in df.iterrows():
        records_for_load.append({
            "key": str(row["key"]),
            "cancer": str(row["cancer"]),
            "idx": int(row["idx"]),
        })

    print(f"  Total samples: {len(records_for_load)}")
    print(f"  Preloading bags ...")
    bag_cache = preload_bags(records_for_load)
    _gc()

    all_metrics: Dict[int, dict] = {}
    for fold in args.folds:
        print(f"\n{'='*65}  FOLD {fold}")
        all_metrics[fold] = run_fold(
            cancer=args.cancer, fold=fold,
            phase=args.phase, device=DEVICE,
            bag_cache=bag_cache, save_root=save_root,
            tasks=tasks,
            wandb_project=getattr(args, "wandb_project", ""))

    del bag_cache; _gc()

    # Summary table
    print(f"\n{'─'*65}")
    print(f"  FINAL TEST RESULTS — {args.cancer.upper()}")
    print(f"{'─'*65}")
    all_task_keys = set()
    for fm in all_metrics.values():
        test_m = fm.get("test", fm)
        all_task_keys.update(test_m.keys())
    all_task_keys.discard("val_primary")

    for tkey in sorted(all_task_keys):
        vals = []
        for fold in args.folds:
            fm = all_metrics.get(fold, {})
            test_m = fm.get("test", fm)
            v = test_m.get(tkey)
            if v is not None: vals.append(v)
        if vals:
            print(f"  {tkey:20s}  {np.mean(vals):.4f} ± {np.std(vals):.4f}  "
                  f"per-fold: {[round(v,4) for v in vals]}")

    summary = {
        "cancer": args.cancer, "tasks": tasks,
        "folds": {
            fold: all_metrics.get(fold, {}).get("test", {})
            for fold in args.folds
        }
    }
    _write_json(save_root / "summary.json", summary)
    print(f"\n  Results → {save_root}/summary.json")


if __name__ == "__main__":
    main()
