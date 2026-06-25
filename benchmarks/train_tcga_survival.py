#!/usr/bin/env python3
"""
train_tcga_survival.py  ·  Multimodal ABMIL — WSI + RNA + CNV (TCGA-GBM)
Overall Survival (Cox PH loss, Breslow approximation)

════════════════════════════════════════════════════════════════════
PHASE 1  Per-modality SingleModalMIL with Cox PH loss
────────────────────────────────────────────────────────────────────
  Modalities (all treated equally, no teacher-student):
    WSI  — WSI_patches  (N_patches × 1536)
    RNA  — RNA_pathways (331 × 199)
    CNV  — CNV_pathways (331,) → unsqueezed to (1 × 331)

  Full-batch Cox per epoch: all training samples forward in one pass,
  Cox partial likelihood (Breslow) computed over the full training set.
  Model selection: highest val C-index.

════════════════════════════════════════════════════════════════════
PHASE 2  Fusion variants  (--p2_variants)
────────────────────────────────────────────────────────────────────
  early / early_cls
  late
  middle / middle_cls
  crossattn / crossattn_cls
  crossmodal / crossmodal_cls
  iterative / iterative_cls
════════════════════════════════════════════════════════════════════
"""

import argparse
import gc
import json
import random
import warnings
from itertools import product as iproduct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_ckpt_utils

warnings.filterwarnings("ignore")

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
# STATUS / CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════

def _write_status(path: Path, completed: bool, **kw) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f: json.dump({"completed": completed, **kw}, f, indent=2)
    tmp.replace(path)

def _read_status(path: Path) -> Optional[dict]:
    if not path.exists(): return None
    try:
        with open(path) as f: return json.load(f)
    except Exception: return None

def _is_completed(save_dir: Path, tag: str = "status") -> bool:
    s = _read_status(save_dir / f"{tag}.json")
    return s is not None and s.get("completed", False)

def _find_resume_epoch(ckpt_dir: Path) -> int:
    if not ckpt_dir.exists(): return 0
    epochs = []
    for cp in ckpt_dir.glob("ep*.pt"):
        try: epochs.append(int(cp.stem[2:]))
        except ValueError: pass
    return max(epochs) if epochs else 0

def _load_checkpoint(ckpt_dir: Path, epoch: int) -> Optional[dict]:
    path = ckpt_dir / f"ep{epoch:04d}.pt"
    if not path.exists(): return None
    try: return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [warn] failed to load {path}: {e}"); return None


# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

SAMPLES_DIR = "/lustre/groups/aih/dinesh.haridoss/mil/tcga_cache_gbm/samples"
SPLITS_CSV  = "/home/aih/dinesh.haridoss/chicago_mil/tcga_gbm_splits.csv"
SAVE_DIR    = "/home/aih/dinesh.haridoss/chicago_mil/results_tcga_survival"

FOLDS      = [0, 1, 2, 3, 4]
PHASE      = None
PHASE1_DIR = None

# (feature_key_in_pt, feat_dim, presence_col)
MODALITY_REGISTRY: Dict[str, Tuple[str, int, str]] = {
    "WSI": ("WSI_patches",  1536, "has_WSI"),
    "RNA": ("RNA_pathways",  199, "has_RNA"),
    "CNV": ("CNV_pathways",  331, "has_CNV"),
}
MODALITIES = list(MODALITY_REGISTRY.keys())

def _feat_key(mod): return MODALITY_REGISTRY[mod][0]
def _feat_dim(mod): return MODALITY_REGISTRY[mod][1]
def _pres_col(mod): return MODALITY_REGISTRY[mod][2]

HIDDEN_DIM = 256
DROPOUT    = 0.4

# Phase 1
P1_LR           = 1e-4
P1_WEIGHT_DECAY = 1e-3
P1_EPOCHS       = 300
P1_EVAL_EVERY   = 25

# Phase 2
P2_LR             = 1e-4
P2_WEIGHT_DECAY   = 1e-3
P2_EPOCHS         = 200
P2_EVAL_EVERY     = 20
P2_MODAL_DROPOUT  = 0.3
P2_N_HEADS        = 4
P2_N_CROSS_LAYERS = 2
P2_ATTN_DROPOUT   = 0.1
P2_MAX_PATCHES    = 4096
P2_MAX_WSI_BLOCK  = 2048
P2_ITER_R         = 2
P2_SLOT_K         = 8

P2_VARIANTS = [
    "early", "early_cls",
    "late",
    "middle", "middle_cls",
    "crossattn", "crossattn_cls",
    "crossmodal", "crossmodal_cls",
    "iterative", "iterative_cls",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42

BagCache = Dict[str, Dict[str, Optional[torch.Tensor]]]


# ══════════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════════

def set_seeds(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def _variant_tag(variant: str, iter_r: int = 2, slot_k: int = 8) -> str:
    if "iterative" in variant:
        suffix = "_cls" if variant.endswith("_cls") else ""
        return f"iterative_r{iter_r}_k{slot_k}{suffix}"
    return variant


# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════

def build_splits(samples_dir, splits_csv, fold):
    import pandas as pd
    df = pd.read_csv(splits_csv)
    fold_col = f"fold_{fold}"
    assert fold_col in df.columns, f"Column {fold_col!r} not in {splits_csv}"

    splits_dict: Dict[str, list] = {"train": [], "val": [], "test": []}
    for _, row in df.iterrows():
        sp = str(row[fold_col])
        if sp not in splits_dict: continue
        idx = f"{int(row['idx']):05d}"
        rec = {
            "idx":      idx,
            "identifier": str(row["identifier"]),
            "os_time":  float(row["os_time"]),
            "os_event": float(row["os_status"]),
        }
        for mod in MODALITIES:
            rec[_pres_col(mod)] = True  # updated after preload
        splits_dict[sp].append(rec)

    for sp, recs in splits_dict.items():
        t = [r["os_time"]  for r in recs]
        e = [r["os_event"] for r in recs]
        print(f"  [fold_{fold}] {sp:5s}  n={len(recs)}  "
              f"events={int(sum(e))}  "
              f"os_time median={np.median(t):.1f}")
    return splits_dict["train"], splits_dict["val"], splits_dict["test"]


def preload_bags(idxs: List[str], samples_dir: str) -> BagCache:
    sd = Path(samples_dir)
    cache: BagCache = {}
    n_loaded      = {m: 0 for m in MODALITIES}
    total_patches = {m: 0 for m in MODALITIES}

    for i, idx in enumerate(sorted(idxs)):
        path  = sd / f"{idx}.pt"
        entry = {m: None for m in MODALITIES}
        if not path.exists():
            cache[idx] = entry; continue
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [warn] load failed {path.name}: {e}")
            cache[idx] = entry; continue

        inp = data.get("inputs", {})
        for mod in MODALITIES:
            t = inp.get(_feat_key(mod))
            if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0:
                continue
            if t.dtype == torch.float16: t = t.float()
            if t.dim() == 1: t = t.unsqueeze(0)   # CNV: (331,) → (1, 331)
            entry[mod] = t
            n_loaded[mod] += 1; total_patches[mod] += t.shape[0]
        del data, inp
        cache[idx] = entry

        if (i + 1) % 50 == 0:
            mb = sum(t.numel()*4/1e6 for e in cache.values()
                     for t in e.values() if t is not None)
            print(f"    preload {i+1}/{len(idxs)}  RAM={mb:.0f}MB", flush=True)
            _gc()

    mb = sum(t.numel()*4/1e6 for e in cache.values()
             for t in e.values() if t is not None)
    for mod in MODALITIES:
        avg = total_patches[mod] / max(n_loaded[mod], 1)
        print(f"  {mod:5s}: files={n_loaded[mod]}  patches={total_patches[mod]}  avg={avg:.0f}")
    print(f"  Total RAM: {mb:.0f} MB")
    return cache


def update_presence_from_cache(records: List[dict], bag_cache: BagCache):
    for rec in records:
        entry = bag_cache.get(rec["idx"], {})
        for mod in MODALITIES:
            rec[_pres_col(mod)] = entry.get(mod) is not None


# ══════════════════════════════════════════════════════════════════
# LOSS & METRICS
# ══════════════════════════════════════════════════════════════════

def cox_ph_loss(logits: torch.Tensor, times: torch.Tensor,
                events: torch.Tensor) -> torch.Tensor:
    """
    Breslow approximation of Cox negative partial log-likelihood.

    logits  : (N,)  risk scores (higher → higher hazard)
    times   : (N,)  observed times (months)
    events  : (N,)  event indicator (1 = event, 0 = censored)

    L = -1/N_events  ·  sum_i { event_i · [h_i - log(sum_{j: t_j >= t_i} exp(h_j))] }

    Sort by time descending → cumulative sum of exp gives the risk set sum at each step.
    """
    order          = torch.argsort(times, descending=True)
    h_sorted       = logits[order]
    e_sorted       = events[order]
    log_risk_set   = torch.logcumsumexp(h_sorted, dim=0)   # (N,)
    log_lik        = (h_sorted - log_risk_set) * e_sorted
    n_events       = e_sorted.sum().clamp(min=1.0)
    return -log_lik.sum() / n_events


def concordance_index(times: np.ndarray, events: np.ndarray,
                      risks: np.ndarray) -> float:
    """
    Harrell's C-index.  risks: higher → higher predicted hazard.
    Comparable pair (i, j): event_i=1 AND t_j > t_i.
    Concordant: risk_i > risk_j.
    """
    concordant = 0.0; comparable = 0.0
    n = len(times)
    for i in range(n):
        if events[i] == 0: continue
        for j in range(n):
            if times[j] <= times[i]: continue
            comparable += 1.0
            if risks[i] > risks[j]:   concordant += 1.0
            elif risks[i] == risks[j]: concordant += 0.5
    return concordant / max(comparable, 1.0)


def _plot_training_curves(history: dict, save_dir: Path, tag: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    n = len(history.get("train_loss", []))
    if n == 0: return
    def xax(k): step = max(n // max(k, 1), 1); return [step*(i+1)-1 for i in range(k)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(history["train_loss"], label="Train loss", color="steelblue", alpha=0.7)
    if history.get("val_loss"):
        ax.plot(xax(len(history["val_loss"])), history["val_loss"],
                "ro-", label="Val loss", markersize=4)
    ax.set_title(f"Cox loss — {tag}"); ax.legend(); ax.grid(True)
    fig.savefig(save_dir / f"loss_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if history.get("val_cindex"):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(xax(len(history["val_cindex"])), history["val_cindex"],
                "go-", markersize=4, label="Val C-index")
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.4)
        ax.set_title(f"C-index — {tag}"); ax.legend(); ax.grid(True)
        fig.savefig(save_dir / f"cindex_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_km_risk_groups(times, events, risks, save_dir: Path, tag: str):
    """Kaplan-Meier stratified by predicted risk (high vs low)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        return   # optional dependency

    median_risk = np.median(risks)
    high = risks >= median_risk
    fig, ax = plt.subplots(figsize=(8, 6))
    for mask, label, color in [(high, "High risk", "red"), (~high, "Low risk", "blue")]:
        km = KaplanMeierFitter()
        km.fit(times[mask], events[mask], label=label)
        km.plot_survival_function(ax=ax, ci_show=True, color=color)
    ax.set_title(f"KM — {tag}"); ax.grid(True, alpha=0.3)
    fig.savefig(save_dir / f"km_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# PHASE 1 MODELS
# ══════════════════════════════════════════════════════════════════

class GatedAttentionEncoder(nn.Module):
    """Gated attention MIL encoder.  Returns rep (H,), alpha (N,), h (N, H)."""
    def __init__(self, feat_dim: int, hidden_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone   = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.att_V    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.att_U    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.att_w    = nn.Linear(hidden_dim, 1, bias=False)
        self.att_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h     = self.backbone(x)
        gate  = self.att_V(h) * self.att_U(h)
        raw   = self.att_w(self.att_drop(gate))
        alpha = F.softmax(raw, dim=0)
        rep   = (alpha * h).sum(dim=0)
        return rep, alpha.squeeze(1), h


class SingleModalMIL(nn.Module):
    """Phase 1: single-modality Cox MIL model. Head outputs a risk score (scalar)."""
    def __init__(self, feat_dim: int, hidden_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.encoder = GatedAttentionEncoder(feat_dim, hidden_dim, dropout)
        self.head    = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rep, _, _ = self.encoder(x)
        return self.head(rep).squeeze()   # scalar risk score


# ══════════════════════════════════════════════════════════════════
# PHASE 1 TRAINING (full-batch Cox per epoch)
# ══════════════════════════════════════════════════════════════════

def p1_train_one_epoch(
    model: SingleModalMIL,
    records: List[dict],
    mod_name: str,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    bag_cache: BagCache,
    scaler: Optional[torch.amp.GradScaler],
) -> float:
    """
    Full-batch Cox per epoch:
      1. Forward all training samples → collect risk logits (with grad)
      2. Compute Cox PH loss over the full training set
      3. Single backward + optimizer step
    """
    model.train()
    random.shuffle(records)

    pres_col = _pres_col(mod_name)
    all_logits: List[torch.Tensor] = []
    all_times:  List[float]        = []
    all_events: List[float]        = []

    optimizer.zero_grad()

    for rec in records:
        if not rec.get(pres_col): continue
        bag = bag_cache.get(rec["idx"], {}).get(mod_name)
        if bag is None: continue
        bag_dev = bag.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logit = model(bag_dev)   # scalar with grad
        all_logits.append(logit)
        all_times.append(rec["os_time"])
        all_events.append(rec["os_event"])

    if len(all_logits) < 2:
        return float("nan")

    logits = torch.stack(all_logits)
    times  = torch.tensor(all_times,  dtype=torch.float32, device=device)
    events = torch.tensor(all_events, dtype=torch.float32, device=device)

    loss = cox_ph_loss(logits, times, events)

    if scaler is not None:
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    else:
        loss.backward(); optimizer.step()

    del all_logits; _gc()
    return loss.item()


@torch.no_grad()
def p1_evaluate(
    model: SingleModalMIL, records: List[dict], mod_name: str,
    device: torch.device, bag_cache: BagCache,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Returns risks, times, events, c_index."""
    model.eval()
    pres_col = _pres_col(mod_name)
    risks, times, events = [], [], []
    for rec in records:
        if not rec.get(pres_col): continue
        bag = bag_cache.get(rec["idx"], {}).get(mod_name)
        if bag is None: continue
        bag_dev = bag.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            risk = model(bag_dev).float().item()
        risks.append(risk); times.append(rec["os_time"]); events.append(rec["os_event"])
        del bag_dev

    risks  = np.array(risks);  times = np.array(times); events = np.array(events)
    ci = concordance_index(times, events, risks) if len(risks) > 1 else 0.5
    return risks, times, events, ci


# ══════════════════════════════════════════════════════════════════
# PHASE 1 RUNNER
# ══════════════════════════════════════════════════════════════════

def run_phase1_modality(
    mod_name: str, fold: int, device: torch.device,
    bag_cache: BagCache, train_recs: List[dict],
    val_recs: List[dict], test_recs: List[dict],
    save_dir: Path,
) -> Path:
    print(f"\n  {'─'*60}")
    print(f"  Phase 1 — {mod_name}  (fold {fold})")
    print(f"  {'─'*60}")

    pres_col = _pres_col(mod_name)
    tr = [r for r in train_recs if r.get(pres_col)]
    vl = [r for r in val_recs   if r.get(pres_col)]
    te = [r for r in test_recs  if r.get(pres_col)]
    print(f"  Present: train={len(tr)}  val={len(vl)}  test={len(te)}")

    save_dir.mkdir(parents=True, exist_ok=True)

    if len(tr) == 0:
        dummy = SingleModalMIL(_feat_dim(mod_name), HIDDEN_DIM, DROPOUT)
        torch.save(dummy.state_dict(), save_dir / "best_model.pt")
        _write_status(save_dir / "status.json", completed=True,
                      best_epoch=0, best_cindex=0.5, note="dummy")
        return save_dir / "best_model.pt"

    if _is_completed(save_dir):
        st = _read_status(save_dir / "status.json")
        print(f"  [{mod_name}] Already completed "
              f"(ep={st.get('best_epoch')}  "
              f"C-idx={st.get('best_cindex', 0):.4f}). Skipping.")
        return save_dir / "best_model.pt"

    ckpt_dir = save_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    model    = SingleModalMIL(_feat_dim(mod_name), HIDDEN_DIM, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=P1_LR,
                                  weight_decay=P1_WEIGHT_DECAY)
    scaler   = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}  feat_dim={_feat_dim(mod_name)}  AMP={scaler is not None}")

    history: Dict[str, List] = {k: [] for k in
                                 ["train_loss", "val_loss", "val_cindex"]}

    resume_epoch = _find_resume_epoch(ckpt_dir)
    start_epoch  = 0
    if 0 < resume_epoch < P1_EPOCHS:
        ckpt = _load_checkpoint(ckpt_dir, resume_epoch)
        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if scaler and ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            for k in history:
                if k in ckpt.get("history", {}): history[k] = ckpt["history"][k]
            start_epoch = resume_epoch
            print(f"  Resumed from epoch {resume_epoch}.")

    for epoch in range(start_epoch, P1_EPOCHS):
        tl = p1_train_one_epoch(model, tr, mod_name, optimizer, device, bag_cache, scaler)
        history["train_loss"].append(tl)
        _gc()

        if (epoch + 1) % P1_EVAL_EVERY == 0:
            _, vl_t, vl_e, val_ci = p1_evaluate(model, vl, mod_name, device, bag_cache)
            # Val Cox loss
            model.eval(); val_loss = 0.0
            with torch.no_grad():
                vlogs, vtimes, vevts = [], [], []
                for rec in vl:
                    if not rec.get(pres_col): continue
                    bag = bag_cache.get(rec["idx"], {}).get(mod_name)
                    if bag is None: continue
                    bd = bag.to(device)
                    vlogs.append(model(bd).float()); vtimes.append(rec["os_time"]); vevts.append(rec["os_event"])
                    del bd
                if len(vlogs) >= 2:
                    vl_ten = torch.stack(vlogs)
                    vt_ten = torch.tensor(vtimes, dtype=torch.float32, device=device)
                    ve_ten = torch.tensor(vevts,  dtype=torch.float32, device=device)
                    val_loss = cox_ph_loss(vl_ten, vt_ten, ve_ten).item()
            model.train()

            history["val_loss"].append(val_loss)
            history["val_cindex"].append(val_ci)

            torch.save({
                "epoch": epoch+1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "history": history,
            }, ckpt_dir / f"ep{epoch+1:04d}.pt")

            print(f"  [{mod_name}] ep {epoch+1:3d}  "
                  f"train_loss={tl:.4f}  val_loss={val_loss:.4f}  "
                  f"val_cindex={val_ci:.4f}  [ckpt]")
            _gc()
        elif (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  [{mod_name}] ep {epoch+1:3d}  train_loss={tl:.4f}")

    # Rescan → best val C-index
    ckpts = sorted(ckpt_dir.glob("ep*.pt"))
    if not ckpts:
        torch.save(model.state_dict(), save_dir / "best_model.pt")
        _write_status(save_dir / "status.json", completed=True,
                      best_epoch=0, best_cindex=0.5, last_epoch=P1_EPOCHS)
        return save_dir / "best_model.pt"

    print(f"\n  [{mod_name}] Rescanning {len(ckpts)} checkpoints ...")
    best_ci, best_ep, best_path = -1.0, 0, ckpts[-1]
    for cp in ckpts:
        ep   = int(cp.stem[2:])
        data = torch.load(cp, map_location="cpu", weights_only=False)
        state = data["model"] if isinstance(data, dict) else data
        model.load_state_dict(state); model.to(device); del data, state
        _, _, _, ci = p1_evaluate(model, vl, mod_name, device, bag_cache)
        print(f"    ep {ep:4d}  val_cindex={ci:.4f}")
        if ci > best_ci: best_ci, best_ep, best_path = ci, ep, cp

    print(f"  [{mod_name}] best ep={best_ep}  val_cindex={best_ci:.4f}")
    data  = torch.load(best_path, map_location="cpu", weights_only=False)
    state = data["model"] if isinstance(data, dict) else data
    model.load_state_dict(state); model.to(device); del data, state
    torch.save(model.state_dict(), save_dir / "best_model.pt")
    _write_status(save_dir / "status.json", completed=True,
                  best_epoch=best_ep, best_cindex=round(best_ci, 4),
                  last_epoch=P1_EPOCHS)

    metrics: dict = {}
    for sn, recs in [("train", tr), ("val", vl), ("test", te)]:
        r, t, e, ci = p1_evaluate(model, recs, mod_name, device, bag_cache)
        metrics[sn] = {"cindex": ci, "risks": r.tolist(), "times": t.tolist(),
                       "events": e.tolist()}
        print(f"  [{mod_name}] {sn:5s}  C-index={ci:.4f}  n={len(r)}")

    with open(save_dir / "metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    with open(save_dir / "history.json", "w") as f: json.dump(history, f)
    _plot_training_curves(history, save_dir / "plots", tag=mod_name)
    if metrics.get("test"):
        _plot_km_risk_groups(
            np.array(metrics["test"]["times"]),
            np.array(metrics["test"]["events"]),
            np.array(metrics["test"]["risks"]),
            save_dir / "plots", tag=mod_name)

    del model, optimizer, scaler; _gc()
    return save_dir / "best_model.pt"


# ══════════════════════════════════════════════════════════════════
# PHASE 2 — SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════

class FFN(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(),
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


class IterativeSlotAttn(nn.Module):
    """Iterative Slot Attention (Locatello et al., NeurIPS 2020)."""
    def __init__(self, hidden_dim: int, n_slots: int = 8,
                 n_iters: int = 3, dropout: float = 0.0):
        super().__init__()
        self.n_slots  = n_slots
        self.n_iters  = n_iters
        self.scale    = hidden_dim ** -0.5
        self.slot_mu        = nn.Parameter(torch.randn(1, n_slots, hidden_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, n_slots, hidden_dim))
        self.norm_slots = nn.LayerNorm(hidden_dim)
        self.norm_input = nn.LayerNorm(hidden_dim)
        self.proj_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gru    = nn.GRUCell(hidden_dim, hidden_dim)
        self.mlp    = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm_mlp = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(self.slot_log_sigma) + 1e-6
        slots = (self.slot_mu + sigma * torch.randn_like(self.slot_mu)).squeeze(0)
        h_norm = self.norm_input(h)
        k = self.proj_k(h_norm); v = self.proj_v(h_norm)
        for _ in range(self.n_iters):
            slots_prev = slots
            q = self.proj_q(self.norm_slots(slots))
            logits = torch.matmul(q, k.T) * self.scale
            attn   = F.softmax(logits, dim=0)
            attn_n = attn / (attn.sum(dim=1, keepdim=True) + 1e-6)
            updates = torch.matmul(attn_n, v)
            slots   = self.gru(updates.view(self.n_slots, -1),
                               slots_prev.view(self.n_slots, -1))
            slots   = slots + self.mlp(slots)
        return slots


class BidirPatchCrossAttn(nn.Module):
    """Bidirectional patch cross-attention (MCAT style)."""
    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads; self.d_k = hidden_dim // n_heads
        self.scale   = self.d_k ** -0.5
        for direction in ("a", "b"):
            other = "b" if direction == "a" else "a"
            setattr(self, f"Wq_{direction}", nn.Linear(hidden_dim, hidden_dim, bias=False))
            setattr(self, f"Wk_{other}",    nn.Linear(hidden_dim, hidden_dim, bias=False))
            setattr(self, f"Wv_{other}",    nn.Linear(hidden_dim, hidden_dim, bias=False))
            setattr(self, f"Wo_{direction}", nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.drop   = nn.Dropout(dropout)
        self.norm_a  = nn.LayerNorm(hidden_dim); self.norm_b  = nn.LayerNorm(hidden_dim)
        self.norm_a2 = nn.LayerNorm(hidden_dim); self.norm_b2 = nn.LayerNorm(hidden_dim)
        self.ffn_a   = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden_dim*2, hidden_dim), nn.Dropout(dropout))
        self.ffn_b   = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden_dim*2, hidden_dim), nn.Dropout(dropout))

    def _mha(self, Q, K, V):
        N_q, N_k = Q.shape[0], K.shape[0]
        h, d = self.n_heads, self.d_k
        q = Q.view(N_q, h, d).transpose(0,1); k = K.view(N_k, h, d).transpose(0,1); v = V.view(N_k, h, d).transpose(0,1)
        w = self.drop(F.softmax(torch.bmm(q, k.transpose(1,2)) * self.scale, dim=-1))
        return torch.bmm(w, v).transpose(0,1).contiguous().view(N_q, -1)

    def forward(self, h_a, h_b):
        out_a = self.Wo_a(self._mha(self.Wq_a(h_a), self.Wk_b(h_b), self.Wv_b(h_b)))
        h_a_r = h_a + out_a
        h_a_enr = self.norm_a2(h_a_r + self.ffn_a(self.norm_a(h_a_r)))
        out_b = self.Wo_b(self._mha(self.Wq_b(h_b), self.Wk_a(h_a), self.Wv_a(h_a)))
        h_b_r = h_b + out_b
        h_b_enr = self.norm_b2(h_b_r + self.ffn_b(self.norm_b(h_b_r)))
        return h_a_enr, h_b_enr


def _abmil_pool(h, att_V, att_U, att_w):
    gate  = att_V(h) * att_U(h)
    alpha = F.softmax(att_w(gate), dim=0)
    return (alpha * h).sum(0)

def _pool(use_cls, tokens, cls_token, cls_attn_mod, cls_norm,
          att_V, att_U, att_w, device):
    if use_cls:
        cls = cls_token.to(device)
        seq = torch.cat([cls, tokens], dim=0).unsqueeze(0)
        out, _ = cls_attn_mod(seq, seq, seq)
        return cls_norm(out.squeeze(0)[0])
    return _abmil_pool(tokens, att_V, att_U, att_w)

def _load_p1_encoder(p1_dir: Path, mod: str, trainable: bool = True) -> GatedAttentionEncoder:
    ckpt  = p1_dir / mod / "best_model.pt"
    assert ckpt.exists(), f"Missing Phase 1 checkpoint: {ckpt}"
    base  = SingleModalMIL(_feat_dim(mod), HIDDEN_DIM, DROPOUT)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    base.load_state_dict(state, strict=False); del state
    enc = base.encoder
    for p in enc.parameters(): p.requires_grad = trainable
    return enc


# ══════════════════════════════════════════════════════════════════
# PHASE 2 — FUSION MODELS
# ══════════════════════════════════════════════════════════════════

class EarlyFusionMIL(nn.Module):
    def __init__(self, encoders, hidden_dim=256, dropout=0.4,
                 modal_dropout=0.3, max_patches=P2_MAX_PATCHES, use_cls=False):
        super().__init__()
        self.encoders      = nn.ModuleDict(encoders)
        self.modal_dropout = modal_dropout
        self.max_patches   = max_patches
        self.use_cls       = use_cls
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, hidden_dim))
            self.cls_attn  = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
            self.cls_norm  = nn.LayerNorm(hidden_dim)
            self.att_V = self.att_U = self.att_w = None
        else:
            self.cls_token = self.cls_attn = self.cls_norm = None
            self.att_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
            self.att_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
            self.att_w = nn.Linear(hidden_dim, 1, bias=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        patches = []
        for mod, enc in self.encoders.items():
            t = bags.get(mod)
            if t is None: continue
            if self.training and random.random() < self.modal_dropout: continue
            t = t.to(device, non_blocking=True)
            if t.shape[0] > self.max_patches:
                t = t[torch.randperm(t.shape[0], device=device)[:self.max_patches]]
            patches.append(enc.backbone(t))
        if not patches:
            return torch.tensor(0.0, device=device, requires_grad=True)
        H_all = torch.cat(patches, dim=0)
        rep = _pool(self.use_cls, H_all, self.cls_token, self.cls_attn,
                    self.cls_norm, self.att_V, self.att_U, self.att_w, device)
        return self.head(rep).squeeze()


class LateFusionMIL(nn.Module):
    def __init__(self, encoders, hidden_dim=256, dropout=0.4, modal_dropout=0.3):
        super().__init__()
        self.encoders      = nn.ModuleDict(encoders)
        self.modal_dropout = modal_dropout
        self.heads = nn.ModuleDict({
            m: nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
            for m in encoders})
        self.log_weights = nn.Parameter(torch.zeros(len(encoders)))
        self.mod_index   = {m: i for i, m in enumerate(encoders)}

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        logits, indices = {}, []
        for mod, enc in self.encoders.items():
            t = bags.get(mod)
            if t is None: continue
            if self.training and random.random() < self.modal_dropout: continue
            rep, _, _ = enc(t.to(device, non_blocking=True))
            logits[mod] = self.heads[mod](rep).squeeze()
            indices.append(self.mod_index[mod])
        if not logits:
            return torch.tensor(0.0, device=device, requires_grad=True)
        if len(logits) == 1: return list(logits.values())[0]
        weights = F.softmax(self.log_weights[torch.tensor(indices, device=device)], dim=0)
        return (weights * torch.stack(list(logits.values()))).sum()


class MiddleFusionMIL(nn.Module):
    def __init__(self, encoders, hidden_dim=256, n_heads=4, n_layers=2,
                 dropout=0.1, modal_dropout=0.3, use_cls=False):
        super().__init__()
        self.encoders      = nn.ModuleDict(encoders)
        self.modal_dropout = modal_dropout
        self.use_cls       = use_cls
        self.transformer   = nn.ModuleList([nn.ModuleDict({
            "attn": nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True),
            "norm": nn.LayerNorm(hidden_dim), "ffn": FFN(hidden_dim, dropout),
        }) for _ in range(n_layers)])
        self.cls_token = nn.Parameter(torch.zeros(1, hidden_dim)) if use_cls else None
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
                                   nn.Linear(hidden_dim, 1))

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        tokens = []
        for mod, enc in self.encoders.items():
            t = bags.get(mod)
            if t is None: continue
            if self.training and random.random() < self.modal_dropout: continue
            rep, _, _ = enc(t.to(device, non_blocking=True))
            tokens.append(rep)
        if not tokens:
            return torch.tensor(0.0, device=device, requires_grad=True)
        if self.use_cls:
            tokens = [self.cls_token.squeeze(0).to(device)] + tokens
        x = torch.stack(tokens, dim=0).unsqueeze(0)
        for L in self.transformer:
            a, _ = L["attn"](x, x, x)
            x = L["ffn"](L["norm"](x + a))
        rep = x.squeeze(0)[0] if self.use_cls else x.squeeze(0).mean(0)
        return self.head(rep).squeeze()


class CrossAttnFusionMIL(nn.Module):
    def __init__(self, encoders, hidden_dim=256, n_heads=4, n_cross_layers=2,
                 dropout=0.1, modal_dropout=0.3, n_slots=8, n_slot_iters=3,
                 max_patches_bidir=256, use_cls=False):
        super().__init__()
        self.encoders          = nn.ModuleDict(encoders)
        self.modal_dropout     = modal_dropout
        self.max_patches_bidir = max_patches_bidir
        self.use_cls           = use_cls
        self.bidir_cross  = BidirPatchCrossAttn(hidden_dim, n_heads, dropout)
        self.slot_attn    = IterativeSlotAttn(hidden_dim, n_slots, n_slot_iters, dropout)
        self.cross_xfmr   = CrossModalTransformer(hidden_dim, n_heads, dropout, n_cross_layers)
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, hidden_dim))
            self.cls_attn  = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            self.cls_norm  = nn.LayerNorm(hidden_dim)
            self.att_V = self.att_U = self.att_w = None
        else:
            self.cls_token = self.cls_attn = self.cls_norm = None
            self.att_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
            self.att_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
            self.att_w = nn.Linear(hidden_dim, 1, bias=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
                                   nn.Linear(hidden_dim, 1))

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        present_h: Dict[str, torch.Tensor] = {}
        for mod, enc in self.encoders.items():
            t = bags.get(mod)
            if t is None: continue
            if self.training and random.random() < self.modal_dropout: continue
            present_h[mod] = enc.backbone(t.to(device, non_blocking=True))
        if not present_h:
            return torch.tensor(0.0, device=device, requires_grad=True)
        if len(present_h) >= 2:
            enriched = {m: h.clone() for m, h in present_h.items()}
            mods = list(present_h.keys())
            for i, m_a in enumerate(mods):
                for m_b in mods[i+1:]:
                    h_a = enriched[m_a]
                    h_b = enriched[m_b]
                    if h_b.shape[0] > self.max_patches_bidir:
                        h_b = h_b[torch.randperm(h_b.shape[0], device=device)[:self.max_patches_bidir]]
                    h_a_e, h_b_e = self.bidir_cross(h_a, h_b)
                    enriched[m_a] = h_a_e; enriched[m_b] = h_b_e
        else:
            enriched = present_h
        slot_list = [self.slot_attn(h) for h in enriched.values()]
        all_slots = self.cross_xfmr(torch.cat(slot_list, dim=0).unsqueeze(0)).squeeze(0)
        rep = _pool(self.use_cls, all_slots, self.cls_token, self.cls_attn,
                    self.cls_norm, self.att_V, self.att_U, self.att_w, device)
        return self.head(rep).squeeze()


class SlotCrossModalMIL(nn.Module):
    def __init__(self, encoders, hidden_dim=256, n_slots=8, n_slot_iters=3,
                 n_heads=4, n_cross_layers=2, dropout=0.1, modal_dropout=0.3, use_cls=False):
        super().__init__()
        self.encoders      = nn.ModuleDict(encoders)
        self.modal_dropout = modal_dropout
        self.use_cls       = use_cls
        self.slot_attn  = IterativeSlotAttn(hidden_dim, n_slots, n_slot_iters, dropout)
        self.cross_xfmr = CrossModalTransformer(hidden_dim, n_heads, dropout, n_cross_layers)
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, hidden_dim))
            self.cls_attn  = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            self.cls_norm  = nn.LayerNorm(hidden_dim)
            self.att_V = self.att_U = self.att_w = None
        else:
            self.cls_token = self.cls_attn = self.cls_norm = None
            self.att_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
            self.att_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
            self.att_w = nn.Linear(hidden_dim, 1, bias=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
                                   nn.Linear(hidden_dim, 1))

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        slot_list = []
        for mod, enc in self.encoders.items():
            t = bags.get(mod)
            if t is None: continue
            if self.training and random.random() < self.modal_dropout: continue
            slots = self.slot_attn(enc.backbone(t.to(device, non_blocking=True)))
            slot_list.append(slots)
        if not slot_list:
            return torch.tensor(0.0, device=device, requires_grad=True)
        all_slots = self.cross_xfmr(torch.cat(slot_list, dim=0).unsqueeze(0)).squeeze(0)
        rep = _pool(self.use_cls, all_slots, self.cls_token, self.cls_attn,
                    self.cls_norm, self.att_V, self.att_U, self.att_w, device)
        return self.head(rep).squeeze()


class IterativeXModalMIL(nn.Module):
    def __init__(self, encoders, hidden_dim=256, n_iter_blocks=2, n_slots=8,
                 n_slot_iters=3, n_heads=4, n_cross_layers=2, dropout=0.1,
                 modal_dropout=0.3, max_wsi_patches=P2_MAX_WSI_BLOCK,
                 use_cls=False, use_grad_ckpt=False):
        super().__init__()
        self.encoders        = nn.ModuleDict(encoders)
        self.modal_dropout   = modal_dropout
        self.n_iter_blocks   = n_iter_blocks
        self.max_wsi_patches = max_wsi_patches
        self.use_cls         = use_cls
        self.use_grad_ckpt   = use_grad_ckpt

        self.self_attn_blocks = nn.ModuleList([nn.ModuleDict({
            "attn": nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True),
            "norm": nn.LayerNorm(hidden_dim), "ffn": FFN(hidden_dim, dropout),
        }) for _ in range(n_iter_blocks)])
        self.cross_attn_blocks = nn.ModuleList([nn.ModuleDict({
            "attn": nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True),
            "norm": nn.LayerNorm(hidden_dim), "ffn": FFN(hidden_dim, dropout),
        }) for _ in range(n_iter_blocks)])
        self.slot_attn  = IterativeSlotAttn(hidden_dim, n_slots, n_slot_iters, dropout)
        self.cross_xfmr = CrossModalTransformer(hidden_dim, n_heads, dropout, n_cross_layers)

        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, hidden_dim))
            self.cls_attn  = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            self.cls_norm  = nn.LayerNorm(hidden_dim)
            self.att_V = self.att_U = self.att_w = None
        else:
            self.cls_token = self.cls_attn = self.cls_norm = None
            self.att_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
            self.att_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
            self.att_w = nn.Linear(hidden_dim, 1, bias=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
                                   nn.Linear(hidden_dim, 1))

    def _iter_block(self, r, h_dict):
        SA = self.self_attn_blocks[r]; CA = self.cross_attn_blocks[r]
        h_self = {}
        for mod, h in h_dict.items():
            x = h.unsqueeze(0); a, _ = SA["attn"](x, x, x)
            h_self[mod] = SA["ffn"](SA["norm"](x + a)).squeeze(0)
        if len(h_self) < 2: return h_self
        h_cross = {}
        for mod, h in h_self.items():
            others = torch.cat([v for k, v in h_self.items() if k != mod], dim=0).unsqueeze(0)
            q = h.unsqueeze(0); a, _ = CA["attn"](q, others, others)
            h_cross[mod] = CA["ffn"](CA["norm"](q + a)).squeeze(0)
        return h_cross

    def _checkpointed_block(self, r, h_dict):
        mods = list(h_dict.keys()); tensors = tuple(h_dict[m] for m in mods)
        def fn(*args):
            out = self._iter_block(r, {mods[i]: args[i] for i in range(len(mods))})
            return tuple(out[m] for m in mods)
        result = grad_ckpt_utils.checkpoint(fn, *tensors, use_reentrant=False)
        if not isinstance(result, tuple): result = (result,)
        return {mods[i]: result[i] for i in range(len(mods))}

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        h_dict: Dict[str, torch.Tensor] = {}
        for mod, enc in self.encoders.items():
            t = bags.get(mod)
            if t is None: continue
            if self.training and random.random() < self.modal_dropout: continue
            t = t.to(device, non_blocking=True)
            if mod == "WSI" and t.shape[0] > self.max_wsi_patches:
                t = t[torch.randperm(t.shape[0], device=device)[:self.max_wsi_patches]]
            h_dict[mod] = enc.backbone(t)
        if not h_dict:
            return torch.tensor(0.0, device=device, requires_grad=True)
        for r in range(self.n_iter_blocks):
            h_dict = (self._checkpointed_block(r, h_dict) if self.use_grad_ckpt
                      else self._iter_block(r, h_dict))
        slot_list = [self.slot_attn(h) for h in h_dict.values()]
        all_slots = self.cross_xfmr(torch.cat(slot_list, dim=0).unsqueeze(0)).squeeze(0)
        rep = _pool(self.use_cls, all_slots, self.cls_token, self.cls_attn,
                    self.cls_norm, self.att_V, self.att_U, self.att_w, device)
        return self.head(rep).squeeze()


# ══════════════════════════════════════════════════════════════════
# PHASE 2 — FACTORY, TRAINING, EVALUATION
# ══════════════════════════════════════════════════════════════════

def build_p2_model(variant: str, p1_dir: Path,
                   modal_dropout: float = P2_MODAL_DROPOUT,
                   iter_r: int = P2_ITER_R, slot_k: int = P2_SLOT_K,
                   n_slot_iters: int = 3, use_grad_ckpt: bool = False) -> nn.Module:
    encoders = {m: _load_p1_encoder(p1_dir, m, trainable=True) for m in MODALITIES}
    n_enc = sum(sum(p.numel() for p in e.parameters()) for e in encoders.values())
    print(f"  [p2:{variant}] encoders={n_enc:,}  slot_iters={n_slot_iters}")

    base    = variant.replace("_cls", "")
    use_cls = variant.endswith("_cls")

    if base == "early":
        return EarlyFusionMIL(encoders, HIDDEN_DIM, DROPOUT,
                               modal_dropout, P2_MAX_PATCHES, use_cls)
    elif base == "late":
        return LateFusionMIL(encoders, HIDDEN_DIM, DROPOUT, modal_dropout)
    elif base == "middle":
        return MiddleFusionMIL(encoders, HIDDEN_DIM, P2_N_HEADS,
                                P2_N_CROSS_LAYERS, P2_ATTN_DROPOUT, modal_dropout, use_cls)
    elif base == "crossattn":
        return CrossAttnFusionMIL(encoders, HIDDEN_DIM, P2_N_HEADS,
                                   P2_N_CROSS_LAYERS, P2_ATTN_DROPOUT,
                                   modal_dropout, slot_k, n_slot_iters, 256, use_cls)
    elif base == "crossmodal":
        return SlotCrossModalMIL(encoders, HIDDEN_DIM, slot_k, n_slot_iters,
                                  P2_N_HEADS, P2_N_CROSS_LAYERS,
                                  P2_ATTN_DROPOUT, modal_dropout, use_cls)
    elif base == "iterative":
        return IterativeXModalMIL(
            encoders=encoders, hidden_dim=HIDDEN_DIM,
            n_iter_blocks=iter_r, n_slots=slot_k, n_slot_iters=n_slot_iters,
            n_heads=P2_N_HEADS, n_cross_layers=P2_N_CROSS_LAYERS,
            dropout=P2_ATTN_DROPOUT, modal_dropout=modal_dropout,
            max_wsi_patches=P2_MAX_WSI_BLOCK,
            use_cls=use_cls, use_grad_ckpt=use_grad_ckpt)
    else:
        raise ValueError(f"Unknown variant: {variant!r}")


def p2_train_one_epoch(model, records, optimizer, device, bag_cache, scaler) -> float:
    """Full-batch Cox per epoch for Phase 2 fusion models."""
    model.train()
    random.shuffle(records)
    optimizer.zero_grad()

    all_logits: List[torch.Tensor] = []
    all_times:  List[float]        = []
    all_events: List[float]        = []

    for rec in records:
        bags = {m: bag_cache.get(rec["idx"], {}).get(m) for m in MODALITIES}
        if all(v is None for v in bags.values()): continue
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logit = model(bags, device)
        if not isinstance(logit, torch.Tensor) or logit.grad_fn is None: continue
        all_logits.append(logit)
        all_times.append(rec["os_time"]); all_events.append(rec["os_event"])

    if len(all_logits) < 2: return float("nan")

    logits = torch.stack(all_logits)
    times  = torch.tensor(all_times,  dtype=torch.float32, device=device)
    events = torch.tensor(all_events, dtype=torch.float32, device=device)
    loss   = cox_ph_loss(logits, times, events)

    if scaler:
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    else:
        loss.backward(); optimizer.step()

    del all_logits; _gc()
    return loss.item()


@torch.no_grad()
def p2_evaluate(model, records, device, bag_cache):
    model.eval()
    risks, times, events = [], [], []
    for rec in records:
        bags = {m: bag_cache.get(rec["idx"], {}).get(m) for m in MODALITIES}
        if all(v is None for v in bags.values()): continue
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            risk = model(bags, device).float().item()
        risks.append(risk); times.append(rec["os_time"]); events.append(rec["os_event"])
    risks = np.array(risks); times = np.array(times); events = np.array(events)
    ci = concordance_index(times, events, risks) if len(risks) > 1 else 0.5
    return risks, times, events, ci


# ══════════════════════════════════════════════════════════════════
# PHASE 2 — RUNNER
# ══════════════════════════════════════════════════════════════════

def run_phase2_variant(model: nn.Module, variant: str, fold: int,
                       device: torch.device, bag_cache: BagCache,
                       train_recs: List[dict], val_recs: List[dict],
                       test_recs: List[dict], save_dir: Path,
                       tag: Optional[str] = None) -> dict:
    vtag = tag or variant
    print(f"\n  {'='*60}")
    print(f"  Phase 2 [{vtag}]  (fold {fold})")
    print(f"  {'='*60}")
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = save_dir / f"ckpts_{vtag}"; ckpt_dir.mkdir(exist_ok=True)

    status_path = save_dir / f"status_{vtag}.json"
    if _is_completed(save_dir, tag=f"status_{vtag}"):
        st = _read_status(status_path)
        print(f"  [{vtag}] Already completed "
              f"(ep={st.get('best_epoch')}  C-idx={st.get('best_cindex', 0):.4f}). Skipping.")
        mf = save_dir / f"metrics_{vtag}.json"
        if mf.exists():
            with open(mf) as f: return json.load(f)
        return {}

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable={n_tr:,}")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=P2_LR, weight_decay=P2_WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = {k: [] for k in ["train_loss", "val_loss", "val_cindex"]}

    resume_epoch = _find_resume_epoch(ckpt_dir); start_epoch = 0
    if 0 < resume_epoch < P2_EPOCHS:
        ckpt = _load_checkpoint(ckpt_dir, resume_epoch)
        if ckpt is not None:
            model.load_state_dict(ckpt.get("model", ckpt))
            if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
            if scaler and ckpt.get("scaler"): scaler.load_state_dict(ckpt["scaler"])
            if "history" in ckpt: history = ckpt["history"]
            model.to(device); start_epoch = resume_epoch

    for epoch in range(start_epoch, P2_EPOCHS):
        tl = p2_train_one_epoch(model, train_recs, optimizer, device, bag_cache, scaler)
        history["train_loss"].append(tl); _gc()

        if (epoch + 1) % P2_EVAL_EVERY == 0:
            _, _, _, val_ci = p2_evaluate(model, val_recs, device, bag_cache)
            # Val Cox loss
            model.eval(); val_loss = 0.0
            with torch.no_grad():
                vlogs, vtimes, vevts = [], [], []
                for rec in val_recs:
                    bags = {m: bag_cache.get(rec["idx"], {}).get(m) for m in MODALITIES}
                    if all(v is None for v in bags.values()): continue
                    lo = model(bags, device).float()
                    vlogs.append(lo); vtimes.append(rec["os_time"]); vevts.append(rec["os_event"])
                if len(vlogs) >= 2:
                    vl_ten = torch.stack(vlogs)
                    vt_ten = torch.tensor(vtimes, dtype=torch.float32, device=device)
                    ve_ten = torch.tensor(vevts,  dtype=torch.float32, device=device)
                    val_loss = cox_ph_loss(vl_ten, vt_ten, ve_ten).item()
            model.train()

            history["val_loss"].append(val_loss)
            history["val_cindex"].append(val_ci)
            torch.save({
                "epoch": epoch+1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "history": history,
            }, ckpt_dir / f"ep{epoch+1:04d}.pt")
            print(f"  [{vtag}] ep {epoch+1:3d}  loss={tl:.4f}/{val_loss:.4f}  "
                  f"C-idx={val_ci:.4f}  [ckpt]")
            _gc()
        elif (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  [{vtag}] ep {epoch+1:3d}  train_loss={tl:.4f}")

    # Rescan → best val C-index
    ckpts = sorted(ckpt_dir.glob("ep*.pt"))
    if not ckpts: return {}

    print(f"\n  [{vtag}] Rescanning {len(ckpts)} checkpoints ...")
    best_ci, best_ep, best_path = -1.0, 0, ckpts[-1]
    for cp in ckpts:
        ep   = int(cp.stem[2:])
        data = torch.load(cp, map_location="cpu", weights_only=False)
        state = data.get("model", data); model.load_state_dict(state); model.to(device); del data, state
        _, _, _, ci = p2_evaluate(model, val_recs, device, bag_cache)
        print(f"    ep {ep:4d}  val_cindex={ci:.4f}")
        if ci > best_ci: best_ci, best_ep, best_path = ci, ep, cp

    print(f"  [{vtag}] best ep={best_ep}  val_cindex={best_ci:.4f}")
    data  = torch.load(best_path, map_location="cpu", weights_only=False)
    state = data.get("model", data); model.load_state_dict(state); model.to(device); del data, state
    torch.save(model.state_dict(), save_dir / f"model_{vtag}.pt")
    _write_status(status_path, completed=True, best_epoch=best_ep,
                  best_cindex=round(best_ci, 4), last_epoch=P2_EPOCHS)

    all_metrics: dict = {}
    for sn, recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        r, t, e, ci = p2_evaluate(model, recs, device, bag_cache)
        all_metrics[sn] = {"cindex": ci, "risks": r.tolist(),
                            "times": t.tolist(), "events": e.tolist()}
        print(f"  [{vtag}] {sn:5s}  C-index={ci:.4f}  n={len(r)}")

    with open(save_dir / f"metrics_{vtag}.json", "w") as f: json.dump(all_metrics, f, indent=2)
    with open(save_dir / f"history_{vtag}.json", "w") as f: json.dump(history, f)
    _plot_training_curves(history, save_dir / "plots", tag=vtag)
    if all_metrics.get("test"):
        _plot_km_risk_groups(
            np.array(all_metrics["test"]["times"]),
            np.array(all_metrics["test"]["events"]),
            np.array(all_metrics["test"]["risks"]),
            save_dir / "plots", tag=vtag)
    del model, optimizer, scaler; _gc()
    return all_metrics


# ══════════════════════════════════════════════════════════════════
# FOLD RUNNER
# ══════════════════════════════════════════════════════════════════

def run_fold(fold: int, phase=None, phase1_dir=None,
             bag_cache: BagCache = None,
             p2_variants: List[str] = None,
             p2_iter_r_list: List[int] = None,
             p2_slot_k_list: List[int] = None,
             p2_n_slot_iters: int = 3,
             p2_grad_ckpt: bool = False) -> dict:
    if p2_variants    is None: p2_variants    = ["iterative"]
    if p2_iter_r_list is None: p2_iter_r_list = [P2_ITER_R]
    if p2_slot_k_list is None: p2_slot_k_list = [P2_SLOT_K]

    set_seeds(SEED)
    fold_dir = Path(SAVE_DIR) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_recs, val_recs, test_recs = build_splits(SAMPLES_DIR, SPLITS_CSV, fold)
    for recs in (train_recs, val_recs, test_recs):
        update_presence_from_cache(recs, bag_cache)

    p1_dir = fold_dir / "phase1"
    p2_dir = fold_dir / "phase2"
    fold_metrics: dict = {}

    # ══ PHASE 1 ══════════════════════════════════════════════════
    if phase in (1, None):
        print(f"\n  ── Phase 1 ─────────────────────────────────────────────")
        for mod in MODALITIES:
            run_phase1_modality(
                mod_name=mod, fold=fold, device=DEVICE,
                bag_cache=bag_cache, train_recs=train_recs,
                val_recs=val_recs, test_recs=test_recs,
                save_dir=p1_dir / mod)

    # ══ PHASE 2 ══════════════════════════════════════════════════
    if phase in (2, None):
        eff_p1 = Path(phase1_dir) if phase1_dir else p1_dir

        for p2_variant in p2_variants:
            base_variant = p2_variant.replace("_cls", "")
            needs_grid   = base_variant in ("iterative", "crossmodal", "crossattn")

            if needs_grid:
                configs = list(iproduct(p2_iter_r_list, p2_slot_k_list))
                print(f"\n  ── P2 [{p2_variant}] grid: {len(configs)} configs ──")
                for r_val, k_val in configs:
                    vtag  = _variant_tag(p2_variant, r_val, k_val)
                    model = build_p2_model(
                        p2_variant, eff_p1, P2_MODAL_DROPOUT,
                        iter_r=r_val, slot_k=k_val,
                        n_slot_iters=p2_n_slot_iters,
                        use_grad_ckpt=p2_grad_ckpt).to(DEVICE)
                    fold_metrics[vtag] = run_phase2_variant(
                        model=model, variant=p2_variant, fold=fold,
                        device=DEVICE, bag_cache=bag_cache,
                        train_recs=train_recs, val_recs=val_recs,
                        test_recs=test_recs, save_dir=p2_dir, tag=vtag)
            else:
                vtag  = p2_variant
                model = build_p2_model(
                    p2_variant, eff_p1, P2_MODAL_DROPOUT,
                    slot_k=p2_slot_k_list[0],
                    n_slot_iters=p2_n_slot_iters).to(DEVICE)
                fold_metrics[vtag] = run_phase2_variant(
                    model=model, variant=p2_variant, fold=fold,
                    device=DEVICE, bag_cache=bag_cache,
                    train_recs=train_recs, val_recs=val_recs,
                    test_recs=test_recs, save_dir=p2_dir, tag=vtag)

    return fold_metrics


# ══════════════════════════════════════════════════════════════════
# ARGUMENT PARSING & MAIN
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="TCGA-GBM Multimodal ABMIL — Survival (Cox PH)")
    p.add_argument("--folds",       nargs="+", type=int, default=None)
    p.add_argument("--phase",       type=int,  default=None,
                   help="1=P1 only  2=P2 only  omit=both")
    p.add_argument("--phase1_dir",  type=str,  default=None)
    p.add_argument("--save_dir",    type=str,  default=SAVE_DIR)
    p.add_argument("--samples_dir", type=str,  default=SAMPLES_DIR)
    p.add_argument("--splits_csv",  type=str,  default=SPLITS_CSV)
    p.add_argument("--p2_variants", nargs="+", type=str,
                   default=["iterative"], choices=P2_VARIANTS)
    p.add_argument("--p2_iter_r",   nargs="+", type=int, default=[P2_ITER_R])
    p.add_argument("--p2_slot_iters", type=int, default=3)
    p.add_argument("--p2_slot_k",   nargs="+", type=int, default=[P2_SLOT_K])
    p.add_argument("--p2_grad_ckpt", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    global FOLDS, PHASE, PHASE1_DIR, SAVE_DIR, SAMPLES_DIR, SPLITS_CSV
    FOLDS       = args.folds if args.folds is not None else FOLDS
    PHASE       = args.phase
    PHASE1_DIR  = args.phase1_dir
    SAVE_DIR    = args.save_dir
    SAMPLES_DIR = args.samples_dir
    SPLITS_CSV  = args.splits_csv
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    print(f"PyTorch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    print(f"\n{'#'*65}")
    print(f"#  TCGA-GBM Multimodal ABMIL — Survival (Cox PH)")
    print(f"#  Modalities : {MODALITIES}")
    print(f"#  Samples dir: {SAMPLES_DIR}")
    print(f"#  Splits CSV : {SPLITS_CSV}")
    print(f"#  P2 variants: {args.p2_variants}")
    print(f"#  device={DEVICE}  folds={FOLDS}  phase={PHASE or '1→2'}")
    print(f"{'#'*65}\n")

    import pandas as pd
    df = pd.read_csv(SPLITS_CSV)
    all_idxs = [f"{int(row['idx']):05d}" for _, row in df.iterrows()]
    print(f"  Total samples: {len(all_idxs)}")
    print(f"  Preloading bags ...")
    bag_cache = preload_bags(all_idxs, SAMPLES_DIR)
    _gc()

    all_fold_metrics: Dict[int, dict] = {}
    for fold in FOLDS:
        print(f"\n{'='*65}  FOLD {fold}")
        all_fold_metrics[fold] = run_fold(
            fold=fold, phase=PHASE, phase1_dir=PHASE1_DIR,
            bag_cache=bag_cache,
            p2_variants=args.p2_variants,
            p2_iter_r_list=args.p2_iter_r,
            p2_slot_k_list=args.p2_slot_k,
            p2_grad_ckpt=args.p2_grad_ckpt,
            p2_n_slot_iters=args.p2_slot_iters)

    del bag_cache; _gc()

    with open(Path(SAVE_DIR) / "all_fold_metrics.json", "w") as f:
        json.dump(all_fold_metrics, f, indent=2)

    if PHASE not in (1,):
        all_tags = set()
        for fm in all_fold_metrics.values(): all_tags.update(fm.keys())

        print(f"\n{'─'*65}")
        print(f"  FINAL TEST RESULTS  (best val C-index per fold)")
        print(f"{'─'*65}")
        for vtag in sorted(all_tags):
            print(f"\n  [{vtag}]")
            print(f"  {'Fold':>4}  {'C-index':>8}  {'n_test':>6}")
            cidxs = []
            for fold in FOLDS:
                tm = all_fold_metrics.get(fold, {}).get(vtag, {}).get("test", {})
                if not tm: continue
                ci = tm.get("cindex", 0.0)
                cidxs.append(ci)
                print(f"  {fold:>4}  {ci:>8.4f}  {len(tm.get('risks', [])):>6}")
            if cidxs:
                print(f"  {'mean':>4}  {np.mean(cidxs):>8.4f}  "
                      f"{'±'+f'{np.std(cidxs):.4f}':>8}")

    print(f"\n  Done. Outputs: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
