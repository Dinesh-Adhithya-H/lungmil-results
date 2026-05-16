#!/usr/bin/env python3
"""
train_mm_abmil_v7.py  ·  Multimodal ABMIL — HE + BAL + CT + Clinical
Single-phase end-to-end multitask training.

Loss: L = L_hinge + λ_cox · L_cox
  L_hinge : hinge loss on ACR-labelled samples (A0*→0, A1*/A2*→1)
  L_cox   : Cox-Breslow loss using gap-time TTE (all samples, censored OK)

Dual-task pooling:
  Early / Late / Middle  →  DualGatedPool (two independent gated ABMIL per task)
  CrossAttn / CrossModal / Iterative  →  DualTaskHead (two CLS tokens on K slots)

Fusion variants (--p2_variants):
  early / early_cls
  late
  middle / middle_cls
  crossattn / crossattn_cls
  crossmodal / crossmodal_cls
  iterative / iterative_cls

Label derivation from acr_grade: A0* → 0,  A1*/A2* → 1,  else → None (excluded from hinge)
TTE: gap-time to next A1/A2 biopsy; censored at last biopsy if no future event.
"""

# Make the mil package importable when src/ lives next to this file.
import sys as _sys, pathlib as _pl
_src = _pl.Path(__file__).parent / "src"
if _src.exists() and str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

import argparse
import gc
import json
import math
import os
import random
import re
import warnings
from itertools import product as iproduct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_ckpt_utils
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_curve,
    roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

try:
    import ctypes
    _libc = ctypes.CDLL("libc.so.6")
    def _malloc_trim(): _libc.malloc_trim(0)
except Exception:
    def _malloc_trim(): pass

def _gc():
    # gc.collect() is unreliable on some HPC nodes — rely solely on CUDA cache
    # flush and optional malloc_trim for CPU fragmentation.
    _malloc_trim()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════
# STATUS FILE HELPERS
# ══════════════════════════════════════════════════════════════════

def _write_status(path: Path, completed: bool, **kwargs) -> None:
    data = {"completed": completed, **kwargs}
    tmp  = path.with_suffix(".tmp")
    with open(tmp, "w") as f: json.dump(data, f, indent=2)
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
        print(f"  [warn] failed to load {path}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

SAMPLES_DIR = "/lustre/groups/aih/dinesh.haridoss/mil/dataset_cache_latest_fixed_large/samples"
SPLITS_CSV  = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
SAVE_DIR    = "./results_mm_abmil_v6"

FOLDS      = [0, 1, 2, 3]
SPLIT      = None
PHASE      = None
PHASE1_DIR = None

MODALITY_REGISTRY: Dict[str, Tuple[str, int, str]] = {
    "HE":       ("HE_cells",  1024, "has_HE"),
    "BAL":      ("BAL_cells", 10,   "has_BAL"),
    "CT":       ("CT_cells",  1024, "has_CT"),
    # Clinical: new .pt format stores clinical_onehot (F, F*n_bins) = (F, F*4).
    # feat_dim is F*n_bins (e.g. 102*4=408). Falls back to raw (1, 107) if absent.
    "Clinical": ("clinical_onehot", 408, "has_Clinical"),
}
MODALITIES         = list(MODALITY_REGISTRY.keys())
TEACHER_MODALITIES = ["HE", "Clinical"]
STUDENT_MODALITIES = ["CT", "BAL"]

def _feat_key(mod): return MODALITY_REGISTRY[mod][0]
def _feat_dim(mod): return MODALITY_REGISTRY[mod][1]
def _pres_col(mod): return MODALITY_REGISTRY[mod][2]

HIDDEN_DIM = 256
DROPOUT    = 0.4

# Phase 1
P1_LR           = 1e-5
P1_WEIGHT_DECAY = 1e-3
P1_EPOCHS       = 400
P1_EVAL_EVERY   = 25
P1_GRAD_ACCUM   = 4
P1_CLR_PROJ_DIM = 128

# Phase 1 contrastive
P1_CLR_TAU    = 0.07
P1_CLR_LAMBDA = 0.1

# v7 multitask + careful CLR
LAMBDA_COX     = 1.0    # weight for Cox loss in combined objective
LAMBDA_CLR     = 0.1    # weight for CLR loss in combined objective
CLR_TAU_TEMP   = 0.15   # softer than SimCLR 0.07 — better for noisy medical labels
CLR_TAU_TIME   = 180.0  # exponential decay half-width for disease_time similarity (days)
N_CLR_WARMUP   = 5      # epochs with lambda_clr=0 before CLR is switched on
P1_SUMMARY_CLR_LAMBDA = 0.1
# Strategy 2: patch-augmentation NT-Xent (separate aug_proj_head)
P1_AUG_CLR_LAMBDA  = 0.05   # weight for aug NT-Xent loss
P1_AUG_SUBSAMPLE   = 0.70   # fraction of patches per augmented view
P1_AUG_MIN_PATCHES = 16     # min bag size to enable augmentation
# Strategy 3: label-supervised SupCon for survival (event as label)
P1_LABEL_SUPCON_LAMBDA = 0.05

# Phase 1 cross-attn / KD / CRD
P1_CROSS_ATTN_LAMBDA = 0.05
P1_KD_LAMBDA         = 0.05
P1_KD_TAU            = 0.5
P1_KD_TOP_K          = 50
P1_CRD_LAMBDA        = 0.1
P1_ATTN_N_HEADS      = 4
P1_ATTN_DROPOUT      = 0.1
P1_MAX_TEACH_PATCHES = 100   # top-K teacher patches by ABMIL attention weight kept for cross-attn

# Phase 2
P2_LR             = 5e-5
P2_WEIGHT_DECAY   = 1e-3
P2_EPOCHS         = 200
P2_EVAL_EVERY     = 20
P2_GRAD_ACCUM     = 4
P2_MODAL_DROPOUT  = 0.3
P2_N_HEADS        = 4
P2_N_CROSS_LAYERS = 2
P2_ATTN_DROPOUT   = 0.1
P2_MAX_PATCHES    = 2048
P2_MAX_HE_BLOCK   = 1024
P2_ITER_R         = 2
P2_SLOT_K         = 8

P2_VARIANTS = ["early", "late", "middle", "crossattn", "crossmodal", "iterative"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42

BagCache     = Dict[str, Dict[str, Optional[torch.Tensor]]]
TeacherCache = Dict[str, Dict[str, Optional[torch.Tensor]]]


# ══════════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════════

def set_seeds(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True

def acr_label(grade_str) -> Optional[int]:
    """A0* → 0, A1*/A2* → 1, anything else → None."""
    if grade_str is None: return None
    if isinstance(grade_str, float) and np.isnan(grade_str): return None
    g = str(grade_str).strip()
    if not g or g.lower() in ("nan", "none", "n/a", "na", "", "?"): return None
    if g.startswith("A0"): return 0
    if g.startswith("A1") or g.startswith("A2"): return 1
    return None


def compute_tte_next_acr(df) -> dict:
    """
    Gap-time approach for recurrent ACR events.
    For each sample (row), compute:
      tte_next_acr  : days from anchor_dt to next A1/A2 biopsy for same patient
      event_next_acr: 1 if that biopsy exists, 0 if censored

    ACR events = biopsies where acr_grade starts with A1 or A2.
    Censored time = days to last biopsy for that patient.
    Returns dict: stem -> (tte, event)
    """
    import pandas as _pd
    if not hasattr(df['anchor_dt'], 'dt'):
        df = df.copy()
        df['anchor_dt'] = _pd.to_datetime(df['anchor_dt'])

    # collect ACR event dates per patient
    acr_mask = df['acr_grade'].apply(
        lambda g: isinstance(g, str) and (g.startswith('A1') or g.startswith('A2'))
    )
    acr_dates: dict = {}
    for _, row in df[acr_mask].iterrows():
        acr_dates.setdefault(row['patient_id'], []).append(row['anchor_dt'])

    last_date: dict = df.groupby('patient_id')['anchor_dt'].max().to_dict()

    result: dict = {}
    for _, row in df.iterrows():
        stem = str(Path(str(row['file'])).stem)
        pid  = row['patient_id']
        t    = row['anchor_dt']
        is_acr_pos = (isinstance(row.get('acr_grade'), str) and
                      (row['acr_grade'].startswith('A1') or row['acr_grade'].startswith('A2')))
        future = sorted([d for d in acr_dates.get(pid, []) if d > t])
        if is_acr_pos:
            # This biopsy IS an ACR event — event at t=0
            tte, ev = 0, 1
        elif future:
            # ACR- biopsy with a future ACR event — use gap time to next event
            tte, ev = (future[0] - t).days, 1
        else:
            # ACR- with no future ACR → censored at last biopsy
            last = last_date.get(pid, t)
            tte, ev = max(int((last - t).days), 0), 0
        result[stem] = (float(tte), int(ev))
    return result

def _variant_tag(variant: str, iter_r: int = 2, slot_k: int = 8) -> str:
    base   = variant.replace("_cls", "")
    suffix = "_cls" if variant.endswith("_cls") else ""
    if base == "iterative":
        return f"iterative_r{iter_r}_k{slot_k}{suffix}"
    if base in ("crossattn", "crossmodal"):
        return f"{base}_k{slot_k}{suffix}"
    return variant


# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════

def build_splits(samples_dir, splits_csv, fold, split=None):
    import pandas as pd
    df = pd.read_csv(splits_csv)
    fold_col = f"split{split}_fold{fold}" if split is not None else f"fold_{fold}"
    assert fold_col in df.columns, f"Column {fold_col!r} not in {splits_csv}"

    splits_dict: Dict[str, list] = {"train": [], "val": [], "test": []}
    n_dropped = 0
    for _, row in df.iterrows():
        grade = row.get("acr_grade")
        if grade is None or (isinstance(grade, float) and np.isnan(grade)):
            n_dropped += 1; continue
        grade_str = str(grade).strip()
        if not grade_str or grade_str.lower() in ("nan","none","n/a","na","","?"):
            n_dropped += 1; continue
        if not re.search(r"A\d", grade_str, re.IGNORECASE):
            n_dropped += 1; continue
        label = 1 if re.search(r"A[12]", grade_str, re.IGNORECASE) else 0
        sp = str(row[fold_col])
        if sp not in splits_dict: n_dropped += 1; continue
        stem = Path(str(row["file"])).stem
        rec  = {"stem": stem, "label": label,
                "patient_id": str(row.get("patient_id", stem))}
        for mod in MODALITIES:
            rec[_pres_col(mod)] = bool(row.get(_pres_col(mod), False))
        for ep, sc, dc in [("clad", "clad_status", "clad_days"),
                           ("death", "death_status", "death_days")]:
            try:
                s = float(row.get(sc, float("nan")))
                d = float(row.get(dc, float("nan")))
                if not math.isnan(d) and not math.isnan(s):
                    rec[f"{ep}_time"]  = max(d, 0.0)
                    rec[f"{ep}_event"] = float(s)
                else:
                    rec[f"{ep}_time"]  = float("nan")
                    rec[f"{ep}_event"] = float("nan")
            except (TypeError, ValueError):
                rec[f"{ep}_time"]  = float("nan")
                rec[f"{ep}_event"] = float("nan")
        splits_dict[sp].append(rec)

    tag = f"split{split}_fold{fold}" if split is not None else f"fold_{fold}"
    print(f"  [{tag}] dropped={n_dropped}  "
          f"train={len(splits_dict['train'])}  val={len(splits_dict['val'])}  "
          f"test={len(splits_dict['test'])}")
    for sn in ("train","val","test"):
        _print_split_stats(tag, sn, splits_dict[sn])
    return splits_dict["train"], splits_dict["val"], splits_dict["test"]

def build_splits_survival(samples_dir, splits_csv, fold, split=None, endpoint='clad'):
    """Like build_splits but filters by survival endpoint instead of ACR grade.
    Records include label=int(surv_event) for compatibility with stats/contrastive code.

    CLAD endpoint logic:
      - Only pre-CLAD samples are kept (clad_days > 0 = days until CLAD event).
        Post-CLAD samples (clad_days <= 0) are discarded — the event has already
        occurred and we cannot predict "time to CLAD" from them.
      - Non-CLAD patients (clad_status=0) are censored observations.  Their
        clad_days is NaN, so we use death_days as a proxy censoring time (they
        were alive and CLAD-free at least until death or last follow-up).
        Samples with no usable proxy time are dropped.

    Death endpoint logic (unchanged):
      - All samples with a valid death_days are used (positive = days until death,
        censored patients have death_days=NaN and are dropped — known limitation).
    """
    import pandas as pd
    df = pd.read_csv(splits_csv, parse_dates=["anchor_dt"])
    fold_col = f"split{split}_fold{fold}" if split is not None else f"fold_{fold}"
    assert fold_col in df.columns, f"Column {fold_col!r} not in {splits_csv}"

    time_col  = f"{endpoint}_days"
    event_col = f"{endpoint}_status"

    # Study end date: latest sample date observed in the entire cohort.
    # Used as censoring time for truly alive patients (no event, no death).
    study_end = df["anchor_dt"].max()

    splits_dict: Dict[str, list] = {"train": [], "val": [], "test": []}
    n_dropped = 0
    for _, row in df.iterrows():
        sp = str(row.get(fold_col, ""))
        if sp not in splits_dict:
            n_dropped += 1; continue

        try:
            t = float(row.get(time_col,  float("nan")))
            e = float(row.get(event_col, float("nan")))
        except (TypeError, ValueError):
            n_dropped += 1; continue

        # ── ACR-specific handling ─────────────────────────────────────────────
        if endpoint == "acr":
            if math.isnan(e):
                n_dropped += 1; continue

            label_val = row.get("label", float("nan"))
            try:
                label_val = float(label_val)
            except (TypeError, ValueError):
                label_val = float("nan")

            if e == 0.0:
                # Never-ACR patient: only use A0-confirmed samples (label==0).
                # Unknown samples (label=NaN) are dropped — we cannot confirm
                # the patient was ACR-free at that visit.
                if math.isnan(label_val) or label_val != 0.0:
                    n_dropped += 1; continue
                try:
                    anchor = pd.Timestamp(row["anchor_dt"])
                    t = float((study_end - anchor).days)
                except Exception:
                    t = float("nan")
                if math.isnan(t) or t <= 0:
                    n_dropped += 1; continue
            else:
                # ACR patient: keep A0-confirmed pre-episode samples AND the
                # first ACR+ sample (acr_days=1, event=1).
                # Unknown pre-episode samples dropped (label=NaN, not confirmed ACR-free).
                # Post-episode samples (acr_days < 0) always dropped.
                if math.isnan(t) or t < 0:
                    n_dropped += 1; continue
                if t > 1 and (math.isnan(label_val) or label_val != 0.0):
                    # Pre-episode but not A0-confirmed → drop
                    n_dropped += 1; continue

        # ── CLAD-specific handling ────────────────────────────────────────────
        elif endpoint == "clad":
            if math.isnan(e):
                n_dropped += 1; continue

            if e == 0.0:
                # Censored patient: no CLAD event.
                # Censoring time priority:
                #   1. death_days — patient was CLAD-free until they died
                #   2. study_end − anchor_dt — still alive at last known study date
                try:
                    proxy_t = float(row.get("death_days", float("nan")))
                except (TypeError, ValueError):
                    proxy_t = float("nan")
                if math.isnan(proxy_t) or proxy_t <= 0:
                    # Alive-censored: use days from this sample to study end
                    try:
                        anchor = pd.Timestamp(row["anchor_dt"])
                        proxy_t = float((study_end - anchor).days)
                    except Exception:
                        proxy_t = float("nan")
                if math.isnan(proxy_t) or proxy_t <= 0:
                    n_dropped += 1; continue
                t = proxy_t
            else:
                # Event patient: keep only pre-CLAD samples (clad_days > 0).
                # clad_days = clad_event_date − anchor_dt; negative means the
                # biopsy was collected after CLAD onset — discard those.
                if math.isnan(t) or t <= 0:
                    n_dropped += 1; continue
        # ── Death (and any other endpoint) ────────────────────────────────────
        else:
            if math.isnan(e):
                n_dropped += 1; continue
            # Alive/censored patients typically have NaN death_days.
            # Use study_end − anchor_dt as proxy censoring time (same logic as CLAD).
            if e == 0.0 and (math.isnan(t) or t <= 0):
                try:
                    anchor = pd.Timestamp(row["anchor_dt"])
                    t = float((study_end - anchor).days)
                except Exception:
                    t = float("nan")
            if math.isnan(t) or t <= 0:
                n_dropped += 1; continue

        stem = Path(str(row["file"])).stem
        rec = {
            "stem":       stem,
            "label":      int(e),   # event flag as binary (0=censored, 1=event)
            "patient_id": str(row.get("patient_id", stem)),
        }
        for mod in MODALITIES:
            rec[_pres_col(mod)] = bool(row.get(_pres_col(mod), False))
        for ep, sc, dc in [("clad", "clad_status", "clad_days"),
                           ("death", "death_status", "death_days"),
                           ("acr",  "acr_status",  "acr_days")]:
            try:
                s = float(row.get(sc, float("nan")))
                d = float(row.get(dc, float("nan")))
                if not math.isnan(d) and not math.isnan(s) and d > 0:
                    rec[f"{ep}_time"]  = d
                    rec[f"{ep}_event"] = float(s)
                else:
                    rec[f"{ep}_time"]  = float("nan")
                    rec[f"{ep}_event"] = float("nan")
            except (TypeError, ValueError):
                rec[f"{ep}_time"]  = float("nan")
                rec[f"{ep}_event"] = float("nan")
        # For censored records, store the proxy time (study_end − anchor_dt)
        if e == 0.0:
            if endpoint == "clad":
                rec["clad_time"]  = t
                rec["clad_event"] = 0.0
            elif endpoint == "acr":
                rec["acr_time"]   = t
                rec["acr_event"]  = 0.0
            else:
                rec["death_time"]  = t
                rec["death_event"] = 0.0
        splits_dict[sp].append(rec)

    tag = f"split{split}_fold{fold}" if split is not None else f"fold_{fold}"
    print(f"  [{tag}] survival({endpoint}) dropped={n_dropped}  "
          f"train={len(splits_dict['train'])}  val={len(splits_dict['val'])}  "
          f"test={len(splits_dict['test'])}")
    for sn in ("train","val","test"):
        _print_split_stats(tag, sn, splits_dict[sn])
    return splits_dict["train"], splits_dict["val"], splits_dict["test"]


def build_splits_multitask(samples_dir, splits_csv, fold, split=None):
    """
    Build splits for v7 multitask objective.

    Label: derived from acr_grade (A0*→0, A1*/A2*→1, else None).
    Survival: gap-time approach — tte_next_acr = days to next A1/A2 biopsy;
              event_next_acr = 1 if event occurred, 0 if censored.
              ALL samples included for Cox (not just pre-episode).
    """
    import pandas as pd
    df = pd.read_csv(splits_csv)
    df['anchor_dt'] = pd.to_datetime(df['anchor_dt'])
    fold_col = f"split{split}_fold{fold}" if split is not None else f"fold_{fold}"
    assert fold_col in df.columns, f"Column {fold_col!r} not in {splits_csv}"

    # Precompute TTE for all rows at once
    tte_map = compute_tte_next_acr(df)

    splits_dict: Dict[str, list] = {"train": [], "val": [], "test": []}
    n_dropped = 0
    for _, row in df.iterrows():
        sp = str(row.get(fold_col, ""))
        if sp not in splits_dict:
            n_dropped += 1; continue

        stem = str(Path(str(row["file"])).stem)
        tte, ev = tte_map.get(stem, (float("nan"), 0))

        rec = {
            "stem":            stem,
            "patient_id":      str(row.get("patient_id", stem)),
            "label":           acr_label(row.get("acr_grade")),
            "tte_next_acr":    tte,
            "event_next_acr":  ev,
            # keep legacy acr_days for CLR time ordering (pre-episode only)
            "acr_days":        float(row["acr_days"]) if pd.notna(row.get("acr_days")) else float("nan"),
            "acr_status":      float(row["acr_status"]) if pd.notna(row.get("acr_status")) else float("nan"),
        }
        for mod in MODALITIES:
            rec[_pres_col(mod)] = bool(row.get(_pres_col(mod), False))

        # CLR disease times: use tte_next_acr where event occurred
        rec["disease_times_clr"] = [tte] if ev == 1 and not math.isnan(tte) else []

        # Other endpoints
        for ep, sc, dc in [("clad", "clad_status", "clad_days"),
                           ("death", "death_status", "death_days")]:
            try:
                s = float(row.get(sc, float("nan")))
                d = float(row.get(dc, float("nan")))
                rec[f"{ep}_time"]  = d if not math.isnan(d) and d > 0 else float("nan")
                rec[f"{ep}_event"] = float(s) if not math.isnan(s) else float("nan")
            except (TypeError, ValueError):
                rec[f"{ep}_time"] = float("nan"); rec[f"{ep}_event"] = float("nan")

        splits_dict[sp].append(rec)

    tag = f"split{split}_fold{fold}" if split is not None else f"fold_{fold}"
    print(f"  [{tag}] n_dropped={n_dropped}")
    for sn, recs in splits_dict.items():
        n_cls  = sum(1 for r in recs if r["label"] is not None)
        n_ev   = sum(1 for r in recs if r["event_next_acr"] == 1)
        n_cens = sum(1 for r in recs if r["event_next_acr"] == 0)
        print(f"  [{tag}] {sn:5s}  total={len(recs)}  cls={n_cls}  surv_event={n_ev}  surv_censored={n_cens}")
    return splits_dict


def update_presence_from_cache(records, bag_cache):
    for rec in records:
        entry = bag_cache.get(rec["stem"], {})
        for mod in MODALITIES:
            rec[_pres_col(mod)] = entry.get(mod) is not None
    return records

def _print_split_stats(fold_tag, split_name, recs):
    parts = []
    for mod in MODALITIES:
        pc = _pres_col(mod)
        n0 = sum(1 for r in recs if r.get(pc) and r["label"] == 0)
        n1 = sum(1 for r in recs if r.get(pc) and r["label"] == 1)
        parts.append(f"{mod}:neg={n0},pos={n1}")
    print(f"  [{fold_tag}] {split_name:5s}  total={len(recs)}  " + "  ".join(parts))

def _load_one_bag(args):
    """Load a single stem's .pt file — runs in a thread pool worker.

    New .pt format changes:
      - Clinical: use top-level `clinical_onehot` (F, F*n_bins) as bag.
        Each row = one clinical feature as an instance. Falls back to
        `inputs["Clinical"]` (1, 107) for old-format files.
      - Structured mods (HE/BAL/CT): also store cluster_count_onehot
        (K, K*n_bins) for use in count-stream models (v7).
    """
    stem, path = args
    entry = {m: None for m in MODALITIES}
    if not path.exists():
        return stem, entry
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [warn] load failed {path.name}: {e}", flush=True)
        return stem, entry
    inp = data.get("inputs", {})

    for mod in MODALITIES:
        if mod == "Clinical":
            # Prefer new clinical_onehot (F, F*n_bins) as bag representation
            coh = data.get("clinical_onehot")
            if coh is not None and isinstance(coh, torch.Tensor) and coh.numel() > 0:
                entry["Clinical"] = coh.float()
                continue
            # Fallback: old-format 1D raw clinical vector
            t = inp.get("Clinical")
            if t is not None and isinstance(t, torch.Tensor) and t.numel() > 0:
                if t.dtype == torch.float16: t = t.float()
                if t.dim() == 1: t = t.unsqueeze(0)
                entry["Clinical"] = t
        else:
            t = inp.get(_feat_key(mod))
            if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0: continue
            if t.dtype == torch.float16: t = t.float()
            if t.dim() == 1: t = t.unsqueeze(0)
            entry[mod] = t

    coords_t = inp.get("HE_coords")
    if coords_t is not None and isinstance(coords_t, torch.Tensor) and coords_t.numel() > 0:
        if coords_t.dtype == torch.float16: coords_t = coords_t.float()
        entry["HE_coords"] = coords_t

    # Store cluster_count_onehot for structured modalities (used in v7 count stream)
    raw_coh = data.get("cluster_count_onehot") or {}
    for mod, agg_key in [("HE", "HE_cells"), ("BAL", "BAL_cells"), ("CT", "CT_cells")]:
        coh = raw_coh.get(agg_key)
        if coh is not None and isinstance(coh, torch.Tensor) and coh.numel() > 0:
            entry[f"{mod}_count_onehot"] = coh.float()

    # Clinical feature names — needed for cross-attn interpretability
    cfn = data.get("clinical_feature_names")
    if cfn is not None:
        entry["_clinical_feature_names"] = cfn

    # Cluster names per modality — needed for cross-attn interpretability
    raw_cn = data.get("cluster_names") or {}
    for mod, agg_key in [("HE", "HE_cells"), ("BAL", "BAL_cells"), ("CT", "CT_cells")]:
        cn = raw_cn.get(agg_key)
        if cn is not None:
            entry[f"_{mod}_cluster_names"] = cn

    inp.clear()
    data.clear() if hasattr(data, "clear") else None
    return stem, entry


def preload_bags(stems, samples_dir, n_workers: int = 8):
    """Load all bag .pt files in parallel using a thread pool.
    I/O-bound: threads release the GIL during file reads.
    n_workers=8 is a safe default on HPC nodes with fast storage."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    sd = Path(samples_dir)
    stems_sorted = sorted(stems)
    args = [(s, sd / f"{s}.pt") for s in stems_sorted]

    cache: BagCache = {}
    n_loaded      = {m: 0 for m in MODALITIES}
    total_patches = {m: 0 for m in MODALITIES}

    print(f"  Preloading {len(stems_sorted)} bags with {n_workers} threads ...")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_load_one_bag, a): a[0] for a in args}
        done = 0
        for fut in as_completed(futs):
            stem, entry = fut.result()
            cache[stem] = entry
            for mod in MODALITIES:
                t = entry.get(mod)
                if t is not None:
                    n_loaded[mod] += 1; total_patches[mod] += t.shape[0]
            done += 1
            if done % 200 == 0:
                mb = sum(t.numel()*4/1e6 for e in cache.values()
                         for t in e.values()
                         if isinstance(t, torch.Tensor))
                print(f"    preload {done}/{len(stems_sorted)}  "
                      f"{'  '.join(f'{m}={n_loaded[m]}' for m in MODALITIES)}  "
                      f"RAM={mb:.0f}MB", flush=True)

    mb = sum(t.numel()*4/1e6 for e in cache.values()
             for t in e.values() if isinstance(t, torch.Tensor))
    for mod in MODALITIES:
        avg = total_patches[mod] / max(n_loaded[mod], 1)
        print(f"  {mod:10s}: files={n_loaded[mod]}  "
              f"patches={total_patches[mod]}  avg={avg:.0f}")
    print(f"  Total RAM: {mb:.0f} MB")
    return cache




# ══════════════════════════════════════════════════════════════════
# LOSS & METRICS
# ══════════════════════════════════════════════════════════════════

def hinge_loss(logit, target, cw):
    y = 2.0 * target - 1.0
    # Avoid creating new GPU scalars on every call — use float literals directly.
    w = torch.where(target > 0.5, logit.new_full((), cw[1]), logit.new_full((), cw[0]))
    return (w * torch.clamp(1.0 - y * logit, min=0.0)).mean()

def compute_class_weights(records):
    n_pos = sum(1 for r in records if r["label"] == 1)
    n_neg = sum(1 for r in records if r["label"] == 0)
    n_tot = max(n_pos + n_neg, 1)
    w_pos = min(n_tot / (2.0 * max(n_pos, 1)), 20.0)
    w_neg = min(n_tot / (2.0 * max(n_neg, 1)), 20.0)
    return (w_neg, w_pos)


def surv_con_loss(
    z:      torch.Tensor,   # (N, D) L2-normalised embeddings
    ttes:   torch.Tensor,   # (N,)  time-to-event (0 = event now)
    events: torch.Tensor,   # (N,)  δ: 1=event, 0=censored
    tau:      float = 0.1,
    tau_time: float = 90.0, # days — soft-positive bandwidth
) -> Optional[torch.Tensor]:
    """
    Soft temporal supervised contrastive loss for survival.

    Positive weight between event samples i and j:
        w_ij = exp(-|T_i - T_j| / tau_time)   if δ_i = δ_j = 1
               0                               otherwise

    Special cases:
      - Two ACR+ (T=0,T=0):  w=1.0   → strongly pulled together
      - ACR+ (T=0) vs soon (T=500): w=exp(-500/90)≈0  → near-zero pull
      - Censored (δ=0): excluded from anchors AND positives;
        they appear only in the denominator (potential negatives)

    Loss (soft-weighted SupCon):
        L = -1/|anchors| Σ_i  [1/W_i · Σ_j w_ij (sim_ij/τ - log_denom_i)]
    """
    N = z.shape[0]
    if N < 2:
        return None

    dev = z.device
    sim = z @ z.T / tau                               # (N, N)

    # Soft positive weights — event pairs only
    T     = ttes.float()
    ev    = events.float()
    dT    = (T.unsqueeze(1) - T.unsqueeze(0)).abs()   # (N, N)
    w_time = torch.exp(-dT / tau_time)                # (N, N)
    ev_mat = ev.unsqueeze(1) * ev.unsqueeze(0)        # 1 only if both δ=1
    W      = w_time * ev_mat
    W.fill_diagonal_(0.)                              # no self-similarity

    # Stable denominator: all pairs except self
    self_mask = torch.eye(N, dtype=torch.bool, device=dev)
    sim_max   = sim.detach().max(dim=1, keepdim=True).values
    exp_sim   = torch.exp(sim - sim_max) * (~self_mask)
    log_denom = torch.log(exp_sim.sum(dim=1) + 1e-8) + sim_max.squeeze(1)

    # Per-anchor loss (only event samples as anchors)
    anchor_mask = ev > 0.5
    if anchor_mask.sum() == 0:
        return None

    pos_weight_sum = W[anchor_mask].sum(dim=1)           # (A,)
    valid = pos_weight_sum > 1e-6
    if valid.sum() == 0:
        return None

    # Weighted log-softmax for each anchor
    log_num = sim[anchor_mask] - sim_max[anchor_mask]    # (A, N)
    per_pair = W[anchor_mask] * (log_num - log_denom[anchor_mask].unsqueeze(1))
    per_anch = -per_pair.sum(dim=1) / pos_weight_sum.clamp(min=1e-8)
    return per_anch[valid].mean()


def surv_rank_loss(
    hazards: torch.Tensor,  # (N,) risk scores (log-hazard)
    ttes:    torch.Tensor,  # (N,) time-to-event
    events:  torch.Tensor,  # (N,) δ
    max_pairs: int = 2048,  # subsample cap to avoid O(N²) blowup from tte=0 anchors
) -> Optional[torch.Tensor]:
    """
    Pairwise ranking loss (Luck et al. / DeepHit-style):
        L = -Σ_{i,j: T_i < T_j, δ_i=1} log σ(h_i - h_j)

    Patient i had the event earlier than j (regardless of whether j is censored)
    → model should predict higher risk for i.

    Censoring: only δ_i=1 required for the anchor; j can be censored or event.
    T_i < T_j ensures the observed event time for i precedes j's follow-up.

    tte=0 (ACR+ now): T_i=0 < T_j for all j with T_j>0 → large gradient signal.
    Capped at max_pairs (random subsample) to keep batch cost manageable.
    """
    N = len(hazards)
    if N < 2:
        return None

    T  = ttes.float()
    ev = events.float()

    # Valid pairs: T_i < T_j AND δ_i = 1
    Ti = T.unsqueeze(1)   # (N,1)
    Tj = T.unsqueeze(0)   # (1,N)
    di = ev.unsqueeze(1)
    mask = (Ti < Tj) & (di > 0.5)  # (N,N) bool

    idx = mask.nonzero(as_tuple=False)  # (P, 2)
    if idx.shape[0] == 0:
        return None

    # Subsample if too many pairs
    if idx.shape[0] > max_pairs:
        perm = torch.randperm(idx.shape[0], device=hazards.device)[:max_pairs]
        idx  = idx[perm]

    hi = hazards[idx[:, 0]]
    hj = hazards[idx[:, 1]]
    return -torch.log(torch.sigmoid(hi - hj) + 1e-8).mean()


def cox_breslow_loss(cox_buffer):
    """Breslow approximation of Cox partial negative log-likelihood.
    cox_buffer: list of (hazard_tensor, time_float, event_float)
    Returns scalar loss tensor or None if no events in buffer.
    """
    if not cox_buffer: return None
    hazards = torch.stack([h for h, t, e in cox_buffer])  # (N,) with grad
    dev     = hazards.device
    times   = torch.tensor([t for h, t, e in cox_buffer], dtype=torch.float32, device=dev)
    events  = torch.tensor([e for h, t, e in cox_buffer], dtype=torch.float32, device=dev)
    if events.sum() == 0: return None
    # Sort ascending by time
    order = torch.argsort(times)
    h_s = hazards[order]
    e_s = events[order]
    # Suffix log-sum-exp: log Σ_{j>=i} exp(h_j) for each i
    h_max = h_s.max().detach()
    exp_h = torch.exp(h_s - h_max)
    suffix_exp = torch.cumsum(exp_h.flip(0), dim=0).flip(0)
    log_risk = torch.log(suffix_exp + 1e-9) + h_max
    nll = (log_risk - h_s) * e_s
    return nll.sum() / e_s.sum().clamp(min=1)


def c_index(hazards, times, events):
    """Harrell's concordance index (higher=better, 0.5=random)."""
    n = len(hazards)
    concordant = discordant = 0
    for i in range(n):
        for j in range(n):
            if events[j] == 1 and times[j] < times[i]:
                if hazards[j] > hazards[i]: concordant += 1
                elif hazards[j] < hazards[i]: discordant += 1
    total = concordant + discordant
    return concordant / total if total > 0 else 0.5


def attention_transfer_loss(
    alpha_self:  torch.Tensor,   # (N_s,) student ABMIL weights — has grad
    alpha_cross: torch.Tensor,   # (N_s,) teacher-guided weights — detached
    tau: float = 0.5,
    top_k: int = 50,
) -> torch.Tensor:
    """
    Attention Transfer (Zagoruyko & Komodakis, ICLR 2017), adapted for MIL.

    Selects top-K instances by alpha_cross (teacher's view), then computes
    KL divergence between temperature-scaled softmax distributions.

    Scale-invariant: only relative ranking matters, not absolute weight values.
    Avoids uniform-attention collapse present in raw KL(alpha_self||alpha_cross).
    Gradient flows through alpha_self (student gate) only.
    """
    N = alpha_self.shape[0]
    k = min(top_k, N)
    top_idx    = alpha_cross.topk(k, dim=0).indices            # teacher's top-K
    p_target   = F.softmax(alpha_cross[top_idx] / tau, dim=0)  # target (detached)
    p_student  = F.softmax(alpha_self[top_idx]  / tau, dim=0)  # student (has grad)
    return F.kl_div(p_student.log(), p_target.detach(), reduction="batchmean")


def crd_loss_fn(
    r_cross:    torch.Tensor,   # (H,) cross-attended student rep — has grad
    r_teacher:  torch.Tensor,   # (H,) cached teacher summary — detached
) -> torch.Tensor:
    """
    CRD-style cosine distance (He et al., ICLR 2020).
    Pulls cross-attended student summary toward teacher summary.
    Both vectors are in raw representation space (H,).
    Loss = 1 - cosine_similarity ∈ [0, 2].
    """
    r_c = F.normalize(r_cross.float(),   dim=0)
    r_t = F.normalize(r_teacher.float(), dim=0).detach()
    return 1.0 - (r_c * r_t).sum()


def batch_supcon_loss(
    buffer: List[Tuple[torch.Tensor, int, str, str]],
    tau: float,
    cw: Tuple[float, float],
    min_multimodal_stems: int = 1,
) -> Optional[torch.Tensor]:
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    buffer: list of (L2-normalised vector, label, stem, mod)

    Positives: same stem + different modality  OR  same label + different stem.
    Negatives: different label.
    Only entries with requires_grad=True contribute loss terms (teacher entries
    are detached and act as passive positives/negatives).
    Class-weighted: minority (rejection) class upweighted.

    Vectorised: masks are built with integer IDs + tensor broadcasting;
    the per-anchor Python loop is replaced by masked tensor ops.
    """
    stem_counts: Dict[str, int] = {}
    for _, _, s, _ in buffer:
        stem_counts[s] = stem_counts.get(s, 0) + 1
    if sum(1 for c in stem_counts.values() if c >= 2) < min_multimodal_stems:
        return None
    B = len(buffer)
    if B < 2: return None

    zs     = torch.stack([b[0] for b in buffer])
    dev    = zs.device
    labels = torch.tensor([b[1] for b in buffer], dtype=torch.long, device=dev)
    stems  = [b[2] for b in buffer]
    mods   = [b[3] for b in buffer]

    # Build integer IDs for stems/mods (avoids O(B²) Python string comparisons)
    stem_vocab = {s: i for i, s in enumerate(dict.fromkeys(stems))}
    mod_vocab  = {m: i for i, m in enumerate(dict.fromkeys(mods))}
    stem_ids   = torch.tensor([stem_vocab[s] for s in stems], dtype=torch.long, device=dev)
    mod_ids    = torch.tensor([mod_vocab[m]  for m in mods],  dtype=torch.long, device=dev)

    self_mask  = torch.eye(B, dtype=torch.bool, device=dev)
    same_label = (labels.unsqueeze(0) == labels.unsqueeze(1))           # (B,B)
    same_stem  = (stem_ids.unsqueeze(0) == stem_ids.unsqueeze(1))       # (B,B)
    diff_mod   = (mod_ids.unsqueeze(0)  != mod_ids.unsqueeze(1))        # (B,B)
    pos_mask   = ((same_stem & diff_mod) | same_label) & ~self_mask     # (B,B)

    sims     = torch.matmul(zs, zs.T) / tau
    exp_sims = torch.exp(sims - sims.detach().max(dim=1, keepdim=True).values)
    exp_sims = exp_sims.masked_fill(self_mask, 0.0)

    # Vectorised per-anchor loss — no Python loop over B
    has_grad    = torch.tensor([b[0].requires_grad for b in buffer],
                                dtype=torch.bool, device=dev)
    has_pos     = pos_mask.any(dim=1)
    anchor_mask = has_grad & has_pos
    if not anchor_mask.any():
        return None

    cw_sum  = cw[0] + cw[1] + 1e-8
    weights = torch.where(labels == 1,
                          torch.full((B,), cw[1] / cw_sum, device=dev),
                          torch.full((B,), cw[0] / cw_sum, device=dev))

    denom     = exp_sims.sum(dim=1, keepdim=True) + 1e-8                # (B,1)
    log_probs = torch.log(exp_sims / denom + 1e-8)                      # (B,B)

    pos_f     = pos_mask.float()
    pos_count = pos_f.sum(dim=1).clamp(min=1)
    per_anc   = -(log_probs * pos_f).sum(dim=1) / pos_count             # (B,)

    return (per_anc * weights)[anchor_mask].mean()


def nt_xent_loss(
    z1: torch.Tensor,   # (N, D) L2-normalised
    z2: torch.Tensor,   # (N, D) L2-normalised
    tau: float,
) -> Optional[torch.Tensor]:
    """
    NT-Xent (SimCLR) for N paired augmented views.
    (z1[i], z2[i]) are positives; all other pairs within the 2N batch are negatives.
    Strategy 2: patch-subsampling augmentation — teaches encoder to be invariant
    to which specific patches are sampled from the same bag.
    """
    N = z1.shape[0]
    if N < 2:
        return None
    z = torch.cat([z1, z2], dim=0)                          # (2N, D)
    sim = torch.matmul(z, z.T) / tau                        # (2N, 2N)
    # Positive index for each row: row i → i+N, row i+N → i
    pos_idx = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N,     device=z.device),
    ])
    # Numerically stable: exclude self-similarities from denominator
    self_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim_max   = sim.detach().max(dim=1, keepdim=True).values
    exp_sim   = torch.exp(sim - sim_max) * (~self_mask).float()
    log_denom = torch.log(exp_sim.sum(dim=1) + 1e-8)
    pos_sim   = sim[torch.arange(2 * N, device=z.device), pos_idx]
    return -(pos_sim - sim_max.squeeze(1) - log_denom).mean()


def temporal_ordered_clr_loss(
    z: torch.Tensor,
    disease_times: torch.Tensor,
    tau_temp: float = CLR_TAU_TEMP,
    tau_time: float = CLR_TAU_TIME,
    uniform_floor: float = 0.01,
) -> torch.Tensor:
    """
    Temporal-ordered contrastive loss (v7 Rule 5: symmetric soft targets + uniform floor).

    z             : [B, D]  L2-normalised projected embeddings
    disease_times : [B]     signed days (positive = pre-event, negative = post-event)
    tau_temp      : temperature for similarity logits (Rule R3: 0.15 not 0.07)
    tau_time      : time-decay half-width in days
    uniform_floor : small uniform mass added to prevent zero targets (Rule R5)

    Gradient flows through z to backbone + proj_head.
    """
    if z.shape[0] < 2:
        return z.new_tensor(0.0)

    B = z.shape[0]
    sim = (z @ z.T) / tau_temp                                          # [B, B]
    dt  = (disease_times[:, None] - disease_times[None, :]).abs()       # [B, B]

    # Soft target: exponential decay + uniform floor to prevent collapse
    target = torch.exp(-dt / tau_time)                                   # [B, B]
    target = target + uniform_floor                                       # add floor
    target.fill_diagonal_(0.0)                                           # remove self
    row_sum = target.sum(dim=1, keepdim=True).clamp(min=1e-8)
    target  = target / row_sum                                           # normalise rows

    # Symmetric: average forward and backward cross-entropy directions
    loss_fwd = -(target * F.log_softmax(sim, dim=1)).sum(dim=1)         # [B]
    loss_bwd = -(target * F.log_softmax(sim, dim=0)).sum(dim=0)         # [B]
    loss = 0.5 * (loss_fwd + loss_bwd)

    return loss.mean()


def _mcc(labels, preds):
    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
        num = tp*tn - fp*fn
        den = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5
        return float(num/den) if den > 0 else 0.0
    except Exception: return 0.0

def compute_metrics(labels, probs, threshold=0.5):
    if np.any(np.isnan(labels)) or np.any(np.isnan(probs)):
        raise ValueError("NaN in compute_metrics")
    if len(np.unique(labels)) < 2:
        print("  [warn] single class in labels — neutral metrics returned")
        return dict(auc=0.5, auprc=0.0, bacc=0.5, mcc=0.0,
                    sens=0.0, spec=0.0, threshold=threshold)
    preds = (probs >= threshold).astype(int)
    m = dict(auc   = roc_auc_score(labels, probs),
             auprc = average_precision_score(labels, probs),
             bacc  = balanced_accuracy_score(labels, preds),
             mcc   = _mcc(labels, preds))
    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
        m["sens"] = tp / max(tp+fn, 1)
        m["spec"] = tn / max(tn+fp, 1)
    except Exception:
        m["sens"] = m["spec"] = 0.0
    m["threshold"] = threshold
    return m


def _plot_training_curves(history, save_dir, tag):
    save_dir.mkdir(parents=True, exist_ok=True)
    n = len(history.get("train_loss", []))
    if n == 0: return
    def xax(k): step = max(n // max(k,1), 1); return [step*(i+1)-1 for i in range(k)]
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(history["train_loss"], label="Train loss", color="steelblue", alpha=0.7)
    if history.get("val_loss"):
        ax.plot(xax(len(history["val_loss"])), history["val_loss"],
                "ro-", label="Val loss", markersize=4)
    ax.set_title(f"Loss — {tag}"); ax.legend(); ax.grid(True)
    fig.savefig(save_dir/f"loss_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    for key, ylabel in [("val_auc","AUC"),("val_bacc","BAcc"),("val_mcc","MCC")]:
        if not history.get(key): continue
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(xax(len(history[key])), history[key], "ro-", markersize=4, label=ylabel)
        ax.set_title(f"{ylabel} — {tag}"); ax.legend(); ax.grid(True)
        fig.savefig(save_dir/f"{key}_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    for key in ("cross_loss","kd_loss","crd_loss","clr_loss",
                "cls_loss","cox_loss","grad_norm"):
        if not history.get(key): continue
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(history[key], label=key, alpha=0.7)
        ax.set_title(f"{key} — {tag}"); ax.legend(); ax.grid(True)
        fig.savefig(save_dir/f"{key}_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

def _plot_roc_pr_confusion(metrics, save_dir, tag):
    save_dir.mkdir(parents=True, exist_ok=True)
    if "test" not in metrics: return
    probs  = np.array(metrics["test"]["probs"])
    labels = np.array(metrics["test"]["labels"])
    if len(np.unique(labels)) < 2: return
    fig, ax = plt.subplots(figsize=(7,6))
    fpr, tpr, _ = roc_curve(labels, probs)
    ax.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc_score(labels,probs):.3f}")
    ax.plot([0,1],[0,1],"k--"); ax.set_title(f"ROC — {tag}"); ax.legend(); ax.grid(True)
    fig.savefig(save_dir/f"roc_{tag}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,6))
    prec, rec, _ = precision_recall_curve(labels, probs)
    ax.plot(rec, prec, lw=2, label=f"AP={average_precision_score(labels,probs):.3f}")
    ax.set_title(f"PR — {tag}"); ax.legend(); ax.grid(True)
    fig.savefig(save_dir/f"pr_{tag}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0,1]).astype(float)
    cm /= (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["Pred 0","Pred 1"],
                yticklabels=["True 0","True 1"], ax=ax)
    ax.set_title(f"Confusion — {tag}")
    fig.savefig(save_dir/f"confusion_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def _plot_pooled_cv(all_fold_metrics, folds, save_dir, tag):
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        ap = np.concatenate([np.array(all_fold_metrics[f][tag]["test"]["probs"]) for f in folds])
        al = np.concatenate([np.array(all_fold_metrics[f][tag]["test"]["labels"]) for f in folds])
    except KeyError: return
    if len(np.unique(al)) < 2: return
    fig, ax = plt.subplots(figsize=(7,6))
    fpr, tpr, _ = roc_curve(al, ap)
    ax.plot(fpr, tpr, lw=2, label=f"Pooled AUC={roc_auc_score(al,ap):.3f}")
    for f in folds:
        try:
            fp2 = np.array(all_fold_metrics[f][tag]["test"]["probs"])
            fl2 = np.array(all_fold_metrics[f][tag]["test"]["labels"])
            if len(np.unique(fl2)) < 2: continue
            ff2, ft2, _ = roc_curve(fl2, fp2)
            ax.plot(ff2, ft2, lw=1, alpha=0.4,
                    label=f"F{f} ({roc_auc_score(fl2,fp2):.3f})")
        except Exception: pass
    ax.plot([0,1],[0,1],"k--")
    ax.set_title(f"ROC pooled — {tag}"); ax.legend(fontsize=8); ax.grid(True)
    fig.savefig(save_dir/f"roc_pooled_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{tag}] pooled AUC={roc_auc_score(al,ap):.3f}  "
          f"AP={average_precision_score(al,ap):.3f}")



# ══════════════════════════════════════════════════════════════════
# PHASE 1 MODELS
# ══════════════════════════════════════════════════════════════════

class GatedAttentionEncoder(nn.Module):
    """
    Gated attention MIL encoder.
    backbone: Linear(feat_dim → H) + ReLU + Dropout
    gate:     att_V (tanh) * att_U (sigmoid) → att_w → softmax → weighted sum
    Returns:  rep (H,), alpha (N,), h (N, H)
    """
    def __init__(self, feat_dim: int = 1024, hidden_dim: int = 256,
                 dropout: float = 0.4, use_spatial: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone   = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
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


# ══════════════════════════════════════════════════════════════════
# BIDIRECTIONAL PATCH CROSS-ATTENTION  
# ══════════════════════════════════════════════════════════════════

class BidirPatchCrossAttn(nn.Module):
    """
    Bidirectional patch-level cross-attention (MCAT / CONCH style).

    Implements explicit Q/K/V projections with manual scaled dot-product
    attention — avoids PyTorch 2.x flash-attention fast-path which silently
    drops Q-path gradients when K/V are detached (as in Phase 1 student
    training where teacher patches are always detached).

    Both directions use SEPARATE Q/K/V projection weights.
    No modality-specific parameters — same module used for all pairs.

    Forward computes:
      a → b:  out_a = softmax(Qa @ Kb.T / √d) @ Vb  (student queries teacher)
      b → a:  out_b = softmax(Qb @ Ka.T / √d) @ Va  (teacher queries student)

    Returns:
      h_a_enr   : (N_a, H)  student patches enriched with teacher context
      h_b_enr   : (N_b, H)  teacher patches enriched with student context
      alpha_a2b : (N_a,)    KD target — how much each student patch is
                             collectively attended by all teacher patches,
                             = mean_j( attn_b2a[j, i] )  ∈ [0,1], sums to 1

    Phase 1: caller passes h_b = h_teach.detach()
             grad flows through h_a (Q path a→b, KV path b→a)
    Phase 2: both h_a and h_b are trainable
    """
    def __init__(self, hidden_dim: int, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads    = n_heads
        self.d_k        = hidden_dim // n_heads
        self.hidden_dim = hidden_dim
        self.scale      = self.d_k ** -0.5

        # a → b direction
        self.Wq_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk_b = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv_b = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wo_a = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # b → a direction (separate weights)
        self.Wq_b = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wo_b = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.drop  = nn.Dropout(dropout)
        self.norm_a  = nn.LayerNorm(hidden_dim)
        self.norm_b  = nn.LayerNorm(hidden_dim)
        self.norm_a2 = nn.LayerNorm(hidden_dim)
        self.norm_b2 = nn.LayerNorm(hidden_dim)
        self.ffn_a   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout))
        self.ffn_b   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout))

    def _attn(self, Q: torch.Tensor, K: torch.Tensor,
              V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Manual multi-head scaled dot-product attention.
        Q: (N_q, H)  K: (N_k, H)  V: (N_k, H)
        Returns: out (N_q, H),  weights (N_q, N_k) averaged over heads.
        Gradient flows correctly through Q even when K/V are detached.
        """
        N_q, N_k = Q.shape[0], K.shape[0]
        h, d = self.n_heads, self.d_k

        q = Q.view(N_q, h, d).transpose(0, 1)   # (h, N_q, d)
        k = K.view(N_k, h, d).transpose(0, 1)   # (h, N_k, d)
        v = V.view(N_k, h, d).transpose(0, 1)   # (h, N_k, d)

        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (h, N_q, N_k)
        attn_w = F.softmax(scores, dim=-1)                      # (h, N_q, N_k)
        attn_w = self.drop(attn_w)

        out = torch.bmm(attn_w, v)                              # (h, N_q, d)
        out = out.transpose(0, 1).contiguous().view(N_q, -1)    # (N_q, H)
        weights = attn_w.mean(dim=0)                            # (N_q, N_k)
        return out, weights

    def forward(
        self,
        h_a: torch.Tensor,   # (N_a, H) — student patches (trainable)
        h_b: torch.Tensor,   # (N_b, H) — teacher patches (detached in Phase 1)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # ── a → b: student patches attend teacher patches ─────────
        Q_a = self.Wq_a(h_a)
        K_b = self.Wk_b(h_b)
        V_b = self.Wv_b(h_b)
        out_a2b, _ = self._attn(Q_a, K_b, V_b)
        out_a2b    = self.Wo_a(out_a2b)
        h_a_res    = h_a + out_a2b
        h_a_enr    = self.norm_a2(h_a_res + self.ffn_a(self.norm_a(h_a_res)))

        # ── b → a: teacher patches attend student patches ──────────
        Q_b = self.Wq_b(h_b)
        K_a = self.Wk_a(h_a)
        V_a = self.Wv_a(h_a)
        out_b2a, attn_b2a = self._attn(Q_b, K_a, V_a)  # attn_b2a: (N_b, N_a)
        out_b2a    = self.Wo_b(out_b2a)
        h_b_res    = h_b + out_b2a
        h_b_enr    = self.norm_b2(h_b_res + self.ffn_b(self.norm_b(h_b_res)))

        # ── KD signal: collective teacher attention per student patch
        # attn_b2a[j, i] = how much teacher patch j attends to student patch i
        # mean over j → per-student-patch weight (N_a,), sums to 1
        alpha_a2b = attn_b2a.mean(dim=0)   # (N_a,)

        return h_a_enr, h_b_enr, alpha_a2b


class IterativeSlotAttn(nn.Module):
    """
    Iterative Slot Attention with GRU update (Locatello et al., NeurIPS 2020).

    Slots compete for patches via softmax(dim=0) — each patch is assigned
    to exactly one slot in the soft sense — producing disentangled summaries.

    Contrast with standard cross-attention (softmax(dim=1)): there each slot
    distributes over all patches independently, slots can duplicate each other,
    and there is no competition. That collapses to weighted-sum pooling after
    one round.

    Algorithm per round t:
      1. Attention logits:  L = LayerNorm(slots_{t-1}) @ LayerNorm(h).T / √d_k
                            shape (K, N)
      2. Competition:       A = softmax(L, dim=0)   ← softmax over SLOTS
                            shape (K, N)  — each patch column sums to 1
      3. Normalise:         A_norm = A / (A.sum(dim=1, keepdim=True) + ε)
                            (each slot's share of the total pool)
      4. Aggregate:         updates = A_norm @ h     shape (K, H)
      5. GRU state:         slots_t = GRU(updates, slots_{t-1})
      6. Residual + LN:     slots_t = LayerNorm(slots_t + MLP(slots_t))

    Parameters shared across all T rounds (single set of weights).
    Slot init S_0: learned nn.Parameter (K, H), shared across modalities.
    No modality-specific parameters.

    Returns: slots (K, H)
    """
    def __init__(self, hidden_dim: int, n_slots: int = 8,
                 n_iters: int = 3, dropout: float = 0.0):
        super().__init__()
        self.n_slots  = n_slots
        self.n_iters  = n_iters
        self.scale    = hidden_dim ** -0.5

        # Learned slot initialisation — shared across all modalities
        self.slot_mu  = nn.Parameter(torch.randn(1, n_slots, hidden_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, n_slots, hidden_dim))

        # Separate LayerNorms for slots and inputs (original paper design)
        self.norm_slots = nn.LayerNorm(hidden_dim)
        self.norm_input = nn.LayerNorm(hidden_dim)

        # Linear projections for attention (no bias, following paper)
        self.proj_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # GRU update (inputs = aggregated update, hidden = current slots)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Slot MLP (post-GRU refinement)
        self.mlp      = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm_mlp = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (N, H)  patch features from backbone
        Returns: slots (K, H)
        """
        N = h.shape[0]
        B = 1   # single-bag (unbatched) processing

        # Sample initial slot states from learned distribution
        # sigma = softplus for numerical stability
        sigma = F.softplus(self.slot_log_sigma) + 1e-6
        if self.training:
            slots = self.slot_mu + sigma * torch.randn_like(self.slot_mu)
        else:
            slots = self.slot_mu.expand_as(self.slot_mu)  # deterministic eval
        slots = slots.squeeze(0)   # (K, H)

        # Pre-compute key and value projections (fixed across iterations)
        h_norm = self.norm_input(h)            # (N, H)
        k      = self.proj_k(h_norm)           # (N, H)
        v      = self.proj_v(h_norm)           # (N, H)

        for _ in range(self.n_iters):
            slots_prev = slots

            # Attention logits:  q @ k.T / sqrt(H)
            q     = self.proj_q(self.norm_slots(slots))  # (K, H)
            logits = torch.matmul(q, k.T) * self.scale   # (K, N)

            # Slot competition: softmax over slots (dim=0)
            # Each patch column becomes a probability distribution over K slots
            attn = F.softmax(logits, dim=0)              # (K, N)

            # Normalise so each slot gets a weighted share (prevents empty slots)
            attn_norm = attn / (attn.sum(dim=1, keepdim=True) + 1e-6)  # (K, N)

            # Aggregate: each slot collects its weighted patches
            updates = torch.matmul(attn_norm, v)         # (K, H)

            # GRU update: treat K slots as a batch
            slots = self.gru(
                updates.view(self.n_slots, -1),
                slots_prev.view(self.n_slots, -1)
            )   # (K, H)

            # Post-GRU MLP with residual
            slots = slots + self.mlp(slots)

        return slots   # (K, H)


# ══════════════════════════════════════════════════════════════════
# PHASE 1 MODEL  (updated to use BidirPatchCrossAttn)
# ══════════════════════════════════════════════════════════════════

class SingleModalMIL(nn.Module):
    """
    Phase 1 model for one modality.

    All modalities: backbone, ABMIL gate, class head, proj_head.
    Student modalities only (use_cross_attn=True):
      bidir_cross (BidirPatchCrossAttn) — enriches student patches
        using teacher patches as context (teacher h_b detached by caller).
      cross_aux_head — classification on pooled enriched patches → L_cross.
      After enrichment, ABMIL runs on enriched h_a to get alpha_self_enr.
      alpha_a2b from bidir_cross is the KD target.
    """
    def __init__(self, feat_dim: int = 1024, hidden_dim: int = 256,
                 dropout: float = 0.4, proj_dim: int = 128,
                 use_cross_attn: bool = False,
                 n_heads: int = P1_ATTN_N_HEADS,
                 attn_dropout: float = P1_ATTN_DROPOUT,
                 use_spatial: bool = False):
        super().__init__()
        self.encoder = GatedAttentionEncoder(feat_dim, hidden_dim, dropout,
                                             use_spatial=use_spatial)
        self.head           = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.proj_head      = ProjectionHead(hidden_dim, proj_dim)
        # Strategy 2: separate projection head for patch-augmentation NT-Xent.
        # Keeps aug invariance gradients from interfering with strategy 1+3 SupCon.
        self.aug_proj_head  = ProjectionHead(hidden_dim, proj_dim)
        self.hazard_head    = nn.Linear(hidden_dim, 1, bias=True)
        nn.init.normal_(self.hazard_head.weight, 0.0, 0.01)
        nn.init.zeros_(self.hazard_head.bias)
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.bidir_cross    = BidirPatchCrossAttn(hidden_dim, n_heads, attn_dropout)
            self.cross_aux_head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor,
                return_extras: bool = False,
                coords=None):
        rep, alpha, h = self.encoder(x, coords=coords)
        logit  = self.head(rep).squeeze()
        proj_z = self.proj_head(rep)
        hazard = self.hazard_head(rep).squeeze()
        if not return_extras:
            return logit
        return logit, {"r_final": rep, "alpha": alpha, "h": h, "proj_z": proj_z, "hazard": hazard}

    def forward_with_cross(
        self,
        h_stud: torch.Tensor,       # (N_s, H) backbone features — trainable
        h_teach: torch.Tensor,      # (N_t, H) backbone features — DETACHED by caller
        target: torch.Tensor,       # (1,) label
        cw: Tuple[float, float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Bidirectional cross-attention forward for student training.

        Returns:
          alpha_self  : (N_s,)  ABMIL weights on original patches (for standard path)
          alpha_a2b   : (N_s,)  cross-attn weights (KD target)
          L_cross     : scalar  aux classification on enriched rep
          r_stud_enr  : (H,)    pooled enriched rep (for CRD)
        """
        # Bidirectional enrichment: student patches attend teacher patches and vice versa
        h_stud_enr, _, alpha_a2b = self.bidir_cross(h_stud, h_teach)
        # h_teach_enr discarded — teacher side is detached, no use in grad path

        # Pool enriched student patches with the same ABMIL gate
        # (gate runs on enriched features — it sees the cross-modal context)
        gate   = self.encoder.att_V(h_stud_enr) * self.encoder.att_U(h_stud_enr)
        raw    = self.encoder.att_w(self.encoder.att_drop(gate))
        alpha_self_enr = F.softmax(raw, dim=0).squeeze(1)   # (N_s,)
        r_stud_enr     = (alpha_self_enr.unsqueeze(1) * h_stud_enr).sum(0)  # (H,)

        L_cross = hinge_loss(self.cross_aux_head(r_stud_enr).unsqueeze(0), target, cw)

        return alpha_self_enr, alpha_a2b, L_cross, r_stud_enr


# ══════════════════════════════════════════════════════════════════
# UPDATED PHASE 1 TRAINING LOOP  (bidir cross-attn KD)
# ══════════════════════════════════════════════════════════════════

def p1_train_one_epoch(
    model: SingleModalMIL,
    records: List[dict],
    mod_name: str,
    optimizer: torch.optim.Optimizer,
    cw: Tuple[float, float],
    device: torch.device,
    bag_cache: BagCache,
    scaler: Optional[torch.amp.GradScaler],
    grad_accum: int,
    # Bidir cross-attn + KD (students only, teacher present)
    use_cross_attn: bool = False,
    teacher_r_cache: Optional[TeacherCache] = None,  # not used for h_t here
    teacher_h_cache: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
    cross_attn_lambda: float = P1_CROSS_ATTN_LAMBDA,
    kd_lambda: float = P1_KD_LAMBDA,
    kd_tau: float = P1_KD_TAU,
    kd_top_k: int = P1_KD_TOP_K,
    # CRD (students only)
    use_crd: bool = False,
    teacher_r_cache_crd: Optional[TeacherCache] = None,
    crd_lambda: float = P1_CRD_LAMBDA,
    # Contrastive on proj_z
    use_contrastive: bool = False,
    teacher_pz_cache: Optional[TeacherCache] = None,
    clr_tau: float = P1_CLR_TAU,
    clr_lambda: float = P1_CLR_LAMBDA,
    # Contrastive on L2(r_self)
    use_summary_clr: bool = False,
    summary_clr_lambda: float = P1_SUMMARY_CLR_LAMBDA,
    # Strategy 2: patch-augmentation NT-Xent
    use_aug_clr: bool = False,
    aug_clr_lambda: float = P1_AUG_CLR_LAMBDA,
    aug_subsample: float = P1_AUG_SUBSAMPLE,
    aug_min_patches: int = P1_AUG_MIN_PATCHES,
    # Spatial PE (HE only)
    use_spatial: bool = False,
) -> Dict[str, float]:
    """
    Phase 1 epoch with bidirectional patch cross-attention.

    Three complementary CLR strategies (all optional, additive at step boundary):
      1. Cross-modal SupCon (same_stem & diff_mod): aligns different modalities of
         the same patient via proj_head. Enabled by use_contrastive.
      2. Patch-augmentation NT-Xent (use_aug_clr): two random patch subsamples of
         the same bag are pulled together via aug_proj_head. Teaches encoder
         invariance to which patches happen to be sampled.
      3. Label-supervised SupCon (same_label): aligns patients with same label via
         proj_head. Enabled by use_contrastive alongside strategy 1.

    Strategies 1+3 share proj_head; strategy 2 uses aug_proj_head — gradient
    flows are isolated between the two projection spaces while the shared encoder
    accumulates all three signals before each optimiser step.
    """
    model.train()
    random.shuffle(records)

    is_student = mod_name in STUDENT_MODALITIES
    can_cross  = (use_cross_attn and is_student
                  and model.use_cross_attn
                  and teacher_h_cache is not None)
    can_crd    = (use_crd and is_student and teacher_r_cache_crd is not None)

    totals = {"task": 0.0, "cross": 0.0, "kd": 0.0, "crd": 0.0, "clr": 0.0}
    counts = {k: 0 for k in totals}

    accum_step = 0
    grad_accumulated = False
    batch_buffer:   List[Tuple[torch.Tensor, int, str, str]] = []
    summary_buffer: List[Tuple[torch.Tensor, int, str, str]] = []
    aug_buffer:     List[Tuple[torch.Tensor, torch.Tensor]]  = []
    optimizer.zero_grad()

    for rec in records:
        stem  = rec["stem"]
        label = rec["label"]
        bag   = bag_cache.get(stem, {}).get(mod_name)
        if bag is None: continue

        bag_dev = bag.to(device, non_blocking=True)
        target  = torch.tensor([label], dtype=torch.float32, device=device)
        use_amp = scaler is not None

        # Spatial 2-D PE coords for HE (None for other modalities)
        he_coords = bag_cache.get(stem, {}).get("HE_coords") if use_spatial else None

        # ── Standard forward ─────────────────────────────────────
        with torch.amp.autocast("cuda", enabled=use_amp):
            logit, extras = model(bag_dev, return_extras=True,
                                  coords=he_coords)
            L_task = hinge_loss(logit.unsqueeze(0), target, cw) / grad_accum

        r_self    = extras["r_final"]
        alpha_self = extras["alpha"]
        h_stud    = extras["h"]
        proj_z    = extras["proj_z"]

        L_total = L_task

        # ── Bidir cross-attn + KD + CRD ──────────────────────────
        if can_cross or can_crd:
            h_teach_cpu = None
            for t_mod in TEACHER_MODALITIES:
                h_t = (teacher_h_cache or {}).get(stem, {}).get(t_mod)
                if h_t is not None:
                    h_teach_cpu = h_t; break

            if h_teach_cpu is not None:
                # Cache already holds exactly the top-K patches by ABMIL attention
                # (selected once in build_teacher_caches) — no further subsampling needed.
                h_teach = h_teach_cpu.to(device, non_blocking=True).detach()

                if can_cross:
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        alpha_self_enr, alpha_a2b, L_cross_raw, r_stud_enr = \
                            model.forward_with_cross(h_stud, h_teach, target, cw)

                        L_cross = L_cross_raw * cross_attn_lambda / grad_accum

                        # KD: alpha_a2b is the per-student-patch attention
                        # averaged over teacher patches — shape (N_s,)
                        # Use alpha_self_enr (on enriched patches) as student side
                        L_kd = (attention_transfer_loss(
                                    alpha_self_enr, alpha_a2b.detach(),
                                    kd_tau, kd_top_k)
                                * kd_lambda / grad_accum)

                    L_total = L_total + L_cross + L_kd
                    totals["cross"] += L_cross.item() * grad_accum; counts["cross"] += 1
                    totals["kd"]    += L_kd.item()    * grad_accum; counts["kd"]    += 1

                    if can_crd:
                        # CRD: pull enriched student summary toward cached teacher summary
                        r_teacher = None
                        for t_mod in TEACHER_MODALITIES:
                            rt = (teacher_r_cache_crd or {}).get(stem, {}).get(t_mod)
                            if rt is not None:
                                r_teacher = rt.to(device, non_blocking=True).detach(); break
                        if r_teacher is not None:
                            with torch.amp.autocast("cuda", enabled=use_amp):
                                L_crd = (crd_loss_fn(r_stud_enr, r_teacher)
                                         * crd_lambda / grad_accum)
                            L_total = L_total + L_crd
                            totals["crd"] += L_crd.item() * grad_accum; counts["crd"] += 1

                elif can_crd and not can_cross:
                    r_teacher = None
                    for t_mod in TEACHER_MODALITIES:
                        rt = (teacher_r_cache_crd or {}).get(stem, {}).get(t_mod)
                        if rt is not None:
                            r_teacher = rt.to(device, non_blocking=True).detach(); break
                    if r_teacher is not None:
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            L_crd = (crd_loss_fn(r_self, r_teacher)
                                     * crd_lambda / grad_accum)
                        L_total = L_total + L_crd
                        totals["crd"] += L_crd.item() * grad_accum; counts["crd"] += 1

                del h_teach   # free GPU memory before backward

        # ── Backward ─────────────────────────────────────────────
        if scaler is not None:
            scaler.scale(L_total).backward()
        else:
            L_total.backward()

        grad_accumulated = True
        totals["task"] += L_task.item() * grad_accum; counts["task"] += 1

        # ── Contrastive buffers ───────────────────────────────────
        # Re-project from detached r_self so these tensors have their own
        # fresh graph and won't fail when CLR backward runs after L_total.backward().
        if use_contrastive:
            proj_z_clr = model.proj_head(r_self.detach())
            batch_buffer.append((proj_z_clr, label, stem, mod_name))
            if teacher_pz_cache is not None:
                for t_mod in TEACHER_MODALITIES:
                    t_pz = teacher_pz_cache.get(stem, {}).get(t_mod)
                    if t_pz is not None:
                        batch_buffer.append((t_pz.to(device, non_blocking=True), label, stem, t_mod))

        if use_summary_clr:
            r_norm = F.normalize(r_self.detach(), dim=0)
            summary_buffer.append((r_norm, label, stem, mod_name))
            if teacher_r_cache_crd is not None:
                for t_mod in TEACHER_MODALITIES:
                    r_t = teacher_r_cache_crd.get(stem, {}).get(t_mod)
                    if r_t is not None:
                        summary_buffer.append(
                            (F.normalize(r_t.to(device, non_blocking=True).detach(), dim=0),
                             label, stem, t_mod))

        # ── Strategy 2: patch-augmentation views ─────────────────
        # Two independent random subsamples of the same bag are pulled together
        # via aug_proj_head (NT-Xent at step boundary).  Uses a fresh forward
        # so its graph is isolated from the L_task graph already backward'd above.
        if use_aug_clr and bag_dev.shape[0] >= aug_min_patches:
            n_keep = max(2, int(bag_dev.shape[0] * aug_subsample))
            idx1 = torch.randperm(bag_dev.shape[0], device=device)[:n_keep]
            idx2 = torch.randperm(bag_dev.shape[0], device=device)[:n_keep]
            with torch.amp.autocast("cuda", enabled=use_amp):
                r1, _, _ = model.encoder(bag_dev[idx1])
                r2, _, _ = model.encoder(bag_dev[idx2])
                z1 = model.aug_proj_head(r1)   # already L2-normalised by ProjectionHead
                z2 = model.aug_proj_head(r2)
            aug_buffer.append((z1, z2))

        accum_step += 1

        if accum_step == grad_accum:
            _p1_step_boundary(
                scaler, optimizer, batch_buffer, summary_buffer,
                use_contrastive, use_summary_clr,
                clr_tau, clr_lambda, summary_clr_lambda, cw, totals, counts,
                aug_buffer=aug_buffer, use_aug_clr=use_aug_clr,
                aug_clr_lambda=aug_clr_lambda)
            batch_buffer.clear(); summary_buffer.clear(); aug_buffer.clear()
            accum_step = 0; grad_accumulated = False

        del bag_dev, target
        if counts["task"] % 200 == 0: _gc()   # less frequent; cuda.empty_cache is synchronous

    if accum_step > 0 and grad_accumulated:
        _p1_step_boundary(
            scaler, optimizer, batch_buffer, summary_buffer,
            use_contrastive, use_summary_clr,
            clr_tau, clr_lambda, summary_clr_lambda, cw, totals, counts,
            aug_buffer=aug_buffer, use_aug_clr=use_aug_clr,
            aug_clr_lambda=aug_clr_lambda)

    return {k: totals[k] / max(counts[k], 1) for k in totals}


# ══════════════════════════════════════════════════════════════════
# TEACHER CACHES  (now includes backbone feature cache)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_teacher_caches(
    teacher_models: Dict[str, SingleModalMIL],
    train_recs: List[dict],
    bag_cache: BagCache,
    device: torch.device,
) -> Tuple[TeacherCache, TeacherCache, Dict]:
    """
    Build three caches from trained teacher models (HE, Clinical):
      r_cache  {stem: {mod: r_final (H,)}}   — ABMIL summary, for CRD
      pz_cache {stem: {mod: proj_z (D,)}}    — proj head, for SupCon
      h_cache  {stem: {mod: h (N, H)}}       — backbone features, for bidir cross-attn

    h_cache is the new addition: stores raw patch-level backbone features
    so the bidirectional cross-attention can attend patch-to-patch rather
    than summary-to-patch.

    Only train_recs processed. All tensors stored on CPU.
    """
    print(f"  [teacher_caches] building for {len(train_recs)} stems "
          f"(r + proj_z + patch features) ...")
    r_cache:  TeacherCache = {}
    pz_cache: TeacherCache = {}
    h_cache:  Dict[str, Dict[str, Optional[torch.Tensor]]] = {}

    for mod, model in teacher_models.items(): model.eval()

    for rec in train_recs:
        stem = rec["stem"]
        r_cache.setdefault(stem,  {m: None for m in teacher_models})
        pz_cache.setdefault(stem, {m: None for m in teacher_models})
        h_cache.setdefault(stem,  {m: None for m in teacher_models})

        for mod, model in teacher_models.items():
            bag = bag_cache.get(stem, {}).get(mod)
            if bag is None: continue
            bag_dev = bag.to(device, non_blocking=True)
            _, extras = model(bag_dev, return_extras=True)
            r_cache[stem][mod]  = extras["r_final"].detach().cpu()
            pz_cache[stem][mod] = extras["proj_z"].detach().cpu()
            # Keep only the top-K patches by ABMIL attention weight.
            # This is semantically consistent with what ABMIL considers
            # important and avoids including low-attention noise patches
            # in cross-attn.  alpha (N,) is already on GPU here.
            h     = extras["h"].detach()        # (N, H) on GPU
            alpha = extras["alpha"].detach()    # (N,)   ABMIL weights
            k     = min(P1_MAX_TEACH_PATCHES, h.shape[0])
            top_idx = alpha.topk(k, dim=0).indices   # highest-attention indices
            h_cache[stem][mod] = h[top_idx].cpu()
            del bag_dev, h, alpha, top_idx

        if len(r_cache) % 200 == 0: _gc()

    for mod in teacher_models:
        n = sum(1 for e in r_cache.values() if e.get(mod) is not None)
        print(f"  [teacher_caches] {mod}: {n}/{len(train_recs)} cached")

    # Pin all cache tensors so CPU→GPU DMA can overlap with GPU compute
    # (non_blocking=True transfers benefit from pinned memory).
    if device.type == "cuda":
        for cache in (r_cache, pz_cache, h_cache):
            for stem_d in cache.values():
                for m, t in stem_d.items():
                    if t is not None:
                        stem_d[m] = t.pin_memory()

    return r_cache, pz_cache, h_cache


@torch.no_grad()
def p1_evaluate(model, records, mod_name, device, bag_cache,
                use_spatial=False, cw=None):
    """Single-pass eval: returns (probs, labels[, val_loss]).
    Pass cw=(w_neg, w_pos) to also compute val_loss in the same pass,
    avoiding a second full loop over the val set."""
    model.eval(); probs, labels, losses = [], [], []
    use_amp = (device.type == "cuda")
    for rec in records:
        bag = bag_cache.get(rec["stem"], {}).get(mod_name)
        if bag is None: continue
        bag_dev   = bag.to(device, non_blocking=True)
        he_coords = bag_cache.get(rec["stem"], {}).get("HE_coords") if use_spatial else None
        with torch.amp.autocast("cuda", enabled=use_amp):
            logit = model(bag_dev, coords=he_coords)
        probs.append(torch.sigmoid(logit.float()).item())
        labels.append(rec["label"])
        if cw is not None:
            ta = logit.new_tensor([rec["label"]])
            losses.append(hinge_loss(logit.unsqueeze(0), ta, cw).item())
        del bag_dev
    val_loss = float(np.mean(losses)) if losses else 0.0
    if cw is not None:
        return np.array(probs), np.array(labels), val_loss
    return np.array(probs), np.array(labels)


def p1_train_one_epoch_survival(
    model: SingleModalMIL,
    records: List[dict],
    mod_name: str,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    bag_cache: BagCache,
    scaler: Optional[torch.amp.GradScaler],
    grad_accum: int,
    surv_endpoint: str = 'clad',
    use_spatial: bool = False,
    # Strategy 2: patch-augmentation NT-Xent (aug_proj_head)
    use_aug_clr: bool = False,
    aug_clr_lambda: float = P1_AUG_CLR_LAMBDA,
    aug_subsample: float = P1_AUG_SUBSAMPLE,
    aug_min_patches: int = P1_AUG_MIN_PATCHES,
    # Strategy 3: label-supervised SupCon (event status as label, proj_head)
    use_label_supcon: bool = False,
    label_supcon_lambda: float = P1_LABEL_SUPCON_LAMBDA,
    clr_tau: float = P1_CLR_TAU,
) -> Dict[str, float]:
    """
    Phase 1 epoch for survival (Cox loss).  Optionally adds:
      Strategy 2: patch-augmentation NT-Xent via aug_proj_head.
      Strategy 3: event-label SupCon via proj_head (event 0/1 as pseudo-label).
    Both run at the grad_accum step boundary, accumulating onto Cox gradients.
    """
    model.train()
    random.shuffle(records)

    cox_buffer: list = []
    aug_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
    label_buffer: List[Tuple[torch.Tensor, int, str, str]] = []
    total_loss = 0.0; clr_total_logged = 0.0; n_steps = 0; n_clr = 0
    accum_step = 0
    use_amp = (scaler is not None)
    use_spatial_for_mod = use_spatial and mod_name == "HE"
    # Equal weights for survival SupCon (no class imbalance weighting needed)
    surv_cw = (1.0, 1.0)
    optimizer.zero_grad()

    def _surv_step_boundary():
        nonlocal total_loss, clr_total_logged, n_steps, n_clr
        L_cox = cox_breslow_loss(cox_buffer)
        has_cox = L_cox is not None and L_cox.requires_grad
        if has_cox:
            if scaler: scaler.scale(L_cox).backward()
            else:      L_cox.backward()
            total_loss += L_cox.item()
            n_steps += 1

        # CLR losses: strategies 2+3 (accumulated on top of Cox gradient)
        clr_total = None
        if use_aug_clr and len(aug_buffer) >= 2:
            z1s = torch.stack([a[0] for a in aug_buffer])
            z2s = torch.stack([a[1] for a in aug_buffer])
            L_aug = nt_xent_loss(z1s, z2s, clr_tau)
            if L_aug is not None:
                clr_total = aug_clr_lambda * L_aug
                clr_total_logged += L_aug.item(); n_clr += 1
        if use_label_supcon and len(label_buffer) >= 2:
            # min_multimodal_stems=0: single-modal survival, no cross-modal pairs
            L_lsup = batch_supcon_loss(label_buffer, clr_tau, surv_cw,
                                       min_multimodal_stems=0)
            if L_lsup is not None:
                term = label_supcon_lambda * L_lsup
                clr_total = clr_total + term if clr_total is not None else term
                clr_total_logged += L_lsup.item(); n_clr += 1
        if clr_total is not None:
            if scaler: scaler.scale(clr_total).backward()
            else:      clr_total.backward()

        if has_cox or clr_total is not None:
            if scaler: scaler.step(optimizer); scaler.update()
            else:      optimizer.step()
        optimizer.zero_grad()
        cox_buffer.clear(); aug_buffer.clear(); label_buffer.clear()
        _gc()

    for rec in records:
        surv_t = rec.get(f"{surv_endpoint}_time", float("nan"))
        surv_e = rec.get(f"{surv_endpoint}_event", float("nan"))
        if not isinstance(surv_t, float) or math.isnan(surv_t):
            continue
        bag = bag_cache.get(rec["stem"], {}).get(mod_name)
        if bag is None:
            continue

        bag_dev   = bag.to(device, non_blocking=True)
        he_coords = bag_cache.get(rec["stem"], {}).get("HE_coords") if use_spatial_for_mod else None

        with torch.amp.autocast("cuda", enabled=use_amp):
            _, extras = model(bag_dev, return_extras=True, coords=he_coords)
        hazard = extras.get("hazard")
        if hazard is None:
            del bag_dev; continue
        r_self = extras["r_final"]

        cox_buffer.append((hazard.float(), surv_t, surv_e))

        # Strategy 2: augmented patch views via aug_proj_head (with grad)
        if use_aug_clr and bag_dev.shape[0] >= aug_min_patches:
            n_keep = max(2, int(bag_dev.shape[0] * aug_subsample))
            idx1 = torch.randperm(bag_dev.shape[0], device=device)[:n_keep]
            idx2 = torch.randperm(bag_dev.shape[0], device=device)[:n_keep]
            with torch.amp.autocast("cuda", enabled=use_amp):
                r1, _, _ = model.encoder(bag_dev[idx1])
                r2, _, _ = model.encoder(bag_dev[idx2])
                z1 = model.aug_proj_head(r1)
                z2 = model.aug_proj_head(r2)
            aug_buffer.append((z1, z2))

        # Strategy 3: event-label SupCon — proj_head on detached r (fresh graph)
        if use_label_supcon:
            pz = model.proj_head(r_self.detach())
            label_buffer.append((pz, int(surv_e), rec["stem"], mod_name))

        accum_step += 1
        del bag_dev

        if accum_step == grad_accum:
            _surv_step_boundary()
            accum_step = 0

    if accum_step > 0:
        _surv_step_boundary()

    return {"task":  total_loss / max(n_steps, 1),
            "cross": 0.0, "kd": 0.0, "crd": 0.0,
            "clr":   clr_total_logged / max(n_clr, 1)}


@torch.no_grad()
def p1_evaluate_survival(model, records, mod_name, device, bag_cache,
                          surv_endpoint='clad', use_spatial=False):
    """Evaluate P1 survival model; returns C-index and mean Cox loss."""
    model.eval()
    hazards, times, events = [], [], []
    cox_buf: list = []
    use_amp = (device.type == "cuda")
    use_spatial_for_mod = use_spatial and mod_name == "HE"
    for rec in records:
        surv_t = rec.get(f"{surv_endpoint}_time", float("nan"))
        surv_e = rec.get(f"{surv_endpoint}_event", float("nan"))
        if not isinstance(surv_t, float) or math.isnan(surv_t):
            continue
        bag = bag_cache.get(rec["stem"], {}).get(mod_name)
        if bag is None:
            continue
        bag_dev   = bag.to(device, non_blocking=True)
        he_coords = bag_cache.get(rec["stem"], {}).get("HE_coords") if use_spatial_for_mod else None
        with torch.amp.autocast("cuda", enabled=use_amp):
            with torch.enable_grad():
                _, extras = model(bag_dev, return_extras=True, coords=he_coords)
        hazard = extras.get("hazard")
        if hazard is None:
            del bag_dev; continue
        hazards.append(hazard.float().item())
        times.append(surv_t)
        events.append(surv_e)
        cox_buf.append((hazard.detach().float(), surv_t, surv_e))
        del bag_dev
    ci = c_index(hazards, times, events) if len(hazards) >= 2 and sum(events) > 0 else 0.5
    val_cox = cox_breslow_loss(cox_buf)
    val_loss = float(val_cox.item()) if val_cox is not None else 0.0
    return ci, val_loss


def _p1_step_boundary(scaler, optimizer, batch_buffer, summary_buffer,
                      use_contrastive, use_summary_clr,
                      clr_tau, clr_lambda, summary_clr_lambda, cw,
                      totals, counts,
                      aug_buffer=None,
                      use_aug_clr: bool = False,
                      aug_clr_lambda: float = P1_AUG_CLR_LAMBDA):
    """
    Single optimiser step covering three complementary CLR strategies:
      1+3: SupCon on proj_head (cross-modal + label positives) — existing
      2:   NT-Xent on aug_proj_head (patch-subsample augmented views) — new

    Strategy 2 uses aug_proj_head (separate from proj_head) so its invariance
    gradient does not directly overwrite the discrimination gradient of strategy 1+3.
    The encoder is shared and accumulates all three signals before each step.
    """
    clr_total = None

    # ── Strategy 2: patch-augmentation NT-Xent ────────────────────
    if use_aug_clr and aug_buffer:
        z1s = torch.stack([a[0] for a in aug_buffer])   # (N, D)
        z2s = torch.stack([a[1] for a in aug_buffer])   # (N, D)
        L_aug = nt_xent_loss(z1s, z2s, clr_tau)
        if L_aug is not None:
            clr_total = aug_clr_lambda * L_aug
            totals["clr"] += L_aug.item(); counts["clr"] += 1

    # ── Strategy 1+3: cross-modal + label SupCon on proj_z ────────
    if use_contrastive and batch_buffer:
        L_clr = batch_supcon_loss(batch_buffer, clr_tau, cw)
        if L_clr is not None:
            term = L_clr * clr_lambda
            clr_total = clr_total + term if clr_total is not None else term
            totals["clr"] += L_clr.item(); counts["clr"] += 1

    # ── Strategy 1+3 on L2(r_self) ────────────────────────────────
    if use_summary_clr and summary_buffer:
        L_sclr = batch_supcon_loss(summary_buffer, clr_tau, cw)
        if L_sclr is not None:
            sclr_term = L_sclr * summary_clr_lambda
            clr_total = clr_total + sclr_term if clr_total is not None else sclr_term
            totals["clr"] += L_sclr.item(); counts["clr"] += 1

    if clr_total is not None:
        s = scaler.scale(clr_total) if scaler else clr_total
        s.backward()
    if scaler is not None:
        scaler.step(optimizer); scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad()


# ══════════════════════════════════════════════════════════════════
# PHASE 1 RUNNER  (updated with h_cache parameter)
# ══════════════════════════════════════════════════════════════════

def run_phase1_modality(
    mod_name: str, fold: int, device: torch.device,
    bag_cache: BagCache, train_recs: List[dict],
    val_recs: List[dict], test_recs: List[dict],
    save_dir: Path,
    use_cross_attn: bool = False,
    teacher_r_cache: Optional[TeacherCache] = None,
    teacher_h_cache: Optional[Dict] = None,
    use_crd: bool = False,
    use_contrastive: bool = False,
    teacher_pz_cache: Optional[TeacherCache] = None,
    use_summary_clr: bool = False,
    clr_tau: float = P1_CLR_TAU,
    clr_lambda: float = P1_CLR_LAMBDA,
    summary_clr_lambda: float = P1_SUMMARY_CLR_LAMBDA,
    cross_attn_lambda: float = P1_CROSS_ATTN_LAMBDA,
    kd_lambda: float = P1_KD_LAMBDA,
    kd_tau: float = P1_KD_TAU,
    kd_top_k: int = P1_KD_TOP_K,
    crd_lambda: float = P1_CRD_LAMBDA,
    # Strategy 2: patch-augmentation NT-Xent (both ACR and survival)
    use_aug_clr: bool = False,
    aug_clr_lambda: float = P1_AUG_CLR_LAMBDA,
    aug_subsample: float = P1_AUG_SUBSAMPLE,
    aug_min_patches: int = P1_AUG_MIN_PATCHES,
    # Strategy 3 for survival: event-label SupCon
    use_label_supcon: bool = False,
    label_supcon_lambda: float = P1_LABEL_SUPCON_LAMBDA,
    use_spatial: bool = False,
    n_epochs: int = P1_EPOCHS,
    patience: int = 0,            # eval periods without improvement → stop (0 = disabled)
    task: str = 'acr',            # 'acr' = hinge classification; 'survival' = Cox only
    surv_endpoint: str = 'clad',
) -> Path:
    print(f"\n  {'─'*60}")
    print(f"  Phase 1 — {mod_name}  (fold {fold})  [task={task}]")

    is_teacher = mod_name in TEACHER_MODALITIES
    if is_teacher or task == 'survival':
        # Survival mode: no cross-attn aux (it uses classification head)
        use_cross_attn = False; teacher_r_cache = None
        teacher_h_cache = None; teacher_pz_cache = None; use_crd = False

    active = []
    if task == 'survival':
        active.append(f"Cox({surv_endpoint})")
        if use_aug_clr:      active.append(f"CLR-aug(λ={aug_clr_lambda},sub={aug_subsample})")
        if use_label_supcon: active.append(f"CLR-label(λ={label_supcon_lambda})")
    else:
        if use_cross_attn:  active.append(f"BidirCrossAttn+KD(τ={kd_tau},K={kd_top_k})")
        if use_crd:         active.append("CRD")
        if use_contrastive: active.append("CLR-proj(s1+s3)")
        if use_summary_clr: active.append("CLR-summary")
        if use_aug_clr:     active.append(f"CLR-aug(λ={aug_clr_lambda},sub={aug_subsample})")
    print(f"  Active: task" + ((" + " + " + ".join(active)) if active else " only"))
    print(f"  {'─'*60}")

    pc = _pres_col(mod_name)
    tr = [r for r in train_recs if r.get(pc)]
    vl = [r for r in val_recs   if r.get(pc)]
    te = [r for r in test_recs  if r.get(pc)]
    print(f"  Present-only: train={len(tr)}  val={len(vl)}  test={len(te)}")

    use_spatial_for_mod = use_spatial and mod_name == "HE"
    if len(tr) == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        dummy = SingleModalMIL(_feat_dim(mod_name), HIDDEN_DIM, DROPOUT,
                               P1_CLR_PROJ_DIM, use_cross_attn=False,
                               use_spatial=use_spatial_for_mod)
        torch.save(dummy.state_dict(), save_dir / "best_model.pt")
        _write_status(save_dir / "status.json", completed=True,
                      best_epoch=0, best_bacc=0.0, last_epoch=0, note="dummy")
        return save_dir / "best_model.pt"

    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = save_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)

    if _is_completed(save_dir):
        st = _read_status(save_dir / "status.json")
        print(f"  [{mod_name}] Already completed "
              f"(best_ep={st.get('best_epoch')}  "
              f"best_bacc={st.get('best_bacc',0):.4f}). Skipping.")
        assert (save_dir / "best_model.pt").exists()
        return save_dir / "best_model.pt"

    cw = compute_class_weights(tr)
    model = SingleModalMIL(
        feat_dim=_feat_dim(mod_name), hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
        proj_dim=P1_CLR_PROJ_DIM,
        use_cross_attn=(use_cross_attn and mod_name in STUDENT_MODALITIES and task == 'acr'),
        use_spatial=use_spatial_for_mod,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=P1_LR, weight_decay=P1_WEIGHT_DECAY)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    n_params  = sum(p.numel() for p in model.parameters())
    if task == 'acr':
        print(f"  Params: {n_params:,}  cw=(neg={cw[0]:.3f}, pos={cw[1]:.3f})  AMP={scaler is not None}")
    else:
        print(f"  Params: {n_params:,}  Cox({surv_endpoint})  AMP={scaler is not None}")

    hist_keys = ["train_loss","val_loss","val_auc","val_bacc","val_mcc",
                 "cross_loss","kd_loss","crd_loss","clr_loss"]
    history: Dict[str, List] = {k: [] for k in hist_keys}

    resume_epoch = _find_resume_epoch(ckpt_dir); start_epoch = 0
    if resume_epoch >= n_epochs:
        print(f"  [{mod_name}] Training complete. Rescanning.")
        start_epoch = n_epochs  # skip training loop — already finished
    elif resume_epoch > 0:
        ckpt = _load_checkpoint(ckpt_dir, resume_epoch)
        if ckpt is not None:
            model.load_state_dict(ckpt["model"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer"])
            if scaler is not None and ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            for k in history:
                if k in ckpt.get("history", {}): history[k] = ckpt["history"][k]
            start_epoch = resume_epoch; model.to(device)
            print(f"  [{mod_name}] Resumed from epoch {resume_epoch}.")

    # Inline best-model tracking (best metric = val_bacc for ACR, val_cidx for survival)
    _best_metric_inline: float = max(history["val_bacc"]) if history["val_bacc"] else -1.0
    _best_ep_inline:   int   = 0
    _no_improve:       int   = 0

    print(f"  [{mod_name}] epochs={n_epochs}  patience={patience or 'off'}  "
          f"eval_every={P1_EVAL_EVERY}")

    for epoch in range(start_epoch, n_epochs):
        if task == 'survival':
            loss_d = p1_train_one_epoch_survival(
                model=model, records=tr, mod_name=mod_name,
                optimizer=optimizer, device=device,
                bag_cache=bag_cache, scaler=scaler, grad_accum=P1_GRAD_ACCUM,
                surv_endpoint=surv_endpoint, use_spatial=use_spatial_for_mod,
                use_aug_clr=use_aug_clr, aug_clr_lambda=aug_clr_lambda,
                aug_subsample=aug_subsample, aug_min_patches=aug_min_patches,
                use_label_supcon=use_label_supcon,
                label_supcon_lambda=label_supcon_lambda, clr_tau=clr_tau,
            )
        else:
            loss_d = p1_train_one_epoch(
                model=model, records=tr, mod_name=mod_name,
                optimizer=optimizer, cw=cw, device=device,
                bag_cache=bag_cache, scaler=scaler, grad_accum=P1_GRAD_ACCUM,
                use_cross_attn=use_cross_attn,
                teacher_r_cache=teacher_r_cache,
                teacher_h_cache=teacher_h_cache,
                cross_attn_lambda=cross_attn_lambda,
                kd_lambda=kd_lambda, kd_tau=kd_tau, kd_top_k=kd_top_k,
                use_crd=use_crd, teacher_r_cache_crd=teacher_r_cache,
                crd_lambda=crd_lambda,
                use_contrastive=use_contrastive, teacher_pz_cache=teacher_pz_cache,
                clr_tau=clr_tau, clr_lambda=clr_lambda,
                use_summary_clr=use_summary_clr, summary_clr_lambda=summary_clr_lambda,
                use_aug_clr=use_aug_clr, aug_clr_lambda=aug_clr_lambda,
                aug_subsample=aug_subsample, aug_min_patches=aug_min_patches,
                use_spatial=use_spatial_for_mod,
            )
        history["train_loss"].append(loss_d["task"])
        history["cross_loss"].append(loss_d["cross"])
        history["kd_loss"].append(loss_d["kd"])
        history["crd_loss"].append(loss_d["crd"])
        history["clr_loss"].append(loss_d["clr"])
        _gc()

        if (epoch + 1) % P1_EVAL_EVERY == 0:
            if task == 'survival':
                val_cidx, val_loss = p1_evaluate_survival(
                    model, vl, mod_name, device, bag_cache,
                    surv_endpoint=surv_endpoint, use_spatial=use_spatial_for_mod)
                model.train()
                history["val_loss"].append(val_loss)
                history["val_auc"].append(val_cidx)   # repurpose val_auc slot for C-index
                history["val_bacc"].append(val_cidx)
                history["val_mcc"].append(0.0)
                torch.save({
                    "epoch": epoch+1, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler else None,
                    "history": history,
                }, ckpt_dir / f"ep{epoch+1:04d}.pt")
                if val_cidx > _best_metric_inline:
                    _best_metric_inline = val_cidx
                    _best_ep_inline     = epoch + 1
                    _no_improve         = 0
                    torch.save(model.state_dict(), save_dir / "best_model.pt")
                    ckpt_tag = "[ckpt*]"
                else:
                    _no_improve += 1
                    ckpt_tag = "[ckpt]"
                print(f"  [{mod_name}] ep {epoch+1:3d}  "
                      f"cox_loss={loss_d['task']:.4f}/{val_loss:.4f}"
                      f"  cidx={val_cidx:.3f}  {ckpt_tag}"
                      + (f"  no_improve={_no_improve}/{patience}" if patience > 0 else ""))
            else:
                # ACR path
                vl_p, vl_l, val_loss = p1_evaluate(model, vl, mod_name, device, bag_cache,
                                                   use_spatial=use_spatial_for_mod, cw=cw)
                vm = compute_metrics(vl_l, vl_p)
                model.train()
                history["val_loss"].append(val_loss)
                history["val_auc"].append(vm["auc"])
                history["val_bacc"].append(vm["bacc"])
                history["val_mcc"].append(vm.get("mcc", 0.0))
                torch.save({
                    "epoch": epoch+1, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler else None,
                    "history": history,
                }, ckpt_dir / f"ep{epoch+1:04d}.pt")
                aux_str = ""
                if loss_d["cross"] > 0: aux_str += f"  cx={loss_d['cross']:.4f}"
                if loss_d["kd"]    > 0: aux_str += f"  kd={loss_d['kd']:.4f}"
                if loss_d["crd"]   > 0: aux_str += f"  crd={loss_d['crd']:.4f}"
                if loss_d["clr"]   > 0: aux_str += f"  clr={loss_d['clr']:.4f}"
                if vm["bacc"] > _best_metric_inline:
                    _best_metric_inline = vm["bacc"]
                    _best_ep_inline     = epoch + 1
                    _no_improve         = 0
                    torch.save(model.state_dict(), save_dir / "best_model.pt")
                    ckpt_tag = "[ckpt*]"
                else:
                    _no_improve += 1
                    ckpt_tag = "[ckpt]"
                print(f"  [{mod_name}] ep {epoch+1:3d}  "
                      f"Lt={loss_d['task']:.4f}/{val_loss:.4f}" + aux_str +
                      f"  auc={vm['auc']:.3f}  bacc={vm['bacc']:.3f}  {ckpt_tag}"
                      + (f"  no_improve={_no_improve}/{patience}" if patience > 0 else ""))
            _gc()
            if patience > 0 and _no_improve >= patience:
                print(f"  [{mod_name}] Early stop: {_no_improve} evals without improvement "
                      f"(best_ep={_best_ep_inline}  best={_best_metric_inline:.4f})")
                break
        elif (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{mod_name}] ep {epoch+1:3d}  train_loss={loss_d['task']:.4f}")

    # ── Finalise best_model.pt ────────────────────────────────────────────────
    ckpts = sorted(ckpt_dir.glob("ep*.pt"))
    best_metric_key = "val_bacc"   # same history key for both tasks

    if (save_dir / "best_model.pt").exists() and _best_ep_inline > 0:
        print(f"\n  [{mod_name}] Using inline best_model.pt "
              f"(ep={_best_ep_inline}  metric={_best_metric_inline:.4f})")
        state = torch.load(save_dir / "best_model.pt",
                           map_location="cpu", weights_only=False)
        state = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state, strict=False); model.to(device); del state
        best_ep_final, best_bacc_final = _best_ep_inline, _best_metric_inline
    elif ckpts:
        print(f"\n  [{mod_name}] Fast-scanning {len(ckpts)} checkpoint histories ...")
        best_bacc_final, best_ep_final, best_path = -1.0, 0, ckpts[-1]
        for cp in ckpts:
            try:
                data = torch.load(cp, map_location="cpu", weights_only=False)
                hist_b = data.get("history", {}).get(best_metric_key, [])
                b = max(hist_b) if hist_b else -1.0
                if b > best_bacc_final:
                    best_bacc_final, best_ep_final, best_path = b, int(cp.stem[2:]), cp
                del data
            except Exception:
                pass
        print(f"  [{mod_name}] best ep≈{best_ep_final}  metric={best_bacc_final:.4f}")
        data  = torch.load(best_path, map_location="cpu", weights_only=False)
        state = data["model"] if isinstance(data, dict) else data
        model.load_state_dict(state, strict=False); model.to(device); del data, state
        torch.save(model.state_dict(), save_dir / "best_model.pt")
    else:
        best_ep_final, best_bacc_final = 0, 0.0
        torch.save(model.state_dict(), save_dir / "best_model.pt")

    _write_status(save_dir / "status.json", completed=True,
                  best_epoch=best_ep_final, best_bacc=round(best_bacc_final, 4),
                  last_epoch=_best_ep_inline or n_epochs)

    # Final metrics
    metrics: dict = {}
    for sn, recs in [("train", tr), ("val", vl), ("test", te)]:
        if task == 'survival':
            ci, _ = p1_evaluate_survival(model, recs, mod_name, device, bag_cache,
                                         surv_endpoint=surv_endpoint,
                                         use_spatial=use_spatial_for_mod)
            metrics[sn] = {"c_index": ci}
            print(f"  [{mod_name}] {sn:5s}  C-index={ci:.4f}")
        else:
            p, l = p1_evaluate(model, recs, mod_name, device, bag_cache,
                               use_spatial=use_spatial_for_mod)
            m    = compute_metrics(l, p)
            m["auprc"] = average_precision_score(l, p) if len(np.unique(l)) > 1 else 0.0
            metrics[sn] = {**m, "probs": p.tolist(), "labels": l.tolist()}
            print(f"  [{mod_name}] {sn:5s}  AUC={m['auc']:.4f}  AUPRC={m['auprc']:.4f}  "
                  f"BAcc={m['bacc']:.4f}  MCC={m.get('mcc',0):.4f}  "
                  f"Sens={m['sens']:.4f}  Spec={m['spec']:.4f}")

    with open(save_dir/"metrics.json","w") as f: json.dump(metrics, f, indent=2)
    with open(save_dir/"history.json","w") as f: json.dump(history, f)
    _plot_training_curves(history, save_dir/"plots", tag=mod_name)
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


def _load_p1_encoder(p1_dir: Path, mod: str,
                     trainable: bool = True,
                     use_spatial: bool = False):
    ckpt_path = p1_dir / mod / "best_model.pt"
    if not ckpt_path.exists():
        # Fallback: pick best available checkpoint using stored val_bacc history.
        # This allows Phase 2 to run even when Phase 1 was interrupted before
        # the final best_model.pt was written (e.g. job killed mid-training).
        ckpt_dir = p1_dir / mod / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("ep*.pt")) if ckpt_dir.exists() else []
        assert ckpts, f"Missing Phase 1 checkpoint: {ckpt_path}"
        best_b, best_cp = -1.0, ckpts[-1]
        for cp in ckpts:
            try:
                data = torch.load(cp, map_location="cpu", weights_only=False)
                hist = data.get("history", {}).get("val_bacc", [])
                b = max(hist) if hist else -1.0
                if b > best_b:
                    best_b, best_cp = b, cp
                del data
            except Exception:
                pass
        print(f"  [warn] {mod}/best_model.pt missing; "
              f"using {best_cp.name} (val_bacc≈{best_b:.4f})")
        ckpt_path = best_cp

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    # Detect actual feat_dim from checkpoint to handle legacy format changes
    # (e.g. Clinical was 107 raw features before switching to 408 one-hot tokens).
    w = state.get("encoder.backbone.0.weight")
    actual_feat_dim = int(w.shape[1]) if w is not None else _feat_dim(mod)
    if actual_feat_dim != _feat_dim(mod):
        print(f"  [info] {mod}: checkpoint feat_dim={actual_feat_dim} "
              f"(registry={_feat_dim(mod)}) — using checkpoint dim")
    # Reconstruct with same use_spatial flag so pos_enc buffers exist if needed
    base  = SingleModalMIL(actual_feat_dim, HIDDEN_DIM, DROPOUT, P1_CLR_PROJ_DIM,
                            use_cross_attn=False,
                            use_spatial=(use_spatial and mod == "HE"))
    base.load_state_dict(state, strict=False)   # pos_enc has no params; backbone loads fine
    enc   = base.encoder
    del state, base
    for p in enc.parameters():
        p.requires_grad = trainable
    return enc

def _load_p1_proj_head(p1_dir: Path, mod: str,
                       frozen: bool = True) -> ProjectionHead:
    ckpt_path = p1_dir / mod / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_dir = p1_dir / mod / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("ep*.pt")) if ckpt_dir.exists() else []
        assert ckpts, f"Missing Phase 1 checkpoint: {ckpt_path}"
        best_b, best_cp = -1.0, ckpts[-1]
        for cp in ckpts:
            try:
                data = torch.load(cp, map_location="cpu", weights_only=False)
                hist = data.get("history", {}).get("val_bacc", [])
                b = max(hist) if hist else -1.0
                if b > best_b:
                    best_b, best_cp = b, cp
                del data
            except Exception:
                pass
        print(f"  [warn] {mod}/best_model.pt missing; "
              f"using {best_cp.name} (val_bacc≈{best_b:.4f})")
        ckpt_path = best_cp
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    ph    = ProjectionHead(HIDDEN_DIM, P1_CLR_PROJ_DIM)
    ph_state = {k[len("proj_head."):]: v
                for k, v in state.items() if k.startswith("proj_head.")}
    if ph_state: ph.load_state_dict(ph_state)
    else: print(f"  [warn] No proj_head in checkpoint for {mod} — random init")
    del state
    for p in ph.parameters(): p.requires_grad = not frozen
    return ph


class DualTaskHead(nn.Module):
    """
    Two independent CLS tokens attending to shared summary tokens.
    Task 1 (cls):  CLS_cls  → r_cls  → cls_head  → logit
    Task 2 (surv): CLS_surv → r_surv → hazard_head → hazard

    Keeps task-specific representations separate so competing objectives
    don't directly constrain each other's pooling weights.
    """
    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cls_token_cls  = nn.Parameter(torch.zeros(1, hidden_dim))
        self.cls_token_surv = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.normal_(self.cls_token_cls,  std=0.02)
        nn.init.normal_(self.cls_token_surv, std=0.02)
        self.cls_attn  = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.surv_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.cls_norm  = nn.LayerNorm(hidden_dim)
        self.surv_norm = nn.LayerNorm(hidden_dim)
        self.cls_head     = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.hazard_head  = nn.Linear(hidden_dim, 1, bias=True)
        nn.init.normal_(self.hazard_head.weight, 0.0, 0.01)
        nn.init.zeros_(self.hazard_head.bias)

    def forward(self, tokens: torch.Tensor, device: torch.device):
        """
        tokens: (K, H) — K summary tokens from cross-modal transformer / slot attn
        Returns: (logit, hazard, r_cls, r_surv)  all scalar or (H,)
        """
        kv = tokens.unsqueeze(0)                                    # (1, K, H)
        q_cls  = self.cls_token_cls.to(device).unsqueeze(0)        # (1, 1, H)
        q_surv = self.cls_token_surv.to(device).unsqueeze(0)       # (1, 1, H)
        r_cls,  _ = self.cls_attn(q_cls,  kv, kv)
        r_surv, _ = self.surv_attn(q_surv, kv, kv)
        r_cls  = self.cls_norm(r_cls).squeeze(0).squeeze(0)        # (H,)
        r_surv = self.surv_norm(r_surv).squeeze(0).squeeze(0)      # (H,)
        logit  = self.cls_head(r_cls).squeeze()
        hazard = self.hazard_head(r_surv).squeeze()
        return logit, hazard, r_cls, r_surv


class DualGatedPool(nn.Module):
    """
    Two independent gated-attention ABMIL pools on the same token set.
    Used by early/late/middle fusion so each task pools differently
    from shared features without competing through a single bottleneck.

    cls pathway:  A_cls = softmax(w_cls · (V_cls(x) ⊙ U_cls(x)))
                  r_cls = A_cls^T x  → logit
    surv pathway: A_surv = softmax(w_surv · (V_surv(x) ⊙ U_surv(x)))
                  r_surv = A_surv^T x → hazard
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.cls_V    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.cls_U    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.cls_w    = nn.Linear(hidden_dim, 1, bias=False)
        self.cls_norm = nn.LayerNorm(hidden_dim)
        self.cls_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

        self.surv_V    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.surv_U    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.surv_w    = nn.Linear(hidden_dim, 1, bias=False)
        self.surv_norm = nn.LayerNorm(hidden_dim)
        self.hazard_head = nn.Linear(hidden_dim, 1, bias=True)
        nn.init.normal_(self.hazard_head.weight, 0.0, 0.01)
        nn.init.zeros_(self.hazard_head.bias)

    def forward(self, x: torch.Tensor, device: torch.device):
        """x: (N, H) — N tokens (patches / modality summaries / cross-modal slots)"""
        A_cls  = torch.softmax(self.cls_w(self.cls_V(x) * self.cls_U(x)),  dim=0)  # (N,1)
        r_cls  = self.cls_norm((A_cls  * x).sum(0))                                 # (H,)
        logit  = self.cls_head(r_cls).squeeze()

        A_surv = torch.softmax(self.surv_w(self.surv_V(x) * self.surv_U(x)), dim=0)
        r_surv = self.surv_norm((A_surv * x).sum(0))
        hazard = self.hazard_head(r_surv).squeeze()

        return logit, hazard, r_cls, r_surv


# ══════════════════════════════════════════════════════════════════
# PHASE 2 — VARIANT 1: EarlyFusionMIL
# ══════════════════════════════════════════════════════════════════

class EarlyFusionMIL(nn.Module):
    """All patches → concat → two separate gated-ABMIL pools (cls / surv)."""
    def __init__(self, encoders, proj_heads, hidden_dim=256, dropout=0.4,
                 modal_dropout=0.3, max_patches_per_mod=P2_MAX_PATCHES,
                 use_cls=False, proj_dim=128):
        super().__init__()
        self.encoders      = nn.ModuleDict(encoders)
        self.proj_heads    = nn.ModuleDict(proj_heads)
        self.modal_dropout = modal_dropout
        self.max_patches   = max_patches_per_mod
        self.use_cls       = use_cls
        self.task_head     = DualGatedPool(hidden_dim, dropout=dropout)

    def forward(self, bags: dict, device: torch.device):
        he_coords = bags.get("HE_coords")
        present_mods = [m for m in self.encoders if bags.get(m) is not None]
        if not present_mods:
            return torch.tensor(0.0, device=device, requires_grad=True)
        # Modal dropout — always keep at least 1 modality
        if self.training:
            kept = [m for m in present_mods if random.random() >= self.modal_dropout]
            if not kept:
                kept = [random.choice(present_mods)]
            present_mods = kept
        # Balance token budget across present modalities
        budget_per_mod = max(1, self.max_patches // len(present_mods))
        patches = []
        for mod in present_mods:
            enc = self.encoders[mod]
            t = bags[mod].to(device, non_blocking=True)
            if t.shape[0] > budget_per_mod:
                t = t[torch.randperm(t.shape[0], device=device)[:budget_per_mod]]
            crds = he_coords if mod == "HE" else None
            patches.append(enc.encode_patches(t, coords=crds))
        H_all = torch.cat(patches, dim=0)  # (N_total, H) balanced across mods
        return self.task_head(H_all, device)


# ══════════════════════════════════════════════════════════════════
# PHASE 2 — VARIANT 2: LateFusionMIL
# ══════════════════════════════════════════════════════════════════

class LateFusionMIL(nn.Module):
    """
    True late fusion: per-modality ABMIL → per-modality cls/surv heads →
    combine decisions with learnable softmax weights.
    Each modality votes independently; the combination is learned.
    """
    def __init__(self, encoders, proj_heads, hidden_dim=256, dropout=0.4,
                 modal_dropout=0.3, proj_dim=128):
        super().__init__()
        self.encoders      = nn.ModuleDict(encoders)
        self.modal_dropout = modal_dropout
        self.cls_heads  = nn.ModuleDict({
            m: nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
            for m in encoders})
        self.surv_heads = nn.ModuleDict({
            m: nn.Linear(hidden_dim, 1, bias=True)
            for m in encoders})
        for m in encoders:
            nn.init.normal_(self.surv_heads[m].weight, 0.0, 0.01)
            nn.init.zeros_(self.surv_heads[m].bias)
        self.log_weights = nn.Parameter(torch.zeros(len(encoders)))
        self.mod_index   = {m: i for i, m in enumerate(encoders)}

    def forward(self, bags: dict, device: torch.device):
        he_coords = bags.get("HE_coords")
        cls_logits: dict = {}; surv_hazards: dict = {}
        reps: dict = {};      indices: list = []
        for mod, enc in self.encoders.items():
            t = bags.get(mod)
            if t is None: continue
            if self.training and random.random() < self.modal_dropout: continue
            crds = he_coords if mod == "HE" else None
            rep, _, _ = enc(t.to(device, non_blocking=True), coords=crds)
            cls_logits[mod]   = self.cls_heads[mod](rep).squeeze()
            surv_hazards[mod] = self.surv_heads[mod](rep).squeeze()
            reps[mod]         = rep
            indices.append(self.mod_index[mod])
        if not reps:
            return torch.tensor(0.0, device=device, requires_grad=True)

        r_cls  = torch.stack(list(reps.values())).mean(0)  # (H,) for UMAP
        r_surv = r_cls

        if len(reps) == 1:
            return (list(cls_logits.values())[0],
                    list(surv_hazards.values())[0],
                    r_cls, r_surv)

        idx     = torch.tensor(indices, device=device)
        weights = F.softmax(self.log_weights[idx], dim=0)
        logit   = (weights * torch.stack(list(cls_logits.values()))).sum()
        hazard  = torch.stack(list(surv_hazards.values())).mean()
        return logit, hazard, r_cls, r_surv


# ══════════════════════════════════════════════════════════════════
# PHASE 2 — VARIANT 3: MiddleFusionMIL
# ══════════════════════════════════════════════════════════════════

class MiddleFusionMIL(nn.Module):
    """ABMIL per mod → summaries → cross-modal transformer → dual gated-ABMIL per task."""
    def __init__(self, encoders, proj_heads, hidden_dim=256, n_heads=4,
                 n_layers=2, dropout=0.1, modal_dropout=0.3, use_cls=False,
                 use_recon=False, proj_dim=128):
        super().__init__()
        self.encoders      = nn.ModuleDict(encoders)
        self.proj_heads    = nn.ModuleDict(proj_heads)
        self.modal_dropout = modal_dropout
        self.use_recon     = use_recon
        self.transformer   = nn.ModuleList([nn.ModuleDict({
            "attn": nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True),
            "norm": nn.LayerNorm(hidden_dim), "ffn": FFN(hidden_dim, dropout),
        }) for _ in range(n_layers)])
        self.task_head = DualGatedPool(hidden_dim, dropout=dropout)
        if use_recon:
            self.recon_decoders = nn.ModuleDict({
                m: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim))
                for m in encoders})

    def forward(self, bags: dict, device: torch.device):
        he_coords = bags.get("HE_coords")
        summary_dict: Dict[str, torch.Tensor] = {}
        for mod, enc in self.encoders.items():
            t = bags.get(mod)
            if t is None: continue
            if self.training and random.random() < self.modal_dropout: continue
            crds = he_coords if mod == "HE" else None
            rep, _, _ = enc(t.to(device, non_blocking=True), coords=crds)
            summary_dict[mod] = rep
        if not summary_dict:
            return torch.tensor(0.0, device=device, requires_grad=True)
        x = torch.stack(list(summary_dict.values()), dim=0).unsqueeze(0)  # (1, M, H)
        if len(summary_dict) >= 2:
            for L in self.transformer:
                a, _ = L["attn"](x, x, x)
                x    = L["ffn"](L["norm"](x + a))
        tokens = x.squeeze(0)  # (M, H)
        if self.use_recon and summary_dict:
            m = random.choice(list(summary_dict.keys()))
            self._last_recon = F.mse_loss(self.recon_decoders[m](tokens.mean(0)),
                                           summary_dict[m].detach())
        else:
            self._last_recon = None
        return self.task_head(tokens, device)


# ══════════════════════════════════════════════════════════════════
# PHASE 2 — VARIANT 4: CrossAttnFusionMIL  (patch-to-patch bidir)
# ══════════════════════════════════════════════════════════════════

class CrossAttnFusionMIL(nn.Module):
    """
    All-pairs bidirectional patch cross-attention (MCAT style).

    For each present pair (m_a, m_b):
      h_a, h_b = backbone features
      h_a_enr += MHA(Q=h_a, KV=h_b)   [shared MHA via BidirPatchCrossAttn]
      h_b_enr += MHA(Q=h_b, KV=h_a)   [same MHA weights]

    After all pairs, each modality's patches have seen context from every
    other present modality. Then iterative slot attention compresses each
    modality's enriched patches to K disentangled slots. All slots go
    through a cross-modal transformer then ABMIL or CLS.

    Guard: bidir cross-attn skipped when < 2 modalities present.
    Shared BidirPatchCrossAttn + IterativeSlotAttn — no mod-specific params.
    """
    def __init__(self, encoders, proj_heads, hidden_dim=256,
                 n_heads=4, n_cross_layers=2, dropout=0.1,
                 modal_dropout=0.3, n_slots=8, n_slot_iters=3,
                 max_patches_bidir=256, use_cls=False, use_recon=False):
        super().__init__()
        self.encoders         = nn.ModuleDict(encoders)
        self.proj_heads       = nn.ModuleDict(proj_heads)
        self.modal_dropout    = modal_dropout
        self.max_patches_bidir = max_patches_bidir
        self.use_cls           = use_cls
        self.use_recon         = use_recon

        # Shared bidir cross-attn (same weights for all pairs)
        self.bidir_cross  = BidirPatchCrossAttn(hidden_dim, n_heads, dropout)
        # Iterative slot attention (shared slot init across all modalities)
        self.slot_attn    = IterativeSlotAttn(hidden_dim, n_slots, n_slot_iters, dropout)
        # Cross-modal transformer over concatenated slots
        self.cross_xfmr   = CrossModalTransformer(hidden_dim, n_heads, dropout, n_cross_layers)
        if use_recon:
            self.recon_decoders = nn.ModuleDict({
                m: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim))
                for m in encoders})
        self.task_head = DualTaskHead(hidden_dim, n_heads=n_heads, dropout=dropout)

    def _cap(self, h: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Cap patches when used as KV to control attention matrix size."""
        if h.shape[0] > self.max_patches_bidir:
            idx = torch.randperm(h.shape[0], device=device)[:self.max_patches_bidir]
            return h[idx]
        return h

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        he_coords = bags.get("HE_coords")
        # Stage 1: backbone features per present modality
        present_h: Dict[str, torch.Tensor] = {}
        for mod, enc in self.encoders.items():
            t = bags.get(mod)
            if t is None: continue
            if self.training and random.random() < self.modal_dropout: continue
            crds = he_coords if mod == "HE" else None
            present_h[mod] = enc.encode_patches(t.to(device, non_blocking=True),
                                                coords=crds)

        if not present_h:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Stage 2: all-pairs bidir patch cross-attention
        # Each modality accumulates enrichment from all others
        if len(present_h) >= 2:
            enriched_h: Dict[str, torch.Tensor] = dict(present_h)  # shallow: no tensor copies
            mods = list(present_h.keys())
            for i, m_a in enumerate(mods):
                for m_b in mods[i+1:]:   # each pair once; bidir handles both directions
                    h_a      = enriched_h[m_a]
                    h_b_orig = enriched_h[m_b]
                    h_b_cap  = self._cap(h_b_orig, device)
                    h_a_enr, h_b_enr, _ = self.bidir_cross(h_a, h_b_cap)
                    enriched_h[m_a] = h_a_enr
                    # Only write back m_b if no capping was applied;
                    # otherwise the reduced patch count would corrupt
                    # subsequent pairs involving m_b as the query side.
                    if h_b_cap.shape[0] == h_b_orig.shape[0]:
                        enriched_h[m_b] = h_b_enr
        else:
            enriched_h = present_h

        # Stage 3: iterative slot attention per modality
        slot_dict: Dict[str, torch.Tensor] = {}
        for mod, h_enr in enriched_h.items():
            slot_dict[mod] = self.slot_attn(h_enr)   # (K, H)

        # Stage 4: cross-modal transformer over all slots (skip if only 1 modality)
        slot_list  = list(slot_dict.values())
        all_slots  = torch.cat(slot_list, dim=0).unsqueeze(0)   # (1, K*n, H)
        if len(slot_list) >= 2:
            all_slots = self.cross_xfmr(all_slots)
        all_slots = all_slots.squeeze(0)                          # (K*n, H)

        # Stage 5: dual task head on all slots
        if self.use_recon and slot_dict:
            m = random.choice(list(slot_dict.keys()))
            self._last_recon = F.mse_loss(self.recon_decoders[m](all_slots.mean(0)),
                                           slot_dict[m].mean(0).detach())
        else:
            self._last_recon = None
        return self.task_head(all_slots, device)


# ══════════════════════════════════════════════════════════════════
# PHASE 2 — VARIANT 5: SlotCrossModalMIL  (iterative slots, no bidir)
# ══════════════════════════════════════════════════════════════════

class SlotCrossModalMIL(nn.Module):
    """
    Each bag -> backbone -> IterativeSlotAttn (K slots, T=3 rounds).
    All slots -> cross-modal transformer -> ABMIL or CLS.
    No type embeddings. Shared slot init across modalities.
    """
    def __init__(self, encoders, proj_heads, hidden_dim=256, n_slots=8,
                 n_slot_iters=3, n_heads=4, n_cross_layers=2, dropout=0.1,
                 modal_dropout=0.3, use_cls=False, use_recon=False, proj_dim=128):
        super().__init__()
        self.encoders      = nn.ModuleDict(encoders)
        self.proj_heads    = nn.ModuleDict(proj_heads)
        self.modal_dropout = modal_dropout
        self.use_cls       = use_cls
        self.use_recon     = use_recon

        self.slot_attn  = IterativeSlotAttn(hidden_dim, n_slots, n_slot_iters, dropout)
        self.cross_xfmr = CrossModalTransformer(hidden_dim, n_heads, dropout, n_cross_layers)
        if use_recon:
            self.recon_decoders = nn.ModuleDict({
                m: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim))
                for m in encoders})
        self.task_head = DualTaskHead(hidden_dim, n_heads=n_heads, dropout=dropout)

    def forward(self, bags: dict, device: torch.device):
        he_coords = bags.get("HE_coords")
        slot_dict: Dict[str, torch.Tensor] = {}
        for mod, enc in self.encoders.items():
            t = bags.get(mod)
            if t is None: continue
            if self.training and random.random() < self.modal_dropout: continue
            crds = he_coords if mod == "HE" else None
            h = enc.encode_patches(t.to(device, non_blocking=True), coords=crds)
            slot_dict[mod] = self.slot_attn(h)   # (K, H)
        if not slot_dict:
            return torch.tensor(0.0, device=device, requires_grad=True)
        slot_list = list(slot_dict.values())
        all_slots = torch.cat(slot_list, dim=0).unsqueeze(0)
        if len(slot_list) >= 2:
            all_slots = self.cross_xfmr(all_slots)
        all_slots = all_slots.squeeze(0)
        if self.use_recon and slot_dict:
            m = random.choice(list(slot_dict.keys()))
            self._last_recon = F.mse_loss(self.recon_decoders[m](all_slots.mean(0)),
                                           slot_dict[m].mean(0).detach())
        else:
            self._last_recon = None
        return self.task_head(all_slots, device)


# ══════════════════════════════════════════════════════════════════
# PHASE 2 — VARIANT 6: IterativeXModalMIL  (patch-level iterative)
# ══════════════════════════════════════════════════════════════════

class IterativeXModalMIL(nn.Module):
    """
    R × iterative blocks at patch level:
      Stage 2a: within-modal self-attention     (always)
      Stage 2b: cross-modal cross-attention     (guard: skip if n_present < 2)
                Q = own patches, KV = concat(all other present patches)
                Symmetric all-pairs — shared CA weights
    Stage 3: IterativeSlotAttn (K slots, T=3 rounds, shared init)
    Stage 4: CrossModalTransformer
    Stage 5: ABMIL or CLS → head

    No type embeddings. HE capped at max_he_patches.
    """
    def __init__(self, encoders, proj_heads, hidden_dim=256,
                 n_iter_blocks=2, n_slots=8, n_slot_iters=3, n_heads=4,
                 n_cross_layers=2, dropout=0.1, modal_dropout=0.3,
                 max_he_patches=P2_MAX_HE_BLOCK,
                 use_cls=False, use_grad_ckpt=False, use_recon=False):
        super().__init__()
        self.encoders       = nn.ModuleDict(encoders)
        self.proj_heads     = nn.ModuleDict(proj_heads)
        self.modal_dropout  = modal_dropout
        self.n_iter_blocks  = n_iter_blocks
        self.max_he_patches = max_he_patches
        self.use_cls        = use_cls
        self.use_grad_ckpt  = use_grad_ckpt

        self.self_attn_blocks = nn.ModuleList([nn.ModuleDict({
            "attn": nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True),
            "norm": nn.LayerNorm(hidden_dim), "ffn": FFN(hidden_dim, dropout),
        }) for _ in range(n_iter_blocks)])

        self.cross_attn_blocks = nn.ModuleList([nn.ModuleDict({
            "attn": nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True),
            "norm": nn.LayerNorm(hidden_dim), "ffn": FFN(hidden_dim, dropout),
        }) for _ in range(n_iter_blocks)])

        # Iterative slot attention replaces single-round slot compression
        self.slot_attn  = IterativeSlotAttn(hidden_dim, n_slots, n_slot_iters, dropout)
        self.cross_xfmr = CrossModalTransformer(hidden_dim, n_heads, dropout, n_cross_layers)
        self.use_recon  = use_recon
        if use_recon:
            self.recon_decoders = nn.ModuleDict({
                m: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim))
                for m in encoders})
        self.task_head = DualTaskHead(hidden_dim, n_heads=n_heads, dropout=dropout)

    def _iter_block(self, r, h_dict):
        SA = self.self_attn_blocks[r]
        CA = self.cross_attn_blocks[r]
        h_self = {}
        for mod, h in h_dict.items():
            x    = h.unsqueeze(0)
            a, _ = SA["attn"](x, x, x)
            h_self[mod] = SA["ffn"](SA["norm"](x + a)).squeeze(0)
        if len(h_self) < 2:
            return h_self
        h_cross = {}
        for mod, h in h_self.items():
            others = torch.cat([v for k, v in h_self.items() if k != mod], dim=0).unsqueeze(0)
            q    = h.unsqueeze(0)
            a, _ = CA["attn"](q, others, others)
            h_cross[mod] = CA["ffn"](CA["norm"](q + a)).squeeze(0)
        return h_cross

    def _checkpointed_block(self, r, h_dict):
        mods    = list(h_dict.keys())
        tensors = tuple(h_dict[m] for m in mods)
        def fn(*args):
            hd = {mods[i]: args[i] for i in range(len(mods))}
            out = self._iter_block(r, hd)
            return tuple(out[m] for m in mods)
        result = grad_ckpt_utils.checkpoint(fn, *tensors, use_reentrant=False)
        if not isinstance(result, tuple): result = (result,)
        return {mods[i]: result[i] for i in range(len(mods))}

    def forward(self, bags: dict, device: torch.device) -> torch.Tensor:
        he_coords = bags.get("HE_coords")
        h_dict: Dict[str, torch.Tensor] = {}
        for mod, enc in self.encoders.items():
            t = bags.get(mod)
            if t is None: continue
            if self.training and random.random() < self.modal_dropout: continue
            t = t.to(device, non_blocking=True)
            if mod == "HE" and t.shape[0] > self.max_he_patches:
                t = t[torch.randperm(t.shape[0], device=device)[:self.max_he_patches]]
            crds = he_coords if mod == "HE" else None
            h_dict[mod] = enc.encode_patches(t, coords=crds)

        if not h_dict:
            return torch.tensor(0.0, device=device, requires_grad=True)

        for r in range(self.n_iter_blocks):
            if self.use_grad_ckpt:
                h_dict = self._checkpointed_block(r, h_dict)
            else:
                h_dict = self._iter_block(r, h_dict)

        # Iterative slot attention (T=3 rounds, slot competition)
        slot_dict: Dict[str, torch.Tensor] = {}
        for mod, h in h_dict.items():
            slot_dict[mod] = self.slot_attn(h)   # (K, H)

        slot_list = list(slot_dict.values())
        all_slots = torch.cat(slot_list, dim=0).unsqueeze(0)
        if len(slot_list) >= 2:
            all_slots = self.cross_xfmr(all_slots)
        all_slots = all_slots.squeeze(0)
        if self.use_recon and slot_dict:
            m = random.choice(list(slot_dict.keys()))
            self._last_recon = F.mse_loss(self.recon_decoders[m](all_slots.mean(0)),
                                           slot_dict[m].mean(0).detach())
        else:
            self._last_recon = None
        return self.task_head(all_slots, device)


# ══════════════════════════════════════════════════════════════════
# PHASE 2 — MODEL FACTORY
# ══════════════════════════════════════════════════════════════════

def build_p2_model(variant: str, p1_dir: Path,
                   modal_dropout: float = P2_MODAL_DROPOUT,
                   iter_r: int = P2_ITER_R,
                   slot_k: int = P2_SLOT_K,
                   n_slot_iters: int = 3,
                   use_grad_ckpt: bool = False,
                   use_recon: bool = False,
                   use_he_spatial: bool = False) -> nn.Module:
    def _p1_ckpt_available(mod: str) -> bool:
        if (p1_dir / mod / "best_model.pt").exists():
            return True
        ckpt_dir = p1_dir / mod / "checkpoints"
        return ckpt_dir.exists() and bool(list(ckpt_dir.glob("ep*.pt")))

    avail_mods = [m for m in MODALITIES if _p1_ckpt_available(m)]
    if not avail_mods:
        raise RuntimeError(f"No Phase 1 checkpoints found in {p1_dir}")
    missing = [m for m in MODALITIES if m not in avail_mods]
    if missing:
        print(f"  [p2:{variant}] Missing modalities (no P1 ckpt): {missing} — running without them")
    encoders   = {m: _load_p1_encoder(p1_dir, m, trainable=True,
                                       use_spatial=(use_he_spatial and m == "HE"))
                  for m in avail_mods}
    proj_heads = {m: _load_p1_proj_head(p1_dir, m, frozen=True)  for m in avail_mods}
    n_enc = sum(sum(p.numel() for p in e.parameters()) for e in encoders.values())
    spatial_str = "  HE=2D-sinPE" if use_he_spatial else ""
    print(f"  [p2:{variant}] encoders={n_enc:,} (trainable)  "
          f"proj_heads=frozen  slot_iters={n_slot_iters}  use_recon={use_recon}"
          f"  modalities={avail_mods}{spatial_str}")

    base    = variant.replace("_cls", "")
    use_cls = variant.endswith("_cls")

    if base == "early":
        return EarlyFusionMIL(encoders, proj_heads, HIDDEN_DIM, DROPOUT,
                               modal_dropout, P2_MAX_PATCHES, use_cls)
    elif base == "late":
        return LateFusionMIL(encoders, proj_heads, HIDDEN_DIM, DROPOUT, modal_dropout)
    elif base == "middle":
        return MiddleFusionMIL(encoders, proj_heads, HIDDEN_DIM, P2_N_HEADS,
                                P2_N_CROSS_LAYERS, P2_ATTN_DROPOUT, modal_dropout,
                                use_cls, use_recon=use_recon)
    elif base == "crossattn":
        return CrossAttnFusionMIL(encoders, proj_heads, HIDDEN_DIM,
                                   P2_N_HEADS, P2_N_CROSS_LAYERS, P2_ATTN_DROPOUT,
                                   modal_dropout, slot_k, n_slot_iters,
                                   max_patches_bidir=256, use_cls=use_cls,
                                   use_recon=use_recon)
    elif base == "crossmodal":
        return SlotCrossModalMIL(encoders, proj_heads, HIDDEN_DIM, slot_k,
                                  n_slot_iters, P2_N_HEADS, P2_N_CROSS_LAYERS,
                                  P2_ATTN_DROPOUT, modal_dropout, use_cls,
                                  use_recon=use_recon)
    elif base == "iterative":
        return IterativeXModalMIL(
            encoders=encoders, proj_heads=proj_heads, hidden_dim=HIDDEN_DIM,
            n_iter_blocks=iter_r, n_slots=slot_k, n_slot_iters=n_slot_iters,
            n_heads=P2_N_HEADS, n_cross_layers=P2_N_CROSS_LAYERS,
            dropout=P2_ATTN_DROPOUT, modal_dropout=modal_dropout,
            max_he_patches=P2_MAX_HE_BLOCK,
            use_cls=use_cls, use_grad_ckpt=use_grad_ckpt, use_recon=use_recon)
    else:
        raise ValueError(f"Unknown p2_variant: {variant!r}.  Choose from: {P2_VARIANTS}")

def p2_train_one_epoch(model, records, optimizer, cw, device, bag_cache,
                       scaler, grad_accum,
                       use_contrastive=False,
                       clr_tau=P1_CLR_TAU, clr_lambda=P1_CLR_LAMBDA,
                       recon_lambda=0.0,
                       cox_lambda=0.0, surv_endpoint='clad',
                       task='acr'):
    model.train()
    random.shuffle(records)
    total_loss = 0.0; n = 0
    accum_step = 0; grad_accumulated = False
    batch_buffer: List[Tuple[torch.Tensor, int, str, str]] = []
    cox_buffer: list = []
    optimizer.zero_grad()

    has_enc  = hasattr(model, "encoders")
    has_proj = hasattr(model, "proj_heads")

    # Reuse a single bags dict every iteration — avoids creating a new dict
    # object per sample (important when Python GC is unreliable).
    bags: dict = {m: None for m in MODALITIES}
    bags["HE_coords"] = None

    # Per-bag OOM counter — bags that OOM this many times in one epoch are
    # permanently skipped for the rest of the epoch to prevent infinite loops.
    OOM_SKIP_THRESHOLD = 3
    oom_per_bag: dict = {}

    for rec in records:
        # Skip bags that repeatedly OOM this epoch
        if oom_per_bag.get(rec["stem"], 0) >= OOM_SKIP_THRESHOLD:
            continue
        # Refill in-place from the pre-loaded CPU cache
        entry = bag_cache.get(rec["stem"], {})
        for m in MODALITIES:
            bags[m] = entry.get(m)
        bags["HE_coords"] = entry.get("HE_coords")
        if all(bags.get(m) is None for m in MODALITIES): continue

        target = torch.tensor([rec["label"]], dtype=torch.float32, device=device)
        surv_time_key  = f"{surv_endpoint}_time"
        surv_event_key = f"{surv_endpoint}_event"
        surv_t = rec.get(surv_time_key, float("nan"))
        has_surv_data = (isinstance(surv_t, float) and not math.isnan(surv_t))

        try:
            # Contrastive reps — before main forward so graph is alive through encoders
            if use_contrastive and has_enc and has_proj:
                for mod, enc in model.encoders.items():
                    bag = bags.get(mod)
                    if bag is None: continue
                    bag_dev = bag.to(device, non_blocking=True)
                    crds = bags.get("HE_coords") if mod == "HE" else None
                    rep, _, _ = enc(bag_dev, coords=crds)
                    pz = model.proj_heads[mod](rep)
                    batch_buffer.append((pz, rec["label"], rec["stem"], mod))

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                result = model(bags, device)
                if isinstance(result, tuple) and len(result) == 3:
                    logit, L_recon, hazard = result
                elif isinstance(result, tuple) and len(result) == 2:
                    logit, L_recon = result; hazard = None
                else:
                    logit = result; L_recon = None; hazard = None
                if not isinstance(logit, torch.Tensor) or logit.grad_fn is None:
                    continue

            # ── ACR path: hinge loss + optional Cox auxiliary ─────────────
            if task == 'acr':
                with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                    loss = hinge_loss(logit.unsqueeze(0), target, cw) / grad_accum
                    if L_recon is not None and recon_lambda > 0:
                        loss = loss + recon_lambda * L_recon / grad_accum

                has_surv = hazard is not None and cox_lambda > 0 and has_surv_data
                if scaler is not None:
                    scaler.scale(loss).backward(retain_graph=has_surv)
                else:
                    loss.backward(retain_graph=has_surv)

                if has_surv:
                    cox_buffer.append((hazard.float(), rec[surv_time_key], rec[surv_event_key]))

                if torch.cuda.is_available(): torch.cuda.empty_cache()
                grad_accumulated = True
                total_loss += loss.item() * grad_accum
                n += 1; accum_step += 1

            # ── Survival path: Cox loss only, accumulated per batch ────────
            else:  # task == 'survival'
                if hazard is not None and has_surv_data:
                    cox_buffer.append((hazard.float(), rec[surv_time_key], rec[surv_event_key]))
                grad_accumulated = True
                n += 1; accum_step += 1

            if accum_step == grad_accum:
                _did_backward = False
                # CLR (ACR mode only — labels needed)
                if task == 'acr' and use_contrastive and batch_buffer:
                    L_clr = batch_supcon_loss(batch_buffer, clr_tau, cw)
                    if L_clr is not None:
                        s = scaler.scale(L_clr * clr_lambda) if scaler else L_clr * clr_lambda
                        s.backward()
                        _did_backward = True
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                # Cox flush
                eff_cox_lambda = cox_lambda if task == 'acr' else 1.0
                if cox_buffer and eff_cox_lambda > 0:
                    L_cox = cox_breslow_loss(cox_buffer)
                    if L_cox is not None and L_cox.requires_grad:
                        if scaler is not None:
                            scaler.scale(L_cox * eff_cox_lambda).backward()
                        else:
                            (L_cox * eff_cox_lambda).backward()
                        _did_backward = True
                        if task == 'survival':
                            total_loss += L_cox.item()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                cox_buffer.clear()
                if _did_backward:
                    if scaler is not None:
                        scaler.step(optimizer); scaler.update()
                    else:
                        optimizer.step()
                optimizer.zero_grad()
                batch_buffer.clear()
                accum_step = 0; grad_accumulated = False
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            batch_buffer.clear()
            cox_buffer.clear()
            accum_step = 0
            grad_accumulated = False
            oom_per_bag[rec["stem"]] = oom_per_bag.get(rec["stem"], 0) + 1
            count = oom_per_bag[rec["stem"]]
            if count >= OOM_SKIP_THRESHOLD:
                print(f"  [OOM-p2] {rec['stem']} hit OOM {count}× — permanently skipping for this epoch", flush=True)
            else:
                print(f"  [OOM-p2] skipped {rec['stem']} ({count}/{OOM_SKIP_THRESHOLD}) — cache cleared", flush=True)

    if accum_step > 0 and grad_accumulated:
        _did_backward = False
        if task == 'acr' and use_contrastive and batch_buffer:
            L_clr = batch_supcon_loss(batch_buffer, clr_tau, cw)
            if L_clr is not None:
                s = scaler.scale(L_clr * clr_lambda) if scaler else L_clr * clr_lambda
                s.backward()
                _did_backward = True
        eff_cox_lambda = cox_lambda if task == 'acr' else 1.0
        if cox_buffer and eff_cox_lambda > 0:
            L_cox = cox_breslow_loss(cox_buffer)
            if L_cox is not None and L_cox.requires_grad:
                if scaler is not None:
                    scaler.scale(L_cox * eff_cox_lambda).backward()
                else:
                    (L_cox * eff_cox_lambda).backward()
                _did_backward = True
                if task == 'survival':
                    total_loss += L_cox.item()
        cox_buffer.clear()
        if _did_backward:
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(n, 1)


@torch.no_grad()
def p2_evaluate(model, records, device, bag_cache, cw=None,
                surv_endpoint='clad', task='acr'):
    """Single-pass eval.
    ACR mode:      returns (probs, labels, val_loss, ci, [], [], [])
    Survival mode: returns (probs, labels, val_loss, ci, hazards, times, events)
      where hazards/times/events are parallel lists for samples with valid survival data.
    """
    model.eval(); probs, labels, losses = [], [], []
    hazard_list, surv_times, surv_events = [], [], []
    use_amp = (device.type == "cuda")
    for rec in records:
        bags = {m: bag_cache.get(rec["stem"], {}).get(m) for m in MODALITIES}
        bags["HE_coords"] = bag_cache.get(rec["stem"], {}).get("HE_coords")
        if all(bags.get(m) is None for m in MODALITIES): continue
        try:
            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model(bags, device)
            if isinstance(result, tuple) and len(result) == 3:
                logit, _, hazard = result
            elif isinstance(result, tuple):
                logit = result[0]; hazard = None
            else:
                logit = result; hazard = None
            probs.append(torch.sigmoid(logit.float()).item())
            labels.append(rec["label"])
            if cw is not None and task == 'acr':
                ta = logit.new_tensor([rec["label"]])
                losses.append(hinge_loss(logit.unsqueeze(0), ta, cw).item())
            if hazard is not None:
                t_val = rec.get(f"{surv_endpoint}_time", float("nan"))
                e_val = rec.get(f"{surv_endpoint}_event", float("nan"))
                if not (isinstance(t_val, float) and math.isnan(t_val)) and t_val > 0:
                    hazard_list.append(hazard.float().item())
                    surv_times.append(float(t_val))
                    surv_events.append(float(e_val) if not math.isnan(float(e_val)) else 0.0)
            del logit
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [OOM-eval] skipped {rec['stem']} — cache cleared", flush=True)
    ci = None
    if len(hazard_list) >= 2 and sum(surv_events) > 0:
        ci = c_index(hazard_list, surv_times, surv_events)
    if task == 'survival':
        cox_buf = [(torch.tensor(h), t, e)
                   for h, t, e in zip(hazard_list, surv_times, surv_events)]
        cox_l = cox_breslow_loss(cox_buf)
        val_loss = cox_l.item() if cox_l is not None else 0.0
        return np.array(probs), np.array(labels), val_loss, ci, hazard_list, surv_times, surv_events
    val_loss = float(np.mean(losses)) if losses else 0.0
    if cw is not None:
        return np.array(probs), np.array(labels), val_loss, ci, [], [], []
    return np.array(probs), np.array(labels), None, ci, [], [], []


# ══════════════════════════════════════════════════════════════════
# PHASE 2 — RUNNER
# ══════════════════════════════════════════════════════════════════

def run_phase2_variant(model: nn.Module, variant: str, fold: int,
                       device: torch.device, bag_cache: BagCache,
                       train_recs: List[dict], val_recs: List[dict],
                       test_recs: List[dict], save_dir: Path,
                       tag: Optional[str] = None,
                       use_contrastive: bool = False,
                       clr_lambda: float = P1_CLR_LAMBDA,
                       clr_tau: float = P1_CLR_TAU,
                       recon_lambda: float = 0.0,
                       patience: int = 0,
                       cox_lambda: float = 0.0,
                       surv_endpoint: str = 'clad',
                       task: str = 'acr') -> dict:
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
              f"(best_ep={st.get('best_epoch')}  "
              f"best_bacc={st.get('best_bacc',0):.4f}). Skipping.")
        mf = save_dir / f"metrics_{vtag}.json"
        if mf.exists():
            with open(mf) as f: return json.load(f)
        return {}

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_fr = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable={n_tr:,}  Frozen={n_fr:,}  task={task}")
    if use_contrastive and task == 'acr': print(f"  Contrastive: λ={clr_lambda}  τ={clr_tau}")
    if recon_lambda > 0: print(f"  Recon: λ={recon_lambda}")
    if task == 'survival':
        print(f"  Survival Cox: endpoint={surv_endpoint}")
    elif cox_lambda > 0:
        print(f"  Cox aux: λ={cox_lambda}  endpoint={surv_endpoint}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=P2_LR, weight_decay=P2_WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    cw     = compute_class_weights(train_recs)
    if task == 'acr':
        print(f"  cw=(neg={cw[0]:.3f}, pos={cw[1]:.3f})  modal_dropout={P2_MODAL_DROPOUT}")
    else:
        print(f"  modal_dropout={P2_MODAL_DROPOUT}")

    history = {k: [] for k in
               ["train_loss","val_loss","val_auc","val_bacc","val_mcc"]}

    resume_epoch = _find_resume_epoch(ckpt_dir); start_epoch = 0
    if resume_epoch >= P2_EPOCHS:
        print(f"  [{vtag}] Already complete. Rescanning.")
        start_epoch = P2_EPOCHS
    elif resume_epoch > 0:
        ckpt = _load_checkpoint(ckpt_dir, resume_epoch)
        if ckpt is not None:
            model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) else ckpt, strict=False)
            if isinstance(ckpt, dict) and "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if isinstance(ckpt, dict) and scaler and ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            if isinstance(ckpt, dict) and "history" in ckpt:
                history = ckpt["history"]
            model.to(device); start_epoch = resume_epoch

    # Inline best tracking (metric = val_bacc for ACR, val_cidx stored in val_bacc for survival)
    _p2_best_metric: float = max(history["val_bacc"]) if history["val_bacc"] else -1.0
    _p2_best_ep:   int   = 0
    _p2_no_improve: int  = 0

    for epoch in range(start_epoch, P2_EPOCHS):
        tl = p2_train_one_epoch(
            model, train_recs, optimizer, cw, device, bag_cache,
            scaler, P2_GRAD_ACCUM,
            use_contrastive=(use_contrastive and task == 'acr'),
            clr_tau=clr_tau, clr_lambda=clr_lambda,
            recon_lambda=recon_lambda,
            cox_lambda=cox_lambda, surv_endpoint=surv_endpoint,
            task=task)
        history["train_loss"].append(tl); _gc()

        if (epoch + 1) % P2_EVAL_EVERY == 0:
            vl_p, vl_l, val_loss, ci, *_ = p2_evaluate(
                model, val_recs, device, bag_cache,
                cw=(cw if task == 'acr' else None),
                surv_endpoint=surv_endpoint, task=task)
            model.train()

            if task == 'survival':
                primary_metric = ci if ci is not None else 0.0
                history["val_loss"].append(val_loss)
                history["val_auc"].append(primary_metric)
                history["val_bacc"].append(primary_metric)
                history["val_mcc"].append(0.0)
                metric_str = f"cidx={primary_metric:.3f}  cox_loss={val_loss:.4f}"
            else:
                vm = compute_metrics(vl_l, vl_p)
                primary_metric = vm["bacc"]
                history["val_loss"].append(val_loss)
                history["val_auc"].append(vm["auc"])
                history["val_bacc"].append(vm["bacc"])
                history["val_mcc"].append(vm.get("mcc", 0.0))
                ci_str = f"  C-idx={ci:.3f}" if ci is not None else ""
                metric_str = f"auc={vm['auc']:.3f}  bacc={vm['bacc']:.3f}{ci_str}"

            torch.save({
                "epoch": epoch+1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "history": history,
            }, ckpt_dir / f"ep{epoch+1:04d}.pt")

            if primary_metric > _p2_best_metric:
                _p2_best_metric = primary_metric
                _p2_best_ep     = epoch + 1
                _p2_no_improve  = 0
                torch.save(model.state_dict(), save_dir / f"model_{vtag}.pt")
                ckpt_tag = "[ckpt*]"
            else:
                _p2_no_improve += 1
                ckpt_tag = "[ckpt]"
            improve_str = (f"  no_improve={_p2_no_improve}/{patience}"
                           if patience > 0 else "")
            print(f"  [{vtag}] ep {epoch+1:3d}  loss={tl:.4f}/{val_loss:.4f}  "
                  f"{metric_str}  {ckpt_tag}{improve_str}")
            _gc()
            if patience > 0 and _p2_no_improve >= patience:
                print(f"  [{vtag}] Early stop: {_p2_no_improve} eval periods "
                      f"without improvement (patience={patience}).")
                break
        elif (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{vtag}] ep {epoch+1:3d}  train_loss={tl:.4f}")

    # ── Finalise best model ────────────────────────────────────────
    ckpts = sorted(ckpt_dir.glob("ep*.pt"))
    if not ckpts and not (save_dir / f"model_{vtag}.pt").exists():
        print(f"  [{vtag}] [warn] no checkpoints."); return {}

    best_model_path = save_dir / f"model_{vtag}.pt"
    if best_model_path.exists() and _p2_best_ep > 0:
        print(f"\n  [{vtag}] Using inline best "
              f"(ep={_p2_best_ep}  metric={_p2_best_metric:.4f})")
        state = torch.load(best_model_path, map_location="cpu", weights_only=False)
        state = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state, strict=False); model.to(device); del state
        best_ep, best_bacc = _p2_best_ep, _p2_best_metric
    elif ckpts:
        print(f"\n  [{vtag}] Fast-scanning {len(ckpts)} checkpoint histories ...")
        best_bacc, best_ep, best_path = -1.0, 0, ckpts[-1]
        for cp in ckpts:
            try:
                data = torch.load(cp, map_location="cpu", weights_only=False)
                hist_b = data.get("history", {}).get("val_bacc", [])
                b = max(hist_b) if hist_b else -1.0
                if b > best_bacc:
                    best_bacc, best_ep, best_path = b, int(cp.stem[2:]), cp
                del data
            except Exception: pass
        print(f"  [{vtag}] best ep≈{best_ep}  metric={best_bacc:.4f}")
        data  = torch.load(best_path, map_location="cpu", weights_only=False)
        state = data["model"] if isinstance(data, dict) else data
        model.load_state_dict(state, strict=False); model.to(device); del data, state
        torch.save(model.state_dict(), best_model_path)
    else:
        best_ep, best_bacc = 0, 0.0

    _write_status(status_path, completed=True,
                  best_epoch=best_ep, best_bacc=round(best_bacc, 4),
                  last_epoch=_p2_best_ep or P2_EPOCHS)

    all_metrics: dict = {}
    for sn, recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        p, l, _, ci, h_list, t_list, e_list = p2_evaluate(model, recs, device, bag_cache,
                                                            surv_endpoint=surv_endpoint, task=task)
        if task == 'survival':
            all_metrics[sn] = {
                "c_index": ci or 0.0,
                "probs":   h_list,   # actual hazard scores (not sigmoid logit)
                "labels":  e_list,   # event flags
                "times":   t_list,   # survival times
            }
            print(f"  [{vtag}] {sn:5s}  C-index={ci:.4f}" if ci else
                  f"  [{vtag}] {sn:5s}  C-index=N/A")
        else:
            m = compute_metrics(l, p)
            m["auprc"] = average_precision_score(l, p) if len(np.unique(l)) > 1 else 0.0
            if ci is not None:
                m["c_index"] = ci
            all_metrics[sn] = {**m, "probs": p.tolist(), "labels": l.tolist()}
            ci_str = f"  C-idx={ci:.4f}" if ci is not None else ""
            print(f"  [{vtag}] {sn:5s}  AUC={m['auc']:.4f}  AUPRC={m['auprc']:.4f}  "
                  f"BAcc={m['bacc']:.4f}  MCC={m.get('mcc',0):.4f}  "
                  f"Sens={m['sens']:.4f}  Spec={m['spec']:.4f}{ci_str}")

    with open(save_dir / f"metrics_{vtag}.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(save_dir / f"history_{vtag}.json", "w") as f:
        json.dump(history, f)
    _plot_training_curves(history, save_dir / "plots", tag=vtag)
    if task == 'acr':
        _plot_roc_pr_confusion(all_metrics, save_dir / "plots", tag=vtag)
    del model, optimizer, scaler; _gc()
    return all_metrics


# ══════════════════════════════════════════════════════════════════
# FOLD RUNNER
# ══════════════════════════════════════════════════════════════════


def run_fold(fold: int, split: Optional[int] = None, phase=None,
             phase1_dir=None, bag_cache: BagCache = None,
             # Phase 1
             use_cross_attn: bool = False, use_crd: bool = False,
             use_contrastive: bool = False, use_summary_clr: bool = False,
             clr_tau: float = P1_CLR_TAU, clr_lambda: float = P1_CLR_LAMBDA,
             summary_clr_lambda: float = P1_SUMMARY_CLR_LAMBDA,
             cross_attn_lambda: float = P1_CROSS_ATTN_LAMBDA,
             kd_lambda: float = P1_KD_LAMBDA, kd_tau: float = P1_KD_TAU,
             kd_top_k: int = P1_KD_TOP_K, crd_lambda: float = P1_CRD_LAMBDA,
             # CLR strategies 2+3 (both ACR and survival)
             use_aug_clr: bool = False,
             aug_clr_lambda: float = P1_AUG_CLR_LAMBDA,
             aug_subsample: float = P1_AUG_SUBSAMPLE,
             aug_min_patches: int = P1_AUG_MIN_PATCHES,
             use_label_supcon: bool = False,
             label_supcon_lambda: float = P1_LABEL_SUPCON_LAMBDA,
             # Modality selection (None → all teacher + student modalities)
             p1_teacher_mods: Optional[List[str]] = None,
             p1_student_mods: Optional[List[str]] = None,
             # Epoch and early-stopping control
             p1_epochs: int = P1_EPOCHS,
             p1_patience: int = 0,
             # Spatial 2-D PE for HE
             p1_he_spatial: bool = False,
             p2_he_spatial: bool = False,
             # Phase 2
             p2_variants: List[str] = None,
             p2_iter_r_list: List[int] = None,
             p2_slot_k_list: List[int] = None,
             p2_n_slot_iters: int = 3,
             p2_grad_ckpt: bool = False,
             p2_use_contrastive: bool = False,
             p2_clr_lambda: float = P1_CLR_LAMBDA,
             p2_clr_tau: float = P1_CLR_TAU,
             p2_use_recon: bool = False,
             p2_recon_lambda: float = 0.1,
             p2_patience: int = 0,
             p2_cox_lambda: float = 0.0,
             p2_surv_endpoint: str = "clad",
             task: str = "acr",
             ) -> dict:
    if p2_variants    is None: p2_variants    = ["iterative"]
    if p2_iter_r_list is None: p2_iter_r_list = [P2_ITER_R]
    if p2_slot_k_list is None: p2_slot_k_list = [P2_SLOT_K]

    set_seeds(SEED)
    fold_tag = f"split{split}_fold{fold}" if split is not None else f"fold_{fold}"
    fold_dir = Path(SAVE_DIR) / fold_tag; fold_dir.mkdir(parents=True, exist_ok=True)

    if task == 'survival':
        train_recs, val_recs, test_recs = build_splits_survival(
            SAMPLES_DIR, SPLITS_CSV, fold, split=split, endpoint=p2_surv_endpoint)
    else:
        train_recs, val_recs, test_recs = build_splits(
            SAMPLES_DIR, SPLITS_CSV, fold, split=split)
    for recs in (train_recs, val_recs, test_recs):
        update_presence_from_cache(recs, bag_cache)
    _print_split_stats(fold_tag, "train", train_recs)
    _print_split_stats(fold_tag, "val",   val_recs)
    _print_split_stats(fold_tag, "test",  test_recs)

    p1_dir = fold_dir / "phase1"
    p2_dir = fold_dir / "phase2"
    fold_metrics: dict = {}

    # ══ PHASE 1 ══════════════════════════════════════════════════
    if phase in (1, None):
        # Resolve which teacher/student modalities to train in this run.
        # None → all; pass a subset (e.g. ["BAL"]) to train only that modality.
        active_teachers = p1_teacher_mods if p1_teacher_mods is not None else TEACHER_MODALITIES
        active_students = p1_student_mods if p1_student_mods is not None else STUDENT_MODALITIES
        print(f"  P1 modalities → teachers={active_teachers}  students={active_students}")

        # Step 1: Train teachers — no cross-attn, no CRD
        for mod in active_teachers:
            run_phase1_modality(
                mod_name=mod, fold=fold, device=DEVICE,
                bag_cache=bag_cache, train_recs=train_recs,
                val_recs=val_recs, test_recs=test_recs,
                save_dir=p1_dir / mod,
                use_cross_attn=False, use_crd=False,
                use_contrastive=(use_contrastive and task == 'acr'),
                use_summary_clr=(use_summary_clr and task == 'acr'),
                clr_tau=clr_tau, clr_lambda=clr_lambda,
                summary_clr_lambda=summary_clr_lambda,
                use_aug_clr=use_aug_clr, aug_clr_lambda=aug_clr_lambda,
                aug_subsample=aug_subsample, aug_min_patches=aug_min_patches,
                use_label_supcon=(use_label_supcon and task == 'survival'),
                label_supcon_lambda=label_supcon_lambda,
                use_spatial=p1_he_spatial,
                n_epochs=p1_epochs, patience=p1_patience,
                task=task, surv_endpoint=p2_surv_endpoint,
            )

        # Step 2: Build teacher caches (r + proj_z + backbone h features).
        # Always load from existing teacher checkpoints — teachers may already be
        # trained in a prior run, and building caches is required for all students.
        teacher_r_cache  = None
        teacher_pz_cache = None
        teacher_h_cache  = None
        need_caches = (use_cross_attn or use_crd or use_contrastive or use_summary_clr)
        if need_caches and active_students:
            print(f"\n  ── Building teacher caches (r + proj_z + patch features) ──")
            teacher_models = {}
            for mod in TEACHER_MODALITIES:
                ckpt_p = p1_dir / mod / "best_model.pt"
                if not ckpt_p.exists():
                    print(f"  [warn] Teacher checkpoint missing: {ckpt_p} — skipping cache for {mod}")
                    continue
                m     = SingleModalMIL(_feat_dim(mod), HIDDEN_DIM, DROPOUT,
                                       P1_CLR_PROJ_DIM, use_cross_attn=False)
                state = torch.load(ckpt_p, map_location="cpu", weights_only=False)
                m.load_state_dict(state, strict=False); del state
                teacher_models[mod] = m.to(DEVICE)
            if teacher_models:
                teacher_r_cache, teacher_pz_cache, teacher_h_cache = build_teacher_caches(
                    teacher_models, train_recs, bag_cache, DEVICE)
            del teacher_models; _gc()

        # Step 3: Train students with full auxiliary suite
        for mod in active_students:
            run_phase1_modality(
                mod_name=mod, fold=fold, device=DEVICE,
                bag_cache=bag_cache, train_recs=train_recs,
                val_recs=val_recs, test_recs=test_recs,
                save_dir=p1_dir / mod,
                use_cross_attn=(use_cross_attn and task == 'acr'),
                teacher_r_cache=teacher_r_cache,
                teacher_h_cache=teacher_h_cache,
                use_crd=(use_crd and task == 'acr'),
                crd_lambda=crd_lambda,
                use_contrastive=(use_contrastive and task == 'acr'),
                teacher_pz_cache=(teacher_pz_cache if use_contrastive and task == 'acr' else None),
                use_summary_clr=(use_summary_clr and task == 'acr'),
                clr_tau=clr_tau, clr_lambda=clr_lambda,
                summary_clr_lambda=summary_clr_lambda,
                cross_attn_lambda=cross_attn_lambda,
                kd_lambda=kd_lambda, kd_tau=kd_tau, kd_top_k=kd_top_k,
                use_aug_clr=use_aug_clr, aug_clr_lambda=aug_clr_lambda,
                aug_subsample=aug_subsample, aug_min_patches=aug_min_patches,
                use_label_supcon=(use_label_supcon and task == 'survival'),
                label_supcon_lambda=label_supcon_lambda,
                use_spatial=p1_he_spatial,
                n_epochs=p1_epochs, patience=p1_patience,
                task=task, surv_endpoint=p2_surv_endpoint,
            )

        del teacher_r_cache, teacher_pz_cache, teacher_h_cache; _gc()

    # ══ PHASE 2 ══════════════════════════════════════════════════
    if phase in (2, None):
        eff_p1 = Path(phase1_dir) / fold_tag / "phase1" if phase1_dir else p1_dir

        for p2_variant in p2_variants:
            base_variant = p2_variant.replace("_cls", "")
            needs_grid   = base_variant in ("iterative", "crossmodal", "crossattn")

            if needs_grid:
                # crossattn/crossmodal don't use iter_r — only iterate slot_k
                if base_variant == "iterative":
                    configs = list(iproduct(p2_iter_r_list, p2_slot_k_list))
                else:
                    configs = [(p2_iter_r_list[0], k) for k in p2_slot_k_list]
                print(f"\n  ── P2 [{p2_variant}] grid: {len(configs)} configs ──")
                for r_val, k_val in configs:
                    vtag  = _variant_tag(p2_variant, r_val, k_val)
                    model = build_p2_model(
                        p2_variant, eff_p1, P2_MODAL_DROPOUT,
                        iter_r=r_val, slot_k=k_val,
                        n_slot_iters=p2_n_slot_iters,
                        use_grad_ckpt=p2_grad_ckpt,
                        use_recon=p2_use_recon,
                        use_he_spatial=p2_he_spatial,
                    ).to(DEVICE)
                    fold_metrics[vtag] = run_phase2_variant(
                        model=model, variant=p2_variant, fold=fold,
                        device=DEVICE, bag_cache=bag_cache,
                        train_recs=train_recs, val_recs=val_recs,
                        test_recs=test_recs, save_dir=p2_dir, tag=vtag,
                        use_contrastive=p2_use_contrastive,
                        clr_lambda=p2_clr_lambda, clr_tau=p2_clr_tau,
                        recon_lambda=p2_recon_lambda if p2_use_recon else 0.0,
                        patience=p2_patience,
                        cox_lambda=p2_cox_lambda,
                        surv_endpoint=p2_surv_endpoint,
                        task=task)
            else:
                vtag  = p2_variant
                model = build_p2_model(
                    p2_variant, eff_p1, P2_MODAL_DROPOUT,
                    slot_k=p2_slot_k_list[0],
                    n_slot_iters=p2_n_slot_iters,
                    use_recon=p2_use_recon,
                    use_he_spatial=p2_he_spatial,
                ).to(DEVICE)
                fold_metrics[vtag] = run_phase2_variant(
                    model=model, variant=p2_variant, fold=fold,
                    device=DEVICE, bag_cache=bag_cache,
                    train_recs=train_recs, val_recs=val_recs,
                    test_recs=test_recs, save_dir=p2_dir, tag=vtag,
                    use_contrastive=p2_use_contrastive,
                    clr_lambda=p2_clr_lambda, clr_tau=p2_clr_tau,
                    recon_lambda=p2_recon_lambda if p2_use_recon else 0.0,
                    patience=p2_patience,
                    cox_lambda=p2_cox_lambda,
                    surv_endpoint=p2_surv_endpoint,
                    task=task)

    return fold_metrics

# ══════════════════════════════════════════════════════════════════
# V7 — END-TO-END MULTITASK (no Phase 1)
# ══════════════════════════════════════════════════════════════════

def build_model_v7(variant, modal_dropout=P2_MODAL_DROPOUT, iter_r=P2_ITER_R,
                   slot_k=P2_SLOT_K, n_slot_iters=3):
    """Fresh encoders from random init — no Phase 1 checkpoint required."""
    encoders   = {m: GatedAttentionEncoder(_feat_dim(m), HIDDEN_DIM, DROPOUT)
                  for m in MODALITIES}
    proj_heads = {}   # unused; kept for API compat with model constructors
    base    = variant.replace("_cls", "")
    use_cls = variant.endswith("_cls")
    kw = dict(hidden_dim=HIDDEN_DIM, dropout=DROPOUT, modal_dropout=modal_dropout)
    if base == "early":
        return EarlyFusionMIL(encoders, proj_heads, use_cls=use_cls, **kw,
                               max_patches_per_mod=P2_MAX_PATCHES)
    if base == "late":
        return LateFusionMIL(encoders, proj_heads, **kw)
    if base == "middle":
        return MiddleFusionMIL(encoders, proj_heads, n_heads=P2_N_HEADS,
                                n_layers=P2_N_CROSS_LAYERS, dropout=P2_ATTN_DROPOUT,
                                modal_dropout=modal_dropout, use_cls=use_cls,
                                hidden_dim=HIDDEN_DIM)
    if base == "crossattn":
        return CrossAttnFusionMIL(encoders, proj_heads, HIDDEN_DIM, P2_N_HEADS,
                                   P2_N_CROSS_LAYERS, P2_ATTN_DROPOUT, modal_dropout,
                                   slot_k, n_slot_iters, use_cls=use_cls)
    if base == "crossmodal":
        return SlotCrossModalMIL(encoders, proj_heads, HIDDEN_DIM, slot_k, n_slot_iters,
                                  P2_N_HEADS, P2_N_CROSS_LAYERS, P2_ATTN_DROPOUT,
                                  modal_dropout, use_cls)
    if base == "iterative":
        return IterativeXModalMIL(encoders=encoders, proj_heads=proj_heads,
                                   hidden_dim=HIDDEN_DIM, n_iter_blocks=iter_r,
                                   n_slots=slot_k, n_slot_iters=n_slot_iters,
                                   n_heads=P2_N_HEADS, n_cross_layers=P2_N_CROSS_LAYERS,
                                   dropout=P2_ATTN_DROPOUT, modal_dropout=modal_dropout,
                                   max_he_patches=P2_MAX_HE_BLOCK, use_cls=use_cls)
    raise ValueError(f"Unknown variant: {variant!r}")


def train_one_epoch_v7(model, records, optimizer, cw, device, bag_cache,
                       scaler, grad_accum, epoch,
                       lambda_cox=1.0, task="cls", surv_endpoint="acr"):
    """Single-task epoch: task=cls → hinge only; task=surv → Cox only."""
    model.train()
    random.shuffle(records)
    optimizer.zero_grad()

    bags: dict = {m: None for m in MODALITIES}
    bags["HE_coords"] = None
    OOM_SKIP = 3
    oom_count: dict = {}

    cls_losses: list = []
    cox_buffer: list = []
    accum_step  = 0
    total_loss  = 0.0
    n_batches   = 0

    def _flush():
        nonlocal total_loss, n_batches
        if not cls_losses and not cox_buffer:
            cls_losses.clear(); cox_buffer.clear()
            optimizer.zero_grad(); return

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            L_total = None

            if cls_losses:
                L_cls = torch.stack(cls_losses).mean()
                L_total = L_cls

            if cox_buffer and lambda_cox > 0:
                L_cox = cox_breslow_loss(cox_buffer)
                if L_cox is not None and L_cox.requires_grad:
                    L_total = (L_total + lambda_cox * L_cox) if L_total is not None \
                              else lambda_cox * L_cox

        if L_total is not None and L_total.requires_grad:
            if scaler is not None:
                scaler.scale(L_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                L_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += L_total.item()
            n_batches  += 1

        optimizer.zero_grad()
        cls_losses.clear(); cox_buffer.clear()
        _gc()

    for rec in records:
        if oom_count.get(rec["stem"], 0) >= OOM_SKIP:
            continue
        entry = bag_cache.get(rec["stem"], {})
        for m in MODALITIES:
            bags[m] = entry.get(m)
        bags["HE_coords"] = entry.get("HE_coords")
        if all(bags[m] is None for m in MODALITIES):
            continue

        try:
            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                result = model(bags, device)
                if isinstance(result, tuple) and len(result) >= 4:
                    logit, hazard, r_cls, r_surv = result[0], result[1], result[2], result[3]
                elif isinstance(result, tuple) and len(result) == 3:
                    logit, hazard, r_cls = result[0], result[1], result[2]; r_surv = r_cls
                else:
                    logit = result; hazard = None; r_cls = None; r_surv = None
                if not isinstance(logit, torch.Tensor) or logit.grad_fn is None:
                    continue

                # Classification loss — only when task=cls
                label = rec.get("label")
                if task != "surv" and label is not None:
                    tgt = torch.tensor([float(label)], device=device)
                    cls_losses.append(hinge_loss(logit.unsqueeze(0), tgt, cw))

                # Cox accumulation — only when task=surv
                if surv_endpoint == "acr":
                    tte = rec.get("tte_next_acr", float("nan"))
                    ev  = rec.get("event_next_acr", 0)
                else:
                    tte = rec.get(f"{surv_endpoint}_time", float("nan"))
                    ev  = rec.get(f"{surv_endpoint}_event", float("nan"))
                if task != "cls" and hazard is not None and not math.isnan(float(tte)) and float(tte) >= 0 and not math.isnan(float(ev)):
                    cox_buffer.append((hazard.float(), float(tte), float(ev)))

            accum_step += 1
            if accum_step >= grad_accum:
                _flush()
                accum_step = 0

        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad()
            cls_losses.clear(); cox_buffer.clear()
            accum_step = 0; _gc()
            oom_count[rec["stem"]] = oom_count.get(rec["stem"], 0) + 1
            if oom_count[rec["stem"]] >= OOM_SKIP:
                print(f"  [OOM-v7] {rec['stem']} OOM {OOM_SKIP}× — skip for epoch")

    if accum_step > 0:
        _flush()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_v7(model, records, device, bag_cache, cw, surv_endpoint="acr"):
    """Multitask eval: classification metrics + Cox C-index."""
    model.eval()
    cls_probs, cls_labels, cls_losses = [], [], []
    hazards, times, events = [], [], []
    use_amp = (device.type == "cuda")

    for rec in records:
        entry = bag_cache.get(rec["stem"], {})
        bags  = {m: entry.get(m) for m in MODALITIES}
        bags["HE_coords"] = entry.get("HE_coords")
        if all(bags[m] is None for m in MODALITIES):
            continue
        try:
            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model(bags, device)
            if not isinstance(result, tuple) or len(result) < 2:
                continue
            logit, hazard = result[0], result[1]

            label = rec.get("label")
            if label is not None:
                cls_probs.append(torch.sigmoid(logit.float()).item())
                cls_labels.append(int(label))
                cls_losses.append(hinge_loss(logit.unsqueeze(0),
                                             torch.tensor([float(label)], device=device),
                                             cw).item())

            if surv_endpoint == "acr":
                tte = rec.get("tte_next_acr", float("nan"))
                ev  = rec.get("event_next_acr", 0)
            else:
                tte = rec.get(f"{surv_endpoint}_time",  float("nan"))
                ev  = rec.get(f"{surv_endpoint}_event", float("nan"))
            if hazard is not None and not math.isnan(float(tte)) and float(tte) >= 0 and not math.isnan(float(ev)):
                hazards.append(hazard.float().item())
                times.append(float(tte))
                events.append(float(ev))

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

    cls_metrics = {}
    if len(cls_probs) >= 4 and len(set(cls_labels)) == 2:
        cls_metrics = compute_metrics(np.array(cls_labels), np.array(cls_probs))

    ci = None
    if len(hazards) >= 2 and sum(events) > 0:
        ci = c_index(hazards, times, events)

    val_loss = float(np.mean(cls_losses)) if cls_losses else 0.0
    return cls_metrics, ci, val_loss


def run_phase1_v7(records, bag_cache, device, save_dir,
                  p1_epochs=50, p1_lr=1e-4,
                  p1_subsample=0.70, p1_min_patches=16,
                  p1_batch=16, p1_tau=0.07, proj_dim=128,
                  lambda_survcon=0.5, lambda_rank=0.3,
                  tau_time=90.0):
    """
    Per-modality Phase 1 pre-training combining three losses:

      1. Aug-NT-Xent  : two random patch subsamples of same bag → pulled together
      2. SurvCon      : soft temporal SupCon on bag embeddings
                        w_ij = exp(-|T_i-T_j|/tau_time) for event pairs (δ=1)
                        - Two ACR+ (T=0,T=0): w=1 → strongly pulled together
                        - ACR+ vs soon (T=500): w≈0 → no pull
                        - Censored: in denominator only, never anchors/positives
      3. Rank loss    : pairwise -log σ(h_i - h_j) for T_i < T_j, δ_i=1
                        uses a small per-modality risk head (discarded after p1)

    Saves enc_{mod}.pt. Returns {mod_name: state_dict}.
    """
    import torch.nn as nn
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, dict] = {}

    for mod_name in MODALITIES:
        ckpt      = save_dir / f"enc_{mod_name}.pt"
        lock_file = save_dir / f"enc_{mod_name}.lock"

        # If another parallel job is writing, wait for it to finish
        import time as _time
        waited = 0
        while lock_file.exists() and waited < 1800:
            if waited == 0:
                print(f"  [p1/{mod_name}] lock held by another job — waiting...", flush=True)
            _time.sleep(10); waited += 10
        if waited >= 1800:
            print(f"  [p1/{mod_name}] WARNING: lock wait timed out, proceeding anyway")

        if ckpt.exists():
            try:
                print(f"  [p1/{mod_name}] checkpoint exists — loading")
                results[mod_name] = torch.load(ckpt, map_location="cpu")
                continue
            except Exception as e:
                print(f"  [p1/{mod_name}] checkpoint corrupt ({e}) — retraining")
                ckpt.unlink(missing_ok=True)

        # Claim the lock before training
        lock_file.touch()

        pres_col = _pres_col(mod_name)
        mod_recs = [r for r in records if r.get(pres_col, False)]
        if not mod_recs:
            print(f"  [p1/{mod_name}] no samples with this modality — skip")
            lock_file.unlink(missing_ok=True)
            continue

        enc       = GatedAttentionEncoder(_feat_dim(mod_name), HIDDEN_DIM, DROPOUT).to(device)
        proj      = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, proj_dim),
        ).to(device)
        risk_head = nn.Linear(HIDDEN_DIM, 1, bias=True).to(device)  # for rank loss only
        nn.init.normal_(risk_head.weight, 0.0, 0.01)
        nn.init.zeros_(risk_head.bias)

        opt    = torch.optim.AdamW(
            list(enc.parameters()) + list(proj.parameters()) + list(risk_head.parameters()),
            lr=p1_lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        use_surv = lambda_survcon > 0 or lambda_rank > 0

        print(f"  [p1/{mod_name}] {len(mod_recs)} bags, {p1_epochs} epochs  "
              f"(survcon={lambda_survcon}, rank={lambda_rank})", flush=True)

        for ep in range(p1_epochs):
            enc.train(); proj.train(); risk_head.train()
            random.shuffle(mod_recs)
            epoch_losses: list = []

            # Buffers: aug-NT-Xent views
            buf1: list = []; buf2: list = []
            # Survival buffers: (z, hazard, tte, event) per sample
            surv_buf: list = []

            def _flush():
                nonlocal buf1, buf2, surv_buf
                losses = []

                # 1. Aug-NT-Xent
                if len(buf1) >= 2:
                    z1b = torch.stack(buf1); z2b = torch.stack(buf2)
                    L_aug = nt_xent_loss(z1b, z2b, p1_tau)
                    if L_aug is not None:
                        losses.append(L_aug)

                # 2. SurvCon + 3. Rank
                if use_surv and len(surv_buf) >= 2:
                    zs   = torch.stack([s[0] for s in surv_buf])          # (B, D)
                    hs   = torch.stack([s[1] for s in surv_buf]).squeeze() # (B,)
                    tts  = torch.tensor([s[2] for s in surv_buf],
                                        dtype=torch.float32, device=device)
                    evs  = torch.tensor([s[3] for s in surv_buf],
                                        dtype=torch.float32, device=device)

                    if lambda_survcon > 0:
                        L_sc = surv_con_loss(zs, tts, evs,
                                             tau=p1_tau, tau_time=tau_time)
                        if L_sc is not None:
                            losses.append(lambda_survcon * L_sc)

                    if lambda_rank > 0:
                        L_rk = surv_rank_loss(hs, tts, evs)
                        if L_rk is not None:
                            losses.append(lambda_rank * L_rk)

                buf1.clear(); buf2.clear(); surv_buf.clear()

                if not losses:
                    return
                loss = torch.stack(losses).sum()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
                opt.zero_grad()
                epoch_losses.append(loss.item())

            opt.zero_grad()
            for rec in mod_recs:
                bag = bag_cache.get(rec["stem"], {}).get(mod_name)
                if bag is None: continue
                bag  = bag.to(device)
                n    = bag.shape[0]
                if n < 2: continue
                tte  = rec.get("tte_next_acr", float("nan"))
                ev   = rec.get("event_next_acr", 0)
                if math.isnan(tte): continue

                k  = min(n, max(p1_min_patches, int(n * p1_subsample)))
                v1 = bag[torch.randperm(n, device=device)[:k]]
                v2 = bag[torch.randperm(n, device=device)[:k]]

                use_amp = scaler is not None
                with torch.amp.autocast("cuda", enabled=use_amp):
                    r1, _, _ = enc(v1)
                    r2, _, _ = enc(v2)
                    z1 = F.normalize(proj(r1.unsqueeze(0)), dim=-1).squeeze(0)
                    z2 = F.normalize(proj(r2.unsqueeze(0)), dim=-1).squeeze(0)
                    h  = risk_head(r1).squeeze()   # log-hazard for rank loss

                buf1.append(z1); buf2.append(z2)
                if use_surv:
                    surv_buf.append((z1.detach().clone(), h, float(tte), float(ev)))

                if len(buf1) >= p1_batch:
                    _flush()
            _flush()

            if (ep + 1) % 10 == 0 or ep == p1_epochs - 1:
                avg = np.mean(epoch_losses) if epoch_losses else float("nan")
                print(f"  [p1/{mod_name}] ep={ep+1:3d}  loss={avg:.4f}", flush=True)

        # Atomic write: save to .tmp then rename so parallel readers never see partial file
        tmp = ckpt.with_suffix(".tmp")
        torch.save(enc.state_dict(), tmp)
        tmp.rename(ckpt)
        lock_file.unlink(missing_ok=True)
        print(f"  [p1/{mod_name}] saved → {ckpt}")
        results[mod_name] = enc.state_dict()
        del enc, proj, risk_head, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def run_phase1_crossmodal_v7(records, bag_cache, device, p1_weights_in, save_dir,
                              p1_epochs=20, p1_lr=5e-5, p1_batch=32, p1_tau=0.1,
                              proj_dim=128):
    """
    Cross-modal alignment step (Phase 1, Step 2).
    For patients with ≥2 modalities, aligns bag-level representations from different
    modalities via NT-Xent: (rep_m1, rep_m2) from the same patient are positives;
    different-patient reps are negatives. Fine-tunes encoders at low LR.
    Saves enc_{mod}_cm.pt and a done-flag; returns updated state_dicts.
    """
    save_dir = Path(save_dir)
    done_flag = save_dir / "crossmodal_done.flag"
    if done_flag.exists():
        print("  [p1/crossmodal] already done — loading updated encoders")
        updated = dict(p1_weights_in)
        for m in MODALITIES:
            ck = save_dir / f"enc_{m}_cm.pt"
            if ck.exists():
                updated[m] = torch.load(ck, map_location="cpu")
        return updated

    # Build encoders from per-modality Phase 1 weights
    encoders = nn.ModuleDict()
    for m in MODALITIES:
        if m not in p1_weights_in:
            continue
        enc = GatedAttentionEncoder(_feat_dim(m), HIDDEN_DIM, DROPOUT).to(device)
        enc.load_state_dict(p1_weights_in[m])
        encoders[m] = enc

    if len(encoders) < 2:
        return p1_weights_in

    # Per-modality projection heads → shared proj_dim space
    proj_heads = nn.ModuleDict({
        m: nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                         nn.Linear(HIDDEN_DIM, proj_dim))
        for m in encoders
    }).to(device)

    def _n_mods(r):
        return sum(1 for m in encoders if r.get(_pres_col(m), False))
    multimod_recs = [r for r in records if _n_mods(r) >= 2]

    if len(multimod_recs) < 4:
        print("  [p1/crossmodal] too few multi-modal bags — skipping")
        return p1_weights_in

    print(f"  [p1/crossmodal] {len(multimod_recs)} multi-modal bags, {p1_epochs} epochs")

    opt = torch.optim.AdamW(
        list(encoders.parameters()) + list(proj_heads.parameters()),
        lr=p1_lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for ep in range(p1_epochs):
        encoders.train(); proj_heads.train()
        random.shuffle(multimod_recs)
        epoch_losses: list = []
        batch_reps: dict = {}   # stem → {mod → z_tensor}
        n_batched = 0

        def _flush_cm():
            nonlocal batch_reps
            z1s, z2s = [], []
            for stem, mod_z in batch_reps.items():
                mods = list(mod_z.keys())
                if len(mods) < 2:
                    continue
                # All C(M,2) positive pairs for this patient
                for i in range(len(mods)):
                    for j in range(i + 1, len(mods)):
                        z1s.append(mod_z[mods[i]])
                        z2s.append(mod_z[mods[j]])
            batch_reps.clear()
            if len(z1s) < 2:
                return
            z1 = torch.stack(z1s)
            z2 = torch.stack(z2s)
            loss = nt_xent_loss(z1, z2, p1_tau)
            if loss is None:
                return
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            opt.zero_grad()
            epoch_losses.append(loss.item())

        opt.zero_grad()
        for rec in multimod_recs:
            stem = rec["stem"]
            reps_for_stem: dict = {}
            use_amp = scaler is not None
            with torch.amp.autocast("cuda", enabled=use_amp):
                for m, enc in encoders.items():
                    if not rec.get(_pres_col(m), False):
                        continue
                    bag = bag_cache.get(stem, {}).get(m)
                    if bag is None:
                        continue
                    rep, _, _ = enc(bag.to(device, non_blocking=True))
                    z = F.normalize(proj_heads[m](rep.unsqueeze(0)).squeeze(0), dim=-1)
                    reps_for_stem[m] = z
            if len(reps_for_stem) >= 2:
                batch_reps[stem] = reps_for_stem
                n_batched += 1
                if n_batched >= p1_batch:
                    _flush_cm()
                    n_batched = 0
        if batch_reps:
            _flush_cm()

        if (ep + 1) % 5 == 0 or ep == p1_epochs - 1:
            avg = np.mean(epoch_losses) if epoch_losses else float("nan")
            print(f"  [p1/crossmodal] ep={ep+1:3d}  loss={avg:.4f}", flush=True)

    # Save updated encoder weights
    updated = dict(p1_weights_in)
    for m, enc in encoders.items():
        ck = save_dir / f"enc_{m}_cm.pt"
        torch.save(enc.state_dict(), ck)
        updated[m] = enc.state_dict()
    done_flag.touch()
    print(f"  [p1/crossmodal] saved updated encoders → {save_dir}")
    del encoders, proj_heads, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return updated


def run_variant_v7(variant, iter_r, slot_k, fold, split, bag_cache,
                   lambda_cox=1.0, n_slot_iters=3, patience=10, task="cls",
                   p1_epochs=0, surv_endpoint="acr", p1_base_dir=None):
    """Single-task training: task=cls → hinge+val_bacc; task=surv → Cox+val_CI.
    surv_endpoint: 'acr' | 'clad' | 'death' — selects which TTE labels to use.
    p1_base_dir: if set, load P1 weights from this dir (e.g. reuse ACR P1 for CLAD/Death).
    """
    tag      = _variant_tag(variant, iter_r, slot_k)
    fold_tag = f"split{split}_fold{fold}" if split is not None else f"fold_{fold}"
    fold_dir = Path(SAVE_DIR) / fold_tag
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = fold_dir / f"ckpts_{tag}"
    ckpt_dir.mkdir(exist_ok=True)

    status_path = fold_dir / f"status_{tag}.json"
    if _is_completed(fold_dir, tag=f"status_{tag}"):
        st = _read_status(status_path)
        print(f"  [v7/{tag}] Already completed (best_ep={st.get('best_epoch')}). Skipping.")
        mf = fold_dir / f"metrics_{tag}.json"
        return json.load(open(mf)) if mf.exists() else {}

    if surv_endpoint == "acr":
        splits_dict = build_splits_multitask(SAMPLES_DIR, SPLITS_CSV, fold, split)
        splits_dict = {"train": splits_dict["train"], "val": splits_dict["val"],
                       "test": splits_dict["test"]}
    else:
        tr, va, te = build_splits_survival(SAMPLES_DIR, SPLITS_CSV, fold, split,
                                           endpoint=surv_endpoint)
        splits_dict = {"train": tr, "val": va, "test": te}
    train_recs, val_recs, test_recs = (splits_dict["train"], splits_dict["val"], splits_dict["test"])
    update_presence_from_cache(train_recs, bag_cache)
    update_presence_from_cache(val_recs,   bag_cache)
    update_presence_from_cache(test_recs,  bag_cache)
    _print_split_stats(fold_tag, "train", train_recs)
    _print_split_stats(fold_tag, "val",   val_recs)
    _print_split_stats(fold_tag, "test",  test_recs)

    # ── Phase 1: per-modality aug-CLR pre-training ───────────────────────────
    # p1_base_dir lets CLAD/Death tasks reuse P1 weights from the ACR run.
    if p1_base_dir is not None:
        phase1_dir = Path(p1_base_dir) / fold_tag / "phase1"
    else:
        phase1_dir = fold_dir / "phase1"
    p1_weights: Dict[str, dict] = {}
    if p1_epochs > 0:
        print(f"\n  [v7/{tag}] === Phase 1 step 1: {p1_epochs} epochs per modality ===")
        all_recs = train_recs + val_recs + test_recs
        p1_weights = run_phase1_v7(
            all_recs, bag_cache, DEVICE, phase1_dir, p1_epochs=p1_epochs)
        cm_epochs = max(10, p1_epochs // 5)
        print(f"\n  [v7/{tag}] === Phase 1 step 2: cross-modal alignment {cm_epochs} epochs ===")
        p1_weights = run_phase1_crossmodal_v7(
            all_recs, bag_cache, DEVICE, p1_weights, phase1_dir,
            p1_epochs=cm_epochs)
    else:
        # Try loading existing Phase 1 weights produced by a sibling variant.
        # Prefer cross-modal fine-tuned weights (enc_{mod}_cm.pt) when available.
        for mod_name in MODALITIES:
            ck_cm = phase1_dir / f"enc_{mod_name}_cm.pt"
            ck    = phase1_dir / f"enc_{mod_name}.pt"
            if ck_cm.exists():
                p1_weights[mod_name] = torch.load(ck_cm, map_location="cpu")
            elif ck.exists():
                p1_weights[mod_name] = torch.load(ck, map_location="cpu")
        if p1_weights:
            print(f"  [v7/{tag}] Reusing Phase 1 weights from {phase1_dir}")

    # lambda_cox for each task leg:
    #   cls       → 0.0  (hinge only)
    #   surv      → lambda_cox  (Cox only)
    #   both_alt  → lambda_cox  (full Cox when surv epoch; 0 when cls epoch — set per-epoch)
    lambda_cox_surv = lambda_cox   # used when the epoch is a surv epoch
    lambda_cox_cls  = 0.0          # used when the epoch is a cls epoch

    model = build_model_v7(variant, iter_r=iter_r, slot_k=slot_k,
                            n_slot_iters=n_slot_iters).to(DEVICE)

    # Load Phase 1 encoder weights into fusion model
    for mod_name, sd in p1_weights.items():
        if mod_name in model.encoders:
            model.encoders[mod_name].load_state_dict(sd, strict=True)
            print(f"  [v7/{tag}] Loaded Phase 1 encoder: {mod_name}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  [v7/{tag}]  task={task}  params={n_params:,}  λ_cox={lambda_cox}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=P2_LR, weight_decay=P2_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=P2_EPOCHS, eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    cw        = compute_class_weights(train_recs)

    history = {"train_loss": [], "val_loss": [], "val_auc": [],
               "val_bacc": [], "val_mcc": [], "val_cindex": []}

    best_metric = -1.0; best_ep = 0; no_improve = 0
    # Track which task each epoch ran (for both_alt logging)
    epoch_tasks: list = []

    for epoch in range(P2_EPOCHS):
        # Determine per-epoch task
        if task == "both_alt":
            ep_task = random.choice(["cls", "surv"])
            ep_lambda = lambda_cox_surv if ep_task == "surv" else lambda_cox_cls
        else:
            ep_task  = task
            ep_lambda = lambda_cox_surv if task == "surv" else lambda_cox_cls
        epoch_tasks.append(ep_task)

        tl = train_one_epoch_v7(
            model, train_recs, optimizer, cw, DEVICE, bag_cache,
            scaler, P2_GRAD_ACCUM, epoch,
            lambda_cox=ep_lambda, task=ep_task, surv_endpoint=surv_endpoint)
        history["train_loss"].append(tl)
        scheduler.step()

        if (epoch + 1) % P2_EVAL_EVERY == 0 or epoch == P2_EPOCHS - 1:
            cls_m, ci, vl = evaluate_v7(model, val_recs, DEVICE, bag_cache, cw,
                                        surv_endpoint=surv_endpoint)
            history["val_loss"].append(vl)
            history["val_auc"].append(cls_m.get("auc", 0.0))
            history["val_bacc"].append(cls_m.get("bacc", 0.0))
            history["val_mcc"].append(cls_m.get("mcc", 0.0))
            history["val_cindex"].append(ci if ci is not None else 0.0)

            # Task-specific model selection
            if task == "surv":
                val_metric = ci if ci is not None else 0.0
            elif task == "both_alt":
                # Combined: average of bacc and C-index (both normalised to [0,1])
                val_metric = 0.5 * cls_m.get("bacc", 0.0) + 0.5 * (ci if ci is not None else 0.0)
            else:
                val_metric = cls_m.get("bacc", 0.0)

            print(f"  [v7/{tag}] ep={epoch+1:3d} [{ep_task:4s}]  loss={tl:.4f}  "
                  f"val_bacc={cls_m.get('bacc',0):.4f}  "
                  f"val_auc={cls_m.get('auc',0):.4f}  "
                  f"val_ci={ci or 0:.4f}", flush=True)

            if val_metric > best_metric:
                best_metric = val_metric; best_ep = epoch + 1; no_improve = 0
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "history": history},
                           ckpt_dir / "best_model.pt")
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    print(f"  [v7/{tag}] Early stop at ep {epoch+1}"); break

    # Load best and evaluate on test
    best_ckpt = ckpt_dir / "best_model.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE)["model"],
                              strict=False)
    test_cls, test_ci, _ = evaluate_v7(model, test_recs, DEVICE, bag_cache, cw,
                                       surv_endpoint=surv_endpoint)
    test_metrics = {**test_cls,
                    "c_index":    test_ci if test_ci is not None else 0.0,
                    "best_epoch": best_ep}
    print(f"  [v7/{tag}] TEST  auc={test_cls.get('auc',0):.4f}  "
          f"bacc={test_cls.get('bacc',0):.4f}  ci={test_ci or 0:.4f}")

    test_metrics["task"] = task
    test_metrics["surv_endpoint"] = surv_endpoint
    with open(fold_dir / f"metrics_{tag}.json", "w") as f:
        json.dump({"val": history, "test": test_metrics, "task": task,
                   "surv_endpoint": surv_endpoint}, f, indent=2)
    _write_status(status_path, completed=True, **test_metrics)
    _plot_training_curves(history, fold_dir, tag)
    return {"val": history, "test": test_metrics, "task": task}


def parse_args():
    p = argparse.ArgumentParser(
        description="Multimodal ABMIL v7 — single-phase multitask (hinge cls + Cox TTE)")

    # Paths
    p.add_argument("--folds",       nargs="+", type=int, default=None)
    p.add_argument("--split",       type=int,  default=None)
    p.add_argument("--save_dir",    type=str,  default=SAVE_DIR)
    p.add_argument("--samples_dir", type=str,  default=SAMPLES_DIR)
    p.add_argument("--splits_csv",  type=str,  default=SPLITS_CSV)

    # Unused stubs retained so existing job scripts that pass Phase-1 flags don't crash
    p.add_argument("--p1_cross_attn",  action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--p1_crd",         action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--p1_contrastive", action="store_true", help=argparse.SUPPRESS)

    # Variants
    p.add_argument("--p2_variants", nargs="+", type=str,
                   default=["iterative"], choices=P2_VARIANTS)
    p.add_argument("--p2_iter_r",     nargs="+", type=int, default=[P2_ITER_R])
    p.add_argument("--p2_slot_iters", type=int, default=3)
    p.add_argument("--p2_slot_k",     nargs="+", type=int, default=[P2_SLOT_K])
    p.add_argument("--p2_max_patches",  type=int, default=P2_MAX_PATCHES)
    p.add_argument("--p2_max_he_block", type=int, default=P2_MAX_HE_BLOCK)

    # Loss weights
    p.add_argument("--lambda_cox",  type=float, default=LAMBDA_COX,
                   help=f"Cox TTE loss weight (default {LAMBDA_COX})")

    # Phase 1
    p.add_argument("--p1_only",   action="store_true",
                   help="Run Phase 1 pre-training only (no Phase 2). "
                        "Saves enc_*.pt to save_dir/split{S}_fold{F}/phase1/ and exits.")
    p.add_argument("--p1_epochs", type=int, default=0,
                   help="Phase 1 aug-CLR epochs per modality (0 = skip)")

    # Early stopping
    p.add_argument("--v7_patience", type=int, default=10,
                   help="Early-stop patience in eval periods (default 10)")
    p.add_argument("--task", type=str, default="cls",
                   choices=["cls", "surv", "both_alt"],
                   help=("Training task: cls=hinge only (val_bacc), "
                         "surv=Cox only (val_CI), "
                         "both_alt=alternating per-epoch (val_bacc+val_CI)"))
    p.add_argument("--surv_endpoint", type=str, default="acr",
                   choices=["acr", "clad", "death"],
                   help="Survival endpoint: acr (default), clad, or death")
    p.add_argument("--p1_base_dir", type=str, default=None,
                   help="Load P1 encoder weights from this dir instead of save_dir "
                        "(e.g. reuse ACR Phase-1 for CLAD/Death runs)")

    return p.parse_args()


def main():
    args = parse_args()

    global FOLDS, SPLIT, SAVE_DIR, SAMPLES_DIR, SPLITS_CSV
    FOLDS       = args.folds if args.folds is not None else FOLDS
    SPLIT       = args.split
    SAVE_DIR    = args.save_dir
    SAMPLES_DIR = args.samples_dir
    SPLITS_CSV  = args.splits_csv
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    print(f"PyTorch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    global P2_MAX_PATCHES, P2_MAX_HE_BLOCK
    P2_MAX_PATCHES  = args.p2_max_patches
    P2_MAX_HE_BLOCK = args.p2_max_he_block
    if torch.cuda.is_available():
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import pandas as pd
    df = pd.read_csv(SPLITS_CSV)
    valid_stems = list({Path(str(row["file"])).stem for _, row in df.iterrows()})
    print(f"  CSV rows: {len(df)}  |  stems: {len(valid_stems)}")

    print(f"  Preloading bags ...")
    bag_cache = preload_bags(valid_stems, SAMPLES_DIR)
    _gc()

    # ── Phase 1 only mode ────────────────────────────────────────────────────────
    if args.p1_only:
        print(f"\n{'#'*65}")
        print(f"#  Multimodal ABMIL v7  [PHASE 1 ONLY]  p1_epochs={args.p1_epochs}")
        print(f"{'#'*65}\n")
        for fold in FOLDS:
            fold_tag  = f"split{SPLIT}_fold{fold}" if SPLIT is not None else f"fold_{fold}"
            fold_dir  = Path(SAVE_DIR) / fold_tag
            fold_dir.mkdir(parents=True, exist_ok=True)
            phase1_dir = fold_dir / "phase1"
            print(f"\n{'='*65}  {fold_tag.upper()}")
            splits_dict = build_splits_multitask(SAMPLES_DIR, SPLITS_CSV, fold, SPLIT)
            all_recs = splits_dict["train"] + splits_dict["val"] + splits_dict["test"]
            update_presence_from_cache(all_recs, bag_cache)
            print(f"  === Phase 1 step 1: {args.p1_epochs} epochs per modality ===")
            p1_weights = run_phase1_v7(all_recs, bag_cache, DEVICE, phase1_dir,
                                       p1_epochs=args.p1_epochs)
            cm_epochs = max(10, args.p1_epochs // 5)
            print(f"  === Phase 1 step 2: cross-modal alignment {cm_epochs} epochs ===")
            run_phase1_crossmodal_v7(all_recs, bag_cache, DEVICE, p1_weights, phase1_dir,
                                     p1_epochs=cm_epochs)
            print(f"  [fold={fold}] Phase 1 complete → {phase1_dir}")
        print("\nPhase 1 done. Exiting.")
        return

    all_fold_metrics: Dict[int, dict] = {}

    print(f"\n{'#'*65}")
    print(f"#  Multimodal ABMIL v7  [task={args.task}, single-phase]")
    print(f"#  variants={args.p2_variants}  λ_cox={args.lambda_cox}")
    print(f"{'#'*65}\n")

    for fold in FOLDS:
        fold_tag = f"split{SPLIT}_fold{fold}" if SPLIT is not None else f"fold_{fold}"
        print(f"\n{'='*65}  {fold_tag.upper()}")
        set_seeds(SEED)
        fold_metrics: dict = {}
        for variant in args.p2_variants:
            base = variant.replace("_cls", "")
            iter_r_list = args.p2_iter_r if base == "iterative" else [2]
            slot_k_list = args.p2_slot_k if base in ("crossattn", "crossmodal", "iterative") else [8]
            for ir in (iter_r_list if base == "iterative" else [2]):
                for sk in slot_k_list:
                    tag = _variant_tag(variant, ir, sk)
                    fold_metrics[tag] = run_variant_v7(
                        variant, ir, sk, fold, SPLIT, bag_cache,
                        lambda_cox=args.lambda_cox,
                        n_slot_iters=args.p2_slot_iters,
                        patience=args.v7_patience,
                        task=args.task,
                        p1_epochs=args.p1_epochs,
                        surv_endpoint=args.surv_endpoint,
                        p1_base_dir=args.p1_base_dir)
        all_fold_metrics[fold] = fold_metrics

    del bag_cache; _gc()

    agg_path = Path(SAVE_DIR) / "all_fold_metrics.json"
    if agg_path.exists():
        try:
            with open(agg_path) as _f:
                _existing = json.load(_f)
            _existing.update({str(k): v for k, v in all_fold_metrics.items()})
            all_fold_metrics = {int(k): v for k, v in _existing.items()}
        except Exception:
            pass
    with open(agg_path, "w") as f:
        json.dump({str(k): v for k, v in all_fold_metrics.items()}, f, indent=2)

    all_tags = set()
    for fm in all_fold_metrics.values(): all_tags.update(fm.keys())
    for vtag in sorted(all_tags):
        completed = [f for f in FOLDS if all_fold_metrics.get(f, {}).get(vtag)]
        if len(completed) > 1:
            pooled_dir = Path(SAVE_DIR) / "pooled"
            _plot_pooled_cv(all_fold_metrics, completed, pooled_dir, vtag)

    print(f"\n{'─'*65}")
    print(f"  FINAL TEST RESULTS  (best val BAcc per fold)")
    print(f"{'─'*65}")
    for vtag in sorted(all_tags):
        print(f"\n  [{vtag}]")
        print(f"  {'Fold':>4}  {'AUC':>6}  {'AUPRC':>6}  "
              f"{'BAcc':>6}  {'MCC':>6}  {'CI':>6}")
        aucs = []
        for fold in FOLDS:
            tm = all_fold_metrics.get(fold, {}).get(vtag, {}).get("test", {})
            if not tm: continue
            aucs.append(tm.get("auc", 0))
            print(f"  {fold:>4}  {tm.get('auc',0):>6.4f}  "
                  f"{tm.get('auprc',0):>6.4f}  {tm.get('bacc',0):>6.4f}  "
                  f"{tm.get('mcc',0):>6.4f}  {tm.get('c_index',0):>6.4f}")
        if aucs:
            print(f"  {'mean':>4}  {np.mean(aucs):>6.4f}  ±{np.std(aucs):.4f}")

    print(f"\n  Done. Outputs: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
