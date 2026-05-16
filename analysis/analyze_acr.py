#!/usr/bin/env python3
"""
analyze_v6.py  —  Unified post-training analysis suite
=======================================================
Four analyses in one script, each independently runnable via --tasks flag:

  1. metrics_table   Per-split metrics table for every Phase 1 + Phase 2 config,
                     showing how each addition (CLR, KD, CRD, bidir-xattn, variant)
                     improves or hurts performance.

  2. rep_maps        Patient-level representation PCA / UMAP colored by:
                       • modality combination present
                       • ACR label (rejection vs no-rejection)
                     For every model config (P1 per-mod + P2 fusion).

  3. attn_maps       HE-only UMAP + global attention heatmap (top patches highlighted).
                     Coloured by: attention weight, ACR label, cluster (if adata provided).

  4. combo_table     Per modality-combination metric tables for each P2 variant
                     (how many samples per combo, what performance on each combo).

Usage
-----
  # Run all tasks
  python analyze_v6.py \\
      --results_dir  ./results_mm_abmil_v6 \\
      --samples_dir  /lustre/.../samples \\
      --splits_csv   ./plots/multimodal_splits_acr_fixed.csv \\
      --output_dir   ./analysis_v6 \\
      --folds        0 1 2 3

  # Run only specific tasks
  python analyze_v6.py --tasks metrics_table combo_table ...

  # With AnnData for HE attention (task 3)
  python analyze_v6.py --tasks attn_maps \\
      --adata_path /lustre/.../adata_v3.h5ad \\
      --cluster_col subclusters_merged
"""

import argparse
import gc
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

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

try:
    from umap import UMAP as UMAPTransform
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("  [info] umap-learn not installed — PCA only for rep_maps")

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

MODALITY_REGISTRY: Dict[str, Tuple[str, int, str]] = {
    "HE":       ("HE_cells",  1024, "has_HE"),
    "BAL":      ("BAL_cells", 10,   "has_BAL"),
    "CT":       ("CT_cells",  1024, "has_CT"),
    "Clinical": ("Clinical",  408,  "has_Clinical"),
}
MODALITIES         = list(MODALITY_REGISTRY.keys())
TEACHER_MODALITIES = ["HE", "Clinical"]
STUDENT_MODALITIES = ["CT", "BAL"]

def _feat_key(m): return MODALITY_REGISTRY[m][0]
def _feat_dim(m): return MODALITY_REGISTRY[m][1]
def _pres_col(m): return MODALITY_REGISTRY[m][2]

METRICS = ["auc", "auprc", "bacc", "mcc", "sens", "spec"]

COMBO_COLORS = {
    "HE":                      "#4e79a7",
    "HE+Clinical":             "#f28e2b",
    "HE+CT":                   "#e15759",
    "HE+BAL":                  "#76b7b2",
    "HE+CT+Clinical":          "#59a14f",
    "HE+BAL+Clinical":         "#edc948",
    "HE+CT+BAL":               "#b07aa1",
    "HE+CT+BAL+Clinical":      "#ff9da7",
    "CT":                      "#9c755f",
    "BAL":                     "#bab0ac",
    "Clinical":                "#d4a6c8",
    "CT+Clinical":             "#86bcb6",
    "BAL+Clinical":            "#f1ce63",
    "CT+BAL":                  "#d37295",
    "CT+BAL+Clinical":         "#a0cbe8",
    "NONE":                    "#cccccc",
}

LABEL_COLORS = {0: "#5c9be0", 1: "#e05c5c"}   # blue=neg, red=pos

# Mapping from logical modality name → "cells" key used in new .pt aggregate dicts
AGG_FEAT_KEYS: Dict[str, str] = {
    "HE":  "HE_cells",
    "BAL": "BAL_cells",
    "CT":  "CT_cells",
}
AGG_MODS = list(AGG_FEAT_KEYS.keys())  # modalities that have cluster aggregates


# ══════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════

def acr_label(grade_str) -> Optional[int]:
    if grade_str is None: return None
    if isinstance(grade_str, float) and np.isnan(grade_str): return None
    grade = str(grade_str).strip()
    if not grade or grade.lower() in ("nan","none","n/a","na","","?"): return None
    if not re.search(r"A", grade, re.IGNORECASE): return None
    return 1 if re.search(r"A[12]", grade, re.IGNORECASE) else 0


def build_records(splits_csv: str, fold: int,
                  split: Optional[str] = None,
                  outer_split: Optional[int] = None) -> List[dict]:
    """
    outer_split: the outer nested-CV split index (0-4).
                 When provided, uses column split{outer_split}_fold{fold}.
                 When None, falls back to fold_{fold} (old single-split layout).
    split:       "train" | "val" | "test" — filter to this inner split.
                 When None, returns all rows.
    """
    df = pd.read_csv(splits_csv)
    if outer_split is not None:
        fold_col = f"split{outer_split}_fold{fold}"
    else:
        fold_col = f"fold_{fold}"
    if fold_col not in df.columns:
        # last-resort: find any column containing fold index
        for c in df.columns:
            if f"fold{fold}" in c:
                fold_col = c; break
    assert fold_col in df.columns, (
        f"No fold column for outer_split={outer_split} fold={fold}. "
        f"Tried: split{outer_split}_fold{fold}, fold_{fold}. "
        f"Available: {[c for c in df.columns if 'fold' in c.lower()]}")

    out = []
    for _, row in df.iterrows():
        lbl = acr_label(row.get("acr_grade"))
        if lbl is None: continue
        sp = str(row[fold_col]).strip()
        if split is not None and sp != split: continue
        stem = Path(str(row["file"])).stem
        rec  = {"stem": stem, "label": lbl, "split": sp,
                "patient_id": str(row.get("patient_id", stem))}
        for mod in MODALITIES:
            rec[_pres_col(mod)] = bool(row.get(_pres_col(mod), False))
        out.append(rec)
    return out


def combo_key(rec: dict) -> str:
    present = [m for m in MODALITIES if rec.get(_pres_col(m))]
    return "+".join(present) if present else "NONE"


def preload_bags(stems: List[str], samples_dir: str,
                 quiet: bool = False) -> Dict:
    sd = Path(samples_dir)
    cache: Dict = {}
    for i, stem in enumerate(sorted(stems)):
        path = sd / f"{stem}.pt"
        entry = {m: None for m in MODALITIES}
        if path.exists():
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                inp  = data.get("inputs", {})
                for mod in MODALITIES:
                    if mod == "Clinical":
                        # Prefer clinical_onehot (F, F*n_bins) — milv2 format
                        coh = data.get("clinical_onehot")
                        if coh is not None and isinstance(coh, torch.Tensor) and coh.numel() > 0:
                            if coh.dtype == torch.float16: coh = coh.float()
                            entry["Clinical"] = coh
                        # milv3 stores raw 1D (102,) in inputs["Clinical"] — not usable
                        # with 408-dim trained model, so skip
                        continue
                    t = inp.get(_feat_key(mod))
                    if t is not None and isinstance(t, torch.Tensor) and t.numel() > 0:
                        if t.dtype == torch.float16: t = t.float()
                        if t.dim() == 1: t = t.unsqueeze(0)
                        entry[mod] = t
                del data, inp
            except Exception as e:
                if not quiet:
                    print(f"  [warn] {path.name}: {e}")
        cache[stem] = entry
        if not quiet and (i+1) % 200 == 0:
            mb = sum(t.numel()*4/1e6 for e in cache.values()
                     for t in e.values() if t is not None)
            print(f"  preload {i+1}/{len(stems)}  RAM={mb:.0f}MB", flush=True)
        if (i+1) % 200 == 0: _gc()
    return cache


def update_presence(records: List[dict], bag_cache: Dict) -> List[dict]:
    for rec in records:
        entry = bag_cache.get(rec["stem"], {})
        for mod in MODALITIES:
            rec[_pres_col(mod)] = entry.get(mod) is not None
        rec["combo"] = combo_key(rec)
    return records


def _token_ids_to_onehot(token_ids: torch.Tensor, K: int, n_bins: int = 4,
                          n_tokens_per_cluster: Optional[int] = None) -> torch.Tensor:
    """Convert per-cluster bin token IDs (K,) to block-diagonal one-hot (K, K*n_bins).
    If n_tokens_per_cluster is set (milv3 global IDs), converts via global_id % n_tokens_per_cluster.
    Bin index >= n_bins (NaN bin) is silently skipped."""
    coh = torch.zeros(K, K * n_bins, dtype=torch.float32)
    for k, tid in enumerate(token_ids.tolist()):
        b = int(tid)
        if n_tokens_per_cluster is not None:
            b = b % n_tokens_per_cluster
        if 0 <= b < n_bins:
            coh[k, k * n_bins + b] = 1.0
    return coh


def _cluster_presence_mask(cluster_labels_list, cluster_names) -> torch.Tensor:
    """Boolean mask (K,): True if that cluster appears in the instance labels."""
    if not cluster_labels_list or not cluster_names:
        return torch.ones(len(cluster_names or []), dtype=torch.bool)
    labels_set = set(cluster_labels_list)
    return torch.tensor([name in labels_set for name in cluster_names], dtype=torch.bool)


def _extract_feat_names(clinical_vocab) -> Optional[List[str]]:
    """Derive ordered feature names from clinical_vocab by stripping bin suffix."""
    if not clinical_vocab:
        return None
    seen: set = set()
    names: List[str] = []
    for entry in clinical_vocab:
        label = entry.get("label", "") if isinstance(entry, dict) else str(entry)
        feat = re.sub(r'_(q\d+|nan|\d+)$', '', label)
        if feat not in seen:
            seen.add(feat)
            names.append(feat)
    return names or None


def preload_aggregates(
    stems: List[str],
    samples_dir: str,
    mods: Optional[List[str]] = None,
    quiet: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Load per-modality cluster-level and instance-level aggregates from .pt format.

    Field mapping from precompute_dataset.py:
      bag_centroids[fk]        -> cluster_centroids
      bag_cluster_names[fk]    -> cluster_names
      bag_count_token_ids[fk]  -> cluster_count_onehot (converted to block-diagonal)
      cluster_labels[fk]       -> instance_cluster_ids
      coords[fk]               -> instance_spatial_coords
      inputs["Clinical"]       -> clinical_onehot
      clinical_vocab           -> clinical_feature_names (extracted)

    Returns
    -------
    agg_cache  : stem -> {mod -> {cluster_centroids, cluster_mask, cluster_names,
                                   cluster_count_onehot, instance_cluster_ids,
                                   instance_spatial_coords}}
    clin_cache : stem -> {clinical_onehot, clinical_feature_names}
    """
    if mods is None:
        mods = AGG_MODS
    sd = Path(samples_dir)
    agg_cache:  Dict = {}
    clin_cache: Dict = {}

    for i, stem in enumerate(sorted(stems)):
        path      = sd / f"{stem}.pt"
        agg_entry = {m: None for m in mods}
        clin_entry: dict = {}

        if path.exists():
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)

                raw_bc   = data.get("bag_centroids")       or {}
                raw_bcn  = data.get("bag_cluster_names")   or {}
                raw_bcti = data.get("bag_count_token_ids") or {}
                raw_cl   = data.get("cluster_labels")      or {}
                raw_co   = data.get("coords")              or {}
                raw_bcm  = data.get("bag_cluster_mask")    or {}

                # milv3: bag_count_token_ids stores global vocab IDs (k*5 + local_bin)
                # where n_tokens_per_cluster=5 (Q1, Q2, Q3, Q4, NaN)
                is_milv3 = "bag_count_vocab" in data or isinstance(raw_bcm, dict)
                n_tok_per_clust = 5 if is_milv3 else None

                for mod in mods:
                    fk = AGG_FEAT_KEYS[mod]
                    centroids = raw_bc.get(fk)
                    if centroids is None:
                        continue
                    cluster_names = raw_bcn.get(fk)
                    K = centroids.shape[0] if isinstance(centroids, torch.Tensor) else 0
                    if K == 0:
                        continue

                    token_ids = raw_bcti.get(fk)
                    if token_ids is not None and isinstance(token_ids, torch.Tensor):
                        coh = _token_ids_to_onehot(token_ids, K, n_bins=4,
                                                    n_tokens_per_cluster=n_tok_per_clust)
                    else:
                        coh = None

                    # milv3: bag_cluster_mask is a dict keyed by feat key → use directly
                    cmask_direct = raw_bcm.get(fk) if isinstance(raw_bcm, dict) else None
                    if cmask_direct is not None and isinstance(cmask_direct, torch.Tensor):
                        cmask = cmask_direct.bool()
                    else:
                        inst_labels = raw_cl.get(fk)
                        cmask = _cluster_presence_mask(inst_labels, cluster_names)

                    agg_entry[mod] = {
                        "cluster_centroids":       centroids,
                        "cluster_mask":            cmask,
                        "cluster_names":           cluster_names,
                        "cluster_count_onehot":    coh,
                        "instance_cluster_ids":    raw_cl.get(fk),
                        "instance_spatial_coords": raw_co.get(fk),
                    }

                # Clinical onehot: milv2 stores clinical_onehot; milv3 only has raw 1D
                cof = data.get("clinical_onehot")
                if cof is None:
                    # milv3: inputs["Clinical"] is 1D raw — not usable as onehot
                    pass
                if cof is not None and isinstance(cof, torch.Tensor) and cof.dim() >= 2:
                    if cof.dtype == torch.float16:
                        cof = cof.float()
                    cfn = _extract_feat_names(data.get("clinical_vocab"))
                    if cfn is None:
                        cfn = data.get("clinical_feature_names")
                    clin_entry = {
                        "clinical_onehot":        cof,
                        "clinical_feature_names": cfn,
                    }
                del data
            except Exception as e:
                if not quiet:
                    print(f"  [agg warn] {path.name}: {e}")

        agg_cache[stem]  = agg_entry
        clin_cache[stem] = clin_entry
        if not quiet and (i + 1) % 200 == 0:
            print(f"  preload agg {i+1}/{len(stems)}", flush=True)
        if (i + 1) % 200 == 0:
            _gc()

    n_loaded = {m: sum(1 for e in agg_cache.values() if e.get(m) is not None) for m in mods}
    n_clin   = sum(1 for e in clin_cache.values() if e)
    for m in mods:
        print(f"  agg {m:5s}: {n_loaded[m]} stems")
    print(f"  clinical: {n_clin} stems")
    return agg_cache, clin_cache


# ══════════════════════════════════════════════════════════════════
# MODEL CLASSES (minimal — match v5/v6 training)
# ══════════════════════════════════════════════════════════════════

class GatedAttentionEncoder(nn.Module):
    def __init__(self, feat_dim=1024, hidden_dim=256, dropout=0.4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone   = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.att_V    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.att_U    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.att_w    = nn.Linear(hidden_dim, 1, bias=False)
        self.att_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        h     = self.backbone(x)
        gate  = self.att_V(h) * self.att_U(h)
        alpha = F.softmax(self.att_w(self.att_drop(gate)), dim=0)
        return (alpha * h).sum(dim=0), alpha.squeeze(1), h

    @torch.no_grad()
    def attn_logits(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.att_w(self.att_V(h) * self.att_U(h)).squeeze(1)


class SingleModalMIL(nn.Module):
    def __init__(self, feat_dim=1024, hidden_dim=256, dropout=0.4, proj_dim=128):
        super().__init__()
        self.encoder   = GatedAttentionEncoder(feat_dim, hidden_dim, dropout)
        self.head      = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim))

    def forward(self, x):
        rep, alpha, h = self.encoder(x)
        return self.head(rep).squeeze()

    @torch.no_grad()
    def rep_and_attn(self, x):
        rep, alpha, h = self.encoder(x)
        return rep, alpha


def load_p1_model(ckpt_path: Path, feat_dim: int,
                  hidden_dim: int, dropout: float,
                  device: torch.device) -> Optional[SingleModalMIL]:
    if not ckpt_path.exists():
        return None
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Auto-detect feat_dim from checkpoint to handle any registry mismatch
    bb_key = "encoder.backbone.0.weight"
    if bb_key in state:
        feat_dim = state[bb_key].shape[1]
    m = SingleModalMIL(feat_dim, hidden_dim, dropout)
    m.load_state_dict(state, strict=False)
    m.to(device).eval()
    return m


def _generic_p2_forward(ckpt_path: Path, model_class, build_fn,
                        bags: dict, device: torch.device) -> Optional[float]:
    """Generic Phase 2 inference — load model, run forward, return prob."""
    if not ckpt_path.exists():
        return None
    try:
        model = build_fn(ckpt_path, device)
        if model is None:
            return None
        model.eval()
        with torch.no_grad():
            logit = model(bags, device)
        return torch.sigmoid(logit.float()).item()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# SCAN RESULTS DIRECTORY
# ══════════════════════════════════════════════════════════════════

def scan_p1_configs(results_dir: Path, folds: List[int],
                    outer_splits: Optional[List[int]] = None) -> List[dict]:
    """
    Detect all Phase 1 configs.
    Handles both single-split layout  (results_dir/fold_N/phase1/)
    and nested-CV layout              (results_dir/split{S}_fold{N}/phase1/).
    """
    configs = []
    # Collect candidate fold dirs
    candidate_dirs = []
    if outer_splits:
        for s in outer_splits:
            for fold in folds:
                d = results_dir / f"split{s}_fold{fold}" / "phase1"
                if d.exists():
                    candidate_dirs.append((fold, s, d))
    # Also check plain fold_N layout
    for fold in folds:
        d = results_dir / f"fold_{fold}" / "phase1"
        if d.exists():
            candidate_dirs.append((fold, None, d))

    for fold, outer_split, fold_dir in candidate_dirs:
        split_tag = f"s{outer_split}_" if outer_split is not None else ""
        for mod in MODALITIES:
            mp = fold_dir / mod / "metrics.json"
            if mp.exists():
                configs.append({
                    "fold": fold,
                    "outer_split": outer_split,
                    "mod": mod,
                    "metrics_path": mp,
                    "model_path":   fold_dir / mod / "best_model.pt",
                    "tag": f"P1_{mod}",
                })
    return configs


def scan_p2_configs(results_dir: Path, folds: List[int],
                    outer_splits: Optional[List[int]] = None) -> List[dict]:
    """
    Detect all Phase 2 variant configs.
    Handles both single-split (fold_N) and nested-CV (split{S}_fold{N}) layouts.
    """
    configs = []
    candidate_dirs = []
    if outer_splits:
        for s in outer_splits:
            for fold in folds:
                d = results_dir / f"split{s}_fold{fold}" / "phase2"
                if d.exists():
                    candidate_dirs.append((fold, s, d))
    for fold in folds:
        d = results_dir / f"fold_{fold}" / "phase2"
        if d.exists():
            candidate_dirs.append((fold, None, d))

    for fold, outer_split, p2_dir in candidate_dirs:
        for mf in sorted(p2_dir.glob("metrics_*.json")):
            vtag = mf.stem.replace("metrics_", "")
            configs.append({
                "fold": fold,
                "outer_split": outer_split,
                "vtag": vtag,
                "metrics_path": mf,
                "model_path":   p2_dir / f"model_{vtag}.pt",
                "tag": f"P2_{vtag}",
            })
    return configs


def load_stored_metrics(metrics_path: Path) -> dict:
    """Load a metrics JSON and return {split: {metric: value}}."""
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except Exception:
        return {}


def load_p1_preds_per_stem(
    results_dir: Path,
    splits_csv: str,
    folds: List[int],
    outer_splits: Optional[List[int]] = None,
) -> Dict:
    """
    Load per-patient P1 predictions aligned to splits_csv records.
    Returns {(fold, mod, stem): {'prob': float, 'label': int, 'split': str}}
    Labels always come from splits_csv, not from stored JSON.
    """
    p1_cfgs = scan_p1_configs(results_dir, folds, outer_splits)
    preds: Dict = {}
    for cfg in p1_cfgs:
        fold        = cfg["fold"]
        mod         = cfg["mod"]
        outer_split = cfg["outer_split"]
        metrics     = load_stored_metrics(cfg["metrics_path"])
        for split_name, sm in metrics.items():
            if not isinstance(sm, dict): continue
            probs = sm.get("probs")
            if not probs: continue
            split_recs = [r for r in build_records(splits_csv, fold, outer_split=outer_split)
                          if r["split"] == split_name and r.get(_pres_col(mod), False)]
            if len(split_recs) != len(probs): continue
            for prob, rec in zip(probs, split_recs):
                key = (fold, mod, rec["stem"])
                preds[key] = {"prob": float(prob), "label": rec["label"], "split": split_name}
    return preds


# ══════════════════════════════════════════════════════════════════
# TASK 1: METRICS TABLE
# ══════════════════════════════════════════════════════════════════

def task_metrics_table(results_dir: Path, folds: List[int],
                       out_dir: Path, splits_csv: str,
                       outer_splits: Optional[List[int]] = None):
    """
    For every Phase 1 and Phase 2 config, load stored metrics.json and
    build a comparison table showing the effect of each addition.
    """
    print("\n" + "="*65)
    print("  TASK 1: Metrics Table")
    print("="*65)
    out_dir.mkdir(parents=True, exist_ok=True)

    p1_cfgs = scan_p1_configs(results_dir, folds, outer_splits)
    p2_cfgs = scan_p2_configs(results_dir, folds, outer_splits)
    print(f"  Found {len(p1_cfgs)} P1 configs, {len(p2_cfgs)} P2 configs")

    # Build flat rows: one per (config_tag, fold, split, metric)
    rows = []
    for cfg in p1_cfgs:
        metrics = load_stored_metrics(cfg["metrics_path"])
        for split, sm in metrics.items():
            if split in ("probs", "labels", "_probs", "_labels"):
                continue
            if not isinstance(sm, dict):
                continue
            row = {
                "phase": "P1", "tag": cfg["tag"],
                "mod": cfg.get("mod", ""), "variant": "",
                "fold": cfg["fold"], "split": split,
            }
            for m in METRICS:
                row[m] = sm.get(m, float("nan"))
            rows.append(row)

    for cfg in p2_cfgs:
        metrics = load_stored_metrics(cfg["metrics_path"])
        for split, sm in metrics.items():
            if not isinstance(sm, dict): continue
            vtag = cfg["vtag"]
            # Parse R, K from iterative tags
            r_val = re.search(r"_r(\d+)", vtag)
            k_val = re.search(r"_k(\d+)", vtag)
            base  = re.sub(r"_r\d+_k\d+", "", vtag)
            row = {
                "phase": "P2", "tag": cfg["tag"],
                "mod": "", "variant": vtag,
                "base_variant": base,
                "R": int(r_val.group(1)) if r_val else None,
                "K": int(k_val.group(1)) if k_val else None,
                "fold": cfg["fold"], "split": split,
            }
            for m in METRICS:
                row[m] = sm.get(m, float("nan"))
            rows.append(row)

    if not rows:
        print("  [warn] No metrics found. Check results_dir and fold structure.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "all_metrics.csv", index=False)

    # ── Summary table: mean ± std across folds, per split ─────────
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        if sub.empty: continue
        _write_metrics_summary_txt(sub, split, out_dir)
        _plot_metrics_heatmap(sub, split, out_dir)
        _plot_ablation_bar(sub, split, out_dir)

    print(f"  Outputs → {out_dir}")


def _mean_std(series: pd.Series) -> Tuple[float, float]:
    v = series.dropna().values
    if len(v) == 0: return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v, ddof=0))


def _write_metrics_summary_txt(sub: pd.DataFrame, split: str,
                                out_dir: Path):
    tags = sorted(sub["tag"].unique())
    lines = []
    lines.append("═"*110)
    lines.append(f"  {split.upper()} RESULTS  — mean ± std across folds")
    lines.append("═"*110)
    hdr = f"  {'Config':<36}" + "".join(f"  {m.upper()[:5]:<14}" for m in METRICS)
    lines.append(hdr)
    lines.append("─"*110)

    last_phase = None
    for tag in tags:
        phase = "P1" if tag.startswith("P1") else "P2"
        if phase != last_phase:
            if last_phase is not None:
                lines.append("─"*110)
            lines.append(f"  ── {phase} ──")
            last_phase = phase
        vdata = sub[sub["tag"] == tag]
        parts = [f"  {tag:<36}"]
        for m in METRICS:
            mn, sd = _mean_std(vdata[m])
            if np.isnan(mn): parts.append(f"  {'N/A':<14}")
            else:             parts.append(f"  {f'{mn:.4f}±{sd:.4f}':<14}")
        lines.append("".join(parts))

    lines.append("═"*110)
    txt = "\n".join(lines)
    p = out_dir / f"summary_{split}.txt"
    p.write_text(txt)
    print(f"  {split}: {p}")


def _plot_metrics_heatmap(sub: pd.DataFrame, split: str, out_dir: Path):
    tags = sorted(sub["tag"].unique())
    data = []
    for tag in tags:
        vdata = sub[sub["tag"] == tag]
        row = {"tag": tag}
        for m in METRICS:
            row[m], _ = _mean_std(vdata[m])
        data.append(row)
    if not data: return
    df_plot = pd.DataFrame(data).set_index("tag")[METRICS]

    fig, ax = plt.subplots(figsize=(len(METRICS)*1.6 + 2, max(4, len(tags)*0.5 + 1)))
    if HAS_SNS:
        sns.heatmap(df_plot.astype(float), annot=True, fmt=".3f",
                    cmap="RdYlGn", vmin=0.4, vmax=1.0, linewidths=0.4,
                    annot_kws={"size": 8}, ax=ax)
    else:
        im = ax.imshow(df_plot.values.astype(float), cmap="RdYlGn",
                       vmin=0.4, vmax=1.0, aspect="auto")
        for i, row in enumerate(df_plot.values):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:.3f}" if not np.isnan(v) else "",
                        ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(METRICS)))
        ax.set_yticks(range(len(tags)))
        ax.set_xticklabels(METRICS)
        ax.set_yticklabels(df_plot.index, fontsize=8)
        plt.colorbar(im, ax=ax)

    ax.set_title(f"Mean metrics across folds — {split.upper()}", fontsize=11)
    plt.yticks(rotation=0, fontsize=8)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    p = out_dir / f"heatmap_{split}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  heatmap saved: {p}")


def _plot_ablation_bar(sub: pd.DataFrame, split: str, out_dir: Path):
    """Bar chart: BAcc and AUC side by side, sorted by mean test BAcc."""
    tags = sorted(sub["tag"].unique())
    p1_tags = [t for t in tags if t.startswith("P1")]
    p2_tags = [t for t in tags if t.startswith("P2")]

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(tags)*0.9 + 2), 6))
    for ax, metric, ylabel in zip(axes, ["bacc", "auc"],
                                  ["Balanced Accuracy", "ROC AUC"]):
        means, stds, labels_plot, colors_plot = [], [], [], []
        for tag in p1_tags + ["---"] + p2_tags:
            if tag == "---":
                means.append(np.nan); stds.append(0.0)
                labels_plot.append(""); colors_plot.append("none")
                continue
            vdata = sub[sub["tag"] == tag]
            mn, sd = _mean_std(vdata[metric])
            means.append(mn); stds.append(sd)
            labels_plot.append(tag.replace("P1_","").replace("P2_",""))
            colors_plot.append("#4e79a7" if tag.startswith("P1") else "#f28e2b")

        x = np.arange(len(means))
        for i, (mn, sd, color) in enumerate(zip(means, stds, colors_plot)):
            if np.isnan(mn): continue
            ax.bar(i, mn, yerr=sd, capsize=4, color=color, alpha=0.82,
                   edgecolor="white", width=0.7)
            ax.text(i, mn + sd + 0.005, f"{mn:.3f}", ha="center",
                    va="bottom", fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels_plot, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel(f"Mean {ylabel} ± std", fontsize=10)
        ax.set_title(f"{ylabel} — {split.upper()}", fontsize=11)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.35)
        ax.legend(handles=[Patch(color="#4e79a7", label="Phase 1"),
                            Patch(color="#f28e2b", label="Phase 2")],
                  fontsize=8)

    fig.suptitle(f"Ablation comparison — {split.upper()} split (mean ± std)",
                 fontsize=12)
    fig.tight_layout()
    p = out_dir / f"ablation_bar_{split}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ablation bar saved: {p}")


# ══════════════════════════════════════════════════════════════════
# TASK 2: REP MAPS (PCA / UMAP colored by combo + label)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_p1_reps(model: SingleModalMIL, bags: Dict,
                mod: str, device: torch.device) -> Optional[torch.Tensor]:
    t = bags.get(mod)
    if t is None: return None
    rep, _ = model.rep_and_attn(t.to(device))
    return rep.cpu()


_TRAIN_MOD_CACHE: Dict = {}

def _get_train_module():
    """Import train_mm_abmil_v6.py once via importlib and cache it.
    Safe because the script is main-guarded (if __name__ == '__main__': main())."""
    if "mod" not in _TRAIN_MOD_CACHE:
        import importlib.util
        train_path = Path(__file__).parent.parent / "train_mm_abmil_v6.py"
        spec = importlib.util.spec_from_file_location("_train_v6", str(train_path))
        tm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tm)
        _TRAIN_MOD_CACHE["mod"] = tm
    return _TRAIN_MOD_CACHE["mod"]


def _vtag_to_variant_params(vtag: str):
    """Parse a variant tag back to (variant_str, iter_r, slot_k).

    Examples:
      crossattn_k8      → (crossattn,     2, 8)
      crossattn_k8_cls  → (crossattn_cls, 2, 8)
      crossmodal_k16    → (crossmodal,    2, 16)
      iterative_r2_k8   → (iterative,     2, 8)
      iterative_r3_k16_cls → (iterative_cls, 3, 16)
      early / early_cls / late / middle / middle_cls → as-is
    """
    r_m = re.search(r'_r(\d+)', vtag)
    k_m = re.search(r'_k(\d+)', vtag)
    iter_r = int(r_m.group(1)) if r_m else 2
    slot_k = int(k_m.group(1)) if k_m else 8
    variant = re.sub(r'_r\d+', '', vtag)
    variant = re.sub(r'_k\d+', '', variant)
    return variant, iter_r, slot_k


@torch.no_grad()
def _try_p2_reps_hook(
    ckpt_path: Path, all_recs: List[dict],
    bags: Dict, device: torch.device,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract P2 pre-head embeddings via a forward hook on the last Linear(*, 1).

    Handles both:
      • full nn.Module saves  (uncommon)
      • state-dict saves      (normal — reconstructs model via build_p2_model)
    Returns (reps, labels, combos, splits) or (None, None, None, None).
    """
    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None, None, None, None

    model = None

    if isinstance(obj, nn.Module):
        model = obj

    elif isinstance(obj, dict):
        # State dict — reconstruct the model skeleton from the training script
        state = obj.get("model", obj)   # handle {model: state_dict} or raw state_dict
        vtag = ckpt_path.stem.replace("model_", "")
        p1_dir = ckpt_path.parent.parent / "phase1"
        variant, iter_r, slot_k = _vtag_to_variant_params(vtag)
        try:
            tm = _get_train_module()
            model = tm.build_p2_model(variant, p1_dir,
                                      iter_r=iter_r, slot_k=slot_k)
            incompatible = model.load_state_dict(state, strict=False)
            if incompatible.unexpected_keys:
                print(f"    [warn] {vtag}: {len(incompatible.unexpected_keys)} unexpected keys")
        except Exception as e:
            print(f"    [warn] Could not reconstruct {vtag}: {e}")
            return None, None, None, None

    if model is None:
        return None, None, None, None

    model = model.to(device).eval()

    # Find last Linear with out_features==1 (the classification head)
    head_layer = None
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.out_features == 1:
            head_layer = m
    if head_layer is None:
        return None, None, None, None

    rep_buf: Dict = {}
    hook_h = head_layer.register_forward_hook(
        lambda m, inp, out: rep_buf.update({"rep": inp[0].detach().cpu().float()}))

    reps, labels, combos, splits_list = [], [], [], []
    for rec in all_recs:
        bag = bags.get(rec["stem"], {})
        if not any(bag.get(m) is not None for m in MODALITIES):
            continue
        rep_buf.clear()
        try:
            bags_dev = {m: bag[m].to(device) if bag.get(m) is not None else None
                        for m in MODALITIES}
            _ = model(bags_dev, device)
        except Exception:
            continue
        if "rep" not in rep_buf:
            continue
        reps.append(rep_buf["rep"].numpy().flatten())
        labels.append(rec["label"])
        combos.append(rec.get("combo", "NONE"))
        splits_list.append(rec["split"])

    hook_h.remove()
    if not reps:
        return None, None, None, None
    return np.stack(reps), np.array(labels), np.array(combos), np.array(splits_list)


def task_rep_maps(results_dir: Path, samples_dir: str, splits_csv: str,
                  folds: List[int], out_dir: Path,
                  hidden_dim: int = 256, dropout: float = 0.4,
                  outer_splits: Optional[List[int]] = None):
    print("\n" + "="*65)
    print("  TASK 2: Representation Maps (PCA / UMAP)")
    print("="*65)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outer_split = outer_splits[0] if outer_splits else None

    for fold in folds:
        print(f"\n  Fold {fold}")
        if outer_split is not None:
            p1_dir = results_dir / f"split{outer_split}_fold{fold}" / "phase1"
        else:
            p1_dir = results_dir / f"fold_{fold}" / "phase1"
        if not p1_dir.exists(): continue

        # Load Phase 1 models
        p1_models = {}
        for mod in MODALITIES:
            ckpt = p1_dir / mod / "best_model.pt"
            m    = load_p1_model(ckpt, _feat_dim(mod), hidden_dim, dropout, device)
            if m is not None:
                p1_models[mod] = m

        if not p1_models:
            print(f"  [skip] No P1 models found for fold {fold}")
            continue

        # Load records for all splits
        all_recs = build_records(splits_csv, fold, outer_split=outer_split)
        stems    = list({r["stem"] for r in all_recs})
        print(f"  Loading {len(stems)} bags …")
        bags     = preload_bags(stems, samples_dir)
        all_recs = update_presence(all_recs, bags)

        # Build representations
        print(f"  Computing P1 representations …")
        for mod, model in p1_models.items():
            reps, labels, combos, splits_list = [], [], [], []
            for rec in all_recs:
                rep = get_p1_reps(model, bags.get(rec["stem"], {}), mod, device)
                if rep is None: continue
                reps.append(rep.numpy())
                labels.append(rec["label"])
                combos.append(rec.get("combo", "NONE"))
                splits_list.append(rec["split"])

            if not reps:
                continue
            reps_arr   = np.stack(reps)
            labels_arr = np.array(labels)
            combos_arr = np.array(combos)
            splits_arr = np.array(splits_list)

            _plot_embedding(
                reps_arr, labels_arr, combos_arr, splits_arr,
                title=f"P1_{mod}  fold{fold}",
                out_path=out_dir / f"repmap_P1_{mod}_fold{fold}.png",
                method="pca_umap",
            )
            _gc()

        # ── Concatenated multimodal P1 embedding ──────────────────
        print(f"  Computing multimodal P1 concat embedding …")
        mm_reps_l, mm_labels_l, mm_combos_l, mm_splits_l = [], [], [], []
        for rec in all_recs:
            stem_bags = bags.get(rec["stem"], {})
            parts = []
            for mod in MODALITIES:
                if mod in p1_models:
                    r = get_p1_reps(p1_models[mod], stem_bags, mod, device)
                    parts.append(r.numpy() if r is not None
                                 else np.zeros(hidden_dim, dtype=np.float32))
                else:
                    parts.append(np.zeros(hidden_dim, dtype=np.float32))
            if all(p.sum() == 0 for p in parts):
                continue
            mm_reps_l.append(np.concatenate(parts))
            mm_labels_l.append(rec["label"])
            mm_combos_l.append(rec.get("combo", "NONE"))
            mm_splits_l.append(rec["split"])
        if mm_reps_l:
            _plot_embedding(
                np.stack(mm_reps_l), np.array(mm_labels_l),
                np.array(mm_combos_l), np.array(mm_splits_l),
                title=f"Multimodal P1-concat  fold{fold}",
                out_path=out_dir / f"repmap_P1_multimodal_concat_fold{fold}.png",
                method="pca_umap",
            )
            _gc()

        # ── P2 model embeddings via forward hook ──────────────────
        p2_dir_path = (results_dir / f"split{outer_split}_fold{fold}" / "phase2"
                       if outer_split is not None
                       else results_dir / f"fold_{fold}" / "phase2")
        if p2_dir_path.exists():
            for mf in sorted(p2_dir_path.glob("model_*.pt")):
                vtag = mf.stem.replace("model_", "")
                print(f"  Trying P2 hook reps: {vtag} …")
                p2r, p2l, p2c, p2s = _try_p2_reps_hook(mf, all_recs, bags, device)
                if p2r is not None and len(p2r) > 5:
                    _plot_embedding(
                        p2r, p2l, p2c, p2s,
                        title=f"P2_{vtag}  fold{fold}",
                        out_path=out_dir / f"repmap_P2_{vtag}_fold{fold}.png",
                        method="pca_umap",
                    )
                    _gc()

        del bags; _gc()

    print(f"  Rep map outputs → {out_dir}")


def _plot_embedding(X: np.ndarray, labels: np.ndarray,
                    combos: np.ndarray, splits: np.ndarray,
                    title: str, out_path: Path,
                    method: str = "pca_umap"):
    """PCA → optionally UMAP, then 2×2 panel: combo/label × train/test."""
    # Reduce dimensions
    if X.shape[1] > 50:
        pca = PCA(n_components=min(50, X.shape[0]-1, X.shape[1]))
        X_red = pca.fit_transform(X)
        var_exp = pca.explained_variance_ratio_[:2].sum()
    else:
        X_red = X; var_exp = 1.0

    if HAS_UMAP and method == "pca_umap" and X.shape[0] > 10:
        try:
            reducer = UMAPTransform(n_components=2, random_state=42,
                                    n_neighbors=min(15, X.shape[0]-1))
            Z = reducer.fit_transform(X_red)
            coord_label = "UMAP"
        except Exception:
            Z = X_red[:, :2]; coord_label = "PCA"
    else:
        pca2 = PCA(n_components=2)
        Z = pca2.fit_transform(X_red)
        coord_label = f"PCA (var={var_exp:.2f})"

    unique_combos = sorted(set(combos))
    combo_c = {c: COMBO_COLORS.get(c, "#aaaaaa") for c in unique_combos}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    kw = dict(s=12, alpha=0.55, linewidths=0, rasterized=True)

    for row_idx, col_by in enumerate(["combo", "label"]):
        for col_idx, split_sel in enumerate(["train", "test"]):
            ax = axes[row_idx, col_idx]
            mask = splits == split_sel
            if not mask.any():
                ax.set_visible(False); continue

            Zs = Z[mask]; ls = labels[mask]; cs = combos[mask]
            ax.set_facecolor("#f8f8f8")

            if col_by == "combo":
                for combo in unique_combos:
                    m2 = cs == combo
                    if m2.any():
                        ax.scatter(Zs[m2, 0], Zs[m2, 1],
                                   color=combo_c[combo], label=combo, **kw)
                ax.legend(fontsize=6, ncol=2, loc="upper right",
                          markerscale=2, framealpha=0.7)
                ax.set_title(f"{split_sel} — by modality combo", fontsize=10)
            else:
                for lv, color, lname in [(0, "#5c9be0", "Neg A0"),
                                          (1, "#e05c5c", "Pos A1/A2")]:
                    m2 = ls == lv
                    if m2.any():
                        ax.scatter(Zs[m2, 0], Zs[m2, 1],
                                   color=color, label=lname, **kw)
                ax.legend(fontsize=9, markerscale=2)
                ax.set_title(f"{split_sel} — by ACR label", fontsize=10)

            ax.set_xlabel(f"{coord_label} 1", fontsize=8)
            ax.set_ylabel(f"{coord_label} 2", fontsize=8)
            ax.grid(True, lw=0.3, alpha=0.5)

    n_pos = int(labels.sum()); n_neg = len(labels) - n_pos
    fig.suptitle(f"{title}\n"
                 f"n={len(labels)}  neg={n_neg}  pos={n_pos}  "
                 f"combos={len(unique_combos)}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════
# TASK 3: HE ATTENTION MAPS
# ══════════════════════════════════════════════════════════════════

def task_attn_maps(results_dir: Path, samples_dir: str, splits_csv: str,
                   folds: List[int], out_dir: Path,
                   hidden_dim: int = 256, dropout: float = 0.4,
                   adata_path: Optional[str] = None,
                   cluster_col: str = "subclusters_merged",
                   max_patches_umap: int = 5000,
                   n_per_split: int = 20,
                   outer_splits: Optional[List[int]] = None):
    """
    For each fold:
      • Load HE Phase-1 model
      • Compute pre-softmax attention logits for each bag
      • If adata provided: use stored UMAP coords, cluster labels
      • Otherwise: compute PCA on backbone features
      • Plot: global (all split patches) + per-sample (top-N patients)
    """
    print("\n" + "="*65)
    print("  TASK 3: HE Attention Maps")
    print("="*65)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = None
    if adata_path and Path(adata_path).exists():
        try:
            import anndata as ad
            print(f"  Loading AnnData from {adata_path} …")
            adata = ad.read_h5ad(adata_path, backed="r")
            print(f"  AnnData shape: {adata.shape}")
            if cluster_col not in adata.obs.columns:
                print(f"  [warn] cluster_col '{cluster_col}' not in adata.obs — skipping clusters")
                cluster_col = None
        except ImportError:
            print("  [warn] anndata not installed — no adata features")

    outer_split = outer_splits[0] if outer_splits else None

    for fold in folds:
        print(f"\n  Fold {fold}")
        if outer_split is not None:
            he_ckpt = results_dir / f"split{outer_split}_fold{fold}" / "phase1" / "HE" / "best_model.pt"
        else:
            he_ckpt = results_dir / f"fold_{fold}" / "phase1" / "HE" / "best_model.pt"
        model   = load_p1_model(he_ckpt, 1024, hidden_dim, dropout, device)
        if model is None:
            print(f"  [skip] No HE model for fold {fold}"); continue

        all_recs  = build_records(splits_csv, fold, outer_split=outer_split)
        he_recs   = [r for r in all_recs if r.get("has_HE", False)]
        if not he_recs:
            he_recs = all_recs
        stems = list({r["stem"] for r in he_recs})
        print(f"  Loading {len(stems)} HE bags …")
        bags   = preload_bags(stems, samples_dir)
        he_recs = update_presence(he_recs, bags)

        fold_attn_dir = out_dir / f"fold_{fold}"
        fold_attn_dir.mkdir(exist_ok=True)

        for split in ["train", "val", "test"]:
            split_recs = [r for r in he_recs if r["split"] == split]
            if not split_recs: continue
            _plot_attn_global(
                model, split_recs, bags, device, split, fold,
                fold_attn_dir, adata, cluster_col, max_patches_umap)
            _plot_attn_per_sample(
                model, split_recs, bags, device, split, fold,
                fold_attn_dir, n_per_split)

        del bags; _gc()

    print(f"  Attention map outputs → {out_dir}")


@torch.no_grad()
def _get_he_attn(model: SingleModalMIL, bag: torch.Tensor,
                 device: torch.device):
    t      = bag.to(device)
    h      = model.encoder.backbone(t)                       # (N, H)
    logits = model.encoder.attn_logits(t)                    # (N,) pre-softmax
    alpha  = F.softmax(
        model.encoder.att_w(model.encoder.att_drop(
            model.encoder.att_V(h) * model.encoder.att_U(h))), dim=0).squeeze(1)
    return h.cpu().numpy(), logits.cpu().numpy(), alpha.cpu().numpy()


def _plot_attn_global(model, recs, bags, device, split, fold,
                      out_dir, adata, cluster_col, max_patches):
    print(f"  Global attn map: {split} ({len(recs)} patients) …")
    all_h, all_logits, all_alpha, all_labels = [], [], [], []
    all_umap, all_clusters = [], []
    use_adata_umap = False

    for rec in recs:
        bag = bags.get(rec["stem"], {}).get("HE")
        if bag is None: continue
        h, logits, alpha = _get_he_attn(model, bag, device)
        all_h.append(h); all_logits.append(logits); all_alpha.append(alpha)
        all_labels.extend([rec["label"]] * len(logits))

        # Try to get pre-computed UMAP from adata
        if adata is not None:
            pid = rec.get("patient_id", rec["stem"])
            mask = (adata.obs["record_id"] == pid).values
            idx  = np.where(mask)[0]
            if len(idx) > 0 and "X_umap" in adata.obsm:
                # subsample to match bag size (take first N)
                n = min(len(logits), len(idx))
                all_umap.append(adata.obsm["X_umap"][idx[:n]])
                if cluster_col and cluster_col in adata.obs.columns:
                    all_clusters.extend(adata.obs[cluster_col].values[idx[:n]])
                use_adata_umap = True

    if not all_h: return

    H      = np.vstack(all_h)
    logits = np.concatenate(all_logits)
    alpha  = np.concatenate(all_alpha)
    labels = np.array(all_labels)

    # Subsample for display
    if len(H) > max_patches:
        idx  = np.random.choice(len(H), max_patches, replace=False)
        H    = H[idx]; logits = logits[idx]; alpha = alpha[idx]
        labels = labels[idx]
        if use_adata_umap and all_umap:
            all_umap_cat = np.vstack(all_umap)
            if len(all_umap_cat) > max_patches:
                all_umap_cat = all_umap_cat[idx]

    # Get 2D coords
    if use_adata_umap and all_umap:
        try:
            U = np.vstack(all_umap)[:len(H)]
            coord_label = "UMAP (precomputed)"
        except Exception:
            use_adata_umap = False

    if not use_adata_umap:
        pca = PCA(n_components=2)
        U   = pca.fit_transform(H)
        coord_label = "PCA (backbone)"

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    kw_bg = dict(s=2, alpha=0.15, linewidths=0, rasterized=True)
    kw    = dict(s=4, alpha=0.5,  linewidths=0, rasterized=True)

    # Panel 1: raw logits (all)
    sc = axes[0,0].scatter(U[:,0], U[:,1], c=logits, cmap="plasma", **kw)
    plt.colorbar(sc, ax=axes[0,0], label="pre-softmax logit")
    axes[0,0].set_title(f"All patches — raw logits\n{split} fold{fold}")

    # Panel 2: top 1% by logit
    t1 = np.percentile(logits, 99)
    m1 = logits >= t1
    axes[0,1].scatter(U[~m1,0], U[~m1,1], c="lightgrey", **kw_bg)
    sc = axes[0,1].scatter(U[m1,0], U[m1,1], c=logits[m1], cmap="Reds",
                            s=10, alpha=0.8, linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=axes[0,1])
    axes[0,1].set_title(f"Top 1% logits  (n={m1.sum():,})\n{split} fold{fold}")

    # Panel 3: top 0.05% by logit
    t005 = np.percentile(logits, 99.95)
    m005 = logits >= t005
    axes[0,2].scatter(U[~m005,0], U[~m005,1], c="lightgrey", **kw_bg)
    sc = axes[0,2].scatter(U[m005,0], U[m005,1], c=logits[m005], cmap="hot",
                            s=14, alpha=0.9, linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=axes[0,2])
    axes[0,2].set_title(f"Top 0.05% logits  (n={m005.sum():,})\n{split} fold{fold}")

    # Panel 4: bag-softmax weights
    sc = axes[1,0].scatter(U[:,0], U[:,1], c=alpha, cmap="viridis", **kw)
    plt.colorbar(sc, ax=axes[1,0], label="ABMIL weight (bag-softmax)")
    axes[1,0].set_title("Bag-softmax weights")

    # Panel 5: ACR label
    lc = np.array([LABEL_COLORS[l] for l in labels])
    axes[1,1].scatter(U[:,0], U[:,1], c=lc, **kw)
    axes[1,1].legend(handles=[Patch(color="#5c9be0", label="Neg A0"),
                               Patch(color="#e05c5c", label="Pos A1/A2")],
                     fontsize=9, markerscale=2)
    axes[1,1].set_title("ACR label")

    # Panel 6: cluster (if available)
    if use_adata_umap and all_clusters and len(all_clusters) >= len(U):
        clusters = np.array(all_clusters)[:len(U)]
        unique_c = sorted(set(clusters), key=str)
        cmap_c   = plt.cm.tab20(np.linspace(0, 1, max(len(unique_c), 20)))
        cc       = {c: cmap_c[i%20] for i, c in enumerate(unique_c)}
        colors_c = np.array([cc[c] for c in clusters])
        axes[1,2].scatter(U[:,0], U[:,1], c=colors_c, **kw)
        handles  = [Patch(color=cc[c], label=str(c)) for c in unique_c[:20]]
        axes[1,2].legend(handles=handles, fontsize=5, ncol=2,
                         loc="upper right", markerscale=2)
        axes[1,2].set_title("Cell cluster")
    else:
        axes[1,2].set_visible(False)

    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xlabel(coord_label.split()[0] + " 1", fontsize=8)
            ax.set_ylabel(coord_label.split()[0] + " 2", fontsize=8)
            ax.grid(True, lw=0.3, alpha=0.4)

    n_pos = int(labels.sum())
    fig.suptitle(f"HE Attention — {split.upper()} | Fold {fold}\n"
                 f"{len(recs)} patients  {len(H):,} patches  "
                 f"neg={len(labels)-n_pos}  pos={n_pos}",
                 fontsize=12)
    fig.tight_layout()
    p = out_dir / f"he_attn_global_{split}_fold{fold}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


def _plot_attn_per_sample(model, recs, bags, device, split, fold,
                          out_dir, n_per_split):
    """Per-patient 3-panel attention plot."""
    per_dir = out_dir / "per_sample"; per_dir.mkdir(exist_ok=True)
    pos_recs = [r for r in recs if r["label"] == 1]
    neg_recs = [r for r in recs if r["label"] == 0]
    half     = n_per_split // 2
    chosen   = (pos_recs[:half] + neg_recs[:half])[:n_per_split]

    for rec in chosen:
        bag = bags.get(rec["stem"], {}).get("HE")
        if bag is None: continue
        h, logits, alpha = _get_he_attn(model, bag, device)

        pca2 = PCA(n_components=2)
        U    = pca2.fit_transform(h)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        kw = dict(s=10, alpha=0.7, linewidths=0, rasterized=True)
        kw_bg = dict(s=4, alpha=0.15, linewidths=0, rasterized=True, c="lightgrey")

        sc = axes[0].scatter(U[:,0], U[:,1], c=logits, cmap="plasma", **kw)
        plt.colorbar(sc, ax=axes[0], label="raw logit")
        axes[0].set_title("Raw attention logits")

        t1  = np.percentile(logits, 99)
        m1  = logits >= t1
        axes[1].scatter(U[~m1,0], U[~m1,1], **kw_bg)
        sc = axes[1].scatter(U[m1,0], U[m1,1], c=logits[m1],
                              cmap="Reds", s=16, alpha=0.9, linewidths=0)
        plt.colorbar(sc, ax=axes[1])
        axes[1].set_title(f"Top 1% (n={m1.sum()})")

        sc = axes[2].scatter(U[:,0], U[:,1], c=alpha, cmap="viridis", **kw)
        plt.colorbar(sc, ax=axes[2], label="ABMIL softmax")
        axes[2].set_title("ABMIL attention weights")

        for ax in axes:
            ax.set_xlabel("PC1", fontsize=8); ax.set_ylabel("PC2", fontsize=8)
            ax.grid(True, lw=0.3, alpha=0.4)

        lbl_str = "Pos_A1A2" if rec["label"] == 1 else "Neg_A0"
        fig.suptitle(f"HE Attention — {rec['patient_id']} | {lbl_str} | "
                     f"{split} | Fold {fold}  ({len(logits)} patches)")
        fig.tight_layout()
        pid_safe = re.sub(r"[^A-Za-z0-9_\-]", "_", str(rec["patient_id"]))
        p = per_dir / f"{split}_{pid_safe}_{lbl_str}_fold{fold}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)

    print(f"  Per-sample: {len(chosen)} plots → {per_dir}")


# ══════════════════════════════════════════════════════════════════
# TASK 4: COMBO TABLE — per modality combination per P2 variant
# ══════════════════════════════════════════════════════════════════

def task_combo_table(results_dir: Path, samples_dir: str, splits_csv: str,
                     folds: List[int], out_dir: Path,
                     hidden_dim: int = 256, dropout: float = 0.4,
                     min_n: int = 3,
                     outer_splits: Optional[List[int]] = None):
    """
    For every P2 variant × fold, load stored probs/labels from metrics JSON,
    then split by modality combination and compute per-combo metrics.
    Produces: CSV, text table, heatmap, and per-metric bar charts.
    """
    print("\n" + "="*65)
    print("  TASK 4: Combo Table")
    print("="*65)
    out_dir.mkdir(parents=True, exist_ok=True)

    p2_cfgs = scan_p2_configs(results_dir, folds, outer_splits)
    if not p2_cfgs:
        print("  [warn] No P2 configs found"); return

    # Load P1 per-patient predictions for baseline comparison
    print(f"  Loading P1 per-patient predictions …")
    p1_stem_preds = load_p1_preds_per_stem(
        results_dir, splits_csv, folds, outer_splits)

    # We need a per-stem combo map: load records once per fold
    fold_stem_combos: Dict[int, Dict[str, Dict]] = {}
    for fold in folds:
        outer_split = outer_splits[0] if outer_splits else None
        recs  = build_records(splits_csv, fold, outer_split=outer_split)
        stems = list({r["stem"] for r in recs})
        print(f"  Fold {fold}: loading {len(stems)} bags for combo detection …")
        bags  = preload_bags(stems, samples_dir, quiet=True)
        recs  = update_presence(recs, bags)
        fold_stem_combos[fold] = {
            r["stem"]: {"combo": r["combo"], "split": r["split"], "label": r["label"]}
            for r in recs}
        del bags; _gc()

    # For each P2 config, load probs+labels from stored JSON
    all_rows = []
    for cfg in p2_cfgs:
        fold    = cfg["fold"]
        vtag    = cfg["vtag"]
        metrics = load_stored_metrics(cfg["metrics_path"])
        stem_info = fold_stem_combos.get(fold, {})

        for split, sm in metrics.items():
            if not isinstance(sm, dict): continue
            probs  = sm.get("probs")
            if probs is None: continue

            # To assign combos we need the positional order of stems.
            # We reproduce build_records order to align with stored probs.
            split_recs = [r for r in build_records(
                              splits_csv, fold, outer_split=outer_split)
                          if r["split"] == split]
            if len(split_recs) != len(probs):
                # Try falling back to label alignment
                continue

            # Update presence from bags
            stems_split = list({r["stem"] for r in split_recs})
            bags = preload_bags(stems_split, samples_dir, quiet=True)
            split_recs = update_presence(split_recs, bags)
            del bags; _gc()

            for prob, rec in zip(probs, split_recs):
                all_rows.append({
                    "fold": fold, "vtag": vtag,
                    "base_variant": re.sub(r"_r\d+_k\d+","", vtag),
                    "split": split,
                    "stem": rec["stem"],
                    "combo": rec["combo"],
                    "prob": float(prob),
                    "label": rec["label"],  # always from CSV
                    **{_pres_col(m): rec.get(_pres_col(m), False) for m in MODALITIES},
                })

    if not all_rows:
        print("  [warn] No probs/labels found in stored metrics. "
              "Ensure Phase 2 metrics JSONs contain 'probs' and 'labels'.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "combo_predictions.csv", index=False)

    # Compute per-combo metrics, aggregated over folds
    combo_rows = []
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        if sub.empty: continue
        for vtag in sorted(sub["vtag"].unique()):
            vsub = sub[sub["vtag"] == vtag]
            for combo in sorted(vsub["combo"].unique()):
                csub = vsub[vsub["combo"] == combo]
                if len(csub) < min_n: continue
                # Aggregate over folds: pool all probs/labels
                probs_all  = csub["prob"].values
                labels_all = csub["label"].values
                n = len(labels_all); n_pos = int(labels_all.sum())
                m = _safe_metrics(labels_all, probs_all)
                combo_rows.append({
                    "split": split, "vtag": vtag,
                    "base_variant": re.sub(r"_r\d+_k\d+","", vtag),
                    "combo": combo, "n": n, "n_pos": n_pos, "n_neg": n-n_pos,
                    **m,
                })

    # Add P1 per-modality rows to combo_rows for comparison
    outer_split_for_p1 = outer_splits[0] if outer_splits else None
    for cfg in scan_p1_configs(results_dir, folds, outer_splits):
        mod = cfg["mod"]; fold = cfg["fold"]
        outer_split_cfg = cfg["outer_split"]
        stem_info = fold_stem_combos.get(fold, {})
        for split in ["train", "val", "test"]:
            split_recs = [r for r in build_records(splits_csv, fold, outer_split=outer_split_cfg)
                          if r["split"] == split]
            sub_rows = []
            for rec in split_recs:
                key = (fold, mod, rec["stem"])
                pred = p1_stem_preds.get(key)
                if pred is None: continue
                combo = stem_info.get(rec["stem"], {}).get("combo", "NONE")
                sub_rows.append({"prob": pred["prob"], "label": rec["label"], "combo": combo})
            if not sub_rows: continue
            sub_df = pd.DataFrame(sub_rows)
            for combo in sorted(sub_df["combo"].unique()):
                csub = sub_df[sub_df["combo"] == combo]
                if len(csub) < min_n: continue
                m = _safe_metrics(csub["label"].values, csub["prob"].values)
                n = len(csub); n_pos = int(csub["label"].sum())
                combo_rows.append({
                    "split": split, "vtag": f"P1_{mod}",
                    "base_variant": f"P1_{mod}",
                    "combo": combo, "n": n, "n_pos": n_pos, "n_neg": n - n_pos,
                    **m,
                })

    if not combo_rows:
        print("  [warn] No combo rows computed"); return

    df_combo = pd.DataFrame(combo_rows)
    df_combo.to_csv(out_dir / "combo_metrics.csv", index=False)

    for split in ["train", "val", "test"]:
        sc = df_combo[df_combo["split"] == split]
        if sc.empty: continue
        _write_combo_table_txt(sc, split, out_dir)
        _plot_combo_heatmap(sc, split, out_dir)
        for metric in ["auc", "bacc", "auprc"]:
            _plot_combo_bar(sc, split, metric, out_dir)
        _plot_combo_count_table(sc, split, out_dir)

    print(f"  Combo table outputs → {out_dir}")


def _safe_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    base = {"auc": float("nan"), "auprc": float("nan"),
            "bacc": float("nan"), "mcc": float("nan"),
            "sens": float("nan"), "spec": float("nan")}
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return base
    try:
        preds = (probs >= 0.5).astype(int)
        base["auc"]  = roc_auc_score(labels, probs)
        base["auprc"] = average_precision_score(labels, probs)
        base["bacc"] = balanced_accuracy_score(labels, preds)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
        num = tp*tn - fp*fn
        den = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5
        base["mcc"]  = float(num/den) if den > 0 else 0.0
        base["sens"] = tp / max(tp+fn, 1)
        base["spec"] = tn / max(tn+fp, 1)
    except Exception:
        pass
    return base


def _write_combo_table_txt(df: pd.DataFrame, split: str, out_dir: Path):
    variants = sorted(df["vtag"].unique())
    combos   = sorted(df["combo"].unique())
    lines    = []
    lines.append("═"*120)
    lines.append(f"  MODALITY COMBO METRICS — {split.upper()}")
    lines.append("═"*120)
    hdr = f"  {'Combo':<28}  {'N':>6}  {'N_pos':>6}"
    for m in ["auc","auprc","bacc","mcc","sens","spec"]:
        hdr += f"  {m.upper():<10}"
    hdr += f"  Variant"
    lines.append(hdr)
    lines.append("─"*120)

    for vtag in variants:
        vsub = df[df["vtag"] == vtag]
        lines.append(f"\n  ── {vtag} ──")
        for combo in combos:
            csub = vsub[vsub["combo"] == combo]
            if csub.empty: continue
            r = csub.iloc[0]
            row = f"  {combo:<28}  {int(r['n']):>6}  {int(r['n_pos']):>6}"
            for m in ["auc","auprc","bacc","mcc","sens","spec"]:
                v = r.get(m, float("nan"))
                row += f"  {f'{v:.4f}' if not np.isnan(v) else 'N/A':<10}"
            row += f"  {vtag}"
            lines.append(row)

    lines.append("═"*120)
    p = out_dir / f"combo_table_{split}.txt"
    p.write_text("\n".join(lines))
    print(f"  Combo table: {p}")


def _plot_combo_heatmap(df: pd.DataFrame, split: str, out_dir: Path):
    """Heatmap: rows = combos, cols = variants, values = BAcc."""
    pivot = df.pivot_table(index="combo", columns="vtag", values="bacc", aggfunc="mean")
    if pivot.empty: return
    pivot.index = [c.replace("Clinical","Clin") for c in pivot.index]

    n_rows, n_cols = len(pivot), len(pivot.columns)
    fig, ax = plt.subplots(figsize=(max(8, n_cols*1.5+1), max(4, n_rows*0.55+1)))
    if HAS_SNS:
        sns.heatmap(pivot.astype(float), annot=True, fmt=".3f",
                    cmap="RdYlGn", vmin=0.4, vmax=1.0, linewidths=0.4,
                    annot_kws={"size": 8}, ax=ax)
    else:
        im = ax.imshow(pivot.values.astype(float), cmap="RdYlGn",
                       vmin=0.4, vmax=1.0, aspect="auto")
        for i in range(n_rows):
            for j in range(n_cols):
                v = pivot.values[i,j]
                ax.text(j, i, f"{v:.3f}" if not np.isnan(v) else "",
                        ha="center", va="center", fontsize=7)
        ax.set_xticks(range(n_cols)); ax.set_yticks(range(n_rows))
        ax.set_xticklabels(pivot.columns, rotation=25, ha="right", fontsize=8)
        ax.set_yticklabels(pivot.index, fontsize=8)
        plt.colorbar(im, ax=ax)

    ax.set_title(f"Balanced Accuracy — {split.upper()}\n"
                 f"rows = modality combo, cols = P1 baseline / P2 variant", fontsize=11)
    plt.xticks(rotation=25, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    p = out_dir / f"combo_heatmap_{split}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Heatmap: {p}")


def _plot_combo_bar(df: pd.DataFrame, split: str,
                    metric: str, out_dir: Path):
    """Grouped bar chart: combos on X, one bar-group per variant.
    P1 baselines use blue-family colors; P2 variants use orange-family colors."""
    combos   = sorted(df["combo"].unique())
    variants = sorted(df["vtag"].unique())
    x        = np.arange(len(combos))
    width    = 0.8 / max(len(variants), 1)
    offsets  = np.linspace(-(len(variants)-1)/2, (len(variants)-1)/2,
                           len(variants)) * width

    fig, ax = plt.subplots(figsize=(max(10, len(combos)*1.4+2), 6))

    # Separate P1 and P2 variants for coloring
    p1_variants = [v for v in variants if v.startswith("P1_")]
    p2_variants = [v for v in variants if not v.startswith("P1_")]
    p1_colors = plt.cm.Blues(np.linspace(0.45, 0.85, max(len(p1_variants), 1)))
    p2_colors = plt.cm.Oranges(np.linspace(0.35, 0.85, max(len(p2_variants), 1)))
    p1_color_map = {v: c for v, c in zip(p1_variants, p1_colors)}
    p2_color_map = {v: c for v, c in zip(p2_variants, p2_colors)}

    def _get_color(vtag):
        if vtag.startswith("P1_"):
            return p1_color_map.get(vtag, "#4e79a7")
        return p2_color_map.get(vtag, "#f28e2b")

    for vi, (vtag, off) in enumerate(zip(variants, offsets)):
        color = _get_color(vtag)
        vsub = df[df["vtag"] == vtag]
        vals, ns = [], []
        for combo in combos:
            csub = vsub[vsub["combo"] == combo]
            vals.append(float(csub[metric].iloc[0]) if not csub.empty else float("nan"))
            ns.append(int(csub["n"].iloc[0]) if not csub.empty else 0)
        vals = np.array(vals)
        label = vtag
        for i, (v, n) in enumerate(zip(vals, ns)):
            if np.isnan(v): continue
            bar = ax.bar(x[i] + off, v, width=width*0.9,
                         color=color, alpha=0.82, label=label if i==0 else "")
            ax.text(x[i]+off, v+0.005, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("Clinical","Clin") for c in combos],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0.3, 1.10)
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_title(f"{metric.upper()} by Modality Combination — {split.upper()}\n"
                 f"Blue = P1 single-modal baselines | Orange = P2 multimodal variants",
                 fontsize=11)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(fontsize=7, ncol=max(1, len(variants)//4),
              loc="upper right")

    # Sample count annotations
    n_per_combo = df.groupby("combo")["n"].max()
    for i, combo in enumerate(combos):
        n = n_per_combo.get(combo, 0)
        ax.text(i, 0.31, f"n={n}", ha="center", va="bottom",
                fontsize=7, color="dimgrey",
                transform=ax.get_xaxis_transform())

    fig.tight_layout()
    p = out_dir / f"combo_bar_{metric}_{split}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Bar chart ({metric}): {p}")


def _plot_combo_count_table(df: pd.DataFrame, split: str, out_dir: Path):
    """Table figure: rows = combos, cols = n / n_pos / n_neg per variant."""
    combos   = sorted(df["combo"].unique())
    variants = sorted(df["vtag"].unique())
    col_labels = ["Combo"]
    for vtag in variants:
        lbl = vtag.replace("P2_","")
        col_labels += [f"{lbl}\nn", f"{lbl}\nn_pos", f"{lbl}\nn_neg"]

    rows_data  = []
    cell_colors = []
    cmap = plt.cm.RdYlGn
    for combo in combos:
        row  = [combo.replace("Clinical","Clin")]
        rc   = ["#ffffff"]
        for vtag in variants:
            csub = df[(df["vtag"]==vtag) & (df["combo"]==combo)]
            if csub.empty:
                row += ["—","—","—"]; rc += ["#f0f0f0","#f0f0f0","#f0f0f0"]
            else:
                r = csub.iloc[0]
                bacc = r.get("bacc", float("nan"))
                color = cmap(np.clip((bacc-0.4)/0.6,0,1))
                hex_c = "#{:02x}{:02x}{:02x}".format(
                    int(color[0]*255), int(color[1]*255), int(color[2]*255))
                row += [str(int(r["n"])), str(int(r["n_pos"])), str(int(r["n_neg"]))]
                rc  += [hex_c, "#d5f5e3", "#fde8d8"]
        rows_data.append(row)
        cell_colors.append(rc)

    fig_h = max(3, len(combos)*0.5 + 2)
    fig_w = max(8, len(col_labels)*1.1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows_data, cellColours=cell_colors,
        colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    ax.set_title(f"Sample counts per combo — {split.upper()}\n"
                 f"(row color = BAcc: green=high, red=low)", fontsize=10, pad=12)
    fig.tight_layout()
    p = out_dir / f"combo_count_table_{split}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Count table: {p}")


# ══════════════════════════════════════════════════════════════════
# TASK 5: CLUSTER PRESENCE HEATMAP
# ══════════════════════════════════════════════════════════════════

def task_cluster_presence_maps(
    samples_dir: str,
    splits_csv: str,
    folds: List[int],
    out_dir: Path,
    mods: Optional[List[str]] = None,
    outer_splits: Optional[List[int]] = None,
):
    """
    Heatmap: rows = patients sorted by (label, split), cols = cluster names.
    Cell = 1 (cluster present in this patient's bag) or 0.
    One figure per modality per fold, showing all splits in a single image.
    Helps identify which clusters are associated with rejection.
    """
    print("\n" + "="*65)
    print("  TASK 5: Cluster Presence Maps")
    print("="*65)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mods is None:
        mods = AGG_MODS
    outer_split = outer_splits[0] if outer_splits else None

    for fold in folds:
        print(f"\n  Fold {fold}")
        all_recs = build_records(splits_csv, fold, outer_split=outer_split)
        stems    = list({r["stem"] for r in all_recs})
        print(f"  Loading aggregates for {len(stems)} stems …")
        agg_cache, _ = preload_aggregates(stems, samples_dir, mods=mods, quiet=True)

        for mod in mods:
            # Collect (stem, cluster_mask, cluster_names, label, split)
            rows_data = []
            for rec in all_recs:
                agg = agg_cache.get(rec["stem"], {}).get(mod)
                if agg is None:
                    continue
                mask = agg.get("cluster_mask")
                names = agg.get("cluster_names")
                if mask is None or names is None:
                    continue
                rows_data.append({
                    "stem":  rec["stem"],
                    "label": rec["label"],
                    "split": rec["split"],
                    "mask":  mask.numpy().astype(np.int8),
                    "names": names,
                })

            if not rows_data:
                print(f"  [skip] {mod} — no data for fold {fold}")
                continue

            # Use cluster names from first available sample
            cluster_names = rows_data[0]["names"]
            K = len(cluster_names)

            # Sort patients: split order, then by label
            split_order = {"train": 0, "val": 1, "test": 2}
            rows_data.sort(key=lambda r: (split_order.get(r["split"], 3), r["label"]))

            matrix    = np.stack([r["mask"][:K] for r in rows_data])  # (N, K)
            labels_v  = np.array([r["label"] for r in rows_data])
            splits_v  = np.array([r["split"] for r in rows_data])

            fig, ax = plt.subplots(figsize=(max(12, K * 0.28 + 2), max(5, len(rows_data) * 0.12 + 1.5)))
            ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                      interpolation="nearest")

            # Side-bar: label color stripe
            n_patients = len(rows_data)
            bar_x = K + 0.5
            for pi in range(n_patients):
                ax.add_patch(plt.Rectangle(
                    (bar_x, pi - 0.5), 1.2, 1,
                    color=LABEL_COLORS[labels_v[pi]], clip_on=False))

            # Split separator lines
            prev_sp = None
            for pi, sp in enumerate(splits_v):
                if sp != prev_sp and prev_sp is not None:
                    ax.axhline(pi - 0.5, color="orange", lw=1.5, ls="--")
                    ax.text(K + 2, pi, sp, fontsize=7, va="center", color="orange")
                prev_sp = sp

            # Cluster-level presence rate annotation at top
            presence_rate = matrix.mean(axis=0)
            for ci, rate in enumerate(presence_rate):
                if rate > 0.5:
                    ax.text(ci, -0.8, f"{rate:.0%}", ha="center", va="bottom",
                            fontsize=5, rotation=90, color="#333333")

            ax.set_xticks(range(K))
            ax.set_xticklabels(cluster_names, rotation=70, ha="right", fontsize=6)
            ax.set_yticks([])
            ax.set_xlabel("Cluster", fontsize=9)
            ax.set_ylabel(f"Patients (n={n_patients})", fontsize=9)
            n_pos = int(labels_v.sum())
            ax.set_title(
                f"{mod} Cluster Presence — Fold {fold}\n"
                f"n={n_patients}  neg={n_patients - n_pos}  pos={n_pos}  "
                f"K={K} clusters  (blue=present, side-bar: red=reject, blue=ctrl)",
                fontsize=10)
            ax.legend(handles=[plt.Rectangle((0,0),1,1, color="#5c9be0", label="Neg A0"),
                                plt.Rectangle((0,0),1,1, color="#e05c5c", label="Pos A1/A2")],
                      fontsize=8, loc="lower right")
            fig.tight_layout()
            p = out_dir / f"cluster_presence_{mod}_fold{fold}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {p}")

        del agg_cache; _gc()

    print(f"  Cluster presence outputs → {out_dir}")


# ══════════════════════════════════════════════════════════════════
# TASK 6: SPATIAL INSTANCE SCATTER
# ══════════════════════════════════════════════════════════════════

def task_spatial_scatter(
    samples_dir: str,
    splits_csv: str,
    folds: List[int],
    out_dir: Path,
    mods: Optional[List[str]] = None,
    n_per_label: int = 5,
    outer_splits: Optional[List[int]] = None,
):
    """
    For each modality with spatial coordinates, scatter instances at their
    spatial positions colored by cluster_id.  Picks n_per_label patients per
    label per fold.  Useful for BAL (tissue coordinates) and HE (slide xy).
    """
    print("\n" + "="*65)
    print("  TASK 6: Spatial Instance Scatter")
    print("="*65)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mods is None:
        mods = AGG_MODS
    outer_split = outer_splits[0] if outer_splits else None

    for fold in folds:
        print(f"\n  Fold {fold}")
        all_recs = build_records(splits_csv, fold, outer_split=outer_split)
        stems    = list({r["stem"] for r in all_recs})
        agg_cache, _ = preload_aggregates(stems, samples_dir, mods=mods, quiet=True)

        for mod in mods:
            per_dir = out_dir / f"{mod}_fold{fold}"
            per_dir.mkdir(exist_ok=True)

            # Collect records that have spatial coords
            valid_recs = []
            for rec in all_recs:
                agg = agg_cache.get(rec["stem"], {}).get(mod)
                if agg is None:
                    continue
                sc = agg.get("instance_spatial_coords")
                if sc is None or sc.ndim < 2 or sc.shape[1] < 2:
                    continue
                valid_recs.append(rec)

            if not valid_recs:
                print(f"  [skip] {mod} fold {fold} — no spatial coords")
                continue

            pos_recs = [r for r in valid_recs if r["label"] == 1]
            neg_recs = [r for r in valid_recs if r["label"] == 0]
            chosen   = (pos_recs[:n_per_label] + neg_recs[:n_per_label])

            for rec in chosen:
                agg    = agg_cache[rec["stem"]][mod]
                sc_t   = agg["instance_spatial_coords"]
                if sc_t is None or not isinstance(sc_t, torch.Tensor):
                    continue
                sc     = sc_t.numpy()  # (N, ≥2)
                raw_ids = agg["instance_cluster_ids"]
                names  = agg.get("cluster_names") or []
                mask   = agg.get("cluster_mask")

                # instance_cluster_ids is a list of string labels
                # convert to integer indices using cluster_names as lookup
                if isinstance(raw_ids, (list, tuple)):
                    name2idx = {n: i for i, n in enumerate(names)}
                    cids = np.array([name2idx.get(str(s), 0) for s in raw_ids], dtype=np.int32)
                    if not names:
                        unique_str = sorted(set(str(s) for s in raw_ids))
                        name2idx = {n: i for i, n in enumerate(unique_str)}
                        cids = np.array([name2idx[str(s)] for s in raw_ids], dtype=np.int32)
                        names = unique_str
                elif isinstance(raw_ids, torch.Tensor):
                    cids = raw_ids.numpy().astype(np.int32)
                else:
                    continue

                unique_clusters = np.unique(cids)
                cmap_c = plt.cm.tab20(np.linspace(0, 1, max(len(unique_clusters), 20)))
                color_map = {int(k): cmap_c[i % 20] for i, k in enumerate(unique_clusters)}
                colors = np.array([color_map[int(c)] for c in cids])

                n_present = int(mask.sum()) if mask is not None else len(unique_clusters)
                lbl_str   = "Pos_A1A2" if rec["label"] == 1 else "Neg_A0"

                fig, axes = plt.subplots(1, 2, figsize=(16, 6))

                # Panel 1: colored by cluster
                axes[0].scatter(sc[:, 0], sc[:, 1], c=colors, s=1.5,
                                alpha=0.4, linewidths=0, rasterized=True)
                handles = [plt.Line2D([0],[0], marker="o", color="w",
                                       markerfacecolor=cmap_c[i % 20], markersize=5,
                                       label=names[int(k)] if int(k) < len(names) else str(k))
                           for i, k in enumerate(unique_clusters[:20])]
                axes[0].legend(handles=handles, fontsize=5, ncol=2, loc="upper right",
                               markerscale=1.5, framealpha=0.7)
                axes[0].set_title(f"Instances by cluster\n{mod} | {rec.get('patient_id', rec['stem'])} | {lbl_str} "
                                   f"| fold {fold}")
                axes[0].set_xlabel("Spatial X"); axes[0].set_ylabel("Spatial Y")

                # Panel 2: instance density (hex or scatter with alpha)
                axes[1].scatter(sc[:, 0], sc[:, 1], c="#888888",
                                s=1.0, alpha=0.2, linewidths=0, rasterized=True)
                # Overlay top-1% densest (KDE proxy via hexbin count)
                hb = axes[1].hexbin(sc[:, 0], sc[:, 1], gridsize=40, cmap="Reds",
                                     mincnt=1, alpha=0.7)
                plt.colorbar(hb, ax=axes[1], label="instance density")
                axes[1].set_title(f"Instance density\nK={len(unique_clusters)} clusters "
                                   f"({n_present} present)  N={len(cids):,} instances")
                axes[1].set_xlabel("Spatial X"); axes[1].set_ylabel("Spatial Y")

                for ax in axes:
                    ax.grid(True, lw=0.3, alpha=0.4)
                fig.tight_layout()
                pid_safe = re.sub(r"[^A-Za-z0-9_\-]", "_", str(rec.get("patient_id", rec["stem"])))
                p = per_dir / f"{lbl_str}_{pid_safe}_{rec['split']}.png"
                fig.savefig(p, dpi=130, bbox_inches="tight")
                plt.close(fig)

            print(f"  {mod} fold {fold}: {len(chosen)} spatial plots → {per_dir}")

        del agg_cache; _gc()

    print(f"  Spatial scatter outputs → {out_dir}")


# ══════════════════════════════════════════════════════════════════
# TASK 7: CLUSTER COUNT DISTRIBUTION
# ══════════════════════════════════════════════════════════════════

def task_cluster_count_viz(
    samples_dir: str,
    splits_csv: str,
    folds: List[int],
    out_dir: Path,
    mods: Optional[List[str]] = None,
    outer_splits: Optional[List[int]] = None,
):
    """
    For each modality, visualise:
      (a) Heatmap: cluster × bin, cell = fraction of patients in that bin,
          split into pos vs neg rows.
      (b) Bar chart: per cluster, show bin fractions for neg vs pos patients.
      (c) Cluster absence rate: which clusters are frequently absent per label.
    """
    print("\n" + "="*65)
    print("  TASK 7: Cluster Count Distribution")
    print("="*65)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mods is None:
        mods = AGG_MODS
    outer_split = outer_splits[0] if outer_splits else None

    for fold in folds:
        print(f"\n  Fold {fold}")
        all_recs = build_records(splits_csv, fold, outer_split=outer_split)
        stems    = list({r["stem"] for r in all_recs})
        agg_cache, _ = preload_aggregates(stems, samples_dir, mods=mods, quiet=True)

        for mod in mods:
            # Gather cluster_count_onehot and cluster_mask per patient
            data_pos, data_neg = [], []
            mask_pos, mask_neg = [], []
            cluster_names = None

            for rec in all_recs:
                agg = agg_cache.get(rec["stem"], {}).get(mod)
                if agg is None:
                    continue
                coh   = agg.get("cluster_count_onehot")  # (K, K*n_bins)
                cmask = agg.get("cluster_mask")
                cnames = agg.get("cluster_names")
                if coh is None or cmask is None:
                    continue
                if cluster_names is None and cnames:
                    cluster_names = cnames

                K      = cmask.shape[0]
                n_bins = coh.shape[1] // K if K > 0 else 4

                # Extract per-cluster bin: argmax within each block, -1 if absent
                coh_np = coh.numpy()
                bins   = np.full(K, -1, dtype=np.int32)
                for k in range(K):
                    block = coh_np[k, k * n_bins: (k + 1) * n_bins]
                    if block.sum() > 0:
                        bins[k] = int(np.argmax(block))

                if rec["label"] == 1:
                    data_pos.append(bins)
                    mask_pos.append(cmask.numpy())
                else:
                    data_neg.append(bins)
                    mask_neg.append(cmask.numpy())

            if not data_pos and not data_neg:
                print(f"  [skip] {mod} fold {fold} — no data")
                continue

            K      = len(data_pos[0]) if data_pos else len(data_neg[0])
            n_bins_val = 4
            if cluster_names is None:
                cluster_names = [str(k) for k in range(K)]

            fold_out = out_dir / f"{mod}_fold{fold}"
            fold_out.mkdir(exist_ok=True)

            # ── (a) Bin-fraction heatmap ─────────────────────────
            def bin_fractions(data_list):
                if not data_list:
                    return np.zeros((K, n_bins_val + 1))
                mat = np.stack(data_list)   # (N, K)
                result = np.zeros((K, n_bins_val + 1))
                for k in range(K):
                    col = mat[:, k]
                    n   = len(col)
                    result[k, -1] = (col == -1).sum() / n  # absent fraction
                    for b in range(n_bins_val):
                        result[k, b] = (col == b).sum() / n
                return result

            frac_pos = bin_fractions(data_pos)  # (K, n_bins+1)
            frac_neg = bin_fractions(data_neg)

            col_labels = [f"Q{b+1}" for b in range(n_bins_val)] + ["absent"]
            fig, axes = plt.subplots(1, 2, figsize=(max(8, n_bins_val * 1.5 + 3),
                                                     max(5, K * 0.22 + 2)),
                                     sharey=True)
            for ax, frac, title_suf, cnt in [
                (axes[0], frac_neg, f"Neg A0 (n={len(data_neg)})", len(data_neg)),
                (axes[1], frac_pos, f"Pos A1/A2 (n={len(data_pos)})", len(data_pos)),
            ]:
                if cnt == 0:
                    ax.set_visible(False); continue
                im = ax.imshow(frac, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                               interpolation="nearest")
                for ki in range(K):
                    for bi in range(n_bins_val + 1):
                        v = frac[ki, bi]
                        if v > 0.05:
                            ax.text(bi, ki, f"{v:.0%}", ha="center", va="center",
                                    fontsize=5, color="black" if v < 0.7 else "white")
                ax.set_xticks(range(n_bins_val + 1))
                ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8)
                ax.set_yticks(range(K))
                ax.set_yticklabels(cluster_names, fontsize=5)
                ax.set_title(f"{mod} count bins — {title_suf}", fontsize=9)
                plt.colorbar(im, ax=ax, label="fraction of patients", shrink=0.7)

            fig.suptitle(f"{mod} Cluster Count Bin Distribution — Fold {fold}", fontsize=11)
            fig.tight_layout()
            p = fold_out / "count_bin_heatmap.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Bin heatmap: {p}")

            # ── (b) Absence rate comparison per cluster ─────────
            def absence_rate(data_list):
                if not data_list:
                    return np.ones(K)
                mat = np.stack(data_list)
                return (mat == -1).mean(axis=0)

            abs_neg = absence_rate(data_neg)
            abs_pos = absence_rate(data_pos)

            # Sort clusters by |pos-neg| absence rate difference
            diff_abs = np.abs(abs_pos - abs_neg)
            sort_idx = np.argsort(-diff_abs)

            fig, ax = plt.subplots(figsize=(max(10, K * 0.35 + 2), 5))
            x = np.arange(K)
            w = 0.38
            ax.bar(x - w/2, abs_neg[sort_idx], width=w, color="#5c9be0", alpha=0.8, label="Neg A0")
            ax.bar(x + w/2, abs_pos[sort_idx], width=w, color="#e05c5c", alpha=0.8, label="Pos A1/A2")
            ax.set_xticks(x)
            sorted_names = [cluster_names[i] if i < len(cluster_names) else str(i)
                            for i in sort_idx]
            ax.set_xticklabels(sorted_names, rotation=70, ha="right", fontsize=6)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Absence rate (cluster_mask=False)", fontsize=9)
            ax.set_title(f"{mod} Cluster Absence Rate — Fold {fold}\n"
                         f"Sorted by |pos−neg| difference", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.35)
            fig.tight_layout()
            p = fold_out / "cluster_absence_rate.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Absence rate: {p}")

        del agg_cache; _gc()

    print(f"  Cluster count viz outputs → {out_dir}")


# ══════════════════════════════════════════════════════════════════
# TASK 8: CLINICAL FEATURE TOKENIZATION OVERVIEW
# ══════════════════════════════════════════════════════════════════

def task_clinical_token_viz(
    samples_dir: str,
    splits_csv: str,
    folds: List[int],
    out_dir: Path,
    outer_splits: Optional[List[int]] = None,
):
    """
    Visualise clinical feature tokenization from clinical_onehot (F, F*n_bins):
      (a) Bin occupancy heatmap: feature × bin — fraction of patients per bin, pos vs neg.
      (b) Feature importance proxy: L2 difference between pos and neg bin distributions.
    """
    print("\n" + "="*65)
    print("  TASK 8: Clinical Feature Tokenization")
    print("="*65)
    out_dir.mkdir(parents=True, exist_ok=True)
    outer_split = outer_splits[0] if outer_splits else None

    for fold in folds:
        print(f"\n  Fold {fold}")
        all_recs = build_records(splits_csv, fold, outer_split=outer_split)
        stems    = list({r["stem"] for r in all_recs})
        _, clin_cache = preload_aggregates(stems, samples_dir, mods=[], quiet=True)

        # Collect clinical onehots per label
        pos_data, neg_data = [], []
        feat_names = None

        for rec in all_recs:
            ce = clin_cache.get(rec["stem"], {})
            coh = ce.get("clinical_onehot")
            cfn = ce.get("clinical_feature_names")
            if coh is None:
                continue
            coh_np = coh.numpy()  # (F, F*n_bins)
            F = coh_np.shape[0]
            if feat_names is None and cfn:
                feat_names = cfn
            if rec["label"] == 1:
                pos_data.append(coh_np)
            else:
                neg_data.append(coh_np)

        if not pos_data and not neg_data:
            print(f"  [skip] fold {fold} — no clinical data")
            continue

        F      = pos_data[0].shape[0] if pos_data else neg_data[0].shape[0]
        n_bins = (pos_data[0].shape[1] // F) if pos_data else 4
        if feat_names is None:
            feat_names = [f"F{f}" for f in range(F)]

        def mean_bin_fracs(data_list):
            if not data_list:
                return np.zeros((F, n_bins))
            stacked = np.stack(data_list)  # (N, F, F*n_bins)
            result  = np.zeros((F, n_bins))
            for f in range(F):
                block = stacked[:, f, f * n_bins: (f + 1) * n_bins]
                absent = (block.sum(axis=1) == 0).mean()
                row    = block.mean(axis=0)
                result[f] = row
            return result

        frac_pos = mean_bin_fracs(pos_data)   # (F, n_bins)
        frac_neg = mean_bin_fracs(neg_data)

        diff = frac_pos - frac_neg             # (F, n_bins)
        importance = np.abs(diff).sum(axis=1)  # (F,) — total variation per feature

        # Sort features by importance
        sort_idx   = np.argsort(-importance)
        top_k      = min(40, F)
        top_idx    = sort_idx[:top_k]
        top_names  = [feat_names[i] if i < len(feat_names) else f"F{i}" for i in top_idx]

        # ── (a) Bin occupancy heatmap, top features ──────────
        fig, axes = plt.subplots(1, 2, figsize=(n_bins * 1.5 + 3, max(5, top_k * 0.25 + 1.5)),
                                  sharey=True)
        col_labels = [f"Q{b+1}" for b in range(n_bins)]
        for ax, frac, title_suf, cnt in [
            (axes[0], frac_neg[top_idx], f"Neg A0 (n={len(neg_data)})", len(neg_data)),
            (axes[1], frac_pos[top_idx], f"Pos A1/A2 (n={len(pos_data)})", len(pos_data)),
        ]:
            if cnt == 0:
                ax.set_visible(False); continue
            im = ax.imshow(frac, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                           interpolation="nearest")
            for fi in range(top_k):
                for bi in range(n_bins):
                    v = frac[fi, bi]
                    if v > 0.08:
                        ax.text(bi, fi, f"{v:.0%}", ha="center", va="center",
                                fontsize=5, color="black" if v < 0.7 else "white")
            ax.set_xticks(range(n_bins))
            ax.set_xticklabels(col_labels, fontsize=9)
            ax.set_yticks(range(top_k))
            ax.set_yticklabels(top_names, fontsize=6)
            ax.set_title(f"Clinical bins — {title_suf}", fontsize=9)
            plt.colorbar(im, ax=ax, label="mean bin fraction", shrink=0.7)

        fig.suptitle(f"Clinical Feature Bin Occupancy — Fold {fold}\n"
                     f"Top {top_k} most discriminative features (sorted by |pos-neg|)",
                     fontsize=10)
        fig.tight_layout()
        p = out_dir / f"clinical_bin_heatmap_fold{fold}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Clinical bin heatmap: {p}")

        # ── (b) Feature importance bar chart ─────────────────
        fig, ax = plt.subplots(figsize=(max(10, top_k * 0.4 + 2), 5))
        ax.bar(range(top_k), importance[top_idx], color="#7b68ee", alpha=0.85)
        ax.set_xticks(range(top_k))
        ax.set_xticklabels(top_names, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("Sum |pos_frac − neg_frac| over bins", fontsize=9)
        ax.set_title(f"Clinical Feature Discriminability — Fold {fold}\n"
                     f"(L1 distance between pos/neg bin distributions)", fontsize=10)
        ax.grid(axis="y", alpha=0.35)
        fig.tight_layout()
        p = out_dir / f"clinical_importance_fold{fold}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Clinical importance: {p}")

        del pos_data, neg_data, clin_cache; _gc()

    print(f"  Clinical token viz outputs → {out_dir}")


# ══════════════════════════════════════════════════════════════════
# TASK: UMAP + HEXBIN VISUALISATION
# ══════════════════════════════════════════════════════════════════

def _fit_umap(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1,
              random_state: int = 42) -> np.ndarray:
    """PCA → UMAP (falls back to PCA-2D if umap-learn absent)."""
    n = X.shape[0]
    # PCA pre-reduction
    n_pca = min(50, n - 1, X.shape[1])
    if n_pca >= 2 and X.shape[1] > n_pca:
        X_red = PCA(n_components=n_pca).fit_transform(X)
    else:
        X_red = X
    if HAS_UMAP and n > 10:
        try:
            return UMAPTransform(
                n_components=2, random_state=random_state,
                n_neighbors=min(n_neighbors, n - 1),
                min_dist=min_dist,
            ).fit_transform(X_red)
        except Exception as exc:
            print(f"  [umap warn] {exc} — falling back to PCA")
    return PCA(n_components=2).fit_transform(X_red)


def _scatter_by_combo(ax, Z, combos, splits,
                      highlight_split: str = "test",
                      title: str = ""):
    """Scatter UMAP coloured by modality combo.
    Points in highlight_split are drawn at full opacity; others at 20%."""
    unique_combos = sorted(set(combos))
    kw_dim  = dict(s=10, alpha=0.18, linewidths=0, rasterized=True)
    kw_hi   = dict(s=16, alpha=0.72, linewidths=0, rasterized=True)
    for combo in unique_combos:
        color = COMBO_COLORS.get(combo, "#aaaaaa")
        mask_hi  = (combos == combo) & (splits == highlight_split)
        mask_dim = (combos == combo) & (splits != highlight_split)
        if mask_dim.any():
            ax.scatter(Z[mask_dim, 0], Z[mask_dim, 1], color=color, **kw_dim)
        if mask_hi.any():
            ax.scatter(Z[mask_hi, 0], Z[mask_hi, 1], color=color,
                       label=combo, **kw_hi)
    ax.legend(fontsize=6.5, ncol=2, loc="best", markerscale=2.2,
              framealpha=0.8, handlelength=1.2, borderpad=0.5)
    ax.set_title(title or "Modality combo", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_facecolor("#f5f5f5")
    ax.grid(True, lw=0.3, alpha=0.4)


def _scatter_by_label(ax, Z, labels, splits,
                      highlight_split: str = "test",
                      title: str = ""):
    """Scatter UMAP coloured by ACR label.
    Other splits shown in light grey; highlight_split in bold."""
    # dim: other splits
    dim_mask = splits != highlight_split
    if dim_mask.any():
        ax.scatter(Z[dim_mask, 0], Z[dim_mask, 1],
                   color="#cccccc", s=8, alpha=0.25, linewidths=0,
                   rasterized=True, label="train/val")
    # highlight: test
    for lv, color, lname in [(0, "#3498db", "ACR neg (A0)"),
                              (1, "#e74c3c", "ACR pos (A1/A2)")]:
        m = (splits == highlight_split) & (labels == lv)
        if m.any():
            ax.scatter(Z[m, 0], Z[m, 1], color=color, s=22,
                       alpha=0.80, linewidths=0.4, edgecolors="white",
                       rasterized=True, label=f"{lname} (n={m.sum()})")
    ax.legend(fontsize=8, loc="best", markerscale=1.8, framealpha=0.85)
    ax.set_title(title or "ACR label", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_facecolor("#f5f5f5")
    ax.grid(True, lw=0.3, alpha=0.4)


def _hexbin_label_ratio(ax, Z, labels, mask=None, gridsize=40, title=""):
    """Hexbin: fraction positive per bin (0=neg, 1=pos)."""
    if mask is None:
        mask = np.ones(len(Z), dtype=bool)
    Zs = Z[mask]; ls = labels[mask].astype(float)
    if len(Zs) < 5:
        ax.set_visible(False); return
    hb = ax.hexbin(Zs[:, 0], Zs[:, 1], C=ls,
                   reduce_C_function=np.mean,
                   gridsize=gridsize, cmap="RdYlGn_r",
                   vmin=0, vmax=1, mincnt=2, linewidths=0.2)
    cb = plt.colorbar(hb, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("Fraction ACR+", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    n_pos = int(ls.sum()); n_neg = len(ls) - n_pos
    ax.set_title(title or f"Rejection hotspots\nn={len(Zs)}  pos={n_pos}  neg={n_neg}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_facecolor("#f0f0f0")


def _hexbin_density_pos_neg(ax_neg, ax_pos, Z, labels, mask=None, gridsize=40):
    """Two hexbin density panels: negative patients (blue) and positive (red)."""
    if mask is None:
        mask = np.ones(len(Z), dtype=bool)
    Zs = Z[mask]; ls = labels[mask]
    xlim = (Zs[:, 0].min() - 0.5, Zs[:, 0].max() + 0.5)
    ylim = (Zs[:, 1].min() - 0.5, Zs[:, 1].max() + 0.5)
    for ax, lv, cmap, lname in [(ax_neg, 0, "Blues", "ACR neg (A0)"),
                                 (ax_pos, 1, "Reds",  "ACR pos (A1/A2)")]:
        m = ls == lv
        if not m.any():
            ax.set_visible(False); continue
        ax.hexbin(Zs[m, 0], Zs[m, 1], gridsize=gridsize,
                  cmap=cmap, mincnt=1, linewidths=0.2)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(f"{lname}\n(n={m.sum()})", fontsize=10, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_facecolor("#f0f0f0")


def _hexbin_modality_presence(axes_grid, Z, combos_or_recs, mods, mask=None,
                               gridsize=40, title_prefix=""):
    """One hexbin density panel per modality: present (solid) vs absent (grey)."""
    if mask is None:
        mask = np.ones(len(Z), dtype=bool)
    Zs = Z[mask]
    if hasattr(combos_or_recs[0], 'get'):
        pres = {m: np.array([r.get(_pres_col(m), False) for r in combos_or_recs])[mask]
                for m in mods}
    else:
        # combos_or_recs is an array of combo strings
        pres = {m: np.array([m in c for c in combos_or_recs])[mask] for m in mods}

    xlim = (Zs[:, 0].min() - 0.5, Zs[:, 0].max() + 0.5)
    ylim = (Zs[:, 1].min() - 0.5, Zs[:, 1].max() + 0.5)
    mod_colors = {"HE": "#e67e22", "BAL": "#2980b9", "CT": "#27ae60",
                  "Clinical": "#8e44ad"}
    for ax, mod in zip(axes_grid, mods):
        p = pres.get(mod, np.zeros(len(Zs), dtype=bool))
        # absent: grey background density
        if (~p).any():
            ax.hexbin(Zs[~p, 0], Zs[~p, 1], gridsize=gridsize,
                      cmap="Greys", mincnt=1, linewidths=0.1, alpha=0.5)
        # present: coloured foreground density
        if p.any():
            ax.hexbin(Zs[p, 0], Zs[p, 1], gridsize=gridsize,
                      color=mod_colors.get(mod, "#555"),
                      mincnt=1, linewidths=0.2, alpha=0.85)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(f"{title_prefix}{mod}  (n={p.sum()} present)",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_facecolor("#f5f5f5")
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color=mod_colors.get(mod, "#555"), label=f"{mod} present"),
            Patch(color="#aaaaaa", alpha=0.5, label=f"{mod} absent"),
        ], fontsize=8, loc="best")


def task_umap_acr(
    results_dir: Path,
    samples_dir: str,
    splits_csv: str,
    folds: List[int],
    out_dir: Path,
    outer_splits: Optional[List[int]] = None,
    best_variant: Optional[str] = None,
):
    """
    For ACR: fit UMAP on P2 pre-head embeddings (train+val+test), then produce
    per fold:
      1. Scatter figure (2 panels): by modality combo | by ACR label
      2. Hexbin figure (2×2 panels): label hotspots | pos density | neg density |
         modality presence (4 mini hexbins, one per modality)
    """
    if not HAS_UMAP:
        print("  [warn] umap-learn not installed — UMAP task requires it"); return
    print("\n" + "="*65)
    print("  TASK: UMAP + Hexbin Visualisation (ACR)")
    print("="*65)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outer_split = outer_splits[0] if outer_splits else None

    # Pick best variant
    if best_variant is None:
        p2_cfgs_all = scan_p2_configs(results_dir, folds, outer_splits)
        vtag_baccs: Dict[str, List[float]] = {}
        for cfg in p2_cfgs_all:
            m = load_stored_metrics(cfg["metrics_path"])
            v = m.get("test", {}).get("bacc", np.nan)
            if not np.isnan(v):
                vtag_baccs.setdefault(cfg["vtag"], []).append(v)
        if not vtag_baccs:
            print("  [warn] No P2 metrics found — skipping"); return
        best_variant = max(vtag_baccs, key=lambda k: np.mean(vtag_baccs[k]))
        print(f"  Best variant: {best_variant}  "
              f"(mean test BAcc = {np.mean(vtag_baccs[best_variant]):.4f})")

    split_tag = f"split{outer_split}" if outer_split is not None else "combined"

    all_fold_Zs:      List[np.ndarray] = []
    all_fold_labels:  List[np.ndarray] = []
    all_fold_combos:  List[np.ndarray] = []
    all_fold_splits:  List[np.ndarray] = []
    all_fold_recs:    List[List[dict]] = []
    all_fold_idx:     List[int] = []   # which fold each point belongs to

    for fold in folds:
        print(f"\n  Fold {fold}")
        if outer_split is not None:
            p2_dir = results_dir / f"split{outer_split}_fold{fold}" / "phase2"
        else:
            p2_dir = results_dir / f"fold_{fold}" / "phase2"
        ckpt = p2_dir / f"model_{best_variant}.pt"
        if not ckpt.exists():
            print(f"  [skip] checkpoint missing: {ckpt}"); continue

        all_recs = build_records(splits_csv, fold, outer_split=outer_split)
        stems    = list({r["stem"] for r in all_recs})
        print(f"  Loading {len(stems)} bags …")
        bags = preload_bags(stems, samples_dir, quiet=True)
        all_recs = update_presence(all_recs, bags)

        print(f"  Extracting P2 embeddings via hook ({best_variant}) …")
        reps, labels, combos, splits_arr = _try_p2_reps_hook(
            ckpt, all_recs, bags, device)
        del bags; _gc()

        if reps is None or len(reps) < 10:
            print(f"  [skip] too few embeddings (got {0 if reps is None else len(reps)})"); continue

        # Align recs to extracted reps (hook only ran on patients with bags)
        rec_by_stem = {r["stem"]: r for r in all_recs}
        # reps come out in the same order as hook iteration (all_recs order)
        # we use the combos/splits arrays returned from the hook to build recs_aligned
        recs_aligned = []
        stems_in_order = [r["stem"] for r in all_recs]
        ri = 0
        for stem in stems_in_order:
            if ri >= len(combos): break
            rec = rec_by_stem.get(stem)
            if rec is None: continue
            # The hook skips patients with no bag — so align by combo match
            # Use combos[ri] to verify against rec["combo"]
            recs_aligned.append(rec)
            ri += 1
        # Trim to actual reps length
        recs_aligned = recs_aligned[:len(reps)]

        print(f"  Fitting UMAP on {len(reps)} points …")
        Z = _fit_umap(reps)

        all_fold_Zs.append(Z)
        all_fold_labels.append(labels)
        all_fold_combos.append(combos)
        all_fold_splits.append(splits_arr)
        all_fold_recs.append(recs_aligned)
        all_fold_idx.extend([fold] * len(Z))

        # ── Per-fold scatter figure ────────────────────────────────────────────
        fig_s, axes_s = plt.subplots(1, 2, figsize=(16, 7))
        _scatter_by_combo(axes_s[0], Z, combos, splits_arr,
                          title=f"Modality combo — Fold {fold}")
        _scatter_by_label(axes_s[1], Z, labels, splits_arr,
                          title=f"ACR label — Fold {fold}")
        fig_s.suptitle(
            f"UMAP — ACR  ({split_tag},  {best_variant})  fold {fold}\n"
            f"n={len(reps)}  pos={int(labels.sum())}  neg={len(labels)-int(labels.sum())}",
            fontsize=12, fontweight="bold"
        )
        fig_s.tight_layout()
        p = out_dir / f"umap_scatter_fold{fold}_{split_tag}_{best_variant}.png"
        fig_s.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig_s)
        print(f"  Saved: {p.name}")

        # ── Per-fold hexbin figure ─────────────────────────────────────────────
        test_mask = splits_arr == "test"
        all_mask  = np.ones(len(Z), dtype=bool)

        fig_h = plt.figure(figsize=(18, 14))
        gs = fig_h.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        # [0,0] Label ratio (all patients)
        ax_lr = fig_h.add_subplot(gs[0, 0])
        _hexbin_label_ratio(ax_lr, Z, labels, mask=all_mask,
                            title=f"Rejection hotspots (all)\nfold {fold}")

        # [0,1] Label ratio (test only)
        ax_lrt = fig_h.add_subplot(gs[0, 1])
        _hexbin_label_ratio(ax_lrt, Z, labels, mask=test_mask,
                            title=f"Rejection hotspots (test)\nfold {fold}")

        # [0,2] Split scatter (train/val/test — small)
        ax_sp = fig_h.add_subplot(gs[0, 2])
        split_colors = {"train": "#2980b9", "val": "#e67e22", "test": "#27ae60"}
        for sp, sc in split_colors.items():
            m = splits_arr == sp
            if m.any():
                ax_sp.scatter(Z[m, 0], Z[m, 1], color=sc, s=8, alpha=0.45,
                              linewidths=0, rasterized=True, label=f"{sp} (n={m.sum()})")
        ax_sp.legend(fontsize=8, markerscale=2)
        ax_sp.set_title(f"Train / Val / Test split\nfold {fold}", fontsize=10, fontweight="bold")
        ax_sp.set_xlabel("UMAP 1", fontsize=8); ax_sp.set_ylabel("UMAP 2", fontsize=8)
        ax_sp.tick_params(labelsize=7); ax_sp.set_facecolor("#f5f5f5")

        # [1,0–2] Modality presence hexbins (4 modalities → 4 sub-axes)
        gs_bot = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=gs[1, :], wspace=0.25)
        axes_mod = [fig_h.add_subplot(gs_bot[0, i]) for i in range(4)]
        _hexbin_modality_presence(
            axes_mod, Z, recs_aligned, MODALITIES, mask=all_mask,
            title_prefix="")

        fig_h.suptitle(
            f"Hexbin — ACR  ({split_tag},  {best_variant})  fold {fold}",
            fontsize=13, fontweight="bold"
        )
        p = out_dir / f"umap_hexbin_fold{fold}_{split_tag}_{best_variant}.png"
        fig_h.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig_h)
        print(f"  Saved: {p.name}")

        _gc()

    # Cross-fold combined UMAP is intentionally omitted: each fold's UMAP is
    # fitted independently so the coordinate systems are incompatible.
    # Per-fold scatter/hexbin figures above are the valid outputs.
    print(f"\n  UMAP outputs → {out_dir}")


# ══════════════════════════════════════════════════════════════════
# TASK: MODALITY BALANCED-ACCURACY COMPARISON
# ══════════════════════════════════════════════════════════════════

def _align_probs_to_records(json_labels, json_probs, csv_recs):
    """Align JSON probs to CSV records via greedy label-sequence matching.

    The JSON may be shorter than csv_recs if some patients were skipped
    during inference (missing .pt files).  A greedy left-to-right scan
    skips CSV positions whose label does not match the current JSON position.

    Returns float32 array of length len(csv_recs); nan for positions not in JSON.
    Returns None if fewer than len(json_labels) positions were aligned.
    """
    csv_labels = [r["label"] for r in csv_recs]
    aligned = np.full(len(csv_labels), np.nan, dtype=np.float32)
    j = 0
    for i, cl in enumerate(csv_labels):
        if j >= len(json_labels):
            break
        if cl == json_labels[j]:
            aligned[i] = json_probs[j]
            j += 1
    n_aligned = int(np.sum(~np.isnan(aligned)))
    if n_aligned != len(json_labels):
        return None
    return aligned


def _safe_bacc(labels, probs, threshold=0.5):
    labels = np.asarray(labels)
    probs  = np.asarray(probs)
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    preds = (probs >= threshold).astype(int)
    return float(balanced_accuracy_score(labels, preds))


def task_modality_bacc(
    results_dir: Path,
    splits_csv: str,
    folds: List[int],
    out_dir: Path,
    outer_splits: Optional[List[int]] = None,
    best_variant: Optional[str] = None,
):
    """3-bar modality comparison for ACR classification (Balanced Accuracy).

    Per modality:
      Green  = P1 unimodal BAcc (test patients who have that modality)
      Blue   = P2 multimodal BAcc on ALL test patients
      Orange = P2 multimodal BAcc on SAME subset as P1 (fair comparison)

    P2-subset BAcc is computed by aligning stored test probs from the metrics
    JSON to CSV records, then filtering to has_{mod} patients.
    """
    print("\n" + "="*65)
    print("  TASK: Modality Balanced-Accuracy Comparison")
    print("="*65)
    out_dir.mkdir(parents=True, exist_ok=True)

    outer_split = outer_splits[0] if outer_splits else None

    # ── Pick best variant (highest mean test BAcc across folds) ──────────────
    if best_variant is None:
        p2_cfgs_all = scan_p2_configs(results_dir, folds, outer_splits)
        vtag_baccs: Dict[str, List[float]] = {}
        for cfg in p2_cfgs_all:
            m = load_stored_metrics(cfg["metrics_path"])
            v = m.get("test", {}).get("bacc", np.nan)
            if not np.isnan(v):
                vtag_baccs.setdefault(cfg["vtag"], []).append(v)
        if not vtag_baccs:
            print("  [warn] No P2 metrics found — skipping"); return
        best_variant = max(vtag_baccs, key=lambda k: np.mean(vtag_baccs[k]))
        print(f"  Best variant: {best_variant}  "
              f"(mean test BAcc = {np.mean(vtag_baccs[best_variant]):.4f})")

    # ── Collect per-fold data ─────────────────────────────────────────────────
    p1_fold_vals: Dict[str, List[float]] = {m: [] for m in MODALITIES}
    p1_fold_sens: Dict[str, List[float]] = {m: [] for m in MODALITIES}
    fold_full_bacc: List[float] = []
    fold_sub_bacc:  Dict[str, List[float]] = {m: [] for m in MODALITIES}
    fold_sub_n:     Dict[str, List[int]]   = {m: [] for m in MODALITIES}

    for fold in folds:
        # P1 per-modality BAcc from stored metrics
        if outer_split is not None:
            p1_dir = results_dir / f"split{outer_split}_fold{fold}" / "phase1"
            p2_dir = results_dir / f"split{outer_split}_fold{fold}" / "phase2"
        else:
            p1_dir = results_dir / f"fold_{fold}" / "phase1"
            p2_dir = results_dir / f"fold_{fold}" / "phase2"

        for mod in MODALITIES:
            mp = p1_dir / mod / "metrics.json"
            if mp.exists():
                m    = load_stored_metrics(mp)
                v    = m.get("test", {}).get("bacc", np.nan)
                sens = m.get("test", {}).get("sens", np.nan)
            else:
                v    = np.nan
                sens = np.nan
            p1_fold_vals[mod].append(v)
            p1_fold_sens[mod].append(sens)

        # P2 full-model BAcc
        p2_mp = p2_dir / f"metrics_{best_variant}.json"
        if p2_mp.exists():
            m = load_stored_metrics(p2_mp)
            full_bacc = m.get("test", {}).get("bacc", np.nan)
            json_probs  = m.get("test", {}).get("probs",  [])
            json_labels = m.get("test", {}).get("labels", [])
        else:
            full_bacc   = np.nan
            json_probs  = []
            json_labels = []
        fold_full_bacc.append(full_bacc)

        # P2 per-modality-subset BAcc from aligned stored probs
        test_recs = build_records(splits_csv, fold, split="test",
                                  outer_split=outer_split)

        aligned_probs = None
        if json_probs and json_labels and test_recs:
            aligned_probs = _align_probs_to_records(
                json_labels, json_probs, test_recs)
            if aligned_probs is None:
                print(f"  [warn] fold {fold}: probs alignment failed "
                      f"(json={len(json_labels)}, csv={len(test_recs)}) — "
                      f"subset BAcc will be NaN")

        for mod in MODALITIES:
            pres_col = _pres_col(mod)
            if aligned_probs is not None:
                mask         = np.array([r.get(pres_col, False) for r in test_recs], dtype=bool)
                valid_probs  = aligned_probs[mask]
                valid_labels = np.array([r["label"] for r in test_recs])[mask]
                clean        = ~np.isnan(valid_probs)
                sub_bacc     = _safe_bacc(valid_labels[clean], valid_probs[clean])
                fold_sub_n[mod].append(int(clean.sum()))
            else:
                sub_bacc = np.nan
                fold_sub_n[mod].append(0)
            fold_sub_bacc[mod].append(sub_bacc)

    # ── Summary statistics ───────────────────────────────────────────────────
    def _mean(vals):
        v = [x for x in vals if np.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")

    def _std(vals):
        v = [x for x in vals if np.isfinite(x)]
        return float(np.std(v)) if len(v) > 1 else 0.0

    p1_means  = {mod: _mean(p1_fold_vals[mod])   for mod in MODALITIES}
    p2_sub_means = {mod: _mean(fold_sub_bacc[mod]) for mod in MODALITIES}
    p2_sub_stds  = {mod: _std(fold_sub_bacc[mod])  for mod in MODALITIES}
    p2_full_mean = _mean(fold_full_bacc)
    mods = MODALITIES
    x    = np.arange(len(mods))

    # Per-fold weighted-average P1 BAcc (weights = n patients with that modality)
    fold_bacc_p1_wavg: List[float] = []
    for fi in range(len(folds)):
        ws, vs = [], []
        for mod in mods:
            n = fold_sub_n[mod][fi] if fi < len(fold_sub_n[mod]) else 0
            v = p1_fold_vals[mod][fi] if fi < len(p1_fold_vals[mod]) else float("nan")
            if n > 0 and np.isfinite(v):
                ws.append(n); vs.append(v)
        fold_bacc_p1_wavg.append(float(np.average(vs, weights=ws)) if ws else float("nan"))
    bacc_p1_wavg = _mean(fold_bacc_p1_wavg)

    # Degenerate P1: always-predict-negative (mean sens == 0 across all folds)
    p1_degen: Dict[str, bool] = {}
    for mod in MODALITIES:
        finite_sens = [s for s in p1_fold_sens[mod] if np.isfinite(s)]
        p1_degen[mod] = bool(finite_sens) and float(np.mean(finite_sens)) == 0.0
    for mod in MODALITIES:
        if p1_degen[mod]:
            print(f"  [WARN] P1_{mod}: degenerate classifier (sens=0 in all folds) — "
                  f"BAcc={p1_means[mod]:.3f} reflects always-predict-negative, "
                  f"not real performance")

    print(f"\n  {'Modality':<10} {'n/fold':>7} {'P1_unimodal':>12} "
          f"{'P2_all_test':>12} {'P2_mod_subset':>14}")
    print("  " + "-" * 62)
    rows_csv = []
    for mod in mods:
        n_mean = int(np.mean(fold_sub_n[mod])) if fold_sub_n[mod] else 0
        print(f"  {mod:<10} {n_mean:>7} {p1_means[mod]:>12.3f} "
              f"{p2_full_mean:>12.3f} {p2_sub_means[mod]:>14.3f}")
        rows_csv.append({
            "modality": mod, "n_per_fold": n_mean,
            "P1_unimodal": p1_means[mod],
            "P2_all_test": p2_full_mean,
            "P2_mod_subset": p2_sub_means[mod],
        })
    print(f"\n  P2 full mean BAcc (all test patients, per-fold avg): {p2_full_mean:.3f}")
    print(f"  P1 weighted-avg BAcc (wt by n patients per modality): {bacc_p1_wavg:.3f}")

    pd.DataFrame(rows_csv).to_csv(
        out_dir / f"modality_bacc_{best_variant}.csv", index=False)

    # Global y-axis: collect all finite values across folds
    all_finite = (
        [v for mv in p1_fold_vals.values()   for v in mv if np.isfinite(v)] +
        [v for mv in fold_sub_bacc.values()  for v in mv if np.isfinite(v)] +
        [v for v in fold_full_bacc if np.isfinite(v)]
    )
    y_min = max(0.0, min(all_finite) - 0.08) if all_finite else 0.40
    y_max = min(1.0, max(all_finite) + 0.10) if all_finite else 1.00
    y_min = 0.05 * np.floor(y_min / 0.05)
    y_max = 0.05 * np.ceil (y_max / 0.05)

    # ── Shared helpers ────────────────────────────────────────────────────────
    def _add_bar_label(ax, bar, ylim_top, fmt=".3f"):
        h = bar.get_height()
        if not np.isfinite(h):
            return
        y_text = h + 0.008
        if y_text + 0.02 > ylim_top:
            y_text = h - 0.025
            color = "white"
        else:
            color = "black"
        ax.text(bar.get_x() + bar.get_width() / 2, y_text,
                f"{h:{fmt}}", ha="center", va="bottom", fontsize=8,
                color=color, clip_on=True)

    def _draw_panel(ax, pf_p1, pf_sub, pf_full, fold_label,
                    pf_p1_wavg=float("nan"), pf_p1_degen=None):
        """Draw a single 3-bar modality comparison panel."""
        bw = 0.26
        ax.set_xlim(-0.5, len(mods) - 0.5)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0.5, color="#aab7b8", linestyle="-", linewidth=0.8, zorder=0)

        b1 = ax.bar(x - bw, pf_p1, bw,
                    label="P1 unimodal  (test patients with that modality)",
                    color="#27ae60", alpha=0.88, zorder=3)
        b2 = ax.bar(x, [pf_full] * len(mods), bw,
                    label=f"P2 multimodal  (all test patients)  {pf_full:.3f}",
                    color="#2980b9", alpha=0.88, zorder=3)
        b3 = ax.bar(x + bw, pf_sub, bw,
                    label="P2 multimodal  (same subset as P1)",
                    color="#e67e22", alpha=0.88, zorder=3)

        # Grey out degenerate P1 bars (always-predict-negative)
        if pf_p1_degen:
            for mi, mod in enumerate(mods):
                if pf_p1_degen.get(mod, False):
                    b1[mi].set_color("#aab7b8")
                    b1[mi].set_hatch("///")
                    b1[mi].set_edgecolor("#666666")

        for bar in list(b1) + list(b2) + list(b3):
            _add_bar_label(ax, bar, y_max)

        if np.isfinite(pf_p1_wavg):
            ax.axhline(pf_p1_wavg, color="#c0392b", linestyle="--", linewidth=1.5,
                       zorder=4, label=f"P1 wt.avg (by n patients)  {pf_p1_wavg:.3f}")

        tick_labels = [f"{m}*" if pf_p1_degen and pf_p1_degen.get(m, False) else m
                       for m in mods]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=12, fontweight="bold")
        ax.set_ylabel("Balanced Accuracy", fontsize=11)
        ax.set_yticks(np.arange(y_min, y_max + 0.01, 0.05))
        ax.tick_params(axis="y", labelsize=9)
        ax.set_title(fold_label, fontsize=12, fontweight="bold", pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.85,
                  handlelength=1.5, borderpad=0.6)
        ax.grid(axis="y", linewidth=0.4, alpha=0.4, zorder=0)
        if pf_p1_degen and any(pf_p1_degen.get(m, False) for m in mods):
            degen_mods = [m for m in mods if pf_p1_degen.get(m, False)]
            ax.text(0.01, 0.01,
                    f"* degenerate (sens=0): {', '.join(degen_mods)}",
                    transform=ax.transAxes, fontsize=8, color="#888888",
                    va="bottom")

    # ── Summary plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(-0.5, len(mods) - 0.5)
    ax.set_ylim(y_min, y_max)
    ax.axhline(0.5, color="#aab7b8", linestyle="-", linewidth=0.8, zorder=0,
               label="Chance (0.5)")

    p1_vals     = [p1_means[m]     for m in mods]
    p2_sub_list = [p2_sub_means[m] for m in mods]
    p2_sub_err  = [p2_sub_stds[m]  for m in mods]

    b1 = ax.bar(x - 0.26, p1_vals, 0.26,
                label="P1 unimodal  (test patients with that modality)",
                color="#27ae60", alpha=0.88, zorder=3)
    b2 = ax.bar(x, [p2_full_mean] * len(mods), 0.26,
                label=f"P2 multimodal  (all test patients)  {p2_full_mean:.3f}",
                color="#2980b9", alpha=0.88, zorder=3)
    b3 = ax.bar(x + 0.26, p2_sub_list, 0.26,
                yerr=p2_sub_err, capsize=5,
                error_kw={"elinewidth": 1.4, "capthick": 1.4},
                label="P2 multimodal  (same subset as P1)  mean±std",
                color="#e67e22", alpha=0.88, zorder=3)

    # Grey out degenerate P1 bars (always-predict-negative)
    for mi, mod in enumerate(mods):
        if p1_degen.get(mod, False):
            b1[mi].set_color("#aab7b8")
            b1[mi].set_hatch("///")
            b1[mi].set_edgecolor("#666666")

    for bar in list(b1) + list(b2) + list(b3):
        _add_bar_label(ax, bar, y_max)

    if np.isfinite(bacc_p1_wavg):
        ax.axhline(bacc_p1_wavg, color="#c0392b", linestyle="--", linewidth=1.8,
                   zorder=4, label=f"P1 wt.avg (by n patients)  {bacc_p1_wavg:.3f}")

    # Per-fold dots on P1 and P2-subset bars
    rng = np.random.default_rng(42)
    for mi, mod in enumerate(mods):
        for fv in p1_fold_vals[mod]:
            if np.isfinite(fv):
                ax.scatter(mi - 0.26 + rng.uniform(-0.05, 0.05), fv,
                           color="white", edgecolors="#1a5e34", s=28, zorder=6,
                           linewidths=1.0)
        for fv in fold_sub_bacc[mod]:
            if np.isfinite(fv):
                ax.scatter(mi + 0.26 + rng.uniform(-0.05, 0.05), fv,
                           color="white", edgecolors="#7d4e00", s=28, zorder=6,
                           linewidths=1.0)

    split_tag = f"split{outer_split}" if outer_split is not None else "combined"
    tick_labels_sum = [f"{m}*" if p1_degen.get(m, False) else m for m in mods]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels_sum, fontsize=13, fontweight="bold")
    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_yticks(np.arange(y_min, y_max + 0.01, 0.05))
    ax.tick_params(axis="y", labelsize=10)
    ax.set_title(
        f"BAcc by modality — ACR  ({split_tag},  {best_variant})\n"
        f"Mean across {len(folds)} folds  |  ±std on orange bars  |  dots = individual folds",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.88,
              handlelength=1.5, borderpad=0.7)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if any(p1_degen.values()):
        degen_mods_sum = [m for m in mods if p1_degen.get(m, False)]
        ax.text(0.01, 0.01,
                f"* degenerate (sens=0 in all folds): {', '.join(degen_mods_sum)}",
                transform=ax.transAxes, fontsize=9, color="#888888", va="bottom")
    plt.tight_layout()

    path = out_dir / f"modality_bacc_{split_tag}_{best_variant}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")

    # ── Per-fold individual PNGs ──────────────────────────────────────────────
    for fi, fold in enumerate(folds):
        pf_p1   = [p1_fold_vals[m][fi] if fi < len(p1_fold_vals[m]) else float("nan")
                   for m in mods]
        pf_sub  = [fold_sub_bacc[m][fi] if fi < len(fold_sub_bacc[m]) else float("nan")
                   for m in mods]
        pf_full = fold_full_bacc[fi] if fi < len(fold_full_bacc) else float("nan")
        pf_wavg = fold_bacc_p1_wavg[fi] if fi < len(fold_bacc_p1_wavg) else float("nan")

        fig_f, ax_f = plt.subplots(figsize=(10, 5.5))
        _draw_panel(ax_f, pf_p1, pf_sub, pf_full,
                    f"Fold {fold} — ACR  ({split_tag},  {best_variant})",
                    pf_p1_wavg=pf_wavg, pf_p1_degen=p1_degen)
        plt.tight_layout()
        path_f = out_dir / f"modality_bacc_fold{fold}_{split_tag}_{best_variant}.png"
        fig_f.savefig(path_f, dpi=150, bbox_inches="tight"); plt.close(fig_f)
        print(f"  Saved: {path_f}")

    # ── 2×2 grid (all folds) ─────────────────────────────────────────────────
    nf    = len(folds)
    ncols = min(2, nf)
    nrows = (nf + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 9, nrows * 5.5),
                                squeeze=False)
    fig2.suptitle(
        f"BAcc by modality — ACR  ({split_tag},  {best_variant})\n"
        f"Green = P1 unimodal  |  Blue = P2 all test  |  Orange = P2 same subset  "
        f"|  Red dashed = P1 wt.avg",
        fontsize=13, fontweight="bold", y=1.01
    )
    for fi, fold in enumerate(folds):
        ax2    = axes2[fi // ncols][fi % ncols]
        pf_p1  = [p1_fold_vals[m][fi] if fi < len(p1_fold_vals[m]) else float("nan")
                  for m in mods]
        pf_sub = [fold_sub_bacc[m][fi] if fi < len(fold_sub_bacc[m]) else float("nan")
                  for m in mods]
        pf_full = fold_full_bacc[fi] if fi < len(fold_full_bacc) else float("nan")
        pf_wavg = fold_bacc_p1_wavg[fi] if fi < len(fold_bacc_p1_wavg) else float("nan")
        _draw_panel(ax2, pf_p1, pf_sub, pf_full, f"Fold {fold}",
                    pf_p1_wavg=pf_wavg, pf_p1_degen=p1_degen)
    for idx in range(nf, nrows * ncols):
        axes2[idx // ncols][idx % ncols].set_visible(False)
    plt.tight_layout()
    path_grid = out_dir / f"modality_bacc_perfold_{split_tag}_{best_variant}.png"
    fig2.savefig(path_grid, dpi=150, bbox_inches="tight"); plt.close(fig2)
    print(f"  Saved: {path_grid}")
    print(f"  Outputs → {out_dir}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Unified v6 analysis suite")
    p.add_argument("--results_dir",  type=str, required=True)
    p.add_argument("--samples_dir",  type=str, required=True)
    p.add_argument("--splits_csv",   type=str, required=True)
    p.add_argument("--output_dir",   type=str, default="./analysis_v6")
    p.add_argument("--folds",        nargs="+", type=int, default=[0,1,2,3])
    p.add_argument("--outer_splits", nargs="+", type=int, default=None,
                   help="Nested-CV outer split indices (0-4). "
                        "When set, looks for split{N}_fold{M} columns. "
                        "When omitted, uses fold_{M} (old layout).")
    p.add_argument("--tasks",        nargs="+",
                   choices=["metrics_table", "rep_maps", "attn_maps", "combo_table",
                            "cluster_presence", "spatial_scatter",
                            "cluster_count_viz", "clinical_token_viz",
                            "unimodal_p2", "modality_bacc", "umap_acr", "all"],
                   default=["all"])
    # Model
    p.add_argument("--hidden_dim",   type=int, default=256)
    p.add_argument("--dropout",      type=float, default=0.4)
    # Attn maps
    p.add_argument("--adata_path",   type=str, default=None)
    p.add_argument("--cluster_col",  type=str, default="subclusters_merged")
    p.add_argument("--max_patches_umap", type=int, default=5000)
    p.add_argument("--n_per_split",  type=int, default=20,
                   help="Per-sample attention plots per split")
    # Combo table
    p.add_argument("--min_combo_n",  type=int, default=3)
    # New aggregate-based tasks
    p.add_argument("--agg_mods",     nargs="+", default=None,
                   choices=AGG_MODS,
                   help="Modalities to use for aggregate tasks (default: all)")
    p.add_argument("--n_spatial_per_label", type=int, default=5,
                   help="Patients per label for spatial_scatter task")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# TASK: UNIMODAL P2 EVALUATION
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def _p2_single_mod_inference(
    model: nn.Module,
    target_mod: str,
    all_recs: List[dict],
    bags: Dict,
    device: torch.device,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Run P2 model with only target_mod set; all other modality slots = None.
    Returns {split_name: (probs_array, labels_array)} for patients that
    have target_mod present. Labels from all_recs (CSV-sourced)."""
    from collections import defaultdict
    split_buf: Dict[str, Tuple[List, List]] = defaultdict(lambda: ([], []))

    for rec in all_recs:
        t = bags.get(rec["stem"], {}).get(target_mod)
        if t is None:
            continue
        bags_single: Dict[str, Optional[torch.Tensor]] = {m: None for m in MODALITIES}
        bags_single[target_mod] = t.to(device)
        try:
            out = model(bags_single, device)
            if isinstance(out, tuple):
                out = out[0]
            if not isinstance(out, torch.Tensor) or out.numel() == 0:
                continue
            prob = float(torch.sigmoid(out.detach().squeeze()))
        except Exception:
            continue
        split_buf[rec["split"]][0].append(prob)
        split_buf[rec["split"]][1].append(rec["label"])

    return {sp: (np.array(ps), np.array(ls))
            for sp, (ps, ls) in split_buf.items()
            if len(ps) >= 5 and len(np.unique(ls)) > 1}


def _plot_unimodal_comparison(
    p1_metrics: Dict,   # {(mod, split, fold): {metric: float}}
    p2_metrics: Dict,   # {(vtag, mod, split, fold): {metric: float}}
    folds: List[int],
    out_dir: Path,
):
    """Bar chart per modality × split: P1 baseline (blue) vs every P2 variant (orange).
    Dashed reference line at P1 mean. Error bars = std over folds."""
    vtags = sorted({k[0] for k in p2_metrics})
    if not vtags:
        print("  [warn] No P2 unimodal results to plot"); return

    all_rows = []

    for mod in MODALITIES:
        for split in ["val", "test"]:
            # ── collect mean±std per variant ──────────────────────
            x_labels, means_bacc, stds_bacc, means_auc, stds_auc, colors = \
                [], [], [], [], [], []

            # P1 baseline
            p1_b = [p1_metrics.get((mod, split, f), {}).get("bacc", np.nan) for f in folds]
            p1_a = [p1_metrics.get((mod, split, f), {}).get("auc",  np.nan) for f in folds]
            p1_b = [v for v in p1_b if not np.isnan(v)]
            p1_a = [v for v in p1_a if not np.isnan(v)]
            if not p1_b:
                continue  # P1 never ran for this mod

            x_labels.append(f"P1\n{mod}")
            means_bacc.append(np.mean(p1_b)); stds_bacc.append(np.std(p1_b))
            means_auc.append(np.mean(p1_a));  stds_auc.append(np.std(p1_a))
            colors.append("#4e79a7")

            # P2 variants
            for vtag in vtags:
                p2_b = [p2_metrics.get((vtag, mod, split, f), {}).get("bacc", np.nan) for f in folds]
                p2_a = [p2_metrics.get((vtag, mod, split, f), {}).get("auc",  np.nan) for f in folds]
                p2_b = [v for v in p2_b if not np.isnan(v)]
                p2_a = [v for v in p2_a if not np.isnan(v)]
                x_labels.append(vtag.replace("_", "\n"))
                means_bacc.append(np.mean(p2_b) if p2_b else np.nan)
                stds_bacc.append(np.std(p2_b)   if p2_b else 0)
                means_auc.append(np.mean(p2_a)  if p2_a else np.nan)
                stds_auc.append(np.std(p2_a)    if p2_a else 0)
                colors.append("#f28e2b")
                # CSV rows
                for fold in folds:
                    for metric in ["bacc", "auc", "mcc", "auprc"]:
                        all_rows.append({
                            "mod": mod, "split": split, "fold": fold,
                            "vtag": vtag,
                            metric: p2_metrics.get((vtag, mod, split, fold), {}).get(metric, np.nan),
                        })

            n_bars = len(x_labels)
            fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.4 * n_bars), 5.5))
            fig.suptitle(
                f"Unimodal performance — {mod} only — {split.upper()}\n"
                f"(P2 models run with {mod} input only; dashed = P1 baseline)",
                fontsize=12)

            p1_mean_b = means_bacc[0]
            p1_mean_a = means_auc[0]
            x = np.arange(n_bars)

            for ax, means, stds, ylabel, p1_ref in [
                (axes[0], means_bacc, stds_bacc, "Balanced Accuracy", p1_mean_b),
                (axes[1], means_auc,  stds_auc,  "AUROC",             p1_mean_a),
            ]:
                bars = ax.bar(x, means, yerr=stds, capsize=4,
                              color=colors, edgecolor="white", linewidth=0.5,
                              error_kw=dict(elinewidth=1.2, ecolor="#444"))
                if not np.isnan(p1_ref):
                    ax.axhline(p1_ref, color="#4e79a7", linestyle="--",
                               linewidth=1.4, alpha=0.7, label=f"P1 baseline ({p1_ref:.3f})")

                # Annotate each bar
                for bar, val in zip(bars, means):
                    if np.isnan(val): continue
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", va="bottom",
                            fontsize=7, color="#333")

                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, fontsize=7.5, rotation=40, ha="right")
                ax.set_ylabel(ylabel, fontsize=10)
                ax.set_ylim(0, 1.08)
                ax.grid(axis="y", alpha=0.25)
                ax.legend(fontsize=8)
                # Shade background for P2 bars to visually separate from P1
                ax.axvspan(0.5, n_bars - 0.5, color="#fff7ed", alpha=0.4, zorder=0)

            plt.tight_layout()
            fname = out_dir / f"unimodal_p2_{mod}_{split}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved {fname.name}")

    # Save CSV
    if all_rows:
        import pandas as pd
        pd.DataFrame(all_rows).to_csv(out_dir / "unimodal_p2_metrics.csv", index=False)


def task_unimodal_p2(
    results_dir: Path,
    samples_dir: str,
    splits_csv: str,
    folds: List[int],
    out_dir: Path,
    outer_splits: Optional[List[int]] = None,
):
    """
    For each P2 variant, run inference with only a single modality present
    (all other modality inputs set to None). Compare against P1 baselines.

    Answers: does a multimodal model degrade vs P1 when given only one modality?
    Uses all patients who have that modality — independent of whether they had
    other modalities during training.
    """
    print("\n" + "="*65)
    print("  TASK: Unimodal P2 Evaluation")
    print("="*65)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    p2_cfgs = scan_p2_configs(results_dir, folds, outer_splits)
    if not p2_cfgs:
        print("  [warn] No P2 configs found"); return

    # ── P1 baselines from stored probs (labels from splits CSV) ────────────
    print("  Loading P1 predictions …")
    p1_stem_preds = load_p1_preds_per_stem(results_dir, splits_csv, folds, outer_splits)
    # Aggregate → {(mod, split, fold): {auc, bacc, …}}
    _p1_buf: Dict = {}
    for (fold, mod, _stem), pred in p1_stem_preds.items():
        k = (fold, mod, pred["split"])
        if k not in _p1_buf:
            _p1_buf[k] = {"probs": [], "labels": []}
        _p1_buf[k]["probs"].append(pred["prob"])
        _p1_buf[k]["labels"].append(pred["label"])

    p1_metrics: Dict = {}   # {(mod, split, fold): {auc, bacc, …}}
    for (fold, mod, split), buf in _p1_buf.items():
        p1_metrics[(mod, split, fold)] = _safe_metrics(
            np.array(buf["labels"]), np.array(buf["probs"]))

    # ── P2 single-mod inference ─────────────────────────────────────────────
    p2_metrics: Dict = {}   # {(vtag, mod, split, fold): {auc, bacc, …}}

    for fold in folds:
        outer_split = outer_splits[0] if outer_splits else None
        all_recs = build_records(splits_csv, fold, outer_split=outer_split)
        stems    = list({r["stem"] for r in all_recs})
        print(f"\n  Fold {fold}: loading {len(stems)} bags …")
        bags     = preload_bags(stems, samples_dir, quiet=True)
        all_recs = update_presence(all_recs, bags)

        fold_cfgs = [c for c in p2_cfgs if c["fold"] == fold]
        print(f"  Fold {fold}: {len(fold_cfgs)} P2 variants to evaluate")

        for cfg in fold_cfgs:
            vtag  = cfg["vtag"]
            ckpt  = cfg["model_path"]
            if not ckpt.exists():
                print(f"    [{vtag}] checkpoint missing — skip")
                continue

            # Load P2 model (reconstruct from state dict)
            print(f"    [{vtag}] loading …", end=" ", flush=True)
            try:
                obj = torch.load(ckpt, map_location="cpu", weights_only=False)
                if isinstance(obj, nn.Module):
                    model = obj
                elif isinstance(obj, dict):
                    state   = obj.get("model", obj)
                    variant, iter_r, slot_k = _vtag_to_variant_params(vtag)
                    p1_dir  = ckpt.parent.parent / "phase1"
                    tm      = _get_train_module()
                    model   = tm.build_p2_model(variant, p1_dir,
                                                iter_r=iter_r, slot_k=slot_k)
                    model.load_state_dict(state, strict=False)
                else:
                    print("unknown format — skip"); continue
            except Exception as e:
                print(f"failed ({e}) — skip"); continue

            model = model.to(device).eval()

            # Run each modality independently
            for mod in MODALITIES:
                split_res = _p2_single_mod_inference(
                    model, mod, all_recs, bags, device)
                for split, (probs, labels) in split_res.items():
                    p2_metrics[(vtag, mod, split, fold)] = _safe_metrics(labels, probs)
            n_mod_results = sum(1 for k in p2_metrics if k[2] in ("val","test") and k[3] == fold and k[0] == vtag)
            print(f"done  ({n_mod_results} mod×split results)")

            del model; _gc()

        del bags; _gc()

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\n  Generating comparison plots …")
    _plot_unimodal_comparison(p1_metrics, p2_metrics, folds, out_dir)
    print(f"  Outputs → {out_dir}")


def main():
    args = parse_args()
    tasks = set(args.tasks)
    if "all" in tasks:
        tasks = {"metrics_table", "rep_maps", "attn_maps", "combo_table",
                 "cluster_presence", "spatial_scatter",
                 "cluster_count_viz", "clinical_token_viz", "unimodal_p2",
                 "modality_bacc", "umap_acr"}

    results_dir = Path(args.results_dir)
    out_root    = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Results dir : {results_dir}")
    print(f"Output dir  : {out_root}")
    print(f"Folds       : {args.folds}")
    print(f"Tasks       : {sorted(tasks)}")

    outer_splits = args.outer_splits  # None or list of ints

    if "metrics_table" in tasks:
        task_metrics_table(
            results_dir, args.folds,
            out_root / "metrics_table", args.splits_csv,
            outer_splits=outer_splits)

    if "rep_maps" in tasks:
        task_rep_maps(
            results_dir, args.samples_dir, args.splits_csv,
            args.folds, out_root / "rep_maps",
            args.hidden_dim, args.dropout,
            outer_splits=outer_splits)

    if "attn_maps" in tasks:
        task_attn_maps(
            results_dir, args.samples_dir, args.splits_csv,
            args.folds, out_root / "attn_maps",
            args.hidden_dim, args.dropout,
            args.adata_path, args.cluster_col,
            args.max_patches_umap, args.n_per_split,
            outer_splits=outer_splits)

    if "combo_table" in tasks:
        task_combo_table(
            results_dir, args.samples_dir, args.splits_csv,
            args.folds, out_root / "combo_table",
            args.hidden_dim, args.dropout, args.min_combo_n,
            outer_splits=outer_splits)

    # ── New aggregate-based tasks (use new .pt format fields) ─────────────────
    if "cluster_presence" in tasks:
        task_cluster_presence_maps(
            args.samples_dir, args.splits_csv,
            args.folds, out_root / "cluster_presence",
            mods=args.agg_mods, outer_splits=outer_splits)

    if "spatial_scatter" in tasks:
        task_spatial_scatter(
            args.samples_dir, args.splits_csv,
            args.folds, out_root / "spatial_scatter",
            mods=args.agg_mods,
            n_per_label=args.n_spatial_per_label,
            outer_splits=outer_splits)

    if "cluster_count_viz" in tasks:
        task_cluster_count_viz(
            args.samples_dir, args.splits_csv,
            args.folds, out_root / "cluster_count_viz",
            mods=args.agg_mods, outer_splits=outer_splits)

    if "clinical_token_viz" in tasks:
        task_clinical_token_viz(
            args.samples_dir, args.splits_csv,
            args.folds, out_root / "clinical_token_viz",
            outer_splits=outer_splits)

    if "unimodal_p2" in tasks:
        task_unimodal_p2(
            results_dir, args.samples_dir, args.splits_csv,
            args.folds, out_root / "unimodal_p2",
            outer_splits=outer_splits)

    if "modality_bacc" in tasks:
        task_modality_bacc(
            results_dir, args.splits_csv,
            args.folds, out_root / "modality_bacc",
            outer_splits=outer_splits)

    if "umap_acr" in tasks:
        task_umap_acr(
            results_dir, args.samples_dir, args.splits_csv,
            args.folds, out_root / "umap_acr",
            outer_splits=outer_splits)

    print(f"\nDone. All outputs in: {out_root}")


if __name__ == "__main__":
    main()