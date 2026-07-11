"""
interpret_slot_shared.py — Interpretability for SharedSlotMIL (v8)
===================================================================

SharedSlotMIL architecture:
  Stage 1+2  Per-modality ModalFFNEncoder + MHASlotAttn → (K, H) per modality
  Stage 3    Mean over present modalities → slots_agg (K, H)
  Stage 4    Per-task gated ABMIL → alpha_task (K, 1) per task
  Stage 5    Per-task head (cls / Cox)

What we extract per patient:
  slot_attn_{mod}     (K, N_mod)   patch→slot routing (MHA weights)
  alpha_{task}        (K,)         per-task slot importance
  h_{mod}             (N, H)       backbone features (optional, large)

What we visualize:
  1. Per-task slot alpha bar charts — which of K=128 slots matter per task
  2. Slot×Cluster heatmaps — which cluster types route into each slot
  3. Sankey/alluvial diagram — cluster types → top active slots → tasks

Usage
-----
  python3 interpretability/interpret_slot_shared.py \\
      --split 1 --fold 0 \\
      --p2-tag shared_combined \\          # or "shared" for per-fold model
      --split-set test \\
      --out-dir interpretability/slot_shared_s1f0

SLURM: see submit_interpret_slot_shared.sh
"""

from __future__ import annotations
import argparse, gc, json, sys, warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")

# ── project imports ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from mil.data.loader   import preload_bags
from mil.data.splits   import build_splits_multitask
from mil.data.registry import MODALITIES
from mil.models.builders import build_model_v8
from mil.models.phase2   import SharedSlotMIL

SAMPLES_DIR  = "/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV   = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
RESULTS_DIR  = Path("/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 42

TASKS        = ["acr_cls", "acr_surv", "clad", "death"]
TASK_LABELS  = {"acr_cls": "ACR-cls", "acr_surv": "ACR-TTE",
                "clad": "CLAD-TTE", "death": "Death-TTE"}
TASK_COLORS  = {"acr_cls": "#1565C0", "acr_surv": "#0277BD",
                "clad": "#FB8500", "death": "#C62828"}
MOD_COLORS   = {"HE": "#E53935", "BAL": "#1E88E5",
                "CT": "#43A047", "Clinical": "#8E24AA"}
MOD_TO_CLUSTER_KEY = {"HE": "HE_cells", "BAL": "BAL_cells", "CT": "CT_cells"}
CLUSTER_MODS = list(MOD_TO_CLUSTER_KEY.keys()) + ["Clinical"]

# Outcome label names per task (cls=binary, TTE=quartile bins)
TASK_LABEL_NAMES = {
    "acr_cls":  {0: "No ACR",       1: "ACR+"},
    "acr_surv": {0: "No ACR (TTE)", 1: "ACR+ (TTE)"},
    "clad":     {0: "Q1 (early)", 1: "Q2", 2: "Q3", 3: "Q4 (late)"},
    "death":    {0: "Q1 (early)", 1: "Q2", 2: "Q3", 3: "Q4 (late)"},
}
TASK_N_LABELS = {"acr_cls": 2, "acr_surv": 2, "clad": 4, "death": 4}
TASK_LABEL_COLORS = {
    "acr_cls":  ["#4CAF50", "#F44336"],
    "acr_surv": ["#81C784", "#E57373"],
    "clad":     ["#C8E6C9", "#66BB6A", "#EF5350", "#B71C1C"],
    "death":    ["#BBDEFB", "#42A5F5", "#EF5350", "#7B1FA2"],
}


# ══════════════════════════════════════════════════════════════════════════════
# ATTENTION CAPTURE
# ══════════════════════════════════════════════════════════════════════════════

class _MHACapture:
    """Forward hook to capture MHA attention weights (K, N)."""
    def __init__(self):
        self.weights: Optional[torch.Tensor] = None
        self._handle = None

    def hook(self, mha_module: nn.MultiheadAttention):
        self._handle = mha_module.register_forward_hook(self._cb)

    def _cb(self, module, inp, output):
        # MHA output: (attn_out, attn_weights) where attn_weights=(1, K, N)
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            w = output[1]
            if w is not None:
                self.weights = w.detach().cpu().squeeze(0)  # (K, N)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@contextmanager
def _capture_slot_attns(model: SharedSlotMIL):
    """Capture per-modality MHA attention weights inside all MHASlotAttn modules."""
    caps: Dict[str, _MHACapture] = {}
    for mod, sa in model.slot_attns.items():
        c = _MHACapture()
        c.hook(sa.mha)
        caps[mod] = c
    try:
        yield caps
    finally:
        for c in caps.values():
            c.remove()


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_shared_slot(model: SharedSlotMIL, bags: dict, device) -> Optional[dict]:
    """
    Extract per-patient interpretability tensors from a SharedSlotMIL forward pass.

    Returns dict with:
      slot_attn_{mod}  (K, N_mod)  patch→slot routing (MHA attn weights, last iter)
      alpha_{task}     (K,)        per-task slot importance
      h_{mod}          (N, H)      backbone patch features (saved for cluster labels)
      mods_present     list[str]
    """
    model.eval()
    result: dict = {}
    h_store: Dict[str, torch.Tensor] = {}

    # Stage 1: backbone features (without slot attn yet)
    for mod, enc in model.encoders.items():
        t = bags.get(mod)
        if t is None:
            continue
        t = t.to(device, non_blocking=True)
        if mod == "HE" and t.shape[0] > model.max_he_patches:
            t = t[:model.max_he_patches]
        h = enc.encode_patches(t)              # (N, H)
        h_store[mod] = h
        result[f"h_{mod}"] = h.cpu().numpy()

    if not h_store:
        return None
    result["mods_present"] = list(h_store.keys())

    # Stage 2: slot attention with capture, Stage 3: mean aggregation, Stage 4: alpha
    with _capture_slot_attns(model) as caps:
        mod_slots: List[torch.Tensor] = []
        for mod, h in h_store.items():
            s = model.slot_attns[mod](h, model.shared_slots)  # (K, H)
            mod_slots.append(s)
            attn_w = caps[mod].weights  # (K, N) or None
            if attn_w is not None:
                result[f"slot_attn_{mod}"] = attn_w.numpy()   # (K, N)

    # Raw pre-softmax QK scores — use same normalised inputs the MHA actually sees:
    #   Q input: norm_q(slots)  (LayerNorm, same as MHASlotAttn.forward)
    #   K input: L2_norm(h)     (same as MHASlotAttn.forward)
    #   scale:   1/sqrt(d_k) where d_k = embed_dim / n_heads
    slots = model.shared_slots  # (K, H)
    for mod, h in h_store.items():
        try:
            sa   = model.slot_attns[mod]
            mha  = sa.mha
            Hd   = mha.embed_dim
            nh   = mha.num_heads
            dk   = Hd // nh
            w    = mha.in_proj_weight; b = mha.in_proj_bias
            slots_normed = sa.norm_q(slots)
            h_normed     = F.normalize(h, dim=-1)
            Q = F.linear(slots_normed, w[:Hd],    b[:Hd]    if b is not None else None)
            K = F.linear(h_normed,    w[Hd:2*Hd], b[Hd:2*Hd] if b is not None else None)
            Q_h = Q.view(-1, nh, dk).transpose(0, 1)        # (nh, K, dk)
            K_h = K.view(-1, nh, dk).transpose(0, 1)        # (nh, N, dk)
            raw = (Q_h @ K_h.transpose(-1, -2)) * (dk ** -0.5)  # (nh, K, N)
            result[f"slot_attn_raw_{mod}"] = raw.mean(0).cpu().numpy()   # (K, N)
        except Exception:
            pass

    if not mod_slots:
        return None

    slots_agg = torch.stack(mod_slots, dim=0).mean(dim=0)     # (K, H)

    # Stage 4: per-task gated ABMIL → alpha
    for task in model.task_names:
        gate  = model.abmil_V[task](slots_agg) * model.abmil_U[task](slots_agg)
        alpha = torch.softmax(model.abmil_w[task](gate), dim=0)  # (K, 1)
        result[f"alpha_{task}"] = alpha.squeeze(1).cpu().numpy()  # (K,)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTER LABEL HELPERS  (matching interpret_mm_abmil.py conventions)
# ══════════════════════════════════════════════════════════════════════════════

def _load_cluster_labels(stem: str, samples_dir) -> Dict[str, List[str]]:
    pt = Path(samples_dir) / f"{stem}.pt"
    if not pt.exists():
        return {}
    try:
        data = torch.load(pt, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    raw = data.get("cluster_labels", {})
    out = {}
    for mod, key in MOD_TO_CLUSTER_KEY.items():
        labs = raw.get(key)
        if labs is not None:
            out[mod] = labs if isinstance(labs, list) else list(labs)
    return out


def _load_clinical_token_labels(stem: str, samples_dir) -> Optional[List[str]]:
    pt = Path(samples_dir) / f"{stem}.pt"
    if not pt.exists():
        return None
    try:
        data = torch.load(pt, map_location="cpu", weights_only=False)
    except Exception:
        return None
    token_ids = data.get("clinical_token_ids")
    vocab     = data.get("clinical_vocab")
    if token_ids is None or vocab is None:
        return None
    id_to_label = {entry["id"]: entry["label"] for entry in vocab}
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return [id_to_label.get(int(tid), f"unk_{tid}") for tid in token_ids]


# ══════════════════════════════════════════════════════════════════════════════
# COHORT EXTRACTION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_extraction(model: SharedSlotMIL, records, bag_cache, device,
                   out_dir: Path, samples_dir: str) -> List[dict]:
    npy_dir = out_dir / "npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, rec in enumerate(records):
        stem  = rec["stem"]
        entry = bag_cache.get(stem, {})
        bags  = {m: entry.get(m) for m in MODALITIES}

        if all(v is None for v in bags.values()):
            continue

        try:
            r = extract_shared_slot(model, bags, device)
        except Exception as e:
            print(f"  [warn] {stem}: {e}")
            r = None

        if r is None:
            continue

        r["stem"]      = stem
        r["label"]     = rec.get("label")
        r["acr_status"]= rec.get("acr_status", float("nan"))
        for ep in ("clad", "death"):
            r[f"{ep}_time"]  = rec.get(f"{ep}_time",  float("nan"))
            r[f"{ep}_event"] = rec.get(f"{ep}_event", float("nan"))

        # Attach cluster labels from .pt file
        cl_map = _load_cluster_labels(stem, samples_dir)
        for mod, labels in cl_map.items():
            ref = r.get(f"h_{mod}")
            n   = ref.shape[0] if ref is not None else len(labels)
            r[f"cluster_labels_{mod}"] = labels[:n]

        alpha_clin = r.get("h_Clinical")
        if alpha_clin is not None:
            clin_labels = _load_clinical_token_labels(stem, samples_dir)
            if clin_labels is not None:
                r["cluster_labels_Clinical"] = clin_labels[:alpha_clin.shape[0]]

        # Save arrays separately from metadata
        np_data = {k: v for k, v in r.items() if isinstance(v, np.ndarray)}
        meta    = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        if np_data:
            np.savez_compressed(npy_dir / f"{stem}.npz", **np_data)
        try:
            with open(npy_dir / f"{stem}_meta.json", "w") as f:
                json.dump(meta, f, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
        except Exception:
            pass

        results.append(r)
        if (i + 1) % 20 == 0:
            print(f"  extracted {i+1}/{len(records)}", flush=True)
        gc.collect()

    print(f"  Saved {len(results)} samples → {npy_dir}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# COHORT AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def _slot_cluster_matrix(slot_KN: np.ndarray, labels: List[str],
                          all_types: Optional[List[str]] = None
                          ) -> Tuple[np.ndarray, List[str]]:
    """(K, N) slot routing + N cluster labels → (K, C) mean routing per cluster type."""
    K = slot_KN.shape[0]
    n = min(slot_KN.shape[1], len(labels))
    A = slot_KN[:, :n]; labs = labels[:n]
    ctypes = all_types if all_types is not None else sorted(set(labs))
    C = len(ctypes); idx_map = {ct: i for i, ct in enumerate(ctypes)}
    result = np.zeros((K, C), dtype=np.float32)
    counts = np.zeros(C, dtype=np.int32)
    for j, ct in enumerate(labs):
        ci = idx_map.get(ct)
        if ci is not None:
            result[:, ci] += A[:, j]; counts[ci] += 1
    mask = counts > 0
    result[:, mask] /= counts[mask]
    return result, ctypes


def _assign_task_labels_all(results: List[dict]) -> Dict[str, List]:
    """
    Per-patient label for each task:
      acr_cls / acr_surv : binary 0/1 from r['label']
      clad / death       : quartile bin (0-3) of time-to-event
    Returns dict[task → list of label_idx or None].
    """
    labels: Dict[str, List] = {t: [] for t in TASKS}

    for r in results:
        lab = r.get("label")
        labels["acr_cls"].append(lab if lab in (0, 1) else None)
        labels["acr_surv"].append(lab if lab in (0, 1) else None)
        # TTE: defer to second pass after computing quantiles
        labels["clad"].append(None)
        labels["death"].append(None)

    for task, time_key in [("clad", "clad_time"), ("death", "death_time")]:
        times = []
        for r in results:
            v = r.get(time_key)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            times.append(fv)

        valid = [t for t in times if t == t]  # non-nan
        if valid:
            qs = np.nanpercentile(valid, [25, 50, 75])
            for i, t in enumerate(times):
                if t != t:
                    labels[task][i] = None
                elif t <= qs[0]:
                    labels[task][i] = 0
                elif t <= qs[1]:
                    labels[task][i] = 1
                elif t <= qs[2]:
                    labels[task][i] = 2
                else:
                    labels[task][i] = 3
        else:
            # fallback: use ACR binary label
            for i, r in enumerate(results):
                lab = r.get("label")
                labels[task][i] = lab if lab in (0, 1) else None

    return labels


def build_task_label_data(results: List[dict], n_slots: int = 128
                          ) -> Dict[str, Dict[int, Dict]]:
    """
    For each task and label bin, compute mean slot-alpha and patient count.
    Returns: {task: {label_idx: {'alpha': (K,), 'n': int}}}
    """
    task_patient_labels = _assign_task_labels_all(results)
    out: Dict[str, Dict[int, Dict]] = {t: {} for t in TASKS}

    for i, r in enumerate(results):
        for task in TASKS:
            alpha = r.get(f"alpha_{task}")
            if alpha is None or len(alpha) != n_slots:
                continue
            lab = task_patient_labels[task][i]
            if lab is None:
                continue
            if lab not in out[task]:
                out[task][lab] = {"alpha_sum": np.zeros(n_slots, dtype=np.float64), "n": 0}
            out[task][lab]["alpha_sum"] += alpha
            out[task][lab]["n"] += 1

    for task in TASKS:
        for lab, d in out[task].items():
            d["alpha"] = (d["alpha_sum"] / d["n"]) if d["n"] > 0 else d["alpha_sum"]
            del d["alpha_sum"]

    return out


def build_cohort_matrices(results: List[dict], n_slots: int = 128):
    """
    Aggregate per-patient arrays into cohort-level matrices.

    Returns
    -------
    mean_alpha_task   : dict[task → (K,)]   mean slot importance per task
    cluster_slot_mats : dict[mod → (C, K)]  cluster→slot flow (transposed for Sankey)
    cluster_types     : dict[mod → List[str]]
    slot_alpha_by_label: dict[task → {0: (K,), 1: (K,)}]  per-label mean alpha
    """
    # Per-task alpha accumulation
    alpha_accum: Dict[str, List[np.ndarray]] = {t: [] for t in TASKS}
    alpha_by_lab: Dict[str, Dict[int, List[np.ndarray]]] = {t: {0: [], 1: []} for t in TASKS}

    for r in results:
        lab = r.get("label")
        for task in TASKS:
            a = r.get(f"alpha_{task}")
            if a is not None and len(a) == n_slots:
                alpha_accum[task].append(a)
                if lab in (0, 1):
                    alpha_by_lab[task][lab].append(a)

    mean_alpha_task = {}
    for task in TASKS:
        if alpha_accum[task]:
            mean_alpha_task[task] = np.stack(alpha_accum[task]).mean(0)

    slot_alpha_by_label = {}
    for task in TASKS:
        slot_alpha_by_label[task] = {}
        for lab in (0, 1):
            if alpha_by_lab[task][lab]:
                slot_alpha_by_label[task][lab] = np.stack(alpha_by_lab[task][lab]).mean(0)

    # Per-modality cluster→slot matrices (soft = softmax-normalized, raw = pre-softmax)
    cluster_slot_mats:     Dict[str, np.ndarray] = {}
    cluster_slot_mats_raw: Dict[str, np.ndarray] = {}
    cluster_types_all:     Dict[str, List[str]]  = {}

    for mod in CLUSTER_MODS:
        # First pass: collect all cluster types
        all_ctypes = None
        for r in results:
            sa = r.get(f"slot_attn_{mod}")
            cl = r.get(f"cluster_labels_{mod}")
            if sa is None or cl is None:
                continue
            n = min(sa.shape[1], len(cl))
            ctypes = sorted(set(cl[:n]))
            if all_ctypes is None or len(ctypes) > len(all_ctypes):
                all_ctypes = ctypes

        if all_ctypes is None:
            continue
        cluster_types_all[mod] = all_ctypes

        # Second pass: accumulate (K, C) matrices
        mats:     List[np.ndarray] = []
        mats_raw: List[np.ndarray] = []
        for r in results:
            cl = r.get(f"cluster_labels_{mod}")
            if cl is None:
                continue
            sa = r.get(f"slot_attn_{mod}")
            if sa is not None:
                kc, _ = _slot_cluster_matrix(sa, cl, all_types=all_ctypes)
                mats.append(kc)
            sa_raw = r.get(f"slot_attn_raw_{mod}")
            if sa_raw is not None:
                kc_r, _ = _slot_cluster_matrix(sa_raw, cl, all_types=all_ctypes)
                mats_raw.append(kc_r)

        if mats:
            cluster_slot_mats[mod]     = np.stack(mats).mean(0).T     # (C, K)
        if mats_raw:
            cluster_slot_mats_raw[mod] = np.stack(mats_raw).mean(0).T # (C, K)

    return (mean_alpha_task, cluster_slot_mats, cluster_slot_mats_raw,
            cluster_types_all, slot_alpha_by_label)


# ══════════════════════════════════════════════════════════════════════════════
# SANKEY / ALLUVIAL DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

def _bezier_band(ax, x0, x1, y0_bot, y0_top, y1_bot, y1_top, color, alpha=0.55):
    """Draw a filled bezier band (ribbon) between two x positions."""
    cx = (x0 + x1) / 2
    verts = [
        (x0, y0_bot), (cx, y0_bot), (cx, y1_bot), (x1, y1_bot),
        (x1, y1_top), (cx, y1_top), (cx, y0_top), (x0, y0_top),
        (x0, y0_bot),
    ]
    codes = [MPath.MOVETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.CLOSEPOLY]
    path  = MPath(verts, codes)
    patch = PathPatch(path, facecolor=color, edgecolor="none", alpha=alpha, zorder=1)
    ax.add_patch(patch)


def plot_sankey(
    mean_alpha_task: Dict[str, np.ndarray],
    cluster_slot_mats: Dict[str, np.ndarray],
    cluster_types_all: Dict[str, List[str]],
    out_path: Path,
    top_k_slots: int = 20,
    top_k_clusters: int = 30,
):
    """
    3-column Sankey alluvial:  cluster types  →  top active slots  →  tasks

    Left nodes  : cluster types per modality
    Middle nodes: top active slots (selected by max alpha across tasks)
    Right nodes : 4 tasks
    """
    if not mean_alpha_task or not cluster_slot_mats:
        return

    tasks = [t for t in TASKS if t in mean_alpha_task]
    if not tasks:
        return

    # ── Select top active slots ───────────────────────────────────────────────
    alpha_stack = np.stack([mean_alpha_task[t] for t in tasks], axis=0)  # (T, K)
    mean_alpha_all = alpha_stack.mean(0)  # (K,) mean across tasks
    top_slot_idx = np.argsort(mean_alpha_all)[-top_k_slots:][::-1]       # top K slots

    # ── Build cluster→slot flow matrix ───────────────────────────────────────
    # Collect all cluster nodes: (mod, cluster_name) → index
    cluster_nodes: List[Tuple[str, str]] = []
    for mod in CLUSTER_MODS:
        if mod not in cluster_types_all:
            continue
        for ct in cluster_types_all[mod]:
            cluster_nodes.append((mod, ct))

    # Compute flow: for each cluster node, how much flows to each top slot?
    # flow_CS[c, s_rank] = cluster_slot_mats[mod][c_local, slot_idx]
    N_C = len(cluster_nodes)
    N_S = len(top_slot_idx)
    N_T = len(tasks)

    flow_CS = np.zeros((N_C, N_S), dtype=np.float32)
    cluster_offset = 0
    for mod in CLUSTER_MODS:
        if mod not in cluster_slot_mats:
            continue
        C_mod = cluster_slot_mats[mod].shape[0]   # (C_mod, K)
        for ci in range(C_mod):
            flow_CS[cluster_offset + ci] = cluster_slot_mats[mod][ci, top_slot_idx]
        cluster_offset += C_mod

    # Normalize per-column (slot) so edge widths are comparable
    col_sum = flow_CS.sum(0) + 1e-8
    flow_CS_norm = flow_CS / col_sum

    # Slot→task flow: alpha[k, t] for each top slot
    flow_ST = np.zeros((N_S, N_T), dtype=np.float32)
    for si, slot_k in enumerate(top_slot_idx):
        for ti, task in enumerate(tasks):
            flow_ST[si, ti] = mean_alpha_task[task][slot_k]
    # Normalize rows (each slot distributes to tasks)
    row_sum = flow_ST.sum(1, keepdims=True) + 1e-8
    flow_ST_norm = flow_ST / row_sum

    # Prune low-weight cluster nodes
    cluster_strength = flow_CS.sum(1)           # total flow per cluster
    sorted_ci = np.argsort(cluster_strength)[-top_k_clusters:][::-1]

    # ── Node heights (proportional to total flow) ─────────────────────────────
    PAD = 0.008                     # gap between nodes
    X_L, X_M, X_R = 0.0, 0.45, 0.90

    # Left column: top cluster nodes
    visible_ci  = sorted(sorted_ci)
    n_L = len(visible_ci)
    h_L = np.array([cluster_strength[i] for i in visible_ci])
    h_L = h_L / (h_L.sum() + 1e-8) * (1 - PAD * n_L)
    y_L = np.cumsum(np.concatenate([[0], h_L + PAD]))[:-1]   # bottom y per node

    # Middle column: top slots
    h_M = mean_alpha_all[top_slot_idx]
    h_M = h_M / (h_M.sum() + 1e-8) * (1 - PAD * N_S)
    y_M = np.cumsum(np.concatenate([[0], h_M + PAD]))[:-1]

    # Right column: tasks (equal height proportional to sum of incoming flow)
    h_T = flow_ST.sum(0)
    h_T = h_T / (h_T.sum() + 1e-8) * (1 - PAD * N_T)
    y_T = np.cumsum(np.concatenate([[0], h_T + PAD]))[:-1]

    # ── Draw ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, max(12, n_L * 0.35 + 2)))
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.02, 1.12)
    ax.axis("off")

    # Left→Middle edges
    # Track per-slot current fill position (bottom)
    slot_fill_bot = y_M.copy()
    for rank, ci in enumerate(visible_ci):
        mod_ci, name_ci = cluster_nodes[ci]
        color = MOD_COLORS.get(mod_ci, "#888888")
        y_c_bot = y_L[rank]; y_c_top = y_L[rank] + h_L[rank]
        src_fill = y_c_bot

        for si in range(N_S):
            w = flow_CS_norm[ci, si] * h_M[si]
            if w < 1e-5:
                continue
            y_s_bot = slot_fill_bot[si]
            y_s_top = y_s_bot + w
            _bezier_band(ax, X_L + 0.05, X_M - 0.02,
                         src_fill, src_fill + w * h_L[rank] / (h_L[rank] + 1e-8),
                         y_s_bot, y_s_top,
                         color=color, alpha=0.45)
            slot_fill_bot[si] += w
            src_fill += w * h_L[rank] / (h_L[rank] + 1e-8)

    # Middle→Right edges
    task_fill_bot = y_T.copy()
    for si in range(N_S):
        color_s = plt.cm.plasma(si / max(N_S - 1, 1))
        y_s_bot = y_M[si]; y_s_top = y_M[si] + h_M[si]
        src_fill = y_s_bot

        for ti, task in enumerate(tasks):
            w = flow_ST_norm[si, ti] * h_M[si]
            if w < 1e-5:
                continue
            y_t_bot = task_fill_bot[ti]
            y_t_top = y_t_bot + w
            _bezier_band(ax, X_M + 0.05, X_R - 0.02,
                         src_fill, src_fill + w,
                         y_t_bot, y_t_top,
                         color=color_s, alpha=0.45)
            task_fill_bot[ti] += w
            src_fill += w

    # Draw nodes
    NODE_W = 0.04
    for rank, ci in enumerate(visible_ci):
        mod_ci, name_ci = cluster_nodes[ci]
        color = MOD_COLORS.get(mod_ci, "#888888")
        rect = mpatches.FancyBboxPatch(
            (X_L, y_L[rank]), NODE_W, h_L[rank],
            boxstyle="square,pad=0", fc=color, ec="none", zorder=3)
        ax.add_patch(rect)
        ax.text(X_L - 0.01, y_L[rank] + h_L[rank] / 2,
                name_ci, ha="right", va="center",
                fontsize=6, color="#222")

    for si, slot_k in enumerate(top_slot_idx):
        color_s = plt.cm.plasma(si / max(N_S - 1, 1))
        rect = mpatches.FancyBboxPatch(
            (X_M, y_M[si]), NODE_W, h_M[si],
            boxstyle="square,pad=0", fc=color_s, ec="none", zorder=3)
        ax.add_patch(rect)
        ax.text(X_M + NODE_W + 0.005, y_M[si] + h_M[si] / 2,
                f"S{slot_k}", ha="left", va="center", fontsize=6, color="#222")

    for ti, task in enumerate(tasks):
        color_t = TASK_COLORS.get(task, "#555")
        rect = mpatches.FancyBboxPatch(
            (X_R, y_T[ti]), NODE_W, h_T[ti],
            boxstyle="square,pad=0", fc=color_t, ec="none", zorder=3)
        ax.add_patch(rect)
        ax.text(X_R + NODE_W + 0.01, y_T[ti] + h_T[ti] / 2,
                TASK_LABELS.get(task, task), ha="left", va="center",
                fontsize=8, color=color_t, fontweight="bold")

    # Column headers
    ax.text(X_L + NODE_W / 2, 1.05, "Instance clusters", ha="center",
            fontsize=10, fontweight="bold")
    ax.text(X_M + NODE_W / 2, 1.05, f"Top-{N_S} slots", ha="center",
            fontsize=10, fontweight="bold")
    ax.text(X_R + NODE_W / 2, 1.05, "Tasks", ha="center",
            fontsize=10, fontweight="bold")

    # Modality legend
    legend_patches = [mpatches.Patch(color=MOD_COLORS[m], label=m)
                      for m in CLUSTER_MODS if m in cluster_types_all]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.8)

    ax.set_title(
        f"Instance clusters → Shared slots → Tasks\n"
        f"(top {N_S} slots by mean α, top {n_L} cluster types by routing strength)",
        fontsize=12, fontweight="bold", y=1.08)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PER-TASK ALPHA BAR CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_task_alpha(mean_alpha_task: Dict[str, np.ndarray],
                    slot_alpha_by_label: Dict[str, Dict[int, np.ndarray]],
                    out_dir: Path, top_k: int = 32):
    """
    Differential slot alpha plots: ACR+ minus ACR− for every task.

    Per-task: signed bar chart sorted by |Δα|.
      Positive bar → slot more important for ACR+ patients.
      Negative bar → slot more important for ACR− patients.

    Comparison heatmap: tasks × top-K differentially-used slots (Δα values,
    diverging colourmap). Shows task-specific slot specialisation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in TASKS if t in mean_alpha_task]
    if not tasks:
        return

    # ── Per-task differential bar chart ──────────────────────────────────────
    for task in tasks:
        alpha_by_lab = slot_alpha_by_label.get(task, {})
        a0 = alpha_by_lab.get(0)   # Non-ACR mean alpha (K,)
        a1 = alpha_by_lab.get(1)   # ACR+    mean alpha (K,)

        if a0 is None or a1 is None:
            # Fallback: just show mean alpha sorted by value (last resort)
            alpha_mean = mean_alpha_task[task]
            top_idx = np.argsort(alpha_mean)[-top_k:][::-1]
            fig, ax = plt.subplots(figsize=(max(10, top_k * 0.5), 4))
            ax.bar(range(top_k), alpha_mean[top_idx],
                   color=TASK_COLORS.get(task, "#888"), alpha=0.85)
            ax.set_xticks(range(top_k))
            ax.set_xticklabels([f"S{i}" for i in top_idx],
                               rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Mean slot α", fontsize=10)
            ax.set_title(f"{TASK_LABELS.get(task, task)} — top-{top_k} by mean α",
                         fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            p = out_dir / f"alpha_task_{task}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  → {p}")
            continue

        diff = a1 - a0                                   # Δα = ACR+ − ACR−  (K,)
        top_idx = np.argsort(np.abs(diff))[-top_k:][::-1]  # top-K by |Δα|
        d_sub   = diff[top_idx]

        fig, ax = plt.subplots(figsize=(max(10, top_k * 0.5), 4))
        colors = ["#e05c5c" if v > 0 else "#5c9be0" for v in d_sub]
        ax.bar(range(top_k), d_sub, color=colors, alpha=0.85,
               edgecolor="#333", linewidth=0.4)
        ax.axhline(0, color="#666", linewidth=0.8, linestyle="--")

        ax.set_xticks(range(top_k))
        ax.set_xticklabels([f"S{i}" for i in top_idx],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Δα  (ACR+ − ACR−)", fontsize=10)
        ax.set_title(
            f"{TASK_LABELS.get(task, task)} — top-{top_k} differentially weighted slots\n"
            f"Red = higher in ACR+,  Blue = higher in ACR−",
            fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # annotate the 5 most extreme slots
        for j in range(min(5, top_k)):
            v = d_sub[j]
            ax.text(j, v + np.sign(v) * 0.00005,
                    f"S{top_idx[j]}", ha="center",
                    va="bottom" if v > 0 else "top", fontsize=6)

        fig.tight_layout()
        p = out_dir / f"alpha_task_{task}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {p}")

    # ── Multi-task differential heatmap (task × top-K slots) ─────────────────
    if len(tasks) < 2:
        return

    # Build Δα matrix: (T, K)
    diff_stack = []
    for task in tasks:
        a0 = slot_alpha_by_label.get(task, {}).get(0)
        a1 = slot_alpha_by_label.get(task, {}).get(1)
        if a0 is not None and a1 is not None:
            diff_stack.append(a1 - a0)
        else:
            diff_stack.append(mean_alpha_task[task])

    D = np.stack(diff_stack, axis=0)               # (T, K)
    # Select slots most variable in Δα across tasks
    slot_score = np.abs(D).max(0)                  # (K,) — max |Δα| across any task
    top_idx    = np.argsort(slot_score)[-top_k:][::-1]
    sub        = D[:, top_idx]                     # (T, top_k)

    vmax = max(np.abs(sub).max(), 1e-8)
    fig, ax = plt.subplots(figsize=(max(12, top_k * 0.55), max(3, len(tasks) * 0.8 + 1.5)))
    im = ax.imshow(sub, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=9)
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"S{i}" for i in top_idx],
                       rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Slot index", fontsize=10)
    ax.set_title(
        f"Task × slot  Δα = ACR+ − ACR−  (top-{top_k} slots by max |Δα|)\n"
        "Red = slot more active in ACR+,  Blue = more active in ACR−",
        fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.03, label="Δα")
    fig.tight_layout()
    p = out_dir / "alpha_task_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# SLOT × CLUSTER HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════

def plot_slot_cluster_heatmaps(
    cluster_slot_mats: Dict[str, np.ndarray],
    cluster_types_all: Dict[str, List[str]],
    mean_alpha_task: Dict[str, np.ndarray],
    out_dir: Path,
    top_k_slots: int = 32,
):
    """
    Per-modality (C × K) heatmap: mean slot routing weight per cluster type.
    Slots sorted by mean alpha across tasks. Only top-K slots shown.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not mean_alpha_task:
        return

    tasks = [t for t in TASKS if t in mean_alpha_task]
    alpha_stack = np.stack([mean_alpha_task[t] for t in tasks], axis=0)
    mean_alpha_all = alpha_stack.mean(0)                         # (K,)
    top_slot_idx = np.argsort(mean_alpha_all)[-top_k_slots:][::-1]

    for mod in CLUSTER_MODS:
        if mod not in cluster_slot_mats or mod not in cluster_types_all:
            continue

        mat    = cluster_slot_mats[mod]       # (C, K)
        ctypes = cluster_types_all[mod]
        C      = mat.shape[0]

        # Filter to top_k_slots most active slots
        mat_sub = mat[:, top_slot_idx]         # (C, top_k)

        # Sort cluster types by total routing strength
        row_sum = mat_sub.sum(1)
        sorted_ci = np.argsort(row_sum)[::-1]
        mat_sorted = mat_sub[sorted_ci]
        ctypes_sorted = [ctypes[i] for i in sorted_ci]

        fig_h = max(4, C * 0.22 + 2)
        fig, ax = plt.subplots(figsize=(max(14, top_k_slots * 0.5), fig_h))
        vmax = mat_sorted.max() + 1e-8
        im = ax.imshow(mat_sorted, aspect="auto", cmap="plasma",
                       vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(C))
        ax.set_yticklabels(ctypes_sorted, fontsize=7)
        ax.set_xticks(range(top_k_slots))
        ax.set_xticklabels([f"S{i}" for i in top_slot_idx],
                           rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Slot index (sorted by mean α)", fontsize=10)
        ax.set_ylabel(f"{mod} cluster type", fontsize=10)
        ax.set_title(f"{mod} — Cluster → Slot routing  "
                     f"(top-{top_k_slots} slots by mean task-α)",
                     fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.03, label="mean routing weight")
        fig.tight_layout()
        p = out_dir / f"slot_cluster_{mod}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# SLOT ALPHA × CLUSTER TYPE: which cluster types correlate with high-alpha slots?
# ══════════════════════════════════════════════════════════════════════════════

def plot_cluster_task_routing(
    cluster_slot_mats: Dict[str, np.ndarray],
    cluster_types_all: Dict[str, List[str]],
    mean_alpha_task: Dict[str, np.ndarray],
    out_dir: Path,
):
    """
    For each modality and task: effective cluster→task contribution.
    Computed as: (C, K) routing × (K,) alpha → (C,) cluster importance per task.

    Shows which cell/cluster types most influence each task, via slot routing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in TASKS if t in mean_alpha_task]
    if not tasks:
        return

    for mod in CLUSTER_MODS:
        if mod not in cluster_slot_mats or mod not in cluster_types_all:
            continue
        mat    = cluster_slot_mats[mod]   # (C, K)
        ctypes = cluster_types_all[mod]   # C names
        C      = mat.shape[0]

        # (C, T) effective contribution matrix
        contrib_CT = np.zeros((C, len(tasks)), dtype=np.float32)
        for ti, task in enumerate(tasks):
            alpha = mean_alpha_task[task]            # (K,)
            contrib_CT[:, ti] = mat @ alpha          # (C,)

        # Sort by total contribution
        total = contrib_CT.sum(1)
        sorted_ci = np.argsort(total)[::-1]
        mat_sorted   = contrib_CT[sorted_ci]
        ctype_sorted = [ctypes[i] for i in sorted_ci]

        fig_h = max(4, C * 0.25 + 2)
        fig, axes = plt.subplots(1, 2, figsize=(14, fig_h))

        # Panel 1: heatmap (C × T)
        ax = axes[0]
        vmax = mat_sorted.max() + 1e-8
        im = ax.imshow(mat_sorted, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(C)); ax.set_yticklabels(ctype_sorted, fontsize=7)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{mod} — Cluster → Task routing\n"
                     f"(routing × alpha per task)",
                     fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.05, label="effective contribution")

        # Panel 2: stacked bar chart
        ax2 = axes[1]
        x = np.arange(C)
        bottom = np.zeros(C)
        for ti, task in enumerate(tasks):
            vals = mat_sorted[:, ti]
            ax2.bar(x, vals, bottom=bottom, label=TASK_LABELS.get(task, task),
                    color=TASK_COLORS.get(task, "#888"), alpha=0.85)
            bottom += vals
        ax2.set_xticks(x)
        ax2.set_xticklabels(ctype_sorted, rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("Effective contribution (routing × α)", fontsize=9)
        ax2.set_title(f"{mod} — Per-cluster task contribution", fontsize=10)
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"{mod}: how do cluster types contribute to each task (via shared slots)?",
            fontsize=12, fontweight="bold")
        fig.tight_layout()
        p = out_dir / f"cluster_task_routing_{mod}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# 4-COLUMN SANKEY: clusters → slots → tasks → outcome labels
# ══════════════════════════════════════════════════════════════════════════════

def plot_sankey_full(
    mean_alpha_task:       Dict[str, np.ndarray],
    cluster_slot_mats_raw: Dict[str, np.ndarray],
    cluster_slot_mats_soft: Dict[str, np.ndarray],
    cluster_types_all:     Dict[str, List[str]],
    task_label_data:       Dict[str, Dict[int, Dict]],
    out_path:              Path,
    top_k_slots:  int = 20,
    top_k_clusters: int = 30,
):
    """
    4-column Sankey:  Instance clusters → Shared slots → Tasks → Outcome labels

    Columns
    -------
    0  Cluster types (top-C by total raw routing strength, colored by modality)
    1  Top shared slots (by mean alpha across tasks)
    2  Tasks (ACR-cls, ACR-TTE, CLAD-TTE, Death-TTE)
    3  Outcome labels  (ACR+/− for cls/surv; Q1-Q4 TTE bins for clad/death)

    Edges
    -----
    Cluster→Slot   raw pre-softmax QK scores (or softmax fallback)
    Slot→Task      per-task slot alpha (ABMIL weights)
    Task→Label     patient count per outcome bin
    """
    if not mean_alpha_task:
        return

    tasks = [t for t in TASKS if t in mean_alpha_task]
    if not tasks:
        return

    # Use raw scores when available, fall back to softmax
    cs_mats = cluster_slot_mats_raw if cluster_slot_mats_raw else cluster_slot_mats_soft
    if not cs_mats:
        return

    # ── Col-1: top slots by mean alpha ───────────────────────────────────────
    alpha_stack    = np.stack([mean_alpha_task[t] for t in tasks], axis=0)  # (T, K)
    mean_alpha_all = alpha_stack.mean(0)                                     # (K,)
    top_slot_idx   = np.argsort(mean_alpha_all)[-top_k_slots:][::-1]
    N_S = len(top_slot_idx)
    N_T = len(tasks)

    # ── Col-0: cluster nodes ─────────────────────────────────────────────────
    cluster_nodes: List[Tuple[str, str]] = []
    for mod in CLUSTER_MODS:
        if mod not in cluster_types_all:
            continue
        for ct in cluster_types_all[mod]:
            cluster_nodes.append((mod, ct))
    N_C = len(cluster_nodes)

    flow_CS = np.zeros((N_C, N_S), dtype=np.float32)
    c_off = 0
    for mod in CLUSTER_MODS:
        n_mod = len(cluster_types_all.get(mod, []))
        if mod in cs_mats:
            mat = cs_mats[mod]  # (C_mod, K)
            for ci in range(n_mod):
                flow_CS[c_off + ci] = mat[ci, top_slot_idx]
        c_off += n_mod

    # Prune to top clusters
    cluster_strength = flow_CS.sum(1)
    visible_ci = sorted(np.argsort(cluster_strength)[-top_k_clusters:][::-1].tolist())
    n_L = len(visible_ci)

    # ── Col-1→Col-2: slot→task ───────────────────────────────────────────────
    flow_ST = np.zeros((N_S, N_T), dtype=np.float32)
    for si, sk in enumerate(top_slot_idx):
        for ti, task in enumerate(tasks):
            flow_ST[si, ti] = mean_alpha_task[task][sk]
    flow_ST_norm = flow_ST / (flow_ST.sum(1, keepdims=True) + 1e-8)

    # ── Col-2→Col-3: task→label ──────────────────────────────────────────────
    label_nodes: List[Tuple[str, int]] = []
    for task in tasks:
        for li in range(TASK_N_LABELS.get(task, 2)):
            label_nodes.append((task, li))
    N_Label = len(label_nodes)

    flow_TL = np.zeros((N_T, N_Label), dtype=np.float32)
    for ti, task in enumerate(tasks):
        tld = task_label_data.get(task, {})
        for lni, (ltask, li) in enumerate(label_nodes):
            if ltask != task:
                continue
            d = tld.get(li)
            if d:
                flow_TL[ti, lni] = float(d["n"])

    # ── Node heights ─────────────────────────────────────────────────────────
    PAD = 0.010

    def _make_nodes(vals):
        vals = np.asarray(vals, dtype=np.float32)
        vals = np.maximum(vals, 1e-6)
        total = vals.sum()
        h = vals / total * (1 - PAD * len(vals))
        h = np.maximum(h, PAD * 0.3)
        y = np.cumsum(np.concatenate([[0.0], h + PAD]))[:-1]
        return h, y

    h_C, y_C = _make_nodes([cluster_strength[i] for i in visible_ci])
    h_S, y_S = _make_nodes(mean_alpha_all[top_slot_idx])
    h_T, y_T = _make_nodes(flow_TL.sum(1) + 1e-6)
    h_L, y_L = _make_nodes(flow_TL.sum(0) + 1e-6)

    # ── Figure ────────────────────────────────────────────────────────────────
    NODE_W = 0.030
    X0, X1, X2, X3 = 0.0, 0.26, 0.54, 0.82

    fig_h = max(13, max(n_L, N_Label) * 0.38 + 3)
    fig, ax = plt.subplots(figsize=(24, fig_h))
    ax.set_xlim(-0.08, 1.06);  ax.set_ylim(-0.03, 1.14)
    ax.axis("off")

    # ── Edges: Col0 → Col1 (cluster → slot) ──────────────────────────────────
    slot_fill_CS = y_S.copy()
    for rank, ci in enumerate(visible_ci):
        mod_ci, _ = cluster_nodes[ci]
        color = MOD_COLORS.get(mod_ci, "#888888")
        total_out = flow_CS[ci].sum() + 1e-8
        src_fill  = y_C[rank]
        for si in range(N_S):
            w = flow_CS[ci, si]
            if w < 1e-7:
                continue
            w_c = w / total_out * h_C[rank]
            w_s = w / (flow_CS[:, si].sum() + 1e-8) * h_S[si]
            _bezier_band(ax, X0 + NODE_W, X1, src_fill, src_fill + w_c,
                         slot_fill_CS[si], slot_fill_CS[si] + w_s,
                         color=color, alpha=0.38)
            slot_fill_CS[si] += w_s
            src_fill += w_c

    # ── Edges: Col1 → Col2 (slot → task) ─────────────────────────────────────
    task_fill_ST = y_T.copy()
    for si in range(N_S):
        c_s = plt.cm.plasma(si / max(N_S - 1, 1))
        src_fill = y_S[si]
        for ti in range(N_T):
            w_s = flow_ST_norm[si, ti] * h_S[si]
            w_t = flow_ST[si, ti] / (flow_ST[:, ti].sum() + 1e-8) * h_T[ti]
            if w_s < 1e-6:
                continue
            _bezier_band(ax, X1 + NODE_W, X2, src_fill, src_fill + w_s,
                         task_fill_ST[ti], task_fill_ST[ti] + w_t,
                         color=c_s, alpha=0.38)
            task_fill_ST[ti] += w_t
            src_fill += w_s

    # ── Edges: Col2 → Col3 (task → label) ────────────────────────────────────
    label_fill_TL = y_L.copy()
    for ti, task in enumerate(tasks):
        total_t = flow_TL[ti].sum() + 1e-8
        src_fill = y_T[ti]
        for lni, (ltask, li) in enumerate(label_nodes):
            if ltask != task:
                continue
            w = flow_TL[ti, lni]
            if w < 0.5:
                continue
            w_t = w / total_t * h_T[ti]
            w_l = w / (flow_TL[:, lni].sum() + 1e-8) * h_L[lni]
            lc = TASK_LABEL_COLORS.get(task, ["#888"])[li % len(TASK_LABEL_COLORS.get(task, ["#888"]))]
            _bezier_band(ax, X2 + NODE_W, X3, src_fill, src_fill + w_t,
                         label_fill_TL[lni], label_fill_TL[lni] + w_l,
                         color=lc, alpha=0.48)
            label_fill_TL[lni] += w_l
            src_fill += w_t

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for rank, ci in enumerate(visible_ci):
        mod_ci, name_ci = cluster_nodes[ci]
        color = MOD_COLORS.get(mod_ci, "#888888")
        ax.add_patch(mpatches.FancyBboxPatch(
            (X0, y_C[rank]), NODE_W, h_C[rank],
            boxstyle="square,pad=0", fc=color, ec="none", zorder=3))
        ax.text(X0 - 0.012, y_C[rank] + h_C[rank] / 2,
                name_ci, ha="right", va="center", fontsize=5.5, color="#111")

    for si, sk in enumerate(top_slot_idx):
        c_s = plt.cm.plasma(si / max(N_S - 1, 1))
        ax.add_patch(mpatches.FancyBboxPatch(
            (X1, y_S[si]), NODE_W, h_S[si],
            boxstyle="square,pad=0", fc=c_s, ec="none", zorder=3))
        ax.text(X1 + NODE_W / 2, y_S[si] + h_S[si] / 2,
                f"S{sk}", ha="center", va="center", fontsize=5.5,
                color="white", fontweight="bold")

    for ti, task in enumerate(tasks):
        color_t = TASK_COLORS.get(task, "#555")
        ax.add_patch(mpatches.FancyBboxPatch(
            (X2, y_T[ti]), NODE_W, h_T[ti],
            boxstyle="square,pad=0", fc=color_t, ec="none", zorder=3))
        ax.text(X2 + NODE_W / 2, y_T[ti] + h_T[ti] / 2,
                TASK_LABELS.get(task, task), ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

    for lni, (ltask, li) in enumerate(label_nodes):
        colors_t = TASK_LABEL_COLORS.get(ltask, ["#888"])
        lc = colors_t[li % len(colors_t)]
        ax.add_patch(mpatches.FancyBboxPatch(
            (X3, y_L[lni]), NODE_W, h_L[lni],
            boxstyle="square,pad=0", fc=lc, ec="white", linewidth=0.5, zorder=3))
        tld    = task_label_data.get(ltask, {})
        d      = tld.get(li)
        n_str  = f" (n={d['n']})" if d else ""
        lname  = TASK_LABEL_NAMES.get(ltask, {}).get(li, f"L{li}")
        ax.text(X3 + NODE_W + 0.012, y_L[lni] + h_L[lni] / 2,
                f"{TASK_LABELS.get(ltask, ltask)}: {lname}{n_str}",
                ha="left", va="center", fontsize=5.5, color="#111")

    # ── Headers ───────────────────────────────────────────────────────────────
    for xc, txt in [(X0 + NODE_W/2, "Instance\nClusters"),
                    (X1 + NODE_W/2, f"Shared Slots\n(top {N_S})"),
                    (X2 + NODE_W/2, "Tasks"),
                    (X3 + NODE_W/2, "Outcome\nLabels")]:
        ax.text(xc, 1.06, txt, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#222")

    # ── Modality legend ───────────────────────────────────────────────────────
    leg_patches = [mpatches.Patch(color=MOD_COLORS[m], label=m)
                   for m in CLUSTER_MODS if m in cluster_types_all]
    ax.legend(handles=leg_patches, loc="lower right", fontsize=8,
              framealpha=0.85, title="Modality")

    score_note = "raw QK scores" if cluster_slot_mats_raw else "softmax attn"
    ax.set_title(
        f"SharedSlotMIL Information Flow  "
        f"(cluster→slot: {score_note};  slot→task: ABMIL α;  task→label: patient counts)",
        fontsize=11, fontweight="bold", y=1.10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_shared_slot_model(
    results_dir: Path,
    split: int,
    fold: int,
    p2_tag: str,
    slot_k: int = 128,
) -> SharedSlotMIL:
    """Load SharedSlotMIL checkpoint."""
    fold_tag  = f"split{split}_fold{fold}"
    save_dir  = results_dir / "phase2" / fold_tag / f"slot_mega_{p2_tag}"
    ckpt_path = save_dir / "model_slot_final.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model_v8(variant="slot", slot_k=slot_k, task="mega")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:3]}")
    if missing:
        print(f"  [warn] missing keys: {missing[:3]}")
    model.to(DEVICE).eval()
    print(f"  Loaded SharedSlotMIL (K={slot_k}) from {ckpt_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Interpretability for SharedSlotMIL (v8)")
    p.add_argument("--split",       type=int, default=1)
    p.add_argument("--fold",        type=int, default=0)
    p.add_argument("--p2-tag",      default="shared_combined",
                   help="p2 tag used when training, e.g. 'shared' or 'shared_combined'")
    p.add_argument("--slot-k",      type=int, default=128)
    p.add_argument("--split-set",   default="test",
                   choices=["train", "val", "test", "all"])
    p.add_argument("--out-dir",     default=None,
                   help="Output directory (default: interpretability/slot_shared_s{split}f{fold})")
    p.add_argument("--samples-dir", default=SAMPLES_DIR)
    p.add_argument("--splits-csv",  default=SPLITS_CSV)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--top-k-slots", type=int, default=20)
    p.add_argument("--top-k-clust", type=int, default=30)
    p.add_argument("--top-k-alpha", type=int, default=32,
                   help="Top-K slots to show in alpha bar charts")
    p.add_argument("--skip-extract", action="store_true",
                   help="Skip extraction (use existing NPZ files)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(_ROOT) / "interpretability" /
        f"slot_shared_s{args.split}f{args.fold}_{args.p2_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_tag = f"split{args.split}_fold{args.fold}"
    print(f"\n{'='*65}")
    print(f"  SharedSlotMIL Interpretability  [{fold_tag}]  tag={args.p2_tag}")
    print(f"  split-set: {args.split_set}  →  {out_dir}")
    print(f"{'='*65}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    splits_dict = build_splits_multitask(args.samples_dir, args.splits_csv,
                                         args.fold, split=args.split)
    if args.split_set == "all":
        records = splits_dict["train"] + splits_dict["val"] + splits_dict["test"]
    else:
        records = splits_dict[args.split_set]
    print(f"  Records: {len(records)}")

    # ── Extract ───────────────────────────────────────────────────────────────
    if args.skip_extract:
        # Reload from NPZ
        npy_dir = out_dir / "npy"
        results = []
        for rec in records:
            npz_path = npy_dir / f"{rec['stem']}.npz"
            meta_path = npy_dir / f"{rec['stem']}_meta.json"
            if not npz_path.exists():
                continue
            r = dict(np.load(npz_path, allow_pickle=True))
            r = {k: v for k, v in r.items()}
            if meta_path.exists():
                meta = json.load(open(meta_path))
                r.update(meta)
            results.append(r)
        print(f"  Reloaded {len(results)} samples from {npy_dir}")
    else:
        model = load_shared_slot_model(
            results_dir, args.split, args.fold, args.p2_tag, args.slot_k)

        all_stems = list({r["stem"] for r in records})
        print(f"  Preloading bags ({len(all_stems)} stems) ...")
        bag_cache = preload_bags(all_stems, args.samples_dir)

        print(f"  Extracting ...")
        results = run_extraction(model, records, bag_cache, DEVICE,
                                 out_dir, args.samples_dir)
        del model, bag_cache; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  {len(results)} samples ready for visualization.\n")
    if not results:
        print("  No results — check checkpoint path and split-set.")
        return

    # ── Cohort aggregation ────────────────────────────────────────────────────
    print("  Building cohort matrices ...")
    (mean_alpha_task, cluster_slot_mats, cluster_slot_mats_raw,
     cluster_types_all, slot_alpha_by_label) = \
        build_cohort_matrices(results, n_slots=args.slot_k)

    print(f"  Tasks with alpha data : {list(mean_alpha_task.keys())}")
    print(f"  Modalities with soft routing: {list(cluster_slot_mats.keys())}")
    print(f"  Modalities with raw routing:  {list(cluster_slot_mats_raw.keys())}")

    print("  Building task→label data ...")
    task_label_data = build_task_label_data(results, n_slots=args.slot_k)
    for task, ld in task_label_data.items():
        counts = {k: d["n"] for k, d in ld.items()}
        print(f"    {task}: {counts}")

    # ── Visualizations ────────────────────────────────────────────────────────
    print("  Plotting per-task alpha bar charts ...")
    plot_task_alpha(mean_alpha_task, slot_alpha_by_label,
                    out_dir / "task_alpha", top_k=args.top_k_alpha)

    print("  Plotting slot×cluster heatmaps ...")
    plot_slot_cluster_heatmaps(cluster_slot_mats, cluster_types_all, mean_alpha_task,
                                out_dir / "slot_cluster", top_k_slots=args.top_k_slots)

    print("  Plotting cluster→task routing ...")
    plot_cluster_task_routing(cluster_slot_mats, cluster_types_all, mean_alpha_task,
                               out_dir / "cluster_task")

    print("  Plotting 3-column Sankey (clusters → slots → tasks) ...")
    plot_sankey(mean_alpha_task, cluster_slot_mats, cluster_types_all,
                out_dir / "sankey_cluster_slot_task.png",
                top_k_slots=args.top_k_slots, top_k_clusters=args.top_k_clust)

    print("  Plotting 4-column Sankey (clusters → slots → tasks → labels) ...")
    plot_sankey_full(
        mean_alpha_task,
        cluster_slot_mats_raw,
        cluster_slot_mats,
        cluster_types_all,
        task_label_data,
        out_dir / "sankey_full.png",
        top_k_slots=args.top_k_slots,
        top_k_clusters=args.top_k_clust,
    )

    # ── Save cohort summary ───────────────────────────────────────────────────
    summary = {
        "split": args.split, "fold": args.fold, "p2_tag": args.p2_tag,
        "split_set": args.split_set, "n": len(results),
        "n_pos": sum(1 for r in results if r.get("label") == 1),
        "n_neg": sum(1 for r in results if r.get("label") == 0),
        "tasks": list(mean_alpha_task.keys()),
        "mods_with_routing": list(cluster_slot_mats.keys()),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Done → {out_dir}\n")


if __name__ == "__main__":
    main()
