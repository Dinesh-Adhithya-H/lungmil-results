"""
analyze_sankey_clean.py — Clean per-modality Sankey for SharedSlotMIL

Flow:  Cluster types  →  Shared slots  →  Tasks
       (raw QK score)    (joint relevance)  (ABMIL alpha)

One plot per modality.  Only positive raw attn edges, threshold weak ones.
Reports slot-structure diagnostics so you can see if the model has collapsed.

Usage
-----
  python3 interpretability/analyze_sankey_clean.py \\
      --split 1 --fold 0 \\
      --p2-tag alt_shared_combined \\
      --split-set test \\
      --npy-dir interpretability/slot_shared_s1f0_alt_shared_combined/npy \\
      --out-dir interpretability/slot_shared_s1f0_alt_shared_combined

SLURM: see submit_analyze_sankey.sh
"""

from __future__ import annotations
import argparse, gc, json, sys, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
import numpy as np
import torch
import torch.nn.functional as F
warnings.filterwarnings("ignore")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from mil.data.loader   import preload_bags
from mil.data.splits   import build_splits_multitask
from mil.data.registry import MODALITIES
from mil.models.builders import build_model_v8
from mil.models.phase2   import SharedSlotMIL

# ── constants ─────────────────────────────────────────────────────────────────
SAMPLES_DIR  = "/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV   = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
RESULTS_DIR  = Path("/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 42

TASKS       = ["acr_cls", "acr_surv", "clad", "death"]
TASK_LABELS = {"acr_cls": "ACR-cls", "acr_surv": "ACR-TTE",
               "clad": "CLAD-TTE", "death": "Death-TTE"}
TASK_COLORS = {"acr_cls": "#1565C0", "acr_surv": "#0277BD",
               "clad": "#FB8500",    "death": "#C62828"}
MOD_COLORS  = {"HE": "#E53935", "BAL": "#1E88E5",
               "CT": "#43A047", "Clinical": "#8E24AA"}
MOD_TO_CLUSTER_KEY = {"HE": "HE_cells", "BAL": "BAL_cells", "CT": "CT_cells"}
CLUSTER_MODS = ["HE", "BAL", "CT", "Clinical"]


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _load_model(results_dir, split, fold, p2_tag, slot_k=128):
    fold_tag  = f"split{split}_fold{fold}"
    ckpt_path = results_dir / "phase2" / fold_tag / f"slot_mega_{p2_tag}" / "model_slot_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Not found: {ckpt_path}")
    model = build_model_v8(variant="slot", slot_k=slot_k, task="mega")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    print(f"  Loaded {ckpt_path.parent.name}  K={slot_k}")
    return model


@torch.no_grad()
def _extract_one(model, bags, device):
    """
    Explicit inference through SharedSlotMIL with CompetitiveSlotAttn.

    Uses return_attn=True to get (K, N) competitive attention weights from the
    final routing iteration.  Raw pre-softmax scores are recomputed from the
    final slot state using the same to_q / to_k projections.

    Saved per patient
    -----------------
    h_{mod}             (N, H)  L2-normalised encoder features (input to slots)
    slot_attn_{mod}     (K, N)  competitive attn weights (renorm; each slot sums to 1)
    slot_attn_raw_{mod} (K, N)  pre-competitive-softmax QK scores (avg over heads)
    slot_rep_{mod}      (K, H)  slot representations after CompetitiveSlotAttn
    slots_agg           (K, H)  mean-aggregated slots across present modalities
    alpha_{task}        (K,)    per-task ABMIL importance weights
    """
    result: Dict[str, object] = {}
    h_store: Dict[str, torch.Tensor] = {}

    # ── Step 1: backbone features per modality ────────────────────────────────
    for mod, enc in model.encoders.items():
        t = bags.get(mod)
        if t is None:
            continue
        t = t.to(device, non_blocking=True)
        if mod == "HE" and t.shape[0] > model.max_he_patches:
            t = t[:model.max_he_patches]
        h = enc.encode_patches(t)   # (N, H)  L2-normed
        h_store[mod] = h

    if not h_store:
        return None
    result["mods_present"] = list(h_store.keys())

    # ── Step 2: CompetitiveSlotAttn with return_attn=True ────────────────────
    mod_slots: List[torch.Tensor] = []

    for mod, h in h_store.items():
        sa = model.slot_attns[mod]
        nh, dk = sa.n_heads, sa.d_k

        result[f"h_{mod}"] = h.cpu().numpy()   # (N, H)

        # Forward pass — returns final slot reps + (K, N) normalised attn
        slots, attn_w = sa(h, model.shared_slots, return_attn=True)
        result[f"slot_attn_{mod}"] = attn_w.cpu().numpy()   # (K, N)
        result[f"slot_rep_{mod}"]  = slots.cpu().numpy()    # (K, H)

        # Raw pre-softmax scores from the final slot state
        # (same computation as inside CompetitiveSlotAttn, final iteration)
        N = h.shape[0]
        h_norm = sa.norm_in(F.normalize(h, dim=-1))             # (N, H)
        k_feat = sa.to_k(h_norm)                                # (N, H)
        q_feat = sa.to_q(sa.norm_q(slots))                      # (K, H)
        K_h = k_feat.view(N, nh, dk).permute(1, 0, 2)          # (nh, N, dk)
        Q_h = q_feat.view(-1, nh, dk).permute(1, 0, 2)         # (nh, K, dk)
        raw = torch.bmm(Q_h, K_h.transpose(1, 2)) * sa.scale   # (nh, K, N)
        result[f"slot_attn_raw_{mod}"] = raw.mean(0).cpu().numpy()  # (K, N)

        mod_slots.append(slots)

    if not mod_slots:
        return None

    # ── Step 3: mean aggregate across present modalities ─────────────────────
    slots_agg = torch.stack(mod_slots, 0).mean(0)   # (K, H)
    result["slots_agg"] = slots_agg.cpu().numpy()

    # ── Step 4: per-task gated ABMIL → alpha ──────────────────────────────────
    for task in model.task_names:
        gate  = model.abmil_V[task](slots_agg) * model.abmil_U[task](slots_agg)
        alpha = torch.softmax(model.abmil_w[task](gate), dim=0)
        result[f"alpha_{task}"] = alpha.squeeze(1).cpu().numpy()   # (K,)

    return result


def _load_cluster_labels(stem, samples_dir):
    pt = Path(samples_dir) / f"{stem}.pt"
    if not pt.exists(): return {}
    try:
        data = torch.load(pt, map_location="cpu", weights_only=False)
    except Exception: return {}
    raw = data.get("cluster_labels", {})
    out = {}
    for mod, key in MOD_TO_CLUSTER_KEY.items():
        labs = raw.get(key)
        if labs is not None:
            out[mod] = labs if isinstance(labs, list) else list(labs)
    token_ids = data.get("clinical_token_ids"); vocab = data.get("clinical_vocab")
    if token_ids is not None and vocab is not None:
        id2l = {e["id"]: e["label"] for e in vocab}
        if isinstance(token_ids, torch.Tensor): token_ids = token_ids.tolist()
        out["Clinical"] = [id2l.get(int(tid), f"tok_{tid}") for tid in token_ids]
    return out


def run_extraction(model, records, bag_cache, npy_dir, samples_dir):
    npy_dir = Path(npy_dir); npy_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, rec in enumerate(records):
        stem  = rec["stem"]
        entry = bag_cache.get(stem, {})
        bags  = {m: entry.get(m) for m in MODALITIES}
        if all(v is None for v in bags.values()): continue
        try:
            r = _extract_one(model, bags, DEVICE)
        except Exception as e:
            print(f"  [warn] {stem}: {e}"); continue
        if r is None: continue
        r["stem"]  = stem
        r["label"] = rec.get("label")
        for ep in ("clad", "death"):
            r[f"{ep}_time"]  = rec.get(f"{ep}_time",  float("nan"))
            r[f"{ep}_event"] = rec.get(f"{ep}_event", float("nan"))
        cl = _load_cluster_labels(stem, samples_dir)
        for mod, labs in cl.items():
            # h_{mod} is (N, H)  → shape[0] = N
            # slot_attn_{mod} is (K, N) → shape[-1] = N  (fallback)
            ref_h = r.get(f"h_{mod}")
            ref_a = r.get(f"slot_attn_{mod}")
            if ref_h is not None:
                n = ref_h.shape[0]
            elif ref_a is not None:
                n = ref_a.shape[-1]
            else:
                n = len(labs)
            r[f"cluster_labels_{mod}"] = labs[:n]
        np_data = {k: v for k, v in r.items() if isinstance(v, np.ndarray)}
        meta    = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        if np_data: np.savez_compressed(npy_dir / f"{stem}.npz", **np_data)
        with open(npy_dir / f"{stem}_meta.json", "w") as f:
            json.dump(meta, f, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
        results.append(r)
        if (i + 1) % 20 == 0:
            print(f"  extracted {i+1}/{len(records)}", flush=True)
        gc.collect()
    print(f"  Saved {len(results)} → {npy_dir}")
    return results


def load_from_npy(records, npy_dir):
    npy_dir = Path(npy_dir)
    results = []
    for rec in records:
        stem = rec["stem"]
        npz  = npy_dir / f"{stem}.npz"
        meta = npy_dir / f"{stem}_meta.json"
        if not npz.exists(): continue
        r = dict(np.load(npz, allow_pickle=True))
        if meta.exists():
            r.update(json.load(open(meta)))
        results.append(r)
    print(f"  Loaded {len(results)} from {npy_dir}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def aggregate(results, n_slots=128):
    """
    Returns:
      alpha_mean   : {task: (K,)}  mean ABMIL alpha across all patients
      alpha_by_lab : {task: {0: (K,), 1: (K,)}}  per-ACR-label mean alpha
      raw_routing  : {mod: (C, K)}  mean raw attn per cluster type per slot
      cluster_types: {mod: [str]}
    """
    K = n_slots

    # Alpha
    a_sum  = {t: np.zeros(K) for t in TASKS}
    a_cnt  = {t: 0           for t in TASKS}
    a_lab  = {t: {0: np.zeros(K), 1: np.zeros(K)} for t in TASKS}
    a_lcnt = {t: {0: 0, 1: 0}                       for t in TASKS}

    # Raw routing  — collect per-patient (C,K) matrices then mean
    raw_lists  : Dict[str, List[np.ndarray]] = {m: [] for m in CLUSTER_MODS}
    ctype_union: Dict[str, List[str]]        = {m: [] for m in CLUSTER_MODS}

    for r in results:
        lab = r.get("label")
        for t in TASKS:
            a = r.get(f"alpha_{t}")
            if a is not None and len(a) == K:
                a_sum[t] += a; a_cnt[t] += 1
                if lab in (0, 1):
                    a_lab[t][lab] += a; a_lcnt[t][lab] += 1

        for mod in CLUSTER_MODS:
            raw = r.get(f"slot_attn_raw_{mod}")   # (K, N) pre-softmax
            cl  = r.get(f"cluster_labels_{mod}")  # [str] length N
            if raw is None or cl is None: continue
            n = min(raw.shape[1], len(cl))
            raw_n = raw[:, :n]; cl_n = cl[:n]
            ctypes = sorted(set(cl_n))
            # expand union
            for ct in ctypes:
                if ct not in ctype_union[mod]:
                    ctype_union[mod].append(ct)
            # (K, C_union) matrix for this patient
            C = len(ctype_union[mod]); idx = {c: i for i, c in enumerate(ctype_union[mod])}
            mat = np.zeros((K, C), np.float32)
            cnt = np.zeros(C, np.int32)
            for j, ct in enumerate(cl_n):
                ci = idx[ct]; mat[:, ci] += raw_n[:, j]; cnt[ci] += 1
            mask = cnt > 0; mat[:, mask] /= cnt[mask]
            raw_lists[mod].append((mat, list(ctype_union[mod])))

    alpha_mean   = {t: a_sum[t] / a_cnt[t] if a_cnt[t] else np.zeros(K) for t in TASKS}
    alpha_by_lab = {}
    for t in TASKS:
        alpha_by_lab[t] = {}
        for l in (0, 1):
            if a_lcnt[t][l]:
                alpha_by_lab[t][l] = a_lab[t][l] / a_lcnt[t][l]

    # Average raw routing  — need to align columns across patients
    raw_routing  : Dict[str, np.ndarray] = {}
    cluster_types: Dict[str, List[str]]  = {}
    for mod in CLUSTER_MODS:
        if not raw_lists[mod]: continue
        ctypes_all = sorted(ctype_union[mod])
        C = len(ctypes_all); idx_all = {c: i for i, c in enumerate(ctypes_all)}
        mats_aligned = []
        for mat_p, ctypes_p in raw_lists[mod]:
            # mat_p is (K, C_p); re-index to full C
            idx_p = {c: i for i, c in enumerate(ctypes_p)}
            mat_a = np.zeros((K, C), np.float32)
            for ct, ci_p in idx_p.items():
                ci_a = idx_all.get(ct)
                if ci_a is not None:
                    mat_a[:, ci_a] = mat_p[:, ci_p]
            mats_aligned.append(mat_a)
        mean_mat = np.stack(mats_aligned, 0).mean(0)  # (K, C)
        raw_routing[mod]  = mean_mat.T                 # (C, K)
        cluster_types[mod] = ctypes_all

    return alpha_mean, alpha_by_lab, raw_routing, cluster_types


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC REPORT
# ══════════════════════════════════════════════════════════════════════════════

def report_diagnostics(alpha_mean, alpha_by_lab, raw_routing, out_dir):
    """Print + save JSON report: slot-structure health check."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    K = next(iter(alpha_mean.values())).shape[0]
    report = {"K": K, "tasks": {}, "routing": {}}
    uniform_alpha = 1.0 / K
    print(f"\n{'='*60}")
    print(f"  Slot-structure diagnostics  (K={K}, uniform=1/{K}={uniform_alpha:.5f})")
    print(f"{'='*60}")

    for t, a in alpha_mean.items():
        s  = a.std()
        mx = a.max()
        top5 = np.argsort(a)[-5:][::-1].tolist()
        collapsed = s < uniform_alpha * 0.1   # std < 10% of uniform value → collapsed
        flag = "  *** COLLAPSED ***" if collapsed else ""
        print(f"  {t:12s}  std={s:.6f}  max={mx:.6f}  top5={top5}{flag}")
        report["tasks"][t] = {"std": float(s), "max": float(mx), "collapsed": bool(collapsed)}

        # Differential (ACR+ vs ACR-)
        a0 = alpha_by_lab.get(t, {}).get(0)
        a1 = alpha_by_lab.get(t, {}).get(1)
        if a0 is not None and a1 is not None:
            diff = a1 - a0
            top_diff = np.argsort(np.abs(diff))[-5:][::-1].tolist()
            print(f"  {'':12s}  max|Δα|={np.abs(diff).max():.6f}  top5_diff={top_diff}")

    print()
    for mod, mat in raw_routing.items():
        # mat: (C, K) mean raw attn per cluster per slot
        positive = np.clip(mat, 0, None)
        slot_score = positive.sum(0)   # (K,) total positive routing per slot
        cluster_score = positive.sum(1)  # (C,) total routing per cluster
        top_slots = np.argsort(slot_score)[-5:][::-1].tolist()
        print(f"  {mod:8s}  routing max={mat.max():.5f}  positive_sum/K={slot_score.mean():.5f}  "
              f"top_slots={top_slots}")
        report["routing"][mod] = {
            "max": float(mat.max()),
            "min": float(mat.min()),
            "mean_positive_per_slot": float(slot_score.mean()),
            "top_slots_by_routing": top_slots,
        }

    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  → {out_dir}/diagnostics.json\n")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# CLEAN PER-MODALITY SANKEY
# ══════════════════════════════════════════════════════════════════════════════

def _bezier_band(ax, x0, x1, y0b, y0t, y1b, y1t, color, alpha=0.5):
    cx = (x0 + x1) / 2
    verts = [(x0, y0b), (cx, y0b), (cx, y1b), (x1, y1b),
             (x1, y1t), (cx, y1t), (cx, y0t), (x0, y0t), (x0, y0b)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="none", alpha=alpha, zorder=1))


def plot_sankey_modality(
    mod: str,
    cluster_types: List[str],
    raw_routing: np.ndarray,        # (C, K) mean raw pre-softmax attn
    alpha_mean: Dict[str, np.ndarray],
    alpha_by_lab: Dict[str, Dict[int, np.ndarray]],
    out_path: Path,
    top_k_slots: int = 12,
    top_k_clusters: int = 15,
    edge_thresh_pct: float = 0.05,
):
    """
    3-column Sankey per modality:
      Cluster types  →  Shared slots  →  Tasks

    Cluster→Slot edges: positive raw QK scores only (pre-softmax).
    Slot→Task edges: ABMIL alpha.
    """
    tasks = [t for t in TASKS if t in alpha_mean]
    if not tasks or len(cluster_types) == 0:
        print(f"  [skip] {mod}: no data")
        return

    # Clip to positive (genuine attention signal)
    pos_routing = np.clip(raw_routing, 0, None)   # (C, K)

    # Select top slots: high raw routing from this modality × high mean alpha
    alpha_stack  = np.stack([alpha_mean[t] for t in tasks], 0).mean(0)  # (K,)
    slot_raw_sum = pos_routing.sum(0)                                     # (K,)
    # Joint score: routing + alpha (both normalized to [0,1])
    slot_score = (slot_raw_sum / (slot_raw_sum.max() + 1e-8) +
                  alpha_stack   / (alpha_stack.max()   + 1e-8))
    top_slot_idx = np.argsort(slot_score)[-top_k_slots:][::-1]
    N_S = len(top_slot_idx)
    N_T = len(tasks)

    # Select top clusters: high max raw attn to any top slot
    sub_routing = pos_routing[:, top_slot_idx]        # (C, K')
    cluster_score = sub_routing.max(1)                 # (C,) max over selected slots
    top_clust_idx = np.argsort(cluster_score)[-top_k_clusters:][::-1]
    # Keep sorted by original order so similar clusters are adjacent
    top_clust_idx = np.sort(top_clust_idx)
    N_C = len(top_clust_idx)

    flow_CS = sub_routing[top_clust_idx]              # (C', K')
    edge_thresh = flow_CS.max() * edge_thresh_pct
    flow_CS[flow_CS < edge_thresh] = 0.0

    # Slot→Task: alpha[slot, task]
    flow_ST = np.zeros((N_S, N_T), np.float32)
    for si, sk in enumerate(top_slot_idx):
        for ti, t in enumerate(tasks):
            flow_ST[si, ti] = alpha_mean[t][sk]

    # ── Node heights ──────────────────────────────────────────────────────────
    PAD = 0.012

    def _nodes(vals):
        vals = np.maximum(vals, 1e-8)
        h = vals / vals.sum() * (1 - PAD * len(vals))
        h = np.maximum(h, PAD * 0.4)
        y = np.cumsum(np.concatenate([[0], h + PAD]))[:-1]
        return h, y

    h_C, y_C = _nodes(flow_CS.sum(1))
    h_S, y_S = _nodes(slot_score[top_slot_idx])
    h_T, y_T = _nodes(flow_ST.sum(0) + 1e-8)

    # ── Figure ────────────────────────────────────────────────────────────────
    NODE_W = 0.04
    X0, X1, X2 = 0.0, 0.40, 0.76
    fig_h = max(10, max(N_C, N_T) * 0.5 + 3)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.set_xlim(-0.18, 1.06); ax.set_ylim(-0.04, 1.14); ax.axis("off")
    mod_color = MOD_COLORS.get(mod, "#888")

    # ── Cluster → Slot edges ──────────────────────────────────────────────────
    slot_fill = y_S.copy()
    for ci, c_orig in enumerate(top_clust_idx):
        total_out = flow_CS[ci].sum() + 1e-8
        src = y_C[ci]
        for si in range(N_S):
            w = flow_CS[ci, si]
            if w < 1e-8: continue
            w_c = w / total_out * h_C[ci]
            w_s = w / (flow_CS[:, si].sum() + 1e-8) * h_S[si]
            _bezier_band(ax, X0 + NODE_W, X1,
                         src, src + w_c,
                         slot_fill[si], slot_fill[si] + w_s,
                         color=mod_color, alpha=0.38)
            slot_fill[si] += w_s; src += w_c

    # ── Slot → Task edges ─────────────────────────────────────────────────────
    task_fill = y_T.copy()
    for si in range(N_S):
        c_s = plt.cm.plasma(si / max(N_S - 1, 1))
        src = y_S[si]
        row_sum = flow_ST[si].sum() + 1e-8
        for ti in range(N_T):
            w_s = flow_ST[si, ti] / row_sum * h_S[si]
            w_t = flow_ST[si, ti] / (flow_ST[:, ti].sum() + 1e-8) * h_T[ti]
            if w_s < 1e-6: continue
            _bezier_band(ax, X1 + NODE_W, X2,
                         src, src + w_s,
                         task_fill[ti], task_fill[ti] + w_t,
                         color=c_s, alpha=0.42)
            task_fill[ti] += w_t; src += w_s

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for ci, c_orig in enumerate(top_clust_idx):
        cname = cluster_types[c_orig]
        ax.add_patch(mpatches.FancyBboxPatch(
            (X0, y_C[ci]), NODE_W, h_C[ci],
            boxstyle="square,pad=0", fc=mod_color, ec="none", zorder=3))
        ax.text(X0 - 0.012, y_C[ci] + h_C[ci] / 2, cname,
                ha="right", va="center", fontsize=7, color="#111")

    for si, sk in enumerate(top_slot_idx):
        c_s = plt.cm.plasma(si / max(N_S - 1, 1))
        ax.add_patch(mpatches.FancyBboxPatch(
            (X1, y_S[si]), NODE_W, h_S[si],
            boxstyle="square,pad=0", fc=c_s, ec="none", zorder=3))
        ax.text(X1 + NODE_W / 2, y_S[si] + h_S[si] / 2,
                f"S{sk}", ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold")

    for ti, t in enumerate(tasks):
        ax.add_patch(mpatches.FancyBboxPatch(
            (X2, y_T[ti]), NODE_W, h_T[ti],
            boxstyle="square,pad=0", fc=TASK_COLORS.get(t, "#555"), ec="none", zorder=3))
        ax.text(X2 + NODE_W + 0.012, y_T[ti] + h_T[ti] / 2,
                TASK_LABELS.get(t, t), ha="left", va="center",
                fontsize=9, color=TASK_COLORS.get(t, "#555"), fontweight="bold")

    # Column headers
    for xc, lbl in [(X0 + NODE_W/2, f"{mod}\ncluster types"),
                    (X1 + NODE_W/2, f"Shared slots\n(top {N_S})"),
                    (X2 + NODE_W/2, "Tasks")]:
        ax.text(xc, 1.06, lbl, ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    # ACR differential annotation on slots
    a0 = alpha_by_lab.get("acr_cls", {}).get(0)
    a1 = alpha_by_lab.get("acr_cls", {}).get(1)
    if a0 is not None and a1 is not None:
        diff = a1 - a0
        for si, sk in enumerate(top_slot_idx):
            d = diff[sk]
            clr = "#c62828" if d > 0 else "#1565c0"
            ax.text(X1 + NODE_W + 0.004, y_S[si] + h_S[si] / 2,
                    f"{'+'if d>0 else ''}{d:.4f}",
                    ha="left", va="center", fontsize=5.5, color=clr)

    ax.set_title(
        f"{mod}  —  cluster types → shared slots → tasks\n"
        f"(edge weight = mean raw pre-softmax attn;  slot label = slot index;  "
        f"Δα annotation = ACR+ − ACR−)",
        fontsize=10, fontweight="bold", y=1.10)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# DIFFERENTIAL ALPHA HEATMAP  (ACR+ vs ACR- per task, per slot)
# ══════════════════════════════════════════════════════════════════════════════

def plot_differential_alpha(alpha_by_lab, alpha_mean, out_dir, top_k=32):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in TASKS if t in alpha_mean]
    diff_rows = []
    for t in tasks:
        a0 = alpha_by_lab.get(t, {}).get(0)
        a1 = alpha_by_lab.get(t, {}).get(1)
        if a0 is not None and a1 is not None:
            diff_rows.append(a1 - a0)
        else:
            diff_rows.append(np.zeros_like(alpha_mean[t]))

    D = np.stack(diff_rows, 0)                        # (T, K)
    top_idx = np.argsort(np.abs(D).max(0))[-top_k:][::-1]
    sub     = D[:, top_idx]
    vmax    = max(np.abs(sub).max(), 1e-8)

    fig, ax = plt.subplots(figsize=(max(14, top_k * 0.55), len(tasks) * 0.9 + 2))
    im = ax.imshow(sub, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=9)
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"S{i}" for i in top_idx], rotation=45, ha="right", fontsize=7)
    ax.set_title(f"Δα = ACR+ − ACR−  per task  (top-{top_k} slots by max |Δα|)\n"
                 "Red = slot more active in ACR+,  Blue = more active in ACR−",
                 fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.03, label="Δα")
    fig.tight_layout()
    p = out_dir / "diff_alpha_heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  → {p}")

    for t, diff in zip(tasks, diff_rows):
        top_idx_t = np.argsort(np.abs(diff))[-top_k:][::-1]
        d_sub = diff[top_idx_t]
        fig, ax = plt.subplots(figsize=(max(10, top_k * 0.45), 4))
        colors = ["#c62828" if v > 0 else "#1565c0" for v in d_sub]
        ax.bar(range(top_k), d_sub, color=colors, alpha=0.85, edgecolor="#333", linewidth=0.4)
        ax.axhline(0, color="#666", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(top_k))
        ax.set_xticklabels([f"S{i}" for i in top_idx_t], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Δα  (ACR+ − ACR−)", fontsize=10)
        ax.set_title(f"{TASK_LABELS.get(t, t)} — top-{top_k} differentially weighted slots\n"
                     "Red = more active in ACR+,  Blue = more active in ACR−",
                     fontsize=10, fontweight="bold")
        for j in range(min(5, top_k)):
            v = d_sub[j]
            ax.text(j, v + np.sign(v) * 1e-5, f"S{top_idx_t[j]}",
                    ha="center", va="bottom" if v > 0 else "top", fontsize=6)
        ax.grid(axis="y", alpha=0.3); fig.tight_layout()
        p = out_dir / f"diff_alpha_{t}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",      type=int,  default=1)
    p.add_argument("--fold",       type=int,  default=0)
    p.add_argument("--p2-tag",     default="alt_shared_combined")
    p.add_argument("--slot-k",     type=int,  default=128)
    p.add_argument("--split-set",  default="test",
                   choices=["train", "val", "test", "all"])
    p.add_argument("--npy-dir",    default=None,
                   help="Where NPZ files live (auto-derived if omitted)")
    p.add_argument("--out-dir",    default=None)
    p.add_argument("--samples-dir", default=SAMPLES_DIR)
    p.add_argument("--splits-csv",  default=SPLITS_CSV)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--top-k-slots",    type=int, default=12)
    p.add_argument("--top-k-clusters", type=int, default=15)
    p.add_argument("--skip-extract",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    results_dir = Path(args.results_dir)

    base = Path(_ROOT) / "interpretability" / f"slot_shared_s{args.split}f{args.fold}_{args.p2_tag}"
    npy_dir = Path(args.npy_dir) if args.npy_dir else base / "npy"
    out_dir = Path(args.out_dir) if args.out_dir else base
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  analyze_sankey_clean  tag={args.p2_tag}  set={args.split_set}")
    print(f"  npy : {npy_dir}")
    print(f"  out : {out_dir}")
    print(f"{'='*60}\n")

    splits_dict = build_splits_multitask(args.samples_dir, args.splits_csv,
                                         args.fold, split=args.split)
    if args.split_set == "all":
        records = splits_dict["train"] + splits_dict["val"] + splits_dict["test"]
    else:
        records = splits_dict[args.split_set]
    print(f"  Records: {len(records)}")

    # ── Extract or reload ─────────────────────────────────────────────────────
    if args.skip_extract:
        results = load_from_npy(records, npy_dir)
    else:
        model = _load_model(results_dir, args.split, args.fold, args.p2_tag, args.slot_k)
        all_stems = list({r["stem"] for r in records})
        print(f"  Preloading {len(all_stems)} bags ...")
        bag_cache = preload_bags(all_stems, args.samples_dir)
        results = run_extraction(model, records, bag_cache, npy_dir, args.samples_dir)
        del model, bag_cache; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if not results:
        print("  No results — check tag and split-set."); return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print("  Aggregating ...")
    alpha_mean, alpha_by_lab, raw_routing, cluster_types = \
        aggregate(results, n_slots=args.slot_k)
    print(f"  Modalities with routing: {list(raw_routing.keys())}")
    print(f"  Cluster type counts: { {m: len(v) for m,v in cluster_types.items()} }")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    report_diagnostics(alpha_mean, alpha_by_lab, raw_routing, out_dir)

    # ── Differential alpha plots ──────────────────────────────────────────────
    print("  Differential alpha plots ...")
    plot_differential_alpha(alpha_by_lab, alpha_mean, out_dir / "diff_alpha", top_k=32)

    # ── Per-modality Sankeys ──────────────────────────────────────────────────
    san_dir = out_dir / "sankey"
    san_dir.mkdir(exist_ok=True)
    for mod in CLUSTER_MODS:
        if mod not in raw_routing:
            print(f"  [skip] {mod}: no raw routing data"); continue
        print(f"  Sankey {mod} ...")
        plot_sankey_modality(
            mod           = mod,
            cluster_types = cluster_types[mod],
            raw_routing   = raw_routing[mod],
            alpha_mean    = alpha_mean,
            alpha_by_lab  = alpha_by_lab,
            out_path      = san_dir / f"sankey_{mod}.png",
            top_k_slots   = args.top_k_slots,
            top_k_clusters= args.top_k_clusters,
        )

    print(f"\n  Done → {out_dir}\n")


if __name__ == "__main__":
    main()
