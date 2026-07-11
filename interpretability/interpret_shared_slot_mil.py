"""
interpret_shared_slot_mil.py  ·  Interpretability + Sankey plots for SharedSlotMIL (v8)
========================================================================================

Extracts per-patient interpretability data from a trained SharedSlotMIL model:

  slot_attn_{mod}  (K, N)  — MHA attention from each slot to each instance
  alpha_{task}     (K,)    — per-task ABMIL weights over the K shared slots
  prediction       float   — logit (acr_cls) or hazard (survival tasks)

Then builds two types of visualisations:

  1. Sankey plots (plotly):
       instance-cluster nodes  →  top-K slot nodes  →  task-outcome nodes
     One Sankey per task, plus a multi-task overview Sankey.

  2. Heatmaps (matplotlib):
       slot × cluster-type    (how much each cluster type loads each slot)
       slot × task alpha      (which slots drive each task, across patients)

Usage
-----
  python3 interpretability/interpret_shared_slot_mil.py \\
      --model-dir  results/mm_abmil_v8/phase2/split1_fold0/slot_mega_alt_shared \\
      --samples-dir /path/to/mil_v2/samples \\
      --splits-csv  /path/to/multimodal_splits_nested_cv.csv \\
      --split 1 --fold 0 --split-set test \\
      --out-dir results/mm_abmil_v8/interpretability/shared_slot \\
      --top-slots 32
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ── project path ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_SRC  = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mil.data.loader import preload_bags
from mil.data.splits import build_splits_multitask
from mil.data.registry import MODALITIES
from mil.models.builders import build_model_v8
from mil.models.encoders import MHASlotAttn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cluster label key mapping (mirrors interpret_mm_abmil.py)
MOD_TO_CLUSTER_KEY = {"HE": "HE_cells", "BAL": "BAL_cells", "CT": "CT_cells"}

# Dark-theme colour palette (matches patient_explorer)
BG    = "#0d1117"
BG2   = "#161b22"
EDGE  = "#30363d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

MOD_COLORS = {
    "HE":       "#4e79a7",
    "BAL":      "#f28e2b",
    "CT":       "#59a14f",
    "Clinical": "#e15759",
}
TASK_COLORS = {
    "acr_cls":  "#58a6ff",
    "acr_surv": "#3fb950",
    "clad":     "#d29922",
    "death":    "#f85149",
}
SLOT_CMAP = "plasma"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_dir: Path, slot_k: int = 128,
               max_he_patches: int = 99999) -> torch.nn.Module:
    """Load a trained SharedSlotMIL from model_slot_final.pt."""
    ckpt_path = model_dir / "model_slot_final.pt"
    if not ckpt_path.exists():
        # fall back to best_val checkpoint
        ckpt_path = model_dir / "best_val.pt"
    assert ckpt_path.exists(), f"No checkpoint found in {model_dir}"

    model = build_model_v8(
        variant="slot",
        slot_k=slot_k,
        max_he_patches=max_he_patches,
        modal_dropout=0.0,
        task="mega",
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE).eval()
    print(f"  Loaded {ckpt_path.name} → {type(model).__name__}  K={model.n_slots}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# FORWARD PASS WITH ATTENTION CAPTURE
# ══════════════════════════════════════════════════════════════════════════════

def _mha_slot_forward_capture(
    slot_mod: MHASlotAttn,
    h: torch.Tensor,
    shared_slots: torch.nn.Parameter,
) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    """
    Replicate MHASlotAttn.forward() and capture last-iter attention weights.

    Returns
    -------
    slots   (K, H) — final slot representations
    attn_KN (K, N) numpy array — mean-head attention from each slot to each instance
    """
    slots = shared_slots.clone()                         # (K, H)
    kv    = F.normalize(h, dim=-1).unsqueeze(0)         # (1, N, H)
    last_attn = None
    for _ in range(slot_mod.n_iters):
        q = slot_mod.norm_q(slots).unsqueeze(0)         # (1, K, H)
        # need_weights=True returns averaged attention over heads: (1, K, N)
        out, attn = slot_mod.mha(q, kv, kv,
                                  need_weights=True,
                                  average_attn_weights=True)
        last_attn = attn                                 # (1, K, N)
        slots = slots + out.squeeze(0)
        slots = slots + slot_mod.mlp(slots)
    attn_KN = (last_attn.squeeze(0).detach().cpu().numpy()
               if last_attn is not None else None)
    return slots, attn_KN


@torch.no_grad()
def extract_patient(
    model,
    bags: dict,
    device: torch.device,
) -> Optional[dict]:
    """
    Run SharedSlotMIL forward pass and capture interpretability tensors.

    Returns dict with:
      slot_attn_{mod}  (K, N_mod) numpy   — last-iter MHA attention per modality
      alpha_{task}     (K,) numpy          — softmax ABMIL slot weights per task
      pred_{task}      float               — raw model output (logit or hazard)
      mods_present     list[str]
    """
    he_coords = bags.get("HE_coords")
    mod_slots: Dict[str, torch.Tensor] = {}
    slot_attn_maps: Dict[str, np.ndarray] = {}

    for mod, enc in model.encoders.items():
        t = bags.get(mod)
        if t is None:
            continue
        t = t.to(device, non_blocking=True)
        crds = he_coords if mod == "HE" else None
        h = enc.encode_patches(t, coords=crds)                     # (N, H)
        slots, attn_KN = _mha_slot_forward_capture(
            model.slot_attns[mod], h, model.shared_slots
        )
        mod_slots[mod]        = slots                               # (K, H)
        slot_attn_maps[mod]   = attn_KN                            # (K, N)

    if not mod_slots:
        return None

    slots_agg = torch.stack(list(mod_slots.values()), dim=0).mean(0)  # (K, H)

    result: dict = {
        "mods_present": list(mod_slots.keys()),
    }
    for mod, attn in slot_attn_maps.items():
        result[f"slot_attn_{mod}"] = attn

    for task in model.task_names:
        gate  = model.abmil_V[task](slots_agg) * model.abmil_U[task](slots_agg)
        alpha = torch.softmax(model.abmil_w[task](gate), dim=0).squeeze(-1)  # (K,)
        rep   = model.norms[task]((alpha.unsqueeze(-1) * slots_agg).sum(0))
        pred  = model.heads[task](rep).squeeze()
        result[f"alpha_{task}"]  = alpha.cpu().numpy()
        result[f"pred_{task}"]   = pred.item()

    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTER LABEL LOADING  (mirrors interpret_mm_abmil.py)
# ══════════════════════════════════════════════════════════════════════════════

def _load_cluster_labels(stem: str, samples_dir: Path) -> Dict[str, List[str]]:
    pt = samples_dir / f"{stem}.pt"
    if not pt.exists():
        return {}
    try:
        data = torch.load(pt, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    raw = data.get("cluster_labels", {})
    out: Dict[str, List[str]] = {}
    for mod, key in MOD_TO_CLUSTER_KEY.items():
        labs = raw.get(key)
        if labs is not None:
            out[mod] = labs if isinstance(labs, list) else list(labs)
    # Clinical: use feature names as labels
    cfn = data.get("clinical_feature_names")
    if cfn is not None:
        out["Clinical"] = list(cfn)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# COHORT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def run_extraction(
    model,
    records: list,
    bag_cache: dict,
    samples_dir: Path,
    device: torch.device,
) -> List[dict]:
    results = []

    for i, rec in enumerate(records):
        stem  = rec["stem"]
        entry = bag_cache.get(stem, {})
        bags  = {m: entry.get(m) for m in MODALITIES}
        bags["HE_coords"] = entry.get("HE_coords")
        if all(v is None for m, v in bags.items() if m in MODALITIES):
            continue

        try:
            r = extract_patient(model, bags, device)
        except Exception as e:
            print(f"  [warn] {stem}: {e}")
            r = None

        if r is None:
            continue

        r["stem"]  = stem
        r["label"] = rec.get("label")

        # survival labels
        for ep in ("clad", "death"):
            r[f"{ep}_time"]  = rec.get(f"{ep}_time",  float("nan"))
            r[f"{ep}_event"] = rec.get(f"{ep}_event", float("nan"))
        r["tte_next_acr"]   = rec.get("tte_next_acr",   float("nan"))
        r["event_next_acr"] = rec.get("event_next_acr", 0)

        # cluster labels per modality (from .pt file)
        cl_map = _load_cluster_labels(stem, samples_dir)
        for mod, labs in cl_map.items():
            # truncate to N instances actually processed
            ref = r.get(f"slot_attn_{mod}")
            n   = ref.shape[1] if ref is not None else len(labs)
            r[f"cluster_labels_{mod}"] = labs[:n]

        results.append(r)
        gc.collect()

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(records)}]", flush=True)

    print(f"  Extracted {len(results)} patients.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# FLOW MATRIX COMPUTATION  (for Sankey)
# ══════════════════════════════════════════════════════════════════════════════

def _outcome_category(r: dict, task: str, n_quantiles: int = 4) -> Optional[str]:
    """Assign a patient to a task outcome category for right-side Sankey nodes."""
    if task == "acr_cls":
        lab = r.get("label")
        if lab is None:
            return None
        return "ACR+" if int(lab) == 1 else "ACR−"

    if task == "acr_surv":
        tte, ev = r.get("tte_next_acr", float("nan")), r.get("event_next_acr", 0)
    elif task == "clad":
        tte, ev = r.get("clad_time", float("nan")), r.get("clad_event", float("nan"))
    elif task == "death":
        tte, ev = r.get("death_time", float("nan")), r.get("death_event", float("nan"))
    else:
        return None

    try:
        tte = float(tte); ev = float(ev)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(tte):
        return None
    if ev == 0:
        return "Censored"
    # Use hazard-score quantile among event patients (assigned later in build_flows)
    return "__event__"  # placeholder replaced in build_flows


def build_flow_matrices(
    results: List[dict],
    task: str,
    K: int,
    n_quantiles: int = 4,
) -> Tuple[
    Dict[str, np.ndarray],   # cluster→slot: {mod_cluster: (K,)}
    Dict[str, np.ndarray],   # slot→outcome: {outcome: (K,)}
    List[str],               # cluster node labels
    List[str],               # outcome node labels
]:
    """
    Build aggregated flow matrices for a Sankey plot for one task.

    cluster→slot flow  : how strongly each cluster type routes to each slot
    slot→outcome flow  : how strongly each slot drives each outcome category
    """
    # ── 1. Cluster → slot ────────────────────────────────────────────────────
    # For each patient and modality: sum attn[k, n] for all n belonging to cluster c
    cluster_to_slot: Dict[str, np.ndarray] = {}  # "mod:cluster_type" -> (K,)
    cluster_count:   Dict[str, int] = {}

    for r in results:
        for mod in r.get("mods_present", []):
            attn = r.get(f"slot_attn_{mod}")   # (K, N)
            labs = r.get(f"cluster_labels_{mod}")
            if attn is None or labs is None:
                continue
            N = min(attn.shape[1], len(labs))
            for n in range(N):
                ct  = str(labs[n])
                key = f"{mod}: {ct}"
                vec = attn[:, n]               # (K,)
                if key not in cluster_to_slot:
                    cluster_to_slot[key] = np.zeros(K, dtype=np.float64)
                    cluster_count[key]   = 0
                cluster_to_slot[key] += vec
                cluster_count[key]   += 1

    # normalise to mean per instance
    for key in cluster_to_slot:
        n = cluster_count[key]
        if n > 0:
            cluster_to_slot[key] /= n

    # ── 2. Slot → outcome ────────────────────────────────────────────────────
    # Use alpha[task][k] as slot contribution; weight by I(patient in outcome category)
    # For survival: compute hazard quartiles among event patients first
    preds_event = []
    stems_event = []
    for r in results:
        alpha = r.get(f"alpha_{task}")
        if alpha is None:
            continue
        cat = _outcome_category(r, task, n_quantiles)
        if cat == "__event__":
            preds_event.append(r[f"pred_{task}"])
            stems_event.append(r["stem"])

    # build quantile thresholds
    q_labels: List[str] = []
    q_thresholds: List[float] = []
    if preds_event:
        preds_arr = np.array(preds_event)
        for qi in range(1, n_quantiles + 1):
            q_thresholds.append(float(np.percentile(preds_arr, qi * 100 / n_quantiles)))
            q_labels.append(f"Q{qi} (event)")
    stem_to_qlab = {}
    for stem, pred in zip(stems_event, preds_event):
        for qi, thresh in enumerate(q_thresholds):
            if pred <= thresh:
                stem_to_qlab[stem] = q_labels[qi]
                break
        else:
            stem_to_qlab[stem] = q_labels[-1]

    # accumulators
    outcome_to_alpha: Dict[str, List[np.ndarray]] = {}
    for r in results:
        alpha = r.get(f"alpha_{task}")
        if alpha is None:
            continue
        cat = _outcome_category(r, task, n_quantiles)
        if cat is None:
            continue
        if cat == "__event__":
            cat = stem_to_qlab.get(r["stem"], "event")
        outcome_to_alpha.setdefault(cat, []).append(alpha)

    slot_to_outcome: Dict[str, np.ndarray] = {}
    for cat, alphas in outcome_to_alpha.items():
        slot_to_outcome[cat] = np.stack(alphas).mean(0)   # (K,) mean alpha

    # outcome node labels: fixed order
    if task == "acr_cls":
        outcome_labels = ["ACR−", "ACR+"]
    else:
        ev_cats  = [f"Q{qi+1} (event)" for qi in range(n_quantiles)
                    if f"Q{qi+1} (event)" in slot_to_outcome]
        cens_cat = ["Censored"] if "Censored" in slot_to_outcome else []
        outcome_labels = ev_cats + cens_cat

    cluster_labels = sorted(cluster_to_slot.keys())

    return cluster_to_slot, slot_to_outcome, cluster_labels, outcome_labels


# ══════════════════════════════════════════════════════════════════════════════
# SANKEY PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_sankey_task(
    cluster_to_slot: Dict[str, np.ndarray],
    slot_to_outcome: Dict[str, np.ndarray],
    cluster_labels:  List[str],
    outcome_labels:  List[str],
    task: str,
    out_path: Path,
    top_slots: int = 32,
    min_flow: float = 1e-4,
) -> None:
    """
    Build a 3-column Sankey:
      Column 0: instance cluster nodes
      Column 1: slot nodes (top_slots by max alpha)
      Column 2: task outcome nodes
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  [warn] plotly not installed — skipping Sankey. pip install plotly")
        return

    K = next(iter(slot_to_outcome.values())).shape[0] if slot_to_outcome else 0
    if K == 0:
        return

    # pick top slots by max alpha across outcomes
    all_alpha = np.stack(list(slot_to_outcome.values()), axis=0)  # (n_outcomes, K)
    max_alpha_per_slot = all_alpha.max(axis=0)                     # (K,)
    top_slot_idx = np.argsort(max_alpha_per_slot)[::-1][:top_slots]
    top_slot_idx = sorted(top_slot_idx.tolist())

    # ── node lists ───────────────────────────────────────────────────────────
    # Col 0: clusters
    cluster_node_ids = {cl: i for i, cl in enumerate(cluster_labels)}
    n_clusters = len(cluster_labels)

    # Col 1: slots
    slot_node_start = n_clusters
    slot_node_ids   = {k: slot_node_start + j for j, k in enumerate(top_slot_idx)}
    n_slots_shown   = len(top_slot_idx)

    # Col 2: outcomes
    outcome_node_start = slot_node_start + n_slots_shown
    outcome_node_ids   = {oc: outcome_node_start + j for j, oc in enumerate(outcome_labels)}
    n_outcomes = len(outcome_labels)

    total_nodes = n_clusters + n_slots_shown + n_outcomes

    # ── colours ──────────────────────────────────────────────────────────────
    node_colors = []
    # cluster nodes: colour by modality prefix
    for cl in cluster_labels:
        mod = cl.split(":")[0].strip()
        node_colors.append(MOD_COLORS.get(mod, MUTED))
    # slot nodes: plasma gradient
    cmap = plt.get_cmap(SLOT_CMAP)
    for j, k in enumerate(top_slot_idx):
        rgba = cmap(j / max(n_slots_shown - 1, 1))
        node_colors.append(f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},0.85)")
    # outcome nodes: task colour
    tc = TASK_COLORS.get(task, "#8b949e")
    for oc in outcome_labels:
        if "ACR+" in oc or "Q1" in oc:
            node_colors.append(tc)
        else:
            node_colors.append(MUTED)

    # ── node labels ──────────────────────────────────────────────────────────
    node_labels = (list(cluster_labels)
                   + [f"S{k}" for k in top_slot_idx]
                   + list(outcome_labels))

    # ── links: cluster → slot ────────────────────────────────────────────────
    src, tgt, val, lnk_col = [], [], [], []

    for cl, flow_K in cluster_to_slot.items():
        cl_id = cluster_node_ids.get(cl)
        if cl_id is None:
            continue
        mod = cl.split(":")[0].strip()
        base_rgba = MOD_COLORS.get(mod, MUTED)
        for k in top_slot_idx:
            v = float(flow_K[k])
            if v < min_flow:
                continue
            src.append(cl_id)
            tgt.append(slot_node_ids[k])
            val.append(v)
            lnk_col.append(base_rgba.replace("#", "rgba(").replace(
                "rgba(", "rgba(") + ",0.35)")
            # re-build rgba string properly
            r, g, b = _hex_to_rgb(base_rgba)
            lnk_col[-1] = f"rgba({r},{g},{b},0.35)"

    # ── links: slot → outcome ─────────────────────────────────────────────────
    for oc, alpha_K in slot_to_outcome.items():
        oc_id = outcome_node_ids.get(oc)
        if oc_id is None:
            continue
        for j, k in enumerate(top_slot_idx):
            v = float(alpha_K[k])
            if v < min_flow:
                continue
            src.append(slot_node_ids[k])
            tgt.append(oc_id)
            val.append(v)
            cmap_rgba = cmap(j / max(n_slots_shown - 1, 1))
            lnk_col.append(f"rgba({int(cmap_rgba[0]*255)},"
                            f"{int(cmap_rgba[1]*255)},"
                            f"{int(cmap_rgba[2]*255)},0.4)")

    if not src:
        print(f"  [warn] No flows for task={task} — Sankey empty")
        return

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=12, thickness=18,
            label=node_labels,
            color=node_colors,
            line=dict(color=EDGE, width=0.5),
        ),
        link=dict(source=src, target=tgt, value=val, color=lnk_col),
        textfont=dict(color=TEXT, size=10),
        arrangement="snap",
    ))

    task_pretty = {"acr_cls": "ACR Classification",
                   "acr_surv": "ACR Time-to-Event",
                   "clad": "CLAD TTE",
                   "death": "Death TTE"}.get(task, task)
    fig.update_layout(
        title=dict(text=f"SharedSlotMIL · {task_pretty}<br>"
                        f"<sub>Instance clusters → Slots → Task outcomes</sub>",
                   font=dict(color=TEXT, size=14)),
        paper_bgcolor=BG,
        plot_bgcolor=BG2,
        font=dict(color=TEXT, size=10),
        margin=dict(l=20, r=20, t=80, b=20),
        height=max(600, n_clusters * 14 + n_outcomes * 30 + 120),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"  → {out_path}")


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 6:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 128, 128, 128


# ══════════════════════════════════════════════════════════════════════════════
# HEATMAP PLOTS (matplotlib)
# ══════════════════════════════════════════════════════════════════════════════

def plot_slot_cluster_heatmap(
    results: List[dict],
    out_dir: Path,
    top_slots: int = 32,
) -> None:
    """
    K×C heatmap: mean slot attention per (slot, cluster_type) pair,
    separately for each modality. Rows = top_slots, columns = cluster types.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for mod in list(MOD_TO_CLUSTER_KEY.keys()) + ["Clinical"]:
        # collect all (K, N) attn + labels across patients
        kc_rows_by_label: Dict[int, List[np.ndarray]] = {0: [], 1: []}
        all_ctypes: Optional[List[str]] = None
        K_ref = None

        for r in results:
            attn = r.get(f"slot_attn_{mod}")
            labs = r.get(f"cluster_labels_{mod}")
            if attn is None or labs is None:
                continue
            K_ref = attn.shape[0]
            N = min(attn.shape[1], len(labs))
            ctypes = sorted(set(str(l) for l in labs[:N]))
            if all_ctypes is None or len(ctypes) > len(all_ctypes):
                all_ctypes = ctypes

        if all_ctypes is None or K_ref is None:
            continue

        C = len(all_ctypes)
        ctype_idx = {ct: ci for ci, ct in enumerate(all_ctypes)}

        # mean attn per (slot, cluster_type) averaged over all instances of that type
        kc_sum   = {lab: np.zeros((K_ref, C)) for lab in (0, 1, -1)}
        kc_count = {lab: np.zeros((K_ref, C)) for lab in (0, 1, -1)}

        for r in results:
            attn = r.get(f"slot_attn_{mod}")
            labs = r.get(f"cluster_labels_{mod}")
            if attn is None or labs is None:
                continue
            label = int(r.get("label", -1)) if r.get("label") is not None else -1
            if label not in kc_sum:
                label = -1
            N = min(attn.shape[1], len(labs))
            for n in range(N):
                ci = ctype_idx.get(str(labs[n]))
                if ci is None:
                    continue
                kc_sum[label][:, ci]   += attn[:, n]
                kc_count[label][:, ci] += 1

        # average
        kc_mean: Dict[int, np.ndarray] = {}
        for lab in kc_sum:
            mask = kc_count[lab] > 0
            mat  = np.zeros_like(kc_sum[lab])
            mat[mask] = kc_sum[lab][mask] / kc_count[lab][mask]
            kc_mean[lab] = mat

        # select top_slots by max attention across all instances
        overall = kc_sum[-1].copy()
        cnt_all  = kc_count[-1].copy()
        cnt_mask = cnt_all > 0
        overall[cnt_mask] = overall[cnt_mask] / cnt_all[cnt_mask]
        slot_score = overall.max(axis=1)        # (K,)
        top_idx = np.argsort(slot_score)[::-1][:top_slots]
        top_idx = sorted(top_idx.tolist())

        panels = []
        if kc_mean[0].any(): panels.append((kc_mean[0], "Non-ACR (label=0)"))
        if kc_mean[1].any(): panels.append((kc_mean[1], "ACR (label=1)"))
        if kc_mean[0].any() and kc_mean[1].any():
            panels.append((kc_mean[1] - kc_mean[0], "ACR − Non-ACR"))
        if not panels:
            panels = [(kc_mean[-1], "All patients")]

        n_active = len(top_idx)
        fig, axes = plt.subplots(1, len(panels),
                                  figsize=(6 * len(panels), max(4, n_active * 0.35 + 2)),
                                  squeeze=False)

        for ax, (mat_full, ptitle) in zip(axes[0], panels):
            mat    = mat_full[top_idx, :]
            is_diff = "−" in ptitle
            vmax   = max(np.abs(mat).max(), 1e-8)
            cmap_n = "RdBu_r" if is_diff else "YlOrRd"
            vmin   = -vmax if is_diff else 0

            im = ax.imshow(mat, aspect="auto", cmap=cmap_n,
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks(range(C))
            ax.set_xticklabels(all_ctypes, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(n_active))
            ax.set_yticklabels([f"S{k}" for k in top_idx], fontsize=7)
            ax.set_xlabel("Cluster / Feature type", fontsize=9)
            ax.set_ylabel("Slot", fontsize=9)
            ax.set_title(ptitle, fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, label="mean attn")

        fig.suptitle(f"{mod} — Slot × Cluster attention  (top {n_active}/{K_ref} slots)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        p = out_dir / f"slot_cluster_{mod}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {p}")


def plot_task_alpha_heatmap(
    results: List[dict],
    tasks: List[str],
    out_dir: Path,
    top_slots: int = 32,
) -> None:
    """
    Heatmap: mean alpha per (task, slot), rows = tasks, columns = top slots.
    Separate panels for ACR+ vs ACR− patients.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    K_ref = None
    for r in results:
        alpha = r.get(f"alpha_{tasks[0]}")
        if alpha is not None:
            K_ref = alpha.shape[0]
            break
    if K_ref is None:
        return

    task_alpha: Dict[str, Dict[int, List[np.ndarray]]] = {
        t: {0: [], 1: [], -1: []} for t in tasks
    }
    for r in results:
        lab = int(r.get("label", -1)) if r.get("label") is not None else -1
        if lab not in (0, 1):
            lab = -1
        for t in tasks:
            alpha = r.get(f"alpha_{t}")
            if alpha is not None:
                task_alpha[t][lab].append(alpha)
                task_alpha[t][-1].append(alpha)

    def _mean_alpha(label: int) -> np.ndarray:
        rows = []
        for t in tasks:
            arr = task_alpha[t][label]
            rows.append(np.stack(arr).mean(0) if arr else np.zeros(K_ref))
        return np.stack(rows)   # (n_tasks, K)

    # top slots by max alpha across all tasks
    mat_all = _mean_alpha(-1)   # (n_tasks, K)
    slot_score = mat_all.max(0)  # (K,)
    top_idx = sorted(np.argsort(slot_score)[::-1][:top_slots].tolist())

    panels = []
    if any(task_alpha[tasks[0]][0]):
        panels.append((_mean_alpha(0), "Non-ACR"))
    if any(task_alpha[tasks[0]][1]):
        panels.append((_mean_alpha(1), "ACR+"))
    if not panels:
        panels = [(_mean_alpha(-1), "All")]

    fig, axes = plt.subplots(1, len(panels),
                              figsize=(0.4 * len(top_idx) * len(panels) + 3,
                                       max(3, len(tasks) * 0.7 + 2)),
                              squeeze=False)

    task_pretty = {"acr_cls": "ACR cls", "acr_surv": "ACR-TTE",
                   "clad": "CLAD-TTE", "death": "Death-TTE"}

    for ax, (mat_full, ptitle) in zip(axes[0], panels):
        mat  = mat_full[:, top_idx]        # (n_tasks, n_slots)
        vmax = max(mat.max(), 1e-8)
        im   = ax.imshow(mat, aspect="auto", cmap="Blues",
                         vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(len(top_idx)))
        ax.set_xticklabels([f"S{k}" for k in top_idx], rotation=90, fontsize=7)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([task_pretty.get(t, t) for t in tasks], fontsize=9)
        ax.set_xlabel("Slot", fontsize=9)
        ax.set_title(ptitle, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, label="mean α")

    fig.suptitle(f"Per-task ABMIL slot weights (top {len(top_idx)}/{K_ref} slots)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "task_alpha_heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def plot_slot_variance(results: List[dict], tasks: List[str], out_dir: Path) -> None:
    """Bar chart of cross-patient variance per slot for each task alpha."""
    out_dir.mkdir(parents=True, exist_ok=True)

    K_ref = None
    for r in results:
        a = r.get(f"alpha_{tasks[0]}")
        if a is not None:
            K_ref = a.shape[0]; break
    if K_ref is None:
        return

    fig, axes = plt.subplots(len(tasks), 1,
                              figsize=(max(12, K_ref * 0.12), 3 * len(tasks)),
                              squeeze=False)
    for ax, task in zip(axes[:, 0], tasks):
        alphas = [r[f"alpha_{task}"] for r in results if r.get(f"alpha_{task}") is not None]
        if not alphas:
            continue
        mat  = np.stack(alphas)          # (N_patients, K)
        var  = mat.var(axis=0)           # (K,)
        mean = mat.mean(axis=0)
        idx  = np.argsort(var)[::-1]
        ax.bar(range(K_ref), var[idx], color=TASK_COLORS.get(task, MUTED), alpha=0.85)
        ax.set_ylabel("Variance", fontsize=8)
        ax.set_title(f"{task} — per-slot alpha variance (ranked)", fontsize=9)
        ax.tick_params(labelsize=7)
        # annotate top-5
        for j in range(min(5, K_ref)):
            k = idx[j]
            ax.text(j, var[k] * 1.02, f"S{k}", ha="center", va="bottom",
                    fontsize=6, color="black")

    fig.suptitle("Slot specialisation: cross-patient alpha variance per task",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "slot_variance_by_task.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE / LOAD EXTRACTED DATA
# ══════════════════════════════════════════════════════════════════════════════

def save_results(results: List[dict], out_dir: Path) -> None:
    npy_dir = out_dir / "npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        stem   = r["stem"]
        np_dat = {k: v for k, v in r.items() if isinstance(v, np.ndarray)}
        meta   = {k: v for k, v in r.items()
                  if not isinstance(v, np.ndarray) and k != "cluster_labels"}
        if np_dat:
            np.savez_compressed(npy_dir / f"{stem}.npz", **np_dat)
        try:
            with open(npy_dir / f"{stem}_meta.json", "w") as f:
                json.dump(meta, f,
                          default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
        except Exception:
            pass
    print(f"  Saved {len(results)} .npz files → {npy_dir}")


def load_results(out_dir: Path) -> List[dict]:
    npy_dir = out_dir / "npy"
    results = []
    for meta_f in sorted(npy_dir.glob("*_meta.json")):
        stem = meta_f.name.replace("_meta.json", "")
        try:
            with open(meta_f) as f:
                r = json.load(f)
        except Exception:
            continue
        npz_f = npy_dir / f"{stem}.npz"
        if npz_f.exists():
            try:
                npz = np.load(npz_f, allow_pickle=False)
                r.update({k: npz[k] for k in npz.files})
            except Exception:
                pass
        # restore cluster_labels from meta
        for k, v in r.items():
            if isinstance(v, list) and k.startswith("cluster_labels_"):
                r[k] = v
        results.append(r)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Interpretability for SharedSlotMIL (v8)")
    parser.add_argument("--model-dir",   required=True, type=Path,
                        help="Directory containing model_slot_final.pt")
    parser.add_argument("--samples-dir", required=True, type=Path,
                        help="Directory with per-patient .pt files")
    parser.add_argument("--splits-csv",  required=True, type=Path)
    parser.add_argument("--split",       type=int, default=1)
    parser.add_argument("--fold",        type=int, default=0)
    parser.add_argument("--split-set",   default="test",
                        choices=["train", "val", "test"],
                        help="Which split to run inference on")
    parser.add_argument("--out-dir",     required=True, type=Path,
                        help="Output directory for interpretability results")
    parser.add_argument("--slot-k",      type=int, default=128)
    parser.add_argument("--top-slots",   type=int, default=32,
                        help="Number of top slots to show in Sankey / heatmaps")
    parser.add_argument("--no-extract",  action="store_true",
                        help="Skip extraction; load from existing --out-dir/npy/")
    parser.add_argument("--sankey-min-flow", type=float, default=1e-4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading splits …")
    splits = build_splits_multitask(
        samples_dir=args.samples_dir,
        splits_csv=args.splits_csv,
        fold=args.fold,
        split=args.split,
    )
    records = splits[args.split_set]
    print(f"  {args.split_set} set: {len(records)} records")

    if not args.no_extract:
        print("Loading bags into memory …")
        bag_cache = preload_bags(records, args.samples_dir)

        # ── load model ────────────────────────────────────────────────────────
        print("Loading model …")
        model = load_model(args.model_dir, slot_k=args.slot_k)

        # ── run extraction ────────────────────────────────────────────────────
        print(f"Extracting {args.split_set} set on {DEVICE} …")
        results = run_extraction(model, records, bag_cache, args.samples_dir, DEVICE)

        # free GPU memory
        del model, bag_cache
        torch.cuda.empty_cache()
        gc.collect()

        print("Saving extraction results …")
        save_results(results, args.out_dir)
    else:
        print("Loading pre-extracted results …")
        results = load_results(args.out_dir)
        print(f"  Loaded {len(results)} records from {args.out_dir}/npy/")

    if not results:
        print("No results to visualise."); return

    tasks = ["acr_cls", "acr_surv", "clad", "death"]
    K = results[0][f"alpha_{tasks[0]}"].shape[0] if results else args.slot_k

    # ── heatmaps ──────────────────────────────────────────────────────────────
    heatmap_dir = args.out_dir / "heatmaps"
    print("\nGenerating slot×cluster heatmaps …")
    plot_slot_cluster_heatmap(results, heatmap_dir, top_slots=args.top_slots)

    print("Generating task alpha heatmap …")
    plot_task_alpha_heatmap(results, tasks, heatmap_dir, top_slots=args.top_slots)

    print("Generating slot variance plot …")
    plot_slot_variance(results, tasks, heatmap_dir)

    # ── Sankey plots ──────────────────────────────────────────────────────────
    sankey_dir = args.out_dir / "sankey"
    print("\nGenerating Sankey plots …")
    for task in tasks:
        if not any(r.get(f"alpha_{task}") is not None for r in results):
            continue
        c2s, s2o, cl_labs, oc_labs = build_flow_matrices(results, task, K)
        if not c2s or not s2o:
            print(f"  [skip] {task}: no flow data")
            continue
        out_html = sankey_dir / f"sankey_{task}.html"
        plot_sankey_task(c2s, s2o, cl_labs, oc_labs, task, out_html,
                         top_slots=args.top_slots,
                         min_flow=args.sankey_min_flow)

    print(f"\nAll done → {args.out_dir}")


if __name__ == "__main__":
    main()
