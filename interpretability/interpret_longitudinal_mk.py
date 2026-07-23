"""
Longitudinal-MK-MT Interpretability — 6-panel publication-quality visualisation.

Model: longitudinal_mk_mt_mega  (PMA → TemporalSAB + ALiBi → recency ABMIL)

Panels
------
L_global  ALiBi head slopes, per-task recency γ, temporal decay curves  [model-level]
L1        Seed concept timeline        per patient: biopsy × modality × seed heat
L2        TemporalSAB cross-biopsy attention  per patient: causal attn heatmap (per task)
L3        Recency ABMIL α per biopsy   per patient: which biopsy dominates each task
L4        Hazard trajectory             per patient: risk evolves as biopsies accumulate

Usage (sbatch only — never run Python on the login node):
  sbatch interpretability/submit_interp_longitudinal.sh [--split 0] [--fold 0] [--n-patients 30]
"""

import argparse, json, math, os, sys, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # interpretability/
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mil.models.builders import build_model_v8
from mil.data.registry import MODALITIES, _pres_col
from mil.data.splits import build_splits_longitudinal

# ── Shared constants, colours, and utilities ──────────────────────────────────
from shared import (
    SPLITS_CSV, SAMPLES_DIR, RESULTS_ROOT, HE_CLUSTER_MAP,
    MOD_ORDER, MOD_COLORS, TASK_COLORS, TASK_LABELS,
    HE_BIO_MAP, HE_BIO_COLORS, bio_label as _bio_label,
    savefig as _savefig,
    PDF_DPI, PNG_DPI,
)

OUT_ROOT = ROOT / "interpretability" / "longitudinal_mk_interp"

# ── Typography (longitudinal-specific rcParams) ────────────────────────────────
FONT_BASE  = 11
FONT_TITLE = 13
FONT_LABEL = 11
FONT_TICK  = 9

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         FONT_BASE,
    "axes.titlesize":    FONT_TITLE,
    "axes.labelsize":    FONT_LABEL,
    "xtick.labelsize":   FONT_TICK,
    "ytick.labelsize":   FONT_TICK,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.6,
    "figure.dpi":        120,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
})

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(split: int, fold: int, device: torch.device, task: str = "mega"):
    vtag = "longitudinal_mk_mt"
    # Map interpretability task names → training checkpoint dir suffixes
    _task_to_dir = {
        "mega":     "mega",
        "acr_cls":  "cls",       # training uses --task cls
        "cls":      "cls",
        "acr_surv": "acr_surv",
        "clad":     "clad_surv",
        "clad_surv":"clad_surv",
        "death":    "death_surv",
        "death_surv":"death_surv",
    }
    if task == "mega":
        ckpt_dir   = f"{vtag}_mega"
        build_task = "mega"
        tasks      = ["acr_cls", "acr_surv", "clad", "death"]
    else:
        dir_suffix = _task_to_dir.get(task, task)
        ckpt_dir   = f"{vtag}_{dir_suffix}"
        # build_task must match what the model was trained with
        build_task = "cls" if task in ("acr_cls", "cls") else dir_suffix
        tasks      = [task]
    ckpt = RESULTS_ROOT / f"split{split}_fold{fold}" / ckpt_dir / f"model_{vtag}_final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    state = torch.load(ckpt, map_location="cpu")

    # ── Infer n_heads and n_seeds from checkpoint ─────────────────────────────
    slopes_shape = state.get("temporal_sab.alibi_slopes", torch.ones(1)).shape
    n_heads_ckpt = slopes_shape[0]   # e.g. 4

    seeds_shape = None
    for k, v in state.items():
        if k.endswith(".seeds"):
            seeds_shape = v.shape
            break
    # PMA seeds may have been saved with an extra batch dim: (1, K, H) → squeeze to (K, H)
    for k in list(state.keys()):
        if k.endswith(".seeds") and state[k].ndim == 3:
            state[k] = state[k].squeeze(0)

    # Infer slot_k from the (now fixed) seed tensor
    n_seeds_ckpt = 16
    for k, v in state.items():
        if k.endswith(".seeds") and v.ndim == 2:
            n_seeds_ckpt = v.shape[0]
            break

    # Build model with checkpoint's n_heads (overrides P2_N_HEADS=1 default)
    # We inject n_heads by temporarily patching after build
    model = build_model_v8(variant=vtag, task=build_task, slot_k=n_seeds_ckpt, n_cross_layers=1)

    # If n_heads in model ≠ checkpoint, rebuild TemporalSAB in place
    if model.temporal_sab.n_heads != n_heads_ckpt:
        from mil.models.encoders import TemporalSAB
        n_sab_layers = len(model.temporal_sab.layers)
        model.temporal_sab = TemporalSAB(
            hidden_dim=256, n_heads=n_heads_ckpt,
            dropout=0.1, n_layers=n_sab_layers)
        print(f"  [load_model] rebuilt TemporalSAB with n_heads={n_heads_ckpt}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [load_model] missing keys: {missing[:5]}")
    if unexpected:
        print(f"  [load_model] unexpected keys: {unexpected[:5]}")

    model.eval()
    print(f"[load_model] Loaded {ckpt.name}  n_heads={n_heads_ckpt}  n_seeds={n_seeds_ckpt}")
    return model.to(device), tasks


# ── Data loading ──────────────────────────────────────────────────────────────

def load_patient_bags(patient: dict, device: torch.device):
    """Load .pt files for each biopsy in the patient timeline.

    Returns (bags_list, transplant_days) where transplant_days is a list of
    float days-from-transplant for each biopsy (None if transplant_date missing).
    """
    import datetime
    bags_list       = []
    transplant_days = []
    transplant_date = None   # resolved from first available .pt

    for stem in patient["stems"]:
        pt_path = Path(SAMPLES_DIR) / f"{stem}.pt"
        if not pt_path.exists():
            bags_list.append({mod: None for mod in MOD_ORDER})
            transplant_days.append(None)
            continue
        data = torch.load(pt_path, map_location="cpu")

        # Resolve transplant_date once
        if transplant_date is None:
            tx_raw = data.get("transplant_date")
            if tx_raw is not None:
                try:
                    transplant_date = pd.Timestamp(str(tx_raw))
                except Exception:
                    pass

        # Anchor time for this biopsy → days from transplant
        anc_raw = data.get("anchor_time")
        if anc_raw is not None and transplant_date is not None:
            try:
                anc_dt = pd.Timestamp(str(anc_raw))
                transplant_days.append(float((anc_dt - transplant_date).days))
            except Exception:
                transplant_days.append(None)
        else:
            transplant_days.append(None)

        inp = data.get("inputs", data)
        bags = {}
        for mod in MOD_ORDER:
            key = f"{mod}_cells"
            feat = inp.get(key)
            if feat is not None and feat.numel() > 0:
                bags[mod] = feat.float()
            else:
                bags[mod] = None
        bags_list.append(bags)

    return bags_list, transplant_days


def load_patient_cluster_names(patient: dict) -> Dict[str, List[str]]:
    """Load cluster_names per modality from the first available biopsy's .pt file."""
    cluster_names: Dict[str, List[str]] = {}
    for stem in patient["stems"]:
        pt_path = Path(SAMPLES_DIR) / f"{stem}.pt"
        if not pt_path.exists():
            continue
        data = torch.load(pt_path, map_location="cpu")
        names_dict = data.get("cluster_names", {})
        for mod, names in names_dict.items():
            if mod not in cluster_names and names:
                cluster_names[mod] = list(names)
        if len(cluster_names) == len(MOD_ORDER):
            break
    return cluster_names


# ── TemporalSAB patcher ───────────────────────────────────────────────────────

def _patch_temporal_sab(temporal_sab: nn.Module, attn_store: dict):
    """
    Replace TemporalSAB.forward to also return per-head attention weights.
    Stores: attn_store['per_layer'] = list of (n_heads, N, N) per SAB layer
            attn_store['alibi_bias'] = (n_heads, N, N) ALiBi component only
    """
    n_heads = temporal_sab.n_heads
    orig_forward = temporal_sab.forward

    def patched_forward(x: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        delta      = days.unsqueeze(1) - days.unsqueeze(0)
        days_range = (days.max() - days.min() + 1.0).clamp(min=1.0)
        dist       = delta.abs() / days_range
        slopes     = temporal_sab.alibi_slopes.abs()
        alibi      = -slopes.view(-1, 1, 1) * dist.unsqueeze(0)   # (n_heads, N, N)
        causal     = (delta < 0).to(x.dtype) * -1e9
        bias       = (alibi + causal.unsqueeze(0)).to(x.dtype)

        attn_store["alibi_bias"] = alibi.detach().cpu().numpy()    # (n_heads, N, N)
        per_layer = []
        x_b = x.unsqueeze(0)
        for L in temporal_sab.layers:
            a, w = L["attn"](x_b, x_b, x_b,
                              attn_mask=bias.view(n_heads, N, N),
                              need_weights=True, average_attn_weights=False)
            per_layer.append(w.squeeze(0).detach().cpu().numpy())   # (n_heads, N, N)
            x_b = L["ffn"](L["norm"](x_b + a))
        attn_store["per_layer"] = per_layer
        return x_b.squeeze(0)

    temporal_sab.forward = patched_forward
    return orig_forward   # caller must restore


# ── Main extraction ───────────────────────────────────────────────────────────

@torch.no_grad()
def extract_patient_longitudinal(
        model: nn.Module,
        patient: dict,
        bags_list: List[Dict[str, Optional[torch.Tensor]]],
        device: torch.device,
        tasks: List[str],
) -> Optional[dict]:
    """
    Run a single patient through the longitudinal model and capture all internals.

    Returns dict with:
      seeds_pre_gate   : (total_tokens, H) — PMA seeds + modal embed, before task gate
      pma_attn         : {(t_idx, mod): (K, N)} — PMA b-cos attention weights
      pma_logits       : {(t_idx, mod): (K, N)} — PMA raw dot-product logits
      tok_biopsy       : list[int] — biopsy index per token
      tok_mod_name     : list[str] — modality name per token
      biopsy_days      : list[float] — days per biopsy (len T)
      gate_mat         : {task: (T, n_mods) np.ndarray}
      temporal_attn    : {task: {'per_layer': [...], 'alibi_bias': ndarray}}
      alpha_per_task   : {task: (N,) np.ndarray} — final ABMIL α weights
      hazard_traj      : {task: list[float]} — per-biopsy hazard (causal accumulation)
      pma_seeds_norms  : (T, n_mods, K) np.ndarray — L2 norm of each PMA seed
      logits           : {task: float}
    """
    m = model
    days_list = patient["days"]
    records   = patient["records"]
    T         = len(days_list)
    K         = m.n_seeds

    # ── Stage 1: per-biopsy PMA seeds with attention capture ──────────────────
    all_seeds:     List[torch.Tensor] = []
    all_days_flat: List[torch.Tensor] = []
    tok_biopsy:    List[int]  = []
    tok_mod_name:  List[str]  = []
    biopsy_presence: List[torch.Tensor] = []
    biopsy_ends:   List[int]  = []
    pma_attn:     Dict[Tuple, np.ndarray] = {}
    pma_logits:   Dict[Tuple, np.ndarray] = {}
    # (T, n_mods, K) — seed norms; nan where modality absent
    n_mods = len(m._mod_order)
    seeds_norms_grid = np.full((T, n_mods, K), np.nan, dtype=np.float32)

    running_total = 0
    for t_idx in range(T):
        bags  = bags_list[t_idx]
        d_val = float(days_list[t_idx])
        present_this: List[str] = []

        for mod, enc in m.encoders.items():
            feat = bags.get(mod)
            if feat is None:
                continue
            feat = feat.to(device, non_blocking=True)
            if mod == "HE" and feat.shape[0] > m.max_he_patches:
                idx  = torch.randperm(feat.shape[0], device=device)[:m.max_he_patches]
                feat = feat[idx]
            h = enc.encode_patches(feat)                     # (N, H)
            # PMA with attention + logits
            s, attn_w, (dots, relu_d, raw_pow) = m.pma[mod](
                h, return_attn=True, return_logits=True)     # s:(K,H), w:(K,N), ...
            # Add modal identity embedding (same as model forward)
            mod_idx_i = torch.tensor(m._mod_idx[mod], device=device)
            s = s + m.modal_embed(mod_idx_i)

            pma_attn[(t_idx, mod)]   = attn_w.detach().cpu().numpy()    # (K, N)
            pma_logits[(t_idx, mod)] = raw_pow.detach().cpu().numpy()   # (K, N) = relu(q·k)^b

            s_cpu = s.detach().cpu()
            mi = m._mod_idx[mod]
            seeds_norms_grid[t_idx, mi, :] = s_cpu.norm(dim=1).numpy()

            all_seeds.append(s)
            all_days_flat.append(
                torch.full((K,), d_val, dtype=torch.float32, device=device))
            tok_biopsy.extend([t_idx] * K)
            tok_mod_name.extend([mod] * K)
            running_total += K
            present_this.append(mod)

        biopsy_ends.append(running_total)
        biopsy_presence.append(torch.tensor(
            [1.0 if mo in present_this else 0.0 for mo in m._mod_order],
            dtype=torch.float32, device=device))

    if not all_seeds:
        return None

    all_tokens_raw = torch.cat(all_seeds,     dim=0)   # (total_tokens, H)
    days_tok       = torch.cat(all_days_flat, dim=0)   # (total_tokens,)
    seeds_pre_gate = all_tokens_raw.detach().cpu().numpy()

    # ── Stage 2: per-task extraction (gated path) ─────────────────────────────
    tok_biopsy_t = torch.tensor(tok_biopsy, dtype=torch.long, device=device)
    tok_mod_t    = torch.tensor(
        [m._mod_idx[mo] for mo in tok_mod_name], dtype=torch.long, device=device)

    gate_mat:       Dict[str, np.ndarray]  = {}
    temporal_attn:  Dict[str, dict]        = {}
    alpha_per_task: Dict[str, np.ndarray]  = {}
    hazard_traj:    Dict[str, List[float]] = {t: [] for t in tasks}
    logits_out:     Dict[str, float]       = {}

    for task in tasks:
        # --- Task gate ---
        if m.use_task_gate and m.task_gate is not None:
            gate_stack = torch.stack([
                m.task_gate.nets[task](biopsy_presence[b]) for b in range(T)
            ])                                           # (T, n_mods)
            gate_mat[task] = gate_stack.detach().cpu().numpy()
            scale          = gate_stack[tok_biopsy_t, tok_mod_t]   # (total_tokens,)
            tokens_in      = all_tokens_raw * scale.unsqueeze(1)
        else:
            gate_mat[task] = np.ones((T, n_mods), dtype=np.float32)
            tokens_in      = all_tokens_raw

        # --- TemporalSAB with attn capture ---
        attn_store: dict = {}
        orig_fwd = _patch_temporal_sab(m.temporal_sab, attn_store)
        tokens_t = m.temporal_sab(tokens_in, days_tok)
        m.temporal_sab.forward = orig_fwd                # restore
        temporal_attn[task] = attn_store                 # {per_layer, alibi_bias}

        # --- Recency ABMIL (capture α) ---
        gate_m  = m.abmil_V[task](tokens_t) * m.abmil_U[task](tokens_t)
        raw_w   = m.abmil_w[task](gate_m).squeeze(-1)   # (N,)
        sigma   = (days_tok.max() - days_tok.min() + 1.0).clamp(min=1.0)

        if task == "acr_surv":
            anchor_day = days_tok[-1].item()
        else:
            anchor_day = float(days_list[-1])
        bias = -m.recency_gamma[task].abs() * (days_tok - anchor_day).abs() / sigma
        alpha = torch.softmax(raw_w + bias, dim=0)      # (N,)
        alpha_per_task[task] = alpha.detach().cpu().numpy()

        # --- Per-biopsy hazard trajectory (causal: use tokens up to end_idx) ---
        for t_idx, end_idx in enumerate(biopsy_ends):
            if end_idx == 0:
                continue
            tok_t  = tokens_t[:end_idx]
            days_t = days_tok[:end_idx]
            anc    = float(days_list[t_idx])
            gate2  = m.abmil_V[task](tok_t) * m.abmil_U[task](tok_t)
            raw2   = m.abmil_w[task](gate2).squeeze(-1)
            sig2   = (days_t.max() - days_t.min() + 1.0).clamp(min=1.0)
            bias2  = -m.recency_gamma[task].abs() * (days_t - anc).abs() / sig2
            alp2   = torch.softmax(raw2 + bias2, dim=0)
            rep2   = (alp2.unsqueeze(1) * tok_t).sum(0)
            haz    = m.heads[task](rep2).squeeze().item()
            hazard_traj[task].append(haz)

        # Final logit from full sequence
        rep_full   = (alpha.unsqueeze(1) * tokens_t).sum(0)
        logits_out[task] = m.heads[task](rep_full).squeeze().item()

    return {
        "seeds_pre_gate":  seeds_pre_gate,       # (total_tokens, H)
        "pma_attn":        pma_attn,             # {(t_idx, mod): (K, N)}
        "pma_logits":      pma_logits,           # {(t_idx, mod): (K, N)}
        "tok_biopsy":      tok_biopsy,           # list[int]
        "tok_mod_name":    tok_mod_name,         # list[str]
        "biopsy_days":     list(days_list),      # list[float]
        "biopsy_ends":     biopsy_ends,          # list[int]
        "gate_mat":        gate_mat,             # {task: (T, n_mods)}
        "temporal_attn":   temporal_attn,        # {task: {per_layer, alibi_bias}}
        "alpha_per_task":  alpha_per_task,       # {task: (N,)}
        "hazard_traj":     hazard_traj,          # {task: [float×T]}
        "seeds_norms_grid":seeds_norms_grid,     # (T, n_mods, K) nan=absent
        "logits":          logits_out,           # {task: float}
        "patient_id":      patient["patient_id"],
        "records":         patient["records"],
        "n_biopsies":      T,
    }


# ── Panel L_global: ALiBi slopes, recency γ, decay curves ────────────────────

def plot_L_global(model: nn.Module, out_dir: Path, tasks: List[str]):
    slopes = model.temporal_sab.alibi_slopes.abs().detach().cpu().numpy()
    n_heads = len(slopes)
    gammas  = {t: model.recency_gamma[t].abs().item() for t in tasks}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Longitudinal-MK-MT — Global Model Properties", fontsize=FONT_TITLE + 1,
                 fontweight="bold", y=1.01)

    # Panel A: ALiBi head slopes
    ax = axes[0]
    xs = np.arange(n_heads)
    colors_heads = plt.cm.plasma(np.linspace(0.15, 0.85, n_heads))
    bars = ax.bar(xs, slopes, color=colors_heads, edgecolor="white", linewidth=0.8, width=0.65)
    for bar, val in zip(bars, slopes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("ALiBi Slope |s_h|")
    ax.set_title("A  ALiBi Head Slopes\n(larger = faster temporal decay)", fontsize=FONT_LABEL)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"H{i}" for i in range(n_heads)], fontsize=FONT_TICK)
    ax.set_ylim(0, max(slopes) * 1.3)
    ax.grid(axis="x", alpha=0)

    # Panel B: Per-task recency γ
    ax = axes[1]
    task_labels = [TASK_LABELS.get(t, t) for t in tasks]
    task_cols   = [TASK_COLORS.get(t, "#888888") for t in tasks]
    gvals = [gammas[t] for t in tasks]
    bars2 = ax.barh(task_labels, gvals, color=task_cols, edgecolor="white",
                    linewidth=0.8, height=0.55)
    for bar, val in zip(bars2, gvals):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"γ = {val:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Recency γ  (larger = stronger recency bias)")
    ax.set_title("B  Per-Task Recency γ\n(downweights distant biopsies)", fontsize=FONT_LABEL)
    ax.set_xlim(0, max(gvals) * 1.5)
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=0)

    # Panel C: ALiBi temporal bias curves per head
    ax = axes[2]
    delta_days = np.linspace(0, 730, 300)
    for hi, (s, col) in enumerate(zip(slopes, colors_heads)):
        bias_curve = -s * delta_days / 730.0   # normalize as in model
        ax.plot(delta_days, bias_curve, color=col, lw=1.8, label=f"H{hi} (s={s:.3f})")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Temporal Distance  Δt  (days)")
    ax.set_ylabel("ALiBi Attention Bias")
    ax.set_title("C  ALiBi Decay Curves\n(attention penalty for temporal distance)",
                 fontsize=FONT_LABEL)
    ax.legend(fontsize=8, ncol=2, loc="lower left",
              framealpha=0.85, edgecolor="lightgrey")

    plt.tight_layout()
    png = _savefig(fig, out_dir, "L_global_model_properties")
    plt.close(fig)
    print(f"  [L_global] → {png.name}")
    return png


# ── Seed label builders ───────────────────────────────────────────────────────

def _build_token_labels(tok_biopsy, tok_mod_name, biopsy_days, n_seeds, prefix=""):
    """Build short labels for each token: 'B1·HE·s00' etc."""
    labels = []
    seen_counts: Dict[Tuple, int] = {}
    for bi, mo in zip(tok_biopsy, tok_mod_name):
        key = (bi, mo)
        idx = seen_counts.get(key, 0)
        labels.append(f"{prefix}B{bi+1}·{mo}·s{idx:02d}")
        seen_counts[key] = idx + 1
    return labels


def _biopsy_spans(tok_biopsy, biopsy_days):
    """Return list of (start_idx, end_idx, day_str) per biopsy."""
    spans = []
    current_b, start = None, 0
    for i, b in enumerate(tok_biopsy):
        if b != current_b:
            if current_b is not None:
                spans.append((start, i, current_b, biopsy_days[current_b]))
            current_b, start = b, i
    if current_b is not None:
        spans.append((start, len(tok_biopsy), current_b, biopsy_days[current_b]))
    return spans  # [(start, end, b_idx, day_value), ...]


# ── Panel L1: Seed concept timeline ──────────────────────────────────────────

# Per-modality colormaps and top-seed count
_MOD_CMAPS = {"HE": "Blues", "BAL": "Greens", "CT": "Oranges", "Clinical": "Purples"}
_L1_TOP_K  = 8   # show top-K seeds per modality by mean activation

def plot_L1(extr: dict, model: nn.Module, out_dir: Path):
    """
    Redesigned L1: shows top-K most active seeds per modality.
    - Per-modality independent colormap normalization (each modality scaled to its own range)
    - Only seeds with non-zero activation shown (top-8 per modality)
    - Separate subplot row per modality for clarity
    - X-axis: day labels, showing biopsy # every ~5 steps
    """
    T      = extr["n_biopsies"]
    K      = model.n_seeds
    days   = extr["biopsy_days"]
    norms  = extr["seeds_norms_grid"]     # (T, n_mods, K)
    pid    = extr["patient_id"]

    # Determine which modalities are actually present (have at least 1 biopsy)
    present_mods = []
    for mi, mo in enumerate(MOD_ORDER):
        col_data = norms[:, mi, :]  # (T, K)
        if not np.all(np.isnan(col_data)) and np.nanmax(col_data) > 0:
            present_mods.append((mi, mo))
    if not present_mods:
        return None

    n_mods = len(present_mods)
    # For each present modality, pick top-K seeds by mean activation
    mod_top_seeds = {}
    for mi, mo in present_mods:
        col_data = norms[:, mi, :]   # (T, K)
        means    = np.nanmean(col_data, axis=0)  # (K,)
        top_ki   = np.argsort(means)[::-1][:_L1_TOP_K]
        mod_top_seeds[(mi, mo)] = sorted(top_ki.tolist())  # keep seed order for readability

    # Total rows = sum of top seeds per modality
    total_rows = sum(len(v) for v in mod_top_seeds.values())

    # Day-aligned column edges for pcolormesh (proportional to actual time gaps)
    days_arr = np.asarray(days, dtype=float)
    if T > 1:
        mids = (days_arr[:-1] + days_arr[1:]) / 2.0
        left_edge  = days_arr[0] - (days_arr[1] - days_arr[0]) / 2
        right_edge = days_arr[-1] + (days_arr[-1] - days_arr[-2]) / 2
        col_edges  = np.concatenate([[left_edge], mids, [right_edge]])
    else:
        col_edges = np.array([days_arr[0] - 15, days_arr[0] + 15])
    day_span = col_edges[-1] - col_edges[0]

    # Figure: one row-band per modality, stacked vertically
    row_heights = [len(mod_top_seeds[(mi, mo)]) for mi, mo in present_mods]
    fig_h = max(5, sum(row_heights) * 0.55 + 1.5)
    fig_w = max(10, min(0.5 * T + 4, 28))

    fig, axes = plt.subplots(
        n_mods, 1, figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": row_heights},
        squeeze=False)
    axes = axes[:, 0]

    fig.suptitle(
        f"L1  Seed Concept Timeline — Patient {pid}  ({T} biopsies)\n"
        f"Top-{_L1_TOP_K} seeds per modality · per-modality normalized · grey = absent\n"
        "X-axis proportional to actual days from transplant",
        fontsize=FONT_TITLE, fontweight="bold", y=1.01)

    for ax_idx, ((mi, mo), ax) in enumerate(zip(present_mods, axes)):
        top_ki   = mod_top_seeds[(mi, mo)]
        n_rows   = len(top_ki)
        sub_data = np.full((n_rows, T), np.nan, dtype=np.float32)
        for row_i, ki in enumerate(top_ki):
            sub_data[row_i] = norms[:, mi, ki]

        # Relative within-biopsy z-score normalization
        sub_rel = sub_data.copy()
        for t in range(T):
            col = sub_data[:, t]
            valid = col[~np.isnan(col)]
            if len(valid) >= 2 and valid.std() > 1e-6:
                sub_rel[:, t] = (col - valid.mean()) / valid.std()
            elif not np.all(np.isnan(col)):
                sub_rel[:, t] = 0.0

        # Fill NaN with 0 for pcolormesh (absent biopsies drawn grey via axvspan)
        sub_plot = np.where(np.isnan(sub_rel), 0.0, sub_rel)

        vlim = max(np.nanpercentile(np.abs(sub_rel[~np.isnan(sub_rel)]), 95), 0.5) \
               if not np.all(np.isnan(sub_rel)) else 1.0

        # Row edges (uniform — one row per seed)
        row_edges = np.arange(n_rows + 1) - 0.5

        # pcolormesh: X = day-aligned column edges, Y = row edges
        cmap_div = plt.get_cmap("RdBu_r")
        cmap_div.set_bad("#cccccc")
        pm = ax.pcolormesh(col_edges, row_edges, sub_plot,
                           cmap=cmap_div, vmin=-vlim, vmax=vlim, shading="flat")

        # Grey-out absent biopsies (modality not present at that timepoint)
        for t in range(T):
            if np.all(np.isnan(sub_data[:, t])):
                ax.axvspan(col_edges[t], col_edges[t + 1],
                           color="#cccccc", alpha=0.55, zorder=5)

        # Thin dividers between biopsies
        for t in range(1, T):
            ax.axvline(col_edges[t], color="white", lw=0.4, alpha=0.5)

        # Y-axis: seed labels
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f"s{ki:02d}" for ki in top_ki],
                           fontsize=8, color=MOD_COLORS[mo], fontweight="bold")
        ax.set_ylim(row_edges[0], row_edges[-1])

        # Modality label on left
        ax.set_ylabel(mo, fontsize=11, color=MOD_COLORS[mo], fontweight="bold",
                      rotation=0, ha="right", va="center", labelpad=38)

        # X-axis ticks at biopsy days — up to 12 shown, plus tick marks for all
        tick_step_t = max(1, T // 10)
        tick_idx    = list(range(0, T, tick_step_t))
        if T - 1 not in tick_idx:
            tick_idx.append(T - 1)
        tick_day_pos = [days_arr[t] for t in tick_idx]
        tick_lbls    = [f"B{t+1}\n{int(days_arr[t])}d" for t in tick_idx]

        ax.set_xlim(col_edges[0], col_edges[-1])
        if ax_idx < n_mods - 1:
            ax.set_xticks(tick_day_pos)
            ax.set_xticklabels([""] * len(tick_day_pos))   # tick marks but no labels
        else:
            ax.set_xticks(tick_day_pos)
            ax.set_xticklabels(tick_lbls, fontsize=8, ha="center")
            ax.set_xlabel("Days from transplant  (proportional spacing →)", fontsize=FONT_LABEL)

        # Colorbar
        cb = plt.colorbar(pm, ax=ax, fraction=0.02, pad=0.01, aspect=15)
        cb.set_label("Rel. activation\n(z-score)", fontsize=7)
        cb.ax.tick_params(labelsize=7)
        cb.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        abs_max = np.nanmax(sub_data) if not np.all(np.isnan(sub_data)) else 0.0
        ax.text(1.01, 0.0, f"max\n{abs_max:.1f}", transform=ax.transAxes,
                fontsize=6, va="bottom", ha="left", color="grey", style="italic")

        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    plt.tight_layout(h_pad=0.3)
    png = _savefig(fig, out_dir, f"L1_seed_timeline_pid{pid}")
    plt.close(fig)
    print(f"  [L1] patient={pid}  → {png.name}")
    return png


# ── Panel L2: TemporalSAB cross-biopsy attention ─────────────────────────────

def _aggregate_biopsy_attn(A_tok, tok_b, T):
    """Collapse token-level (N, N) attention to biopsy-level (T, T) by summing blocks."""
    B = np.zeros((T, T), dtype=np.float64)
    tok_b_arr = np.asarray(tok_b)
    for qi in range(T):
        for ki in range(T):
            q_mask = tok_b_arr == qi
            k_mask = tok_b_arr == ki
            if q_mask.any() and k_mask.any():
                B[qi, ki] = A_tok[np.ix_(q_mask, k_mask)].sum()
    return B


def plot_L2(extr: dict, model: nn.Module, out_dir: Path, tasks: List[str]):
    """
    Redesigned L2: biopsy-level (T×T) attention heatmap.
    Aggregates token-level TemporalSAB weights to per-biopsy totals — far more readable
    than the raw N×N token matrix for patients with many modalities per biopsy.
    """
    pid    = extr["patient_id"]
    days   = extr["biopsy_days"]
    T      = extr["n_biopsies"]
    tok_b  = extr["tok_biopsy"]

    # X/Y tick labels: "B1\nDay 0", every ~5 biopsies
    tick_step = max(1, T // 8)
    tick_pos  = list(range(0, T, tick_step))
    if T - 1 not in tick_pos and (T - 1 - tick_pos[-1]) > tick_step // 2:
        tick_pos.append(T - 1)
    tick_lbl  = [f"B{t+1}\n{int(days[t])}d" for t in tick_pos]

    ntasks = len(tasks)
    fig, axes = plt.subplots(1, ntasks, figsize=(5.5 * ntasks, 5.0))
    if ntasks == 1:
        axes = [axes]
    fig.suptitle(
        f"L2  TemporalSAB Cross-Biopsy Attention — Patient {pid}  ({T} biopsies)\n"
        "Summed token attention per biopsy pair · averaged over heads & layers · causal lower-triangle",
        fontsize=FONT_TITLE, fontweight="bold", y=1.03)

    for ax, task in zip(axes, tasks):
        tattn = extr["temporal_attn"].get(task, {})
        per_layer = tattn.get("per_layer", [])
        if not per_layer:
            ax.set_visible(False)
            continue

        # Average over layers then heads → (N, N) token attention
        A_tok = np.mean(per_layer, axis=0)   # (n_heads, N, N) mean over layers
        A_tok = A_tok.mean(axis=0)           # (N, N) mean over heads

        # Aggregate to biopsy-level T×T
        B = _aggregate_biopsy_attn(A_tok, tok_b, T)

        # Normalize each query row so we can see relative key focus (not just row-sum)
        row_sums = B.sum(axis=1, keepdims=True)
        B_norm   = np.where(row_sums > 0, B / np.maximum(row_sums, 1e-9), 0.0)

        vmax = np.percentile(B_norm[B_norm > 0], 95) if (B_norm > 0).any() else 0.1
        im   = ax.imshow(B_norm, aspect="equal", cmap="YlOrRd",
                         vmin=0, vmax=vmax, interpolation="nearest")

        # Diagonal line to highlight self-attention
        ax.plot([0, T - 1], [0, T - 1], color="white", lw=0.8, ls="--", alpha=0.5)

        # Grid lines every 5 biopsies for readability
        for t in range(0, T, max(1, T // 8)):
            ax.axhline(t - 0.5, color="white", lw=0.6, alpha=0.5)
            ax.axvline(t - 0.5, color="white", lw=0.6, alpha=0.5)

        # Mark ACR+ biopsies: red tick labels + small triangle on both axes
        records = extr.get("records", [])
        acr_pos_biops = [t for t, rec in enumerate(records) if rec.get("label") == 1]

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, fontsize=8, rotation=0, ha="center")
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lbl, fontsize=8)

        # Overlay small red dots on the diagonal for ACR+ biopsies
        for t in acr_pos_biops:
            ax.plot(t, t, "v", color="#e53935", ms=5, zorder=6, alpha=0.85)
            # Red vertical/horizontal guide lines at low alpha
            ax.axvline(t, color="#e53935", lw=0.5, ls=":", alpha=0.25, zorder=1)
            ax.axhline(t, color="#e53935", lw=0.5, ls=":", alpha=0.25, zorder=1)

        ax.set_xlabel("Key biopsy (past / causal →)", fontsize=9)
        ax.set_ylabel("Query biopsy (current)", fontsize=9)
        ax.set_title(TASK_LABELS.get(task, task), fontsize=10,
                     color=TASK_COLORS.get(task, "#555"), fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Row-norm attention", fontsize=7)
        cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    png = _savefig(fig, out_dir, f"L2_temporal_sab_attn_pid{pid}")
    plt.close(fig)
    print(f"  [L2] patient={pid}  → {png.name}")
    return png


# ── Panel L2b: ALiBi bias heatmap (one task, shared structure) ───────────────

def plot_L2b_alibi(extr: dict, model: nn.Module, out_dir: Path, ref_task: str = None):
    """Show the ALiBi bias component per head for one reference task."""
    pid  = extr["patient_id"]
    days = extr["biopsy_days"]
    tok_b  = extr["tok_biopsy"]
    tok_m  = extr["tok_mod_name"]
    spans  = _biopsy_spans(tok_b, days)
    tok_labels = _build_token_labels(tok_b, tok_m, days, model.n_seeds)

    task = ref_task or (list(extr["temporal_attn"].keys())[0] if extr["temporal_attn"] else None)
    if task is None:
        return
    alibi = extr["temporal_attn"].get(task, {}).get("alibi_bias")
    if alibi is None:
        return

    n_heads = alibi.shape[0]
    ncols = min(4, n_heads)
    nrows = math.ceil(n_heads / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows))
    axes_flat = np.array(axes).flatten()
    fig.suptitle(
        f"L2b  ALiBi Bias per Attention Head — Patient {pid}\n"
        "Bias = −|slope_h| × |days_q − days_k| / days_range  (causal future = −∞)",
        fontsize=FONT_TITLE, y=1.01)

    vabs = np.abs(alibi).max()
    N_tok = alibi.shape[1]  # total tokens
    # Build causal mask: token ordering is chronological (biopsy 0 first)
    # Upper triangle (col > row) = future keys → masked at runtime with -∞
    causal_mask = np.triu(np.ones((N_tok, N_tok), dtype=bool), k=1)
    # Build biopsy-aligned tick positions (midpoint of each biopsy's token span)
    biopsy_mids = []
    biopsy_tick_lbls = []
    for s_start, s_end, b_idx, day_v in spans:
        mid = (s_start + s_end - 1) / 2
        biopsy_mids.append(mid)
        biopsy_tick_lbls.append(f"B{b_idx+1}\n{int(day_v)}d")
    # Only label every Nth biopsy to avoid crowding
    n_biopsies = len(spans)
    label_step = max(1, n_biopsies // 8)

    for hi in range(n_heads):
        ax = axes_flat[hi]
        # Plot ALiBi bias (pre-causal-mask component)
        bias_plot = alibi[hi].copy().astype(float)
        im = ax.imshow(bias_plot, aspect="auto", cmap="RdBu",
                       vmin=-vabs, vmax=0, interpolation="nearest")
        # Overlay causal mask (future tokens) in grey
        masked_rgba = np.zeros((N_tok, N_tok, 4))
        masked_rgba[causal_mask] = [0.6, 0.6, 0.6, 0.65]   # grey, semi-transparent
        ax.imshow(masked_rgba, aspect="auto", interpolation="nearest")
        # Diagonal line separating causal/non-causal
        ax.plot([0 - 0.5, N_tok - 0.5], [0 - 0.5, N_tok - 0.5],
                color="white", lw=1.0, ls="--", alpha=0.7)
        # Biopsy boundary grid
        for s_start, s_end, b_idx, day_v in spans:
            ax.axhline(s_end - 0.5, color="white", lw=0.6, alpha=0.5)
            ax.axvline(s_end - 0.5, color="white", lw=0.6, alpha=0.5)
        slope_v = model.temporal_sab.alibi_slopes.abs()[hi].item()
        ax.set_title(f"Head {hi}  (slope={slope_v:.3f})", fontsize=9)
        # Day-aligned ticks
        shown_idx = list(range(0, n_biopsies, label_step))
        if n_biopsies - 1 not in shown_idx:
            shown_idx.append(n_biopsies - 1)
        tick_pos  = [biopsy_mids[i] for i in shown_idx]
        tick_lbls = [biopsy_tick_lbls[i] for i in shown_idx]
        ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lbls, fontsize=6, rotation=45, ha="right")
        ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lbls, fontsize=6)
        ax.set_xlabel("Key biopsy (past →)", fontsize=7)
        ax.set_ylabel("Query biopsy (→)", fontsize=7)
        # Mark causal region label
        ax.text(0.72, 0.05, "causal\nmask\n(−∞)", transform=ax.transAxes,
                fontsize=7, ha="center", va="bottom", color="grey",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.7))
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=6)

    for hi in range(n_heads, len(axes_flat)):
        axes_flat[hi].set_visible(False)

    plt.tight_layout()
    png = _savefig(fig, out_dir, f"L2b_alibi_bias_pid{pid}")
    plt.close(fig)
    print(f"  [L2b] patient={pid}  → {png.name}")
    return png


# ── Panel L3: Recency ABMIL α per biopsy per task ────────────────────────────

def plot_L3(extr: dict, model: nn.Module, out_dir: Path, tasks: List[str]):
    """
    Bar chart: total ABMIL α weight assigned to each biopsy, per task.
    Shows which historical biopsies the model relies on most.
    """
    pid   = extr["patient_id"]
    days  = extr["biopsy_days"]
    T     = extr["n_biopsies"]
    tok_b = extr["tok_biopsy"]

    # For dense timelines (T > 15), just show biopsy number to avoid label overlap
    if T > 15:
        biopsy_xticklabels = [f"B{t+1}" for t in range(T)]
    else:
        biopsy_xticklabels = [f"B{t+1}\nDay {int(days[t])}" for t in range(T)]

    ntasks = len(tasks)
    # Cap total width so figure stays printable; bars compress gracefully
    panel_w = max(4.0, min(0.28 * T + 1.5, 9))
    total_w = min(panel_w * ntasks, 22)
    fig, axes = plt.subplots(1, ntasks, figsize=(total_w, 5.0), sharey=False)
    if ntasks == 1:
        axes = [axes]
    fig.suptitle(
        f"L3  Recency ABMIL — Weight per Biopsy — Patient {pid}  ({T} biopsies)\n"
        "Total ABMIL α per biopsy · top-5 labelled · anchor = last biopsy",
        fontsize=FONT_TITLE, fontweight="bold", y=1.02)

    days_arr = np.asarray(days, dtype=float)
    # Days-proportional bar widths
    if T > 1:
        gaps = np.diff(days_arr)
        bar_widths = np.append(gaps, gaps[-1]) * 0.85
    else:
        bar_widths = np.array([30.0])
    bar_widths = np.clip(bar_widths, 1.0, None)
    day_span = days_arr[-1] - days_arr[0]

    for ax, task in zip(axes, tasks):
        alpha = extr["alpha_per_task"].get(task)
        if alpha is None:
            ax.set_visible(False)
            continue

        # Sum α over tokens belonging to each biopsy
        biopsy_alpha = np.zeros(T, dtype=np.float32)
        for i, b in enumerate(tok_b):
            biopsy_alpha[b] += alpha[i]

        col = TASK_COLORS.get(task, "#777")
        # Gradient color: earlier = lighter; bars at actual day positions
        shade = [plt.cm.Blues(0.35 + 0.55 * (t / max(T - 1, 1))) for t in range(T)]
        bars = ax.bar(days_arr, biopsy_alpha, color=shade, edgecolor=col,
                      linewidth=0.8, width=bar_widths, align="edge")

        # Only label top-5 bars
        top5_idx = set(np.argsort(biopsy_alpha)[::-1][:5])
        for i, (bar, val) in enumerate(zip(bars, biopsy_alpha)):
            if i in top5_idx:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color=col)

        # ACR event markers
        records = extr.get("records", [])
        for t, rec in enumerate(records):
            if rec.get("label") == 1:
                ax.axvline(days_arr[t], color="#e53935", lw=0.9, ls="--", alpha=0.4, zorder=0)

        # X-axis: actual days with minimum spacing to prevent label crowding
        day_range_l3 = days_arr[-1] - days_arr[0]
        min_gap_l3 = max(30.0, day_range_l3 / 10)
        tick_idx = [0]
        for t in range(1, T - 1):
            if days_arr[t] - days_arr[tick_idx[-1]] >= min_gap_l3:
                tick_idx.append(t)
        if T - 1 not in tick_idx:
            tick_idx.append(T - 1)
        ax.set_xticks([days_arr[t] for t in tick_idx])
        ax.set_xticklabels([f"B{t+1}\n{int(days_arr[t])}d" for t in tick_idx], fontsize=8)
        ax.set_xlim(days_arr[0] - max(15, 0.01 * day_span),
                    days_arr[-1] + max(20, 0.03 * day_span))
        ax.set_xlabel("Days from transplant  (→)", fontsize=FONT_LABEL)
        ax.set_ylabel("Σ ABMIL α weight", fontsize=FONT_LABEL)
        ax.set_title(TASK_LABELS.get(task, task), fontsize=10, color=col)
        ax.set_ylim(0, biopsy_alpha.max() * 1.35)

        # Recency γ annotation
        gamma_v = model.recency_gamma[task].abs().item()
        ax.text(0.97, 0.97, f"γ = {gamma_v:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.8))

    plt.tight_layout()
    png = _savefig(fig, out_dir, f"L3_recency_abmil_pid{pid}")
    plt.close(fig)
    print(f"  [L3] patient={pid}  → {png.name}")
    return png


# ── Panel L4: Hazard trajectory ───────────────────────────────────────────────

def plot_L4(extr: dict, model: nn.Module, out_dir: Path, tasks: List[str]):
    """
    Line chart: model output trajectory plotted on biopsy days.
    X-axis = days from first biopsy (causal accumulation).
    For acr_cls: y = classification logit. For survival tasks: y = hazard (log-risk).
    Vertical markers: ACR+ (red dashed), CLAD event (orange solid), Death (black solid).
    """
    pid     = extr["patient_id"]
    T       = extr["n_biopsies"]
    traj    = extr["hazard_traj"]
    records = extr["records"]

    # Prefer days from transplant; fall back to days from first biopsy
    tx_days = extr.get("transplant_days", [])
    if tx_days and all(v is not None for v in tx_days[:T]):
        days   = np.asarray(tx_days[:T], dtype=float)
        x_label = "Days from transplant (causal accumulation →)"
        use_tx  = True
    else:
        days   = np.asarray(extr["biopsy_days"], dtype=float)
        x_label = "Days from first biopsy (causal accumulation →)"
        use_tx  = False

    # Determine y-axis label based on tasks present
    has_cls_only = all(t == "acr_cls" for t in tasks)
    has_surv     = any(t != "acr_cls" for t in tasks)
    if has_cls_only:
        ylabel = "Classification Logit"
    elif has_surv and "acr_cls" in tasks:
        ylabel = "Model output (logit / hazard)"
    else:
        ylabel = "Hazard Score (log-risk)"

    fig, ax = plt.subplots(figsize=(14, 5))

    for task in tasks:
        hvals = traj.get(task, [])
        if not hvals:
            continue
        n   = len(hvals)
        xs  = days[:n]
        col = TASK_COLORS.get(task, "#777")
        lbl = TASK_LABELS.get(task, task)
        # For cls task, label as logit explicitly
        if task == "acr_cls":
            lbl = f"{lbl} (logit)"
        ax.plot(xs, hvals, "o-", color=col, lw=2.0, ms=5, label=lbl)
        ax.annotate(f"{hvals[-1]:.3f}", xy=(xs[-1], hvals[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=8, color=col, fontweight="bold", va="center")

    # ── Event annotations ─────────────────────────────────────────────────────
    import matplotlib.lines as mlines
    y_top = ax.get_ylim()[1]
    y_bot = ax.get_ylim()[0]
    legend_extra  = []
    acr_pos_seen  = False
    acr_neg_seen  = False
    clad_marked   = False
    death_marked  = False

    for t_idx, rec in enumerate(records):
        bday = days[t_idx]

        # ACR biopsy label
        label = rec.get("label")
        if label == 1:
            ax.axvline(bday, color="#e53935", lw=1.2, ls="--", alpha=0.55, zorder=0)
            ax.text(bday, y_top, "ACR+", rotation=90, ha="center", va="top",
                    fontsize=7, color="#e53935", alpha=0.9, fontweight="bold")
            acr_pos_seen = True
        elif label == 0:
            ax.axvline(bday, color="#43a047", lw=0.7, ls=":", alpha=0.3, zorder=0)
            acr_neg_seen = True

        # CLAD event: clad_time is days from transplant (absolute).
        # If using transplant x-axis, plot at clad_time directly.
        # If using first-biopsy x-axis, plot at bday + clad_time - biopsy_tx_day.
        if not clad_marked and rec.get("clad_event") == 1.0:
            ct = rec.get("clad_time", float("nan"))
            if ct == ct and ct > 0:  # not NaN, positive
                event_x = float(ct) if use_tx else (bday + float(ct) - (tx_days[t_idx] or bday))
                ax.axvline(event_x, color="#FF6F00", lw=2.0, ls="-", alpha=0.8, zorder=1)
                ax.text(event_x, y_bot + (y_top - y_bot) * 0.05,
                        "CLAD", rotation=90, ha="center", va="bottom",
                        fontsize=8, color="#FF6F00", fontweight="bold")
                clad_marked = True

        # Death event: death_time is days from transplant (absolute).
        if not death_marked and rec.get("death_event") == 1.0:
            dt = rec.get("death_time", float("nan"))
            if dt == dt and dt > 0:  # not NaN, positive
                event_x = float(dt) if use_tx else (bday + float(dt) - (tx_days[t_idx] or bday))
                ax.axvline(event_x, color="#212121", lw=2.0, ls="-", alpha=0.8, zorder=1)
                ax.text(event_x, y_bot + (y_top - y_bot) * 0.15,
                        "Death", rotation=90, ha="center", va="bottom",
                        fontsize=8, color="#212121", fontweight="bold")
                death_marked = True

    # Legend entries for markers
    if acr_pos_seen:
        legend_extra.append(mlines.Line2D([], [], color="#e53935", lw=1.2, ls="--",
                                          label="ACR+ biopsy"))
    if acr_neg_seen:
        legend_extra.append(mlines.Line2D([], [], color="#43a047", lw=0.7, ls=":",
                                          label="No-ACR biopsy"))
    if clad_marked:
        legend_extra.append(mlines.Line2D([], [], color="#FF6F00", lw=2.0, ls="-",
                                          label="CLAD event"))
    if death_marked:
        legend_extra.append(mlines.Line2D([], [], color="#212121", lw=2.0, ls="-",
                                          label="Death event"))

    # Min-gap tick spacing: skip ticks closer than max(30d, range/12)
    day_range_l4 = float(days[-1] - days[0]) if T > 1 else 1.0
    min_gap_l4   = max(30.0, day_range_l4 / 12)
    tick_idx_l4  = [0]
    for t in range(1, T - 1):
        if days[t] - days[tick_idx_l4[-1]] >= min_gap_l4:
            tick_idx_l4.append(t)
    if T - 1 not in tick_idx_l4:
        tick_idx_l4.append(T - 1)
    tick_xs   = [days[t] for t in tick_idx_l4]
    tick_lbls = [f"B{t+1}\n{int(days[t])}d" for t in tick_idx_l4]
    ax.set_xticks(tick_xs)
    ax.set_xticklabels(tick_lbls, fontsize=8)
    ax.set_xlabel(x_label, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_title(
        f"L4  Model Output Trajectory — Patient {pid}\n"
        "Per-biopsy model output (causal); vertical lines = ACR labels / CLAD / Death events",
        fontsize=FONT_TITLE)
    handles_existing, labels_existing = ax.get_legend_handles_labels()
    all_handles = handles_existing + legend_extra
    all_labels  = labels_existing + [h.get_label() for h in legend_extra]
    ax.legend(handles=all_handles, labels=all_labels, fontsize=9, loc="best",
              framealpha=0.9, edgecolor="lightgrey")
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    # Extend x-axis right to show any events after last biopsy
    x_right = days[-1] + max(30, 0.04 * (days[-1] - days[0] + 1))
    if clad_marked or death_marked:
        x_right = max(x_right, days[-1] + max(200, 0.15 * (days[-1] - days[0] + 1)))
    ax.set_xlim(days[0] - max(20, 0.02 * (days[-1] - days[0] + 1)), x_right)

    plt.tight_layout()
    png = _savefig(fig, out_dir, f"L4_hazard_trajectory_pid{pid}")
    plt.close(fig)
    print(f"  [L4] patient={pid}  → {png.name}")
    return png


# ── Panel L5: PMA seed–cluster affinity (per biopsy) ─────────────────────────

def plot_L5_pma_affinity(extr: dict, model: nn.Module, out_dir: Path,
                          cluster_names: Dict[str, List[str]]):
    """
    Per-biopsy, per-modality: heatmap of PMA seed × cluster affinity.
    Rows = seeds (s0..sK-1), Cols = clusters (sorted by biology for HE).
    Cell = sum of b-cos attention mass on cluster patches.
    """
    pid  = extr["patient_id"]
    days = extr["biopsy_days"]
    T    = extr["n_biopsies"]

    present_mods = [mo for mo in MOD_ORDER
                    if any((t_idx, mo) in extr["pma_attn"] for t_idx in range(T))]
    if not present_mods:
        return

    K = model.n_seeds

    for mo in present_mods:
        # Collect affinity matrices across biopsies that have this modality
        biopsy_affinities = []
        for t_idx in range(T):
            attn = extr["pma_attn"].get((t_idx, mo))
            if attn is None:
                continue
            biopsy_affinities.append((t_idx, attn))

        if not biopsy_affinities:
            continue

        cnames = cluster_names.get(mo, [])
        n_clus = len(cnames) if cnames else 0

        if n_clus == 0:
            # No cluster names: show raw attention mass per patch position
            for t_idx, attn in biopsy_affinities:
                N = attn.shape[1]
                # Summarize: top-20 patches by total attention
                top_n = min(30, N)
                mass  = attn.sum(axis=0)   # (N,)
                top_i = np.argsort(mass)[::-1][:top_n]
                aff_top = attn[:, top_i]   # (K, top_n)

                fig, ax = plt.subplots(figsize=(max(6, 0.25 * top_n), 3))
                im = ax.imshow(aff_top, aspect="auto", cmap="YlOrRd",
                               vmin=0, vmax=aff_top.max(), interpolation="nearest")
                ax.set_yticks(range(K))
                ax.set_yticklabels([f"s{ki:02d}" for ki in range(K)], fontsize=8)
                ax.set_xlabel(f"Top-{top_n} patches by attention mass", fontsize=9)
                ax.set_ylabel("Seed", fontsize=9)
                ax.set_title(
                    f"L5  PMA Seed Affinity — {mo} | B{t_idx+1} Day {int(days[t_idx])} — Patient {pid}",
                    fontsize=FONT_TITLE - 1)
                cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
                cb.set_label("b-cos attn weight", fontsize=8)
                _savefig(fig, out_dir, f"L5_pma_affinity_pid{pid}_{mo}_B{t_idx+1}")
                plt.close(fig)
                print(f"  [L5] patient={pid} mod={mo} B{t_idx+1} (raw)")
            continue

        # HE/CT: use cluster names
        if mo == "HE":
            sort_order = sorted(range(n_clus),
                                key=lambda i: (
                                    list(HE_BIO_COLORS.keys()).index(
                                        HE_BIO_MAP.get(cnames[i], "Unknown"))
                                    if HE_BIO_MAP.get(cnames[i], "Unknown") in HE_BIO_COLORS
                                    else len(HE_BIO_COLORS),
                                    i))
            top_n = min(n_clus, 54)    # show all HE clusters
        elif mo == "Clinical":
            top_n = min(40, n_clus)
            sort_order = None           # will sort by attention mass below
        else:
            top_n = min(n_clus, 37)
            sort_order = None

        for t_idx, attn in biopsy_affinities:
            N = attn.shape[1]   # num patches for this biopsy
            if N != n_clus:
                # Cluster names vs patch count mismatch — skip cluster-level view
                continue

            if sort_order is not None:
                so = sort_order[:top_n]
            else:
                mass = attn.sum(axis=0)
                so   = list(np.argsort(mass)[::-1][:top_n])

            aff_sub = attn[:, so]                 # (K, top_n)
            clus_lbls = [cnames[i] for i in so]
            # Truncate long clinical feature names
            clus_lbls = [c[:25] + "…" if len(c) > 26 else c for c in clus_lbls]

            fig_w = max(8, 0.22 * len(so))
            fig, ax = plt.subplots(figsize=(fig_w, 3.5))
            im = ax.imshow(aff_sub, aspect="auto", cmap="YlOrRd",
                           vmin=0, vmax=np.percentile(aff_sub, 98),
                           interpolation="nearest")
            ax.set_yticks(range(K))
            ax.set_yticklabels([f"{mo}·s{ki:02d}" for ki in range(K)], fontsize=8)
            ax.set_xticks(range(len(so)))
            ax.set_xticklabels(clus_lbls, rotation=75, ha="right", fontsize=6)
            ax.set_xlabel("Cluster / Feature  (sorted by biology or attention mass)", fontsize=9)
            ax.set_ylabel("Seed Index", fontsize=9)
            ax.set_title(
                f"L5  PMA b-cos Affinity — {mo} | B{t_idx+1} Day {int(days[t_idx])} — Patient {pid}",
                fontsize=FONT_TITLE - 1)

            # Biological category color bands for HE
            if mo == "HE" and HE_BIO_MAP:
                prev_cat, cat_start = None, 0
                for ci, idx in enumerate(so):
                    cat = HE_BIO_MAP.get(cnames[idx], "Unknown")
                    if cat != prev_cat and prev_cat is not None:
                        ax.axvspan(cat_start - 0.5, ci - 0.5, alpha=0.08,
                                   color=HE_BIO_COLORS.get(prev_cat, "#ccc"), zorder=0)
                    prev_cat  = cat
                    cat_start = ci

            cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
            cb.set_label("b-cos attn", fontsize=8)
            plt.tight_layout()
            _savefig(fig, out_dir, f"L5_pma_affinity_pid{pid}_{mo}_B{t_idx+1}")
            plt.close(fig)
            print(f"  [L5] patient={pid} mod={mo} B{t_idx+1}")


# ── Panel L6: Gate matrix across biopsies ────────────────────────────────────

def plot_L6_gate(extr: dict, model: nn.Module, out_dir: Path, tasks: List[str]):
    """
    Per-task: heatmap (biopsy × modality) of task gate values.
    Shows which modalities the model trusts at each timepoint for each task.
    """
    pid     = extr["patient_id"]
    days    = extr["biopsy_days"]
    T       = extr["n_biopsies"]
    gate_d  = extr["gate_mat"]

    if not gate_d:
        return

    ntasks = len(tasks)
    # Cap figure height so long timelines stay readable
    row_h = 0.30 if T > 20 else 0.45
    fig_h = max(4, min(18, row_h * T + 2.0))
    fig, axes = plt.subplots(1, ntasks, figsize=(3.5 * ntasks, fig_h))
    if ntasks == 1:
        axes = [axes]
    fig.suptitle(
        f"L6  Task-Adaptive Modality Gate — Patient {pid}\n"
        "Gate value ≈ 1 = modality trusted, ≈ 0 = suppressed before TemporalSAB",
        fontsize=FONT_TITLE, y=1.03)

    mod_lbls = model._mod_order

    # For tall patients skip annotation text and show fewer y-ticks
    tick_step = max(1, T // 20)
    show_ticks = list(range(0, T, tick_step))
    if T - 1 not in show_ticks:
        show_ticks.append(T - 1)
    biopsy_tick_lbls = [f"B{t+1} {int(days[t])}d" for t in show_ticks]
    annotate = T <= 25  # skip per-cell text for very tall grids

    # Compute global vmin/vmax across all tasks so colormap is comparable
    all_gate_vals = np.concatenate([gate_d[t].flatten() for t in tasks if gate_d.get(t) is not None])
    g_lo = np.percentile(all_gate_vals, 1)
    g_hi = np.percentile(all_gate_vals, 99)
    spread = g_hi - g_lo
    if spread < 0.04:   # nearly uniform — widen range to 0.04 centred on mean
        g_mid = all_gate_vals.mean()
        g_lo  = max(0.0, g_mid - 0.02)
        g_hi  = min(1.0, g_mid + 0.02)

    for ax, task in zip(axes, tasks):
        gate = gate_d.get(task)         # (T, n_mods)
        if gate is None:
            ax.set_visible(False)
            continue

        im = ax.imshow(gate, aspect="auto", cmap="YlGn",
                       vmin=g_lo, vmax=g_hi, interpolation="nearest")
        g_mid_thresh = (g_lo + g_hi) / 2
        if annotate:
            for r in range(T):
                for c in range(len(mod_lbls)):
                    ax.text(c, r, f"{gate[r, c]:.2f}", ha="center", va="center",
                            fontsize=7, fontweight="bold",
                            color="white" if gate[r, c] > g_mid_thresh else "black")

        ax.set_xticks(range(len(mod_lbls)))
        ax.set_xticklabels(mod_lbls, fontsize=9, fontweight="bold")
        ax.set_yticks(show_ticks)
        ax.set_yticklabels(biopsy_tick_lbls, fontsize=7)
        ax.set_title(TASK_LABELS.get(task, task), fontsize=9,
                     color=TASK_COLORS.get(task, "#555"))
        # Modality color ticks
        for ci, mo in enumerate(mod_lbls):
            ax.get_xticklabels()[ci].set_color(MOD_COLORS.get(mo, "black"))

        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(f"Gate weight [{g_lo:.3f}–{g_hi:.3f}]", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    plt.tight_layout()
    png = _savefig(fig, out_dir, f"L6_task_gate_pid{pid}")
    plt.close(fig)
    print(f"  [L6] patient={pid}  → {png.name}")
    return png


# ── Panel L7: Modality contribution to ABMIL attention per biopsy ────────────

def plot_L7_mod_contrib(extr: dict, model: nn.Module, out_dir: Path, tasks: List[str]):
    """
    Stacked area chart: fraction of total ABMIL α attributable to each modality,
    per biopsy per task.  Shows how the model's reliance on each modality shifts
    over the transplant timeline.
    """
    pid    = extr["patient_id"]
    days   = extr["biopsy_days"]
    T      = extr["n_biopsies"]
    tok_b  = extr["tok_biopsy"]
    tok_m  = extr["tok_mod_name"]

    ntasks = len(tasks)
    fig, axes = plt.subplots(1, ntasks, figsize=(min(4.5 * ntasks, 18), 4.5), sharey=False)
    if ntasks == 1:
        axes = [axes]
    fig.suptitle(
        f"L7  Modality Contribution to ABMIL α — Patient {pid}  ({T} biopsies)\n"
        "Fraction of biopsy attention attributable to each modality  (stacked bars, ordered by time)",
        fontsize=FONT_TITLE, fontweight="bold", y=1.02)

    days_arr = np.asarray(days, dtype=float)

    for ax, task in zip(axes, tasks):
        alpha = extr["alpha_per_task"].get(task)
        if alpha is None:
            ax.set_visible(False)
            continue

        # Per biopsy, per modality: sum of α over tokens belonging to that (biopsy, mod)
        mod_biopsy_alpha = {mo: np.zeros(T) for mo in MOD_ORDER}
        for i, (b, mo) in enumerate(zip(tok_b, tok_m)):
            if mo in mod_biopsy_alpha:
                mod_biopsy_alpha[mo][b] += alpha[i]

        # Normalize to fractions within each biopsy
        total_per_biopsy = sum(mod_biopsy_alpha[mo] for mo in MOD_ORDER)
        total_per_biopsy = np.where(total_per_biopsy > 0, total_per_biopsy, 1.0)
        fracs = {mo: mod_biopsy_alpha[mo] / total_per_biopsy for mo in MOD_ORDER}

        # Bar widths proportional to inter-biopsy intervals (days-aligned x-axis)
        if T > 1:
            gaps = np.diff(days_arr)
            # Each bar fills ~85% of the gap before the next biopsy
            bar_widths = np.append(gaps, gaps[-1]) * 0.85
        else:
            bar_widths = np.array([30.0])
        bar_widths = np.clip(bar_widths, 1.0, None)

        xs     = days_arr
        bottom = np.zeros(T)
        for mo in MOD_ORDER:
            frac = fracs[mo]
            mask = frac > 0.005
            ax.bar(xs[mask], frac[mask], bottom=bottom[mask], width=bar_widths[mask],
                   color=MOD_COLORS[mo], alpha=0.88, label=mo, edgecolor="none",
                   align="edge")
            bottom += frac

        # Horizontal guide at 50%
        ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)

        # X-axis: actual days with minimum spacing to prevent label crowding
        day_range = days_arr[-1] - days_arr[0]
        min_day_gap = max(30.0, day_range / 12)   # at most ~12 ticks, min 30-day gap
        tick_idx = [0]
        for t in range(1, T - 1):
            if days_arr[t] - days_arr[tick_idx[-1]] >= min_day_gap:
                tick_idx.append(t)
        if T - 1 not in tick_idx:
            tick_idx.append(T - 1)
        tick_days = [days_arr[t] for t in tick_idx]
        tick_lbls = [f"B{t+1}\n{int(days_arr[t])}d" for t in tick_idx]

        ax.set_xlim(days_arr[0] - max(15, 0.01 * day_range),
                    days_arr[-1] + max(20, 0.03 * day_range))
        ax.set_ylim(0, 1)
        ax.set_xticks(tick_days)
        ax.set_xticklabels(tick_lbls, fontsize=8)
        ax.set_xlabel("Days from transplant  (→)", fontsize=FONT_LABEL)
        ax.set_ylabel("Fraction of ABMIL α", fontsize=FONT_LABEL)

        # Mark ACR events
        records = extr.get("records", [])
        for t, rec in enumerate(records):
            if rec.get("label") == 1:
                ax.axvline(days_arr[t], color="#e53935", lw=0.9, ls="--", alpha=0.45, zorder=0)
        ax.set_title(TASK_LABELS.get(task, task), fontsize=10,
                     color=TASK_COLORS.get(task, "#555"), fontweight="bold")

    # Shared legend on last axis
    handles = [mpatches.Patch(color=MOD_COLORS[mo], label=mo) for mo in MOD_ORDER]
    axes[-1].legend(handles=handles, fontsize=8, loc="upper right",
                    framealpha=0.85, title="Modality", title_fontsize=8)

    plt.tight_layout()
    png = _savefig(fig, out_dir, f"L7_mod_contrib_pid{pid}")
    plt.close(fig)
    print(f"  [L7] patient={pid}  → {png.name}")
    return png


# ── Panel L8: Seed temporal trend (correlation with day) ─────────────────────

def plot_L8_seed_trend(extr: dict, model: nn.Module, out_dir: Path):
    """
    For each modality × seed, compute Spearman rank correlation of PMA seed L2 norm
    with biopsy day.  Shows whether a seed concept is rising or falling over time.
    Displayed as a heatmap (seed × sign of trend) + bar chart of top-trending seeds.
    """
    from scipy.stats import spearmanr

    pid   = extr["patient_id"]
    T     = extr["n_biopsies"]
    days  = np.array(extr["biopsy_days"])
    norms = extr["seeds_norms_grid"]   # (T, n_mods, K)
    K     = model.n_seeds

    # Only include modalities with at least 4 present biopsies (needed for Spearman)
    _MIN_OBS = 4
    present_mods = []
    for mi, mo in enumerate(MOD_ORDER):
        col = norms[:, mi, :]
        n_present = int((~np.all(np.isnan(col), axis=1)).sum())
        if n_present >= _MIN_OBS:
            present_mods.append((mi, mo, n_present))
    if not present_mods:
        return None

    # Compute Spearman r between day and L2 norm for each seed
    results_rho = {}
    for mi, mo, n_obs in present_mods:
        rhos = []
        for ki in range(K):
            series = norms[:, mi, ki]
            valid  = ~np.isnan(series)
            if valid.sum() >= _MIN_OBS:
                rho, pval = spearmanr(days[valid], series[valid])
                rhos.append(rho)
            else:
                rhos.append(np.nan)
        results_rho[(mi, mo, n_obs)] = np.array(rhos)

    n_mods = len(present_mods)
    fig, axes = plt.subplots(1, n_mods, figsize=(min(4.5 * n_mods, 18), 4.5),
                             gridspec_kw={"wspace": 0.45})
    if n_mods == 1:
        axes = [axes]
    n_obs_str = ", ".join(f"{mo}={n}" for _, mo, n in present_mods)
    fig.suptitle(
        f"L8  Seed Temporal Trend — Patient {pid}  ({n_obs_str} biopsies per modality)\n"
        "Spearman ρ: biopsy day vs PMA seed L2 norm  ·  red = rising over time · blue = falling",
        fontsize=FONT_TITLE, fontweight="bold", y=1.02)

    for ax, (mi, mo, n_obs) in zip(axes, present_mods):
        rhos = results_rho[(mi, mo, n_obs)]
        valid = ~np.isnan(rhos)
        xs = np.arange(K)
        colors = ["#d62728" if r > 0 else "#1f77b4" for r in rhos]
        bars = ax.bar(xs[valid], rhos[valid],
                      color=[colors[i] for i in range(K) if valid[i]],
                      edgecolor="white", linewidth=0.6, width=0.75)
        ax.axhline(0, color="#555", lw=0.8)
        ax.axhline(0.5, color="#d62728", lw=0.6, ls="--", alpha=0.5)
        ax.axhline(-0.5, color="#1f77b4", lw=0.6, ls="--", alpha=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"s{ki:02d}" for ki in range(K)],
                           fontsize=8, rotation=45, ha="right",
                           color=MOD_COLORS[mo], fontweight="bold")
        ax.set_ylabel("Spearman ρ (day vs norm)", fontsize=9)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{mo}  (n={n_obs} biopsies)", fontsize=11,
                     color=MOD_COLORS[mo], fontweight="bold")
        # Annotate top-2 and bottom-2 seeds; offset from zero (not bar tip) to avoid crowding
        top2 = np.argsort(np.where(valid, rhos, -99))[-2:]
        bot2 = np.argsort(np.where(valid, rhos,  99))[:2]
        _label_y_pos = 0.13   # fixed distance above zero for positive labels
        _label_y_neg = -0.13  # fixed distance below zero for negative labels
        for k in list(top2) + list(bot2):
            if not valid[k]:
                continue
            y_txt = _label_y_pos if rhos[k] >= 0 else _label_y_neg
            ax.text(k, y_txt, f"s{k:02d}", ha="center",
                    fontsize=8, fontweight="bold",
                    va="bottom" if rhos[k] >= 0 else "top",
                    color="#d62728" if rhos[k] > 0 else "#1f77b4")

    plt.tight_layout()
    png = _savefig(fig, out_dir, f"L8_seed_trend_pid{pid}")
    plt.close(fig)
    print(f"  [L8] patient={pid}  → {png.name}")
    return png


# ── Patient summary figure ────────────────────────────────────────────────────

def plot_patient_summary(extr: dict, model: nn.Module, out_dir: Path, tasks: List[str]):
    """
    One-page summary per patient.  All three time-axis panels share the same
    x-axis (days from transplant) so vertical alignment is exact.
    """
    pid     = extr["patient_id"]
    days    = np.array(extr["biopsy_days"], dtype=float)
    T       = extr["n_biopsies"]
    records = extr["records"]
    logits  = extr["logits"]

    # Use actual day positions for bars; compute a sensible bar half-width
    day_span   = float(days[-1] - days[0]) if T > 1 else 1.0
    med_gap    = float(np.median(np.diff(days))) if T > 1 else day_span
    bar_hw     = max(8.0, med_gap * 0.14)   # half-width of each grouped bar

    # Build figure with sharex between top-left, bottom-left, bottom-right
    fig = plt.figure(figsize=(14, 7))
    ax00 = fig.add_subplot(2, 2, 1)                    # modality timeline
    ax01 = fig.add_subplot(2, 2, 2)                    # hazard scores (own x)
    ax10 = fig.add_subplot(2, 2, 3, sharex=ax00)      # hazard trajectory
    ax11 = fig.add_subplot(2, 2, 4, sharex=ax00)      # biopsy ABMIL α
    fig.suptitle(f"Patient {pid} — Longitudinal Summary  ({T} biopsies)",
                 fontsize=14, fontweight="bold")

    # Shared ACR label vertical lines helper
    def _draw_acr_vlines(ax):
        for t, rec in enumerate(records):
            lbl = rec.get("label")
            if lbl is not None:
                ax.axvline(days[t], color="#e53935" if lbl else "#43a047",
                           lw=1.2, ls="--", alpha=0.45, zorder=0)

    # ── 1. Modality timeline (top-left, defines shared x) ────────────────────
    for mi, mo in enumerate(MOD_ORDER):
        for t in range(T):
            has_mod = (t, mo) in extr["pma_attn"]
            ax00.scatter(days[t], mi,
                         color=MOD_COLORS[mo] if has_mod else "#ccc",
                         s=80 if has_mod else 40, zorder=3 if has_mod else 2,
                         marker="s" if has_mod else "x",
                         edgecolors="white" if has_mod else "none",
                         linewidths=0.5)
    ax00.set_yticks(range(len(MOD_ORDER)))
    ax00.set_yticklabels(MOD_ORDER, fontsize=10)
    for mi, mo in enumerate(MOD_ORDER):
        ax00.get_yticklabels()[mi].set_color(MOD_COLORS[mo])
    ax00.set_title("Modality Availability  (■ = present, × = absent)", fontsize=10)
    plt.setp(ax00.get_xticklabels(), visible=False)   # bottom panels carry the label
    _draw_acr_vlines(ax00)

    # ── 2. Final hazard scores (top-right, independent x) ────────────────────
    task_lbls   = [TASK_LABELS.get(t, t) for t in tasks]
    hazard_vals = [logits.get(t, 0.0) for t in tasks]
    cols        = [TASK_COLORS.get(t, "#777") for t in tasks]
    bars = ax01.barh(task_lbls, hazard_vals, color=cols, edgecolor="white",
                     linewidth=0.8, height=0.55)
    ax01.axvline(0, color="grey", lw=1.0, ls=":")
    xspan = max(abs(v) for v in hazard_vals) * 1.5 + 0.01
    for bar, val in zip(bars, hazard_vals):
        offset = xspan * 0.04
        ax01.text(val + (offset if val >= 0 else -offset),
                  bar.get_y() + bar.get_height() / 2,
                  f"{val:.3f}", va="center",
                  ha="left" if val >= 0 else "right",
                  fontsize=9, fontweight="bold")
    ax01.set_xlabel("Final Log-Risk Score", fontsize=FONT_LABEL)
    ax01.set_title("Final Hazard Scores (full timeline)", fontsize=10)
    ax01.invert_yaxis()
    ax01.set_xlim(-xspan, xspan)

    # ── 3. Hazard trajectory — plotted on actual biopsy days ─────────────────
    for task in tasks:
        hvals = extr["hazard_traj"].get(task, [])
        if not hvals:
            continue
        n = min(len(hvals), T)
        col = TASK_COLORS.get(task, "#777")
        ax10.plot(days[:n], hvals[:n], "o-", color=col, lw=1.8, ms=5,
                  label=TASK_LABELS.get(task, task))
    ax10.set_ylabel("Hazard Score", fontsize=FONT_LABEL)
    ax10.set_xlabel("Days from transplant", fontsize=FONT_LABEL)
    ax10.set_title("Hazard Trajectory (causal)", fontsize=10)
    ax10.legend(fontsize=8, framealpha=0.9)
    ax10.axhline(0, color="grey", lw=0.6, ls=":")
    _draw_acr_vlines(ax10)

    # ── 4. ABMIL biopsy contribution — bars at actual day positions ───────────
    n_tasks = len(tasks)
    for ti, task in enumerate(tasks):
        alpha = extr["alpha_per_task"].get(task)
        if alpha is None:
            continue
        biopsy_alpha = np.zeros(T, dtype=np.float32)
        for i, b in enumerate(extr["tok_biopsy"]):
            biopsy_alpha[b] += alpha[i]
        offset = (ti - n_tasks / 2 + 0.5) * bar_hw * 2
        col = TASK_COLORS.get(task, "#777")
        ax11.bar(days + offset, biopsy_alpha, width=bar_hw * 1.8,
                 color=col, alpha=0.82, label=TASK_LABELS.get(task, task),
                 edgecolor="none")
    ax11.set_ylabel("Total ABMIL α", fontsize=FONT_LABEL)
    ax11.set_xlabel("Days from transplant", fontsize=FONT_LABEL)
    ax11.set_title("Biopsy Contribution per Task", fontsize=10)
    ax11.legend(fontsize=7.5, framealpha=0.9, ncol=2)
    _draw_acr_vlines(ax11)

    # Shared x-axis: add sparse biopsy index ticks on ax10 (visible) and ax11
    # Use min-gap approach so dense early biopsies don't overlap
    day_range_l0 = float(days[-1] - days[0]) if T > 1 else 1.0
    min_gap_l0   = max(30.0, day_range_l0 / 12)
    tick_idx_l0  = [0]
    for t in range(1, T - 1):
        if days[t] - days[tick_idx_l0[-1]] >= min_gap_l0:
            tick_idx_l0.append(t)
    if T - 1 not in tick_idx_l0:
        tick_idx_l0.append(T - 1)
    shared_tick_days = [days[t] for t in tick_idx_l0]
    shared_tick_lbls = [f"B{t+1}\n{int(days[t])}d" for t in tick_idx_l0]
    for ax_shared in (ax10, ax11):
        ax_shared.set_xticks(shared_tick_days)
        ax_shared.set_xticklabels(shared_tick_lbls, fontsize=7.5)

    plt.tight_layout()
    png = _savefig(fig, out_dir, f"L0_summary_pid{pid}")
    plt.close(fig)
    print(f"  [summary] patient={pid}  → {png.name}")
    return png


# ── Population-level: aggregate across all test patients ─────────────────────

def plot_population_seed_trends(all_extractions: List[dict], model: nn.Module,
                                out_dir: Path):
    """
    Population-level heatmap: Spearman ρ (biopsy day vs PMA seed L2 norm) for each
    patient × modality × seed.  Reveals which seed concepts rise or fall consistently
    across the cohort vs which are patient-specific.
    Rows = seeds (grouped by modality), Columns = patients (sorted by n_biopsies).
    """
    from scipy.stats import spearmanr

    _MIN_OBS = 4
    K = model.n_seeds

    # Collect ρ matrix: {(mi, mo): (n_patients, K)} — NaN where patient lacks modality
    n_pats = len(all_extractions)
    pid_list = [e["patient_id"] for e in all_extractions]
    # Sort patients by n_biopsies desc for visual ordering
    order_p = sorted(range(n_pats), key=lambda i: -all_extractions[i]["n_biopsies"])
    pid_sorted = [pid_list[i] for i in order_p]

    rho_mats = {}  # (mi, mo) -> (n_patients, K) array
    for mi, mo in enumerate(MOD_ORDER):
        mat = np.full((n_pats, K), np.nan)
        for pi, extr in enumerate(all_extractions):
            days  = np.array(extr["biopsy_days"])
            norms = extr["seeds_norms_grid"]   # (T, n_mods, K)
            col   = norms[:, mi, :]            # (T, K)
            n_present = int((~np.all(np.isnan(col), axis=1)).sum())
            if n_present < _MIN_OBS:
                continue
            for ki in range(K):
                series = col[:, ki]
                valid  = ~np.isnan(series)
                if valid.sum() >= _MIN_OBS:
                    rho, _ = spearmanr(days[valid], series[valid])
                    mat[pi, ki] = rho
        rho_mats[(mi, mo)] = mat

    # Only include modalities with at least 5 patients having data
    present = [(mi, mo) for (mi, mo), mat in rho_mats.items()
               if (~np.isnan(mat)).any(axis=1).sum() >= 5]
    if not present:
        return

    n_mods = len(present)
    # Build combined row matrix: (n_mods * K, n_patients)
    row_labels, row_colors = [], []
    big_mat = []
    for mi, mo in present:
        mat = rho_mats[(mi, mo)][order_p, :]   # (n_patients, K) reordered
        for ki in range(K):
            row_labels.append(f"{mo}·s{ki:02d}")
            row_colors.append(MOD_COLORS[mo])
            big_mat.append(mat[:, ki])
    big_mat = np.stack(big_mat)   # (n_rows, n_patients)

    n_rows = len(row_labels)
    fig_h = max(5.0, n_rows * 0.28 + 1.5)
    fig_w = max(8.0, n_pats * 0.22 + 2.0)
    fig, ax = plt.subplots(figsize=(min(fig_w, 22), min(fig_h, 14)))

    im = ax.imshow(big_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto",
                   interpolation="nearest")

    # Modality block separators
    n_rows_total = len(present) * K
    cumk = 0
    for bi, (mi, mo) in enumerate(present):
        # Modality label to the right of the heatmap, vertically centred on block
        frac_y_lo = 1.0 - (cumk + K) / n_rows_total
        frac_y_hi = 1.0 - cumk / n_rows_total
        ax.annotate(mo, xy=(1.01, (frac_y_lo + frac_y_hi) / 2),
                    xycoords="axes fraction", ha="left", va="center",
                    fontsize=9, fontweight="bold", color=MOD_COLORS[mo])
        cumk += K
        if bi < len(present) - 1:
            ax.axhline(cumk - 0.5, color="#222222", lw=2.5, zorder=5)

    # Mean ρ per row — embed directly in y-tick labels (no twinx conflict with colorbar)
    mean_rho = np.nanmean(big_mat, axis=1)
    lbl_fs = max(6, min(8, int(200 / n_rows)))
    ytick_labels = []
    for ri, (lbl, mr) in enumerate(zip(row_labels, mean_rho)):
        if np.isnan(mr):
            ytick_labels.append(lbl)
        else:
            ytick_labels.append(f"{lbl}  {mr:+.2f}")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(ytick_labels, fontsize=lbl_fs, fontfamily="monospace")
    for tick, col in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(col)

    # X-axis: all patient IDs
    x_step = max(1, n_pats // 20)
    ax.set_xticks(range(0, n_pats, x_step))
    ax.set_xticklabels(pid_sorted[::x_step], rotation=55, ha="right", fontsize=7)
    ax.set_xlabel("Patient  (sorted by n_biopsies desc)", fontsize=9)
    ax.set_ylabel("PMA seed  (mean ρ shown in label)", fontsize=9)

    # Horizontal colorbar below the heatmap — no right-side conflict
    cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                      fraction=0.03, pad=0.12, shrink=0.6, aspect=35)
    cb.set_label("Spearman ρ (biopsy day vs PMA seed L2 norm)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Lpop  Population Seed Temporal Trends — N={n_pats} patients\n"
        "Spearman ρ: biopsy day vs PMA seed L2 norm  ·  red = rising · blue = falling · grey = insufficient data",
        fontsize=FONT_TITLE, fontweight="bold", y=1.01)

    plt.tight_layout()
    png = _savefig(fig, out_dir, "Lpop_seed_trends")
    plt.close(fig)
    print(f"  [pop_seed_trends] → {png.name}")
    return png


def plot_population_alpha(all_extractions: List[dict], model: nn.Module,
                          out_dir: Path, tasks: List[str]):
    """
    Scatter: for each patient, x = normalized biopsy position (0=first, 1=last),
    y = ABMIL α, coloured by task. Shows population-level recency bias.
    """
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.5 * len(tasks), 4))
    if len(tasks) == 1:
        axes = [axes]
    fig.suptitle("Population-Level Recency ABMIL Weights\n"
                 "x = normalised biopsy position, y = ABMIL α  (N test patients)",
                 fontsize=FONT_TITLE)

    for ax, task in zip(axes, tasks):
        col = TASK_COLORS.get(task, "#777")
        all_pos, all_alp = [], []

        for extr in all_extractions:
            T     = extr["n_biopsies"]
            alpha = extr["alpha_per_task"].get(task)
            if alpha is None:
                continue
            biopsy_alpha = np.zeros(T, dtype=np.float32)
            for i, b in enumerate(extr["tok_biopsy"]):
                biopsy_alpha[b] += alpha[i]
            for t in range(T):
                pos = t / max(T - 1, 1)
                all_pos.append(pos)
                all_alp.append(float(biopsy_alpha[t]))

        if not all_pos:
            ax.set_visible(False)
            continue

        all_pos = np.array(all_pos)
        all_alp = np.array(all_alp)
        ax.scatter(all_pos, all_alp, color=col, alpha=0.3, s=14, edgecolors="none")

        # Smoothed rolling mean
        sort_i = np.argsort(all_pos)
        w = max(5, len(all_pos) // 20)
        pad_alp = np.pad(all_alp[sort_i], w // 2, mode="edge")
        smooth = np.convolve(pad_alp, np.ones(w) / w, mode="valid")[:len(all_pos)]
        ax.plot(all_pos[sort_i], smooth, color=col, lw=3.0, label="rolling mean")

        # Linear regression slope — shows overall recency direction
        slope, intercept = np.polyfit(all_pos, all_alp, 1)
        xs_fit = np.array([0.0, 1.0])
        ax.plot(xs_fit, slope * xs_fit + intercept,
                color=col, lw=1.2, ls="--", alpha=0.7)

        ax.set_xlabel("Normalised biopsy position  (0=first, 1=last)", fontsize=9)
        ax.set_ylabel("ABMIL α weight", fontsize=9)
        ax.set_title(TASK_LABELS.get(task, task), fontsize=10, color=col)
        gamma_v = model.recency_gamma[task].abs().item()
        slope_sign = "↑" if slope > 0 else "↓"
        ax.text(0.03, 0.97, f"γ={gamma_v:.3f}\nslope={slope:.4f} {slope_sign}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, alpha=0.8))

    plt.tight_layout()
    png = _savefig(fig, out_dir, "Lpop_alpha_recency")
    plt.close(fig)
    print(f"  [pop_alpha] → {png.name}")
    return png


# ── Lpop_K: Population seed attribution chain ────────────────────────────────

def plot_population_seed_attribution(all_extractions: List[dict],
                                     cluster_names_pool: Dict[str, List[str]],
                                     model: nn.Module, out_dir: Path,
                                     tasks: List[str]):
    """
    Lpop_K: Which seeds drive predictions, and what instances do they attend to?

    For each task:
      Row 0 — mean ABMIL α per (mod, seed_k) for high-risk vs low-risk patients
              + signed difference bar (red=enriched in high-risk, blue=low-risk)
      Row 1 — per-modality: ABMIL-weighted cluster affinity of top-5 differential
              seeds, averaged separately over high- and low-risk groups.

    Token → seed mapping:  tokens are stored in blocks of K per (biopsy, modality).
    We average alpha over biopsies so each patient contributes one (M*K,) vector.
    """
    K = model.n_seeds

    # Determine which modalities appear in ≥3 patients
    present_mods = [mo for mo in MOD_ORDER
                    if sum(any((ti, mo) in extr["pma_attn"]
                               for ti in range(extr["n_biopsies"]))
                           for extr in all_extractions) >= 3]
    if not present_mods:
        return
    total_seeds = len(present_mods) * K

    ep_keys = {
        "acr_cls":    "label",
        "acr_surv":   "logits",
        "clad":       "logits",
        "death":      "logits",
    }

    for task in tasks:
        outcome_src = ep_keys.get(task, "logits")

        # ── Per-patient: mean alpha over biopsies per (mod, seed_k) ──────────
        canonical_alphas, outcomes, valid_extrs = [], [], []
        for extr in all_extractions:
            alpha = extr["alpha_per_task"].get(task)
            if alpha is None:
                continue
            out_val = (float(extr["records"][0].get("label", float("nan")))
                       if outcome_src == "label"
                       else extr["logits"].get(task, float("nan")))
            if np.isnan(out_val):
                continue

            tok_biopsy   = extr["tok_biopsy"]
            tok_mod_name = extr["tok_mod_name"]
            mod_seed_sum   = {mo: np.zeros(K, dtype=np.float64) for mo in present_mods}
            mod_seed_count = {mo: 0 for mo in present_mods}

            # Walk token blocks (each block = K tokens for one biopsy×mod)
            i = 0
            n_tok = len(alpha)
            while i < n_tok:
                mo = tok_mod_name[i]
                if mo in present_mods and i + K <= n_tok:
                    mod_seed_sum[mo]   += alpha[i:i + K]
                    mod_seed_count[mo] += 1
                i += K

            can = []
            for mo in present_mods:
                cnt = mod_seed_count[mo]
                can.extend(mod_seed_sum[mo] / cnt if cnt > 0 else [0.0] * K)
            canonical_alphas.append(np.array(can, dtype=np.float32))
            outcomes.append(out_val)
            valid_extrs.append(extr)

        if len(canonical_alphas) < 6:
            continue
        alphas   = np.stack(canonical_alphas)   # (N, total_seeds)
        outcomes = np.array(outcomes)

        # Split into high / low risk
        if task == "acr_cls":
            hi_m = outcomes == 1
            lo_m = outcomes == 0
            hi_lbl, lo_lbl = "ACR+", "ACR−"
        else:
            med  = np.median(outcomes)
            hi_m = outcomes >= med
            lo_m = outcomes <  med
            hi_lbl, lo_lbl = "High risk", "Low risk"

        if hi_m.sum() < 3 or lo_m.sum() < 3:
            continue

        alpha_hi   = alphas[hi_m].mean(0)
        alpha_lo   = alphas[lo_m].mean(0)
        alpha_diff = alpha_hi - alpha_lo

        # Seed labels and modality spans
        seed_labels, seed_colors, mod_spans = [], [], {}
        for mo in present_mods:
            s = len(seed_labels)
            for k in range(K):
                seed_labels.append(f"{mo[:3]}·s{k:02d}")
                seed_colors.append(MOD_COLORS.get(mo, "#888"))
            mod_spans[mo] = (s, s + K)

        # ── Figure ────────────────────────────────────────────────────────────
        n_mods = len(present_mods)
        fig = plt.figure(figsize=(max(16, total_seeds * 0.23), 4 + 3.8 * n_mods))
        gs_outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.5,
                                     height_ratios=[2, n_mods * 3.8])

        # Row 0: alpha bars
        gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0],
                                                  width_ratios=[2, 2, 1], wspace=0.1)
        x = np.arange(total_seeds)
        for col_i, (vals, title) in enumerate([
                (alpha_lo, f"{lo_lbl}  (n={lo_m.sum()})"),
                (alpha_hi, f"{hi_lbl}  (n={hi_m.sum()})")]):
            ax = fig.add_subplot(gs_top[col_i])
            ax.bar(x, vals, color=seed_colors, width=0.85, alpha=0.8)
            for mo in present_mods[1:]:
                ax.axvline(mod_spans[mo][0] - 0.5, color="#aaa", lw=0.7, ls="--")
            for mo in present_mods:
                mid = (mod_spans[mo][0] + mod_spans[mo][1]) / 2
                ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.01,
                        mo, ha="center", va="bottom", fontsize=7,
                        color=MOD_COLORS.get(mo, "#888"), fontweight="bold")
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_xticks([]); ax.set_ylabel("Mean ABMIL α", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)

        ax_diff = fig.add_subplot(gs_top[2])
        diff_cols = ["#E53935" if v > 0 else "#1E88E5" for v in alpha_diff]
        ax_diff.bar(x, alpha_diff, color=diff_cols, width=0.85, alpha=0.85)
        ax_diff.axhline(0, color="#333", lw=0.8)
        ax_diff.set_title(f"Δα  ({hi_lbl}−{lo_lbl})", fontsize=9, fontweight="bold")
        ax_diff.set_xticks([]); ax_diff.set_ylabel("Δ Mean α", fontsize=8)
        ax_diff.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Lpop_K — Seed attribution: {task} | N={len(canonical_alphas)} patients",
                     fontsize=11, fontweight="bold")

        # Row 1: per-mod cluster affinity for top differential seeds
        gs_bot = gridspec.GridSpecFromSubplotSpec(n_mods, 2, subplot_spec=gs_outer[1],
                                                  hspace=0.55, wspace=0.4)

        for mi, mo in enumerate(present_mods):
            s0, s1 = mod_spans[mo]
            mod_diff_vals = alpha_diff[s0:s1]      # (K,) seed diffs for this mod
            top_seeds = np.argsort(np.abs(mod_diff_vals))[::-1][:5]

            cnames = cluster_names_pool.get(mo, [])

            for col_i, (mask, grp_lbl, extrs_g) in enumerate([
                    (lo_m, lo_lbl, [e for e, m in zip(valid_extrs, hi_m) if not m]),
                    (hi_m, hi_lbl, [e for e, m in zip(valid_extrs, hi_m) if m])]):

                ax = fig.add_subplot(gs_bot[mi, col_i])
                if not extrs_g:
                    ax.set_visible(False); continue

                # Collect ABMIL-weighted PMA cluster affinity per seed
                # weighted_aff[k, c] = mean over patients of alpha_k * pma_attn[k, c]
                aff_acc  = np.zeros((K, max(1, len(cnames))), dtype=np.float64)
                aff_cnt  = np.zeros((K, max(1, len(cnames))), dtype=np.float64)

                for extr in extrs_g:
                    pat_alpha = extr["alpha_per_task"].get(task)
                    if pat_alpha is None:
                        continue
                    tok_mod_name_e = extr["tok_mod_name"]
                    # Collect per-seed alpha (averaged over biopsies) for this mod
                    seed_sum = np.zeros(K); seed_cnt = 0
                    i = 0
                    n_t = len(pat_alpha)
                    while i < n_t:
                        if tok_mod_name_e[i] == mo and i + K <= n_t:
                            seed_sum += pat_alpha[i:i + K]
                            seed_cnt += 1
                        i += K
                    if seed_cnt == 0:
                        continue
                    seed_w = seed_sum / seed_cnt  # (K,) mean alpha for this mod

                    # Average pma_attn over biopsies for this modality
                    pma_sum = None; pma_cnt = 0
                    for t_idx in range(extr["n_biopsies"]):
                        pa = extr["pma_attn"].get((t_idx, mo))
                        if pa is None:
                            continue
                        n_c = pa.shape[1]
                        if pma_sum is None:
                            pma_sum = np.zeros((K, n_c))
                        if pa.shape == pma_sum.shape:
                            pma_sum += pa
                            pma_cnt += 1
                    if pma_sum is None or pma_cnt == 0:
                        continue
                    pma_mean = pma_sum / pma_cnt   # (K, n_clus)
                    n_c = pma_mean.shape[1]
                    if aff_acc.shape[1] != n_c:
                        aff_acc = np.zeros((K, n_c)); aff_cnt = np.zeros((K, n_c))

                    # ABMIL-weighted affinity: alpha_k × pma_k,c
                    weighted = seed_w[:, None] * pma_mean   # (K, n_c)
                    aff_acc  += weighted
                    aff_cnt  += 1

                if aff_cnt.max() == 0:
                    ax.set_visible(False); continue

                mean_aff = np.where(aff_cnt > 0, aff_acc / aff_cnt, 0)  # (K, n_c)
                top_aff  = mean_aff[top_seeds]    # (5, n_c)
                n_c      = top_aff.shape[1]
                clus_nms = [cnames[c][:20] if c < len(cnames) else str(c)
                            for c in range(n_c)]

                im = ax.imshow(top_aff, aspect="auto", cmap="YlOrRd",
                               vmin=0, vmax=np.percentile(top_aff, 98).clip(1e-8))
                ax.set_xticks(range(n_c))
                ax.set_xticklabels(clus_nms, rotation=55, ha="right", fontsize=5)
                ax.set_yticks(range(len(top_seeds)))
                ax.set_yticklabels(
                    [f"s{top_seeds[j]:02d} (Δ={mod_diff_vals[top_seeds[j]]:+.4f})"
                     for j in range(len(top_seeds))], fontsize=6)
                ax.set_title(f"{mo} — {grp_lbl} | top-5 Δseeds",
                             fontsize=8, color=MOD_COLORS.get(mo, "#888"),
                             fontweight="bold")
                plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02).ax.tick_params(labelsize=6)

                # HE category shading on x-axis
                if mo == "HE" and HE_BIO_MAP and cnames:
                    prev_cat, cat_start = None, 0
                    for ci in range(n_c):
                        cat = HE_BIO_MAP.get(cnames[ci] if ci < len(cnames) else "", "Unknown")
                        if cat != prev_cat and prev_cat is not None:
                            ax.axvspan(cat_start - 0.5, ci - 0.5, alpha=0.07,
                                       color=HE_BIO_COLORS.get(prev_cat, "#ccc"), zorder=0)
                        prev_cat  = cat
                        cat_start = ci

        png = _savefig(fig, out_dir, f"Lpop_K_seed_attribution_{task}")
        plt.close(fig)
        print(f"  [pop_K] {task} → {png.name}")


# ── W&B logging ───────────────────────────────────────────────────────────────

def log_to_wandb(model: nn.Module, all_extractions: List[dict],
                 tasks: List[str], out_dir: Path,
                 split: int, fold: int, project: str):
    """
    Comprehensive W&B logging for longitudinal-MK-MT interpretability.

    Logs:
      • Model-level scalars: ALiBi slopes, recency γ per task
      • Per-patient table: n_biopsies, hazard per task, dominant biopsy, ACR label
      • Hazard trajectory line series (wandb.plot)
      • ABMIL biopsy-weight population summary (scatter)
      • All panel PNGs organised by panel type
    """
    try:
        import wandb
    except ImportError:
        print("  [wandb] wandb not installed — skipping")
        return

    fold_tag = f"split{split}_fold{fold}"

    # Learned model parameters
    slopes = model.temporal_sab.alibi_slopes.abs().detach().cpu().numpy()
    gammas = {t: model.recency_gamma[t].abs().item() for t in tasks}

    config = {
        "variant":   "longitudinal_mk_mt",
        "split":     split,
        "fold":      fold,
        "tasks":     tasks,
        "n_seeds":   model.n_seeds,
        "n_patients": len(all_extractions),
        **{f"alibi_slope_h{i}": float(slopes[i]) for i in range(len(slopes))},
        **{f"recency_gamma_{t}": float(gammas[t]) for t in tasks},
    }

    try:
        run = wandb.init(
            project=project,
            name=f"longitudinal_mk_mt_{fold_tag}",
            group="longitudinal_mk_mt",
            config=config,
            reinit=True,
        )
    except Exception as e:
        print(f"  [wandb] init failed: {e}")
        return

    # ── Scalars: model properties ─────────────────────────────────────────────
    scalar_log = {}
    for i, s in enumerate(slopes):
        scalar_log[f"model/alibi_slope_h{i}"] = float(s)
    for t in tasks:
        scalar_log[f"model/recency_gamma/{t}"] = float(gammas[t])

    # ── Table: per-patient summary ────────────────────────────────────────────
    table_cols = (["patient_id", "n_biopsies", "acr_label_last"] +
                  [f"hazard_{t}" for t in tasks] +
                  ["dominant_biopsy_acr_surv"])
    table = wandb.Table(columns=table_cols)

    # ── Hazard trajectory line series ─────────────────────────────────────────
    traj_xs_all  = {t: [] for t in tasks}
    traj_ys_all  = {t: [] for t in tasks}
    traj_keys    = {t: [] for t in tasks}

    for extr in all_extractions:
        pid = extr["patient_id"]
        T   = extr["n_biopsies"]
        last_label = extr["records"][-1].get("label") if extr["records"] else None

        hazards = {t: extr["logits"].get(t, float("nan")) for t in tasks}

        # Dominant biopsy for acr_surv (highest α total)
        dom_biopsy = -1
        alpha_as = extr["alpha_per_task"].get("acr_surv")
        if alpha_as is not None:
            ba = np.zeros(T)
            for i, b in enumerate(extr["tok_biopsy"]):
                ba[b] += alpha_as[i]
            dom_biopsy = int(np.argmax(ba)) + 1   # 1-indexed

        row = [str(pid), T, last_label] + [hazards.get(t, float("nan")) for t in tasks] + [dom_biopsy]
        table.add_data(*row)

        # Trajectory data
        for t in tasks:
            hvals = extr["hazard_traj"].get(t, [])
            for bi, hv in enumerate(hvals):
                traj_xs_all[t].append(bi + 1)
                traj_ys_all[t].append(float(hv))
                traj_keys[t].append(str(pid))

    scalar_log["patients/summary_table"] = table

    # Trajectory line series (one per task)
    for t in tasks:
        if traj_xs_all[t]:
            try:
                scalar_log[f"trajectory/{t}"] = wandb.plot.scatter(
                    wandb.Table(data=list(zip(traj_xs_all[t], traj_ys_all[t])),
                                columns=["biopsy_n", "hazard"]),
                    x="biopsy_n", y="hazard",
                    title=f"Hazard trajectory — {TASK_LABELS.get(t, t)}")
            except Exception:
                pass

    wandb.log(scalar_log)

    # ── Images: all PNGs organised by panel type ──────────────────────────────
    png_map: Dict[str, List] = {
        "L_global":         [],
        "L0_summary":       [],
        "L1_seed_timeline": [],
        "L2_sab_attn":      [],
        "L2b_alibi":        [],
        "L3_abmil":         [],
        "L4_hazard":        [],
        "L5_pma":           [],
        "L6_gate":          [],
        "L7_mod_contrib":   [],
        "L8_seed_trend":    [],
        "Lpop_K":           [],
        "Lpop":             [],
        "other":            [],
    }

    _prefix_map = {
        "L_global":         "L_global",
        "L0_summary":       "L0_summary",
        "L1_seed_timeline": "L1_seed_timeline",
        "L2b_alibi":        "L2b_alibi",
        "L2_temporal_sab":  "L2_sab_attn",
        "L3_recency":       "L3_abmil",
        "L4_hazard":        "L4_hazard",
        "L5_pma":           "L5_pma",
        "L6_task":          "L6_gate",
        "L7_mod":           "L7_mod_contrib",
        "L8_seed":          "L8_seed_trend",
        "Lpop_K":           "Lpop_K",
        "Lpop":             "Lpop",
    }

    for png_path in sorted(out_dir.glob("*.png")):
        stem = png_path.stem
        bucket = "other"
        for prefix, key in _prefix_map.items():
            if stem.startswith(prefix):
                bucket = key
                break
        # Build a clean caption: panel type + patient ID if present
        caption_parts = [stem]
        png_map[bucket].append(
            wandb.Image(str(png_path), caption=" | ".join(caption_parts)))

    img_log = {}
    for bucket, imgs in png_map.items():
        if imgs:
            img_log[f"panels/{bucket}"] = imgs

    if img_log:
        wandb.log(img_log)
        total = sum(len(v) for v in img_log.values())
        print(f"  [wandb] uploaded {total} PNGs across {len(img_log)} panel groups")

    run.finish()
    print(f"  [wandb] run: {run.url}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Longitudinal-MK-MT interpretability")
    parser.add_argument("--split",      type=int, default=0)
    parser.add_argument("--fold",       type=int, default=0)
    parser.add_argument("--task",       type=str, default="mega",
                        choices=["mega", "acr_cls", "cls", "acr_surv", "clad", "clad_surv", "death", "death_surv"],
                        help="'mega' for multitask model; or per-task variant")
    parser.add_argument("--n-patients", type=int, default=30,
                        help="Max test patients to analyse (sorted by #biopsies desc)")
    parser.add_argument("--min-biopsies", type=int, default=2,
                        help="Skip patients with fewer biopsies than this")
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument("--wandb-project", default="chicago-mil-interpretability",
                        help="W&B project name (set to 'none' to skip W&B logging)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[main] device={device}  split={args.split}  fold={args.fold}  task={args.task}")

    # Output directory — separate subdir for per-task vs mega
    task_suffix = args.task if args.task != "mega" else "mega"
    out_dir = OUT_ROOT / f"split{args.split}_fold{args.fold}_{task_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tasks = load_model(args.split, args.fold, device, task=args.task)
    print(f"[main] tasks={tasks}  n_seeds={model.n_seeds}")

    # Global model plots (don't need patient data)
    print("[main] → L_global")
    plot_L_global(model, out_dir, tasks)

    # Load test patients
    print("[main] Loading longitudinal splits …")
    splits = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV,
                                       fold=args.fold, split=args.split)
    test_patients = splits.get("test", [])
    print(f"[main] {len(test_patients)} test patients found")

    # Filter and sort by number of biopsies (most informative first)
    test_patients = [p for p in test_patients if len(p["stems"]) >= args.min_biopsies]
    test_patients.sort(key=lambda p: len(p["stems"]), reverse=True)
    test_patients = test_patients[:args.n_patients]
    print(f"[main] Processing {len(test_patients)} patients (≥{args.min_biopsies} biopsies)")

    all_extractions    = []
    cluster_names_pool = {}   # accumulated cluster names across patients

    for pi, patient in enumerate(test_patients):
        pid = patient["patient_id"]
        T   = len(patient["stems"])
        print(f"  [{pi+1}/{len(test_patients)}] patient={pid}  biopsies={T}")

        try:
            bags_list, transplant_days = load_patient_bags(patient, device)
            cluster_names = load_patient_cluster_names(patient)
            # Accumulate cluster names (first seen per modality wins)
            for mo, nms in cluster_names.items():
                if mo not in cluster_names_pool and nms:
                    cluster_names_pool[mo] = nms

            extr = extract_patient_longitudinal(model, patient, bags_list, device, tasks)
            if extr is not None:
                extr["transplant_days"] = transplant_days
            if extr is None:
                print(f"    skipped (no valid bags)")
                continue

            all_extractions.append(extr)

            # Per-patient plots
            plot_patient_summary(extr, model, out_dir, tasks)
            plot_L1(extr, model, out_dir)
            plot_L2(extr, model, out_dir, tasks)
            plot_L2b_alibi(extr, model, out_dir, ref_task=tasks[0])
            plot_L3(extr, model, out_dir, tasks)
            plot_L4(extr, model, out_dir, tasks)
            plot_L5_pma_affinity(extr, model, out_dir, cluster_names)
            plot_L6_gate(extr, model, out_dir, tasks)
            plot_L7_mod_contrib(extr, model, out_dir, tasks)
            plot_L8_seed_trend(extr, model, out_dir)

        except Exception as exc:
            import traceback
            print(f"    ERROR: {exc}")
            traceback.print_exc()
            continue

    # Population-level plots
    if all_extractions:
        print(f"[main] Population plots (N={len(all_extractions)} patients)")
        plot_population_alpha(all_extractions, model, out_dir, tasks)
        plot_population_seed_trends(all_extractions, model, out_dir)
        plot_population_seed_attribution(all_extractions, cluster_names_pool,
                                         model, out_dir, tasks)

    # W&B logging
    if args.wandb_project.lower() != "none" and all_extractions:
        print(f"[main] W&B logging → project '{args.wandb_project}'")
        log_to_wandb(model, all_extractions, tasks, out_dir,
                     args.split, args.fold, args.wandb_project)

    print(f"[main] Done. Figures in {out_dir}/")


if __name__ == "__main__":
    main()
