"""
slot_collapse_check.py — Diagnose attention collapse in SharedSlotMIL on real data.

Generates the same 4-panel plots as synthetic_slot_test.py:
  1. feature_space_journey.png
       P1: raw patch features UMAP (colored by modality + ACR class)
       P2: encoded features UMAP + slot init (×) + slot post-attn (◆)
       P3: slot Voronoi assignment (argmax competitive attn)
       P4: ABMIL alpha bar — mean per slot for ACR+ vs ACR- patients

  2. routing_by_class.png
       Mean attn weight (slot k ← patch from patient class c) per slot
       Rows: ACR+  vs  ACR-  |  Columns: HE / BAL / CT / Clinical

  3. slot_alignment.png
       Which slot has peak Δalpha (ACR+ − ACR−)?  Is it consistent across modalities?

Collapse signals to watch:
  • inter-slot std ≈ 0   → all K slots converged to same rep (mean-pool collapse)
  • alpha entropy ≈ log(K) → attention is uniform, no slot specialises
  • routing: one slot gets ALL patches in BOTH classes equally → not learning

Usage
-----
  sbatch interpretability/submit_slot_collapse_check.sh
  # or for a quick CPU dry-run:
  python3 interpretability/slot_collapse_check.py \\
      --split 1 --fold 0 --p2-tag alt_shared_comp \\
      --split-set test --n-patients 60 --mods HE BAL
"""

from __future__ import annotations
import argparse, gc, sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from mil.data.loader   import preload_bags
from mil.data.splits   import build_splits_multitask
from mil.data.registry import MODALITIES
from mil.models.builders import build_model_v8

SAMPLES_DIR = "/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV  = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
RESULTS_DIR = Path("/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 42

MOD_COLORS = {
    "HE":       "#E53935",
    "BAL":      "#1E88E5",
    "CT":       "#43A047",
    "Clinical": "#8E24AA",
}
# Lighter versions for class-0 (ACR-)
MOD_COLORS_LIGHT = {
    "HE":       "#FFCDD2",
    "BAL":      "#BBDEFB",
    "CT":       "#C8E6C9",
    "Clinical": "#E1BEE7",
}

MAX_PATCHES_PER_MOD = 200   # per patient for UMAP (subsampled)
MAX_HE_PATCHES      = 1024  # hard cap for GPU


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(results_dir: Path, split: int, fold: int, p2_tag: str,
               slot_k: int = 128):
    fold_tag  = f"split{split}_fold{fold}"
    ckpt_path = results_dir / "phase2" / fold_tag / f"slot_mega_{p2_tag}" / "model_slot_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = build_model_v8(variant="slot", slot_k=slot_k, task="mega")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    print(f"  Loaded: {ckpt_path.parent.name}  K={slot_k}")
    return model


# Variant → (builder_variant, ckpt_prefix)
_NEW_VARIANTS = {
    "set_mil_mt":        ("set_mil_mt",        "model_set_mil_mt_final.pt"),
    "mario_kempes_mt":   ("set_mil_mt",         "model_mario_kempes_mt_final.pt"),
    "longitudinal_mk_mt":("longitudinal_mk_mt", "model_longitudinal_mk_mt_final.pt"),
    "longitudinal_mk":   ("longitudinal_mk",    "model_longitudinal_mk_final.pt"),
}


def load_model_new(results_dir: Path, split: int, fold: int,
                   variant: str, slot_k: int = 16):
    """Load a SetTransformerMIL / LongitudinalMIL checkpoint (new models)."""
    if variant not in _NEW_VARIANTS:
        raise ValueError(f"Unknown new variant {variant!r}. Choose: {list(_NEW_VARIANTS)}")
    builder_variant, ckpt_name = _NEW_VARIANTS[variant]
    fold_tag  = f"split{split}_fold{fold}"
    task_dir  = f"{variant}_mega"
    ckpt_path = results_dir / "phase2" / fold_tag / task_dir / ckpt_name
    if not ckpt_path.exists():
        # try ep_*.pt in ckpts subdir as fallback
        ckpts = sorted((results_dir / "phase2" / fold_tag / task_dir
                        ).glob(f"ckpts_{variant}_final/ep_*.pt"))
        if ckpts:
            ckpt_path = ckpts[-1]
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = build_model_v8(variant=builder_variant, slot_k=slot_k, task="mega")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    # Fix seeds shape: old checkpoints saved seeds as (1, K, H), model now expects (K, H)
    for k in list(state.keys()):
        if k.endswith(".seeds") and state[k].dim() == 3 and state[k].shape[0] == 1:
            state[k] = state[k].squeeze(0)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    print(f"  Loaded new model {variant}  K={slot_k}  ckpt={ckpt_path.name}")
    return model


@torch.no_grad()
def extract_patient_new(model, bags: dict, mods: List[str],
                        max_patches: int, cluster_map: Optional[dict] = None,
                        cluster_labels: Optional[dict] = None) -> dict:
    """
    Extract PMA attention weights, seed reps, patch encodings, and pre-softmax
    logits for new SetTransformerMIL / LongitudinalMIL models.

    cluster_labels: {mod: List[str]} per-patch cluster name strings, aligned with
                    the patch features. Used to build seed×cluster affinity heatmaps.

    Returns dict with (for each modality):
      h_enc_{mod}         (N_sub, H)   encoded + L2-normed
      pma_attn_{mod}      (K, N_sub)   post-normalisation attention weights
      pma_logits_{mod}    (K, N_sub)   pre-softmax raw dot products (q@k, avg heads)
      cluster_labels_{mod} (N_sub,)    cluster name per patch (str array)
      seed_rep_{mod}      (K, H)       post-PMA seed representations
    And:
      alpha_{task}        (K*M,)       ABMIL alpha over concatenated seeds
    """
    result: dict = {}

    for mod in mods:
        t = bags.get(mod)
        if t is None:
            continue
        t = t.to(DEVICE, non_blocking=True)

        # Track subsampling for cluster label alignment
        cl_raw = cluster_labels.get(mod) if cluster_labels else None
        cl_arr  = np.array(cl_raw) if cl_raw is not None else None

        if t.shape[0] > MAX_HE_PATCHES:
            idx = torch.randperm(t.shape[0], device=DEVICE)[:MAX_HE_PATCHES]
            t = t[idx]
            if cl_arr is not None:
                cl_arr = cl_arr[idx.cpu().numpy()]

        enc = model.encoders[mod]
        h   = enc.encode_patches(t)          # (N, H) L2-normed

        N = h.shape[0]
        if N > max_patches:
            sub_idx = torch.randperm(N, device=DEVICE)[:max_patches]
            h_sub = h[sub_idx]
            if cl_arr is not None:
                cl_arr = cl_arr[sub_idx.cpu().numpy()]
        else:
            h_sub, sub_idx = h, None

        result[f"h_enc_{mod}"] = h_sub.cpu().float().numpy()
        if cl_arr is not None:
            result[f"cluster_labels_{mod}"] = cl_arr  # (N_sub,) str array

        # PMA forward — post-norm weights + all three pre-norm stages
        pma = model.pma[mod]
        seeds_out, attn_w, (dots, relu_dots, raw_pow) = pma(h, return_attn=True, return_logits=True)

        def _sub(t):
            return t[:, sub_idx] if sub_idx is not None else t

        result[f"pma_attn_{mod}"]      = _sub(attn_w).cpu().float().numpy()    # (K, N) post-norm
        result[f"pma_dots_{mod}"]      = _sub(dots).cpu().float().numpy()      # (K, N) raw q·k
        result[f"pma_relu_{mod}"]      = _sub(relu_dots).cpu().float().numpy() # (K, N) relu(q·k)
        result[f"pma_raw_pow_{mod}"]   = _sub(raw_pow).cpu().float().numpy()   # (K, N) relu(q·k)^b
        result[f"pma_logits_{mod}"]    = result[f"pma_dots_{mod}"]             # alias for compat
        result[f"seed_rep_{mod}"]   = seeds_out.cpu().float().numpy()   # (K, H)
        # Aliases so existing plot functions (slot_attn, slot_rep, h_raw) work unchanged
        result[f"slot_attn_{mod}"] = result[f"pma_attn_{mod}"]
        result[f"slot_rep_{mod}"]  = result[f"seed_rep_{mod}"]
        result[f"h_raw_{mod}"]     = result[f"h_enc_{mod}"]

    if not any(k.startswith("seed_rep_") for k in result):
        return result

    # Per-task ABMIL alpha over concatenated seeds (same as model's forward)
    present = [m for m in mods if f"seed_rep_{m}" in result]
    seed_list = [torch.tensor(result[f"seed_rep_{m}"], device=DEVICE) for m in present]
    seeds_cat = torch.cat(seed_list, dim=0)   # (M*K, H)

    for task in model.task_names:
        gate  = model.abmil_V[task](seeds_cat) * model.abmil_U[task](seeds_cat)
        alpha = torch.softmax(model.abmil_w[task](gate), dim=0).squeeze(1)
        result[f"alpha_{task}"] = alpha.cpu().float().numpy()   # (M*K,)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Per-patient extraction
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_patient(model, bags: dict, mods: List[str], max_patches: int) -> dict:
    """
    Returns dict with, for each modality:
      h_raw_{mod}       (N_sub, feat_dim)   raw subsampled patches
      h_enc_{mod}       (N_sub, H)          encoded + L2-normed
      slot_attn_{mod}   (K, N_sub)          competitive attn weights
      slot_rep_{mod}    (K, H)              post-attn slot reps
    And:
      alpha_{task}      (K,)                ABMIL alpha per slot per task
    """
    result: dict = {}
    K = model.shared_slots.shape[0]

    for mod in mods:
        t = bags.get(mod)
        if t is None:
            continue
        t = t.to(DEVICE, non_blocking=True)
        # cap patches
        if t.shape[0] > MAX_HE_PATCHES:
            idx = torch.randperm(t.shape[0], device=DEVICE)[:MAX_HE_PATCHES]
            t = t[idx]

        enc = model.encoders[mod]
        h   = enc.encode_patches(t)          # (N, H)

        # subsample for UMAP storage (keep GPU compact)
        N = h.shape[0]
        if N > max_patches:
            sub_idx = torch.randperm(N, device=DEVICE)[:max_patches]
            h_sub   = h[sub_idx]
            t_sub   = t[sub_idx]
        else:
            h_sub, t_sub, sub_idx = h, t, None

        result[f"h_raw_{mod}"] = t_sub.cpu().float().numpy()
        result[f"h_enc_{mod}"] = h_sub.cpu().numpy()

        # slot attention (full N, not subsampled — needed for slot rep quality)
        sa    = model.slot_attns[mod]
        slots, attn_w = sa(h, model.shared_slots, return_attn=True)  # (K,H), (K,N)

        # subsampled attn for plotting
        if sub_idx is not None:
            attn_sub = attn_w[:, sub_idx]
        else:
            attn_sub = attn_w
        result[f"slot_attn_{mod}"] = attn_sub.cpu().numpy()   # (K, N_sub)
        result[f"slot_rep_{mod}"]  = slots.cpu().numpy()      # (K, H)

    if not any(k.startswith("slot_rep_") for k in result):
        return result

    # Aggregate slots across present modalities for ABMIL alpha
    mod_slots = [torch.tensor(result[f"slot_rep_{m}"], device=DEVICE)
                 for m in mods if f"slot_rep_{m}" in result]
    slots_agg = torch.stack(mod_slots).mean(0)   # (K, H)

    for task in model.task_names:
        gate  = model.abmil_V[task](slots_agg) * model.abmil_U[task](slots_agg)
        alpha = torch.softmax(model.abmil_w[task](gate), dim=0).squeeze(1)  # (K,)
        result[f"alpha_{task}"] = alpha.cpu().numpy()

    return result


# ══════════════════════════════════════════════════════════════════════════════
# UMAP helper
# ══════════════════════════════════════════════════════════════════════════════

def _fit_umap(X: np.ndarray, seed: int = SEED) -> np.ndarray:
    try:
        import umap as umap_lib
        return umap_lib.UMAP(n_components=2, random_state=seed,
                             n_neighbors=15, min_dist=0.1,
                             metric="cosine").fit_transform(X)
    except ImportError:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        X50  = PCA(n_components=min(50, X.shape[1])).fit_transform(X)
        return TSNE(n_components=2, random_state=seed,
                    perplexity=min(30, len(X) - 1)).fit_transform(X50)


# ══════════════════════════════════════════════════════════════════════════════
# Collapse statistics
# ══════════════════════════════════════════════════════════════════════════════

def print_collapse_stats(patient_results: List[dict], mods: List[str],
                         task: str = "acr_cls"):
    K = next((r[f"slot_rep_{m}"].shape[0] for r in patient_results for m in mods if f"slot_rep_{m}" in r), 16)
    print(f"\n  {'='*56}")
    print(f"  Collapse diagnostics  (K={K}  task={task})")
    print(f"  {'='*56}")

    # Inter-slot std per patient (0 = collapsed)
    for mod in mods:
        stds = [r[f"slot_rep_{mod}"].std(0).mean()
                for r in patient_results if f"slot_rep_{mod}" in r]
        if stds:
            print(f"  {mod:10s} inter-slot std:  mean={np.mean(stds):.5f}  "
                  f"min={np.min(stds):.5f}  (0=collapsed)")

    # Alpha entropy per class
    alphas1 = [r[f"alpha_{task}"] for r in patient_results
               if r.get("label_acr_cls") == 1 and f"alpha_{task}" in r]
    alphas0 = [r[f"alpha_{task}"] for r in patient_results
               if r.get("label_acr_cls") == 0 and f"alpha_{task}" in r]
    unif_ent = np.log(K)
    if alphas1:
        a1 = np.stack(alphas1).mean(0)
        ent1 = -np.sum(a1 * np.log(a1 + 1e-9))
        print(f"  ACR+ alpha entropy:  {ent1:.3f}  (uniform={unif_ent:.3f}, "
              f"lower=more specialised)")
    if alphas0:
        a0 = np.stack(alphas0).mean(0)
        ent0 = -np.sum(a0 * np.log(a0 + 1e-9))
        print(f"  ACR- alpha entropy:  {ent0:.3f}  (uniform={unif_ent:.3f})")
    if alphas1 and alphas0:
        da = np.stack(alphas1).mean(0) - np.stack(alphas0).mean(0)
        k_star = int(np.argmax(np.abs(da)))
        print(f"  Peak |Δalpha| slot:  k*={k_star}  Δ={da[k_star]:+.4f}")
        print(f"  alpha[k*] ACR+={np.stack(alphas1).mean(0)[k_star]:.4f}  "
              f"ACR-={np.stack(alphas0).mean(0)[k_star]:.4f}  "
              f"uniform={1/K:.4f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: feature_space_journey
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_space_journey(patient_results: List[dict], mods: List[str],
                               init_slots_np: np.ndarray,
                               k_star: int, task: str, out_dir: Path):
    K = init_slots_np.shape[0]
    # Gather patch clouds
    raw_pts, enc_pts, mod_lbl, class_lbl = [], [], [], []
    for r in patient_results:
        cl = r.get("label_acr_cls", -1)
        for mod in mods:
            rw = r.get(f"h_raw_{mod}")
            en = r.get(f"h_enc_{mod}")
            if rw is None:
                continue
            raw_pts.append(rw)
            enc_pts.append(en)
            mod_lbl.extend([mod] * len(rw))
            class_lbl.extend([cl] * len(rw))
    if not raw_pts:
        print("  No patch data for feature_space_journey"); return
    raw_pts   = np.concatenate(raw_pts)
    enc_pts   = np.concatenate(enc_pts)
    mod_lbl   = np.array(mod_lbl)
    class_lbl = np.array(class_lbl)
    N = len(raw_pts)

    # Mean post-attn slot reps (averaged across patients per modality)
    mean_slots = {}
    for mod in mods:
        reps = [r[f"slot_rep_{mod}"] for r in patient_results if f"slot_rep_{mod}" in r]
        if reps:
            mean_slots[mod] = np.stack(reps).mean(0)   # (K, H)
    mean_slot_np = np.stack(list(mean_slots.values())).mean(0)  # (K, H)

    # ABMIL alpha per class
    a1 = [r[f"alpha_{task}"] for r in patient_results
          if r.get("label_acr_cls") == 1 and f"alpha_{task}" in r]
    a0 = [r[f"alpha_{task}"] for r in patient_results
          if r.get("label_acr_cls") == 0 and f"alpha_{task}" in r]
    from collections import Counter as _Counter
    _shapes = [x.shape for x in (a1 + a0)]
    _modal  = _Counter(_shapes).most_common(1)[0][0] if _shapes else None
    a1 = [x for x in a1 if x.shape == _modal] if _modal else []
    a0 = [x for x in a0 if x.shape == _modal] if _modal else []
    alpha1 = np.stack(a1).mean(0) if a1 else np.ones(K) / K
    alpha0 = np.stack(a0).mean(0) if a0 else np.ones(K) / K

    # Slot assignment from attn
    slot_assign = []
    for r in patient_results:
        for mod in mods:
            attn = r.get(f"slot_attn_{mod}")
            if attn is None:
                continue
            slot_assign.extend(np.argmax(attn, axis=0).tolist())
    slot_assign = np.array(slot_assign[:N])

    # Subsample to ≤8k pts for UMAP speed (random_state forces n_jobs=1)
    _UMAP_MAX = 8000
    if N > _UMAP_MAX:
        rng = np.random.default_rng(SEED)
        sub_idx = rng.choice(N, _UMAP_MAX, replace=False)
        raw_pts_u   = raw_pts[sub_idx]
        enc_pts_u   = enc_pts[sub_idx]
        mod_lbl_u   = mod_lbl[sub_idx]
        class_lbl_u = class_lbl[sub_idx]
        slot_assign_u = slot_assign[sub_idx] if len(slot_assign) == N else slot_assign[:_UMAP_MAX]
        N_u = _UMAP_MAX
    else:
        raw_pts_u, enc_pts_u = raw_pts, enc_pts
        mod_lbl_u, class_lbl_u, slot_assign_u = mod_lbl, class_lbl, slot_assign
        N_u = N

    print(f"  Fitting UMAP — raw ({raw_pts_u.shape})...")
    raw_2d = _fit_umap(raw_pts_u)

    print(f"  Fitting UMAP — encoded + slots ({enc_pts_u.shape[0] + K*2})...")
    joint    = np.concatenate([enc_pts_u, init_slots_np, mean_slot_np], axis=0)
    joint_2d = _fit_umap(joint)
    enc_2d   = joint_2d[:N_u]
    init_2d  = joint_2d[N_u:N_u + K]
    post_2d  = joint_2d[N_u + K:]

    # Colors per point
    def pt_color(mod, cl):
        return MOD_COLORS[mod] if cl == 1 else MOD_COLORS_LIGHT.get(mod, "#ccc")

    c_inst = np.array([pt_color(mod_lbl_u[i], class_lbl_u[i]) for i in range(N_u)])
    SLOT_CMAP = plt.colormaps.get_cmap("tab20").resampled(K)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        f"SharedSlotMIL — Feature space journey  (K={K} slots, task={task})\n"
        f"Peak Δalpha slot = S{k_star}*  |  "
        f"saturated color = ACR+,  pastel = ACR-",
        fontsize=12, fontweight="bold", y=1.01)

    # Panel 1: raw features
    ax = axes[0, 0]
    for mod in mods:
        for cl, marker, alpha_v in [(1, "o", 0.6), (0, "o", 0.3)]:
            mask = (mod_lbl_u == mod) & (class_lbl_u == cl)
            if mask.sum() == 0: continue
            label = f"{mod} ACR+" if cl == 1 else f"{mod} ACR-"
            ax.scatter(raw_2d[mask, 0], raw_2d[mask, 1],
                       c=pt_color(mod, cl), s=6, alpha=alpha_v,
                       label=label, rasterized=True)
    ax.set_title("Raw patch features (before encoder)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, markerscale=2, ncol=2)
    ax.set_xticks([]); ax.set_yticks([])

    # Panel 2: encoded + slot positions
    ax = axes[0, 1]
    ax.scatter(enc_2d[:, 0], enc_2d[:, 1], c=c_inst,
               s=4, alpha=0.4, rasterized=True, zorder=1)
    for k in range(K):
        col = "#fdd835" if k == k_star else "#aaa"
        ax.scatter(init_2d[k, 0], init_2d[k, 1], marker="x",
                   c=col, s=(120 if k == k_star else 40),
                   linewidths=(2 if k == k_star else 0.8), zorder=3, alpha=0.9)
    for k in range(K):
        col = "#f57f17" if k == k_star else SLOT_CMAP(k / K)
        ms  = 200 if k == k_star else 50
        ax.scatter(post_2d[k, 0], post_2d[k, 1], marker="D",
                   c=col, s=ms, edgecolors="black", linewidths=0.5, zorder=4)
    # Annotate k* only (K=128 is too dense to label all)
    ax.annotate(f"S{k_star}*", (post_2d[k_star, 0], post_2d[k_star, 1]),
                fontsize=8, fontweight="bold", color="#f57f17",
                xytext=(4, 6), textcoords="offset points")
    from matplotlib.lines import Line2D
    proxies = [
        *[Line2D([0],[0], color=MOD_COLORS[m],  marker="o", ls="none", ms=5,
                 label=f"{m} ACR+") for m in mods if m in MOD_COLORS],
        *[Line2D([0],[0], color=MOD_COLORS_LIGHT.get(m,"#ccc"), marker="o", ls="none", ms=5,
                 label=f"{m} ACR-") for m in mods if m in MOD_COLORS_LIGHT],
        Line2D([0],[0], color="#aaa",    marker="x", ls="none", ms=6, label="slot init"),
        Line2D([0],[0], color="#aaa",    marker="D", ls="none", ms=5, label="slot post-attn"),
        Line2D([0],[0], color="#f57f17", marker="D", ls="none", ms=8, label=f"S{k_star}* (peak Δα)"),
    ]
    ax.legend(handles=proxies, fontsize=6, ncol=2)
    ax.set_title(f"Encoded features (H={enc_pts.shape[1]} sphere)\n"
                 "× = slot init  ◆ = slot post-attn", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # Panel 3: slot assignment (Voronoi)
    ax = axes[1, 0]
    for k in range(K):
        mask = slot_assign_u == k
        if mask.sum() == 0: continue
        clr = "#f57f17" if k == k_star else SLOT_CMAP(k / K)
        ax.scatter(enc_2d[mask, 0], enc_2d[mask, 1],
                   c=clr, s=5, alpha=0.6, rasterized=True)
    ax.scatter(post_2d[k_star, 0], post_2d[k_star, 1], marker="D",
               c="#f57f17", s=200, edgecolors="black", linewidths=1, zorder=5)
    ax.annotate(f"S{k_star}*", (post_2d[k_star, 0], post_2d[k_star, 1]),
                fontsize=8, fontweight="bold", color="#f57f17",
                xytext=(4, 6), textcoords="offset points")
    ax.set_title("Slot Voronoi assignment (argmax competitive attn)\n"
                 "Collapse: one color dominates everything",
                 fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    # fraction going to each slot
    counts = np.bincount(slot_assign_u, minlength=K)
    dom_slot = int(np.argmax(counts)
                   )
    dom_frac = counts[dom_slot] / counts.sum()
    ax.text(0.02, 0.02, f"Most dominant slot: S{dom_slot} ({dom_frac*100:.1f}%)\n"
            f"(collapse if one slot >>{'1/K'} = {100/K:.1f}%)",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(fc="white", alpha=0.8, ec="none"))

    # Panel 4: ABMIL alpha bar
    ax = axes[1, 1]
    x = np.arange(K)
    # Show as line for K=128 (bar chart too dense)
    ax.plot(x, alpha1, color="#c62828", lw=1.5, label=f"ACR+ (n={len(a1)})", alpha=0.8)
    ax.plot(x, alpha0, color="#1565c0", lw=1.5, label=f"ACR- (n={len(a0)})", alpha=0.8)
    ax.axhline(1 / K, color="#555", lw=1, ls="--", label=f"uniform 1/{K}")
    da = alpha1 - alpha0
    ax.fill_between(x, alpha1, alpha0, where=(alpha1 > alpha0),
                    color="#c62828", alpha=0.15, label="ACR+ higher")
    ax.fill_between(x, alpha1, alpha0, where=(alpha1 < alpha0),
                    color="#1565c0", alpha=0.15, label="ACR- higher")
    ax.axvline(k_star, color="#f57f17", lw=2, ls=":", label=f"k*={k_star}")
    ax.set_xlabel("Slot index"); ax.set_ylabel("Mean ABMIL α")
    ax.set_title(f"ABMIL alpha per slot — {task}\n"
                 f"Collapse: flat line at 1/{K}={1/K:.4f}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.text(0.02, 0.98, f"S{k_star}*: ACR+={alpha1[k_star]:.4f}  "
            f"ACR-={alpha0[k_star]:.4f}  Δ={da[k_star]:+.4f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(fc="white", alpha=0.8, ec="none"))

    fig.tight_layout()
    p = out_dir / "feature_space_journey.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: routing_by_class
# ══════════════════════════════════════════════════════════════════════════════

def plot_routing_by_class(patient_results: List[dict], mods: List[str],
                          k_star: int, out_dir: Path):
    K = next((r[f"slot_rep_{m}"].shape[0] for r in patient_results for m in mods if f"slot_rep_{m}" in r), 16)

    # For each mod × class: mean attn weight per slot averaged over patches
    # attn shape is (K, N_sub) per patient — we want mean over N_sub → (K,)
    class_labels = [0, 1]
    mat: Dict[str, Dict[int, np.ndarray]] = {m: {} for m in mods}
    for mod in mods:
        for cl in class_labels:
            vecs = []
            for r in patient_results:
                if r.get("label_acr_cls") != cl:
                    continue
                attn = r.get(f"slot_attn_{mod}")
                if attn is None:
                    continue
                vecs.append(attn.mean(1))    # (K,) — mean across patches
            mat[mod][cl] = np.stack(vecs).mean(0) if vecs else np.zeros(K)

    present = [m for m in mods if m in mat and 0 in mat[m] and 1 in mat[m]]
    if not present:
        print("  No data for routing_by_class"); return

    n_mods = len(present)
    fig, axes = plt.subplots(2, n_mods, figsize=(5 * n_mods, 8))
    if n_mods == 1:
        axes = axes[:, np.newaxis]

    vmax = max(mat[m][cl].max() for m in present for cl in class_labels)
    x    = np.arange(K)

    for col, mod in enumerate(present):
        for row, cl in enumerate(class_labels):
            ax = axes[row, col]
            v  = mat[mod][cl]
            label = "ACR+" if cl == 1 else "ACR-"
            ax.bar(x, v,
                   color=["#f57f17" if k == k_star else
                           (MOD_COLORS[mod] if cl == 1 else MOD_COLORS_LIGHT.get(mod, "#ccc"))
                           for k in range(K)],
                   width=1.0, edgecolor="none")
            ax.axhline(1 / K, lw=1, ls="--", color="#555")
            ax.axvline(k_star, lw=1.5, ls=":", color="#f57f17")
            ax.set_ylim(0, vmax * 1.1)
            ax.set_xlabel("Slot index")
            ax.set_ylabel("Mean attn weight")
            ax.set_title(f"{mod} — {label}", fontsize=9, fontweight="bold")
            ax.text(k_star, vmax * 0.9, f"k*={k_star}", fontsize=7,
                    color="#f57f17", ha="center")

    fig.suptitle(
        "Mean routing weight per slot (averaged over patches)\n"
        "Collapse: flat bar charts for both classes",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "routing_by_class.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: slot_alignment
# ══════════════════════════════════════════════════════════════════════════════

def plot_slot_alignment(patient_results: List[dict], mods: List[str],
                        task: str, out_dir: Path):
    K = next((r[f"slot_rep_{m}"].shape[0] for r in patient_results for m in mods if f"slot_rep_{m}" in r), 16)

    # Δalpha per task
    a1 = [r[f"alpha_{task}"] for r in patient_results
          if r.get("label_acr_cls") == 1 and f"alpha_{task}" in r]
    a0 = [r[f"alpha_{task}"] for r in patient_results
          if r.get("label_acr_cls") == 0 and f"alpha_{task}" in r]
    if not a1 or not a0:
        print("  Skipping slot_alignment (missing labels)"); return

    # Find modal shape across both lists and align both to it
    from collections import Counter
    all_shapes = [x.shape for x in a1 + a0]
    modal_shape = Counter(all_shapes).most_common(1)[0][0]
    a1 = [x for x in a1 if x.shape == modal_shape]
    a0 = [x for x in a0 if x.shape == modal_shape]
    if not a1 or not a0:
        print("  Skipping slot_alignment (inconsistent alpha shapes)"); return

    alpha1 = np.stack(a1).mean(0)   # (K,)
    alpha0 = np.stack(a0).mean(0)
    da     = alpha1 - alpha0
    k_star = int(np.argmax(np.abs(da)))

    # Per-patient alpha[k*] distribution
    ak1 = [r[f"alpha_{task}"][k_star] for r in patient_results
           if r.get("label_acr_cls") == 1 and f"alpha_{task}" in r]
    ak0 = [r[f"alpha_{task}"][k_star] for r in patient_results
           if r.get("label_acr_cls") == 0 and f"alpha_{task}" in r]

    # Top routing slot per modality (which slot do ACR+ patients route most to?)
    top_mod: Dict[str, int] = {}
    for mod in mods:
        vecs = []
        for r in patient_results:
            if r.get("label_acr_cls") != 1:
                continue
            attn = r.get(f"slot_attn_{mod}")
            if attn is None:
                continue
            vecs.append(attn.mean(1))
        if vecs:
            mean_attn = np.stack(vecs).mean(0)
            top_mod[mod] = int(np.argmax(mean_attn))

    # ── figure ─────────────────────────────────────────────────────────────
    n_mods = len([m for m in mods if m in top_mod])
    fig    = plt.figure(figsize=(16, 10))
    gs     = fig.add_gridspec(2, max(2, n_mods), hspace=0.45, wspace=0.35)

    # Row 0: routing bar per modality for ACR+ patients
    for col, mod in enumerate([m for m in mods if m in top_mod]):
        ax = fig.add_subplot(gs[0, col])
        vecs = [r[f"slot_attn_{mod}"].mean(1) for r in patient_results
                if r.get("label_acr_cls") == 1 and f"slot_attn_{mod}" in r]
        mv   = np.stack(vecs).mean(0)
        kmod = top_mod[mod]
        ax.bar(np.arange(K), mv,
               color=["#f57f17" if k == kmod else MOD_COLORS[mod] for k in range(K)],
               width=1.0, edgecolor="none", alpha=0.85)
        ax.axhline(1 / K, lw=1, ls="--", color="#555")
        ax.set_title(f"{mod} — ACR+ routing\nTop slot: k={kmod}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Slot"); ax.set_ylabel("Mean attn")

    # Row 1 left: Δalpha
    ax = fig.add_subplot(gs[1, 0])
    colors = ["#c62828" if v > 0 else "#1565c0" for v in da]
    colors[k_star] = "#f57f17"
    ax.bar(np.arange(K), da, color=colors, width=1.0, edgecolor="none")
    ax.axhline(0, lw=1, color="#555")
    ax.axvline(k_star, lw=2, ls=":", color="#f57f17")
    ax.set_xlabel("Slot index")
    ax.set_ylabel("α(ACR+) − α(ACR−)")
    ax.set_title(f"Δalpha per slot ({task})\nPeak at k*={k_star}  Δ={da[k_star]:+.4f}",
                 fontsize=9, fontweight="bold")

    # Row 1 right: per-patient alpha[k*] distribution
    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(min(min(ak1, default=[0]), min(ak0, default=[0])),
                       max(max(ak1, default=[1/K]), max(ak0, default=[1/K])) + 1e-6, 25)
    ax.hist(ak1, bins=bins, color="#c62828", alpha=0.6, density=True, label=f"ACR+ (n={len(ak1)})")
    ax.hist(ak0, bins=bins, color="#1565c0", alpha=0.6, density=True, label=f"ACR- (n={len(ak0)})")
    ax.axvline(1 / K, lw=1.5, ls="--", color="#555", label=f"uniform 1/{K}")
    ax.set_xlabel(f"alpha[k*={k_star}]")
    ax.set_ylabel("Density")
    ax.set_title(f"S{k_star}* alpha distribution per patient\n"
                 f"ACR+={np.mean(ak1):.4f}  ACR-={np.mean(ak0):.4f}",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)

    # Summary text
    same_slots = all(top_mod.get(m, -1) == top_mod.get(mods[0], -2)
                     for m in mods if m in top_mod)
    summary = "\n".join([
        f"Cross-modal alignment summary",
        f"Peak Δalpha slot (k*): {k_star}",
        *[f"Top routing slot {mod}: {top_mod.get(mod,'N/A')}" for mod in mods if mod in top_mod],
        f"Same slot across mods: {'YES — aligned' if same_slots else 'NO — misaligned'}",
        f"alpha[k*] ACR+ = {np.mean(ak1):.4f}",
        f"alpha[k*] ACR- = {np.mean(ak0):.4f}",
        f"Δalpha = {np.mean(ak1)-np.mean(ak0):+.4f}",
        f"Uniform baseline = {1/K:.4f}",
    ])
    fig.text(0.76, 0.08, summary, fontsize=8, va="bottom",
             fontfamily="monospace",
             bbox=dict(fc="#f5f5f5", ec="#ccc", boxstyle="round,pad=0.5"))

    fig.suptitle(
        f"Slot alignment — {task}\n"
        f"Collapse: all Δalpha ≈ 0,  routing flat,  peak slot random across modalities",
        fontsize=11, fontweight="bold")

    p = out_dir / "slot_alignment.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    return k_star


# ══════════════════════════════════════════════════════════════════════════════
# Plot 4: seed × cluster heatmap  (pre-softmax affinity)
# ══════════════════════════════════════════════════════════════════════════════

def _build_cluster_matrix(patient_results, mod, key, K, sorted_clusters):
    """Average key (K,N) tensor over patches per cluster, then over patients → (K, C)."""
    sums   = {c: np.zeros(K) for c in sorted_clusters}
    counts = {c: 0 for c in sorted_clusters}
    for r in patient_results:
        vals   = r.get(key)               # (K, N)
        labels = r.get(f"cluster_labels_{mod}")  # (N,) str
        if vals is None or labels is None:
            continue
        for c in sorted_clusters:
            mask = labels == c
            if mask.sum() == 0:
                continue
            sums[c]   += vals[:, mask].mean(axis=1)
            counts[c] += 1
    mat = np.stack([sums[c] / counts[c] if counts[c] > 0 else np.zeros(K)
                    for c in sorted_clusters], axis=1)  # (K, C)
    return mat


def plot_seed_cluster_routing(patient_results: List[dict], mods: List[str],
                              out_dir: Path):
    """
    For each modality: 5 panels showing how attention evolves through each step,
    plus the diff between consecutive steps.

    Panels per modality:
      1. q·k          — raw dot product
      2. relu(q·k)    — negatives zeroed
      3. relu(q·k)^b  — sharpened (pre-normalisation)
      4. Δ relu−dot   — what relu kills (negatives lost)
      5. Δ pow−relu   — what pow(b) sharpens (low positives suppressed)
    """
    mods_present = [m for m in mods
                    if any(f"pma_dots_{m}" in r and f"cluster_labels_{m}" in r
                           for r in patient_results)]
    if not mods_present:
        print("  Skipping seed_cluster_routing (no cluster labels or logits available)")
        return

    K = next((r[f"pma_dots_{m}"].shape[0]
               for r in patient_results for m in mods_present
               if f"pma_dots_{m}" in r), 16)

    # Collect sorted clusters per modality (by frequency across patients)
    def get_sorted_clusters(mod):
        cnt: Dict[str, int] = {}
        for r in patient_results:
            labels = r.get(f"cluster_labels_{mod}")
            if labels is None: continue
            for c in np.unique(labels):
                cnt[c] = cnt.get(c, 0) + 1
        return sorted(cnt, key=lambda c: -cnt[c])[:30]

    STAGES = [
        ("q·k",         "pma_dots",    "RdBu_r",  True),   # symmetric around 0
        ("relu(q·k)",   "pma_relu",    "Reds",     False),  # always ≥ 0
        ("relu(q·k)^b", "pma_raw_pow", "Reds",     False),
        ("Δ relu − dot\n(what relu kills)",  None, "Blues",  False),  # computed
        ("Δ pow − relu\n(what pow sharpens)", None, "Reds",  False),
    ]
    N_PANELS = len(STAGES)

    for mod in mods_present:
        sorted_clusters = get_sorted_clusters(mod)
        C = len(sorted_clusters)
        n_pts = sum(1 for r in patient_results if f"pma_dots_{mod}" in r
                    and f"cluster_labels_{mod}" in r)

        mats = {}
        for _, key, _, _ in STAGES[:3]:
            mats[key] = _build_cluster_matrix(
                patient_results, mod, f"{key}_{mod}", K, sorted_clusters)

        # Diff panels
        delta_relu = mats["pma_dots"] - mats["pma_relu"]   # what relu zeroed out (was negative → now 0)
        # flip sign: positives here = values that were negative in dot (lost by relu)
        delta_relu = -delta_relu.clip(max=0)                # show magnitude of negative dots killed
        delta_pow  = mats["pma_relu"] - mats["pma_raw_pow"] / (mats["pma_raw_pow"].max() + 1e-9) * mats["pma_relu"].max()
        # simpler: show raw_pow normalised to same scale as relu to see sharpening
        relu_max = mats["pma_relu"].max() + 1e-9
        pow_norm = mats["pma_raw_pow"] / (mats["pma_raw_pow"].max() + 1e-9) * relu_max
        delta_pow = pow_norm - mats["pma_relu"]             # positive = amplified, negative = suppressed

        computed = [delta_relu, delta_pow]

        fig, axes = plt.subplots(1, N_PANELS,
                                 figsize=(5 * N_PANELS, max(6, K * 0.4 + 3)))
        panel_mats = [mats["pma_dots"], mats["pma_relu"],
                      mats["pma_raw_pow"], delta_relu, delta_pow]

        for idx, (ax, (title, _, cmap, symmetric), mat) in enumerate(
                zip(axes, STAGES, panel_mats)):
            vmax = np.abs(mat).max() + 1e-9
            vmin = -vmax if symmetric else 0
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, shrink=0.5)

            ax.set_xticks(np.arange(C))
            ax.set_xticklabels(sorted_clusters, rotation=70, ha="right", fontsize=6)
            ax.set_yticks(np.arange(K))
            ax.set_yticklabels([f"s{k}" for k in range(K)], fontsize=6)
            ax.set_title(f"{mod} — {title}\nK={K} C={C} n={n_pts}",
                         fontsize=8, fontweight="bold")

            # Orange box: strongest cluster per seed (only for first 3 panels)
            if idx < 3:
                for k in range(K):
                    best = int(np.argmax(mat[k]))
                    ax.add_patch(plt.Rectangle((best - 0.5, k - 0.5), 1, 1,
                                               fill=False, edgecolor="#f57f17", lw=1.2))

        fig.suptitle(
            f"Seed × Cluster routing — {mod}\n"
            f"q·k → relu → relu^b  |  orange = strongest cluster per seed",
            fontsize=10, fontweight="bold")
        fig.tight_layout()
        p = out_dir / f"seed_cluster_routing_{mod}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",       type=int,  default=1)
    p.add_argument("--fold",        type=int,  default=0)
    p.add_argument("--p2-tag",      default="alt_shared_comp",
                   help="For old slot models: subfolder tag. "
                        "For new models: pass --variant instead.")
    p.add_argument("--variant",     default=None,
                   choices=list(_NEW_VARIANTS.keys()),
                   help="New model variant to analyse (overrides --p2-tag)")
    p.add_argument("--slot-k",      type=int,  default=16,
                   help="Number of PMA seeds (must match trained model)")
    p.add_argument("--split-set",   default="test",
                   choices=["train", "val", "test", "all"])
    p.add_argument("--n-patients",  type=int,  default=80,
                   help="Max patients (0=all)")
    p.add_argument("--mods",        nargs="+", default=["HE", "BAL", "CT", "Clinical"])
    p.add_argument("--task",        default="acr_cls")
    p.add_argument("--max-patches", type=int,  default=MAX_PATCHES_PER_MOD,
                   help="Max patches per patient per modality for UMAP")
    p.add_argument("--out-dir",     default=None)
    p.add_argument("--samples-dir", default=SAMPLES_DIR)
    p.add_argument("--splits-csv",  default=SPLITS_CSV)
    p.add_argument("--results-dir",    default=str(RESULTS_DIR))
    p.add_argument("--cluster-csv",    default=None,
                   help="Optional CSV with columns stem,cluster to colour patches by cluster")
    p.add_argument("--no-checkpoint",  action="store_true",
                   help="Skip loading checkpoint — run on randomly initialised model")
    p.add_argument("--wandb-project",  default="chicago-mil",
                   help="W&B project name (default: chicago-mil)")
    p.add_argument("--no-wandb",       action="store_true",
                   help="Disable W&B logging")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)

    use_new = args.variant is not None
    results_dir = Path(args.results_dir)
    suffix = args.variant if use_new else ("randinit" if args.no_checkpoint else args.p2_tag)
    tag_dir = (Path(_ROOT) / "interpretability" /
               f"slot_collapse_s{args.split}f{args.fold}_{suffix}")
    out_dir = Path(args.out_dir) if args.out_dir else tag_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  slot_collapse_check  split={args.split}  fold={args.fold}")
    print(f"  variant={suffix}  set={args.split_set}  mods={args.mods}")
    print(f"  out: {out_dir}")
    print(f"{'='*60}\n")

    # Optional cluster map: stem → cluster_id
    cluster_map: Optional[dict] = None
    if args.cluster_csv:
        import pandas as _pd
        _df = _pd.read_csv(args.cluster_csv)
        if "stem" in _df.columns and "cluster" in _df.columns:
            cluster_map = dict(zip(_df["stem"], _df["cluster"]))
            print(f"  Cluster map loaded: {len(cluster_map)} stems, "
                  f"{_df['cluster'].nunique()} clusters")

    if use_new:
        if args.no_checkpoint:
            bv = _NEW_VARIANTS[args.variant][0]
            model = build_model_v8(variant=bv, slot_k=args.slot_k, task="mega")
            model.to(DEVICE).eval()
            print(f"  Random-init {bv} (K={args.slot_k})")
        else:
            model = load_model_new(results_dir, args.split, args.fold,
                                   args.variant, args.slot_k)
        # Seeds live in model.pma[mod].seeds; use first present modality for init display
        first_mod = args.mods[0] if args.mods[0] in model.pma else list(model.pma.keys())[0]
        init_slots_np = model.pma[first_mod].seeds.detach().cpu().numpy()   # (K, H)
    else:
        if args.no_checkpoint:
            model = build_model_v8(variant="slot", slot_k=args.slot_k, task="mega")
            model.to(DEVICE).eval()
        else:
            model = load_model(results_dir, args.split, args.fold,
                               args.p2_tag, args.slot_k)
        init_slots_np = model.shared_slots.detach().cpu().numpy()

    print(f"  Init seeds std: {init_slots_np.std(0).mean():.5f}")

    splits_dict = build_splits_multitask(args.samples_dir, args.splits_csv,
                                          args.fold, split=args.split)
    if args.split_set == "all":
        records = (splits_dict["train"] + splits_dict["val"] +
                   splits_dict["test"])
    else:
        records = splits_dict[args.split_set]
    if args.n_patients > 0:
        records = records[:args.n_patients]
    print(f"  Patients: {len(records)}")

    stems     = [r["stem"] for r in records]
    bag_cache = preload_bags(stems, args.samples_dir)

    # Preload per-patch cluster labels for the seed×cluster heatmap
    # Maps stem → {mod: np.array of cluster name strings, shape (N_patches,)}
    MOD_CL_KEY = {"HE": "HE_cells", "BAL": "BAL_cells", "CT": "CT_cells"}
    cluster_label_cache: dict = {}
    if use_new:
        import torch as _torch
        _samples_path = Path(args.samples_dir)
        for stem in stems:
            pt_path = _samples_path / f"{stem}.pt"
            if not pt_path.exists():
                continue
            try:
                _d = _torch.load(pt_path, map_location="cpu", weights_only=False)
                _cl = _d.get("cluster_labels", {})
                cluster_label_cache[stem] = {
                    mod: np.array(_cl[key]) if _cl.get(key) is not None else None
                    for mod, key in MOD_CL_KEY.items()
                }
            except Exception:
                pass

    patient_results: List[dict] = []
    for i, rec in enumerate(records):
        bags = {m: bag_cache.get(rec["stem"], {}).get(m) for m in MODALITIES}
        try:
            if use_new:
                _cl_labels = cluster_label_cache.get(rec["stem"])
                r = extract_patient_new(model, bags, args.mods, args.max_patches,
                                        cluster_map, _cl_labels)
            else:
                r = extract_patient(model, bags, args.mods, args.max_patches)
        except Exception as e:
            print(f"  [warn] {rec['stem']}: {e}"); continue
        _lbl = rec.get("label", rec.get("acr_label", None))
        r["label_acr_cls"] = int(_lbl) if _lbl is not None else -1
        r["stem"] = rec["stem"]
        patient_results.append(r)
        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(records)}", flush=True)
        gc.collect()

    del model, bag_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not patient_results:
        print("  No patients processed — exiting"); return

    # For new models, remap attn key names so existing plot functions work
    if use_new:
        for r in patient_results:
            for mod in args.mods:
                if f"pma_attn_{mod}" in r:
                    r[f"slot_attn_{mod}"] = r[f"pma_attn_{mod}"]
                if f"seed_rep_{mod}" in r:
                    r[f"slot_rep_{mod}"] = r[f"seed_rep_{mod}"]

    # ── collapse stats ────────────────────────────────────────────────────────
    # For new models, alpha covers M*K tokens (all modalities concatenated).
    # Report per-modality seed collapse separately.
    if use_new:
        K = args.slot_k
        n_mods_present = sum(1 for m in args.mods
                             if any(f"slot_rep_{m}" in r for r in patient_results))
        print(f"\n  {'='*56}")
        print(f"  New-model collapse diagnostics  K={K}  total_tokens={K*n_mods_present}")
        print(f"  {'='*56}")
        for mod in args.mods:
            stds = [r[f"slot_rep_{mod}"].std(0).mean()
                    for r in patient_results if f"slot_rep_{mod}" in r]
            if stds:
                print(f"  {mod:10s} inter-seed std: mean={np.mean(stds):.5f}  "
                      f"min={np.min(stds):.5f}  (0=collapsed)")
        # Attention entropy per modality
        for mod in args.mods:
            ents = []
            for r in patient_results:
                w = r.get(f"pma_attn_{mod}")
                if w is None: continue
                w_mean = w.mean(0)   # (N,) — average over seeds
                w_mean = w_mean / (w_mean.sum() + 1e-9)
                ent = -np.sum(w_mean * np.log(w_mean + 1e-9))
                ents.append(ent)
            if ents:
                N_avg = np.mean([r.get(f"pma_attn_{mod}", np.zeros((1,1))).shape[1]
                                 for r in patient_results if f"pma_attn_{mod}" in r])
                print(f"  {mod:10s} patch attn entropy: {np.mean(ents):.3f}  "
                      f"(uniform=log({int(N_avg):.0f})={np.log(max(N_avg,1)):.2f})")
        print()
    else:
        print_collapse_stats(patient_results, args.mods, task=args.task)

    # Find k* from alpha difference
    a1_all = [r[f"alpha_{args.task}"] for r in patient_results
              if r.get("label_acr_cls") == 1 and f"alpha_{args.task}" in r]
    a0_all = [r[f"alpha_{args.task}"] for r in patient_results
              if r.get("label_acr_cls") == 0 and f"alpha_{args.task}" in r]
    # Keep only arrays with the most common shape (patients differ by #modalities present)
    def _modal_shape(arrs):
        from collections import Counter
        if not arrs: return None
        shapes = [a.shape for a in arrs]
        return Counter(shapes).most_common(1)[0][0]
    modal = _modal_shape(a1_all + a0_all)
    a1 = [a for a in a1_all if a.shape == modal] if modal else []
    a0 = [a for a in a0_all if a.shape == modal] if modal else []
    if a1 and a0:
        da     = np.stack(a1).mean(0) - np.stack(a0).mean(0)
        k_star = int(np.argmax(np.abs(da)))
    else:
        k_star = 0
    print(f"  k* = {k_star}")

    print("\n  Generating plots...")
    plot_feature_space_journey(patient_results, args.mods, init_slots_np,
                               k_star, args.task, out_dir)
    plot_routing_by_class(patient_results, args.mods, k_star, out_dir)
    plot_slot_alignment(patient_results, args.mods, args.task, out_dir)
    plot_seed_cluster_routing(patient_results, args.mods, out_dir)

    print(f"\n  All plots → {out_dir}")

    # ── wandb logging ─────────────────────────────────────────────────────────
    if not args.no_wandb:
        try:
            import wandb
            run_name = f"{suffix}_s{args.split}f{args.fold}_{args.split_set}"
            run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                group=f"attn_collapse_{suffix}",
                config={
                    "variant": suffix,
                    "split": args.split,
                    "fold": args.fold,
                    "split_set": args.split_set,
                    "slot_k": args.slot_k,
                    "task": args.task,
                    "n_patients": len(patient_results),
                    "mods": args.mods,
                    "no_checkpoint": args.no_checkpoint,
                },
                reinit=True,
            )

            # ── scalar collapse metrics ───────────────────────────────────────
            log_dict: dict = {"k_star": k_star}

            # inter-seed std per modality
            for mod in args.mods:
                stds = [r[f"slot_rep_{mod}"].std(0).mean()
                        for r in patient_results if f"slot_rep_{mod}" in r]
                if stds:
                    log_dict[f"collapse/inter_seed_std_{mod}"] = float(np.mean(stds))
                    log_dict[f"collapse/inter_seed_std_{mod}_min"] = float(np.min(stds))

            # PMA patch-attention entropy per modality
            for mod in args.mods:
                key = f"pma_attn_{mod}" if use_new else f"slot_attn_{mod}"
                ents = []
                N_vals = []
                for r in patient_results:
                    w = r.get(key)
                    if w is None: continue
                    w_mean = w.mean(0)
                    w_mean = w_mean / (w_mean.sum() + 1e-9)
                    ents.append(-np.sum(w_mean * np.log(w_mean + 1e-9)))
                    N_vals.append(w.shape[1])
                if ents:
                    N_avg = float(np.mean(N_vals))
                    log_dict[f"collapse/attn_entropy_{mod}"] = float(np.mean(ents))
                    log_dict[f"collapse/attn_entropy_{mod}_uniform"] = float(np.log(max(N_avg, 1)))
                    log_dict[f"collapse/attn_entropy_frac_{mod}"] = (
                        float(np.mean(ents)) / float(np.log(max(N_avg, 1))))

            # alpha entropy per class — align to modal shape first
            a1_raw = [r[f"alpha_{args.task}"] for r in patient_results
                      if r.get("label_acr_cls") == 1 and f"alpha_{args.task}" in r]
            a0_raw = [r[f"alpha_{args.task}"] for r in patient_results
                      if r.get("label_acr_cls") == 0 and f"alpha_{args.task}" in r]
            if a1_raw or a0_raw:
                from collections import Counter as _Counter
                _modal = _Counter([x.shape for x in a1_raw + a0_raw]).most_common(1)[0][0]
                a1 = [x for x in a1_raw if x.shape == _modal]
                a0 = [x for x in a0_raw if x.shape == _modal]
            else:
                a1, a0 = [], []
            K_total = _modal[0] if (a1 or a0) else args.slot_k
            if a1:
                m1 = np.stack(a1).mean(0)
                log_dict["collapse/alpha_entropy_acr_pos"] = float(
                    -np.sum(m1 * np.log(m1 + 1e-9)))
                log_dict[f"collapse/alpha_kstar_acr_pos"] = float(m1[k_star]) if k_star < len(m1) else float("nan")
            if a0:
                m0 = np.stack(a0).mean(0)
                log_dict["collapse/alpha_entropy_acr_neg"] = float(
                    -np.sum(m0 * np.log(m0 + 1e-9)))
                log_dict[f"collapse/alpha_kstar_acr_neg"] = float(m0[k_star]) if k_star < len(m0) else float("nan")
            if a1 and a0:
                log_dict["collapse/alpha_kstar_delta"] = float(m1[k_star] - m0[k_star]) if k_star < len(m1) and k_star < len(m0) else float("nan")
            log_dict["collapse/alpha_uniform"] = float(np.log(K_total))

            run.log(log_dict)

            # ── images ────────────────────────────────────────────────────────
            img_files = {
                "feature_space_journey":  out_dir / "feature_space_journey.png",
                "routing_by_class":       out_dir / "routing_by_class.png",
                "slot_alignment":         out_dir / "slot_alignment.png",
            }
            # per-modality seed_cluster_routing panels (one file per mod)
            for _mod in args.mods:
                _k = f"seed_cluster_routing_{_mod}"
                img_files[_k] = out_dir / f"seed_cluster_routing_{_mod}.png"
            img_log = {}
            for caption, path in img_files.items():
                if path.exists():
                    img_log[f"plots/{caption}"] = wandb.Image(
                        str(path),
                        caption=f"{suffix} s{args.split}f{args.fold} — {caption}")
            if img_log:
                run.log(img_log)
                print(f"  wandb: logged {len(img_log)} images to "
                      f"{args.wandb_project}/{run_name}")

            run.finish()
        except Exception as _e:
            print(f"  [wandb] error: {_e}")


if __name__ == "__main__":
    main()
