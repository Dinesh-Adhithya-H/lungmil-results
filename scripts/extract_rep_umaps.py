"""
extract_rep_umaps.py — UMAP of multi-level representations from mk_mt final model.

Extracts 4 representation levels for all test patients (split S, fold 0):
  1. patch_reps  : after ModalFFNEncoder, max PATCH_SAMPLE per patient per mod
  2. pma_seeds   : after PMA(h) — (K, H) per modality per patient
  3. post_sab    : after task-gated SAB — (M*K, H) per task per patient
  4. final_rep   : ABMIL weighted sum — (H,) per task per patient

Figures saved to OUT_DIR:
  A_patient_reps.pdf   — patient-level, prediction + ground-truth label/event
  B_postsab_tokens.pdf — post-SAB seeds: by modality + prediction
  C_pma_seeds.pdf      — PMA seeds before SAB: by modality
  D_patches.pdf        — sampled patch embeddings: by modality
  E_pre_post_sab_*.pdf — joint pre/post SAB embedding with displacement arrows
  F_tte_hexbin.pdf     — hexbin heatmaps: mean TTE and event rate per hex (beehive)

Color scheme (analysis/CLAUDE.md):
  prediction/hazard → RdBu_r (red=high risk, blue=low risk)
  TTE hexbin        → RdBu   (red=short TTE=high risk, blue=long TTE=low risk)
  event rate hexbin → Reds   (red=high event rate=high risk)
  ACR label / event → red #E53935 (positive), blue #1E88E5 (negative/censored)
  modality          → MOD_COLORS dict

Usage:
  sbatch scripts/submit_rep_umaps.sh
"""
from __future__ import annotations
import argparse, gc, sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from mil.data.loader   import preload_bags
from mil.data.splits   import build_splits_multitask
from mil.data.registry import MODALITIES, _feat_dim
from mil.models.builders import build_model_v8

# ── constants ─────────────────────────────────────────────────────────────────
SAMPLES_DIR = Path("/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples")
SPLITS_CSV  = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
RESULTS_DIR = _ROOT / "results/mm_abmil_v8"

PATCH_SAMPLE = 64    # patches sampled per modality per patient for UMAP D
MAX_HE       = 2048  # mirrors P2_MAX_PATCHES in builder

MOD_COLORS = {
    "HE":       "#E53935",
    "BAL":      "#1E88E5",
    "CT":       "#43A047",
    "Clinical": "#8E24AA",
}
TASK_LABELS = {
    "acr_cls":  "ACR rejection (BACC)",
    "acr_surv": "ACR survival (hazard)",
    "clad":     "CLAD survival (hazard)",
    "death":    "Death survival (hazard)",
}

# ── model loading ──────────────────────────────────────────────────────────────
def load_model(split: int, fold: int, device: torch.device):
    ckpt = RESULTS_DIR / f"phase2/split{split}_fold{fold}/set_mil_mt_mega/model_set_mil_mt_final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model = build_model_v8(variant="set_mil_mt", task="mega")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f"  Loaded {ckpt}")
    return model


# ── representation extraction ──────────────────────────────────────────────────
@torch.no_grad()
def extract_reps(model, bags: dict, device: torch.device, patch_sample: int = PATCH_SAMPLE):
    """
    Returns:
      patch_reps : {mod: (N_sample, H)} — sampled encoder output
      pma_seeds  : {mod: (K, H)} — after PMA
      post_sab   : {task: (M_present*K, H)} — after task-gated SAB
      final_rep  : {task: (H,)} — ABMIL weighted sum
      preds      : {task: float} — scalar prediction per task
    """
    import random
    patch_reps: Dict[str, np.ndarray] = {}
    pma_seeds:  Dict[str, np.ndarray] = {}

    present_mods: List[str] = []
    mod_seeds_t:  List[torch.Tensor] = []

    for mod, enc in model.encoders.items():
        t = bags.get(mod)
        if t is None:
            continue
        t = t.to(device, non_blocking=True)
        if mod == "HE" and t.shape[0] > MAX_HE:
            t = t[:MAX_HE]

        h = enc.encode_patches(t)               # (N, H)

        # sample patches for UMAP D
        N = h.shape[0]
        idx = torch.randperm(N, device=device)[:min(patch_sample, N)]
        patch_reps[mod] = h[idx].cpu().float().numpy()

        # PMA seeds
        s = model.pma[mod](h)                   # (K, H)
        mod_idx = torch.tensor(model._mod_idx[mod], device=device)
        s = s + model.modal_embed(mod_idx)
        pma_seeds[mod] = s.cpu().float().numpy()

        present_mods.append(mod)
        mod_seeds_t.append(s)

    if not present_mods:
        return None

    # task-gated SAB (mk_mt always uses gated path)
    presence = torch.tensor(
        [1.0 if m in present_mods else 0.0 for m in model._mod_order],
        dtype=torch.float32, device=device,
    )
    gates = model.task_gate(presence)

    post_sab:  Dict[str, np.ndarray] = {}
    final_rep: Dict[str, np.ndarray] = {}
    preds:     Dict[str, float]      = {}

    for task in model.task_names:
        gate_w = gates[task]
        gated = [mod_seeds_t[j] * gate_w[model._mod_idx[present_mods[j]]]
                 for j in range(len(present_mods))]
        tokens = torch.cat(gated, dim=0)        # (M*K, H)
        for layer in model.sab:
            tokens = layer(tokens)

        post_sab[task] = tokens.cpu().float().numpy()

        attn  = model.abmil_V[task](tokens) * model.abmil_U[task](tokens)
        alpha = torch.softmax(model.abmil_w[task](attn), dim=0)
        rep   = (alpha * tokens).sum(0)         # (H,)
        final_rep[task] = rep.cpu().float().numpy()

        logit = model.heads[task](rep).squeeze()
        preds[task] = float(torch.sigmoid(logit).item()
                           if task == "acr_cls"
                           else logit.item())

    return patch_reps, pma_seeds, post_sab, final_rep, preds


# ── UMAP fitting ───────────────────────────────────────────────────────────────
def fit_umap(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 42,
            pca_dims: int = 50):
    """PCA pre-reduction to pca_dims before UMAP — prevents the circular artifact
    that arises when all high-dim points are roughly equidistant."""
    from sklearn.decomposition import PCA
    if X.shape[1] > pca_dims:
        X = PCA(n_components=pca_dims, random_state=seed).fit_transform(X)
    try:
        import umap
        return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                         random_state=seed).fit_transform(X)
    except ImportError:
        from sklearn.manifold import TSNE
        perp = min(30, max(5, X.shape[0] // 5))
        return TSNE(n_components=2, random_state=seed, perplexity=perp,
                    n_iter=1000).fit_transform(X)


def _balance_by_modality(arrays: list, mod_ids: np.ndarray, rng: np.random.Generator,
                          max_per_mod: int = 5000):
    """Subsample each modality to max_per_mod tokens so no single modality dominates."""
    keep = []
    for m_id in np.unique(mod_ids):
        idx = np.where(mod_ids == m_id)[0]
        if len(idx) > max_per_mod:
            idx = rng.choice(idx, max_per_mod, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return arrays[keep], mod_ids[keep], keep


def _savefig(fig, path: Path):
    """Save figure as both PDF and PNG."""
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.with_suffix('.pdf')}  +  .png")


def _scatter(ax, xy, c, cmap, title, vmin=None, vmax=None, s=12, alpha=0.7,
             cbar=True, label_colors=None):
    if label_colors is not None:
        ax.scatter(xy[:, 0], xy[:, 1], c=label_colors, s=s, alpha=alpha,
                   linewidths=0)
    else:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=cmap, s=s, alpha=alpha,
                        vmin=vmin, vmax=vmax, linewidths=0)
        if cbar:
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")


# ── Figure A: patient-level final reps ────────────────────────────────────────
def plot_patient_reps(all_final: dict, all_preds: dict, all_labels: dict,
                      task_umaps: dict, out_dir: Path):
    """4×2 grid: rows = tasks, cols = [prediction UMAP | event/label UMAP].
    task_umaps: pre-computed {task: (N_pat, 2)} UMAP coordinates."""
    tasks = ["acr_cls", "acr_surv", "clad", "death"]
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    fig.suptitle("Patient-level final representations (mk_mt)\n"
                 "Left: model prediction  |  Right: ground-truth label/event",
                 fontsize=11, fontweight="bold")

    for row, task in enumerate(tasks):
        xy    = task_umaps[task]
        preds = np.array(all_preds[task])

        # Left: prediction
        cmap = "RdBu_r"
        vmin, vmax = np.percentile(preds, 1), np.percentile(preds, 99)
        _scatter(axes[row, 0], xy, preds, cmap, f"{task} — prediction (red=high risk)",
                 vmin=vmin, vmax=vmax, s=18)

        # Right: ground truth
        labels = all_labels[task]                     # list of (label, is_event, tte)
        evt  = np.array([l[1] for l in labels], dtype=float)
        lbl  = np.array([l[0] for l in labels], dtype=float)
        if task == "acr_cls":
            colors = ["#E53935" if v == 1 else ("#1E88E5" if v == 0 else "#AAAAAA")
                      for v in lbl]
            _scatter(axes[row, 1], xy, None, None, f"{task} — label (red=ACR+, blue=ACR−)",
                     s=18, label_colors=colors, cbar=False)
        else:
            colors = ["#E53935" if e == 1 else "#1E88E5" for e in evt]
            _scatter(axes[row, 1], xy, None, None,
                     f"{task} — event (red=event, blue=censored)",
                     s=18, label_colors=colors, cbar=False)

    fig.tight_layout()
    _savefig(fig, out_dir / "A_patient_reps")


# ── Figure F: TTE / event hexbin heatmaps ─────────────────────────────────────
def plot_tte_hexbin(all_labels: dict, all_preds: dict, task_umaps: dict, out_dir: Path,
                   gridsize: int = 25):
    """
    Beehive hexbin heatmaps on the patient-level UMAP embedding.

    For each task, 3 panels:
      [mean TTE per hex]  [event rate per hex]  [model prediction per hex]

    Hexagon color = mean value of all patients that fall inside it.
    Hexagons with no patients are transparent (mincnt=1).

    TTE color:   RdBu (red=short TTE=high risk, blue=long TTE=low risk)
    Event color: Reds (red=high event rate=high risk)
    Pred color:  RdBu_r (red=high hazard/prob=high risk)

    For acr_cls: TTE panel replaced by label fraction (ACR+ rate per hex).
    """
    tasks = ["acr_cls", "acr_surv", "clad", "death"]
    TASK_TTE_LABEL = {
        "acr_cls":  "ACR+ fraction",
        "acr_surv": "mean TTE (days)",
        "clad":     "mean TTE to CLAD (days)",
        "death":    "mean TTE to death (days)",
    }

    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    fig.suptitle("Patient representation space — TTE / event hexbin heatmaps\n"
                 "Each hexagon = mean value of all patients inside it",
                 fontsize=12, fontweight="bold")

    for row, task in enumerate(tasks):
        xy    = task_umaps[task]                      # (N_pat, 2)
        x, y  = xy[:, 0], xy[:, 1]
        preds = np.array(all_preds[task])             # (N_pat,)
        labels = all_labels[task]                     # list of (label, event, tte)

        evt_arr = np.array([l[1] if l[1] is not None else np.nan for l in labels], dtype=float)
        tte_arr = np.array([l[2] if l[2] is not None else np.nan for l in labels], dtype=float)
        lbl_arr = np.array([l[0] if l[0] is not None else np.nan for l in labels], dtype=float)

        # ── Panel 0: TTE (or label fraction for acr_cls) ──────────────────────
        ax = axes[row, 0]
        if task == "acr_cls":
            # Use label: 1=ACR+, 0=ACR-, NaN=excluded
            valid = ~np.isnan(lbl_arr)
            c_vals = np.where(valid, lbl_arr, np.nan)
            hb = ax.hexbin(x, y, C=c_vals, gridsize=gridsize,
                           reduce_C_function=np.nanmean,
                           cmap="RdBu_r", mincnt=1, linewidths=0.2)
            cb = plt.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label("ACR+ fraction", fontsize=7)
            ax.set_title(f"{task}\nACR+ fraction per hex (red=high)", fontsize=8, fontweight="bold")
        else:
            valid = ~np.isnan(tte_arr)
            c_vals = np.where(valid, tte_arr, np.nan)
            # Clip extreme TTE at 99th percentile for colour scale
            vmax_tte = np.nanpercentile(c_vals[valid], 99) if valid.any() else 1.0
            hb = ax.hexbin(x, y, C=c_vals, gridsize=gridsize,
                           reduce_C_function=np.nanmean,
                           cmap="RdBu", vmin=0, vmax=vmax_tte,
                           mincnt=1, linewidths=0.2)
            cb = plt.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label("days", fontsize=7)
            ax.set_title(f"{task}\n{TASK_TTE_LABEL[task]} (red=short=high risk)",
                         fontsize=8, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

        # ── Panel 1: event rate per hex ────────────────────────────────────────
        ax = axes[row, 1]
        valid = ~np.isnan(evt_arr)
        c_evt = np.where(valid, evt_arr, np.nan)
        hb = ax.hexbin(x, y, C=c_evt, gridsize=gridsize,
                       reduce_C_function=np.nanmean,
                       cmap="Reds", vmin=0, vmax=1,
                       mincnt=1, linewidths=0.2)
        cb = plt.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("event rate", fontsize=7)
        ax.set_title(f"{task}\nevent rate per hex (red=high)", fontsize=8, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

        # ── Panel 2: model prediction per hex ─────────────────────────────────
        ax = axes[row, 2]
        vmin_p = np.percentile(preds, 1); vmax_p = np.percentile(preds, 99)
        hb = ax.hexbin(x, y, C=preds, gridsize=gridsize,
                       reduce_C_function=np.nanmean,
                       cmap="RdBu_r", vmin=vmin_p, vmax=vmax_p,
                       mincnt=1, linewidths=0.2)
        cb = plt.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("prediction", fontsize=7)
        ax.set_title(f"{task}\nmodel prediction per hex (red=high risk)",
                     fontsize=8, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

    fig.tight_layout()
    _savefig(fig, out_dir / "F_tte_hexbin")


# ── Figure B: post-SAB tokens ─────────────────────────────────────────────────
def plot_postsab(all_postsab: dict, all_preds: dict, all_mod_labels: dict, out_dir: Path):
    """Two panels per task: by modality | by task prediction."""
    rng = np.random.default_rng(42)
    tasks = list(all_postsab.keys())
    fig, axes = plt.subplots(len(tasks), 2, figsize=(10, 4 * len(tasks)))
    if len(tasks) == 1: axes = axes[None]
    fig.suptitle("Post-SAB seed tokens (mk_mt)\n"
                 "Left: modality identity  |  Right: task prediction mapped onto token space",
                 fontsize=11, fontweight="bold")

    for row, task in enumerate(tasks):
        tokens = np.concatenate(all_postsab[task], axis=0)    # (N_pat*M*K, H)
        mod_ids = np.concatenate(all_mod_labels[task], axis=0) # (N_pat*M*K,) int
        preds_tok = np.concatenate(
            [np.full(p.shape[0], v) for p, v in
             zip(all_postsab[task], all_preds[task])], axis=0)

        # Balance modalities so no single one dominates
        tokens, mod_ids, keep = _balance_by_modality(tokens, mod_ids, rng)
        preds_tok = preds_tok[keep]

        print(f"  Fitting UMAP for post-SAB {task} ({tokens.shape}) ...", flush=True)
        xy = fit_umap(tokens, n_neighbors=30)

        # Left: modality colour
        mod_list = list(MOD_COLORS.keys())
        c_mod = [list(MOD_COLORS.values())[m] for m in mod_ids]
        _scatter(axes[row, 0], xy, None, None, f"{task} post-SAB — modality",
                 s=4, alpha=0.4, label_colors=c_mod, cbar=False)
        from matplotlib.patches import Patch
        axes[row, 0].legend(
            handles=[Patch(color=MOD_COLORS[m], label=m) for m in mod_list if m in MOD_COLORS],
            fontsize=7, loc="lower right", framealpha=0.7)

        # Right: prediction
        vmin, vmax = np.percentile(preds_tok, 1), np.percentile(preds_tok, 99)
        _scatter(axes[row, 1], xy, preds_tok, "RdBu_r",
                 f"{task} post-SAB — {task} prediction",
                 vmin=vmin, vmax=vmax, s=4, alpha=0.4)

    fig.tight_layout()
    _savefig(fig, out_dir / "B_postsab_tokens")


# ── Figure C: PMA seeds ────────────────────────────────────────────────────────
def plot_pma_seeds(all_pma: dict, out_dir: Path):
    """One UMAP of all PMA seeds, coloured by modality."""
    rng = np.random.default_rng(42)
    tokens = np.concatenate([np.concatenate(v) for v in all_pma.values()], axis=0)
    mod_ids = np.concatenate([
        np.full(sum(s.shape[0] for s in v), i)
        for i, v in enumerate(all_pma.values())
    ])
    tokens, mod_ids, _ = _balance_by_modality(tokens, mod_ids, rng)

    print(f"  Fitting UMAP for PMA seeds ({tokens.shape}) ...", flush=True)
    xy = fit_umap(tokens, n_neighbors=30)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    fig.suptitle("PMA seeds before SAB — coloured by modality", fontsize=11, fontweight="bold")
    mods = list(all_pma.keys())
    c = [MOD_COLORS.get(mods[i], "#999") for i in mod_ids]
    ax.scatter(xy[:, 0], xy[:, 1], c=c, s=4, alpha=0.4, linewidths=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=MOD_COLORS[m], label=m) for m in mods if m in MOD_COLORS],
              fontsize=9, framealpha=0.8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.tight_layout()
    _savefig(fig, out_dir / "C_pma_seeds")


# ── Figure D: patch-level embeddings ──────────────────────────────────────────
def plot_patches(all_patches: dict, out_dir: Path):
    """One UMAP of sampled patch embeddings, coloured by modality."""
    rng = np.random.default_rng(42)
    mods = [m for m in MODALITIES if m in all_patches and len(all_patches[m]) > 0]
    patches = np.concatenate([np.concatenate(all_patches[m]) for m in mods], axis=0)
    mod_ids = np.concatenate([
        np.full(sum(p.shape[0] for p in all_patches[m]), i)
        for i, m in enumerate(mods)
    ])
    patches, mod_ids, _ = _balance_by_modality(patches, mod_ids, rng)

    print(f"  Fitting UMAP for patch embeddings ({patches.shape}) ...", flush=True)
    xy = fit_umap(patches, n_neighbors=30)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    fig.suptitle("Sampled patch embeddings after ModalFFNEncoder\n"
                 "coloured by modality", fontsize=11, fontweight="bold")
    c = [MOD_COLORS.get(mods[i], "#999") for i in mod_ids]
    ax.scatter(xy[:, 0], xy[:, 1], c=c, s=4, alpha=0.35, linewidths=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=MOD_COLORS[m], label=m) for m in mods],
              fontsize=9, framealpha=0.8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.tight_layout()
    _savefig(fig, out_dir / "D_patches")


# ── Figure E: pre vs post SAB joint embedding ────────────────────────────────
def plot_pre_post_sab(all_pma: dict, all_postsab: dict, all_mod_labels: dict,
                      out_dir: Path, task: str = "acr_cls"):
    """
    Joint UMAP of pre-SAB (PMA seeds) and post-SAB tokens for one task.
    Both sets embedded together so positions are comparable.

    4 panels:
      [pre by modality]  [post by modality]
      [pre vs post stage — grey=pre, colour=post]  [displacement arrows (mean per modality)]
    """
    rng = np.random.default_rng(42)
    mods = list(all_pma.keys())

    # Build pre-SAB tokens (modality-balanced)
    pre_tokens = np.concatenate([np.concatenate(all_pma[m]) for m in mods], axis=0)
    pre_mod_ids = np.concatenate([
        np.full(sum(s.shape[0] for s in all_pma[m]), i) for i, m in enumerate(mods)
    ])
    pre_tokens, pre_mod_ids, _ = _balance_by_modality(pre_tokens, pre_mod_ids, rng, max_per_mod=3000)

    # Build post-SAB tokens for chosen task (modality-balanced)
    post_tokens = np.concatenate(all_postsab[task], axis=0)
    post_mod_ids = np.concatenate(all_mod_labels[task], axis=0)
    post_tokens, post_mod_ids, _ = _balance_by_modality(post_tokens, post_mod_ids, rng, max_per_mod=3000)

    # Joint UMAP — embed both together so positions are in the same space
    combined = np.concatenate([pre_tokens, post_tokens], axis=0)
    n_pre = len(pre_tokens)
    print(f"  Fitting joint UMAP pre+post SAB ({combined.shape}) ...", flush=True)
    xy = fit_umap(combined, n_neighbors=30)
    xy_pre  = xy[:n_pre]
    xy_post = xy[n_pre:]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Pre-SAB vs Post-SAB token space  [task={task}]\n"
                 "Joint UMAP — same coordinate system for both",
                 fontsize=11, fontweight="bold")

    from matplotlib.patches import Patch

    # Panel 1: pre-SAB by modality
    ax = axes[0]
    c_pre = [MOD_COLORS.get(mods[i], "#999") for i in pre_mod_ids]
    ax.scatter(xy_pre[:, 0],  xy_pre[:, 1],  c=c_pre,  s=5, alpha=0.5, linewidths=0)
    ax.set_title("Pre-SAB (PMA seeds)\ncoloured by modality", fontsize=9, fontweight="bold")
    ax.legend(handles=[Patch(color=MOD_COLORS[m], label=m) for m in mods if m in MOD_COLORS],
              fontsize=7, framealpha=0.8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

    # Panel 2: post-SAB by modality
    ax = axes[1]
    c_post = [MOD_COLORS.get(mods[i], "#999") for i in post_mod_ids]
    ax.scatter(xy_post[:, 0], xy_post[:, 1], c=c_post, s=5, alpha=0.5, linewidths=0)
    ax.set_title(f"Post-SAB ({task})\ncoloured by modality", fontsize=9, fontweight="bold")
    ax.legend(handles=[Patch(color=MOD_COLORS[m], label=m) for m in mods if m in MOD_COLORS],
              fontsize=7, framealpha=0.8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

    # Panel 3: overlay — grey pre, coloured post
    ax = axes[2]
    ax.scatter(xy_pre[:, 0],  xy_pre[:, 1],  c="#CCCCCC", s=3, alpha=0.3,
               linewidths=0, label="pre-SAB")
    ax.scatter(xy_post[:, 0], xy_post[:, 1], c=c_post,   s=5, alpha=0.5,
               linewidths=0, label="post-SAB")
    # Draw mean displacement arrow per modality
    for i, m in enumerate(mods):
        pre_m  = xy_pre [pre_mod_ids  == i]
        post_m = xy_post[post_mod_ids == i]
        if len(pre_m) == 0 or len(post_m) == 0:
            continue
        mu_pre  = pre_m.mean(0)
        mu_post = post_m.mean(0)
        ax.annotate("", xy=mu_post, xytext=mu_pre,
                    arrowprops=dict(arrowstyle="->", color=MOD_COLORS.get(m, "#333"),
                                   lw=2.0))
        ax.text(mu_post[0], mu_post[1], m, fontsize=7,
                color=MOD_COLORS.get(m, "#333"), fontweight="bold")
    ax.set_title("Overlay: grey=pre, colour=post\narrows = mean displacement per modality",
                 fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

    fig.tight_layout()
    _savefig(fig, out_dir / f"E_pre_post_sab_{task}")


# ── main ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",       type=int, default=0)
    p.add_argument("--fold",        type=int, default=0)
    p.add_argument("--split-set",   default="test", choices=["train", "val", "test"])
    p.add_argument("--patch-sample",type=int, default=PATCH_SAMPLE,
                   help="Max patches sampled per patient per modality (UMAP D)")
    p.add_argument("--samples-dir", default=str(SAMPLES_DIR))
    p.add_argument("--splits-csv",  default=str(SPLITS_CSV))
    return p.parse_args()


def main():
    args  = parse_args()
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    out_dir = RESULTS_DIR / f"analysis/rep_umaps/split{args.split}_fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {out_dir}\n")

    # Load model
    model = load_model(args.split, args.fold, device)

    # Load test records
    splits = build_splits_multitask(args.samples_dir, args.splits_csv,
                                    args.fold, split=args.split)
    records = splits[args.split_set]
    print(f"  {args.split_set} records: {len(records)}\n")

    stems = [r["stem"] for r in records]
    bag_cache = preload_bags(stems, args.samples_dir)

    # Collector arrays
    tasks = ["acr_cls", "acr_surv", "clad", "death"]
    all_final:    Dict[str, List] = {t: [] for t in tasks}
    all_preds:    Dict[str, List] = {t: [] for t in tasks}
    all_labels:   Dict[str, List] = {t: [] for t in tasks}
    all_postsab:  Dict[str, List] = {t: [] for t in tasks}
    all_mod_lbls: Dict[str, List] = {t: [] for t in tasks}  # modality id per token
    all_patches:  Dict[str, List] = {m: [] for m in MODALITIES}
    all_pma:      Dict[str, List] = {m: [] for m in MODALITIES}

    K = model.n_seeds

    for i, rec in enumerate(records):
        entry = bag_cache.get(rec["stem"], {})
        bags  = {m: entry.get(m) for m in MODALITIES}
        bags["HE_coords"] = entry.get("HE_coords")

        result = extract_reps(model, bags, device, args.patch_sample)
        if result is None:
            continue
        patch_reps, pma_seeds, post_sab, final_rep, preds = result

        for task in tasks:
            all_final[task].append(final_rep[task])
            all_preds[task].append(preds[task])
            # labels/events
            if task == "acr_cls":
                all_labels[task].append((rec.get("label"), None, None))
            elif task == "acr_surv":
                all_labels[task].append((None, rec.get("acr_status", rec.get("event_next_acr")),
                                         rec.get("acr_days", rec.get("tte_next_acr"))))
            elif task == "clad":
                all_labels[task].append((None, rec.get("clad_event"), rec.get("clad_time")))
            elif task == "death":
                all_labels[task].append((None, rec.get("death_event"), rec.get("death_time")))

            # post-SAB token mod labels
            # tokens are ordered: present_mods in model._mod_order order
            present = [m for m in model._mod_order if bags.get(m) is not None]
            mod_ids = np.concatenate([
                np.full(K, model._mod_idx[m]) for m in present
            ])
            all_postsab[task].append(post_sab[task])
            all_mod_lbls[task].append(mod_ids)

        for mod, arr in patch_reps.items():
            all_patches[mod].append(arr)
        for mod, arr in pma_seeds.items():
            all_pma[mod].append(arr)

        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(records)}", flush=True)
        gc.collect()

    del model, bag_cache
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"\n  Collected {len(all_final['acr_cls'])} patients\n")

    # Pre-compute per-task patient UMAP (shared by Fig A and Fig F)
    print("  Fitting per-task patient UMAP embeddings ...")
    task_umaps: Dict[str, np.ndarray] = {}
    for task in tasks:
        reps = np.stack(all_final[task])
        print(f"    {task}: {reps.shape}", flush=True)
        task_umaps[task] = fit_umap(reps)

    # Plot
    print("  Plotting A: patient final reps ...")
    plot_patient_reps(all_final, all_preds, all_labels, task_umaps, out_dir)

    print("  Plotting F: TTE / event hexbin heatmaps ...")
    plot_tte_hexbin(all_labels, all_preds, task_umaps, out_dir)

    print("  Plotting B: post-SAB tokens ...")
    plot_postsab(all_postsab, all_preds, all_mod_lbls, out_dir)

    print("  Plotting C: PMA seeds ...")
    pma_nonempty = {m: v for m, v in all_pma.items() if v}
    if pma_nonempty:
        plot_pma_seeds(pma_nonempty, out_dir)

    print("  Plotting D: patch embeddings ...")
    patches_nonempty = {m: v for m, v in all_patches.items() if v}
    if patches_nonempty:
        plot_patches(patches_nonempty, out_dir)

    print("  Plotting E: pre vs post SAB joint embedding ...")
    pma_nonempty2 = {m: v for m, v in all_pma.items() if v}
    if pma_nonempty2:
        for task in tasks:
            plot_pre_post_sab(pma_nonempty2, all_postsab, all_mod_lbls, out_dir, task=task)

    print(f"\n  Done — all figures saved to {out_dir}\n")


if __name__ == "__main__":
    main()
