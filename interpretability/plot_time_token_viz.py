"""
Visualization of learned time tokens from LongMIL (longitudinal_mk_no_alibi_v2).

The time token maps 2D input (anchor_day, sample_day) → 4-dim vector via
biopsy_time_net[task]: Linear(2→16) → ReLU → Linear(16→4).

Plots:
  1. Input space: scatter of (anchor_day, sample_day) pairs per test patient,
     colored by patient risk score / days_diff / patient trajectory
  2. UMAP of 4-dim time tokens, colored by anchor_day, sample_day, and diff
  3. 4 task × 2 coloring → 2×4 panel

Output: figures/interpretability/time_token_viz/
Usage: sbatch interpretability/submit_time_token_viz.sh
"""

import json, math, sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mil.models.builders import build_model_v8
from mil.data.splits import build_splits_longitudinal
from interpretability.shared import SPLITS_CSV, SAMPLES_DIR, RESULTS_ROOT

PHASE2_ROOT = ROOT / "results" / "mm_abmil_v8" / "phase2"
OUT_ROOT    = ROOT / "figures" / "interpretability" / "time_token_viz"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

VARIANT  = "longitudinal_mk_no_alibi_v2"
LOG_MAX  = math.log(1 + 1825.0)

BG = "#FAF6F2"

TASK_CFG = {
    "acr_cls":   {"dir": "acr_cls",   "label": "ACR classification", "color": "#E07B20"},
    "acr_surv":  {"dir": "acr_surv",  "label": "Time to first ACR",  "color": "#952030"},
    "clad_surv": {"dir": "clad_surv", "label": "Time to CLAD",       "color": "#1A5C8A"},
    "death_surv":{"dir": "death_surv","label": "Time to death",      "color": "#2D8A4E"},
}

FS_TITLE  = 18
FS_LABEL  = 14
FS_TICK   = 12
FS_ANNOT  = 12

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": FS_TICK,
    "axes.titlesize": FS_LABEL, "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
})


def log_norm(d):
    return math.log(1 + d) / LOG_MAX


TASK_DIR = {
    "acr_cls":   "cls",
    "acr_surv":  "acr_surv",
    "clad_surv": "clad_surv",
    "death_surv":"death_surv",
}
TASK_BUILD = {
    "acr_cls":   "cls",
    "acr_surv":  "acr_surv",
    "clad_surv": "clad_surv",
    "death_surv":"death_surv",
}
TASK_INTERNAL = {
    "acr_cls":   "acr_cls",
    "acr_surv":  "acr_surv",
    "clad_surv": "clad",
    "death_surv":"death",
}


def load_model(task_key, split, device):
    task_dir = TASK_DIR[task_key]
    ckpt = (PHASE2_ROOT / f"split{split}_fold0"
            / f"{VARIANT}_{task_dir}" / f"model_{VARIANT}_final.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint: {ckpt}")

    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    # Infer n_seeds from checkpoint
    n_seeds = 8
    for k, v in state.items():
        if k.endswith(".seeds") and isinstance(v, torch.Tensor):
            if v.ndim == 3: state[k] = v.squeeze(0)
            n_seeds = state[k].shape[0]
            break

    model = build_model_v8(variant=VARIANT, task=TASK_BUILD[task_key],
                           slot_k=n_seeds, n_cross_layers=1)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


def extract_time_tokens(model, splits, task_key, device):
    """
    For each test biopsy pair (anchor_biopsy, prev_biopsy), extract:
      - anchor_day_norm, prev_day_norm (input features)
      - time_vec (4-dim output)
      - days_diff = anchor_day - prev_day
      - patient_id

    Returns dict with arrays.
    """
    time_net = model.biopsy_time_net[TASK_INTERNAL[task_key]]

    anchor_norms, prev_norms, diffs, vecs, pids = [], [], [], [], []

    test_pats = splits["test"]
    with torch.no_grad():
        for pat in test_pats:
            days = pat["days"]   # list of floats (days from transplant for each biopsy)
            pid  = pat.get("patient_id", "?")
            T    = len(days)

            for t_idx in range(T):
                anchor_day = float(days[t_idx])
                # Previous biopsies: indices 0..t_idx (the anchor is at t_idx)
                n_prev = t_idx + 1
                feats = torch.tensor(
                    [[log_norm(anchor_day), log_norm(float(days[i]))]
                     for i in range(n_prev)],
                    dtype=torch.float32, device=device,
                )   # (n_prev, 2)
                tvecs = time_net(feats).cpu().numpy()   # (n_prev, 4)

                for i in range(n_prev):
                    anchor_norms.append(log_norm(anchor_day))
                    prev_norms.append(log_norm(float(days[i])))
                    diffs.append(anchor_day - float(days[i]))
                    vecs.append(tvecs[i])
                    pids.append(pid)

    return {
        "anchor_norm": np.array(anchor_norms),
        "prev_norm":   np.array(prev_norms),
        "diff_days":   np.array(diffs),
        "vecs":        np.array(vecs),        # (N, 4)
        "pids":        pids,
    }


def umap_embed(vecs):
    from umap import UMAP
    reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                   metric="cosine", random_state=42)
    return reducer.fit_transform(vecs)


def plot_input_space(ax, data, title, task_color):
    """Scatter of (anchor_norm, prev_norm) colored by diff_days."""
    diff = data["diff_days"]
    sc = ax.scatter(data["anchor_norm"], data["prev_norm"],
                    c=diff, cmap="viridis", s=4, alpha=0.4, linewidths=0)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("Anchor day (log-norm)", fontsize=FS_LABEL)
    ax.set_ylabel("Sample day (log-norm)", fontsize=FS_LABEL)
    ax.set_title(f"{title}\nInput space", fontsize=FS_LABEL, color=task_color, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.set_facecolor(BG)
    ax.spines[["top","right"]].set_visible(False)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Days diff (anchor−sample)", fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=FS_TICK)


def plot_umap_colored(ax, xy, c_vals, cmap, label, title, task_color):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=c_vals, cmap=cmap,
                    s=5, alpha=0.5, linewidths=0)
    ax.set_title(f"{title}\nUMAP — colored by {label}",
                 fontsize=FS_LABEL, color=task_color, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=FS_LABEL)
    ax.set_ylabel("UMAP 2", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_facecolor(BG)
    ax.spines[["top","right"]].set_visible(False)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label(label, fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=FS_TICK)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Use split 0 only (user request: one split, focused)
    SPLIT = 0
    splits = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=SPLIT)
    print(f"Test patients: {len(splits['test'])}", flush=True)

    # Per-task extraction and plotting
    # 3-panel per task: input space | UMAP colored by anchor | UMAP colored by diff
    for task_key, cfg in TASK_CFG.items():
        print(f"\n=== {cfg['label']} ===", flush=True)
        try:
            model = load_model(task_key, SPLIT, device)
        except FileNotFoundError as e:
            print(f"  [skip] {e}", flush=True)
            continue

        data = extract_time_tokens(model, splits, task_key, device)
        n = len(data["anchor_norm"])
        print(f"  {n} (anchor, prev) pairs", flush=True)

        if n < 10:
            print("  Too few points, skip", flush=True)
            continue

        print("  Running UMAP...", flush=True)
        xy = umap_embed(data["vecs"])

        fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor=BG)
        fig.patch.set_facecolor(BG)

        plot_input_space(axes[0], data, cfg["label"], cfg["color"])
        plot_umap_colored(axes[1], xy, data["anchor_norm"], "plasma",
                          "Anchor day (log-norm)", cfg["label"], cfg["color"])
        plot_umap_colored(axes[2], xy, data["diff_days"], "viridis",
                          "Days diff (anchor−sample)", cfg["label"], cfg["color"])

        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT_ROOT / f"time_token_{task_key}.{ext}",
                        dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"  Saved time_token_{task_key}", flush=True)
        del model; torch.cuda.empty_cache()

    # Combined 4-task summary: input space only (quick overview)
    print("\n=== Combined input space figure ===", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(18, 13), facecolor=BG)
    fig.patch.set_facecolor(BG)
    axes = axes.flatten()

    for ax, (task_key, cfg) in zip(axes, TASK_CFG.items()):
        try:
            model = load_model(task_key, SPLIT, device)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "No checkpoint", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(cfg["label"]); continue

        data = extract_time_tokens(model, splits, task_key, device)
        plot_input_space(ax, data, cfg["label"], cfg["color"])
        del model; torch.cuda.empty_cache()

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_ROOT / f"time_token_input_space_all.{ext}",
                    dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    print(f"\nDone. → {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
