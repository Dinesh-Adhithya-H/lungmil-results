"""
Plot 5-split averaged biopsy time-weighting heatmap for longitudinal_mk_no_alibi.

For each task (acr_surv, clad_surv, death_surv):
  - Load biopsy_weight_net weights from all 5 split checkpoints (CPU-only, small MLP)
  - Evaluate w(curr_day, prev_day) on 100x100 grid
  - Average across splits; compute std
  - Plot mean heatmap + std heatmap side by side

Run via: sbatch analysis/submit_L_global_avg.sh
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = Path(__file__).resolve().parent.parent
CKPT_ROOT = ROOT / "results" / "mm_abmil_v8" / "phase2"
OUT_ROOT  = ROOT / "figures" / "interpretability"
BG = "#FAF6F2"

TASKS = {
    "acr_cls":   {"ckpt_dir": "longitudinal_mk_no_alibi_cls",        "net_key": "acr_cls",
                  "label": "ACR classification", "color": "#6A1B9A"},
    "acr_surv":  {"ckpt_dir": "longitudinal_mk_no_alibi_acr_surv",   "net_key": "acr_surv",
                  "label": "ACR survival",       "color": "#1565C0"},
    "clad_surv": {"ckpt_dir": "longitudinal_mk_no_alibi_clad_surv",  "net_key": "clad",
                  "label": "CLAD survival",      "color": "#2E7D32"},
    "death_surv":{"ckpt_dir": "longitudinal_mk_no_alibi_death_surv", "net_key": "death",
                  "label": "Death survival",     "color": "#952030"},
}

GRID_N   = 100
DAY_MAX  = 2000


def make_net():
    return nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())


def load_net(ckpt_path: Path, net_key: str) -> nn.Module:
    state = torch.load(ckpt_path, map_location="cpu")
    net = make_net()
    prefix = f"biopsy_weight_net.{net_key}."
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    net.load_state_dict(sub)
    net.eval()
    return net


def eval_grid(net: nn.Module) -> np.ndarray:
    curr_range = np.linspace(0, DAY_MAX, GRID_N)
    prev_range = np.linspace(0, DAY_MAX, GRID_N)
    CURR, PREV = np.meshgrid(curr_range, prev_range, indexing="ij")
    pairs = np.stack([CURR.ravel(), PREV.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        W = net(torch.from_numpy(pairs)).squeeze(1).numpy().reshape(GRID_N, GRID_N)
    W[PREV > CURR] = np.nan
    return W


def compute_split_matrices(task_key: str, cfg: dict):
    mats = []
    for split in range(5):
        ckpt = CKPT_ROOT / f"split{split}_fold0" / cfg["ckpt_dir"] / "model_longitudinal_mk_no_alibi_final.pt"
        if not ckpt.exists():
            print(f"  MISSING: {ckpt}")
            continue
        net = load_net(ckpt, cfg["net_key"])
        mats.append(eval_grid(net))
        print(f"  split {split} done")
    if not mats:
        return None, None, 0
    arr = np.stack(mats)
    mean = np.nanmean(arr, axis=0)
    std  = np.nanstd(arr, axis=0)
    return mean, std, len(mats)


def plot_task(task_key: str, cfg: dict, mean: np.ndarray, std: np.ndarray, n_splits: int):
    out_dir = OUT_ROOT / task_key
    out_dir.mkdir(parents=True, exist_ok=True)

    extent = [0, DAY_MAX, 0, DAY_MAX]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Learned biopsy time-weighting — {cfg['label']}  (mean ± std, n={n_splits} splits)",
                 fontsize=12, fontweight="bold")

    # Mean
    ax = axes[0]
    ax.set_facecolor(BG)
    im = ax.imshow(mean, origin="lower", aspect="auto", extent=extent,
                   cmap="RdBu_r", vmin=0, vmax=1, interpolation="bilinear")
    ax.plot([0, DAY_MAX], [0, DAY_MAX], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Previous biopsy date (days post-transplant)", fontsize=10)
    ax.set_ylabel("Current biopsy date (days post-transplant)", fontsize=10)
    ax.set_title("Mean weight  w ∈ (0,1)", fontsize=10, color=cfg["color"], fontweight="bold")
    fig.colorbar(im, ax=ax, label="Mean weight", pad=0.02)

    # Std
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    vmax_std = np.nanpercentile(std, 95) if np.any(~np.isnan(std)) else 0.2
    im2 = ax2.imshow(std, origin="lower", aspect="auto", extent=extent,
                     cmap="Oranges", vmin=0, vmax=max(vmax_std, 0.05),
                     interpolation="bilinear")
    ax2.plot([0, DAY_MAX], [0, DAY_MAX], "k--", lw=1, alpha=0.5)
    ax2.set_xlabel("Previous biopsy date (days post-transplant)", fontsize=10)
    ax2.set_ylabel("Current biopsy date (days post-transplant)", fontsize=10)
    ax2.set_title("Std across splits", fontsize=10, color="#555", fontweight="bold")
    fig.colorbar(im2, ax=ax2, label="Std weight", pad=0.02)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"L_global_weight_heatmap_avg.{ext}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
        print(f"  → {path}")
    plt.close(fig)


def plot_combined(results: dict):
    tasks_with_data = [(k, cfg, r["mean"], r["std"], r["n"])
                       for (k, cfg), r in zip(TASKS.items(), results.values())
                       if r["mean"] is not None]
    if not tasks_with_data:
        print("No data to combine")
        return

    n = len(tasks_with_data)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10), facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.suptitle("Learned biopsy time-weighting — all tasks (5-split mean ± std)",
                 fontsize=13, fontweight="bold")
    extent = [0, DAY_MAX, 0, DAY_MAX]

    for ti, (task_key, cfg, mean, std, n_splits) in enumerate(tasks_with_data):
        # Mean row
        ax = axes[0, ti]
        ax.set_facecolor(BG)
        im = ax.imshow(mean, origin="lower", aspect="auto", extent=extent,
                       cmap="RdBu_r", vmin=0, vmax=1, interpolation="bilinear")
        ax.plot([0, DAY_MAX], [0, DAY_MAX], "k--", lw=1, alpha=0.4)
        ax.set_title(f"{cfg['label']}\nMean weight (n={n_splits})", fontsize=10,
                     color=cfg["color"], fontweight="bold")
        if ti == 0:
            ax.set_ylabel("Current biopsy date (days)", fontsize=9)
        ax.set_xlabel("Previous biopsy date (days)", fontsize=9)
        fig.colorbar(im, ax=ax, label="w", pad=0.02, shrink=0.8)

        # Std row
        ax2 = axes[1, ti]
        ax2.set_facecolor(BG)
        vmax_std = max(np.nanpercentile(std, 95) if np.any(~np.isnan(std)) else 0.1, 0.05)
        im2 = ax2.imshow(std, origin="lower", aspect="auto", extent=extent,
                         cmap="Oranges", vmin=0, vmax=vmax_std, interpolation="bilinear")
        ax2.plot([0, DAY_MAX], [0, DAY_MAX], "k--", lw=1, alpha=0.4)
        ax2.set_title("Std across splits", fontsize=9, color="#555")
        if ti == 0:
            ax2.set_ylabel("Current biopsy date (days)", fontsize=9)
        ax2.set_xlabel("Previous biopsy date (days)", fontsize=9)
        fig.colorbar(im2, ax=ax2, label="σ", pad=0.02, shrink=0.8)

    fig.tight_layout()
    out = OUT_ROOT / "agg" / "L_global_weight_heatmap_avg_all.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", facecolor=BG)
    print(f"  → {out}")
    plt.close(fig)


if __name__ == "__main__":
    results = {}
    for task_key, cfg in TASKS.items():
        print(f"\n=== {task_key} ===")
        mean, std, n = compute_split_matrices(task_key, cfg)
        results[task_key] = {"mean": mean, "std": std, "n": n}
        if mean is not None:
            plot_task(task_key, cfg, mean, std, n)

    print("\n=== Combined ===")
    plot_combined(results)
    print("\nDone.")
