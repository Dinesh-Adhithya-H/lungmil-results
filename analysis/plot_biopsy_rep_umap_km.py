"""
Biopsy-level UMAP and Kaplan-Meier curves from extracted LongMK per-biopsy reps.

Reads: results/mm_abmil_v8/biopsy_reps/{task}_split{s}.pt  (5 splits × 4 tasks)

Outputs per task:
  figures/interpretability/{task}/biopsy_rep_umap.pdf/png
  figures/interpretability/{task}/biopsy_rep_umap.pdf/png   (colored variants)
  figures/km_curves/{task}_km_longi_biopsy.pdf/png

Usage: sbatch analysis/submit_biopsy_rep_umap_km.sh
       (Never run Python directly on the login node.)
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "interpretability"))
sys.path.insert(0, str(ROOT / "src"))

from shared import PDF_DPI, PNG_DPI, TASK_COLORS, TASK_LABELS

import umap as umap_lib
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

REPS_DIR = ROOT / "results" / "mm_abmil_v8" / "biopsy_reps"
FIG_ROOT  = ROOT / "figures"

TASK_LABELS_SHORT = {
    "acr_cls":   "ACR Classification",
    "acr_surv":  "ACR Survival",
    "clad_surv": "CLAD Survival",
    "death_surv":"Death Survival",
}

RISK_CMAP = "RdBu_r"
DAY_CMAP  = "viridis"

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.2,
    "figure.dpi":      120,
})


# ── Data loading ────────────────────────────────────────────────────────────────

def load_task(task_key: str):
    """Load and concatenate all 5 splits for a task."""
    all_data = []
    for split in range(5):
        p = REPS_DIR / f"biopsy_reps_{task_key}_split{split}.pt"
        if not p.exists():
            print(f"  Missing: {p}")
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        d["split_idx"] = np.full(len(d["patient_ids"]), split)
        all_data.append(d)

    if not all_data:
        return None

    merged = {
        "reps":        torch.cat([d["reps"]  for d in all_data], dim=0),
        "risk":        torch.cat([d["risk"]  for d in all_data], dim=0),
        "tte":         torch.cat([d["tte"]   for d in all_data], dim=0),
        "event":       torch.cat([d["event"] for d in all_data], dim=0),
        "label":       torch.cat([d["label"] for d in all_data], dim=0),
        "biopsy_days": torch.cat([d["biopsy_days"] for d in all_data], dim=0),
        "patient_ids": sum([d["patient_ids"] for d in all_data], []),
        "stems":       sum([d["stems"]       for d in all_data], []),
        "split_idx":   np.concatenate([d["split_idx"] for d in all_data]),
    }
    return merged


# ── UMAP ────────────────────────────────────────────────────────────────────────

def compute_umap(reps: torch.Tensor, n_neighbors: int = 15, seed: int = 42):
    X = reps.numpy().astype(np.float32)
    # L2-normalize for cosine distance
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.clip(norms, 1e-8, None)
    reducer = umap_lib.UMAP(n_neighbors=n_neighbors, n_components=2,
                             metric="euclidean", random_state=seed)
    emb = reducer.fit_transform(X)
    return emb


def plot_umap(data: dict, emb: np.ndarray, task_key: str, out_dir: Path):
    """6-panel UMAP (2×3): risk | day | split | event | TTE hexbin | ACR label hexbin."""
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Biopsy-Level Representations — {TASK_LABELS_SHORT[task_key]}",
                 fontsize=13, y=1.01)
    axes = axes.flatten()

    risk   = data["risk"].numpy()
    bdays  = data["biopsy_days"].numpy()
    splits = data["split_idx"]
    tte    = data["tte"].numpy()
    label  = data["label"].numpy()
    event  = data["event"].numpy()
    x, y   = emb[:, 0], emb[:, 1]

    # Panel 1: risk score
    ax = axes[0]
    sc = ax.scatter(x, y, c=risk, cmap=RISK_CMAP, s=5, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.04, label="Risk score")
    ax.set_title("Risk score"); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    # Panel 2: biopsy day
    ax = axes[1]
    sc = ax.scatter(x, y, c=bdays, cmap=DAY_CMAP, s=5, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.04, label="Days post-Tx")
    ax.set_title("Days post-transplant"); ax.set_xlabel("UMAP 1")

    # Panel 3: split
    ax = axes[2]
    sc = ax.scatter(x, y, c=splits, cmap=plt.cm.tab10, vmin=0, vmax=4,
                    s=5, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.04, label="Split", ticks=[0,1,2,3,4])
    ax.set_title("Outer split"); ax.set_xlabel("UMAP 1")

    # Panel 4: event / ACR label (scatter)
    ax = axes[3]
    if task_key == "acr_cls":
        mask_v = ~np.isnan(label)
        colors = np.where(label == 1, "#d32f2f", "#1976d2")
        ax.scatter(x[~mask_v], y[~mask_v], c="#cccccc", s=3, alpha=0.3, rasterized=True)
        ax.scatter(x[mask_v],  y[mask_v],  c=colors[mask_v], s=5, alpha=0.7, rasterized=True)
        ax.legend(handles=[Patch(color="#d32f2f", label="ACR+"),
                            Patch(color="#1976d2", label="ACR−")], fontsize=8)
        ax.set_title("ACR label (scatter)")
    else:
        mask_e = event == 1
        mask_c = (event == 0) & ~np.isnan(event)
        ax.scatter(x[mask_c], y[mask_c], c="#1976d2", s=3, alpha=0.4, label="Censored", rasterized=True)
        ax.scatter(x[mask_e], y[mask_e], c="#d32f2f", s=7, alpha=0.8, label="Event",    rasterized=True)
        ax.legend(fontsize=8)
        ax.set_title("Event status (scatter)")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    # Panel 5: TTE hexbin — colormap centered at median TTE
    ax = axes[4]
    from matplotlib.colors import TwoSlopeNorm
    mask_tte = ~np.isnan(tte)
    tte_v    = tte[mask_tte]
    med_tte  = float(np.nanmedian(tte_v))
    vmin_tte = float(np.nanmin(tte_v))
    vmax_tte = float(np.nanmax(tte_v))
    norm_tte = TwoSlopeNorm(vcenter=med_tte, vmin=vmin_tte, vmax=vmax_tte)
    hb = ax.hexbin(x[mask_tte], y[mask_tte], C=tte_v,
                   gridsize=35, cmap="RdBu", norm=norm_tte,
                   reduce_C_function=np.median, linewidths=0.2)
    cb = plt.colorbar(hb, ax=ax, fraction=0.04, label="Median TTE (days)")
    cb.ax.axhline(med_tte, color="k", lw=1.0, linestyle="--")
    ax.set_title(f"TTE hexbin (median-centered, med={med_tte:.0f}d)")
    ax.set_xlabel("UMAP 1")

    # Panel 6: ACR label hexbin (fraction ACR+ per bin)
    ax = axes[5]
    mask_lbl = ~np.isnan(label)
    if mask_lbl.sum() > 10:
        hb = ax.hexbin(x[mask_lbl], y[mask_lbl], C=label[mask_lbl],
                       gridsize=35, cmap="RdBu_r", vmin=0, vmax=1,
                       reduce_C_function=np.mean, linewidths=0.2)
        plt.colorbar(hb, ax=ax, fraction=0.04, label="Fraction ACR+")
        ax.set_title("ACR label hexbin (frac +)")
    else:
        ax.text(0.5, 0.5, "No ACR labels", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="#888")
        ax.set_title("ACR label hexbin")
    ax.set_xlabel("UMAP 1")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "biopsy_rep_umap"
    for ext, dpi in [(".pdf", PDF_DPI), (".png", PNG_DPI)]:
        fig.savefig(out_dir / f"{stem}{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved UMAP → {out_dir / stem}.pdf")


# ── Kaplan-Meier ────────────────────────────────────────────────────────────────

def plot_km(data: dict, task_key: str, out_dir: Path):
    """KM curves stratified by risk tertile (low/mid/high)."""
    if task_key == "acr_cls":
        print(f"  Skipping KM for {task_key} (classification task).")
        return

    risk  = data["risk"].numpy()
    tte   = data["tte"].numpy()
    event = data["event"].numpy()

    valid = (~np.isnan(tte)) & (~np.isnan(event)) & (~np.isnan(risk))
    if valid.sum() < 10:
        print(f"  Skipping KM for {task_key}: too few valid samples ({valid.sum()}).")
        return

    risk_v  = risk[valid]
    tte_v   = tte[valid]
    event_v = event[valid]

    q33 = np.percentile(risk_v, 33)
    q67 = np.percentile(risk_v, 67)
    low  = risk_v <= q33
    mid  = (risk_v > q33) & (risk_v <= q67)
    high = risk_v > q67

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {"Low":  "#1976d2", "Mid": "#ffa726", "High": "#d32f2f"}
    masks  = {"Low": low, "Mid": mid, "High": high}

    for grp, mask in masks.items():
        if mask.sum() < 3:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(tte_v[mask], event_v[mask], label=f"{grp} (n={mask.sum()})")
        kmf.plot_survival_function(ax=ax, color=colors[grp], ci_show=True)

    # log-rank p: low vs high
    if low.sum() >= 3 and high.sum() >= 3:
        lr = logrank_test(tte_v[low], tte_v[high],
                          event_observed_A=event_v[low],
                          event_observed_B=event_v[high])
        ax.text(0.98, 0.98, f"log-rank p = {lr.p_value:.3g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)

    ax.set_title(f"KM by Predicted Risk Tertile\n{TASK_LABELS_SHORT[task_key]} — LongMK (biopsy-level)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{task_key}_km_longi_biopsy"
    for ext, dpi in [(".pdf", PDF_DPI), (".png", PNG_DPI)]:
        fig.savefig(out_dir / f"{stem}{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved KM → {out_dir / stem}.pdf")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+",
                        default=["acr_cls","acr_surv","clad_surv","death_surv"])
    parser.add_argument("--n-neighbors", type=int, default=15)
    args = parser.parse_args()

    for task_key in args.tasks:
        print(f"\n{'='*60}\n{task_key}\n{'='*60}")
        data = load_task(task_key)
        if data is None or len(data["patient_ids"]) == 0:
            print(f"  No data found for {task_key}. Run extraction first.")
            continue

        N = len(data["patient_ids"])
        print(f"  Total biopsy points: N={N}")

        interp_dir = FIG_ROOT / "interpretability" / task_key
        km_dir     = FIG_ROOT / "km_curves"

        print("  Computing UMAP...")
        emb = compute_umap(data["reps"], n_neighbors=args.n_neighbors)

        print("  Plotting UMAP...")
        plot_umap(data, emb, task_key, interp_dir)

        print("  Plotting KM curves...")
        plot_km(data, task_key, km_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
