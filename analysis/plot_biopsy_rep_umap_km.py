"""
Biopsy-level UMAP and Kaplan-Meier curves from extracted LongMK per-biopsy reps.

Reads: results/mm_abmil_v8/biopsy_reps/biopsy_reps_{task}_split{s}.pt  (5 splits × 4 tasks)

Outputs per task:
  figures/interpretability/{task}/biopsy_rep_umap.pdf/png

Usage: sbatch analysis/submit_biopsy_rep_umap_km.sh
       (Never run Python directly on the login node.)
"""

import argparse
import sys
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "interpretability"))
sys.path.insert(0, str(ROOT / "src"))

from shared import PDF_DPI, PNG_DPI

import umap as umap_lib
from lifelines.statistics import logrank_test

REPS_DIR = ROOT / "results" / "mm_abmil_v8" / "biopsy_reps"
FIG_ROOT  = ROOT / "figures"

TASK_LABELS = {
    "acr_cls":   "ACR Classification",
    "acr_surv":  "ACR Survival",
    "clad_surv": "CLAD Survival",
    "death_surv":"Death Survival",
}

# Match patient rep palette
HI         = "#952030"
LO         = "#1A5C8A"
BG         = "#FAF6F2"
SPLIT_COLS = ["#E41A1C","#377EB8","#4DAF4A","#984EA3","#FF7F00"]
MS         = 7     # marker size for biopsy scatter (slightly smaller than patient rep)

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi":      120,
})


# ── Data loading ────────────────────────────────────────────────────────────────

def load_task(task_key: str):
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
        "reps":        torch.cat([d["reps"]        for d in all_data], dim=0),
        "risk":        torch.cat([d["risk"]        for d in all_data], dim=0),
        "tte":         torch.cat([d["tte"]         for d in all_data], dim=0),
        "event":       torch.cat([d["event"]       for d in all_data], dim=0),
        "label":       torch.cat([d["label"]       for d in all_data], dim=0),
        "biopsy_days": torch.cat([d["biopsy_days"] for d in all_data], dim=0),
        "patient_ids": sum([d["patient_ids"] for d in all_data], []),
        "stems":       sum([d["stems"]       for d in all_data], []),
        "split_idx":   np.concatenate([d["split_idx"] for d in all_data]),
    }
    return merged


# ── UMAP ────────────────────────────────────────────────────────────────────────

def compute_umap(reps: torch.Tensor, n_neighbors: int = 15, seed: int = 42):
    X = reps.numpy().astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.clip(norms, 1e-8, None)
    reducer = umap_lib.UMAP(n_neighbors=n_neighbors, n_components=2,
                             metric="euclidean", random_state=seed)
    return reducer.fit_transform(X)


# ── KM helper (patient-dedup) ───────────────────────────────────────────────────

def _km_step(ev, tte, color, ax, label=""):
    """Simple KM step plot (expects 1-D numpy arrays)."""
    order = np.argsort(tte)
    ev_o, tte_o = ev[order], tte[order]
    at_risk = len(ev_o)
    surv = 1.0
    t_pts, s_pts = [0.0], [1.0]
    for i in range(len(ev_o)):
        if ev_o[i] == 1:
            surv *= (at_risk - 1) / at_risk
        at_risk -= 1
        t_pts.append(tte_o[i])
        s_pts.append(surv)
    ax.step(t_pts, s_pts, where="post", color=color, lw=1.8, label=label)


def biopsy_km_data(data: dict):
    """
    Build biopsy-level KM inputs.

    tte[i] = event_date - biopsy_date (days):
      > 0 : biopsy before event → valid prospective observation
      < 0 : biopsy after event  → exclude (event already occurred)
      nan : no outcome data

    For censored patients (event==0), tte > 0 is time to last follow-up.
    """
    risk  = data["risk"].numpy()
    tte   = data["tte"].numpy()
    event = data["event"].numpy()
    # Keep only biopsies with a valid positive TTE (prospective biopsies)
    valid = (~np.isnan(risk)) & (~np.isnan(tte)) & (~np.isnan(event)) & (tte > 0)
    return risk[valid], tte[valid], event[valid]


# ── Main plot: 1×3 compact (risk | event-rate hexbin | avg-TTE hexbin) ──────────

def _ax_style(ax, title, fig_ref, hexbin_obj=None, cbar_label="", bg=BG):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("UMAP-1", fontsize=9)
    ax.set_ylabel("UMAP-2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_facecolor(bg)
    ax.spines[["top", "right"]].set_visible(False)
    if hexbin_obj is not None:
        cb = fig_ref.colorbar(hexbin_obj, ax=ax, pad=0.02, fraction=0.046, shrink=0.88)
        cb.set_label(cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        return cb


def plot_biopsy_umap(data: dict, emb: np.ndarray, task_key: str, out_dir: Path):
    is_surv = task_key != "acr_cls"

    risk   = data["risk"].numpy()
    tte    = data["tte"].numpy()
    label  = data["label"].numpy()
    event  = data["event"].numpy()
    pids   = np.array(data["patient_ids"])
    N      = len(pids)
    n_pts  = len(set(pids.tolist()))
    x, y   = emb[:, 0], emb[:, 1]

    # ── Risk / cls score ─────────────────────────────────────────────────────────
    valid_r = ~np.isnan(risk)
    scores  = np.full_like(risk, float("nan"))
    if is_surv:
        if valid_r.sum() > 1:
            scores[valid_r] = (rankdata(risk[valid_r]) - 1) / max(valid_r.sum() - 1, 1)
        score_lbl = "Risk percentile"
    else:
        scores[valid_r] = 1.0 / (1.0 + np.exp(-risk[valid_r]))
        score_lbl = "P(ACR+)"

    tte_yrs = tte / 365.25

    # Masks for hexbins — events only (event==1), ignore censored
    if is_surv:
        # event rate hexbin: prospective biopsies with event=1 OR event=0, but only
        # biopsies where the patient eventually has the event (event==1)
        evt_mask     = (~np.isnan(event)) & (event == 1)          # only eventual events
        # avg TTE hexbin: prospective (tte>0) AND event==1 (censored excluded)
        tte_hex_mask = evt_mask & (~np.isnan(tte)) & (tte > 0)
    else:
        # cls: event rate = label (positive class), censored n/a
        evt_mask     = ~np.isnan(label)
        tte_hex_mask = np.zeros(N, dtype=bool)   # not meaningful for cls

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=BG)
    fig.suptitle(
        f"Biopsy Representation UMAP  —  {TASK_LABELS[task_key]}"
        f"  ({N} biopsies, {n_pts} patients)",
        fontsize=12, fontweight="bold", y=1.01, color="#1A1018")

    # ① Risk / cls score scatter
    ax1 = axes[0]
    valid_s = ~np.isnan(scores)
    sc = ax1.scatter(x[valid_s], y[valid_s], c=scores[valid_s], cmap="RdBu_r",
                     s=MS, alpha=0.75, linewidths=0, vmin=0, vmax=1)
    if (~valid_s).any():
        ax1.scatter(x[~valid_s], y[~valid_s], c="lightgrey",
                    s=MS * 0.6, alpha=0.25, linewidths=0)
    cb1 = fig.colorbar(sc, ax=ax1, pad=0.02, fraction=0.046, shrink=0.88)
    cb1.set_label(score_lbl, fontsize=8); cb1.ax.tick_params(labelsize=7)
    _ax_style(ax1, f"① {score_lbl}", fig)

    # ② Event-rate hexbin — only biopsies where event occurs (event==1), censored excluded
    ax2 = axes[1]
    gs = max(10, min(30, int(np.sqrt(evt_mask.sum()) // 2)))
    if evt_mask.sum() > 10:
        if is_surv:
            # count density of event biopsies — shows where events cluster in UMAP
            hx2 = ax2.hexbin(x[evt_mask], y[evt_mask], gridsize=gs,
                              cmap="Reds", linewidths=0.2, mincnt=1)
            _ax_style(ax2, "② Event density\n(event biopsies only, censored excluded)",
                      fig, hx2, "Biopsy count")
        else:
            hx2 = ax2.hexbin(x[evt_mask], y[evt_mask], C=label[evt_mask],
                              reduce_C_function=np.mean, gridsize=gs,
                              cmap="RdBu_r", linewidths=0.2)
            hx2.set_clim(0, 1)
            _ax_style(ax2, "② ACR+ label density\n(labelled biopsies only)",
                      fig, hx2, "Mean label (0–1)")
    else:
        ax2.text(0.5, 0.5, "no event data", ha="center", va="center",
                 transform=ax2.transAxes)
        _ax_style(ax2, "② Event density", fig)

    # ③ Avg TTE hexbin — event biopsies only (event==1), prospective (tte>0), censored excluded
    ax3 = axes[2]
    if is_surv and tte_hex_mask.sum() > 10:
        gs3 = max(10, min(30, int(np.sqrt(tte_hex_mask.sum()) // 2)))
        tte_v = tte_yrs[tte_hex_mask]
        tte_clip = np.clip(tte_v, 0, np.nanpercentile(tte_v, 95))
        hx3 = ax3.hexbin(x[tte_hex_mask], y[tte_hex_mask], C=tte_clip,
                          reduce_C_function=np.mean, gridsize=gs3,
                          cmap="RdBu", linewidths=0.2)
        tte_hex_v = hx3.get_array()
        med = float(np.nanmedian(tte_hex_v))
        half = max(abs(float(tte_hex_v.max()) - med),
                   abs(med - float(tte_hex_v.min())), 1e-6)
        vmin = max(0.0, med - half)   # TTE cannot be negative
        hx3.set_clim(vmin, med + half)
        n_ev = int(tte_hex_mask.sum())
        _ax_style(ax3, f"③ Avg TTE to event (yrs)\n(event biopsies only, n={n_ev}, blue=longer)",
                  fig, hx3, f"Mean TTE yrs (med={med:.1f})", bg="#f0ede8")
    elif not is_surv:
        ax3.text(0.5, 0.5, "TTE not applicable\n(classification task)",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=9)
        _ax_style(ax3, "③ Avg TTE", fig)
    else:
        ax3.text(0.5, 0.5, "no TTE data", ha="center", va="center",
                 transform=ax3.transAxes)
        _ax_style(ax3, "③ Avg TTE", fig)

    fig.patch.set_facecolor(BG)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "biopsy_rep_umap"
    for ext, dpi in [("pdf", PDF_DPI), ("png", PNG_DPI)]:
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight",
                    facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {out_dir / stem}.pdf")


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
            print(f"  No data found for {task_key}.")
            continue

        N = len(data["patient_ids"])
        n_pts = len(set(data["patient_ids"]))
        print(f"  Biopsies: {N},  Patients: {n_pts}")

        out_dir = FIG_ROOT / "interpretability" / task_key
        print("  Computing UMAP...")
        emb = compute_umap(data["reps"], n_neighbors=args.n_neighbors)

        print("  Plotting...")
        plot_biopsy_umap(data, emb, task_key, out_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
