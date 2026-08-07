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


# ── Main plot: 2×4 matching patient rep ────────────────────────────────────────

def plot_biopsy_umap(data: dict, emb: np.ndarray, task_key: str, out_dir: Path):
    from matplotlib.patches import Patch

    is_surv = task_key != "acr_cls"

    risk    = data["risk"].numpy()
    bdays   = data["biopsy_days"].numpy()
    splits  = data["split_idx"]
    tte     = data["tte"].numpy()
    label   = data["label"].numpy()
    event   = data["event"].numpy()
    pids    = np.array(data["patient_ids"])
    N       = len(pids)
    n_pts   = len(set(pids.tolist()))
    x, y    = emb[:, 0], emb[:, 1]

    # Risk score: rank-percentile for survival, sigmoid for cls
    valid_r = ~np.isnan(risk)
    scores  = np.full_like(risk, float("nan"))
    if is_surv:
        if valid_r.sum() > 1:
            scores[valid_r] = (rankdata(risk[valid_r]) - 1) / max(valid_r.sum() - 1, 1)
        score_lbl = "Risk percentile"
    else:
        scores[valid_r] = 1.0 / (1.0 + np.exp(-risk[valid_r]))
        score_lbl = "P(ACR+)"

    # TTE in years for display
    tte_yrs = tte / 365.25

    fig = plt.figure(figsize=(24, 11), facecolor=BG)
    fig.suptitle(
        f"Biopsy-Level Representation Space  —  {TASK_LABELS[task_key]}"
        f"  ({N} biopsies, {n_pts} patients)",
        fontsize=12, fontweight="bold", y=0.98, color="#1A1018")
    grd = gridspec.GridSpec(2, 4, figure=fig, wspace=0.28, hspace=0.42)

    ax_score   = fig.add_subplot(grd[0, 0])
    ax_ev      = fig.add_subplot(grd[0, 1])
    ax_tte     = fig.add_subplot(grd[0, 2])
    ax_ev_hex  = fig.add_subplot(grd[0, 3])
    ax_days    = fig.add_subplot(grd[1, 0])
    ax_km      = fig.add_subplot(grd[1, 1])
    ax_split   = fig.add_subplot(grd[1, 2])
    ax_tte_hex = fig.add_subplot(grd[1, 3])

    def _sc(ax, vals, cmap, title, vmin=None, vmax=None, alpha=0.75, sz=MS):
        valid = ~np.isnan(vals)
        sc = ax.scatter(x[valid], y[valid], c=vals[valid], cmap=cmap,
                        s=sz, alpha=alpha, linewidths=0,
                        vmin=(vmin if vmin is not None else float(np.nanmin(vals))),
                        vmax=(vmax if vmax is not None else float(np.nanmax(vals))))
        if (~valid).any():
            ax.scatter(x[~valid], y[~valid], c="lightgrey",
                       s=sz * 0.6, alpha=0.25, linewidths=0)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046, shrink=0.85)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("UMAP-1", fontsize=8); ax.set_ylabel("UMAP-2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_facecolor(BG)
        ax.spines[["top","right"]].set_visible(False)
        return cb

    # ① Risk score
    _sc(ax_score, scores, "RdBu_r", f"① Risk score\n({score_lbl})", vmin=0, vmax=1)

    # ② Biopsy timing relative to event (pre/post for survival; label for cls)
    if is_surv:
        # pre-event biopsy (tte>0) vs post-event (tte<0) vs censored (ev=0)
        pre_ev  = (~np.isnan(tte)) & (tte > 0) & (event == 1)   # before event, event happened
        post_ev = (~np.isnan(tte)) & (tte < 0) & (event == 1)   # after event already occurred
        cens    = (~np.isnan(tte)) & (event == 0)                # censored
        no_data = np.isnan(tte) | np.isnan(event)
        ax_ev.scatter(x[no_data],  y[no_data],  c="lightgrey", s=MS*0.5, alpha=0.25, linewidths=0)
        ax_ev.scatter(x[cens],     y[cens],     c=LO,          s=MS,     alpha=0.6,  linewidths=0, label="censored")
        ax_ev.scatter(x[pre_ev],   y[pre_ev],   c=HI,          s=MS,     alpha=0.85, linewidths=0, label="pre-event")
        ax_ev.scatter(x[post_ev],  y[post_ev],  c="#FF8A00",   s=MS,     alpha=0.85, linewidths=0, label="post-event")
        ax_ev.legend(fontsize=6.5, framealpha=0.8, markerscale=1.3)
        ax_ev.set_title("② Biopsy timing vs event\n(red=pre-event, orange=post, blue=censored)",
                        fontsize=9, fontweight="bold", pad=4)
    else:
        _sc(ax_ev, label, "RdBu_r",
            "② True label\n(positive=red, negative=blue)", vmin=0, vmax=1)
    ax_ev.set_xlabel("UMAP-1", fontsize=8); ax_ev.set_ylabel("UMAP-2", fontsize=8)
    ax_ev.tick_params(labelsize=7); ax_ev.set_facecolor(BG)
    ax_ev.spines[["top","right"]].set_visible(False)

    # ③ Biopsy-level TTE scatter (only prospective biopsies, tte>0)
    valid_t = ~np.isnan(tte_yrs) & (tte > 0)   # prospective only
    if is_surv and valid_t.sum() > 5:
        tte_pos = tte_yrs[valid_t]
        tte_clip = np.clip(tte_pos, 0, np.nanpercentile(tte_pos, 95))
        sc_tte = ax_tte.scatter(
            x[valid_t], y[valid_t],
            c=tte_clip, cmap="plasma_r",
            s=MS, alpha=0.80, linewidths=0,
            vmin=0, vmax=float(np.nanpercentile(tte_clip, 95)))
        cb_tte = fig.colorbar(sc_tte, ax=ax_tte, pad=0.02, fraction=0.046, shrink=0.85)
        cb_tte.ax.tick_params(labelsize=7); cb_tte.set_label("TTE (years)", fontsize=7)
        ev_pre  = valid_t & (event == 1)
        cen_pre = valid_t & (event == 0)
        ax_tte.scatter(x[ev_pre],  y[ev_pre],  marker="x", s=18,
                       color=HI, linewidths=0.9, alpha=0.9, label="event", zorder=3)
        ax_tte.scatter(x[cen_pre], y[cen_pre], marker="o", s=6,
                       color="none", edgecolors=LO, linewidths=0.5, alpha=0.5,
                       label="censored", zorder=2)
        ax_tte.legend(fontsize=6.5, framealpha=0.75, loc="lower right", markerscale=1.2)
        ax_tte.set_title("③ Biopsy-level TTE (prospective only)\n(short=bright, ×=event, ○=censored)",
                         fontsize=9, fontweight="bold", pad=4)
        # Grey out post-event biopsies
        post_e = (~np.isnan(tte_yrs)) & (tte < 0)
        if post_e.sum() > 0:
            ax_tte.scatter(x[post_e], y[post_e], c="#cccccc", s=MS*0.5, alpha=0.3,
                           linewidths=0, label="post-event", zorder=1)
    else:
        ev_cols = np.where(event == 1, HI, LO)
        valid_all = ~np.isnan(event)
        ax_tte.scatter(x[valid_all], y[valid_all], c=ev_cols[valid_all],
                       s=MS, alpha=0.75, linewidths=0)
        ax_tte.set_title("③ TTE / event\n(red=event, blue=censored)",
                         fontsize=9, fontweight="bold", pad=4)
    ax_tte.set_xlabel("UMAP-1", fontsize=8); ax_tte.set_ylabel("UMAP-2", fontsize=8)
    ax_tte.tick_params(labelsize=7); ax_tte.set_facecolor(BG)
    ax_tte.spines[["top","right"]].set_visible(False)

    # ④ Event density hexbin — prospective biopsies only (tte>0) for survival
    valid_ev = ~np.isnan(event) & (tte > 0 if is_surv else np.ones(N, bool))
    if valid_ev.sum() > 10:
        hx_ev = ax_ev_hex.hexbin(
            x[valid_ev], y[valid_ev], C=event[valid_ev],
            reduce_C_function=np.mean,
            gridsize=30, cmap="RdBu_r", linewidths=0.2)
        hex_vals = hx_ev.get_array()
        med_ev   = float(np.nanmedian(hex_vals))
        half_ev  = max(abs(float(hex_vals.max()) - med_ev),
                       abs(med_ev - float(hex_vals.min())), 1e-6)
        hx_ev.set_clim(med_ev - half_ev, med_ev + half_ev)
        cb_ev = fig.colorbar(hx_ev, ax=ax_ev_hex, pad=0.02, fraction=0.046, shrink=0.85)
        cb_ev.set_label(f"Event rate (med={med_ev:.2f})", fontsize=7)
        cb_ev.ax.tick_params(labelsize=7)
        label_iv = "④ Event density" if is_surv else "④ Positive label density"
        ax_ev_hex.set_title(f"{label_iv}\n(median-centred, red=above median)",
                            fontsize=9, fontweight="bold", pad=4)
    else:
        ax_ev_hex.text(0.5, 0.5, "no event data", ha="center", va="center",
                       transform=ax_ev_hex.transAxes)
    ax_ev_hex.set_xlabel("UMAP-1", fontsize=8); ax_ev_hex.set_ylabel("UMAP-2", fontsize=8)
    ax_ev_hex.tick_params(labelsize=7); ax_ev_hex.set_facecolor("#f0ede8")
    ax_ev_hex.spines[["top","right"]].set_visible(False)

    # ⑤ Days post-transplant (biopsy-specific)
    valid_d = ~np.isnan(bdays)
    bdays_yrs = bdays / 365.25
    if valid_d.sum() > 10:
        _sc(ax_days, bdays_yrs, "viridis",
            "⑤ Days post-transplant\n(biopsy timeline)", alpha=0.75)
        ax_days.collections[-1]  # colorbar already added by _sc
    else:
        ax_days.text(0.5, 0.5, "no biopsy day data", ha="center", va="center",
                     transform=ax_days.transAxes)
        ax_days.set_title("⑤ Days post-transplant", fontsize=9, fontweight="bold", pad=4)
        ax_days.set_facecolor(BG)
        ax_days.spines[["top","right"]].set_visible(False)

    # ⑥ KM: top vs bottom risk tertile — biopsy-level (prospective biopsies only, tte>0)
    ax_km.set_facecolor(BG)
    ax_km.spines[["top","right"]].set_visible(False)
    if is_surv:
        b_risk, b_tte, b_ev = biopsy_km_data(data)
        if len(b_risk) >= 20:
            q33, q67 = np.nanpercentile(b_risk, [33, 67])
            hi_m = b_risk >= q67
            lo_m = b_risk <= q33
            if hi_m.sum() > 5 and lo_m.sum() > 5:
                _km_step(b_ev[hi_m], b_tte[hi_m] / 365.25, HI, ax_km,
                         f"High risk (n={hi_m.sum()})")
                _km_step(b_ev[lo_m], b_tte[lo_m] / 365.25, LO, ax_km,
                         f"Low risk (n={lo_m.sum()})")
                if hi_m.sum() >= 3 and lo_m.sum() >= 3:
                    lr = logrank_test(b_tte[hi_m], b_tte[lo_m],
                                      event_observed_A=b_ev[hi_m],
                                      event_observed_B=b_ev[lo_m])
                    ax_km.text(0.98, 0.98, f"log-rank p={lr.p_value:.3g}",
                               transform=ax_km.transAxes, ha="right", va="top", fontsize=8)
                ax_km.set_xlabel("Time from biopsy (years)", fontsize=8)
                ax_km.set_ylabel("Event-free probability", fontsize=8)
                ax_km.legend(fontsize=7, framealpha=0.8)
                ax_km.set_ylim(-0.05, 1.05)
    else:
        ax_km.text(0.5, 0.5, "KM not applicable\n(classification task)",
                   ha="center", va="center", transform=ax_km.transAxes, fontsize=9)
    ax_km.set_title("⑥ KM: top vs bottom risk tertile\n(biopsy-level, tte>0 prospective only)",
                    fontsize=9, fontweight="bold", pad=4)
    ax_km.tick_params(labelsize=7)

    # ⑦ CV split
    uniq_splits = sorted(set(splits.tolist()))
    for si, sp in enumerate(uniq_splits):
        idx = np.where(splits == sp)[0]
        col = SPLIT_COLS[si % len(SPLIT_COLS)]
        ax_split.scatter(x[idx], y[idx], color=col, s=MS, alpha=0.7,
                         linewidths=0, label=f"s{int(sp)}")
    ax_split.set_title("⑦ CV split annotation", fontsize=9, fontweight="bold", pad=4)
    ax_split.set_xlabel("UMAP-1", fontsize=8); ax_split.set_ylabel("UMAP-2", fontsize=8)
    ax_split.tick_params(labelsize=7); ax_split.set_facecolor(BG)
    ax_split.spines[["top","right"]].set_visible(False)
    ax_split.legend(fontsize=6.5, framealpha=0.8, ncol=2, markerscale=1.2)

    # ⑧ Avg TTE hexbin (median-centred)
    valid_th = ~np.isnan(tte_yrs)
    if valid_th.sum() > 10:
        tte_clip_hex = np.clip(tte_yrs, 0, np.nanpercentile(tte_yrs[valid_th], 95))
        hx_tte = ax_tte_hex.hexbin(
            x[valid_th], y[valid_th],
            C=tte_clip_hex[valid_th],
            reduce_C_function=np.mean,
            gridsize=30, cmap="RdBu", linewidths=0.2)
        tte_hex_v = hx_tte.get_array()
        med_th    = float(np.nanmedian(tte_hex_v))
        half_th   = max(abs(float(tte_hex_v.max()) - med_th),
                        abs(med_th - float(tte_hex_v.min())), 1e-6)
        hx_tte.set_clim(med_th - half_th, med_th + half_th)
        cb_th = fig.colorbar(hx_tte, ax=ax_tte_hex, pad=0.02, fraction=0.046, shrink=0.85)
        cb_th.set_label(f"Mean TTE yrs (med={med_th:.1f})", fontsize=7)
        cb_th.ax.tick_params(labelsize=7)
    else:
        ax_tte_hex.text(0.5, 0.5, "no TTE data", ha="center", va="center",
                        transform=ax_tte_hex.transAxes)
    ax_tte_hex.set_title("⑧ Avg TTE per hexagon\n(median-centred, blue=longer TTE)",
                         fontsize=9, fontweight="bold", pad=4)
    ax_tte_hex.set_xlabel("UMAP-1", fontsize=8); ax_tte_hex.set_ylabel("UMAP-2", fontsize=8)
    ax_tte_hex.tick_params(labelsize=7); ax_tte_hex.set_facecolor("#f0ede8")
    ax_tte_hex.spines[["top","right"]].set_visible(False)

    fig.patch.set_facecolor(BG)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

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
