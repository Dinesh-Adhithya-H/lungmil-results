"""
UMAP visualizations — task-type-aware panels.

Classification tasks (acr_cls, acr_alt_cls):
  fig_umap_{task}_{variant}_cls_test.pdf — 3×3 cls panel on r_cls embedding:
    Row 0: [ACR label red/blue | modality combo | # modalities]
    Row 1: [cls_prob high=red  | fold           | hexbin cls_prob]
    Row 2: [hexbin proportion+ | hexbin error   | hexbin # modalities]

TTE / survival tasks (acr_tte, acr_alt_tte, clad, death):
  fig_umap_{task}_{variant}_tte_test.pdf — 3×3 tte panel on r_tte embedding:
    Row 0: [event vs censored red/blue | modality combo | # modalities]
    Row 1: [hazard high=red | TTE events only short=red | fold]
    Row 2: [hexbin event density | hexbin mean TTE | hexbin mean hazard]

Multitask (acr_alt_cls/acr_alt_tte share same checkpoint):
  Both cls and tte panels generated — each from its own head embedding.

Cross-variant summaries:
  fig_umap_{task}_comparison.pdf
  fig_umap_{task}_event_vs_censored.pdf  (tte tasks only)
  fig_umap_{task}_patient_test.pdf       (patient-level mean-pool, tte head)
"""
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ..config import (COMBO_COLORS, DEFAULT_COMBO_COLOR, ENDPOINT, FOLD_COLORS,
                      NATURE_RC, TASKS, TWO_COL_W, VARIANT_DISPLAY, VARIANT_TAGS,
                      CMAP_HAZARD, CMAP_TTE, CMAP_DENSITY)
from ..io import ordered_tags
from ..inference import get_or_run

try:
    from umap import UMAP as UMAPTransform
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(arr, lo=1, hi=99):
    v = arr[~np.isnan(arr)]
    if len(v) == 0:
        return 0.0, 1.0
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def _median_norm(arr, vmin, vmax):
    """TwoSlopeNorm centered at the data median — white maps to typical value."""
    v = arr[~np.isnan(arr.astype(float))]
    if len(v) == 0 or vmax <= vmin:
        return None
    med = float(np.median(v))
    eps = max(1e-6 * abs(vmax - vmin), 1e-10)
    med = float(np.clip(med, vmin + eps, vmax - eps))
    try:
        return mcolors.TwoSlopeNorm(vcenter=med, vmin=vmin, vmax=vmax)
    except Exception:
        return None


def _save(fig, path_stem: Path):
    fig.savefig(str(path_stem) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(path_stem) + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _scatter(ax, xy, vals, title, cmap, vmin, vmax, label=""):
    v = vals.astype(float)
    norm = _median_norm(v, vmin, vmax)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, cmap=cmap,
                    norm=norm, vmin=(None if norm else vmin), vmax=(None if norm else vmax),
                    s=10, alpha=0.75, linewidths=0, rasterized=True)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cb.set_label(label, fontsize=6)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=7, fontweight="bold")


_GRIDSIZE = 25        # shared gridsize across all hexbin panels
_EMPTY_COLOR = "#FFE57F"  # yellow for bins with no valid metric (distinct from RdBu)


def _hexbin(ax, xy, vals, title, cmap, vmin, vmax, label=""):
    """
    Shared-grid hexbin with zero overlap between empty and data bins.

    xy   = ALL test coordinates (defines the bin grid, mincnt=1)
    vals = metric values; NaN where metric is not applicable for a sample

    Both layers use the SAME extent (derived from xy) and SAME gridsize,
    guaranteeing identical bin positions so the foreground perfectly covers
    background bins where data exists — no overlap, no partial bleed.

    Layer 1: all occupied bins → solid yellow (#FFE57F), thin grey edge
    Layer 2: bins with valid metric → RdBu-style colormap, overwrites yellow
    Bins that remain yellow have no valid metric value.
    """
    v = vals.astype(float)
    valid = ~np.isnan(v)

    # Compute extent from the full cloud so both layers use identical grid origin
    pad_x = 0.01 * (xy[:, 0].max() - xy[:, 0].min())
    pad_y = 0.01 * (xy[:, 1].max() - xy[:, 1].min())
    extent = (
        float(xy[:, 0].min() - pad_x), float(xy[:, 0].max() + pad_x),
        float(xy[:, 1].min() - pad_y), float(xy[:, 1].max() + pad_y),
    )
    kw = {"gridsize": _GRIDSIZE, "mincnt": 1, "extent": extent}

    # Layer 1: all occupied bins → solid yellow.
    # Use a single-color ListedColormap so the yellow is guaranteed regardless of counts.
    _yellow_cmap = mcolors.ListedColormap([_EMPTY_COLOR])
    bg = ax.hexbin(xy[:, 0], xy[:, 1],
                   C=np.ones(len(xy)), reduce_C_function=np.mean,
                   cmap=_yellow_cmap, vmin=0.5, vmax=1.5,
                   linewidths=0.5, **kw)
    bg.set_edgecolor("#BDBDBD")

    if valid.sum() < 4:
        ax.set_title(title, fontsize=7)
        return

    norm = _median_norm(v[valid], vmin, vmax)
    # Layer 2: colored bins where metric is valid — no edge so background never bleeds through
    hb = ax.hexbin(xy[valid, 0], xy[valid, 1], C=v[valid],
                   cmap=cmap, reduce_C_function=np.nanmean,
                   norm=norm, vmin=(None if norm else vmin), vmax=(None if norm else vmax),
                   linewidths=0, edgecolors="none",
                   **kw)
    cb = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cb.set_label(label, fontsize=6)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=7, fontweight="bold")


def _clear_axes(axes):
    for ax in np.array(axes).flat:
        ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ["top", "right", "left", "bottom"]:
            ax.spines[sp].set_visible(False)


def _uniform_lim(axes, xy):
    """Apply identical xlim/ylim to all axes so scatter and hexbin panels align."""
    pad = 0.05 * max(float(float(xy[:, 0].max() - xy[:, 0].min())), float(float(xy[:, 1].max() - xy[:, 1].min())), 1e-6)
    xl = (float(xy[:, 0].min()) - pad, float(xy[:, 0].max()) + pad)
    yl = (float(xy[:, 1].min()) - pad, float(xy[:, 1].max()) + pad)
    for ax in np.array(axes).flat:
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)


def _combo_scatter(ax, xy_s, combos):
    for c_name in sorted(set(combos), key=lambda x: -np.sum(combos == x)):
        mk = combos == c_name
        ax.scatter(xy_s[mk, 0], xy_s[mk, 1],
                   c=COMBO_COLORS.get(c_name, DEFAULT_COMBO_COLOR),
                   s=10, alpha=0.72, linewidths=0, label=c_name, rasterized=True)
    ax.legend(fontsize=5, loc="upper right", ncol=2, framealpha=0.75,
              markerscale=1.2, handlelength=1, borderpad=0.4)
    ax.set_title("Modality combo", fontsize=7, fontweight="bold")


def _fold_scatter(ax, xy_s, folds_v):
    for fi in sorted(set(folds_v.astype(int))):
        mk = folds_v.astype(int) == fi
        ax.scatter(xy_s[mk, 0], xy_s[mk, 1],
                   c=FOLD_COLORS[fi % len(FOLD_COLORS)],
                   s=10, alpha=0.72, linewidths=0,
                   label=f"Fold {fi} (n={int(mk.sum())})", rasterized=True)
    ax.legend(fontsize=5, loc="upper right", framealpha=0.75, markerscale=1.2)
    ax.set_title("Fold (test assignment)", fontsize=7, fontweight="bold")


# ── Classification 3×3 panel ─────────────────────────────────────────────────

def _plot_3x3_cls(df: pd.DataFrame, xy: np.ndarray, title: str, out_path: Path):
    """
    3×3 UMAP for classification task head (r_cls embedding), test set only.
    Row 0: [ACR label red/blue | modality combo | # modalities]
    Row 1: [cls_prob high=red  | fold           | hexbin cls_prob]
    Row 2: [hexbin proportion+ | hexbin |error| | hexbin # modalities]
    """
    mask = df["data_split"].values == "test"
    if mask.sum() == 0:
        print(f"[umap] No test samples — skip {out_path.name}"); return

    df_s   = df[mask].reset_index(drop=True)
    xy_s   = xy[mask]

    # Ground-truth ACR label (0=neg, 1=pos, nan=unknown)
    lbl    = df_s["acr_label"].values.astype(float) if "acr_label" in df_s.columns \
             else np.full(len(df_s), float("nan"))
    prob   = df_s["cls_prob"].values.astype(float)  if "cls_prob"  in df_s.columns \
             else np.full(len(df_s), float("nan"))
    combos = df_s["combo"].values if "combo" in df_s.columns else np.full(len(df_s), "Unknown")
    n_mods = df_s["n_mods"].values.astype(float) if "n_mods" in df_s.columns else np.zeros(len(df_s))
    folds_v = df_s["fold"].values.astype(float) if "fold" in df_s.columns else np.zeros(len(df_s))

    n_pos  = int((lbl == 1).sum())
    n_neg  = int((lbl == 0).sum())
    n_sp   = int(mask.sum())
    pmin, pmax = _pct(prob)

    plt.rcParams.update(NATURE_RC)
    fig, axes = plt.subplots(3, 3, figsize=(TWO_COL_W * 1.4, TWO_COL_W * 1.3))
    fig.patch.set_facecolor("white")
    _clear_axes(axes)

    # ── Row 0 ─────────────────────────────────────────────────────────────────
    # [0,0] ACR label: positive=red, negative=blue
    ax = axes[0, 0]
    lbl_k = ~np.isnan(lbl)
    if lbl_k.sum() > 0:
        c_lbl = np.where(lbl[lbl_k] == 1.0, "#E53935", "#1E88E5")
        ax.scatter(xy_s[lbl_k, 0], xy_s[lbl_k, 1], c=c_lbl,
                   s=10, alpha=0.75, linewidths=0, rasterized=True)
    ax.legend(handles=[Patch(color="#E53935", label=f"ACR+ (n={n_pos})"),
                       Patch(color="#1E88E5",  label=f"ACR- (n={n_neg})")],
              fontsize=6, loc="best", framealpha=0.8)
    ax.set_title("ACR label: positive=red, negative=blue", fontsize=7, fontweight="bold")

    # [0,1] modality combo
    _combo_scatter(axes[0, 1], xy_s, combos)

    # [0,2] # modalities
    sc_nm = axes[0, 2].scatter(xy_s[:, 0], xy_s[:, 1], c=n_mods,
                                cmap="Blues", vmin=0.5, vmax=4.5,
                                s=10, alpha=0.75, linewidths=0, rasterized=True)
    cb = plt.colorbar(sc_nm, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cb.set_ticks([1, 2, 3, 4]); cb.ax.tick_params(labelsize=6)
    axes[0, 2].set_title("# modalities (more=darker)", fontsize=7, fontweight="bold")

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    # [1,0] cls_prob: high=red (more likely ACR+)
    _scatter(axes[1, 0], xy_s, prob,
             "Predicted ACR+ prob (high=red)", "RdBu_r", pmin, pmax, "P(ACR+)")

    # [1,1] fold membership
    _fold_scatter(axes[1, 1], xy_s, folds_v)

    # [1,2] hexbin mean cls_prob: high=red (higher risk region)
    _hexbin(axes[1, 2], xy_s, prob,
            "Mean P(ACR+) per region (red=higher risk)", "RdBu_r", pmin, pmax, "mean P(ACR+)")

    # ── Row 2 hexbins — shared grid (mincnt=1 on xy_s), empty bins = yellow ──
    # [2,0] proportion ACR+ per bin: high=red (more positive cases)
    _hexbin(axes[2, 0], xy_s, lbl,
            "Proportion ACR+ per region (red=more+)", "RdBu_r", 0.0, 1.0, "prop ACR+")

    # [2,1] mean prediction error |pred-label|: high=red (worse prediction)
    err = np.abs(prob - lbl)
    err[np.isnan(err)] = float("nan")
    _hexbin(axes[2, 1], xy_s, err,
            "Mean |error| per region (red=worse)", "RdBu_r", 0.0, 1.0, "mean |err|")

    # [2,2] mean # modalities per bin: more modalities = darker blue
    _hexbin(axes[2, 2], xy_s, n_mods,
            "Mean # modalities per region", "Blues", 0.5, 4.5, "# mods")

    _uniform_lim(axes, xy_s)  # identical xlim/ylim on all 9 panels
    fig.suptitle(f"{title}  [TEST n={n_sp}  ACR+={n_pos}  ACR-={n_neg}]",
                 fontsize=8, fontweight="bold", y=1.002)
    plt.tight_layout()
    _save(fig, out_path)
    print(f"[umap] → {out_path.name}.pdf")


# ── TTE / Survival 3×3 panel ─────────────────────────────────────────────────

def _plot_3x3_tte(df: pd.DataFrame, xy: np.ndarray, endpoint: str,
                  title: str, out_path: Path):
    """
    3×3 UMAP for TTE/survival task head (r_tte embedding), test set only.
    Row 0: [event vs censored red/blue | modality combo | # modalities]
    Row 1: [hazard high=red | TTE events-only short=red | fold]
    Row 2: [hexbin event density | hexbin mean TTE | hexbin mean hazard]
    """
    cfg    = ENDPOINT[endpoint]
    ep_lbl = cfg["label"]
    ev_key = cfg["ev_key"]
    tte_key = cfg["tte_key"]

    mask = df["data_split"].values == "test"
    if mask.sum() == 0:
        print(f"[umap] No test samples — skip {out_path.name}"); return

    df_s   = df[mask].reset_index(drop=True)
    xy_s   = xy[mask]

    events = df_s[ev_key].values.astype(float)  if ev_key  in df_s.columns else np.full(len(df_s), float("nan"))
    ttes   = df_s[tte_key].values.astype(float) if tte_key in df_s.columns else np.full(len(df_s), float("nan"))
    haz    = df_s["hazard"].values.astype(float)
    combos = df_s["combo"].values if "combo" in df_s.columns else np.full(len(df_s), "Unknown")
    n_mods = df_s["n_mods"].values.astype(float) if "n_mods" in df_s.columns else np.zeros(len(df_s))
    folds_v = df_s["fold"].values.astype(float) if "fold" in df_s.columns else np.zeros(len(df_s))

    n_ev   = int((events == 1.0).sum())
    n_cens = int((events == 0.0).sum())
    n_sp   = int(mask.sum())
    hmin, hmax = _pct(haz)

    plt.rcParams.update(NATURE_RC)
    fig, axes = plt.subplots(3, 3, figsize=(TWO_COL_W * 1.4, TWO_COL_W * 1.3))
    fig.patch.set_facecolor("white")
    _clear_axes(axes)

    # ── Row 0 ─────────────────────────────────────────────────────────────────
    # [0,0] event vs censored: event=red, censored=blue
    ax = axes[0, 0]
    ev_k = ~np.isnan(events)
    if ev_k.sum() > 0:
        c_ev = np.where(events[ev_k] == 1.0, "#E53935", "#1E88E5")
        ax.scatter(xy_s[ev_k, 0], xy_s[ev_k, 1], c=c_ev,
                   s=10, alpha=0.75, linewidths=0, rasterized=True)
    ax.legend(handles=[Patch(color="#E53935", label=f"{ep_lbl} event (n={n_ev})"),
                       Patch(color="#1E88E5",  label=f"Censored (n={n_cens})")],
              fontsize=6, loc="best", framealpha=0.8)
    ax.set_title(f"{ep_lbl}: event=red, censored=blue", fontsize=7, fontweight="bold")

    # [0,1] modality combo
    _combo_scatter(axes[0, 1], xy_s, combos)

    # [0,2] # modalities
    sc_nm = axes[0, 2].scatter(xy_s[:, 0], xy_s[:, 1], c=n_mods,
                                cmap="Blues", vmin=0.5, vmax=4.5,
                                s=10, alpha=0.75, linewidths=0, rasterized=True)
    cb = plt.colorbar(sc_nm, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cb.set_ticks([1, 2, 3, 4]); cb.ax.tick_params(labelsize=6)
    axes[0, 2].set_title("# modalities (more=darker)", fontsize=7, fontweight="bold")

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    # [1,0] hazard — high hazard=red (high risk)
    _scatter(axes[1, 0], xy_s, haz,
             "Hazard score (high=red=high risk)", CMAP_HAZARD, hmin, hmax, "hazard")

    # [1,1] TTE events only — short TTE=red (imminent event)
    ax = axes[1, 1]
    ev_mask = (events == 1) & ~np.isnan(ttes)
    if ev_mask.sum() > 0:
        tte_ev = ttes[ev_mask]
        vmax_t = float(np.percentile(tte_ev, 98))
        norm_t = _median_norm(tte_ev, 0, max(vmax_t, 1))
        sc_t = ax.scatter(xy_s[ev_mask, 0], xy_s[ev_mask, 1], c=tte_ev,
                          cmap=CMAP_TTE, norm=norm_t,
                          vmin=(None if norm_t else 0), vmax=(None if norm_t else max(vmax_t, 1)),
                          s=10, alpha=0.8, linewidths=0, rasterized=True)
        cb2 = plt.colorbar(sc_t, ax=ax, fraction=0.046, pad=0.04, label="days")
        cb2.ax.tick_params(labelsize=6)
        ax.set_title(f"Days to {ep_lbl} (events only): short=red", fontsize=7, fontweight="bold")
    else:
        ax.set_title(f"Days to {ep_lbl} — no events", fontsize=7)

    # [1,2] fold membership
    _fold_scatter(axes[1, 2], xy_s, folds_v)

    # ── Row 2 hexbins — shared grid (mincnt=1 on xy_s), empty bins = yellow ──
    # Red = high risk / close to event throughout
    # Valid event mask: event happened AND TTE is recorded
    # Ensures event-density and TTE hexbins agree on the same set of samples.
    ev_with_tte = (events == 1.0) & (~np.isnan(ttes))

    # [2,0] event proportion: only count events that have a valid TTE (consistent with [2,1])
    events_valid = np.where(ev_with_tte, 1.0, np.where(events == 0.0, 0.0, np.nan))
    _hexbin(axes[2, 0], xy_s, events_valid,
            f"Event density (red=more {ep_lbl})", "RdBu_r", 0.0, 1.0,
            f"prop {ep_lbl}")

    # [2,1] mean TTE — events with valid TTE only; yellow = no qualifying events in bin
    tte_ev_only = np.where(ev_with_tte, ttes, np.nan)
    vmax_t = (float(np.nanpercentile(tte_ev_only[~np.isnan(tte_ev_only)], 98))
              if (~np.isnan(tte_ev_only)).sum() > 0 else 1.0)
    _hexbin(axes[2, 1], xy_s, tte_ev_only,
            f"Mean TTE — events only (red=imminent; yellow=no events)", CMAP_TTE, 0.0, vmax_t,
            "mean days")

    # [2,2] mean hazard: high=red (high risk region)
    _hexbin(axes[2, 2], xy_s, haz,
            "Mean hazard (red=high risk)", CMAP_HAZARD, hmin, hmax, "mean hazard")

    _uniform_lim(axes, xy_s)  # identical xlim/ylim on all 9 panels
    fig.suptitle(f"{title}  [TEST n={n_sp}  events={n_ev}  censored={n_cens}]",
                 fontsize=8, fontweight="bold", y=1.002)
    plt.tight_layout()
    _save(fig, out_path)
    print(f"[umap] → {out_path.name}.pdf")


# ── Patient-level UMAP ────────────────────────────────────────────────────────

def _plot_patient(df: pd.DataFrame, reps: np.ndarray, endpoint: str,
                  title: str, out_path: Path):
    """Mean-pool per patient (test set), 1×3: event status | TTE events | combo."""
    cfg    = ENDPOINT[endpoint]
    ep_lbl = cfg["label"]
    ev_key = cfg["ev_key"]
    tte_key = cfg["tte_key"]

    mask = df["data_split"].values == "test"
    if mask.sum() == 0:
        return

    df_s = df[mask].reset_index(drop=True)
    rep_s = reps[mask]

    if "patient_id" not in df_s.columns:
        return

    pids  = df_s["patient_id"].values
    upids = list(dict.fromkeys(pids))

    pat_reps, pat_ev, pat_tte, pat_haz, pat_combo = [], [], [], [], []
    for pid in upids:
        idx = np.where(pids == pid)[0]
        pat_reps.append(rep_s[idx].mean(axis=0))
        ev  = df_s[ev_key].values[idx] if ev_key in df_s.columns else np.array([float("nan")])
        tte = df_s[tte_key].values[idx] if tte_key in df_s.columns else np.array([float("nan")])
        haz = df_s["hazard"].values[idx]
        c   = df_s["combo"].values[idx][0] if "combo" in df_s.columns else "Unknown"
        e_v = float(np.nanmax(ev)) if not all(np.isnan(ev.astype(float))) else float("nan")
        t_v = float(np.nanmin(tte.astype(float))) if not all(np.isnan(tte.astype(float))) else float("nan")
        pat_ev.append(e_v); pat_tte.append(t_v)
        pat_haz.append(float(np.nanmean(haz))); pat_combo.append(c)

    pat_reps = np.stack(pat_reps)
    pat_ev   = np.array(pat_ev);  pat_tte = np.array(pat_tte)
    pat_haz  = np.array(pat_haz); pat_combo = np.array(pat_combo)

    if not HAS_UMAP or len(pat_reps) < 10:
        return

    try:
        from umap import UMAP as _UMAP
        pat_xy = _UMAP(n_neighbors=min(15, len(pat_reps) - 1),
                       min_dist=0.3, random_state=42, n_jobs=2).fit_transform(pat_reps)
    except Exception as e:
        print(f"[umap] patient UMAP failed: {e}"); return

    plt.rcParams.update(NATURE_RC)
    fig, ax_p = plt.subplots(1, 3, figsize=(TWO_COL_W * 1.3, 3.5))
    fig.patch.set_facecolor("white")
    for ax in ax_p:
        ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ["top", "right", "left", "bottom"]: ax.spines[sp].set_visible(False)

    # Event vs censored
    ev_k = ~np.isnan(pat_ev)
    if ev_k.sum() > 0:
        c_ev = np.where(pat_ev[ev_k] == 1.0, "#E53935", "#1E88E5")
        ax_p[0].scatter(pat_xy[ev_k, 0], pat_xy[ev_k, 1], c=c_ev,
                        s=30, alpha=0.85, linewidths=0, rasterized=True)
    n_ev = int((pat_ev == 1).sum()); n_cens = int((pat_ev == 0).sum())
    ax_p[0].legend(handles=[Patch(color="#E53935", label=f"Event (n={n_ev})"),
                             Patch(color="#1E88E5",  label=f"Censored (n={n_cens})")],
                   fontsize=6, framealpha=0.8)
    ax_p[0].set_title(f"{ep_lbl} status", fontsize=8, fontweight="bold")

    # Hazard
    hmin, hmax = _pct(pat_haz)
    norm_ph = _median_norm(pat_haz, hmin, hmax)
    sc2 = ax_p[1].scatter(pat_xy[:, 0], pat_xy[:, 1], c=pat_haz,
                           cmap=CMAP_HAZARD, norm=norm_ph,
                           vmin=(None if norm_ph else hmin), vmax=(None if norm_ph else hmax),
                           s=30, alpha=0.85, linewidths=0, rasterized=True)
    plt.colorbar(sc2, ax=ax_p[1], fraction=0.046, pad=0.04, label="mean hazard")
    ax_p[1].set_title("Hazard (patient mean)", fontsize=8, fontweight="bold")

    # Modality combo
    for c_name in sorted(set(pat_combo), key=lambda x: -np.sum(pat_combo == x)):
        mk = pat_combo == c_name
        ax_p[2].scatter(pat_xy[mk, 0], pat_xy[mk, 1],
                        c=COMBO_COLORS.get(c_name, DEFAULT_COMBO_COLOR),
                        s=30, alpha=0.85, linewidths=0, label=c_name)
    ax_p[2].legend(fontsize=6, loc="upper right", ncol=2, framealpha=0.75)
    ax_p[2].set_title("Modality combo", fontsize=8, fontweight="bold")

    fig.suptitle(f"{title} — patient-level UMAP (test, n={len(upids)} patients)",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_path)
    print(f"[umap] → {out_path.name}.pdf")


# ── Event vs censored figure ──────────────────────────────────────────────────

def _plot_event_vs_censored(variant_data: Dict, stored_xy: Dict,
                             endpoint: str, out_path: Path):
    """1×3: events colored by TTE | censored colored by hazard | all + hazard."""
    cfg    = ENDPOINT[endpoint]
    ep_lbl = cfg["label"]
    ev_key = cfg["ev_key"]
    tte_key = cfg["tte_key"]

    best_tag = next((t for t in ["middle", "early", "late"] if t in stored_xy), None)
    if best_tag is None and stored_xy:
        best_tag = next(iter(stored_xy))
    if best_tag is None:
        return

    df  = variant_data[best_tag]["df"]
    xy  = stored_xy[best_tag]["tte"]
    mask = df["data_split"].values == "test"
    if mask.sum() == 0:
        return
    df_t = df[mask].reset_index(drop=True)
    xy_t = xy[mask]

    events_all = df_t[ev_key].values.astype(float)  if ev_key  in df_t.columns else np.full(len(df_t), float("nan"))
    ttes_all   = df_t[tte_key].values.astype(float) if tte_key in df_t.columns else np.full(len(df_t), float("nan"))
    haz_all    = df_t["hazard"].values.astype(float)
    ev_m  = (events_all == 1) & ~np.isnan(ttes_all)
    cens_m = (events_all == 0)

    plt.rcParams.update(NATURE_RC)
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL_W * 1.4, 4.0))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ["top", "right", "left", "bottom"]: ax.spines[sp].set_visible(False)

    # [0] Events colored by TTE — short=red
    if ev_m.sum() > 0:
        tte_ev = ttes_all[ev_m]
        vmax_t = float(np.percentile(tte_ev, 98))
        norm_t = _median_norm(tte_ev, 0, max(vmax_t, 1))
        sc1 = axes[0].scatter(xy_t[ev_m, 0], xy_t[ev_m, 1], c=tte_ev,
                               cmap=CMAP_TTE, norm=norm_t,
                               vmin=(None if norm_t else 0), vmax=(None if norm_t else max(vmax_t, 1)),
                               s=12, alpha=0.8, linewidths=0, rasterized=True)
        plt.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04, label="days to event")
    axes[0].set_title(f"Events (n={ev_m.sum()})\nColored by TTE: short=red",
                       fontsize=8, fontweight="bold")

    # [1] Censored colored by hazard — high=red
    if cens_m.sum() > 0:
        haz_c = haz_all[cens_m]
        hmin, hmax = _pct(haz_c)
        norm_c = _median_norm(haz_c, hmin, hmax)
        sc2 = axes[1].scatter(xy_t[cens_m, 0], xy_t[cens_m, 1], c=haz_c,
                               cmap=CMAP_HAZARD, norm=norm_c,
                               vmin=(None if norm_c else hmin), vmax=(None if norm_c else hmax),
                               s=12, alpha=0.8, linewidths=0, rasterized=True)
        plt.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04, label="hazard")
    axes[1].set_title(f"Censored (n={cens_m.sum()})\nColored by hazard: high=red",
                       fontsize=8, fontweight="bold")

    # [2] All — hazard colored, events outlined
    hmin, hmax = _pct(haz_all)
    norm_a = _median_norm(haz_all, hmin, hmax)
    sc3 = axes[2].scatter(xy_t[:, 0], xy_t[:, 1], c=haz_all,
                           cmap=CMAP_HAZARD, norm=norm_a,
                           vmin=(None if norm_a else hmin), vmax=(None if norm_a else hmax),
                           s=10, alpha=0.7, linewidths=0, rasterized=True)
    plt.colorbar(sc3, ax=axes[2], fraction=0.046, pad=0.04, label="hazard")
    if ev_m.sum() > 0:
        axes[2].scatter(xy_t[ev_m, 0], xy_t[ev_m, 1], c="none",
                        edgecolors=cfg["color"], s=18, linewidths=0.6,
                        alpha=0.9, rasterized=True, label=f"{ep_lbl} event")
        axes[2].legend(fontsize=6, loc="best", framealpha=0.8)
    axes[2].set_title("All — hazard (events outlined)", fontsize=8, fontweight="bold")

    fig.suptitle(
        f"{VARIANT_DISPLAY.get(best_tag, best_tag)} ({endpoint.upper()}) — "
        f"Event vs Censored  [events={ev_m.sum()}  censored={cens_m.sum()}]",
        fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_path)
    print(f"[umap] → {out_path.name}.pdf")


# ── All-variant comparison ────────────────────────────────────────────────────

def _plot_comparison_tte(variant_data: Dict, stored_xy: Dict,
                          endpoint: str, out_path: Path):
    """All variants side-by-side: top=event status, bottom=hazard (TTE tasks)."""
    cfg    = ENDPOINT[endpoint]
    ep_lbl = cfg["label"]
    ev_key = cfg["ev_key"]
    tags   = [t for t in VARIANT_TAGS if t in stored_xy]
    if len(tags) < 2:
        return

    plt.rcParams.update(NATURE_RC)
    fig, axes = plt.subplots(2, len(tags), figsize=(TWO_COL_W * len(tags) / 3, 5.5))
    if len(tags) == 1:
        axes = axes[:, np.newaxis]
    fig.patch.set_facecolor("white")

    for ti, tag in enumerate(tags):
        df   = variant_data[tag]["df"]
        xy_t = stored_xy[tag]["tte"]
        test = df["data_split"].values == "test"
        if test.sum() == 0:
            continue
        df_t  = df[test]; xy_s = xy_t[test]
        events = df_t[ev_key].values.astype(float) if ev_key in df_t.columns else np.full(len(df_t), float("nan"))
        haz    = df_t["hazard"].values.astype(float)

        ax0 = axes[0, ti]; ax1 = axes[1, ti]
        for ax in [ax0, ax1]:
            ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ["top", "right", "left", "bottom"]:
                ax.spines[sp].set_visible(False)

        ev_k = ~np.isnan(events)
        if ev_k.sum() > 0:
            c_ev = ["#E53935" if e == 1 else "#1E88E5" for e in events[ev_k]]
            ax0.scatter(xy_s[ev_k, 0], xy_s[ev_k, 1], c=c_ev,
                        s=8, alpha=0.75, linewidths=0, rasterized=True)
        n_ev = int((events == 1).sum()); n_cens = int((events == 0).sum())
        ax0.set_title(f"{VARIANT_DISPLAY.get(tag, tag)}\n(ev={n_ev} cens={n_cens})",
                      fontsize=7, fontweight="bold")
        if ti == 0:
            ax0.set_ylabel(f"{ep_lbl} event", fontsize=7)

        hmin, hmax = _pct(haz)
        norm_h = _median_norm(haz, hmin, hmax)
        sc = ax1.scatter(xy_s[:, 0], xy_s[:, 1], c=haz, cmap=CMAP_HAZARD,
                         norm=norm_h, vmin=(None if norm_h else hmin), vmax=(None if norm_h else hmax),
                         s=8, alpha=0.75, linewidths=0, rasterized=True)
        if ti == len(tags) - 1:
            plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label="hazard")
        if ti == 0:
            ax1.set_ylabel("Hazard", fontsize=7)

    fig.suptitle(f"All variants — {ep_lbl} event status & hazard (test set)",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_path)
    print(f"[umap] → {out_path.name}.pdf")


def _plot_comparison_cls(variant_data: Dict, stored_xy: Dict, out_path: Path):
    """All variants side-by-side: top=ACR label, bottom=cls_prob (cls tasks)."""
    tags = [t for t in VARIANT_TAGS if t in stored_xy]
    if len(tags) < 2:
        return

    plt.rcParams.update(NATURE_RC)
    fig, axes = plt.subplots(2, len(tags), figsize=(TWO_COL_W * len(tags) / 3, 5.5))
    if len(tags) == 1:
        axes = axes[:, np.newaxis]
    fig.patch.set_facecolor("white")

    for ti, tag in enumerate(tags):
        df   = variant_data[tag]["df"]
        xy_t = stored_xy[tag]["cls"]
        test = df["data_split"].values == "test"
        if test.sum() == 0:
            continue
        df_t  = df[test]; xy_s = xy_t[test]
        lbl  = df_t["acr_label"].values.astype(float) if "acr_label" in df_t.columns else np.full(len(df_t), float("nan"))
        prob = df_t["cls_prob"].values.astype(float)  if "cls_prob"  in df_t.columns else np.full(len(df_t), float("nan"))

        ax0 = axes[0, ti]; ax1 = axes[1, ti]
        for ax in [ax0, ax1]:
            ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ["top", "right", "left", "bottom"]:
                ax.spines[sp].set_visible(False)

        lbl_k = ~np.isnan(lbl)
        if lbl_k.sum() > 0:
            c_lbl = ["#E53935" if l == 1 else "#1E88E5" for l in lbl[lbl_k]]
            ax0.scatter(xy_s[lbl_k, 0], xy_s[lbl_k, 1], c=c_lbl,
                        s=8, alpha=0.75, linewidths=0, rasterized=True)
        n_pos = int((lbl == 1).sum()); n_neg = int((lbl == 0).sum())
        ax0.set_title(f"{VARIANT_DISPLAY.get(tag, tag)}\n(ACR+={n_pos} ACR-={n_neg})",
                      fontsize=7, fontweight="bold")
        if ti == 0:
            ax0.set_ylabel("ACR label", fontsize=7)

        pmin, pmax = _pct(prob)
        norm_p = _median_norm(prob, pmin, pmax)
        sc = ax1.scatter(xy_s[:, 0], xy_s[:, 1], c=prob, cmap="RdBu_r",
                         norm=norm_p, vmin=(None if norm_p else pmin), vmax=(None if norm_p else pmax),
                         s=8, alpha=0.75, linewidths=0, rasterized=True)
        if ti == len(tags) - 1:
            plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label="P(ACR+)")
        if ti == 0:
            ax1.set_ylabel("P(ACR+)", fontsize=7)

    fig.suptitle("All variants — ACR label & predicted probability (test set)",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_path)
    print(f"[umap] → {out_path.name}.pdf")


# ── Main task dispatcher ──────────────────────────────────────────────────────

def task_umap(
    task:        str,
    output_dir:  Path,
    endpoint:    str,
    variant_data: Optional[Dict] = None,
    results_dir: Optional[Path] = None,
    splits_csv:  Optional[Path] = None,
    samples_dir: Optional[Path] = None,
    splits:      Optional[List[int]] = None,
    folds:       Optional[List[int]] = None,
    device_str:  str = "cpu",
    chicago_mil_dir: Optional[Path] = None,
    umap_n_neighbors: int = 30,
    umap_min_dist:    float = 0.3,
):
    if not HAS_UMAP:
        print("[umap] umap-learn not installed — skipping."); return

    plt.rcParams.update(NATURE_RC)
    out = output_dir
    out.mkdir(parents=True, exist_ok=True)

    if variant_data is not None:
        vd = variant_data
    else:
        if results_dir is None or splits_csv is None or samples_dir is None:
            print("[umap] Need results_dir/splits_csv/samples_dir or variant_data"); return
        vd = get_or_run(results_dir, splits_csv, samples_dir,
                        splits or [0], folds or [0, 1, 2, 3],
                        endpoint, output_dir, device_str, chicago_mil_dir)
    if not vd:
        return

    # Determine task type from TASKS registry
    task_type = TASKS.get(task, (None, None, None, None, "tte"))[4]
    do_cls = task_type == "cls"
    do_tte = task_type == "tte"
    # For any task key not in registry default to tte
    if not do_cls and not do_tte:
        do_tte = True

    tags = [t for t in VARIANT_TAGS if t in vd]
    tags += [t for t in sorted(vd) if t not in tags]
    stored_xy: Dict[str, Dict] = {}

    for tag in tags:
        df    = vd[tag]["df"]
        r_cls = vd[tag]["reps_cls"]
        r_tte = vd[tag]["reps_tte"]
        v_out = out / tag
        v_out.mkdir(exist_ok=True)
        disp  = VARIANT_DISPLAY.get(tag, tag)

        reducer = UMAPTransform(n_neighbors=umap_n_neighbors,
                                min_dist=umap_min_dist, random_state=42, n_jobs=4)

        xy_cls, xy_tte = None, None

        if do_cls:
            print(f"[umap] {tag}: UMAP on r_cls ({r_cls.shape}) …")
            xy_cls = reducer.fit_transform(r_cls)
            _plot_3x3_cls(df, xy_cls,
                          f"{disp} ({task.upper()}) — classification head",
                          v_out / f"fig_umap_{tag}_cls_test")

        if do_tte:
            print(f"[umap] {tag}: UMAP on r_tte ({r_tte.shape}) …")
            xy_tte = reducer.fit_transform(r_tte)
            _plot_3x3_tte(df, xy_tte, endpoint,
                          f"{disp} ({task.upper()}) — survival head",
                          v_out / f"fig_umap_{tag}_tte_test")
            _plot_patient(df, r_tte, endpoint,
                          f"{disp} ({task.upper()})",
                          v_out / f"fig_umap_{tag}_patient_test")

        stored_xy[tag] = {"cls": xy_cls, "tte": xy_tte}

    # Summary figures
    if do_tte:
        _plot_comparison_tte(vd, stored_xy, endpoint,
                             out / f"fig_umap_{task}_comparison")
        _plot_event_vs_censored(vd, stored_xy, endpoint,
                                out / f"fig_umap_{task}_event_vs_censored")
    if do_cls:
        _plot_comparison_cls(vd, stored_xy,
                             out / f"fig_umap_{task}_comparison_cls")
