#!/usr/bin/env python3
"""
Per-patient timeline visualization — one rich PNG per patient, saved to lustre.

Each patient plot shows:
  - Header: patient ID, outcomes (ACR/CLAD/Death), transplant date, FU
  - Main timeline: biopsy dots colored by ACR grade, TTE arrows with day labels
  - Per-sample modality panel: colored dots (H&E / BAL / CT / Clinical) at each timepoint
  - Per-sample info strip: date · grade · TTE · event flag annotated below each sample
  - CLAD inset box: onset date, days post-transplant
  - Death marker: date + days post-transplant
  - Stats footer: complete patient summary

TTE logic:
  ACR+ biopsy (A1/A2/A3/A4)   → event=1, tte=0   (event IS this biopsy)
  ACR− with future ACR         → event=1, tte=days to next ACR
  ACR− no future ACR           → event=0, censored at last biopsy
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

warnings.filterwarnings("ignore")

CSV_PATH   = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
OUT_DIR    = "/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/patient_plots"
CACHE_BASE = "/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/analysis/v7_full/inference_cache_alt"

C = dict(
    acr0       = "#3A86FF",   # label=0 (A0)
    acr1       = "#E63946",   # label=1 (A1/A2)
    no_label   = "#AAAAAA",   # label=NaN — no biopsy label
    tte_event  = "#2DC653",
    tte_censor = "#999FAE",
    clad       = "#FB8500",
    death      = "#1B2A3B",
    he         = "#7B2D8B",
    bal        = "#0B6E4F",
    ct         = "#1565C0",
    clin       = "#546E7A",
    spine      = "#C8CDD4",
    bg_acr     = "#FFF5F5",
    bg_none    = "#F6F8FB",
    bg_clad    = "#FFF8EC",
)

def _dot_style(label_val):
    """Return (color, size) based on label: NaN=grey, 0=blue, 1=red."""
    if pd.isna(label_val):
        return C["no_label"], 28
    elif float(label_val) == 1.0:
        return C["acr1"], 90
    else:
        return C["acr0"], 50

MODALITIES = [
    ("has_HE",       "HE",  C["he"]),
    ("has_BAL",      "BAL", C["bal"]),
    ("has_CT",       "CT",  C["ct"]),
    ("has_Clinical", "Clin",C["clin"]),
]


# ── TTE ────────────────────────────────────────────────────────────────────────

def _grade_key(g):
    if not isinstance(g, str):
        return "A0"
    for k in ["A4","A3","A2","A1","A0"]:
        if g.startswith(k):
            return k
    return "A0"


def compute_tte(df: pd.DataFrame) -> pd.DataFrame:
    """
    TTE uses label column (0/1) as ground truth:
      label=1  → event=1, tte=0   (this biopsy IS the ACR event)
      label=0, future label=1 exists → event=1, tte=days to that biopsy
      label=0, no future label=1   → event=0, censored at last biopsy
      label=NaN                    → event=NaN, tte=NaN (no biopsy label)
    """
    df = df.copy()
    df["anchor_dt"] = pd.to_datetime(df["anchor_dt"])

    # ACR event dates: only rows with label==1
    is_pos = df["label"] == 1.0
    acr_dates: dict = {}
    for _, row in df[is_pos].iterrows():
        acr_dates.setdefault(row["patient_id"], []).append(row["anchor_dt"])
    last_date = df.groupby("patient_id")["anchor_dt"].max().to_dict()

    ttes, events = [], []
    for _, row in df.iterrows():
        pid = row["patient_id"]; t = row["anchor_dt"]
        lbl = row["label"]

        if pd.isna(lbl):
            # No biopsy label — exclude from TTE
            ttes.append(np.nan); events.append(np.nan)
        elif float(lbl) == 1.0:
            ttes.append(0.0); events.append(1)
        else:
            future = sorted([d for d in acr_dates.get(pid, []) if d > t])
            if future:
                ttes.append(float((future[0] - t).days)); events.append(1)
            else:
                last = last_date.get(pid, t)
                ttes.append(float(max((last - t).days, 0))); events.append(0)

    df["tte_acr"]   = ttes
    df["event_acr"] = events
    return df


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["anchor_dt"])
    tx = df.groupby("patient_id")["anchor_dt"].min().rename("transplant_dt")
    df = df.merge(tx, on="patient_id")
    df["days_from_tx"] = (df["anchor_dt"] - df["transplant_dt"]).dt.days
    df = compute_tte(df)
    return df


def patient_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("patient_id").agg(
        n_samples    = ("file",          "count"),
        first_day    = ("days_from_tx",  "min"),
        last_day     = ("days_from_tx",  "max"),
        any_acr      = ("label",         lambda x: int((x == 1.0).any())),
        n_acr_pos    = ("label",         lambda x: int((x == 1.0).sum())),
        n_events     = ("event_acr",     "sum"),
        clad_status  = ("clad_status",   "first"),
        clad_days    = ("clad_days",     lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan),
        death_status = ("death_status",  "first"),
        death_days   = ("death_days",    lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan),
        n_he         = ("has_HE",        "sum"),
        n_bal        = ("has_BAL",       "sum"),
        n_ct         = ("has_CT",        "sum"),
        n_clin       = ("has_Clinical",  "sum"),
        transplant_dt= ("transplant_dt", "first"),
    ).reset_index()


# ── Per-patient figure ──────────────────────────────────────────────────────────

# ── Patient UMAP computation ────────────────────────────────────────────────────

def load_patient_umaps(cache_base: str, n_neighbors: int = 15,
                       min_dist: float = 0.25) -> dict:
    """
    Load inference cache for the 'middle' variant, compute per-patient
    mean-pooled r_cls representations for the test split, run UMAP per fold.

    Returns: {fold: pd.DataFrame with cols patient_id, umap_x, umap_y,
                                              label, tte_next_acr, combo, n_mods}
    Returns {} if cache not found.
    """
    import pickle, numpy as np
    arr_path = cache_base + "_arrays.npz"
    pkl_path = cache_base + "_meta.pkl"
    try:
        from pathlib import Path as _P
        if not (_P(arr_path).exists() and _P(pkl_path).exists()):
            print("[umap] Inference cache not found — skipping UMAP panels")
            return {}
        try:
            from umap import UMAP
        except ImportError:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn", "-q"])
            from umap import UMAP

        arrays = np.load(arr_path, allow_pickle=True)
        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)

        if "middle" not in meta:
            print("[umap] 'middle' variant not in cache — skipping")
            return {}

        df_all  = meta["middle"].copy()
        reps_all = arrays["middle__reps_cls"]

        if "data_split" not in df_all.columns:
            print("[umap] Cache missing data_split column — skipping")
            return {}

        result = {}
        for fold_id in sorted(df_all["fold"].unique()):
            fold_mask = df_all["fold"] == fold_id
            df_fold   = df_all.loc[fold_mask].copy().reset_index(drop=True)
            reps_fold = reps_all[np.where(fold_mask.values)[0]]

            if len(df_fold) == 0:
                continue

            # Mean-pool biopsies → one vector per patient (all splits)
            pids_all      = df_fold["patient_id"].values
            unique_pids   = list(dict.fromkeys(pids_all))
            pat_reps      = np.array([
                reps_fold[pids_all == p].mean(axis=0) for p in unique_pids
            ])

            # Patient-level metadata: first biopsy per patient for TTE horizon
            first_bx = (df_fold.sort_values("anchor_dt")
                        .groupby("patient_id", sort=False)
                        .first()
                        .reindex(unique_pids)
                        .reset_index())
            # Patient-level event flag (any event across all biopsies)
            evt_flag = (df_fold.groupby("patient_id")["event_next_acr"]
                        .apply(lambda x: 1.0 if (x == 1).any() else 0.0)
                        .reindex(unique_pids).values)
            first_bx["label"] = evt_flag

            # Mark which patients are test
            test_pids = set(df_fold.loc[df_fold["data_split"] == "test", "patient_id"])
            first_bx["is_test"] = first_bx["patient_id"].isin(test_pids)

            # UMAP on ALL patients for global context
            n_nb = min(n_neighbors, len(unique_pids) - 1)
            xy   = UMAP(n_neighbors=n_nb, min_dist=min_dist,
                        random_state=42, verbose=False).fit_transform(pat_reps)

            first_bx["umap_x"] = xy[:, 0]
            first_bx["umap_y"] = xy[:, 1]
            result[int(fold_id)] = first_bx
            n_test = int(first_bx["is_test"].sum())
            print(f"[umap] fold {fold_id}: {len(unique_pids)} total patients embedded "
                  f"({n_test} test, {len(unique_pids)-n_test} bg)")

        return result

    except Exception as e:
        print(f"[umap] Failed to compute patient UMAPs: {e}")
        import traceback; traceback.print_exc()
        return {}


def _draw_umap_panels(fig, gs_row, pid: str, umap_dfs: dict):
    """
    Draw one row of UMAP panels in the given GridSpec row.
    One column per fold × 3 coloring subsets (TTE / modality combo / ACR label).
    Each panel: all test patients as small dots, highlighted patient as ★.
    """
    folds = sorted(umap_dfs.keys())
    n_folds = len(folds)
    if n_folds == 0:
        return

    # Sub-gridspec: n_folds cols × 3 rows (TTE / combo / ACR)
    inner = gridspec.GridSpecFromSubplotSpec(
        3, n_folds, subplot_spec=gs_row, hspace=0.05, wspace=0.08
    )

    row_titles = ["ACR label", "TTE (days)", "Modality combo"]

    for ci, fold_id in enumerate(folds):
        df_u     = umap_dfs[fold_id]
        xy       = df_u[["umap_x", "umap_y"]].values
        pid_mask = df_u["patient_id"].values == str(pid)
        is_test  = df_u["is_test"].values if "is_test" in df_u.columns else np.ones(len(df_u), bool)
        bg_mask  = ~is_test & ~pid_mask   # train/val background

        def _draw_bg(ax):
            """Faint grey dots for all non-test patients (global context)."""
            if bg_mask.sum():
                ax.scatter(xy[bg_mask, 0], xy[bg_mask, 1],
                           c="#D0D0D0", s=10, alpha=0.25, linewidths=0, zorder=1)

        def _draw_star(ax):
            if pid_mask.sum():
                ax.scatter(xy[pid_mask, 0], xy[pid_mask, 1],
                           marker="*", s=380, c="#FFD700",
                           edgecolors="#333", linewidths=0.8, zorder=7)

        # ── Row 0: ACR label ─────────────────────────────────────────────────
        ax = fig.add_subplot(inner[0, ci])
        labels = df_u["label"].values
        _draw_bg(ax)
        for lv, col in [(1.0, C["acr1"]), (0.0, C["acr0"]), (float("nan"), C["no_label"])]:
            if np.isnan(lv):
                m = np.array([pd.isna(x) for x in labels])
            else:
                m = np.array([not pd.isna(x) and float(x) == lv for x in labels])
            m = m & is_test & ~pid_mask
            if m.sum():
                ax.scatter(xy[m, 0], xy[m, 1], c=col, s=22, alpha=0.82,
                           linewidths=0, zorder=3)
        _draw_star(ax)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_linewidth(0.5)
        if ci == 0:
            ax.set_ylabel("ACR label", fontsize=6.5, labelpad=2)
        ax.set_title(f"Fold {fold_id}", fontsize=6.5, pad=2)

        # ── Row 1: TTE ───────────────────────────────────────────────────────
        ax = fig.add_subplot(inner[1, ci])
        ttes = pd.to_numeric(df_u["tte_next_acr"],   errors="coerce").values
        evts = pd.to_numeric(df_u["event_next_acr"], errors="coerce").values
        _draw_bg(ax)
        # Test event patients: colored by TTE (yellow=short, red=long wait)
        ev_mask  = (evts == 1) & is_test & ~pid_mask
        cen_mask = (evts != 1) & ~np.isnan(ttes) & is_test & ~pid_mask
        tte_ev   = ttes[ev_mask]
        vmax     = float(np.nanpercentile(tte_ev, 95)) if ev_mask.sum() > 1 else 365.0
        vmax     = max(vmax, 1.0)
        if ev_mask.sum():
            sc = ax.scatter(xy[ev_mask, 0], xy[ev_mask, 1],
                            c=tte_ev, cmap="YlOrRd",
                            vmin=0, vmax=vmax,
                            s=22, alpha=0.88, linewidths=0, zorder=3)
            if ci == n_folds - 1:
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="TTE (d)")
        if cen_mask.sum():
            ax.scatter(xy[cen_mask, 0], xy[cen_mask, 1],
                       c="#888888", s=18, alpha=0.65, linewidths=0, zorder=3)
        _draw_star(ax)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_linewidth(0.5)
        if ci == 0:
            ax.set_ylabel("TTE (YlOrRd=event\ngrey=censored)", fontsize=5.5, labelpad=2)

        # ── Row 2: Modality combo ─────────────────────────────────────────────
        ax = fig.add_subplot(inner[2, ci])
        combos      = df_u["combo"].fillna("Unknown").values
        test_combos = sorted(set(combos[is_test]))
        cmap_c      = plt.cm.get_cmap("tab20", max(len(test_combos), 1))
        combo_col_map = {c: cmap_c(i) for i, c in enumerate(test_combos)}
        _draw_bg(ax)
        for c in test_combos:
            m = (combos == c) & is_test & ~pid_mask
            if m.sum():
                ax.scatter(xy[m, 0], xy[m, 1],
                           c=[combo_col_map[c]], s=22, alpha=0.82,
                           linewidths=0, zorder=3, label=c)
        _draw_star(ax)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_linewidth(0.5)
        if ci == 0:
            ax.set_ylabel("Modality", fontsize=6.5, labelpad=2)

        # Modality legend on last column
        if ci == n_folds - 1 and len(test_combos) <= 12:
            ax.legend(fontsize=4.5, loc="upper right",
                      framealpha=0.85, handlelength=0.8, handletextpad=0.4,
                      borderpad=0.4, labelspacing=0.25)

    # Row label on the left
    for ri, rtitle in enumerate(row_titles):
        pass   # already set via ax.set_ylabel above


def plot_patient(pid: str, subs: pd.DataFrame, row: pd.Series, out_dir: Path,
                 umap_dfs: dict = None):
    subs = subs.sort_values("days_from_tx").reset_index(drop=True)
    n_tp  = len(subs)
    fu    = int(row["last_day"] - row["first_day"])

    # Figure width scales with number of timepoints (min 16, max 40 inches)
    fig_w = max(16, min(40, n_tp * 0.85 + 5))
    has_umap = bool(umap_dfs)
    fig_h    = 18.5 if has_umap else 11.5
    n_rows   = 7    if has_umap else 6
    h_ratios = [0.6, 2.6, 0.6, 0.6, 0.6, 0.6] + ([4.0] if has_umap else [])
    # Rows: header / timeline / HE / BAL / CT / Clin [/ UMAP panels]
    fig = plt.figure(figsize=(fig_w, fig_h),
                     facecolor=C["bg_acr"] if row["any_acr"] else C["bg_none"])

    gs = gridspec.GridSpec(
        n_rows, 1,
        height_ratios=h_ratios,
        hspace=0.0,
        left=0.07, right=0.93, top=0.95, bottom=0.08,
    )

    ax_hdr  = fig.add_subplot(gs[0])
    ax_tl   = fig.add_subplot(gs[1])
    ax_he   = fig.add_subplot(gs[2])
    ax_bal  = fig.add_subplot(gs[3])
    ax_ct   = fig.add_subplot(gs[4])
    ax_clin = fig.add_subplot(gs[5])
    mod_axes = [ax_he, ax_bal, ax_ct, ax_clin]

    x_vals = subs["days_from_tx"].values.astype(float)
    pad_l  = max(80, fu * 0.05)
    pad_r  = max(150, fu * 0.08)
    x_lo   = x_vals.min() - pad_l
    x_hi   = x_vals.max() + pad_r

    all_axes = [ax_hdr, ax_tl] + mod_axes
    for ax in all_axes:
        ax.set_xlim(x_lo, x_hi)
        ax.set_yticks([])
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color(C["spine"])
        ax.patch.set_alpha(0)

    # ── Row 0: Header ───────────────────────────────────────────────────────────
    ax_hdr.set_ylim(0, 1)
    ax_hdr.spines["bottom"].set_visible(False)
    ax_hdr.set_xticks([])

    outcomes = []
    if row["any_acr"]:            outcomes.append(f"ACR+ ({row['n_acr_pos']:.0f} episodes)")
    if row["clad_status"] == 1.0: outcomes.append("CLAD")
    if row["death_status"] == 1.0:outcomes.append("DEATH")
    outcome_str = "  ·  ".join(outcomes) if outcomes else "event-free"
    tx_date_str = str(row.get("transplant_dt", ""))[:10]

    hdr_col = (C["acr1"] if row["any_acr"] else
               C["clad"] if row["clad_status"] == 1.0 else
               C["death"] if row["death_status"] == 1.0 else "#1E6B3C")

    ax_hdr.axhline(0.08, color=hdr_col, lw=3, alpha=0.7)
    ax_hdr.text(0.005, 0.58, f"Patient  {pid}",
                transform=ax_hdr.transAxes, fontsize=15,
                fontweight="bold", color=hdr_col, va="center")
    ax_hdr.text(0.22,  0.58, f"Outcomes: {outcome_str}",
                transform=ax_hdr.transAxes, fontsize=10.5,
                color="#2C3E50", va="center")
    ax_hdr.text(0.62,  0.58,
                f"Transplant: {tx_date_str}   Follow-up: {fu} days   "
                f"n = {n_tp} biopsies",
                transform=ax_hdr.transAxes, fontsize=9.5,
                color="#555", va="center")

    # ── Row 1: Main timeline ────────────────────────────────────────────────────
    ax_tl.set_ylim(-1.0, 1.15)
    ax_tl.set_xticks([])

    # Year gridlines
    if fu > 365:
        for yr in range(1, fu // 365 + 2):
            xyr = yr * 365
            if x_lo < xyr < x_hi:
                ax_tl.axvline(xyr, color=C["spine"], lw=0.7, linestyle="--",
                              alpha=0.5, zorder=0)
                ax_tl.text(xyr, 1.05, f"yr{yr}", fontsize=6.5,
                           ha="center", color=C["spine"])

    # Spine
    ax_tl.plot([x_vals.min(), x_vals.max()], [0, 0],
               color=C["spine"], lw=2.2, solid_capstyle="round", zorder=1)

    # TTE arrows + spans
    for _, s in subs.iterrows():
        if pd.isna(s["tte_acr"]) or pd.isna(s["event_acr"]):
            continue
        x0  = float(s["days_from_tx"])
        tte = float(s["tte_acr"])
        ev  = int(s["event_acr"])

        if tte == 0:
            # ACR event at this biopsy — red diamond above spine
            ax_tl.scatter(x0, 0.58, marker="D", color=C["acr1"],
                          s=70, zorder=8, linewidth=0.8,
                          edgecolors="white", alpha=0.95)
            ax_tl.text(x0, 0.72, "tte=0", fontsize=6, ha="center",
                       color=C["acr1"], fontweight="bold",
                       path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
        else:
            col = C["tte_event"] if ev == 1 else C["tte_censor"]
            alp = 0.65         if ev == 1 else 0.38
            x1  = min(x0 + tte, x_hi - 30)
            ax_tl.fill_betweenx([-0.14, 0.14], x0, x1,
                                 color=col, alpha=alp * 0.28, zorder=2)
            ax_tl.annotate(
                "", xy=(x1, 0.34), xytext=(x0 + 1, 0.34),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.4,
                                alpha=alp, shrinkA=0, shrinkB=0), zorder=4,
            )
            # TTE label on arrow
            mid = (x0 + x1) / 2
            suffix = "d ✓" if ev == 1 else "d (cens)"
            ax_tl.text(mid, 0.44, f"{int(tte)}{suffix}",
                       ha="center", va="bottom", fontsize=6.2,
                       color=col, alpha=0.9,
                       path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    # Biopsy dots
    for _, s in subs.iterrows():
        col, size = _dot_style(s.get("label"))
        ax_tl.scatter(float(s["days_from_tx"]), 0,
                      color=col, s=size, zorder=6,
                      alpha=0.93, edgecolors="white", linewidth=0.9)

    # Biopsy date label below dot
    for _, s in subs.iterrows():
        col, _ = _dot_style(s.get("label"))
        dt  = str(s["anchor_dt"])[:10]
        ax_tl.text(float(s["days_from_tx"]), -0.25,
                   dt, ha="center", va="top", fontsize=5.5,
                   color=col, rotation=55,
                   path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])

    # CLAD line + inset box
    if row["clad_status"] == 1.0 and pd.notna(row["clad_days"]):
        cd_abs = subs["days_from_tx"].values + subs["clad_days"].values
        cd_abs = cd_abs[np.isfinite(cd_abs)]
        if len(cd_abs):
            cd = float(np.nanmedian(cd_abs))
            ax_tl.axvline(cd, color=C["clad"], lw=2.2,
                          linestyle=(0, (5, 3)), alpha=0.9, zorder=5)
            ax_tl.text(cd + 6, 0.95,
                       f"CLAD\n{int(cd)}d post-Tx",
                       fontsize=7.5, color=C["clad"], fontweight="bold",
                       va="top",
                       bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8EC",
                                 ec=C["clad"], alpha=0.92),
                       path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    # Death line + box
    if row["death_status"] == 1.0 and pd.notna(row["death_days"]):
        dd = float(row["death_days"])
        ax_tl.axvline(dd, color=C["death"], lw=2.0,
                      linestyle=":", alpha=0.85, zorder=5)
        ax_tl.text(dd + 6, -0.60,
                   f"† Death\n{int(dd)}d post-Tx",
                   fontsize=7.5, color=C["death"], fontweight="bold",
                   va="bottom",
                   bbox=dict(boxstyle="round,pad=0.3", fc="#F0F4F8",
                             ec=C["death"], alpha=0.92))

    # Timeline y-axis label
    ax_tl.text(x_lo + 10, 0.02, "Biopsies", fontsize=7.5,
               color="#777", va="bottom", style="italic")

    # ── Modality rows (one per modality) ────────────────────────────────────────
    for ax, (col_name, mlabel, mcol) in zip(mod_axes, MODALITIES):
        ax.set_ylim(-0.6, 0.6)
        ax.spines["bottom"].set_color(C["spine"])

        # Baseline
        ax.plot([x_vals.min(), x_vals.max()], [0, 0],
                color=C["spine"], lw=0.8, alpha=0.4, zorder=1)

        n_present_total = int(subs[col_name].sum()) if col_name in subs.columns else 0

        for _, s in subs.iterrows():
            xp      = float(s["days_from_tx"])
            present = bool(s.get(col_name, 0))
            if present:
                # Filled colored tick
                ax.plot([xp, xp], [-0.38, 0.38], color=mcol,
                        lw=2.8, solid_capstyle="round", zorder=4, alpha=0.85)
                # Small dot at center
                ax.scatter(xp, 0, color=mcol, s=30, zorder=5,
                           edgecolors="white", linewidth=0.5)
            else:
                # Faint cross = absent
                ax.scatter(xp, 0, color="#D0D5DC", s=12,
                           marker="x", linewidth=0.8, zorder=3, alpha=0.6)

        # Label left
        ax.text(x_lo + 10, 0, mlabel,
                fontsize=8.5, color=mcol, fontweight="bold",
                va="center", ha="left")
        # Count right
        ax.text(x_hi - 8, 0, f"{n_present_total}/{n_tp}",
                fontsize=8, color=mcol, va="center", ha="right", alpha=0.85)

        # Disable x-ticks except bottom row
        if ax is not ax_clin:
            ax.set_xticks([])
        else:
            ax.tick_params(axis="x", labelsize=7.5, color=C["spine"])
            ax.set_xlabel("Days from transplant", fontsize=9, color="#666", labelpad=3)

    # ── Footer stats box ─────────────────────────────────────────────────────────
    fold_cols = [c for c in subs.columns if c.startswith("split0_fold")]
    fold_str  = ("Folds: " + "  ".join(
        [f"f{i}={str(subs[c].iloc[0])[:2]}" for i, c in enumerate(fold_cols)])
        if fold_cols else "")

    clad_str  = (f"CLAD @ {row['clad_days']:.0f}d post-Tx"
                 if row["clad_status"] == 1.0 and pd.notna(row["clad_days"])
                 else "No CLAD")
    death_str = (f"Death @ {row['death_days']:.0f}d post-Tx"
                 if row["death_status"] == 1.0 and pd.notna(row["death_days"])
                 else "Alive")

    ev1 = subs[subs["event_acr"] == 1]
    ev0 = subs[subs["event_acr"] == 0]
    tte_str = (f"TTE events={len(ev1)} (tte=0: {(ev1['tte_acr']==0).sum()}, "
               f"tte>0: {(ev1['tte_acr']>0).sum()})   censored={len(ev0)}")

    lines = [
        f"Patient: {pid}   Transplant: {tx_date_str}   Follow-up: {fu} days   "
        f"n_biopsies: {n_tp}   ACR+ biopsies: {row['n_acr_pos']:.0f}",
        f"Outcomes: {outcome_str}   {clad_str}   {death_str}",
        f"{tte_str}",
        f"Modality counts — HE: {row['n_he']:.0f}   BAL: {row['n_bal']:.0f}   "
        f"CT: {row['n_ct']:.0f}   Clinical: {row['n_clin']:.0f}   {fold_str}",
    ]

    fig.text(0.50, 0.025, "\n".join(lines),
             ha="center", va="bottom", fontsize=7.5,
             family="monospace", color="#2C3E50",
             bbox=dict(boxstyle="round,pad=0.6", fc="white",
                       ec=C["spine"], alpha=0.88))

    # ── Legend ──────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=C["no_label"], label="No label (no biopsy grade)"),
        mpatches.Patch(color=C["acr0"],     label="label=0 (A0, no rejection)"),
        mpatches.Patch(color=C["acr1"],     label="label=1 (A1/A2, ACR+)"),
        Line2D([0],[0], color=C["tte_event"],  lw=2,  label="→ TTE to future ACR (event=1)"),
        Line2D([0],[0], color=C["tte_censor"], lw=2,  linestyle="--",
               label="→ TTE censored (event=0)"),
        mpatches.Patch(color=C["clad"], alpha=0.8, label="CLAD onset"),
        mpatches.Patch(color=C["death"],alpha=0.8, label="Death"),
    ] + [
        Line2D([0],[0], marker="|", color=mcol, markersize=9, lw=0,
               markeredgewidth=2.5, label=f"{mlabel} present")
        for _, mlabel, mcol in MODALITIES
    ]
    fig.legend(handles=legend_items, loc="upper right",
               fontsize=7.5, frameon=True, framealpha=0.93,
               edgecolor=C["spine"], ncol=1,
               bbox_to_anchor=(0.998, 0.96))

    # ── UMAP row (optional) ──────────────────────────────────────────────────────
    if has_umap:
        _draw_umap_panels(fig, gs[6], pid, umap_dfs)

    out_path = out_dir / f"{pid}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Compact overview grid ───────────────────────────────────────────────────────

def plot_overview_grid(df: pd.DataFrame, pat_df: pd.DataFrame, out_dir: Path):
    patients = (
        pat_df.assign(fu=pat_df["last_day"] - pat_df["first_day"])
        .sort_values(["any_acr", "fu"], ascending=[False, False])
        ["patient_id"].tolist()
    )
    n    = len(patients)
    cols = 5
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 6.5, rows * 2.2),
        facecolor="white",
        gridspec_kw={"hspace": 0.65, "wspace": 0.06},
    )
    axes = axes.flatten()

    for idx, pid in enumerate(patients):
        ax   = axes[idx]
        subs = df[df["patient_id"] == pid].sort_values("days_from_tx")
        row  = pat_df[pat_df["patient_id"] == pid].iloc[0]
        x    = subs["days_from_tx"].values.astype(float)

        ax.set_facecolor(C["bg_acr"] if row["any_acr"] else C["bg_none"])
        ax.plot([x.min(), x.max()], [0, 0], color=C["spine"], lw=1.5,
                solid_capstyle="round", zorder=1)

        # TTE spans
        for _, s in subs.iterrows():
            if pd.isna(s["tte_acr"]) or pd.isna(s["event_acr"]):
                continue
            x0  = float(s["days_from_tx"])
            tte = float(s["tte_acr"])
            ev  = int(s["event_acr"])
            if tte == 0:
                ax.scatter(x0, 0.44, marker="D", color=C["acr1"],
                           s=20, zorder=6, linewidth=0)
            elif tte > 0:
                col = C["tte_event"] if ev == 1 else C["tte_censor"]
                x1  = x0 + tte
                ax.fill_betweenx([-0.14, 0.14], x0, x1, color=col,
                                 alpha=0.18, zorder=2)
                ax.annotate("", xy=(x1, 0.28), xytext=(x0, 0.28),
                            arrowprops=dict(arrowstyle="->", color=col,
                                            lw=0.9, alpha=0.6,
                                            shrinkA=0, shrinkB=0), zorder=3)

        # Biopsy dots
        for _, s in subs.iterrows():
            col, size = _dot_style(s.get("label"))
            ax.scatter(float(s["days_from_tx"]), 0,
                       color=col, s=max(12, size * 0.35),
                       zorder=5, alpha=0.90, edgecolors="white", linewidth=0.4)

        # CLAD / Death
        if row["clad_status"] == 1.0 and pd.notna(row["clad_days"]):
            cd_abs = subs["days_from_tx"].values + subs["clad_days"].values
            cd_abs = cd_abs[np.isfinite(cd_abs)]
            if len(cd_abs):
                ax.axvline(float(np.nanmedian(cd_abs)), color=C["clad"],
                           lw=1.3, linestyle=(0,(4,3)), alpha=0.8, zorder=4)
        if row["death_status"] == 1.0 and pd.notna(row["death_days"]):
            ax.axvline(float(row["death_days"]), color=C["death"],
                       lw=1.0, linestyle=":", alpha=0.75, zorder=4)

        # Modality presence dots at bottom
        for mi, (col_name, _, mcol) in enumerate(MODALITIES):
            present = int(subs[col_name].sum()) > 0 if col_name in subs.columns else False
            ax.scatter(0.02 + mi * 0.055, -0.62,
                       color=mcol if present else "#E0E0E0",
                       s=16, zorder=5, clip_on=False,
                       transform=ax.get_yaxis_transform())

        ax.set_ylim(-0.72, 0.62)
        ax.set_yticks([])
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color(C["spine"])
        ax.tick_params(axis="x", labelsize=5.5, color=C["spine"])

        title_col = (C["acr1"] if row["any_acr"] else
                     C["clad"] if row["clad_status"] == 1.0 else "#334")
        ax.set_title(f"{pid}", fontsize=7.5, fontweight="bold",
                     color=title_col, pad=2)
        ax.set_xlabel(
            f"n={row['n_samples']:.0f}  ACR+={row['n_acr_pos']:.0f}  "
            f"FU={int(row['last_day']-row['first_day'])}d",
            fontsize=5.5, color="#778", labelpad=1,
        )

    for ax in axes[n:]:
        ax.set_visible(False)

    legend_items = [
        mpatches.Patch(color=C["acr0"],        label="A0 no rejection"),
        mpatches.Patch(color=C["acr1"],        label="A1/A2 ACR+"),
        Line2D([0],[0], color=C["tte_event"],  lw=1.8, label="TTE→ACR"),
        Line2D([0],[0], color=C["tte_censor"], lw=1.8, linestyle="--",
               label="censored"),
        Line2D([0],[0], color=C["clad"],  lw=1.5, linestyle=(0,(4,3)), label="CLAD"),
        Line2D([0],[0], color=C["death"], lw=1.2, linestyle=":", label="Death"),
    ] + [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=mcol,
               markersize=6, label=mlabel)
        for _, mlabel, mcol in MODALITIES
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=5,
               fontsize=7.5, frameon=True, framealpha=0.95,
               edgecolor=C["spine"], bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("Patient Timelines — ACR / CLAD / Death / TTE  ·  modality dots: HE BAL CT Clin",
                 fontsize=13, fontweight="bold", color="#1a1a2e", y=1.002)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    for fmt, dpi in [("pdf", 180), ("png", 130)]:
        out_path = out_dir / f"patient_timelines_overview.{fmt}"
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi, facecolor="white")
        print(f"  Saved: {out_path}")
    plt.close(fig)


# ── TTE distribution ────────────────────────────────────────────────────────────

def plot_tte_distribution(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), facecolor="white")
    clip = 2000
    bins = np.linspace(0, clip, 45)

    ax = axes[0]
    ev1 = df[df["event_acr"] == 1]["tte_acr"].clip(upper=clip)
    ev0 = df[df["event_acr"] == 0]["tte_acr"].clip(upper=clip)
    ax.hist(ev1, bins=bins, color=C["tte_event"],  alpha=0.75, edgecolor="white",
            lw=0.4, label=f"event=1  (n={len(ev1)})")
    ax.hist(ev0, bins=bins, color=C["tte_censor"], alpha=0.50, edgecolor="white",
            lw=0.4, label=f"censored (n={len(ev0)})")
    ax.axvline(float(ev1.median()), color=C["tte_event"],  lw=1.8, linestyle="--",
               label=f"median event = {ev1.median():.0f}d")
    ax.axvline(float(ev0.median()), color=C["tte_censor"], lw=1.8, linestyle="--",
               label=f"median censor = {ev0.median():.0f}d")
    ax.set_xlabel("TTE (days, clipped @ 2000)", fontsize=11)
    ax.set_ylabel("# samples", fontsize=11)
    ax.set_title("TTE: event vs censored", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)

    ax = axes[1]
    apos = df[df["label"] == 1.0]["tte_acr"].clip(upper=clip)
    aneg = df[df["label"] == 0.0]["tte_acr"].clip(upper=clip)
    ax.hist(apos, bins=bins, color=C["acr1"], alpha=0.72, edgecolor="white",
            lw=0.4, label=f"ACR+  (n={len(apos)})")
    ax.hist(aneg, bins=bins, color=C["acr0"], alpha=0.55, edgecolor="white",
            lw=0.4, label=f"ACR−  (n={len(aneg)})")
    p_ev1 = int(df[df["label"]==1.0]["event_acr"].sum())
    n_ev1 = int(df[df["label"]==0.0]["event_acr"].sum())
    ax.text(0.97, 0.96,
            f"ACR+: event={p_ev1}, censored={len(apos)-p_ev1}\n"
            f"ACR−: event={n_ev1}, censored={len(aneg)-n_ev1}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C["spine"], alpha=0.9))
    ax.set_xlabel("TTE (days, clipped @ 2000)", fontsize=11)
    ax.set_ylabel("# samples", fontsize=11)
    ax.set_title("TTE: by ACR label", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("ACR Time-to-Event Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = out_dir / "tte_distribution.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",           default=CSV_PATH)
    p.add_argument("--out_dir",       default=OUT_DIR)
    p.add_argument("--no_tte",        action="store_true")
    p.add_argument("--overview_only", action="store_true")
    p.add_argument("--no_umap",       action="store_true",
                   help="Skip UMAP panels (faster, smaller files)")
    args = p.parse_args()

    print("Loading data and computing TTE...")
    df     = load_data(args.csv)
    pat_df = patient_summary(df)
    out    = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ev1 = df[df["event_acr"] == 1]
    ev0 = df[df["event_acr"] == 0]
    print(f"\nTTE summary ({len(df)} total samples):")
    print(f"  event=1 : {len(ev1):4d}  "
          f"(tte=0: {(ev1['tte_acr']==0).sum()}, tte>0: {(ev1['tte_acr']>0).sum()})")
    print(f"  event=0 : {len(ev0):4d}  censored  "
          f"median_tte={ev0['tte_acr'].median():.0f}d")

    # Load patient UMAP embeddings (one per fold, test set only)
    umap_dfs = {}
    if not args.no_umap and not args.overview_only:
        print("\nComputing per-fold patient UMAPs from inference cache...")
        umap_dfs = load_patient_umaps(CACHE_BASE)
        print(f"  Loaded UMAPs for folds: {sorted(umap_dfs.keys())}")

    # Per-patient plots
    if not args.overview_only:
        per_pat_dir = out / "per_patient"
        per_pat_dir.mkdir(exist_ok=True)
        print(f"\nPlotting {len(pat_df)} per-patient plots → {per_pat_dir}")
        for i, (_, row) in enumerate(pat_df.iterrows()):
            pid  = row["patient_id"]
            subs = df[df["patient_id"] == pid]
            plot_patient(pid, subs, row, per_pat_dir, umap_dfs=umap_dfs)
            if (i + 1) % 20 == 0 or (i + 1) == len(pat_df):
                print(f"  {i+1}/{len(pat_df)} done...")
        print(f"  All {len(pat_df)} per-patient plots saved to {per_pat_dir}")

    print("\nPlotting overview grid...")
    plot_overview_grid(df, pat_df, out)

    print("Plotting TTE distribution...")
    plot_tte_distribution(df, out)

    print(f"\nAll plots saved to {out}")


if __name__ == "__main__":
    main()
