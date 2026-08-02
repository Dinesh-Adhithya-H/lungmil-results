"""
Regenerate G panel (patient rep UMAP) from cached results_raw.npy.
No GPU or model inference needed — loads existing npy and re-plots.

Usage:
    python interpretability/regen_G_panel.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from umap import UMAP

REPO = Path("/home/aih/dinesh.haridoss/chicago_mil")
INTERP_ROOT = REPO / "interpretability" / "set_mil_mt_interp"
FIG_ROOT    = REPO / "figures" / "interpretability"

CONFIGS = [
    dict(
        name      = "acr_cls",
        npy_path  = INTERP_ROOT / "set_mil_mt_no_sab_split2_fold0_cls" / "results_raw.npy",
        out_dir   = FIG_ROOT / "acr_cls",
        task      = "acr_cls",
        score_key = "acr_cls",
        score_lbl = "P(ACR+)",
        ev_key    = "event_acr",
        tte_key   = "tte_acr",
        label_key = "label",
        split_lbl = "split2_fold0",
    ),
    dict(
        name      = "clad",
        npy_path  = INTERP_ROOT / "set_mil_mt_split2_fold0_clad_surv" / "results_raw.npy",
        out_dir   = FIG_ROOT / "clad",
        task      = "clad_surv",
        score_key = "clad_surv",
        score_lbl = "Risk percentile (0=low, 1=high)",
        ev_key    = "event_clad",
        tte_key   = "tte_clad",
        label_key = "label",
        split_lbl = "split2_fold0",
    ),
]

MS  = 12   # marker size
CMAP_HAZARD = "RdBu_r"
CMAP_TTE    = "RdYlBu_r"


def _uniform_lim(axes, xy, pad=0.05):
    xr = np.ptp(xy[:, 0]); yr = np.ptp(xy[:, 1])
    xl = (xy[:, 0].min() - pad * xr, xy[:, 0].max() + pad * xr)
    yl = (xy[:, 1].min() - pad * yr, xy[:, 1].max() + pad * yr)
    for ax in axes:
        ax.set_xlim(xl); ax.set_ylim(yl)


def _km_curve(t_arr, e_arr):
    order = np.argsort(t_arr)
    t_s = t_arr[order]; e_s = e_arr[order]
    n = len(t_s); surv = 1.0; surv_list = [1.0]; t_list = [0.0]
    for i in range(n):
        if e_s[i] == 1:
            surv *= (1 - 1.0 / max(n - i, 1))
        surv_list.append(surv); t_list.append(t_s[i])
    return np.array(t_list), np.array(surv_list)


def regen(cfg):
    print(f"\n=== {cfg['name']} ===")
    npy = cfg["npy_path"]
    if not npy.exists():
        print(f"  [skip] {npy} not found")
        return

    raw = np.load(str(npy), allow_pickle=True)
    results = raw.item() if raw.ndim == 0 else list(raw)
    if isinstance(results, dict):
        results = [results]
    N = len(results)
    print(f"  loaded {N} biopsy records")

    # ── Extract final representations ────────────────────────────────
    reps, scores, labs, ev_arr, tte_arr, pmod_arr, anchor_arr = [], [], [], [], [], [], []
    for r in results:
        fr = r.get("final_reps")
        if fr is None:
            continue
        if isinstance(fr, dict):
            fr = list(fr.values())[0] if fr else None
        if fr is None:
            continue
        fr = np.array(fr).ravel()
        reps.append(fr)

        lg = r.get("logits", {})
        sc = lg.get(cfg["score_key"], np.nan) if isinstance(lg, dict) else np.nan
        scores.append(float(sc) if sc is not None else np.nan)

        lb = r.get(cfg["label_key"])
        labs.append(float(lb) if lb is not None else np.nan)

        ev_arr.append(float(r.get(cfg["ev_key"], np.nan) or np.nan))
        tte_arr.append(float(r.get(cfg["tte_key"], np.nan) or np.nan))

        pm = r.get("present_mods", set())
        pmod_arr.append("+".join(sorted(pm)) if pm else "unknown")

        adt = r.get("anchor_dt")
        try:
            # anchor_dt may be a Timestamp, datetime, string, or numeric
            if adt is None:
                anchor_arr.append(np.nan)
            elif hasattr(adt, "toordinal"):
                anchor_arr.append(float(adt.toordinal()))
            elif hasattr(adt, "timestamp"):
                anchor_arr.append(float(adt.timestamp()) / 86400)
            else:
                anchor_arr.append(float(adt))
        except (TypeError, ValueError):
            anchor_arr.append(np.nan)

    if len(reps) < 10:
        print("  [skip] not enough valid reps")
        return

    reps = np.stack(reps)
    scores  = np.array(scores)
    labs    = np.array(labs)
    ev_arr  = np.array(ev_arr)
    tte_arr = np.array(tte_arr)
    anchor_arr = np.array(anchor_arr)

    # Rank-normalize scores for percentile display
    valid_sc = ~np.isnan(scores)
    sc_pct = np.full_like(scores, np.nan)
    if valid_sc.sum() > 2:
        from scipy.stats import rankdata
        sc_pct[valid_sc] = rankdata(scores[valid_sc]) / valid_sc.sum()

    # ── UMAP ─────────────────────────────────────────────────────────
    print("  computing UMAP …")
    reducer = UMAP(n_components=2, metric="cosine", n_neighbors=15,
                   min_dist=0.1, random_state=42)
    xy = reducer.fit_transform(reps.astype(np.float32))

    # ── Figure: 2 rows × 4 cols ───────────────────────────────────────
    fig = plt.figure(figsize=(22, 12))
    grd = gridspec.GridSpec(2, 4, figure=fig, wspace=0.25, hspace=0.38)
    ax0 = fig.add_subplot(grd[0, 0])   # label UMAP
    ax1 = fig.add_subplot(grd[0, 1])   # risk score UMAP
    ax2 = fig.add_subplot(grd[0, 2])   # TTE UMAP
    ax3 = fig.add_subplot(grd[0, 3])   # modality combo UMAP
    ax4 = fig.add_subplot(grd[1, 0])   # anchor day UMAP
    ax5 = fig.add_subplot(grd[1, 1])   # risk × TTE scatter
    ax6 = fig.add_subplot(grd[1, 2])   # KM curve
    ax7 = fig.add_subplot(grd[1, 3])   # CV split (placeholder — single split here)

    fig.suptitle(f"G — {cfg['task']} | {cfg['split_lbl']}  N={N}",
                 fontsize=13, fontweight="bold", y=1.01)

    FS_TITLE = 11
    FS_LABEL = 10
    FS_TICK  = 9
    FS_LEG   = 9

    umap_axes = [ax0, ax1, ax2, ax3, ax4]
    for ax in umap_axes:
        ax.set_facecolor("#f9f9f9")
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_linewidth(0.5)

    # 0: ACR label
    m0 = labs == 0; m1 = labs == 1; mn = np.isnan(labs)
    ax0.scatter(xy[m0, 0], xy[m0, 1], s=MS, c="#1E88E5", alpha=0.65,
                edgecolors="none", label=f"ACR− (n={m0.sum()})", zorder=3)
    ax0.scatter(xy[m1, 0], xy[m1, 1], s=MS, c="#E53935", alpha=0.80,
                edgecolors="none", label=f"ACR+ (n={m1.sum()})", zorder=4)
    if mn.any():
        ax0.scatter(xy[mn, 0], xy[mn, 1], s=MS * 0.4, c="#bbb", alpha=0.4, zorder=2)
    ax0.set_title("ACR label", fontsize=FS_TITLE, fontweight="bold")
    ax0.legend(markerscale=1.5, fontsize=FS_LEG, framealpha=0.8)

    # 1: risk score
    disp_sc = sc_pct if cfg["task"] != "acr_cls" else scores
    valid1 = ~np.isnan(disp_sc)
    if valid1.sum() > 5:
        vl = np.nanpercentile(disp_sc[valid1], 2)
        vh = np.nanpercentile(disp_sc[valid1], 98)
        norm1 = matplotlib.colors.Normalize(vmin=vl, vmax=vh)
        sc1 = ax1.scatter(xy[valid1, 0], xy[valid1, 1],
                          c=disp_sc[valid1], cmap=CMAP_HAZARD, norm=norm1,
                          s=MS, alpha=0.82, edgecolors="none", zorder=3)
        cb1 = fig.colorbar(sc1, ax=ax1, shrink=0.75, pad=0.02)
        cb1.ax.tick_params(labelsize=FS_TICK)
        cb1.set_label(cfg["score_lbl"], fontsize=FS_TICK)
    ax1.set_title(cfg["score_lbl"], fontsize=FS_TITLE, fontweight="bold")

    # 2: TTE
    ev_m = (~np.isnan(tte_arr)) & (ev_arr == 1)
    ce_m = (~np.isnan(tte_arr)) & (ev_arr == 0)
    vcenter = 0
    if ev_m.sum() > 5:
        vl2 = np.nanpercentile(tte_arr[ev_m], 2)
        vh2 = np.nanpercentile(tte_arr[ev_m], 98)
        vcenter = float(np.nanmedian(tte_arr[ev_m]))
        vcenter = float(np.clip(vcenter, vl2 + 1e-3, vh2 - 1e-3))
        tte_norm = matplotlib.colors.TwoSlopeNorm(vcenter=vcenter, vmin=vl2, vmax=vh2)
        if ce_m.any():
            ax2.scatter(xy[ce_m, 0], xy[ce_m, 1], s=MS * 0.3, c="#cccccc",
                        alpha=0.5, edgecolors="none", zorder=2,
                        label=f"Censored (n={ce_m.sum()})")
        sc2 = ax2.scatter(xy[ev_m, 0], xy[ev_m, 1],
                          c=tte_arr[ev_m], cmap=CMAP_TTE, norm=tte_norm,
                          s=MS * 1.4, alpha=0.85, edgecolors="white", linewidths=0.3,
                          zorder=4, label=f"Event (n={ev_m.sum()})")
        cb2 = fig.colorbar(sc2, ax=ax2, shrink=0.75, pad=0.02)
        cb2.ax.tick_params(labelsize=FS_TICK)
        cb2.set_label("TTE (days)", fontsize=FS_TICK)
        ax2.legend(fontsize=FS_LEG - 1, framealpha=0.8)
    ax2.set_title(f"TTE  (events, med={vcenter:.0f}d)", fontsize=FS_TITLE, fontweight="bold")

    # 3: modality combo
    combo_counts = {}
    for pm in pmod_arr:
        combo_counts[pm] = combo_counts.get(pm, 0) + 1
    top_combos = sorted(combo_counts, key=lambda x: -combo_counts[x])[:8]
    cmap_tab = plt.cm.get_cmap("tab10")
    combo_color = {c: cmap_tab(i / 10) for i, c in enumerate(top_combos)}
    display_combos = [pm if pm in top_combos else "Other" for pm in pmod_arr]
    other_m = np.array([c == "Other" for c in display_combos])
    if other_m.any():
        ax3.scatter(xy[other_m, 0], xy[other_m, 1], s=MS * 0.6, color="#aaa",
                    alpha=0.5, label=f"Other (n={other_m.sum()})", zorder=2)
    for ci, combo in enumerate(top_combos):
        mask = np.array([c == combo for c in display_combos])
        if not mask.any():
            continue
        ax3.scatter(xy[mask, 0], xy[mask, 1], s=MS * 1.4,
                    color=combo_color[combo], alpha=0.82, edgecolors="none",
                    label=f"{combo} (n={mask.sum()})", zorder=3 + ci)
    ax3.set_title("Modality combo", fontsize=FS_TITLE, fontweight="bold")
    ax3.legend(fontsize=7, markerscale=1.4, framealpha=0.8, ncol=1,
               handlelength=1.2, title="combo (n)", title_fontsize=7)

    # 4: anchor day
    valid4 = ~np.isnan(anchor_arr)
    if valid4.sum() > 5:
        vl4 = np.nanpercentile(anchor_arr[valid4], 2)
        vh4 = np.nanpercentile(anchor_arr[valid4], 98)
        norm4 = matplotlib.colors.Normalize(vmin=vl4, vmax=vh4)
        sc4 = ax4.scatter(xy[valid4, 0], xy[valid4, 1],
                          c=anchor_arr[valid4], cmap="viridis", norm=norm4,
                          s=MS, alpha=0.80, edgecolors="none", zorder=3)
        cb4 = fig.colorbar(sc4, ax=ax4, shrink=0.75, pad=0.02)
        cb4.ax.tick_params(labelsize=FS_TICK)
        cb4.set_label("Anchor day (d)", fontsize=FS_TICK)
    ax4.set_title("Anchor day post-transplant", fontsize=FS_TITLE, fontweight="bold")

    _uniform_lim([ax0, ax1, ax2, ax3, ax4], xy)

    # 5: risk × TTE scatter
    for ax in [ax5, ax6, ax7]:
        ax.set_facecolor("#f9f9f9")
        ax.spines[["top", "right"]].set_visible(False)

    valid5 = ~(np.isnan(disp_sc) | np.isnan(tte_arr))
    if valid5.sum() > 10:
        colors5 = ["#E53935" if e == 1 else "#1E88E5" for e in ev_arr[valid5]]
        ax5.scatter(disp_sc[valid5], tte_arr[valid5], c=colors5, s=MS * 0.7,
                    alpha=0.65, edgecolors="none", zorder=3)
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            lo = lowess(tte_arr[valid5], disp_sc[valid5], frac=0.5)
            ax5.plot(lo[:, 0], lo[:, 1], color="#E65100", lw=2.0, zorder=5, label="LOWESS")
        except Exception:
            pass
        ax5.set_xlabel(cfg["score_lbl"], fontsize=FS_LABEL)
        ax5.set_ylabel("TTE (days)", fontsize=FS_LABEL)
        ax5.tick_params(labelsize=FS_TICK)
        ax5.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E53935',
                       markersize=7, label=f'Event (n={(ev_arr[valid5]==1).sum()})'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E88E5',
                       markersize=7, label=f'Censored (n={(ev_arr[valid5]==0).sum()})'),
            plt.Line2D([0], [0], color='#E65100', lw=2, label='LOWESS'),
        ], fontsize=FS_LEG, framealpha=0.8)
    ax5.set_title("Risk vs TTE", fontsize=FS_TITLE, fontweight="bold")

    # 6: KM curve
    valid6 = ~(np.isnan(disp_sc) | np.isnan(tte_arr) | np.isnan(ev_arr))
    if valid6.sum() > 20:
        sc6 = disp_sc[valid6]; tte6 = tte_arr[valid6]; ev6 = ev_arr[valid6]
        t33 = np.percentile(sc6, 33); t67 = np.percentile(sc6, 67)
        hi = sc6 >= t67; lo = sc6 <= t33
        if hi.sum() >= 5:
            th, sh = _km_curve(tte6[hi], ev6[hi])
            ax6.step(th, sh, where="post", color="#E53935", lw=2,
                     label=f"High risk (≥P67, n={hi.sum()})")
        if lo.sum() >= 5:
            tl, sl = _km_curve(tte6[lo], ev6[lo])
            ax6.step(tl, sl, where="post", color="#1E88E5", lw=2,
                     label=f"Low risk (≤P33, n={lo.sum()})")
        ax6.set_ylim(0, 1.05); ax6.set_xlim(left=0)
        ax6.set_xlabel("Days from transplant", fontsize=FS_LABEL)
        ax6.set_ylabel("Survival probability", fontsize=FS_LABEL)
        ax6.tick_params(labelsize=FS_TICK)
        ax6.legend(fontsize=FS_LEG, framealpha=0.8)
    ax6.set_title("KM: top vs bottom tertile", fontsize=FS_TITLE, fontweight="bold")

    # 7: n biopsies per patient (group by patient_id)
    pat_ids = [r.get("patient_id") for r in results]
    biopsy_counts = {}
    for pid in pat_ids:
        if pid:
            biopsy_counts[pid] = biopsy_counts.get(pid, 0) + 1
    n_bx = np.array([biopsy_counts.get(r.get("patient_id"), 1) for r in results
                     if r.get("final_reps") is not None], dtype=float)
    if len(n_bx) == len(xy):
        vn = np.percentile(n_bx, 2); vx = np.percentile(n_bx, 98)
        normn = matplotlib.colors.Normalize(vmin=vn, vmax=vx)
        sc7 = ax7.scatter(xy[:, 0], xy[:, 1], c=n_bx, cmap="plasma",
                          norm=normn, s=MS, alpha=0.80, edgecolors="none")
        cb7 = fig.colorbar(sc7, ax=ax7, shrink=0.75, pad=0.02)
        cb7.ax.tick_params(labelsize=FS_TICK)
        cb7.set_label("# biopsies (patient)", fontsize=FS_TICK)
    ax7.set_facecolor("#f9f9f9")
    ax7.set_xticks([]); ax7.set_yticks([])
    ax7.set_xlim(ax0.get_xlim()); ax7.set_ylim(ax0.get_ylim())
    ax7.set_title("# biopsies per patient", fontsize=FS_TITLE, fontweight="bold")
    ax7.spines[:].set_linewidth(0.5)

    # save
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    stem = cfg["out_dir"] / f"G_final_rep_hexbin_{cfg['task']}"
    fig.savefig(str(stem) + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {stem}.png")


if __name__ == "__main__":
    for cfg in CONFIGS:
        regen(cfg)
    print("\n[done]")
