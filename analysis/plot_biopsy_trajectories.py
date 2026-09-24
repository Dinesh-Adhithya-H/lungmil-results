"""
Per-patient biopsy trajectory UMAP.

Layout per disease: 4 rows (TP/FN/TN/FP) × (2 context + N_PAT patient) cols.
  Col 0 : hazard-score hexbin   (all biopsies, coolwarm)
  Col 1 : TTE hexbin            (all biopsies, plasma_r + event ✕)
  Cols 2…: one patient per cell, grey density bg, trajectory with visit numbers

Each trajectory cell:
  - grey log-density hexbin background
  - arrows on every segment (light→dark gradient = early→late)
  - numbered visit labels: 1, 2, 3, ...
  - ○ first  ★ last
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler

ROOT     = Path(__file__).resolve().parent.parent
REPS_DIR = ROOT / "results" / "mm_abmil_v8" / "biopsy_reps"
OUT_DIR  = ROOT / "figures" / "interpretability" / "trajectories"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG = "#F7F4F1"

COL = {
    "TP": "#B71C1C",
    "FN": "#E65100",
    "TN": "#1565C0",
    "FP": "#6A1B9A",
}
QUAD_LABEL = {
    "TP": "True Positive\n(Event soon · HIGH risk)",
    "FN": "False Negative\n(Event soon · LOW risk)",
    "TN": "True Negative\n(No event · LOW risk)",
    "FP": "False Positive\n(No event · HIGH risk)",
}
QUADS  = ["TP", "FN", "TN", "FP"]
N_PAT  = 5

FS_SUPTITLE = 24
FS_CTX      = 14
FS_ROW      = 14
FS_CELL     = 11
FS_NUM      =  9   # visit number labels

TASKS = {
    "death_surv": {"label": "Post-Transplant Mortality (C-index 0.759)"},
    "acr_surv":   {"label": "ACR Survival (C-index 0.655)"},
    "clad_surv":  {"label": "CLAD Survival (C-index 0.595)"},
}
THRESHOLDS = {
    "death_surv": {"soon": 500, "long": 900},
    "acr_surv":   {"soon": 200, "long": 600},
    "clad_surv":  {"soon": 500, "long": 900},
}


def load_all_splits(task, n_splits=5):
    records = []
    for s in range(n_splits):
        p = REPS_DIR / f"biopsy_reps_{task}_split{s}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        pids  = list(d["patient_ids"])
        days  = d["biopsy_days"].tolist()
        reps  = d["reps"].numpy()
        risk  = d["risk"].numpy()  if "risk"  in d else np.full(len(pids), np.nan)
        tte   = d["tte"].numpy()   if "tte"   in d else np.full(len(pids), np.nan)
        event = d["event"].numpy() if "event" in d else np.full(len(pids), np.nan)
        for i in range(len(pids)):
            records.append({
                "patient_id":  pids[i],
                "biopsy_days": float(days[i]),
                "rep":         reps[i],
                "risk":        float(risk[i]),
                "tte":         float(tte[i]),
                "event":       float(event[i]),
            })
    return records


def umap_embed(X, n_neighbors=15, min_dist=0.15):
    try:
        import umap as umap_mod
        return umap_mod.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                              metric="cosine", random_state=42).fit_transform(X)
    except ImportError:
        from sklearn.manifold import TSNE
        print("  [warn] umap unavailable, using t-SNE")
        return TSNE(n_components=2, random_state=42).fit_transform(X)


def per_patient_summary(records):
    pdict = {}
    for i, r in enumerate(records):
        pid = r["patient_id"]
        if pid not in pdict:
            pdict[pid] = {"indices": [], "days": [], "risks": [],
                          "event": r["event"], "tte": r["tte"]}
        pdict[pid]["indices"].append(i)
        pdict[pid]["days"].append(r["biopsy_days"])
        pdict[pid]["risks"].append(r["risk"])
    for info in pdict.values():
        info["mean_risk"] = float(np.nanmean(info["risks"]))
    return pdict


def classify_quadrant(event, tte, mean_risk, risk_median, thr):
    high_risk = mean_risk >= risk_median
    if event == 1 and tte < thr["soon"]:
        return "TP" if high_risk else "FN"
    if event == 0 and tte > thr["long"]:
        return "TN" if not high_risk else "FP"
    return None


def select_patients(pdict, quad, n=N_PAT):
    cands = [(pid, info) for pid, info in pdict.items()
             if info.get("quadrant") == quad and len(info["indices"]) >= 2]
    cands.sort(key=lambda x: -len(x[1]["indices"]))
    return [pid for pid, _ in cands[:n]]


def draw_trajectory(ax, xy_all, indices, days, color, lw=1.8, ms=42):
    """Trajectory with arrows + numbered visit labels."""
    order  = np.argsort(days)
    xy_s   = xy_all[indices][order]
    n      = len(xy_s)
    c_rgba = np.array(mcolors.to_rgba(color))

    # arrows
    for i in range(n - 1):
        x0, y0 = xy_s[i];  x1, y1 = xy_s[i + 1]
        if np.sqrt((x1-x0)**2 + (y1-y0)**2) < 0.04:
            continue
        t         = (i + 1) / max(n - 1, 1)
        seg_alpha = 0.28 + 0.68 * t
        seg_lw    = lw * (0.5 + 0.7 * t)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->",
                                    color=(*c_rgba[:3], seg_alpha),
                                    lw=seg_lw, mutation_scale=10 + 5 * t))

    # dots with size/alpha gradient
    for i, (x, y) in enumerate(xy_s):
        t = i / max(n - 1, 1)
        ax.scatter(x, y, s=ms * (0.4 + 0.7 * t),
                   color=(*c_rgba[:3], 0.32 + 0.62 * t),
                   edgecolors="white", linewidths=0.5, zorder=5)

    # numbered visit labels
    for i, (x, y) in enumerate(xy_s):
        ax.text(x, y, str(i + 1), ha="center", va="center",
                fontsize=FS_NUM, fontweight="bold", color="white", zorder=8,
                bbox=dict(boxstyle="round,pad=0.08", facecolor=color,
                          edgecolor="none", alpha=0.85))

    # first = open circle, last = star (drawn on top of number)
    ax.scatter(*xy_s[0],  s=ms * 2.2, marker="o", zorder=6,
               facecolors="none", edgecolors=color, linewidths=2.0)
    ax.scatter(*xy_s[-1], s=ms * 3.2, marker="*", color=color, zorder=6,
               edgecolors="white", linewidths=0.6)


def add_context_hazard(ax, xy, risks, xlim, ylim, vabs):
    """Hazard score hexbin: coolwarm, all biopsies."""
    ax.hexbin(xy[:, 0], xy[:, 1], C=risks,
              reduce_C_function=np.nanmean,
              gridsize=38, cmap="coolwarm",
              vmin=-vabs, vmax=vabs,
              linewidths=0, zorder=1)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_facecolor(BG)


def add_context_tte(ax, xy, ttes, events, xlim, ylim):
    """TTE scatter + event ✕ markers."""
    ax.hexbin(xy[:, 0], xy[:, 1], gridsize=38, bins="log",
              cmap="Greys", mincnt=1, linewidths=0, alpha=0.35, zorder=1)
    tte_clip = np.clip(ttes, 0, np.nanpercentile(ttes, 99))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=tte_clip, cmap="plasma_r",
                    s=4, alpha=0.65, linewidths=0, zorder=2, rasterized=True)
    ev_idx = np.where(events == 1)[0]
    ax.scatter(xy[ev_idx, 0], xy[ev_idx, 1], marker="x", s=16,
               color="#B71C1C", linewidths=0.9, alpha=0.7, zorder=3)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_facecolor(BG)
    return sc


def plot_trajectories(task, records):
    print(f"\n=== {task} ===  ({len(records)} biopsies)")
    if not records:
        return

    pdict = per_patient_summary(records)
    print(f"  Patients: {len(pdict)}")

    all_reps = np.stack([r["rep"] for r in records])
    X = StandardScaler().fit_transform(all_reps)
    print(f"  UMAP embed {X.shape}...")
    xy = umap_embed(X)
    print(f"  Done.")

    risks  = np.array([r["risk"]  for r in records])
    ttes   = np.array([r["tte"]   for r in records])
    events = np.array([r["event"] for r in records])
    vabs   = float(np.nanpercentile(np.abs(risks), 98))

    all_risks   = np.array([info["mean_risk"] for info in pdict.values()])
    risk_median = float(np.nanmedian(all_risks))
    thr         = THRESHOLDS[task]

    for pid, info in pdict.items():
        info["quadrant"] = classify_quadrant(
            info["event"], info["tte"], info["mean_risk"], risk_median, thr)

    selected = {q: select_patients(pdict, q) for q in QUADS}
    for q in QUADS:
        print(f"  {q}: {sum(1 for i in pdict.values() if i.get('quadrant')==q)} patients, "
              f"showing {len(selected[q])}")

    pad  = 0.8
    xlim = (xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ylim = (xy[:, 1].min() - pad, xy[:, 1].max() + pad)

    # ── Layout: 4 rows × N_PAT patient cols ─────────────────────────────────
    nrows  = len(QUADS)
    cw_pat = 5.2
    ch     = 5.2

    fig = plt.figure(figsize=(N_PAT * cw_pat + 1.2, nrows * ch + 1.4),
                     facecolor=BG)
    fig.patch.set_facecolor(BG)

    cfg = TASKS[task]
    fig.suptitle(
        f"Per-Patient Biopsy Trajectories — {cfg['label']}\n"
        f"Background: model hazard score (red=high, blue=low)  ·  "
        f"Numbered visits in order  ·  ○ first  ★ last",
        fontsize=FS_SUPTITLE, fontweight="bold", y=1.005
    )

    gs = fig.add_gridspec(nrows, N_PAT, hspace=0.32, wspace=0.15,
                          left=0.10, right=0.99, top=0.93, bottom=0.05)

    for row_i, quad in enumerate(QUADS):
        col   = COL[quad]
        pids  = selected[quad]

        for pi in range(N_PAT):
            ax = fig.add_subplot(gs[row_i, pi])
            ax.set_facecolor(BG)
            ax.hexbin(xy[:, 0], xy[:, 1], C=risks,
                      reduce_C_function=np.nanmean,
                      gridsize=35, cmap="coolwarm",
                      vmin=-vabs, vmax=vabs,
                      linewidths=0, alpha=0.70, zorder=1)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xticks([]); ax.set_yticks([])
            ax.spines[:].set_visible(False)

            # row label on leftmost cell
            if pi == 0:
                ax.set_ylabel(QUAD_LABEL[quad], fontsize=FS_ROW,
                              color=col, fontweight="bold", labelpad=6)

            if pi < len(pids):
                pid  = pids[pi]
                info = pdict[pid]
                draw_trajectory(ax, xy, info["indices"], info["days"],
                                color=col, lw=2.0, ms=44)
                ev_str = (f"event {int(info['tte'])}d"
                          if info["event"] == 1 else f"cens. {int(info['tte'])}d")
                ax.set_title(
                    f"{pid}  ({ev_str})\nrisk={info['mean_risk']:.2f}  "
                    f"n={len(info['indices'])} visits",
                    fontsize=FS_CELL, color=col, fontweight="bold", pad=3)
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=14, color="#ccc")

    # shared hazard colorbar on right
    sm_h = plt.cm.ScalarMappable(cmap="coolwarm",
                                  norm=mcolors.Normalize(vmin=-vabs, vmax=vabs))
    sm_h.set_array([])
    cax_h = fig.add_axes([1.002, 0.15, 0.010, 0.70])
    cb_h  = fig.colorbar(sm_h, cax=cax_h)
    cb_h.set_label("Model hazard score", fontsize=10)
    cb_h.ax.tick_params(labelsize=9)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_els = (
        [Line2D([0],[0], marker="o", color=COL[q], markersize=8, lw=0,
                label=QUAD_LABEL[q].split("\n")[0]) for q in QUADS]
        + [Line2D([0],[0], marker="o", color="grey", markersize=7, lw=0,
                  markerfacecolor="none", label="First biopsy (○)"),
           Line2D([0],[0], marker="*", color="grey", markersize=10, lw=0,
                  label="Last biopsy (★)")]
    )
    fig.legend(handles=legend_els, fontsize=8, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.03),
               framealpha=0.92, edgecolor="#ccc")

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"biopsy_trajectories_{task}.{ext}", dpi=180,
                    bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved biopsy_trajectories_{task}")


for task in TASKS:
    records = load_all_splits(task)
    if records:
        plot_trajectories(task, records)
    else:
        print(f"  [skip] no data for {task}")

print("\nAll done.")
