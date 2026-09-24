"""
Trajectory showcase — slideshow version.
One figure per task, 2×2 grid: TP / FN / TN / FP (one best patient each).
Each panel = full UMAP hazard background + single patient trajectory.
Large, clear, suitable for presentations.
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from sklearn.preprocessing import StandardScaler

ROOT     = Path(__file__).resolve().parent.parent
REPS_DIR = ROOT / "results" / "mm_abmil_v8" / "biopsy_reps"
OUT_DIR  = ROOT / "figures" / "interpretability" / "trajectories"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG = "#F7F4F1"

COL = {
    "TP": "#B71C1C",   # dark red
    "FN": "#E65100",   # orange
    "TN": "#1565C0",   # dark blue
    "FP": "#6A1B9A",   # purple
}
QUAD_LABEL = {
    "TP": "True Positive\n(Died early · HIGH risk)",
    "FN": "False Negative\n(Died early · LOW risk)",
    "TN": "True Negative\n(Long survival · LOW risk)",
    "FP": "False Positive\n(Long survival · HIGH risk)",
}
QUADS = ["TP", "FN", "TN", "FP"]

TASKS = {
    "death_surv": {"label": "Post-Transplant Mortality", "cindex": "0.759"},
    "acr_surv":   {"label": "ACR Survival",              "cindex": "0.655"},
    "clad_surv":  {"label": "CLAD Survival",             "cindex": "0.595"},
}
THRESHOLDS = {
    "death_surv": {"soon": 500, "long": 900},
    "acr_surv":   {"soon": 200, "long": 600},
    "clad_surv":  {"soon": 500, "long": 900},
}

FS_SUPTITLE = 26
FS_PANEL    = 18
FS_NUM      = 11


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


def pick_best(pdict, quad, min_visits=5):
    """Pick the single clearest example per quadrant: most visits, no NaN risk."""
    cands = [
        (pid, info) for pid, info in pdict.items()
        if info.get("quadrant") == quad
        and len(info["indices"]) >= min_visits
        and not np.isnan(info["mean_risk"])
    ]
    if not cands:
        cands = [(pid, info) for pid, info in pdict.items()
                 if info.get("quadrant") == quad and len(info["indices"]) >= 2
                 and not np.isnan(info["mean_risk"])]
    if not cands:
        return None, None
    # for TP/FN sort by descending/ascending risk extremeness + many visits
    if quad == "TP":
        cands.sort(key=lambda x: (-x[1]["mean_risk"], -len(x[1]["indices"])))
    elif quad == "FN":
        cands.sort(key=lambda x: (x[1]["mean_risk"], -len(x[1]["indices"])))
    elif quad == "TN":
        cands.sort(key=lambda x: (x[1]["mean_risk"], -len(x[1]["indices"])))
    else:  # FP
        cands.sort(key=lambda x: (-x[1]["mean_risk"], -len(x[1]["indices"])))
    pid, info = cands[0]
    return pid, info


def draw_trajectory(ax, xy_all, indices, days, color, lw=2.5, ms=90):
    order  = np.argsort(days)
    xy_s   = xy_all[indices][order]
    n      = len(xy_s)
    c_rgba = np.array(mcolors.to_rgba(color))

    # gradient arrows
    for i in range(n - 1):
        x0, y0 = xy_s[i]; x1, y1 = xy_s[i + 1]
        if np.sqrt((x1-x0)**2 + (y1-y0)**2) < 0.03:
            continue
        t         = (i + 1) / max(n - 1, 1)
        seg_alpha = 0.30 + 0.65 * t
        seg_lw    = lw * (0.55 + 0.65 * t)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->",
                                    color=(*c_rgba[:3], seg_alpha),
                                    lw=seg_lw, mutation_scale=14 + 6 * t))

    # dots with gradient
    for i, (x, y) in enumerate(xy_s):
        t = i / max(n - 1, 1)
        ax.scatter(x, y, s=ms * (0.45 + 0.65 * t),
                   color=(*c_rgba[:3], 0.35 + 0.60 * t),
                   edgecolors="white", linewidths=0.7, zorder=5)

    # visit number labels
    for i, (x, y) in enumerate(xy_s):
        ax.text(x, y, str(i + 1), ha="center", va="center",
                fontsize=FS_NUM, fontweight="bold", color="white", zorder=8,
                bbox=dict(boxstyle="round,pad=0.10", facecolor=color,
                          edgecolor="none", alpha=0.88))

    # first ○, last ★
    ax.scatter(*xy_s[0],  s=ms * 2.5, marker="o", zorder=6,
               facecolors="none", edgecolors=color, linewidths=2.5)
    ax.scatter(*xy_s[-1], s=ms * 3.5, marker="*", color=color, zorder=6,
               edgecolors="white", linewidths=0.8)


def plot_showcase(task, records):
    print(f"\n=== {task} ===  ({len(records)} biopsies)")
    if not records:
        return

    pdict = per_patient_summary(records)
    all_reps = np.stack([r["rep"] for r in records])
    X = StandardScaler().fit_transform(all_reps)
    print(f"  UMAP embed {X.shape}...")
    xy = umap_embed(X)
    print("  Done.")

    risks  = np.array([r["risk"]  for r in records])
    ttes   = np.array([r["tte"]   for r in records])
    events = np.array([r["event"] for r in records])
    vabs   = float(np.nanpercentile(np.abs(risks[~np.isnan(risks)]), 98))

    all_risks   = np.array([info["mean_risk"] for info in pdict.values()])
    risk_median = float(np.nanmedian(all_risks[~np.isnan(all_risks)]))
    thr         = THRESHOLDS[task]

    for pid, info in pdict.items():
        info["quadrant"] = classify_quadrant(
            info["event"], info["tte"], info["mean_risk"], risk_median, thr)

    pad  = 0.8
    xlim = (xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ylim = (xy[:, 1].min() - pad, xy[:, 1].max() + pad)

    cfg = TASKS[task]

    # ── 2×2 figure, one panel per quadrant ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(20, 18), facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Patient Trajectories in Biopsy Representation Space\n"
        f"{cfg['label']}  ·  C-index {cfg['cindex']}  ·  "
        f"Background: model hazard score (red=high, blue=low)\n"
        f"Numbered visits in order  ·  ○ first visit  ★ last visit",
        fontsize=FS_SUPTITLE, fontweight="bold", y=1.01
    )

    quad_pos = [("TP", 0, 0), ("FN", 0, 1), ("TN", 1, 0), ("FP", 1, 1)]

    for quad, ri, ci in quad_pos:
        ax = axes[ri, ci]
        ax.set_facecolor(BG)

        # Background: full UMAP hazard hexbin (slightly muted)
        hb = ax.hexbin(xy[:, 0], xy[:, 1], C=risks,
                       reduce_C_function=np.nanmean,
                       gridsize=40, cmap="coolwarm",
                       vmin=-vabs, vmax=vabs,
                       linewidths=0, alpha=0.65, zorder=1)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_visible(False)

        pid, info = pick_best(pdict, quad)
        col = COL[quad]

        if pid is not None:
            draw_trajectory(ax, xy, info["indices"], info["days"],
                            color=col, lw=2.8, ms=100)
            ev_str = (f"event {int(info['tte'])}d"
                      if info["event"] == 1 else f"censored {int(info['tte'])}d")
            n_visits = len(info["indices"])
            ax.set_title(
                f"{QUAD_LABEL[quad]}\n"
                f"{pid}  ·  {ev_str}  ·  {n_visits} visits  ·  risk={info['mean_risk']:.2f}",
                fontsize=FS_PANEL, color=col, fontweight="bold", pad=8
            )
        else:
            n_quad = sum(1 for i in pdict.values() if i.get("quadrant") == quad)
            ax.set_title(f"{QUAD_LABEL[quad]}\n(no clear example found; n={n_quad})",
                         fontsize=FS_PANEL, color=col, fontweight="bold", pad=8)
            ax.text(0.5, 0.5, "—", ha="center", va="center",
                    transform=ax.transAxes, fontsize=20, color="#bbb")

        # Count label bottom-left
        n_quad = sum(1 for i in pdict.values() if i.get("quadrant") == quad)
        ax.text(0.01, 0.01, f"n={n_quad} patients in quadrant",
                transform=ax.transAxes, fontsize=11, color=col,
                va="bottom", ha="left", alpha=0.75)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap="coolwarm",
                                norm=mcolors.Normalize(vmin=-vabs, vmax=vabs))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.65])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Model hazard score", fontsize=13)
    cb.ax.tick_params(labelsize=11)

    # Legend
    legend_els = [
        Line2D([0],[0], marker="o", color=COL[q], markersize=10, lw=0,
               label=QUAD_LABEL[q].replace("\n", " — ")) for q in QUADS
    ] + [
        Line2D([0],[0], marker="o", color="grey", markersize=9, lw=0,
               markerfacecolor="none", label="First visit (○)"),
        Line2D([0],[0], marker="*", color="grey", markersize=12, lw=0,
               label="Last visit (★)"),
    ]
    fig.legend(handles=legend_els, fontsize=12, ncol=3,
               loc="lower center", bbox_to_anchor=(0.46, -0.04),
               framealpha=0.95, edgecolor="#ccc",
               handlelength=1.6, handleheight=1.4, borderpad=0.8)

    plt.subplots_adjust(hspace=0.25, wspace=0.08,
                        left=0.02, right=0.90, top=0.91, bottom=0.06)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"trajectory_showcase_{task}.{ext}", dpi=180,
                    bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved trajectory_showcase_{task}")


for task in TASKS:
    records = load_all_splits(task)
    if records:
        plot_showcase(task, records)
    else:
        print(f"  [skip] no data for {task}")

print("\nAll done.")
