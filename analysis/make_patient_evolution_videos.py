"""
Animated per-patient evolution videos for LongMIL showcase patients.

For each split × task × each of the 4 showcase patients:
  - One MP4 showing biopsy visits appearing one by one
  - Left panel:  UMAP trajectory building up (visit 1 → 1+2 → ... → all)
  - Right panel: Swimmer plot with BRICKS coloured by per-visit risk score

Output: figures/video/{task}_split{s}_patient_{pid}_evolution.mp4

Usage: sbatch analysis/submit_patient_evolution_videos.sh
"""

import subprocess, sys, tempfile
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler

ROOT    = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# ── locate bundled ffmpeg ─────────────────────────────────────────────────────
FFMPEG = "/home/aih/dinesh.haridoss/.local/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
print(f"ffmpeg: {FFMPEG}", flush=True)

# ── paths / constants ─────────────────────────────────────────────────────────
REPS_DIR = ROOT / "results" / "mm_abmil_v8" / "biopsy_reps"
OUT_DIR  = ROOT / "figures" / "video"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG       = "#FAF6F2"
HI       = "#952030"
LO       = "#1A5C8A"
DAY2MO   = 30.4375
SPLITS   = list(range(5))

TASK_CFG = {
    "acr_cls":   {"label": "ACR classification", "score_type": "cls",
                  "ev_key": "event_acr",   "tte_key": "tte_acr"},
    "acr_surv":  {"label": "Time to first ACR",  "score_type": "surv",
                  "ev_key": "event_acr",   "tte_key": "tte_acr"},
    "clad_surv": {"label": "Time to CLAD",       "score_type": "surv",
                  "ev_key": "event_clad",  "tte_key": "tte_clad"},
    "death_surv":{"label": "Time to death",      "score_type": "surv",
                  "ev_key": "event_death", "tte_key": "tte_death"},
}
SHOWCASE_COLS = ["#952030", "#1A5C8A", "#E07B20", "#2D8A4E"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _f(v):
    try:
        v = float(v)
        return float("nan") if (v != v) else v
    except Exception:
        return float("nan")


def load_records(task_key, split):
    import torch
    pt = REPS_DIR / f"biopsy_reps_{task_key}_split{split}.pt"
    if not pt.exists():
        raise FileNotFoundError(str(pt))
    d = torch.load(str(pt), map_location="cpu", weights_only=False)

    is_cls = task_key == "acr_cls"
    ev_key, tte_key = TASK_CFG[task_key]["ev_key"], TASK_CFG[task_key]["tte_key"]

    pid_arr    = d["patient_ids"]          # list of patient_id strings
    bdays      = d["biopsy_days"].numpy()  # (N,) days from t0
    rep        = d["reps"].numpy()         # (N, 256)
    risk       = d["risk"].numpy()         # (N,) raw model output
    tte_arr    = d["tte"].numpy()          # (N,)
    event_arr  = d["event"].numpy()        # (N,)

    records = []
    for i in range(len(pid_arr)):
        if is_cls:
            ev_v  = float(d.get("label", event_arr)[i]) if "label" in d else _f(event_arr[i])
        else:
            ev_v  = _f(event_arr[i])
        records.append({
            "patient_id": str(pid_arr[i]),
            "rep":        rep[i],
            "risk":       _f(risk[i]),
            "biopsy_day": _f(bdays[i]),
            "ev":         ev_v,
            "tte":        _f(tte_arr[i]),
        })
    return records


def risk_scores(raw_risks, score_type):
    r = np.asarray(raw_risks, dtype=float)
    valid = ~np.isnan(r)
    out = np.full_like(r, float("nan"))
    if score_type == "cls":
        # sigmoid → [0,1]
        out[valid] = 1.0 / (1.0 + np.exp(-r[valid]))
    else:
        # percentile rank
        vals = r[valid]
        ranks = np.argsort(np.argsort(vals)).astype(float) / max(len(vals) - 1, 1)
        out[valid] = ranks
    return out


def patient_groups(records, scores_all):
    pid_data = defaultdict(list)
    for i, r in enumerate(records):
        pid_data[r["patient_id"]].append({
            "idx":        i,
            "biopsy_day": _f(r["biopsy_day"]),
            "score":      float(scores_all[i]) if not np.isnan(scores_all[i]) else float("nan"),
            "ev":         _f(r["ev"]),
            "tte":        _f(r["tte"]),
        })
    patients = []
    for pid, visits in pid_data.items():
        vs = sorted(visits, key=lambda v: v["biopsy_day"])
        sc_vals = [v["score"] for v in vs if not np.isnan(v["score"])]
        patients.append({
            "patient_id": pid,
            "visits":     vs,
            "mean_score": float(np.mean(sc_vals)) if sc_vals else float("nan"),
            "ev":         _f(vs[0]["ev"]),
            "tte":        _f(vs[0]["tte"]),
            "biopsy_mo":  [v["biopsy_day"] / DAY2MO for v in vs],
        })
    return patients


def umap_embed(X, n_neighbors=30, min_dist=0.08):
    from umap import UMAP
    reducer = UMAP(n_components=2, n_neighbors=min(n_neighbors, len(X) - 1),
                   min_dist=min_dist, metric="cosine", random_state=42)
    return reducer.fit_transform(X)


def select_showcase(patients):
    all_pts = [p for p in patients if not np.isnan(p["mean_score"])]
    events  = sorted([p for p in all_pts if p["ev"] == 1],
                     key=lambda p: len(p["visits"]), reverse=True)
    if len(events) < 4:
        extra = sorted([p for p in all_pts if p["ev"] != 1],
                       key=lambda p: len(p["visits"]), reverse=True)
        showcase = (events + extra)[:4]
    else:
        showcase = events[:4]
    return sorted(showcase, key=lambda p: p["mean_score"], reverse=True)


# ── frame drawing ─────────────────────────────────────────────────────────────

def draw_frame(task_key, cfg, xy, scores_all, patient, t_visible, col):
    """
    Draw one animation frame showing visits 1..t_visible.
    Left:  UMAP trajectory (visits 1..t)
    Right: Swimmer with coloured bricks (one brick per visit interval)
    """
    cmap_risk = plt.cm.RdBu_r
    norm_risk  = Normalize(vmin=0, vmax=1)
    score_lbl  = "P(ACR+)" if cfg["score_type"] == "cls" else "Risk percentile"

    pid    = patient["patient_id"]
    visits = patient["visits"]
    T      = len(visits)
    idxs   = [v["idx"] for v in visits]
    bmo    = patient["biopsy_mo"]
    ev     = patient["ev"]
    tte    = patient["tte"]
    tte_mo = tte / DAY2MO if not np.isnan(tte) else float("nan")

    fig = plt.figure(figsize=(22, 9), facecolor=BG)
    fig.patch.set_facecolor(BG)

    ev_sym  = "Event" if ev == 1 else "Censored"
    tte_str = f"  TTE={tte_mo:.0f} mo" if not np.isnan(tte_mo) else ""
    fig.suptitle(
        f"{cfg['label']} — Patient {pid}  ({ev_sym}{tte_str})"
        f"  [visit {t_visible}/{T}]",
        fontsize=18, fontweight="bold", color=col, y=1.01,
    )

    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], figure=fig,
                           wspace=0.14, left=0.05, right=0.97,
                           top=0.92, bottom=0.12)
    ax_umap = fig.add_subplot(gs[0])
    ax_swim = fig.add_subplot(gs[1])

    # ── LEFT: UMAP ──────────────────────────────────────────────────────────
    valid = ~np.isnan(scores_all)
    if valid.sum() > 10:
        hx = ax_umap.hexbin(xy[valid, 0], xy[valid, 1], C=scores_all[valid],
                            reduce_C_function=np.mean, gridsize=25,
                            cmap=cmap_risk, linewidths=0.1, alpha=0.55)
        hx.set_clim(0, 1)
    ax_umap.scatter(xy[:, 0], xy[:, 1], c="lightgrey", s=4, alpha=0.3,
                    linewidths=0, zorder=2)

    # Draw arrows for visible visits
    for j in range(t_visible - 1):
        i0, i1 = idxs[j], idxs[j + 1]
        ax_umap.annotate("", xy=(xy[i1, 0], xy[i1, 1]),
                         xytext=(xy[i0, 0], xy[i0, 1]),
                         arrowprops=dict(arrowstyle="-|>", color=col,
                                         lw=4.5, alpha=0.95, mutation_scale=28),
                         zorder=6)

    vis_idxs = idxs[:t_visible]
    # Past visits slightly faded
    if t_visible > 1:
        ax_umap.scatter(xy[vis_idxs[:-1], 0], xy[vis_idxs[:-1], 1],
                        color=col, s=300, linewidths=2.0, edgecolors="white",
                        alpha=0.6, zorder=7)
        for rank, idx in enumerate(vis_idxs[:-1]):
            ax_umap.annotate(str(rank + 1), (xy[idx, 0], xy[idx, 1]),
                             fontsize=11, color="white", fontweight="bold",
                             ha="center", va="center", zorder=8)

    # Current visit: larger and fully opaque
    cur_idx = vis_idxs[-1]
    ax_umap.scatter([xy[cur_idx, 0]], [xy[cur_idx, 1]],
                    color=col, s=480, linewidths=2.5, edgecolors="white",
                    alpha=1.0, zorder=9)
    ax_umap.annotate(str(t_visible), (xy[cur_idx, 0], xy[cur_idx, 1]),
                     fontsize=13, color="white", fontweight="bold",
                     ha="center", va="center", zorder=10)
    ax_umap.annotate(f"  {pid}", (xy[cur_idx, 0], xy[cur_idx, 1]),
                     fontsize=13, color=col, fontweight="bold",
                     xytext=(12, 8), textcoords="offset points", zorder=11)

    ax_umap.set_title("UMAP trajectory (numbers = visit order)",
                      fontsize=13, fontweight="bold", pad=6)
    ax_umap.set_xlabel("UMAP-1", fontsize=11)
    ax_umap.set_ylabel("UMAP-2", fontsize=11)
    ax_umap.tick_params(labelsize=10)
    ax_umap.set_facecolor(BG)
    ax_umap.spines[["top", "right"]].set_visible(False)

    # ── RIGHT: Brick swimmer ─────────────────────────────────────────────────
    bar_h = 0.55
    y     = 0.0

    # Draw one coloured brick per completed interval
    for i in range(t_visible - 1):
        seg_start = bmo[i]
        seg_end   = bmo[i + 1]
        seg_w     = max(seg_end - seg_start, 0.05)
        v_score   = visits[i]["score"]
        seg_col   = cmap_risk(norm_risk(v_score)) if not np.isnan(v_score) else "#aaaaaa"
        ax_swim.barh(y, seg_w, left=seg_start, height=bar_h,
                     color=seg_col, alpha=0.92, linewidth=0.8,
                     edgecolor="white", zorder=3)

    # Biopsy visit circles (only visible ones)
    if t_visible > 0:
        ax_swim.scatter(bmo[:t_visible], [y] * t_visible,
                        s=80, color="white", edgecolors="#333333",
                        linewidths=0.9, zorder=4, alpha=0.95)

    # Event / censored marker at the last visible visit
    cur_mo = bmo[t_visible - 1]
    if t_visible == T:
        # Final visit — show event/censored marker
        if ev == 1:
            ax_swim.scatter([cur_mo], [y], marker="x", s=200,
                            color=HI, linewidths=2.5, zorder=5)
            ax_swim.text(cur_mo + 0.5, y,
                         f"  ✗ event  TTE={tte_mo:.0f} mo"
                         f"  risk={visits[-1]['score']:.2f}",
                         fontsize=11, va="center", color="#333333")
        else:
            ax_swim.scatter([cur_mo], [y], marker="|", s=200,
                            color=LO, linewidths=2.5, zorder=5)
            ax_swim.text(cur_mo + 0.5, y,
                         f"  ○ censored  TTE={tte_mo:.0f} mo"
                         f"  risk={visits[-1]['score']:.2f}",
                         fontsize=11, va="center", color="#333333")
    else:
        # Show current risk score
        cur_score = visits[t_visible - 1]["score"]
        if not np.isnan(cur_score):
            ax_swim.text(cur_mo + 0.5, y,
                         f"  risk={cur_score:.2f}",
                         fontsize=11, va="center", color=col, fontweight="bold")

    # Fix x-axis to the full timeline so bars don't jump
    total_mo = max(bmo) if bmo else 1.0
    if not np.isnan(tte_mo):
        total_mo = max(total_mo, tte_mo)
    ax_swim.set_xlim(-1, total_mo * 1.18)
    ax_swim.set_ylim(-0.6, 0.6)

    # Colorbar
    sm = ScalarMappable(cmap=cmap_risk, norm=norm_risk)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_swim, pad=0.02, fraction=0.03, shrink=0.4)
    cb.set_label(score_lbl, fontsize=11)
    cb.ax.tick_params(labelsize=10)

    # Legend
    leg_h = [
        Line2D([0], [0], marker="x", color=HI,
               markersize=9, linewidth=0, label="Event"),
        Line2D([0], [0], marker="|", color=LO,
               markersize=9, linewidth=1.5, label="Censored"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#333333", markersize=7, label="Biopsy visit"),
    ]
    ax_swim.legend(handles=leg_h, fontsize=10, framealpha=0.85, loc="upper left")
    ax_swim.set_xlabel("Months from first biopsy", fontsize=12)
    ax_swim.set_yticks([])
    ax_swim.set_title("Risk evolution over biopsy visits\n(brick colour = LongMIL risk score at each visit)",
                      fontsize=13, fontweight="bold", pad=6)
    ax_swim.tick_params(axis="x", labelsize=10)
    ax_swim.set_facecolor(BG)
    ax_swim.spines[["top", "right", "left"]].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ── video compilation ─────────────────────────────────────────────────────────

def compile_video(frame_paths, durations, out_path):
    """Compile list of PNGs with given per-frame durations into MP4."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for fp, dur in zip(frame_paths, durations):
            f.write(f"file '{fp}'\nduration {dur:.3f}\n")
        # ffmpeg concat needs a final repeat
        f.write(f"file '{frame_paths[-1]}'\nduration 0.04\n")
        list_path = f.name

    cmd = [
        FFMPEG, "-y",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,"
               "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0xFAF6F2,"
               "fps=25",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    Path(list_path).unlink(missing_ok=True)
    if r.returncode != 0:
        print(f"  [ffmpeg error]\n{r.stderr[-400:]}", flush=True)
        return False
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    for split in SPLITS:
        print(f"\n{'='*60}", flush=True)
        print(f"=== SPLIT {split} ===", flush=True)
        print(f"{'='*60}", flush=True)
        for task_key, cfg in TASK_CFG.items():
            print(f"\n=== {cfg['label']} (split {split}) ===", flush=True)
            try:
                records = load_records(task_key, split)
            except FileNotFoundError as e:
                print(f"  [skip] {e}", flush=True)
                continue

            risks_all  = np.array([_f(r["risk"]) for r in records])
            scores_all = risk_scores(risks_all, cfg["score_type"])

            X = np.stack([r["rep"] for r in records]).astype(np.float32)
            X = StandardScaler().fit_transform(X)
            print(f"  UMAP on {len(records)} biopsies...", flush=True)
            xy = umap_embed(X)

            patients   = patient_groups(records, scores_all)
            showcases  = select_showcase(patients)

            for pi, patient in enumerate(showcases):
                pid    = patient["patient_id"]
                col    = SHOWCASE_COLS[pi % len(SHOWCASE_COLS)]
                visits = patient["visits"]
                T      = len(visits)
                print(f"  Patient {pi+1}: {pid} ({T} visits)", flush=True)

                out_mp4 = OUT_DIR / f"{task_key}_split{split}_patient_{pid}_evolution.mp4"
                if out_mp4.exists():
                    print(f"    [skip] already exists", flush=True)
                    continue

                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    frame_paths = []
                    durations   = []

                    for t in range(1, T + 1):
                        fig = draw_frame(task_key, cfg, xy, scores_all, patient, t, col)
                        fp  = tmp / f"frame_{t:03d}.png"
                        fig.savefig(fp, dpi=120, bbox_inches="tight", facecolor=BG)
                        plt.close(fig)
                        frame_paths.append(str(fp))
                        durations.append(3.0 if (t == 1 or t == T) else 1.5)

                    ok = compile_video(frame_paths, durations, out_mp4)
                    if ok:
                        mb = out_mp4.stat().st_size / 1e6
                        print(f"    Saved {out_mp4.name}  ({mb:.1f} MB)", flush=True)

    print(f"\nDone. → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
