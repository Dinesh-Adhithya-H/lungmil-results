"""
Unified patient-rep UMAP — identical 6-panel figure for all 4 tasks.

Reads results_raw.npy from:
  SetMIL  ACR cls  → interpretability/set_mil_mt_interp/all_splits_cls/
  SetMIL  CLAD     → interpretability/set_mil_mt_interp/all_splits_clad_surv/
  LongMK  ACR surv → interpretability/longitudinal_mk_interp/longitudinal_mk_no_alibi_split{s}_fold0_acr_surv/ (all 5 splits merged)
  LongMK  Death    → interpretability/longitudinal_mk_interp/longitudinal_mk_no_alibi_split{s}_fold0_death_surv/

Output → figures/interpretability/{task}/unified_rep_umap_{task}.png/.pdf
         figures/interpretability/agg/unified_rep_umap_all_tasks.png/.pdf

Run via:  sbatch analysis/submit_unified_rep_umap.sh
"""

from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler

ROOT   = Path(__file__).resolve().parent.parent
INTERP = ROOT / "interpretability"
FIG_ROOT = ROOT / "figures" / "interpretability"

# ── palette (H&E slide deck) ──────────────────────────────────────────────────
HI  = "#952030"   # red   = high risk / event
LO  = "#1A5C8A"   # blue  = low risk / censored
MU  = "#7D6D78"   # muted
BG  = "#FAF6F2"
SPLIT_COLS = ["#E41A1C","#377EB8","#4DAF4A","#984EA3","#FF7F00"]

def _umap_embed(X, n_neighbors=15, min_dist=0.1, metric="cosine"):
    try:
        import umap
        return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                         metric=metric, random_state=42).fit_transform(X)
    except ImportError:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=42).fit_transform(X)

# ── data loading ──────────────────────────────────────────────────────────────
TASK_CFG = {
    "acr_cls": {
        "label":      "ACR classification",
        "metric":     "cls",
        "rep_key":    "acr_cls",
        "ev_key":     "event_acr",
        "tte_key":    "tte_acr",
        "score_type": "cls",
        "npy_paths":  [INTERP / "set_mil_mt_interp" / "all_splits_cls" / "results_raw.npy"],
    },
    "clad_surv": {
        "label":      "CLAD survival",
        "metric":     "C-index",
        "rep_key":    "clad_surv",
        "ev_key":     "event_clad",
        "tte_key":    "tte_clad",
        "score_type": "surv",
        "npy_paths":  [INTERP / "set_mil_mt_interp" / "all_splits_clad_surv" / "results_raw.npy"],
    },
    "acr_surv": {
        "label":      "ACR survival",
        "metric":     "C-index",
        "rep_key":    "acr_surv",
        "ev_key":     "event_acr",
        "tte_key":    "tte_acr",
        "score_type": "surv",
        "npy_paths":  [
            INTERP / f"longitudinal_mk_interp/longitudinal_mk_no_alibi_split{s}_fold0_acr_surv/results_raw.npy"
            for s in range(5)
        ],
    },
    "death_surv": {
        "label":      "Death survival",
        "metric":     "C-index",
        "rep_key":    "death_surv",
        "ev_key":     "event_death",
        "tte_key":    "tte_death",
        "score_type": "surv",
        "npy_paths":  [
            INTERP / f"longitudinal_mk_interp/longitudinal_mk_no_alibi_split{s}_fold0_death_surv/results_raw.npy"
            for s in range(5)
        ],
    },
}


def load_records(npy_paths, rep_key):
    records = []
    for p in npy_paths:
        if not p.exists():
            print(f"  [warn] not found: {p}")
            continue
        raw = np.load(p, allow_pickle=True)
        items = list(raw) if raw.ndim > 0 else [raw.item()]
        for item in items:
            if not isinstance(item, dict):
                continue
            # Support both SetMIL (final_reps dict) and longitudinal (rep_full dict)
            rep_dict = item.get("final_reps") or item.get("rep_full") or {}
            rep = rep_dict.get(rep_key)
            if rep is None:
                # Try without _surv suffix
                for k in rep_dict:
                    if rep_key.split("_")[0] in k:
                        rep = rep_dict[k]
                        break
            if rep is None:
                continue
            records.append({
                "rep":        np.asarray(rep, dtype=np.float32),
                "logit":      float(item.get("logits", {}).get(rep_key, float("nan"))
                                    if isinstance(item.get("logits"), dict)
                                    else float("nan")),
                "label":      item.get("label"),
                "ev":         item.get(list(filter(lambda k: "event" in k, item.keys()))[0]
                                       if any("event" in k for k in item.keys()) else "label"),
                "tte":        item.get(list(filter(lambda k: "tte" in k, item.keys()))[0]
                                       if any("tte" in k for k in item.keys()) else "label"),
                "present_mods": item.get("present_mods", []),
                "_split":     int(item.get("_split", -1)),
            })
    return records


def load_records_v2(npy_paths, rep_key, ev_key, tte_key):
    records = []
    for p in npy_paths:
        if not p.exists():
            print(f"  [warn] not found: {p}")
            continue
        raw = np.load(p, allow_pickle=True)
        items = list(raw) if raw.ndim > 0 else [raw.item()]
        for item in items:
            if not isinstance(item, dict):
                continue
            rep_dict = item.get("final_reps") or item.get("rep_full") or {}
            rep = rep_dict.get(rep_key)
            if rep is None:
                for k in rep_dict:
                    if any(t in k for t in rep_key.split("_")):
                        rep = rep_dict[k]
                        break
            if rep is None:
                continue
            logit_dict = item.get("logits", {})
            logit = float(logit_dict.get(rep_key, float("nan"))
                          if isinstance(logit_dict, dict) else float("nan"))
            records.append({
                "rep":         np.asarray(rep, dtype=np.float32),
                "logit":       logit,
                "label":       item.get("label"),
                "ev":          item.get(ev_key),
                "tte":         item.get(tte_key),
                "present_mods":item.get("present_mods", []),
                "_split":      int(item.get("_split", -1)),
            })
    return records


def _try_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _km_curve(ev, tte, color, ax, label=""):
    """Simple KM step plot."""
    from collections import defaultdict
    order = np.argsort(tte)
    ev_o = ev[order]
    tte_o = tte[order]
    n = len(ev_o)
    at_risk = n
    surv = 1.0
    t_pts, s_pts = [0.0], [1.0]
    for i in range(n):
        if ev_o[i] == 1:
            surv *= (at_risk - 1) / at_risk
        at_risk -= 1
        t_pts.append(tte_o[i])
        s_pts.append(surv)
    ax.step(t_pts, s_pts, where="post", color=color, lw=1.5, label=label)


def make_unified_umap(task_key, cfg, out_dir):
    records = load_records_v2(cfg["npy_paths"], cfg["rep_key"], cfg["ev_key"], cfg["tte_key"])
    if len(records) < 20:
        print(f"  [skip] {task_key}: only {len(records)} patients")
        return

    X = np.stack([r["rep"] for r in records]).astype(np.float32)
    X = StandardScaler().fit_transform(X)
    print(f"  [umap] {task_key}: N={len(records)}, X={X.shape}")
    xy = _umap_embed(X, n_neighbors=min(15, len(records)-1), min_dist=0.1, metric="cosine")

    logits  = np.array([_try_float(r["logit"]) for r in records])
    ev_raw  = np.array([_try_float(r["ev"])    for r in records])
    tte_raw = np.array([_try_float(r["tte"])   for r in records])
    labels  = np.array([_try_float(r["label"]) for r in records])
    splits  = np.array([r["_split"]            for r in records])

    # Risk score: sigmoid for cls, percentile for surv
    if cfg["score_type"] == "cls":
        valid_l = ~np.isnan(logits)
        scores = np.full_like(logits, float("nan"))
        scores[valid_l] = 1.0 / (1.0 + np.exp(-logits[valid_l]))
        score_lbl = "P(ACR+)"
    else:
        from scipy.stats import rankdata
        valid_l = ~np.isnan(logits)
        scores = np.full_like(logits, float("nan"))
        if valid_l.sum() > 1:
            scores[valid_l] = (rankdata(logits[valid_l]) - 1) / max(valid_l.sum() - 1, 1)
        score_lbl = "Risk percentile"

    # TTE normalise to years for display
    tte_yrs = tte_raw / 365.25

    # Modality combos
    raw_combos = ["+".join(sorted(r["present_mods"])) for r in records]
    cc = Counter(raw_combos)
    major = sorted([c for c, n in cc.items() if n >= 5], key=lambda c: -cc[c])
    try:
        _tab20 = matplotlib.colormaps["tab20"]
    except (KeyError, AttributeError):
        _tab20 = plt.cm.get_cmap("tab20")
    combo_col = {c: _tab20(i % 20) for i, c in enumerate(major)}
    combo_col["Other"] = "#aaaaaa"
    disp_combos = [c if c in combo_col else "Other" for c in raw_combos]

    # ── Build figure: 2×3 grid ────────────────────────────────────────────────
    MS = 9
    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    fig.suptitle(f"Patient Rep Space  —  {cfg['label']}  (N={len(records)})",
                 fontsize=12, fontweight="bold", y=0.98, color="#1A1018")
    grd = gridspec.GridSpec(2, 3, figure=fig, wspace=0.28, hspace=0.38)
    ax_score = fig.add_subplot(grd[0, 0])
    ax_label = fig.add_subplot(grd[0, 1])
    ax_tte   = fig.add_subplot(grd[0, 2])
    ax_mod   = fig.add_subplot(grd[1, 0])
    ax_km    = fig.add_subplot(grd[1, 1])
    ax_split = fig.add_subplot(grd[1, 2])

    def _scatter(ax, vals, cmap, title, vmin=None, vmax=None, alpha=0.8, s=MS):
        valid = ~np.isnan(vals)
        sc = ax.scatter(xy[valid, 0], xy[valid, 1], c=vals[valid], cmap=cmap,
                        s=s, alpha=alpha, linewidths=0,
                        vmin=(vmin if vmin is not None else np.nanmin(vals)),
                        vmax=(vmax if vmax is not None else np.nanmax(vals)))
        if (~valid).any():
            ax.scatter(xy[~valid, 0], xy[~valid, 1], c="lightgrey",
                       s=s * 0.6, alpha=0.3, linewidths=0)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046, shrink=0.85)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("UMAP-1", fontsize=8); ax.set_ylabel("UMAP-2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_facecolor(BG)
        ax.spines[["top","right"]].set_visible(False)

    # 0 — risk score
    _scatter(ax_score, scores, "RdBu_r", f"① Risk score\n({score_lbl})", vmin=0, vmax=1)

    # 1 — label / event
    _scatter(ax_label, labels, "RdBu_r", "② True label\n(ACR+/event=red)", vmin=0, vmax=1)

    # 2 — TTE (years), event markers
    valid_t = ~np.isnan(tte_yrs)
    ev_cols = np.where(ev_raw == 1, HI, LO)
    ax_tte.scatter(xy[valid_t, 0], xy[valid_t, 1],
                   c=ev_cols[valid_t], s=np.clip(12 - tte_yrs[valid_t] * 1.2, 3, 14),
                   alpha=0.75, linewidths=0)
    ax_tte.set_title(f"③ TTE (size∝urgency)\n(red=event, blue=censored)", fontsize=9, fontweight="bold", pad=4)
    ax_tte.set_xlabel("UMAP-1", fontsize=8); ax_tte.set_ylabel("UMAP-2", fontsize=8)
    ax_tte.tick_params(labelsize=7); ax_tte.set_facecolor(BG)
    ax_tte.spines[["top","right"]].set_visible(False)

    # 3 — modality combination
    for combo in list(combo_col.keys()):
        idx = [i for i, c in enumerate(disp_combos) if c == combo]
        if not idx:
            continue
        ax_mod.scatter(xy[idx, 0], xy[idx, 1], color=combo_col[combo],
                       s=MS, alpha=0.75, linewidths=0, label=combo[:30])
    ax_mod.set_title("④ Modality combination", fontsize=9, fontweight="bold", pad=4)
    ax_mod.set_xlabel("UMAP-1", fontsize=8); ax_mod.set_ylabel("UMAP-2", fontsize=8)
    ax_mod.tick_params(labelsize=7); ax_mod.set_facecolor(BG)
    ax_mod.spines[["top","right"]].set_visible(False)
    ax_mod.legend(fontsize=5.5, framealpha=0.7, markerscale=1.2,
                  loc="lower right", ncol=1, borderpad=0.4)

    # 4 — KM: top vs bottom tertile
    valid_km = (~np.isnan(scores)) & (~np.isnan(tte_raw)) & (~np.isnan(ev_raw))
    if valid_km.sum() >= 20:
        q33, q67 = np.nanpercentile(scores[valid_km], [33, 67])
        hi_mask = valid_km & (scores >= q67)
        lo_mask = valid_km & (scores <= q33)
        if hi_mask.sum() > 5 and lo_mask.sum() > 5:
            _km_curve(ev_raw[hi_mask], tte_raw[hi_mask] / 365.25, HI, ax_km, "Top tertile (high risk)")
            _km_curve(ev_raw[lo_mask], tte_raw[lo_mask] / 365.25, LO, ax_km, "Bottom tertile (low risk)")
            ax_km.set_xlabel("Time (years)", fontsize=8)
            ax_km.set_ylabel("Event-free probability", fontsize=8)
            ax_km.legend(fontsize=7, framealpha=0.8)
    ax_km.set_title("⑤ KM: top vs bottom risk tertile", fontsize=9, fontweight="bold", pad=4)
    ax_km.tick_params(labelsize=7); ax_km.set_facecolor(BG)
    ax_km.spines[["top","right"]].set_visible(False)
    ax_km.set_ylim(-0.05, 1.05)

    # 5 — CV split
    uniq_splits = sorted(set(splits.tolist()))
    for si, sp in enumerate(uniq_splits):
        idx = np.where(splits == sp)[0]
        col = SPLIT_COLS[si % len(SPLIT_COLS)]
        ax_split.scatter(xy[idx, 0], xy[idx, 1], color=col, s=MS,
                         alpha=0.7, linewidths=0, label=f"s{sp}" if sp >= 0 else "all")
    ax_split.set_title("⑥ CV split annotation", fontsize=9, fontweight="bold", pad=4)
    ax_split.set_xlabel("UMAP-1", fontsize=8); ax_split.set_ylabel("UMAP-2", fontsize=8)
    ax_split.tick_params(labelsize=7); ax_split.set_facecolor(BG)
    ax_split.spines[["top","right"]].set_visible(False)
    ax_split.legend(fontsize=6.5, framealpha=0.8, ncol=2, markerscale=1.2)

    fig.patch.set_facecolor(BG)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"unified_rep_umap_{task_key}.{ext}", dpi=180,
                    bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [done] unified_rep_umap_{task_key} → {out_dir}")


# ── Run all tasks ─────────────────────────────────────────────────────────────
for task_key, cfg in TASK_CFG.items():
    task_out = FIG_ROOT / task_key
    make_unified_umap(task_key, cfg, task_out)

# ── 4-panel combined overview ─────────────────────────────────────────────────
print("[agg] building 4-panel overview...")
TASK_KEYS = list(TASK_CFG.keys())
TASK_LABELS_BRIEF = {"acr_cls":"ACR cls","clad_surv":"CLAD surv","acr_surv":"ACR surv","death_surv":"Death surv"}

agg_dir = FIG_ROOT / "agg"
agg_dir.mkdir(parents=True, exist_ok=True)

fig_agg, axes_agg = plt.subplots(2, 2, figsize=(16, 13), facecolor=BG)
fig_agg.suptitle("Patient Representation Space — All Tasks (Risk Score coloring)",
                 fontsize=12, fontweight="bold")

for ax, tk in zip(axes_agg.flat, TASK_KEYS):
    cfg = TASK_CFG[tk]
    records = load_records_v2(cfg["npy_paths"], cfg["rep_key"], cfg["ev_key"], cfg["tte_key"])
    ax.set_facecolor(BG)
    if len(records) < 20:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(TASK_LABELS_BRIEF[tk])
        continue
    X = StandardScaler().fit_transform(np.stack([r["rep"] for r in records]).astype(np.float32))
    xy_agg = _umap_embed(X, n_neighbors=min(15, len(records)-1), min_dist=0.1, metric="cosine")
    logits = np.array([_try_float(r["logit"]) for r in records])
    if cfg["score_type"] == "cls":
        valid = ~np.isnan(logits)
        scores = np.full_like(logits, float("nan"))
        scores[valid] = 1.0 / (1.0 + np.exp(-logits[valid]))
    else:
        from scipy.stats import rankdata
        valid = ~np.isnan(logits)
        scores = np.full_like(logits, float("nan"))
        if valid.sum() > 1:
            scores[valid] = (rankdata(logits[valid]) - 1) / max(valid.sum() - 1, 1)
    v = ~np.isnan(scores)
    sc = ax.scatter(xy_agg[v, 0], xy_agg[v, 1], c=scores[v], cmap="RdBu_r",
                    s=8, alpha=0.8, linewidths=0, vmin=0, vmax=1)
    if (~v).any():
        ax.scatter(xy_agg[~v, 0], xy_agg[~v, 1], c="lightgrey", s=5, alpha=0.3, linewidths=0)
    fig_agg.colorbar(sc, ax=ax, pad=0.02, fraction=0.046, shrink=0.85)
    ax.set_title(f"{TASK_LABELS_BRIEF[tk]}  (N={len(records)})", fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP-1", fontsize=8); ax.set_ylabel("UMAP-2", fontsize=8)
    ax.tick_params(labelsize=7); ax.spines[["top","right"]].set_visible(False)

fig_agg.tight_layout()
for ext in ("png", "pdf"):
    fig_agg.savefig(agg_dir / f"unified_rep_umap_all_tasks.{ext}", dpi=180,
                    bbox_inches="tight", facecolor=BG)
plt.close(fig_agg)
print(f"[done] agg overview → {agg_dir}")
