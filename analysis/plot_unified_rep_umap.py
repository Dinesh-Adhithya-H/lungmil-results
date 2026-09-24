"""
Per-biopsy patient representation UMAP — three figures per task × split:

  Figure 1: technical_umap_{task}    (1×3) — biopsy index, modality combo, anchor day
  Figure 2: risk_events_umap_{task}  (2×2) — risk score, event indicator,
                                             TTE hexbin (months, red=sooner),
                                             event density hexbin
  Figure 3: trajectories_{task}      asymmetric — trajectory UMAP (per-patient
                                     connected paths) + swimmer/timeline plot
  Figure 3b: trajectories_{task}_showcase   4-patient showcase version

Each UMAP point = one biopsy visit (per-biopsy reps from biopsy_reps_{task}_split{s}.pt).
Patient trajectories are drawn as connected arrow-paths in the UMAP.

Run via: sbatch analysis/submit_unified_rep_umap.sh
"""

from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.preprocessing import StandardScaler

ROOT     = Path(__file__).resolve().parent.parent
BIOREPS     = ROOT / "results" / "mm_abmil_v8" / "biopsy_reps"
BIOREPS_ALL = ROOT / "results" / "mm_abmil_v8" / "biopsy_reps_all"
FIG_ROOT = ROOT / "figures" / "interpretability" / "patient_rep_umap"

SPLITS_CSV = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")

BG         = "#FAF6F2"
HI         = "#952030"   # red   = high risk / event
LO         = "#1A5C8A"   # blue  = low risk / censored

DAY2MO = 30.4375  # days → months


# ── Splits CSV lookups ─────────────────────────────────────────────────────────

def _build_lookups():
    df = pd.read_csv(SPLITS_CSV)
    MOD_MAP = {"HE": "has_HE", "BAL": "has_BAL", "CT": "has_CT", "Clinical": "has_Clinical"}

    df["_dt"] = pd.to_datetime(df["anchor_dt"], errors="coerce")
    earliest = df["_dt"].min()

    patient_lkp = {}
    for pid, grp in df.groupby("patient_id"):
        r0 = grp.iloc[0]

        def _f(c):
            return float(r0[c]) if c in r0.index and pd.notna(r0[c]) else float("nan")

        mods = [m for m, col in MOD_MAP.items()
                if col in grp.columns and grp[col].any()]

        sorted_grp = grp.sort_values("_dt")
        dates = sorted_grp["_dt"].dropna()
        if len(dates):
            t0 = dates.iloc[0]
            anchor_mo = (t0 - earliest).days / DAY2MO
        else:
            anchor_mo = float("nan")

        patient_lkp[pid] = {
            "label":        _f("label"),
            "event_acr":    _f("acr_status"),
            "tte_acr":      _f("acr_days"),
            "event_clad":   _f("clad_status"),
            "tte_clad":     _f("clad_days"),
            "event_death":  _f("death_status"),
            "tte_death":    _f("death_days"),
            "present_mods": mods,
            "anchor_mo":    anchor_mo,
        }

    return patient_lkp


PATIENT_LOOKUP = _build_lookups()
print(f"[lookup] {len(PATIENT_LOOKUP)} patients")


# ── Task config ────────────────────────────────────────────────────────────────

_TASK_BASE = {
    "acr_cls":   {"label": "ACR classification", "metric": "BACC",    "rep_key": "acr_cls",
                  "ev_key": "event_acr",   "tte_key": "tte_acr",   "score_type": "cls",
                  "pt_stem": "acr_cls",    "is_cls": True},
    "acr_surv":  {"label": "Time to first ACR",   "metric": "C-index", "rep_key": "acr_surv",
                  "ev_key": "event_acr",   "tte_key": "tte_acr",   "score_type": "surv",
                  "pt_stem": "acr_surv",   "is_cls": False},
    "clad_surv": {"label": "Time to CLAD",        "metric": "C-index", "rep_key": "clad",
                  "ev_key": "event_clad",  "tte_key": "tte_clad",  "score_type": "surv",
                  "pt_stem": "clad_surv",  "is_cls": False},
    "death_surv":{"label": "Time to death",       "metric": "C-index", "rep_key": "death",
                  "ev_key": "event_death", "tte_key": "tte_death", "score_type": "surv",
                  "pt_stem": "death_surv", "is_cls": False},
}

def _pt_path(task_key, split):
    stem = _TASK_BASE[task_key]["pt_stem"]
    # Prefer the all-patients file (train+val+test); fall back to test-only
    all_path = BIOREPS_ALL / f"biopsy_reps_all_{stem}_split{split}.pt"
    if all_path.exists():
        return all_path
    return BIOREPS / f"biopsy_reps_{stem}_split{split}.pt"


# ── Biopsy record loading ──────────────────────────────────────────────────────

def load_biopsy_records(pt_path, ev_key, tte_key, is_cls=False):
    """Load per-biopsy records from biopsy_reps .pt file.

    For classification tasks (is_cls=True), ev is taken from the per-biopsy
    label tensor (ACR+ = 1, ACR− = 0) rather than the patient-level event field.
    """
    if not pt_path.exists():
        print(f"  [warn] not found: {pt_path}")
        return []

    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    pids    = d["patient_ids"]        # list[str], len=N
    stems   = d["stems"]              # list[str]
    bdays   = d["biopsy_days"].numpy()   # (N,) days from patient t0
    reps    = d["reps"].numpy()          # (N, 256)
    risk    = d["risk"].numpy()          # (N,) raw model output
    tte     = d["tte"].numpy()           # (N,)
    event   = d["event"].numpy()         # (N,)
    label   = d["label"].numpy()         # (N,) biopsy-level class label

    # Build per-patient biopsy index (0 = first visit)
    pid_order: dict = defaultdict(list)
    for i, pid in enumerate(pids):
        pid_order[pid].append(i)
    biopsy_idx_arr = np.zeros(len(pids), dtype=int)
    pid_n_biopsies: dict = {}
    for pid, idxs in pid_order.items():
        sorted_idxs = sorted(idxs, key=lambda i: bdays[i])
        for rank, i in enumerate(sorted_idxs):
            biopsy_idx_arr[i] = rank
        pid_n_biopsies[pid] = len(idxs)

    split_types = d.get("split_types", None)  # list[str] or None

    records = []
    for i, pid in enumerate(pids):
        lkp = PATIENT_LOOKUP.get(pid, {})

        def _lkp(k, tensor):
            v = float(tensor[i])
            if np.isnan(v):
                v = float(lkp.get(k, float("nan")) or float("nan"))
            return v

        if is_cls:
            # For classification: use per-biopsy label (ACR+ = 1, ACR− = 0)
            ev_v  = float(label[i])
            tte_v = float("nan")
        else:
            ev_v  = _lkp(ev_key,  event)
            tte_v = _lkp(tte_key, tte)

        records.append({
            "rep":          reps[i].astype(np.float32),
            "risk":         float(risk[i]),
            "biopsy_day":   float(bdays[i]),
            "biopsy_idx":   int(biopsy_idx_arr[i]),
            "patient_id":   pid,
            "ev":           ev_v,
            "tte":          tte_v,
            "present_mods": lkp.get("present_mods", []),
            "anchor_mo":    lkp.get("anchor_mo", float("nan")),
            "n_biopsies":   pid_n_biopsies[pid],
            "split_type":   split_types[i] if split_types is not None else "test",
        })
    return records


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _risk_scores(raw_risks, score_type):
    raw = np.asarray(raw_risks, dtype=float)
    valid = ~np.isnan(raw)
    scores = np.full_like(raw, float("nan"))
    if score_type == "cls":
        scores[valid] = 1.0 / (1.0 + np.exp(-raw[valid]))
    else:
        from scipy.stats import rankdata
        if valid.sum() > 1:
            scores[valid] = (rankdata(raw[valid]) - 1) / max(valid.sum() - 1, 1)
    return scores


# ── UMAP embedding ─────────────────────────────────────────────────────────────

def _umap_embed(X, n_neighbors=15, min_dist=0.1, metric="cosine"):
    try:
        import umap
        return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                         metric=metric, random_state=42).fit_transform(X)
    except ImportError:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=42).fit_transform(X)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _style_ax(ax, title, fs=9):
    ax.set_title(title, fontsize=fs, fontweight="bold", pad=4)
    ax.set_xlabel("UMAP-1", fontsize=8)
    ax.set_ylabel("UMAP-2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)


def _scatter(fig, ax, xy, vals, cmap, title, vmin=None, vmax=None,
             alpha=0.8, s=9, cbar_label=""):
    valid = ~np.isnan(vals)
    vl = vmin if vmin is not None else np.nanmin(vals)
    vh = vmax if vmax is not None else np.nanmax(vals)
    sc = ax.scatter(xy[valid, 0], xy[valid, 1], c=vals[valid], cmap=cmap,
                    s=s, alpha=alpha, linewidths=0, vmin=vl, vmax=vh)
    if (~valid).any():
        ax.scatter(xy[~valid, 0], xy[~valid, 1], c="lightgrey",
                   s=s * 0.6, alpha=0.3, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046, shrink=0.85)
    cb.ax.tick_params(labelsize=7)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=7)
    _style_ax(ax, title)
    return sc


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Technical validation  (1 × 3)
# ══════════════════════════════════════════════════════════════════════════════

def make_technical_fig(task_key, cfg, records, xy, out_dir, split=None):
    anchors   = np.array([_f(r["anchor_mo"])  for r in records])
    bday_arr  = np.array([_f(r["biopsy_day"]) for r in records])

    raw_combos = ["+".join(sorted(r["present_mods"])) for r in records]
    cc    = Counter(raw_combos)
    major = sorted([c for c, n in cc.items() if n >= 3], key=lambda c: -cc[c])
    try:
        _tab20 = matplotlib.colormaps["tab20"]
    except (KeyError, AttributeError):
        _tab20 = plt.cm.get_cmap("tab20")
    combo_col  = {c: _tab20(i % 20) for i, c in enumerate(major)}
    combo_col["Other"] = "#aaaaaa"
    disp_combo = [c if c in combo_col else "Other" for c in raw_combos]

    # Check if split_type info is available (from _all_ files)
    split_types = [r.get("split_type", "test") for r in records]
    has_split_type = len(set(split_types)) > 1  # True if train+val+test present

    N = len(records)
    split_lbl = f"split {split}, " if split is not None else ""
    merged = ["train" if st in ("train", "val") else "test" for st in split_types]
    SPLIT_TYPE_COL = {"train": "#4878CF", "test": "#D65F5F"}

    fig, axes = plt.subplots(1, 3, figsize=(19, 6), facecolor=BG)
    fig.suptitle(
        f"Technical Validation — {cfg['label']}  ({split_lbl}N={N} sample visits)",
        fontsize=11, fontweight="bold", y=1.01, color="#1A1018",
    )

    # ① Train / Test split
    ax = axes[0]
    if has_split_type:
        for stype, col in SPLIT_TYPE_COL.items():
            idx = [i for i, st in enumerate(merged) if st == stype]
            if not idx:
                continue
            ax.scatter(xy[idx, 0], xy[idx, 1], color=col, s=7, alpha=0.75,
                       linewidths=0, label=f"{stype} (n={len(idx)})")
        ax.legend(fontsize=8, framealpha=0.85, markerscale=1.5, loc="lower right")
        _style_ax(ax, f"① Train / Test\n(blue=train+val, red=test; split {split})")
    else:
        ax.scatter(xy[:, 0], xy[:, 1], color="#D65F5F", s=7, alpha=0.75,
                   linewidths=0, label=f"test (n={N})")
        ax.legend(fontsize=8, framealpha=0.85, markerscale=1.5, loc="lower right")
        _style_ax(ax, f"① Test only\n(train reps not yet extracted; N={N})")

    # ② Modality combination
    ax = axes[1]
    for combo in list(combo_col.keys()):
        idx = [i for i, c in enumerate(disp_combo) if c == combo]
        if not idx:
            continue
        n_pts = len(idx)
        ax.scatter(xy[idx, 0], xy[idx, 1], color=combo_col[combo], s=7,
                   alpha=0.75, linewidths=0, label=f"{combo[:25]} (n={n_pts})")
    ax.legend(fontsize=5.5, framealpha=0.7, markerscale=1.2,
              loc="lower right", ncol=1, borderpad=0.4)
    _style_ax(ax, "② Modality combination")

    # ③ Anchor day (months into study — when the patient started)
    ax = axes[2]
    valid_a = ~np.isnan(anchors)
    if valid_a.sum() > 10:
        sc = ax.scatter(xy[valid_a, 0], xy[valid_a, 1], c=anchors[valid_a],
                        cmap="plasma", s=7, alpha=0.82, linewidths=0)
        if (~valid_a).any():
            ax.scatter(xy[~valid_a, 0], xy[~valid_a, 1], c="lightgrey",
                       s=5, alpha=0.3, linewidths=0)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046, shrink=0.85)
        cb.set_label("Anchor sample — months into study", fontsize=7)
        cb.ax.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, "anchor date\nnot available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888888")
    _style_ax(ax, "③ Anchor day\n(temporal coverage of representations)")

    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        dpi = 2000 if ext == "png" else 300
        fig.savefig(out_dir / f"technical_umap_{task_key}.{ext}",
                    dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [done] technical_umap_{task_key}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Risk & Events  (2 × 2)
# ══════════════════════════════════════════════════════════════════════════════

def make_risk_events_fig(task_key, cfg, records, xy, out_dir):
    risks   = np.array([_f(r["risk"]) for r in records])
    ev_raw  = np.array([_f(r["ev"])   for r in records])
    tte_raw = np.array([_f(r["tte"])  for r in records])
    scores  = _risk_scores(risks, cfg["score_type"])
    tte_mo  = tte_raw / DAY2MO

    score_lbl = "P(ACR+)" if cfg["score_type"] == "cls" else "Risk percentile"
    pos_lbl   = "ACR+" if cfg["score_type"] == "cls" else "Event"
    neg_lbl   = "ACR−" if cfg["score_type"] == "cls" else "Censored"

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor=BG)
    fig.suptitle(f"Risk & Events — {cfg['label']}  (N={len(records)} sample visits)",
                 fontsize=11, fontweight="bold", y=1.01, color="#1A1018")

    # ① Risk score scatter
    _scatter(fig, axes[0, 0], xy, scores, "RdBu_r",
             f"① Risk score  ({score_lbl})",
             vmin=0, vmax=1, cbar_label=score_lbl)

    # ② Event indicator — all biopsies colored by patient outcome
    ax = axes[0, 1]
    valid_e  = ~np.isnan(ev_raw)
    ev_cols  = np.where(ev_raw == 1, HI, LO)
    ax.scatter(xy[valid_e, 0],  xy[valid_e, 1],  c=ev_cols[valid_e],
               s=7, alpha=0.78, linewidths=0)
    if (~valid_e).any():
        ax.scatter(xy[~valid_e, 0], xy[~valid_e, 1], c="lightgrey",
                   s=5, alpha=0.3, linewidths=0)
    n_ev  = int((ev_raw[valid_e] == 1).sum())
    n_cen = int((ev_raw[valid_e] == 0).sum())
    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=HI,
               markersize=6, label=f"{pos_lbl} (n visits={n_ev})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LO,
               markersize=6, label=f"{neg_lbl} (n visits={n_cen})"),
    ]
    ax.legend(handles=leg, fontsize=7, framealpha=0.8, loc="lower right")
    _style_ax(ax, f"② Event indicator\n({pos_lbl}=red, {neg_lbl}=blue)")

    # ③ TTE hexbin (survival) OR biopsy-day scatter (classification)
    ax = axes[1, 0]
    is_cls = cfg.get("is_cls", False)
    if is_cls:
        # For classification: show biopsy day in months (temporal spread)
        bday_mo = np.array([_f(r["biopsy_day"]) for r in records]) / DAY2MO
        valid_bd = ~np.isnan(bday_mo)
        if valid_bd.sum() > 5:
            sc_bd = ax.scatter(xy[valid_bd, 0], xy[valid_bd, 1],
                               c=bday_mo[valid_bd], cmap="plasma",
                               s=8, alpha=0.75, linewidths=0)
            cb_bd = fig.colorbar(sc_bd, ax=ax, pad=0.02, fraction=0.046, shrink=0.85)
            cb_bd.set_label("Months from first sample", fontsize=7)
            cb_bd.ax.tick_params(labelsize=7)
        _style_ax(ax, "③ Sample timeline\n(months from patient's first sample)")
    else:
        ev_mask  = ~np.isnan(tte_mo) & (ev_raw == 1)
        cen_mask = ~np.isnan(ev_raw) & (ev_raw == 0)

        if ev_mask.sum() > 5:
            max_tte = float(np.nanpercentile(tte_mo[ev_mask], 95))
            tte_clipped = np.clip(tte_mo, 0, max_tte)

            hx = ax.hexbin(
                xy[ev_mask, 0], xy[ev_mask, 1],
                C=tte_clipped[ev_mask],
                reduce_C_function=np.mean,
                gridsize=30,
                cmap="RdBu",        # 0 months → red; max → blue
                linewidths=0.15)
            hx.set_clim(0, max_tte)
            cb = fig.colorbar(hx, ax=ax, pad=0.02, fraction=0.046, shrink=0.85)
            cb.set_label("Mean TTE at event (months)", fontsize=7)
            cb.ax.tick_params(labelsize=7)

            # Censored overlaid as white dots
            ax.scatter(xy[cen_mask, 0], xy[cen_mask, 1], marker="o", s=5,
                       color="white", edgecolors="#aaaaaa", linewidths=0.3,
                       alpha=0.7, label=f"Censored (n={cen_mask.sum()})", zorder=3)
            ax.legend(fontsize=6.5, framealpha=0.78, loc="lower right", markerscale=1.1)
        else:
            ax.text(0.5, 0.5, "no event TTE data", ha="center", va="center",
                    transform=ax.transAxes)

        _style_ax(ax, "③ TTE hexbin (events only)\nred=sooner, blue=later (months); white=censored")
    ax.set_facecolor("#f0ede8")

    # ④ Event density hexbin — white=0, red=1 (linear)
    from matplotlib.colors import LinearSegmentedColormap
    white_red = LinearSegmentedColormap.from_list("white_red", ["white", HI])
    ax = axes[1, 1]
    valid_ev = ~np.isnan(ev_raw)
    if valid_ev.sum() > 10:
        hx_ev = ax.hexbin(
            xy[valid_ev, 0], xy[valid_ev, 1],
            C=ev_raw[valid_ev],
            reduce_C_function=np.mean,
            gridsize=30,
            cmap=white_red,
            linewidths=0.15)
        hx_ev.set_clim(0, 1)
        cb_ev = fig.colorbar(hx_ev, ax=ax, pad=0.02, fraction=0.046, shrink=0.85)
        cb_ev.set_label("Event rate (0=none → 1=all)", fontsize=7)
        cb_ev.ax.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, "no event data", ha="center", va="center",
                transform=ax.transAxes)

    _style_ax(ax, "④ Event density hexbin\nwhite=0 (no events), red=1 (all events)")
    ax.set_facecolor("#f0ede8")

    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        dpi = 2000 if ext == "png" else 300
        fig.savefig(out_dir / f"risk_events_umap_{task_key}.{ext}",
                    dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [done] risk_events_umap_{task_key}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Patient trajectories
#   Panel A: per-patient connected paths in UMAP (colour = risk score at each biopsy)
#   Panel B: swimmer/timeline plot (sorted by risk)
# ══════════════════════════════════════════════════════════════════════════════

def _patient_groups(records, scores_all):
    """Group records by patient_id, sorted chronologically by biopsy_day."""
    pid_data: dict = defaultdict(list)
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
        visits_sorted = sorted(visits, key=lambda v: v["biopsy_day"])
        scores_valid  = [v["score"] for v in visits_sorted if not np.isnan(v["score"])]
        mean_score    = float(np.mean(scores_valid)) if scores_valid else float("nan")
        ev  = _f(visits_sorted[0]["ev"])
        tte = _f(visits_sorted[0]["tte"])
        bdays_mo = [v["biopsy_day"] / DAY2MO for v in visits_sorted]
        patients.append({
            "patient_id":  pid,
            "visits":      visits_sorted,     # list of {idx, biopsy_day, score, ev, tte}
            "mean_score":  mean_score,
            "ev":          ev,
            "tte":         tte,
            "biopsy_mo":   bdays_mo,
        })
    return patients


def _pat_colors(patients):
    """Assign a unique tab20 color to each patient."""
    try:
        _tab20 = matplotlib.colormaps["tab20"]
    except (KeyError, AttributeError):
        _tab20 = plt.cm.get_cmap("tab20")
    return {p["patient_id"]: _tab20(i % 20) for i, p in enumerate(patients)}


def _draw_risk_hexbin_umap(ax, fig, xy, scores_all, cfg, show_grey=True):
    """Panel A: hexbin background coloured by mean risk, scatter dots on top."""
    cmap_risk = plt.cm.RdBu_r
    norm_risk  = Normalize(vmin=0, vmax=1)
    score_lbl  = "P(ACR+)" if cfg["score_type"] == "cls" else "Risk percentile"

    valid = ~np.isnan(scores_all)
    if valid.sum() > 10:
        hx = ax.hexbin(xy[valid, 0], xy[valid, 1], C=scores_all[valid],
                       reduce_C_function=np.mean, gridsize=25,
                       cmap=cmap_risk, linewidths=0.1, alpha=0.85)
        hx.set_clim(0, 1)
        cb = fig.colorbar(hx, ax=ax, pad=0.02, fraction=0.046, shrink=0.85)
        cb.set_label(score_lbl, fontsize=7)
        cb.ax.tick_params(labelsize=7)

    # Scatter dots on top, coloured by risk
    ax.scatter(xy[valid, 0], xy[valid, 1], c=scores_all[valid],
               cmap=cmap_risk, norm=norm_risk,
               s=5, alpha=0.6, linewidths=0, zorder=3)
    if show_grey and (~valid).any():
        ax.scatter(xy[~valid, 0], xy[~valid, 1], c="lightgrey",
                   s=3, alpha=0.3, linewidths=0)

    _style_ax(ax, f"① Risk score\n(hexbin background + scatter; {score_lbl})")


def _draw_patient_trajectory_umap(ax, fig, xy, patients, pat_colors, cfg,
                                   highlight_pids=None, large=False):
    """Panel B/showcase: each patient drawn in a unique color with connected path."""
    score_lbl = "P(ACR+)" if cfg["score_type"] == "cls" else "Risk percentile"
    lw_base  = 1.8 if large else 0.9
    ms_dot   = 25  if large else 8
    fs_label = 7   if large else 5

    # Grey background dots
    ax.scatter(xy[:, 0], xy[:, 1], c="lightgrey", s=2, alpha=0.25,
               linewidths=0, zorder=1)

    for p in patients:
        pid    = p["patient_id"]
        visits = p["visits"]
        col    = pat_colors.get(pid, "#888888")
        is_hl  = highlight_pids is not None and pid in highlight_pids
        lw     = lw_base * (2.5 if is_hl else 1.0)
        alpha  = 0.95 if is_hl else 0.65
        zbase  = 4 if is_hl else 2

        idxs = [v["idx"] for v in visits]

        # Path segments
        for j in range(len(visits) - 1):
            i0, i1 = idxs[j], idxs[j + 1]
            ax.plot([xy[i0, 0], xy[i1, 0]], [xy[i0, 1], xy[i1, 1]],
                    color=col, lw=lw, alpha=alpha, zorder=zbase)

        # Biopsy dots
        ax.scatter(xy[idxs, 0], xy[idxs, 1], color=col, s=ms_dot,
                   linewidths=0.4, edgecolors="white" if is_hl else "none",
                   alpha=0.9, zorder=zbase + 1)

        # Label: patient ID at last biopsy
        last = idxs[-1]
        ev_sym = "✗" if p["ev"] == 1 else "○"
        ax.annotate(
            f"{pid} {ev_sym}",
            (xy[last, 0], xy[last, 1]),
            fontsize=fs_label, color=col,
            fontweight="bold" if is_hl else "normal",
            xytext=(4, 4), textcoords="offset points",
            zorder=zbase + 2,
        )

    panel_lbl = "② Per-patient trajectories\n(one colour per patient; label at last sample)"
    if highlight_pids:
        panel_lbl = "② Highlighted patient trajectories\n(one colour per patient)"
    _style_ax(ax, panel_lbl, fs=9 if large else 8)


def _draw_temporal_risk_trajectories(ax, patients, cfg):
    """Panel ②: X=months, Y=risk score — one polyline per test patient.

    Patients are coloured by mean risk (RdBu_r). Event marked with ✗,
    censored with |. Lines alpha-blended so overlapping paths still legible.
    """
    cmap_risk = plt.cm.RdBu_r
    norm_risk  = Normalize(vmin=0, vmax=1)
    score_lbl  = "P(ACR+)" if cfg["score_type"] == "cls" else "Risk percentile"
    is_cls     = cfg.get("is_cls", False)

    max_mo = 0.0
    for p in patients:
        if p["biopsy_mo"]:
            max_mo = max(max_mo, max(p["biopsy_mo"]))
        tte_mo = p["tte"] / DAY2MO if not np.isnan(p["tte"]) else float("nan")
        if not np.isnan(tte_mo):
            max_mo = max(max_mo, tte_mo)
    max_mo = max(max_mo, 1.0)

    for p in patients:
        visits   = p["visits"]
        bmo      = p["biopsy_mo"]
        sc_pct   = p["mean_score"]
        ev       = p["ev"]
        tte_d    = p["tte"]
        tte_mo   = tte_d / DAY2MO if not np.isnan(tte_d) else float("nan")

        col   = cmap_risk(norm_risk(sc_pct)) if not np.isnan(sc_pct) else "#aaaaaa"
        alpha = 0.75

        scores_t = [v["score"] for v in visits]
        valid_t  = [(m, s) for m, s in zip(bmo, scores_t)
                    if not np.isnan(s) and not np.isnan(m)]
        if len(valid_t) < 1:
            continue

        xs = [v[0] for v in valid_t]
        ys = [v[1] for v in valid_t]

        ax.plot(xs, ys, color=col, lw=1.2, alpha=alpha, zorder=2)
        ax.scatter(xs, ys, color=col, s=14, alpha=0.9,
                   linewidths=0.3, edgecolors="white", zorder=3)

        # Event or censored marker at last biopsy
        last_x, last_y = xs[-1], ys[-1]
        if not np.isnan(ev):
            if ev == 1:
                ax.scatter([last_x], [last_y], marker="x", s=60,
                           color=HI, linewidths=1.5, zorder=5)
                if not np.isnan(tte_mo) and not is_cls:
                    ax.axvline(tte_mo, color=col, lw=0.5,
                               alpha=0.25, ls="--", zorder=1)
            else:
                ax.scatter([last_x], [last_y], marker="|", s=60,
                           color=LO, linewidths=1.5, zorder=5)

    # Colorbar
    sm = ScalarMappable(cmap=cmap_risk, norm=norm_risk)
    sm.set_array([])
    cb = ax.get_figure().colorbar(sm, ax=ax, pad=0.02,
                                   fraction=0.035, shrink=0.75)
    cb.set_label(score_lbl, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    leg_h = [
        Line2D([0], [0], marker="x", color=HI, markersize=8,
               linewidth=0, label="Event (last sample)"),
        Line2D([0], [0], marker="|", color=LO, markersize=8,
               linewidth=1.5, label="Censored (last sample)"),
    ]
    ax.legend(handles=leg_h, fontsize=8, framealpha=0.85, loc="upper right")

    ax.set_xlim(0, max_mo * 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Months from first sample", fontsize=9)
    ax.set_ylabel(score_lbl, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "② Temporal risk trajectories\n"
        "(each line = one test patient; colour = mean risk; × = event, | = censored)",
        fontsize=9, fontweight="bold", pad=4,
    )


def _draw_trajectory_umap(ax, fig, xy, records, scores_all, cfg, highlight_pids=None,
                           show_arrows=True):
    """Legacy wrapper — now calls _draw_risk_hexbin_umap."""
    _draw_risk_hexbin_umap(ax, fig, xy, scores_all, cfg)


def _draw_swimmer_panel(ax, fig, patients, score_lbl, cfg, large=False):
    """Draw swimmer / timeline plot for a list of patient dicts."""
    cmap_risk = plt.cm.RdBu_r
    norm_risk  = Normalize(vmin=0, vmax=1)
    n_pat      = len(patients)
    bar_h      = 0.6 if large else 0.55
    fs_ann     = 10  if large else max(3.0, min(5.5, 260 / max(n_pat, 1)))
    ms_dot     = 60  if large else 10
    ms_end     = 120 if large else 35

    y_ticks, y_labels = [], []

    for yi, p in enumerate(reversed(patients)):
        y       = float(yi)
        sc_pct  = p["mean_score"]
        ev      = p["ev"]
        tte_d   = p["tte"]
        tte_mo  = tte_d / DAY2MO if not np.isnan(tte_d) else float("nan")
        bmo     = p["biopsy_mo"]
        bar_col = cmap_risk(norm_risk(sc_pct)) if not np.isnan(sc_pct) else "#aaaaaa"

        start_mo = bmo[0] if bmo else 0.0
        end_mo   = bmo[-1] if bmo else (tte_mo if not np.isnan(tte_mo) else 0.0)
        width    = max(end_mo - start_mo, 0.1)

        ax.barh(y, width, left=start_mo, height=bar_h,
                color=bar_col, alpha=0.88, linewidth=0)

        if bmo:
            ax.scatter(bmo, [y] * len(bmo), s=ms_dot,
                       color="white", edgecolors="#333333",
                       linewidths=0.6 if large else 0.5, zorder=4, alpha=0.95)

        if not np.isnan(ev):
            if ev == 1:
                ax.scatter([end_mo], [y], marker="x", s=ms_end,
                           color=HI, linewidths=1.8 if large else 1.4, zorder=5)
            else:
                ax.scatter([end_mo], [y], marker="|", s=ms_end,
                           color=LO, linewidths=1.8 if large else 1.4, zorder=5)

        ev_sym = "✗ event" if ev == 1 else "○ censored"
        ann_parts = []
        if not np.isnan(tte_mo):
            ann_parts.append(f"{ev_sym}  TTE={tte_mo:.0f} mo")
        if large and not np.isnan(sc_pct):
            ann_parts.append(f"  risk={sc_pct:.2f}")
        ann = "  ".join(ann_parts)
        if ann:
            ax.text(end_mo + (1.0 if large else 0.5), y, ann,
                    fontsize=fs_ann, va="center", color="#444444")

        y_ticks.append(y)
        pid_str = str(p["patient_id"])
        y_labels.append(pid_str if large else (pid_str[-10:] if len(pid_str) > 10 else pid_str))

    sm = ScalarMappable(cmap=cmap_risk, norm=norm_risk)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.01,
                      fraction=0.012 if not large else 0.02,
                      shrink=0.55, location="right")
    cb.set_label(score_lbl, fontsize=9 if large else 8)
    cb.ax.tick_params(labelsize=8 if large else 7)

    leg_h = [
        Line2D([0], [0], marker="x", color=HI,
               markersize=8 if large else 7, linewidth=0, label="Event"),
        Line2D([0], [0], marker="|", color=LO,
               markersize=8 if large else 7, linewidth=1.5, label="Censored"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#333333",
               markersize=6 if large else 5, label="Sample visit"),
    ]
    ax.legend(handles=leg_h, fontsize=9 if large else 7.5,
              framealpha=0.85, loc="lower right")

    fs_y = 10 if large else max(3.0, min(7.0, 280 / max(n_pat, 1)))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=fs_y)
    ax.set_xlabel("Months from first sample", fontsize=10 if large else 9)
    ax.set_ylim(-0.8, n_pat - 0.2)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9 if large else 8)
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)

    if not large and n_pat > 10:
        xlim = ax.get_xlim()
        ax.text(xlim[0], n_pat - 1, "High risk ↑", fontsize=7, color=HI,
                va="top", ha="left")
        ax.text(xlim[0], 0, "Low risk ↓", fontsize=7, color=LO,
                va="bottom", ha="left")


def make_trajectories_fig(task_key, cfg, records, xy, out_dir):
    risks_all  = np.array([_f(r["risk"]) for r in records])
    scores_all = _risk_scores(risks_all, cfg["score_type"])
    patients   = _patient_groups(records, scores_all)
    score_lbl  = "P(ACR+)" if cfg["score_type"] == "cls" else "Risk percentile"

    # Separate test vs train+val patients — only test shown as trajectories
    test_pids = {r["patient_id"] for r in records if r.get("split_type", "test") == "test"}
    all_sorted = sorted(
        [p for p in patients if not np.isnan(p["mean_score"])],
        key=lambda p: p["mean_score"], reverse=True,
    )
    test_sorted  = [p for p in all_sorted if p["patient_id"] in test_pids]
    n_test       = len(test_sorted)
    n_all        = len(all_sorted)

    pat_colors = _pat_colors(test_sorted)

    # Figure height driven by test-patient swimmer rows
    fig_h = max(30, min(n_test * 0.65 + 10, 160))
    fig = plt.figure(figsize=(120, fig_h), facecolor=BG)
    fig.suptitle(
        f"Patient Trajectories — {cfg['label']}  "
        f"(N={n_test} test patients shown; {n_all} total, {len(records)} sample visits)",
        fontsize=28, fontweight="bold", y=1.002, color="#1A1018",
    )
    gs = gridspec.GridSpec(
        1, 3, width_ratios=[1.6, 1.6, 2.6], figure=fig,
        wspace=0.14, left=0.04, right=0.97, top=0.97, bottom=0.04,
    )
    ax_risk  = fig.add_subplot(gs[0])
    ax_traj  = fig.add_subplot(gs[1])
    ax_swim  = fig.add_subplot(gs[2])

    # Panel A: risk hexbin UMAP (all biopsies)
    _draw_risk_hexbin_umap(ax_risk, fig, xy, scores_all, cfg)

    # Panel B: per-patient trajectory UMAP — TEST patients only, train as grey bg
    train_idxs = [i for i, r in enumerate(records) if r.get("split_type", "test") != "test"]
    test_idxs  = [i for i, r in enumerate(records) if r.get("split_type", "test") == "test"]

    # Draw test patient trajectories — only those with ≥2 biopsy visits
    try:
        _tab20 = matplotlib.colormaps["tab20"]
    except (KeyError, AttributeError):
        _tab20 = plt.cm.get_cmap("tab20")

    multi_visit = [p for p in test_sorted if len(p["visits"]) >= 2]
    single_visit = [p for p in test_sorted if len(p["visits"]) < 2]

    # Single-visit patients: just a dot, no label
    for pi, p in enumerate(single_visit):
        col  = _tab20((multi_visit.__len__() + pi) % 20)
        idxs = [v["idx"] for v in p["visits"]]
        ax_traj.scatter(xy[idxs, 0], xy[idxs, 1], color=col, s=30,
                        linewidths=0.8, edgecolors="white",
                        alpha=0.80, zorder=3)

    col_idx = 0
    for p in multi_visit:
        pid    = p["patient_id"]
        visits = p["visits"]
        col    = _tab20(col_idx % 20)
        col_idx += 1
        idxs   = [v["idx"] for v in visits]

        # Path segments with direction arrows
        for j in range(len(visits) - 1):
            i0, i1 = idxs[j], idxs[j + 1]
            ax_traj.annotate("", xy=(xy[i1, 0], xy[i1, 1]),
                             xytext=(xy[i0, 0], xy[i0, 1]),
                             arrowprops=dict(arrowstyle="-|>",
                                            color=col, lw=2.2, alpha=0.90,
                                            mutation_scale=12),
                             zorder=4)

        # Biopsy dots
        ax_traj.scatter(xy[idxs, 0], xy[idxs, 1], color=col, s=40,
                        linewidths=0.8, edgecolors="white",
                        alpha=0.95, zorder=5)

        # Label at last biopsy
        last   = idxs[-1]
        ev_sym = "✗" if p["ev"] == 1 else "○"
        ax_traj.annotate(
            f"{pid} {ev_sym}",
            (xy[last, 0], xy[last, 1]),
            fontsize=9, color=col, fontweight="bold",
            xytext=(8, 5), textcoords="offset points",
            zorder=6,
        )

    n_traj = len(multi_visit)
    _style_ax(ax_traj,
              f"② Test patient trajectories  (n={n_traj} with ≥2 samples)\n"
              f"(one colour per patient; →=time direction)",
              fs=11)

    # Panel C: swimmer plot (test patients only, sorted by risk)
    ax_swim.set_title(
        "③ Test patient swimmer plot\n"
        "(sorted by risk: top = high risk  |  × = event  |  | = censored)",
        fontsize=9, fontweight="bold", pad=4,
    )
    _draw_swimmer_panel(ax_swim, fig, test_sorted, score_lbl, cfg, large=False)

    fig.patch.set_facecolor(BG)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"trajectories_{task_key}.{ext}",
                    dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [done] trajectories_{task_key}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3b: 4-patient showcase
# ══════════════════════════════════════════════════════════════════════════════

def make_trajectories_showcase_fig(task_key, cfg, records, xy, out_dir):
    """4-patient showcase: 4 event patients with the longest trajectories (most visits).

    No grey background — only the 4 selected patients shown, each a distinct bold colour.
    """
    SHOWCASE_COLS = ["#952030", "#1A5C8A", "#E07B20", "#2D8A4E"]  # red, blue, amber, green

    risks_all  = np.array([_f(r["risk"]) for r in records])
    scores_all = _risk_scores(risks_all, cfg["score_type"])
    patients   = _patient_groups(records, scores_all)
    score_lbl  = "P(ACR+)" if cfg["score_type"] == "cls" else "Risk percentile"

    all_pts = [p for p in patients if not np.isnan(p["mean_score"])]
    # Prefer patients with events; fallback to any if fewer than 4 events exist
    events = sorted([p for p in all_pts if p["ev"] == 1],
                    key=lambda p: len(p["visits"]), reverse=True)
    if len(events) < 4:
        extra = sorted([p for p in all_pts if p["ev"] != 1],
                       key=lambda p: len(p["visits"]), reverse=True)
        showcase = (events + extra)[:4]
    else:
        showcase = events[:4]

    if len(showcase) < 2:
        print(f"  [skip showcase] insufficient patients for {task_key}")
        return

    showcase_sorted = sorted(showcase, key=lambda p: p["mean_score"], reverse=True)
    showcase_pids   = {p["patient_id"] for p in showcase_sorted}
    pat_colors_sc   = {p["patient_id"]: SHOWCASE_COLS[i]
                       for i, p in enumerate(showcase_sorted)}

    fig = plt.figure(figsize=(60, 22), facecolor=BG)
    fig.suptitle(
        f"Patient Trajectory Showcase — {cfg['label']}\n"
        f"4 event patients with most sample visits  (one colour per patient)",
        fontsize=22, fontweight="bold", y=1.02, color="#1A1018",
    )
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.0], figure=fig,
                           wspace=0.10, left=0.04, right=0.98,
                           top=0.92, bottom=0.10)
    ax_traj = fig.add_subplot(gs[0])
    ax_swim = fig.add_subplot(gs[1])

    # Panel A: risk hexbin UMAP + 4 patient trajectories overlaid (no grey)
    _draw_risk_hexbin_umap(ax_traj, fig, xy, scores_all, cfg, show_grey=False)

    for pi, p in enumerate(showcase_sorted):
        pid    = p["patient_id"]
        visits = p["visits"]
        col    = pat_colors_sc[pid]
        idxs   = [v["idx"] for v in visits]

        for j in range(len(visits) - 1):
            i0, i1 = idxs[j], idxs[j + 1]
            ax_traj.annotate("", xy=(xy[i1, 0], xy[i1, 1]),
                             xytext=(xy[i0, 0], xy[i0, 1]),
                             arrowprops=dict(arrowstyle="-|>",
                                            color=col, lw=5.5, alpha=0.97,
                                            mutation_scale=32),
                             zorder=6)

        ax_traj.scatter(xy[idxs, 0], xy[idxs, 1], color=col, s=380,
                        linewidths=2.0, edgecolors="white",
                        alpha=1.0, zorder=7)

        for rank, idx in enumerate(idxs):
            ax_traj.annotate(
                str(rank + 1),
                (xy[idx, 0], xy[idx, 1]),
                fontsize=13, color="white", fontweight="bold",
                ha="center", va="center", zorder=8,
            )

        last   = idxs[-1]
        ev_sym = "✗" if p["ev"] == 1 else "○"
        ax_traj.annotate(
            f"  {pid} {ev_sym}",
            (xy[last, 0], xy[last, 1]),
            fontsize=16, color=col, fontweight="bold",
            xytext=(12, 8), textcoords="offset points",
            zorder=9,
        )

    ax_traj.set_title(
        "① Risk UMAP + patient trajectories\n"
        "(hexbin = mean risk; arrows = time direction; numbers = visit order)",
        fontsize=16, fontweight="bold", pad=6,
    )
    ax_traj.tick_params(labelsize=13)
    ax_traj.set_xlabel(ax_traj.get_xlabel(), fontsize=14)
    ax_traj.set_ylabel(ax_traj.get_ylabel(), fontsize=14)

    # Panel B: large swimmer for 4 patients
    ax_swim.set_title(
        "② 4-patient swimmer plot\n"
        "(× = event  |  ○ = sample visit  |  bar colour = mean risk)",
        fontsize=16, fontweight="bold", pad=6,
    )
    _draw_swimmer_panel(ax_swim, fig, showcase_sorted, score_lbl, cfg, large=True)

    fig.patch.set_facecolor(BG)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        dpi = 2000 if ext == "png" else 300
        fig.savefig(out_dir / f"trajectories_{task_key}_showcase.{ext}",
                    dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [done] trajectories_{task_key}_showcase")


def make_individual_showcase_figs(task_key, cfg, records, xy, out_dir):
    """One figure per showcase patient: UMAP (left) + swimmer (right)."""
    SHOWCASE_COLS = ["#952030", "#1A5C8A", "#E07B20", "#2D8A4E"]

    risks_all  = np.array([_f(r["risk"]) for r in records])
    scores_all = _risk_scores(risks_all, cfg["score_type"])
    patients   = _patient_groups(records, scores_all)
    score_lbl  = "P(ACR+)" if cfg["score_type"] == "cls" else "Risk percentile"

    all_pts = [p for p in patients if not np.isnan(p["mean_score"])]
    events  = sorted([p for p in all_pts if p["ev"] == 1],
                     key=lambda p: len(p["visits"]), reverse=True)
    if len(events) < 4:
        extra   = sorted([p for p in all_pts if p["ev"] != 1],
                         key=lambda p: len(p["visits"]), reverse=True)
        showcase = (events + extra)[:4]
    else:
        showcase = events[:4]

    if len(showcase) < 1:
        print(f"  [skip individual showcase] no patients for {task_key}")
        return

    showcase_sorted = sorted(showcase, key=lambda p: p["mean_score"], reverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pi, p in enumerate(showcase_sorted):
        col  = SHOWCASE_COLS[pi % len(SHOWCASE_COLS)]
        pid  = p["patient_id"]
        visits = p["visits"]
        idxs   = [v["idx"] for v in visits]

        fig = plt.figure(figsize=(28, 10), facecolor=BG)
        fig.patch.set_facecolor(BG)

        ev_sym = "Event" if p["ev"] == 1 else "Censored"
        tte_mo = p["tte"] / DAY2MO if not np.isnan(p["tte"]) else float("nan")
        tte_str = f"  TTE={tte_mo:.0f} mo" if not np.isnan(tte_mo) else ""
        fig.suptitle(
            f"{cfg['label']} — Patient {pid}  ({ev_sym}{tte_str})",
            fontsize=20, fontweight="bold", color=col, y=1.01,
        )

        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], figure=fig,
                               wspace=0.12, left=0.05, right=0.97,
                               top=0.92, bottom=0.10)
        ax_traj = fig.add_subplot(gs[0])
        ax_swim = fig.add_subplot(gs[1])

        # Left: UMAP with grey background + this patient's trajectory
        _draw_risk_hexbin_umap(ax_traj, fig, xy, scores_all, cfg, show_grey=True)

        for j in range(len(visits) - 1):
            i0, i1 = idxs[j], idxs[j + 1]
            ax_traj.annotate("", xy=(xy[i1, 0], xy[i1, 1]),
                             xytext=(xy[i0, 0], xy[i0, 1]),
                             arrowprops=dict(arrowstyle="-|>",
                                            color=col, lw=5.5, alpha=0.97,
                                            mutation_scale=32),
                             zorder=6)

        ax_traj.scatter(xy[idxs, 0], xy[idxs, 1], color=col, s=380,
                        linewidths=2.0, edgecolors="white", alpha=1.0, zorder=7)

        for rank, idx in enumerate(idxs):
            ax_traj.annotate(str(rank + 1), (xy[idx, 0], xy[idx, 1]),
                             fontsize=13, color="white", fontweight="bold",
                             ha="center", va="center", zorder=8)

        last = idxs[-1]
        ax_traj.annotate(f"  {pid}", (xy[last, 0], xy[last, 1]),
                         fontsize=14, color=col, fontweight="bold",
                         xytext=(12, 8), textcoords="offset points", zorder=9)

        ax_traj.set_title("UMAP trajectory\n(numbers = visit order)",
                          fontsize=14, fontweight="bold", pad=6)
        ax_traj.tick_params(labelsize=12)

        # Right: swimmer for this patient only
        ax_swim.set_title("Timeline", fontsize=14, fontweight="bold", pad=6)
        _draw_swimmer_panel(ax_swim, fig, [p], score_lbl, cfg, large=True)

        for ext in ("png", "pdf"):
            dpi = 2000 if ext == "png" else 300
            fig.savefig(out_dir / f"trajectories_{task_key}_showcase_pat{pi+1}.{ext}",
                        dpi=dpi, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"  [done] trajectories_{task_key}_showcase_pat{pi+1} ({pid})")


# ── Main ───────────────────────────────────────────────────────────────────────

for task_key, base_cfg in _TASK_BASE.items():
    for split in range(5):
        print(f"\n[task={task_key}  split={split}]")
        pt_path = _pt_path(task_key, split)
        if not pt_path.exists():
            print(f"  [skip] pt not found: {pt_path}")
            continue

        cfg = base_cfg
        records = load_biopsy_records(pt_path, cfg["ev_key"], cfg["tte_key"],
                                      is_cls=cfg.get("is_cls", False))
        if len(records) < 10:
            print(f"  [skip] only {len(records)} biopsies")
            continue

        X  = np.stack([r["rep"] for r in records]).astype(np.float32)
        X  = StandardScaler().fit_transform(X)
        print(f"  [umap] N_biopsies={len(records)}, X={X.shape}")
        xy = _umap_embed(X, n_neighbors=min(100, len(records) - 1),
                         min_dist=0.25, metric="cosine")

        out = FIG_ROOT / task_key / f"split{split}"
        make_technical_fig(task_key, cfg, records, xy, out, split=split)
        make_risk_events_fig(task_key, cfg, records, xy, out)
        make_trajectories_fig(task_key, cfg, records, xy, out)
        make_trajectories_showcase_fig(task_key, cfg, records, xy, out)
        make_individual_showcase_figs(task_key, cfg, records, xy, out)

print("\n[all done]")
