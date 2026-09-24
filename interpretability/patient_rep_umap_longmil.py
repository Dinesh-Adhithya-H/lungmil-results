"""
Patient representation UMAP for LongMIL (longitudinal_mk_no_alibi_v2).

One split (split 0). For each task × 2 figure styles:
  scatter: individual biopsy rep dots — train grey, test coloured
  hexbin:  density/mean aggregation — cleaner for large N

Plus trajectory figure: all test patient trajectories on one shared UMAP.

Output: figures/interpretability/patient_rep_umap/
Usage: sbatch interpretability/submit_patient_rep_umap_longmil.sh
"""

import math, sys, types, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mil.models.builders import build_model_v8
from mil.data.splits import build_splits_longitudinal
from mil.data.registry import MODALITIES
from mil.data.loader import preload_bags
from interpretability.shared import SPLITS_CSV, SAMPLES_DIR, RESULTS_ROOT

VARIANT   = "longitudinal_mk_no_alibi_v2"
BASE_ROOT = ROOT / "figures" / "interpretability" / "patient_rep_umap"
BASE_ROOT.mkdir(parents=True, exist_ok=True)
ALL_SPLITS = [0, 1, 2, 3, 4]

TASKS = {
    "cls":       {"label": "ACR Classification", "internal": "acr_cls",  "type": "cls",  "color": "#e53935"},
    "acr_surv":  {"label": "ACR Survival",        "internal": "acr_surv", "type": "surv", "color": "#ff7043"},
    "clad_surv": {"label": "CLAD Survival",       "internal": "clad",     "type": "surv", "color": "#7e57c2"},
    "death_surv":{"label": "Death Survival",      "internal": "death",    "type": "surv", "color": "#26a69a"},
}

BG    = "#F8F4F0"
TRAIN_C = "#C8C0BC"   # muted grey for train context
TEST_HI = "#E87070"   # high-risk red (light)
TEST_LO = "#6EB5E8"   # low-risk blue (light)
EVENT_COL    = "#FF9999"   # trajectory: event patients — light red
CENSORED_COL = "#C0C0C0"  # trajectory: censored patients — light grey
TAB20 = mcm.get_cmap("tab20")
MOD_COLORS = {"HE": "#4C9BE8", "BAL": "#3FB950", "CT": "#D4A017", "Clinical": "#A97ECC"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(task_suffix, device, split):
    ckpt = RESULTS_ROOT / f"split{split}_fold0" / \
           f"{VARIANT}_{task_suffix}" / f"model_{VARIANT}_final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    for k in list(state.keys()):
        if k.endswith(".seeds") and state[k].ndim == 3:
            state[k] = state[k].squeeze(0)
    n_seeds = 8
    for k, v in state.items():
        if k.endswith(".seeds") and v.ndim == 2:
            n_seeds = v.shape[0]; break
    model = build_model_v8(variant=VARIANT, task=task_suffix, slot_k=n_seeds, n_cross_layers=1)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


# ── Rep extraction ────────────────────────────────────────────────────────────

def extract_reps(task_suffix, internal_key, split_set, split_label, device, bag_cache, model):
    orig_cm_cls = type(model)._concat_meta
    anchor_day_store = [None]

    def patched_cm(self, task, tokens, tok_biopsy, tok_mod, days_arr, anchor_day, n_biopsies, dev):
        if task == internal_key:
            anchor_day_store[0] = float(anchor_day)
        return orig_cm_cls(self, task, tokens, tok_biopsy, tok_mod, days_arr, anchor_day, n_biopsies, dev)

    model._concat_meta = types.MethodType(patched_cm, model)

    orig_abmil = type(model)._abmil_rep
    pat_recs = []

    def patched_abmil(self, task, tokens, days, anchor_day):
        out = orig_abmil(self, task, tokens, days, anchor_day)
        if task == internal_key:
            logit = self.heads[task](out).squeeze().item()
            pat_recs.append({
                "anchor_day": anchor_day_store[0] if anchor_day_store[0] is not None else float(anchor_day),
                "rep":   out.detach().cpu().numpy(),
                "logit": logit,
            })
        return out

    model._abmil_rep = types.MethodType(patched_abmil, model)

    all_records = []
    with torch.no_grad():
        for pat in split_set:
            pid       = pat.get("patient_id", "?")
            days_raw  = pat["days"]
            bags_list = [{m: bag_cache.get(s, {}).get(m) for m in MODALITIES}
                         for s in pat["stems"]]
            valid_idx = [i for i, d in enumerate(days_raw)
                         if d is not None and not (isinstance(d, float) and math.isnan(d))]
            if len(valid_idx) < 2:
                continue
            bags_v = [bags_list[i] for i in valid_idx]
            days_v = [float(days_raw[i]) for i in valid_idx]
            recs_v = [pat["records"][i] for i in valid_idx if i < len(pat.get("records", []))]
            if len(recs_v) < len(days_v):
                recs_v += [{}] * (len(days_v) - len(recs_v))
            if all(all(b.get(m) is None for m in MODALITIES) for b in bags_v):
                continue

            ev, tte = float("nan"), float("nan")
            if internal_key == "acr_surv":
                # Scan forward: pick first biopsy with event_next_acr==1 (pre/within episode).
                # Reversed-scan is wrong: post-episode biopsies have tte=nan (not None) so
                # they would be picked last, masking all ACR events.
                for rec in pat.get("records", []):
                    en = rec.get("event_next_acr")
                    tt = rec.get("tte_next_acr")
                    if en == 1 and tt is not None and not (isinstance(tt, float) and math.isnan(tt)):
                        tte = float(tt)
                        ev  = 1.0
                        break
                if math.isnan(ev):
                    # Censored patient: use last non-nan censored biopsy
                    for rec in reversed(pat.get("records", [])):
                        en = rec.get("event_next_acr")
                        tt = rec.get("tte_next_acr")
                        if en == 0 and tt is not None and not (isinstance(tt, float) and math.isnan(tt)):
                            tte = float(tt)
                            ev  = 0.0
                            break
            else:
                for rec in reversed(pat.get("records", [])):
                    if internal_key == "acr_cls":
                        if rec.get("label") is not None:
                            ev = float(rec["label"]); break
                    elif internal_key == "clad":
                        if rec.get("clad_time") is not None:
                            tte = float(rec["clad_time"])
                            ev  = float(rec.get("clad_event", float("nan"))); break
                    elif internal_key == "death":
                        if rec.get("death_time") is not None:
                            tte = float(rec["death_time"])
                            ev  = float(rec.get("death_event", float("nan"))); break

            present_mods = sorted({m for b in bags_v for m in MODALITIES if b.get(m) is not None})
            pat_recs.clear()
            try:
                model({"bags_list": bags_v, "days": days_v, "records": recs_v}, device)
            except Exception as e:
                print(f"    [skip] {pid}: {e}", flush=True)
                continue

            for r in pat_recs:
                all_records.append({
                    "pid":          pid,
                    "anchor_day":   r["anchor_day"],
                    "rep":          r["rep"],
                    "logit":        r["logit"],
                    "ev":           ev,
                    "tte":          tte,
                    "present_mods": present_mods,
                    "split_label":  split_label,
                })

    del model._abmil_rep
    del model._concat_meta
    return all_records


# ── UMAP ─────────────────────────────────────────────────────────────────────

def umap_embed(X):
    from umap import UMAP
    return UMAP(n_neighbors=min(30, len(X) - 1), min_dist=0.15,
                metric="cosine", random_state=42).fit_transform(X)


# ── Figure 1: overview panels — scatter + hexbin twins ───────────────────────

def _prep_arrays(records, tcfg):
    """Shared array computation used by both scatter and hexbin figures."""
    n = len(records)
    sl     = np.array([r["split_label"] for r in records])
    logits = np.array([r["logit"]       for r in records], dtype=float)
    ev_arr = np.array([r["ev"]          for r in records], dtype=float)
    tte_arr= np.array([r["tte"]         for r in records], dtype=float) / 30.44  # → months
    anch   = np.array([r["anchor_day"]  for r in records], dtype=float)
    mods   = ["+".join(r["present_mods"]) for r in records]
    train_m = sl == "train"
    test_m  = sl == "test"

    valid_l = ~np.isnan(logits)
    scores  = np.full(n, float("nan"))
    if tcfg["type"] == "cls":
        scores[valid_l] = 1.0 / (1.0 + np.exp(-logits[valid_l]))
        score_lbl = "P(ACR+)"
    else:
        from scipy.stats import rankdata
        if valid_l.sum() > 1:
            scores[valid_l] = (rankdata(logits[valid_l]) - 1) / max(valid_l.sum() - 1, 1)
        score_lbl = "Risk percentile"

    cc = Counter(mods)
    top_mods = sorted(cc, key=lambda c: -cc[c])[:8]
    mod_col = {m: TAB20(i / 8) for i, m in enumerate(top_mods)}
    mod_col["Other"] = "#aaaaaa"
    disp_mods = [m if m in mod_col else "Other" for m in mods]

    # TTE: events only (ev==1), in months
    p95_tte = float("nan")
    valid_tte = test_m & (ev_arr == 1) & ~np.isnan(tte_arr)
    if valid_tte.sum() > 3:
        p95_tte = np.nanpercentile(tte_arr[valid_tte], 95)

    def _div_range(vals, mask):
        """Return (median, vmin, vmax) for a diverging colormap centered at median."""
        v = vals[mask & ~np.isnan(vals)]
        if len(v) < 2:
            return 0.0, -1.0, 1.0
        med = float(np.median(v))
        dev = float(np.max(np.abs(v - med)))
        dev = max(dev, 1e-6)
        return med, med - dev, med + dev

    # Diverging ranges for each continuous variable (test set)
    score_med, score_vmin, score_vmax = _div_range(scores, test_m)
    ev_med,    ev_vmin,    ev_vmax    = _div_range(ev_arr, test_m)
    tte_med,   tte_vmin,   tte_vmax   = _div_range(tte_arr, valid_tte)
    anch_med,  anch_vmin,  anch_vmax  = _div_range(anch, test_m)

    return dict(n=n, sl=sl, logits=logits, ev_arr=ev_arr, tte_arr=tte_arr,
                anch=anch, mods=mods, train_m=train_m, test_m=test_m,
                scores=scores, score_lbl=score_lbl, mod_col=mod_col,
                disp_mods=disp_mods, p95_tte=p95_tte, valid_tte=valid_tte,
                score_med=score_med, score_vmin=score_vmin, score_vmax=score_vmax,
                ev_med=ev_med, ev_vmin=ev_vmin, ev_vmax=ev_vmax,
                tte_med=tte_med, tte_vmin=tte_vmin, tte_vmax=tte_vmax,
                anch_med=anch_med, anch_vmin=anch_vmin, anch_vmax=anch_vmax)


def _ax_base(ax, title, xy, train_m, BG=BG):
    ax.scatter(xy[train_m, 0], xy[train_m, 1], c=TRAIN_C, s=12,
               alpha=0.22, linewidths=0, zorder=1)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_facecolor(BG)
    ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def _cbar(fig, ax, sc, label):
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.04, shrink=0.8)
    cb.set_label(label, fontsize=8); cb.ax.tick_params(labelsize=7)


def _tsn(vals, mask):
    """TwoSlopeNorm centred at median, vmin=p2, vmax=p98 (asymmetric, no extreme colours)."""
    v = vals[mask & ~np.isnan(vals)]
    if len(v) < 3:
        return None, -1.0, 1.0
    p2, med, p98 = np.percentile(v, [2, 50, 98])
    if p2 >= med: p2 = med - 1e-6
    if p98 <= med: p98 = med + 1e-6
    norm = mcolors.TwoSlopeNorm(vcenter=float(med), vmin=float(p2), vmax=float(p98))
    return norm, float(med), float(p98)


def plot_combined_umap(records, xy, task_suffix, tcfg, split, out_root):
    """8-panel 2×4 combined: scatter (①-⑥) + trajectories (⑦) + avg TTE hexbin (⑧)."""
    import random as _rnd
    d = _prep_arrays(records, tcfg)
    MS = 18; ALPHA = 0.82
    valid_te  = d["test_m"] & ~np.isnan(d["scores"])
    valid_ev  = d["test_m"] & ~np.isnan(d["ev_arr"])
    valid_tte = d["valid_tte"]
    tte_yrs   = d["tte_arr"] / 12.0   # months → years

    # TwoSlopeNorm for each continuous variable — median-centred, p2–p98 range
    norm_score, med_score, _ = _tsn(d["scores"], valid_te)
    norm_tte,   med_tte,   _ = _tsn(tte_yrs,    valid_tte)
    norm_anch,  med_anch,  _ = _tsn(d["anch"],  d["test_m"])

    fig, axes = plt.subplots(2, 4, figsize=(26, 12), facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"LongMIL patient rep UMAP — {tcfg['label']}  split {split}"
        f"  (N={d['n']} biopsy anchors)",
        fontsize=13, fontweight="bold", y=1.00
    )
    axes = axes.flatten()

    # ① train vs test
    _ax_base(axes[0], "① Train vs Test", xy, d["train_m"])
    axes[0].scatter(xy[d["test_m"], 0], xy[d["test_m"], 1],
                    c=TEST_HI, s=MS, alpha=ALPHA, linewidths=0, zorder=2)
    axes[0].legend(handles=[
        mpatches.Patch(color=TRAIN_C, label=f"Train ({d['train_m'].sum()})"),
        mpatches.Patch(color=TEST_HI, label=f"Test ({d['test_m'].sum()})")],
        fontsize=8, framealpha=0.85)

    # ② risk score — RdBu_r, median-centred via TwoSlopeNorm
    _ax_base(axes[1], f"② Risk score (Risk percentile)", xy, d["train_m"])
    kw2 = dict(norm=norm_score) if norm_score is not None else dict(vmin=0, vmax=1)
    sc = axes[1].scatter(xy[valid_te, 0], xy[valid_te, 1], c=d["scores"][valid_te],
                         cmap="RdBu_r", s=MS, alpha=ALPHA, linewidths=0, zorder=2, **kw2)
    _cbar(fig, axes[1], sc, f"Risk percentile (white=median {med_score:.2f})")

    # ③ event — binary categorical scatter (red=event, blue=censored)
    _ax_base(axes[2], "③ Event (red=1, blue=0)", xy, d["train_m"])
    ev0 = valid_ev & (d["ev_arr"] == 0)
    ev1 = valid_ev & (d["ev_arr"] == 1)
    if ev0.sum() > 0:
        axes[2].scatter(xy[ev0, 0], xy[ev0, 1], c=TEST_LO, s=MS, alpha=ALPHA,
                        linewidths=0, zorder=2, label=f"Censored ({ev0.sum()})")
    if ev1.sum() > 0:
        axes[2].scatter(xy[ev1, 0], xy[ev1, 1], c=TEST_HI, s=MS, alpha=ALPHA,
                        linewidths=0, zorder=3, label=f"Event ({ev1.sum()})")
    axes[2].legend(fontsize=8, framealpha=0.85)

    # ④ TTE (years) — viridis_r, median-centred, non-events faded
    _ax_base(axes[3], f"④ TTE (years, bright=short)", xy, d["train_m"])
    non_ev_te = d["test_m"] & ~valid_tte
    if non_ev_te.sum() > 0:
        axes[3].scatter(xy[non_ev_te, 0], xy[non_ev_te, 1],
                        c="#cccccc", s=MS * 0.7, alpha=0.25, linewidths=0, zorder=2)
    if valid_tte.sum() >= 1:
        kw4 = dict(norm=norm_tte) if norm_tte is not None else dict(vmin=0, vmax=1)
        sc = axes[3].scatter(xy[valid_tte, 0], xy[valid_tte, 1],
                             c=tte_yrs[valid_tte], cmap="RdBu_r",
                             s=MS, alpha=ALPHA, linewidths=0, zorder=3, **kw4)
        _cbar(fig, axes[3], sc, f"TTE yrs (n={valid_tte.sum()}, white=median {med_tte:.1f})")
        axes[3].set_title(f"④ TTE (years, red=short, blue=far)", fontsize=11, fontweight="bold", pad=6)
    else:
        axes[3].text(0.5, 0.5, "no events in test set", ha="center", va="center",
                     transform=axes[3].transAxes, fontsize=9, color="#888")

    # ⑤ modality combo
    _ax_base(axes[4], "⑤ Modality combination", xy, d["train_m"])
    for combo, col in d["mod_col"].items():
        idx = [i for i, m in enumerate(d["disp_mods"]) if m == combo and d["test_m"][i]]
        if not idx: continue
        axes[4].scatter(xy[idx, 0], xy[idx, 1], color=col, s=MS, alpha=ALPHA,
                        linewidths=0, zorder=2, label=combo[:28])
    axes[4].legend(fontsize=6, framealpha=0.8, loc="lower right")

    # ⑥ anchor day — viridis, median-centred
    _ax_base(axes[5], "⑥ Anchor day (days post-Tx)", xy, d["train_m"])
    kw6 = dict(norm=norm_anch) if norm_anch is not None else dict(vmin=0, vmax=3000)
    sc = axes[5].scatter(xy[d["test_m"], 0], xy[d["test_m"], 1],
                         c=d["anch"][d["test_m"]], cmap="viridis",
                         s=MS, alpha=ALPHA, linewidths=0, zorder=2, **kw6)
    _cbar(fig, axes[5], sc, f"Days post-Tx (white=median {med_anch:.0f}d)")

    # ⑦ test patient trajectories (up to 60 patients for legibility)
    ax7 = axes[6]
    ax7.set_facecolor(BG); ax7.spines[["top","right"]].set_visible(False)
    ax7.tick_params(labelsize=7)
    ax7.set_xlabel("UMAP-1", fontsize=7); ax7.set_ylabel("UMAP-2", fontsize=7)
    ax7.scatter(xy[d["train_m"], 0], xy[d["train_m"], 1],
                c=TRAIN_C, s=6, alpha=0.18, linewidths=0, zorder=1)

    sl_arr = np.array([r["split_label"] for r in records])
    pids_all = [r["pid"] for r in records]
    anch_all = np.array([r["anchor_day"] for r in records], dtype=float)
    ev_all   = np.array([r["ev"]         for r in records], dtype=float)

    test_pids = sorted(set(p for p, s in zip(pids_all, sl_arr) if s == "test"))
    _rnd.seed(42)
    show_pids = _rnd.sample(test_pids, min(60, len(test_pids)))
    n_ev_traj = sum(
        1 for pid in show_pids
        for j in [next((j for j, (p, s) in enumerate(zip(pids_all, sl_arr))
                        if p == pid and s == "test"), None)]
        if j is not None and not math.isnan(ev_all[j]) and ev_all[j] == 1
    )

    for pi, pid in enumerate(show_pids):
        idx = [j for j, (p, s) in enumerate(zip(pids_all, sl_arr)) if p == pid and s == "test"]
        if not idx: continue
        idx_s = sorted(idx, key=lambda j: anch_all[j])
        pts = xy[idx_s]; T = len(idx_s)
        ev_val = ev_all[idx_s[0]]
        is_ev  = not math.isnan(ev_val) and ev_val == 1
        col = TAB20(pi / max(len(show_pids), 1))
        ls  = "-" if is_ev else "--"
        lw  = 1.6 if is_ev else 0.8
        alp = 0.85 if is_ev else 0.4
        if T >= 2:
            for k in range(T - 1):
                ax7.plot(pts[k:k+2, 0], pts[k:k+2, 1], ls=ls,
                         color=col, lw=lw, alpha=alp, zorder=3)
        sizes = np.linspace(10, 35, T) if is_ev else np.linspace(6, 20, T)
        ax7.scatter(pts[:, 0], pts[:, 1], s=sizes, c=[col]*T,
                    alpha=0.9 if is_ev else 0.5, linewidths=0, zorder=4)
        ax7.scatter(pts[0, 0], pts[0, 1], marker="^", s=20 if is_ev else 10,
                    c=[col], alpha=1.0, linewidths=0, zorder=5)
        if is_ev:
            ax7.scatter(pts[-1, 0], pts[-1, 1], marker="*", s=120,
                        c=[EVENT_COL], linewidths=1.2, edgecolors=TEST_HI, zorder=6)
        else:
            ax7.scatter(pts[-1, 0], pts[-1, 1], marker="o", s=20,
                        c=[CENSORED_COL], linewidths=0.6, edgecolors="white", alpha=0.7, zorder=6)

    ax7.set_title(f"⑦ Test patient trajectories (n={len(show_pids)} pts)",
                  fontsize=10, fontweight="bold", pad=6)
    ax7.legend(handles=[
        plt.Line2D([0],[0], color="#888", lw=1.6, ls="-",  label=f"Event (solid)"),
        plt.Line2D([0],[0], color="#888", lw=0.8, ls="--", label="Censored (dashed)"),
    ], fontsize=7, framealpha=0.85, loc="lower right")

    # ⑧ avg TTE hexbin — events only
    ax8 = axes[7]
    ax8.set_facecolor(BG); ax8.spines[["top","right"]].set_visible(False)
    ax8.tick_params(labelsize=7)
    ax8.set_xlabel("UMAP-1", fontsize=7); ax8.set_ylabel("UMAP-2", fontsize=7)
    ax8.set_title(f"⑧ Avg TTE — events only\n(n={valid_tte.sum()}, bright=short TTE)",
                  fontsize=10, fontweight="bold", pad=6)
    if valid_tte.sum() >= 6:
        kw8 = dict(norm=norm_tte) if norm_tte is not None else {}
        hb = ax8.hexbin(xy[valid_tte, 0], xy[valid_tte, 1],
                        C=tte_yrs[valid_tte], reduce_C_function=np.mean,
                        gridsize=22, cmap="RdBu_r", linewidths=0.2, mincnt=1, **kw8)
        cb = fig.colorbar(hb, ax=ax8, pad=0.02, fraction=0.04, shrink=0.8)
        cb.set_label(f"Mean TTE yrs (white=median {med_tte:.1f})", fontsize=7)
        cb.ax.tick_params(labelsize=7)
    else:
        ax8.text(0.5, 0.5, "too few events for hexbin", ha="center", va="center",
                 transform=ax8.transAxes, fontsize=9, color="#888")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_root.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_root / f"split{split}_{task_suffix}_umap.{ext}",
                    dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    {task_suffix}/split{split}_{task_suffix}_umap", flush=True)


def plot_hexbin_overview(records, xy, task_suffix, tcfg, split, out_root):
    d = _prep_arrays(records, tcfg)
    GS = 28   # hexbin gridsize
    valid_te  = d["test_m"] & ~np.isnan(d["scores"])
    valid_ev  = d["test_m"] & ~np.isnan(d["ev_arr"])
    valid_tte = d["valid_tte"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Patient Biopsy Representations — {tcfg['label']}  ·  Split {split}  [hexbin]\n"
                 f"N={d['n']}: {d['train_m'].sum()} train (grey) + {d['test_m'].sum()} test (coloured)",
                 fontsize=13, fontweight="bold", y=0.99)
    axes = axes.flatten()

    def hx(ax, mask, C, cmap, label, vmin=None, vmax=None):
        if mask.sum() < 6:
            ax.text(0.5, 0.5, "too few points", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9); return
        kw = dict(gridsize=GS, cmap=cmap, linewidths=0.2, mincnt=1)
        if vmin is not None: kw["vmin"] = vmin
        if vmax is not None: kw["vmax"] = vmax
        hb = ax.hexbin(xy[mask, 0], xy[mask, 1], C=C[mask] if C is not None else None,
                       reduce_C_function=np.mean if C is not None else None, **kw)
        cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.04, shrink=0.8)
        cb.set_label(label, fontsize=8); cb.ax.tick_params(labelsize=7)

    def base(ax, title):
        # Train density in grey
        if d["train_m"].sum() > 5:
            ax.hexbin(xy[d["train_m"], 0], xy[d["train_m"], 1],
                      gridsize=GS, cmap="Greys", linewidths=0, mincnt=1, alpha=0.3, zorder=1)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.set_facecolor(BG)
        ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    # ① density: train vs test overlay
    base(axes[0], "① Test density (train in grey)")
    if d["test_m"].sum() > 5:
        hb = axes[0].hexbin(xy[d["test_m"], 0], xy[d["test_m"], 1],
                            gridsize=GS, cmap="Reds", linewidths=0, mincnt=1, alpha=0.85, zorder=2)
        cb = fig.colorbar(hb, ax=axes[0], pad=0.02, fraction=0.04, shrink=0.8)
        cb.set_label("Test count", fontsize=8); cb.ax.tick_params(labelsize=7)

    # ② risk score — diverging around median
    base(axes[1], f"② Mean risk — {d['score_lbl']} (÷median)")
    hx(axes[1], valid_te, d["scores"], "RdBu_r", f"{d['score_lbl']} (white=median {d['score_med']:.2f})",
       vmin=d["score_vmin"], vmax=d["score_vmax"])

    # ③ event density — binary categorical (blue=no event, red=event)
    base(axes[2], "③ Event density (red=event, blue=no event)")
    ev0 = valid_ev & (d["ev_arr"] == 0)
    ev1 = valid_ev & (d["ev_arr"] == 1)
    if ev0.sum() > 5:
        axes[2].hexbin(xy[ev0, 0], xy[ev0, 1], gridsize=GS, cmap="Blues",
                       linewidths=0.2, mincnt=1, alpha=0.55, zorder=2)
    if ev1.sum() >= 1:
        axes[2].hexbin(xy[ev1, 0], xy[ev1, 1], gridsize=GS, cmap="Reds",
                       linewidths=0.2, mincnt=1, alpha=0.85, zorder=3)
    axes[2].text(0.02, 0.98, f"No event: {ev0.sum()}  |  Event: {ev1.sum()}",
                 transform=axes[2].transAxes, fontsize=8, va="top", color="#333")

    # ④ mean TTE — events only, months, diverging (red=short)
    base(axes[3], f"④ Mean TTE — events only ({valid_tte.sum()}), months (red=short)")
    if valid_tte.sum() >= 1:
        hx(axes[3], valid_tte, d["tte_arr"], "RdBu",
           f"Mean TTE months (white=median {d['tte_med']:.1f}, red=short)",
           vmin=d["tte_vmin"], vmax=d["tte_vmax"])
    else:
        axes[3].text(0.5, 0.5, "no events in test set", ha="center", va="center",
                     transform=axes[3].transAxes, fontsize=9, color="#888")

    # ⑤ density per modality (separate overlays)
    base(axes[4], "⑤ Test density — top modalities")
    mod_cmaps = ["Blues", "Greens", "Oranges", "Purples", "Reds"]
    for mi, (combo, col) in enumerate(list(d["mod_col"].items())[:5]):
        idx = np.array([i for i, m in enumerate(d["disp_mods"]) if m == combo and d["test_m"][i]])
        if len(idx) < 3: continue
        axes[4].hexbin(xy[idx, 0], xy[idx, 1], gridsize=GS, cmap=mod_cmaps[mi % len(mod_cmaps)],
                       linewidths=0, mincnt=1, alpha=0.55, zorder=2+mi)
    axes[4].set_facecolor(BG); axes[4].set_xlabel("UMAP 1", fontsize=8)
    axes[4].set_ylabel("UMAP 2", fontsize=8); axes[4].tick_params(labelsize=7)
    axes[4].spines[["top","right"]].set_visible(False)

    # ⑥ anchor day — diverging around median (clamp vmin≥0)
    base(axes[5], "⑥ Anchor day — days post-Tx (÷median)")
    anch_vmin_clamp = max(0, d["anch_vmin"])
    hx(axes[5], d["test_m"], d["anch"], "RdBu_r",
       f"Days post-Tx (white≈median {d['anch_med']:.0f}d)",
       vmin=anch_vmin_clamp, vmax=d["anch_vmax"])

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(out_root / f"{task_suffix}.{ext}",
                    dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    hexbin/{task_suffix}", flush=True)


def plot_overview(records, xy, task_suffix, tcfg, split, scatter_root, hexbin_root):
    # Combined 8-panel figure saved to per-task subdir
    task_root = BASE_ROOT / task_suffix
    plot_combined_umap(records, xy, task_suffix, tcfg, split, task_root)
    # Keep hexbin for reference
    plot_hexbin_overview(records, xy, task_suffix, tcfg, split, hexbin_root)


# ── Figure 2: all test trajectories on one shared UMAP ───────────────────────

def plot_trajectories_shared(records, xy, task_suffix, tcfg, split, out_root):
    """
    All test patient trajectories on one shared UMAP.
    - Each patient has a unique tab20 color (rainbow, like the overview)
    - EVENT patients: solid lines, thick, red star endpoint (✗ label)
    - CENSORED patients: dashed lines, lighter alpha, grey circle endpoint
    - Dots grow in size from first → last biopsy (chronological)
    """
    sl     = np.array([r["split_label"] for r in records])
    pids   = [r["pid"]        for r in records]
    anch   = np.array([r["anchor_day"]  for r in records], dtype=float)
    ev_arr = np.array([r["ev"]          for r in records], dtype=float)
    tte_arr= np.array([r["tte"]         for r in records], dtype=float) / 30.44  # → months

    train_m   = sl == "train"
    test_pids = sorted(set(p for p, s in zip(pids, sl) if s == "test"))
    n_pats    = len(test_pids)
    pid_to_idx = {pid: i for i, pid in enumerate(test_pids)}

    # Separate event vs censored patients for legend
    event_pids    = []
    censored_pids = []
    for pid in test_pids:
        idx = [j for j, (p, s) in enumerate(zip(pids, sl)) if p == pid and s == "test"]
        ev  = ev_arr[idx[0]] if idx else float("nan")
        (event_pids if not math.isnan(ev) and ev == 1 else censored_pids).append(pid)

    fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_xlabel("UMAP 1", fontsize=9); ax.set_ylabel("UMAP 2", fontsize=9)
    ax.tick_params(labelsize=8)

    # Train background
    ax.scatter(xy[train_m, 0], xy[train_m, 1], c=TRAIN_C, s=8,
               alpha=0.18, linewidths=0, zorder=1)

    for pid in test_pids:
        idx = [j for j, (p, s) in enumerate(zip(pids, sl)) if p == pid and s == "test"]
        if not idx:
            continue
        idx_sorted = sorted(idx, key=lambda j: anch[j])
        pts  = xy[idx_sorted]
        T    = len(idx_sorted)
        ev_val  = ev_arr[idx_sorted[0]]
        tte_val = tte_arr[idx_sorted[0]]

        is_event = not math.isnan(ev_val) and ev_val == 1
        # Unique per-patient color (tab20 palette)
        col     = TAB20(pid_to_idx[pid] / max(n_pats, 1))
        ls      = "-"   if is_event else "--"
        lw      = 2.0   if is_event else 1.0
        alpha_l = 0.85  if is_event else 0.45

        # Connecting lines
        if T >= 2:
            for k in range(T - 1):
                ax.plot(pts[k:k+2, 0], pts[k:k+2, 1], ls=ls,
                        color=col, lw=lw, alpha=alpha_l, zorder=3, solid_capstyle="round")
                # Directional arrow at midpoint
                mid  = (pts[k] + pts[k+1]) / 2
                dp   = pts[k+1] - pts[k]
                dp_n = dp / (np.linalg.norm(dp) + 1e-8)
                ax.annotate("", xy=mid + dp_n * 0.12, xytext=mid - dp_n * 0.12,
                            arrowprops=dict(arrowstyle="-|>", color=col,
                                            lw=lw * 0.8, mutation_scale=9),
                            zorder=4)

        # Dots: growing size first → last
        sizes   = np.linspace(15, 50, T) if is_event else np.linspace(10, 30, T)
        alpha_d = 0.9 if is_event else 0.55
        ax.scatter(pts[:, 0], pts[:, 1], s=sizes, c=[col] * T,
                   alpha=alpha_d, linewidths=0.4, edgecolors="white", zorder=5)

        # First biopsy: triangle up
        ax.scatter(pts[0, 0], pts[0, 1], marker="^", s=40 if is_event else 20,
                   c=[col], alpha=1.0, linewidths=0.8, edgecolors="white", zorder=6)

        # Last biopsy: red star (event) or grey circle (censored)
        if is_event:
            ax.scatter(pts[-1, 0], pts[-1, 1], marker="*", s=200,
                       c=[EVENT_COL], linewidths=1.5, edgecolors=TEST_HI, zorder=7)
        else:
            ax.scatter(pts[-1, 0], pts[-1, 1], marker="o", s=40,
                       c=[CENSORED_COL], linewidths=1.0, edgecolors="white", alpha=0.8, zorder=7)

        # Endpoint label: pid + outcome
        tte_str = f"{tte_val:.0f}m" if not math.isnan(tte_val) else "?"
        outcome = f"✗ {tte_str}" if is_event else "○cens"
        label_col = TEST_HI if is_event else "#888"
        ax.text(pts[-1, 0] + 0.15, pts[-1, 1] + 0.15,
                f"{pid}\n{outcome}", fontsize=5.5, color=label_col,
                va="bottom", ha="left", zorder=8,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.55, ec="none"))

    n_ev   = len(event_pids)
    n_cens = len(censored_pids)
    ax.set_title(
        f"Test Patient Trajectories — {tcfg['label']}  ·  Split {split}\n"
        f"{n_pats} patients  ·  {n_ev} events (solid line, ★ red endpoint)  ·  "
        f"{n_cens} censored (dashed, ○ grey endpoint)\n"
        f"▲ = first biopsy  ·  dot size grows with time  ·  each patient = unique colour",
        fontsize=11, fontweight="bold", pad=10,
    )

    legend_els = [
        mpatches.Patch(color=TRAIN_C, alpha=0.5, label="Train biopsies (context)"),
        plt.Line2D([0], [0], color="#888", lw=2.0, ls="-",
                   label=f"Event patient — solid line (n={n_ev})"),
        plt.Line2D([0], [0], color="#888", lw=1.0, ls="--",
                   label=f"Censored patient — dashed (n={n_cens})"),
        plt.scatter([], [], marker="*", s=120, c=[EVENT_COL], edgecolors=TEST_HI,
                    linewidths=1.5, label="Last biopsy — event ★"),
        plt.scatter([], [], marker="o", s=40, c=[CENSORED_COL],
                    edgecolors="white", label="Last biopsy — censored ○"),
        plt.scatter([], [], marker="^", s=40, c="grey",
                    edgecolors="white", label="First biopsy ▲"),
    ]
    ax.legend(handles=legend_els, fontsize=8, framealpha=0.88,
              loc="lower right", markerscale=1.0)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_root / f"{task_suffix}.{ext}",
                    dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    trajectories/{task_suffix}", flush=True)


# ── Figure 3: 4-task combined trajectory overview (2×2) ──────────────────────

def plot_4task_overview(task_data, split, out_root):
    """2×2: one trajectory panel per task, all on same style."""
    fig, axes = plt.subplots(2, 2, figsize=(22, 16), facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Patient Biopsy Trajectories — All Tasks  ·  Split {split}",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for ax, (task_suffix, (records, xy, tcfg)) in zip(axes, task_data.items()):
        sl    = np.array([r["split_label"] for r in records])
        pids  = [r["pid"]       for r in records]
        anch  = np.array([r["anchor_day"]  for r in records], dtype=float)
        ev_arr= np.array([r["ev"]          for r in records], dtype=float)

        train_m = sl == "train"
        test_pids = sorted(set(p for p, s in zip(pids, sl) if s == "test"))

        ax.set_facecolor(BG)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("UMAP 1", fontsize=7)
        ax.set_ylabel("UMAP 2", fontsize=7)

        ax.scatter(xy[train_m, 0], xy[train_m, 1], c=TRAIN_C, s=8,
                   alpha=0.18, linewidths=0, zorder=1)

        for pi, pid in enumerate(test_pids):
            idx = [j for j, (p, s) in enumerate(zip(pids, sl)) if p == pid and s == "test"]
            if not idx:
                continue
            idx_sorted = sorted(idx, key=lambda j: anch[j])
            pts  = xy[idx_sorted]
            days = anch[idx_sorted]
            T    = len(idx_sorted)
            col  = TAB20(pi / max(len(test_pids), 1))
            if T >= 2:
                for k in range(T - 1):
                    ax.plot(pts[k:k+2, 0], pts[k:k+2, 1], "-",
                            color=col, lw=1.2, alpha=0.65, zorder=3)
            ax.scatter(pts[:, 0], pts[:, 1], s=np.linspace(12, 36, T),
                       c=[col] * T, alpha=0.9, linewidths=0, zorder=5)
            ev_val = ev_arr[idx_sorted[-1]] if idx_sorted else float("nan")
            end_ec = TEST_HI if not math.isnan(ev_val) and ev_val == 1 else "white"
            ax.scatter(pts[-1, 0], pts[-1, 1], marker="*", s=60, c=[col],
                       linewidths=0.8, edgecolors=end_ec, zorder=6)

        ax.set_title(f"{tcfg['label']}\n({len(test_pids)} test patients)",
                     fontsize=10, fontweight="bold", color=tcfg["color"])

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(out_root / f"all_tasks_trajectories.{ext}",
                    dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    all_tasks_trajectories", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Preload all bags once across all splits (same 4210 files)
    print("Preloading all bags (shared across splits)...", flush=True)
    all_stems = set()
    for s in ALL_SPLITS:
        sp = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=s)
        for pats in sp.values():
            for pat in pats:
                all_stems.update(pat["stems"])
    bag_cache = preload_bags(list(all_stems), SAMPLES_DIR, n_workers=4)
    print(f"  {len(bag_cache)} stems loaded", flush=True)

    for split in ALL_SPLITS:
        print(f"\n{'='*60}", flush=True)
        print(f"  SPLIT {split}", flush=True)
        print(f"{'='*60}", flush=True)

        # Per-split output dirs
        split_root   = BASE_ROOT / f"split{split}"
        scatter_root = split_root / "scatter"
        hexbin_root  = split_root / "hexbin"
        traj_root    = split_root / "trajectories"
        for d in (scatter_root, hexbin_root, traj_root):
            d.mkdir(parents=True, exist_ok=True)

        splits = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=split)

        task_data = {}
        for task_suffix, tcfg in TASKS.items():
            internal = tcfg["internal"]
            print(f"\n  [{tcfg['label']}]", flush=True)
            try:
                model = load_model(task_suffix, device, split)
            except FileNotFoundError as e:
                print(f"    [skip] {e}", flush=True)
                continue

            train_recs = extract_reps(task_suffix, internal,
                                      splits["train"] + splits.get("val", []),
                                      "train", device, bag_cache, model)
            test_recs  = extract_reps(task_suffix, internal,
                                      splits["test"], "test", device, bag_cache, model)
            all_recs   = train_recs + test_recs
            print(f"    train: {len(train_recs)}  test: {len(test_recs)}", flush=True)

            if len(all_recs) < 20:
                print("    [skip] too few records", flush=True)
                del model; continue

            X = np.stack([r["rep"] for r in all_recs]).astype(np.float32)
            X = StandardScaler().fit_transform(X)
            print(f"    UMAP {X.shape}...", flush=True)
            xy = umap_embed(X)

            plot_overview(all_recs, xy, task_suffix, tcfg, split, scatter_root, hexbin_root)
            plot_trajectories_shared(all_recs, xy, task_suffix, tcfg, split, traj_root)
            task_data[task_suffix] = (all_recs, xy, tcfg)

            del model; torch.cuda.empty_cache()

        if len(task_data) == 4:
            print(f"\n  Combined 4-task figure...", flush=True)
            plot_4task_overview(task_data, split, split_root)

    print(f"\nDone. → {BASE_ROOT}", flush=True)


if __name__ == "__main__":
    main()
