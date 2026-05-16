#!/usr/bin/env python3
"""
analyze_v7_endpoints_full.py — Combined analysis for all 4 tasks:
  ACR-cls, ACR-surv, ACR-alt, CLAD, Death.

Tasks:
  benchmark_combined  All experiments in one figure (JSON metrics only, no GPU)
  umap_clad           UMAP panels for CLAD survival model (needs GPU + inference)
  umap_death          UMAP panels for Death survival model (needs GPU + inference)
  combo_perf_clad     Per-modality-combo C-index for CLAD (needs GPU)
  combo_perf_death    Per-modality-combo C-index for Death (needs GPU)

Inference helpers are adapted from analyze_v7_full.py with endpoint awareness.
"""

import argparse
import importlib.util
import json
import math
import pickle
import re as _re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

try:
    from umap import UMAP as UMAPTransform
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[warn] umap-learn not installed; umap tasks will be skipped.")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VARIANT_TAGS = [
    "early", "late", "middle",
    "crossattn_k8", "crossmodal_k8", "iterative_r2_k8",
]
VARIANT_DISPLAY = {
    "early":            "Early",
    "late":             "Late",
    "middle":           "Middle",
    "crossattn_k8":     "Cross-Attn",
    "crossmodal_k8":    "Cross-Modal",
    "iterative_r2_k8":  "Iterative",
}

# All experiments: key → (results_dir_suffix, primary_metric, display_label, color)
EXPERIMENTS = {
    "acr_cls":  ("results_mm_abmil_v7_cls_p1v3",  "bacc",    "ACR Cls",  "#1565C0"),
    "acr_surv": ("results_mm_abmil_v7_surv_p1v3", "c_index", "ACR Surv", "#1976D2"),
    "acr_alt":  ("results_mm_abmil_v7_alt_p1v3",  "c_index", "ACR Alt",  "#42A5F5"),
    "clad":     ("results_mm_abmil_v7_clad",       "c_index", "CLAD",     "#FB8500"),
    "death":    ("results_mm_abmil_v7_death",       "c_index", "Death",    "#C62828"),
}

ENDPOINT_CFG = {
    "clad":  {"time_col": "clad_days",  "event_col": "clad_status",
              "tte_key": "clad_time",   "ev_key": "clad_event",
              "label": "CLAD", "color": "#FB8500"},
    "death": {"time_col": "death_days", "event_col": "death_status",
              "tte_key": "death_time",  "ev_key": "death_event",
              "label": "Death", "color": "#C62828"},
}

COMBO_COLORS = {
    "Clin":           "#546E7A",
    "CT":             "#1565C0",
    "CT+Clin":        "#0B6E4F",
    "HE":             "#7B2D8B",
    "HE+Clin":        "#C2185B",
    "HE+CT":          "#E65100",
    "HE+CT+Clin":     "#E63946",
    "BAL+Clin":       "#00838F",
    "BAL+CT":         "#558B2F",
    "BAL+CT+Clin":    "#6A1B9A",
    "HE+BAL":         "#AD1457",
    "HE+BAL+Clin":    "#F9A825",
    "HE+BAL+CT":      "#4527A0",
    "HE+BAL+CT+Clin": "#BF360C",
    "BAL":            "#2E7D32",
}
DEFAULT_COMBO_COLOR = "#9E9E9E"

NATURE_RC = {
    "font.family":       "sans-serif",
    "font.size":         8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.linewidth":    0.5,
    "grid.alpha":        0.4,
    "grid.linestyle":    "--",
    "grid.color":        "#AAAAAA",
    "figure.dpi":        150,
    "savefig.dpi":       300,
}
ONE_COL_W = 3.5
TWO_COL_W = 7.2

# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _combo_from_row(row) -> str:
    parts = []
    for mod, col in [("HE","has_HE"),("BAL","has_BAL"),("CT","has_CT"),("Clin","has_Clinical")]:
        v = row.get(col, False)
        if v is True or str(v).lower() in ("true","1","1.0"):
            parts.append(mod)
    return "+".join(parts) if parts else "Unknown"

def _n_mods(combo: str) -> int:
    return len(combo.split("+")) if combo and combo != "Unknown" else 0

def _save_fig(fig, path_stem: Path):
    fig.savefig(str(path_stem) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(path_stem) + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def _pct(arr, lo=1, hi=99):
    a = np.asarray(arr, dtype=float)
    v = a[~np.isnan(a)]
    if len(v) == 0:
        return 0.0, 1.0
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))

def _scatter(ax, xy, values, title, cmap, vmin=None, vmax=None, s=12, alpha=0.7):
    values = np.asarray(values, dtype=float)
    mask   = ~np.isnan(values)
    if mask.sum() == 0:
        ax.set_title(title, fontsize=7); return
    vm = vmin if vmin is not None else float(np.nanmin(values))
    vx = vmax if vmax is not None else float(np.nanmax(values))
    sc = ax.scatter(xy[mask,0], xy[mask,1], c=values[mask], cmap=cmap,
                    vmin=vm, vmax=vx, s=s, alpha=alpha, linewidths=0, rasterized=True)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=7, fontweight="bold")

def _hexbin(ax, xy, values, title, cmap, vmin, vmax, cbar_label=""):
    values = np.asarray(values, dtype=float)
    mask   = ~np.isnan(values)
    if mask.sum() < 4:
        ax.set_title(title, fontsize=7); return
    hb = ax.hexbin(xy[mask,0], xy[mask,1], C=values[mask], gridsize=25,
                   reduce_C_function=np.nanmean, cmap=cmap,
                   vmin=vmin, vmax=vmax, mincnt=1, linewidths=0.2)
    cb = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=6)
    ax.set_title(title, fontsize=7, fontweight="bold")

def _safe_cindex(tte, event, hazard):
    tte   = np.asarray(tte,   dtype=float)
    ev    = np.asarray(event, dtype=float)
    hz    = np.asarray(hazard,dtype=float)
    valid = ~np.isnan(tte) & ~np.isnan(ev) & ~np.isnan(hz) & (tte >= 0)
    tte, ev, hz = tte[valid], ev[valid], hz[valid]
    if len(tte) < 2 or ev.sum() == 0:
        return float("nan")
    concordant = discordant = 0
    for i in range(len(tte)):
        if ev[i] == 0:
            continue
        for j in range(len(tte)):
            if i == j: continue
            if tte[i] < tte[j]:
                concordant += hz[i] > hz[j]
                discordant += hz[i] < hz[j]
    total = concordant + discordant
    return float(concordant / total) if total > 0 else float("nan")

# ─────────────────────────────────────────────────────────────────────────────
# Metric loading from JSON
# ─────────────────────────────────────────────────────────────────────────────

def _load_metrics_dir(results_dir: Path, folds: List[int]) -> Dict:
    """Load metrics_*.json from results_dir/split0_fold{f}/. Returns data[tag][fold]=test_dict."""
    data: Dict = {}
    for f in folds:
        fold_dir = results_dir / f"split0_fold{f}"
        if not fold_dir.exists():
            continue
        for mfile in sorted(fold_dir.glob("metrics_*.json")):
            tag = mfile.stem.replace("metrics_", "")
            try:
                m = json.loads(mfile.read_text())
                data.setdefault(tag, {})[f] = m.get("test", {})
            except Exception as e:
                print(f"[load] {mfile}: {e}")
    return data

def _fold_stats(data: Dict, tag: str, metric: str) -> Tuple[float, float]:
    vals = [float(fd[metric]) for fd in data.get(tag,{}).values()
            if metric in fd and fd[metric] is not None and not math.isnan(float(fd[metric]))]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))

def _ordered_tags(data: Dict) -> List[str]:
    order = [t for t in VARIANT_TAGS if t in data]
    order += [t for t in sorted(data) if t not in order]
    return order

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: Combined benchmark — all experiments in one figure
# ─────────────────────────────────────────────────────────────────────────────

def task_benchmark_combined(results_base: Path, output_dir: Path, folds: List[int]):
    plt.rcParams.update(NATURE_RC)
    out = output_dir / "benchmark_combined"
    out.mkdir(parents=True, exist_ok=True)

    # Load all experiments
    exp_data: Dict[str, Dict] = {}
    for exp_key, (dir_suf, prim, label, color) in EXPERIMENTS.items():
        rd = results_base / dir_suf
        d  = _load_metrics_dir(rd, folds)
        exp_data[exp_key] = d
        tags = _ordered_tags(d)
        mu, sd = _fold_stats(d, tags[0], prim) if tags else (float("nan"), float("nan"))
        print(f"[benchmark_combined] {label:10s}: {len(d)} variants loaded  "
              f"best {prim}={mu:.3f}±{sd:.3f} ({tags[0] if tags else '-'})")

    # ── Figure 1: 5-panel bar chart (one per experiment) ─────────────────────
    n_exp  = len(EXPERIMENTS)
    fig, axes = plt.subplots(1, n_exp, figsize=(TWO_COL_W * 1.5, 4.5), sharey=False)
    fig.patch.set_facecolor("white")

    for ax, (exp_key, (dir_suf, prim, label, color)) in zip(axes, EXPERIMENTS.items()):
        data  = exp_data[exp_key]
        tags  = _ordered_tags(data)
        if not tags:
            ax.set_title(label, fontsize=8, fontweight="bold"); ax.set_visible(True); continue

        mus   = [_fold_stats(data, t, prim)[0] for t in tags]
        sds   = [_fold_stats(data, t, prim)[1] for t in tags]
        x     = np.arange(len(tags))
        disps = [VARIANT_DISPLAY.get(t, t) for t in tags]

        bars = ax.bar(x, mus, yerr=sds, capsize=3, color=color,
                      alpha=0.82, width=0.6, edgecolor="white", lw=0.5,
                      error_kw={"elinewidth": 1.0, "ecolor": "#555"})
        ax.axhline(0.5, color="#aaa", lw=0.8, ls=":", zorder=0)

        # star best
        best_i = int(np.nanargmax(mus)) if not all(np.isnan(v) for v in mus) else -1
        if best_i >= 0 and not np.isnan(mus[best_i]):
            ax.bar(x[best_i], mus[best_i], yerr=sds[best_i], capsize=3,
                   color=color, alpha=1.0, width=0.6,
                   edgecolor="#333", lw=1.2,
                   error_kw={"elinewidth": 1.0, "ecolor": "#333"})

        ax.set_xticks(x)
        ax.set_xticklabels(disps, rotation=45, ha="right", fontsize=6)
        metric_label = "C-index" if prim == "c_index" else "BAcc"
        ax.set_ylabel(metric_label, fontsize=7)
        ax.set_title(label, fontsize=8, fontweight="bold", color=color)

        ylo = max(0.3, float(np.nanmin(mus)) - 0.08) if not all(np.isnan(mus)) else 0.3
        yhi = min(1.0, float(np.nanmax(mus)) + 0.12) if not all(np.isnan(mus)) else 1.0
        ax.set_ylim(ylo, yhi)

        for xi, (mu, sd) in enumerate(zip(mus, sds)):
            if not np.isnan(mu):
                ax.text(xi, mu + (sd if not np.isnan(sd) else 0) + 0.01,
                        f"{mu:.3f}", ha="center", va="bottom", fontsize=5.5)

        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", lw=0.5, alpha=0.4, ls="--")

    fig.suptitle("Benchmark — All Endpoints (mean ± std, split 0, 4 folds, test set)",
                 fontsize=9, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_fig(fig, out / "fig_benchmark_all_tasks")
    print(f"  Saved: {out}/fig_benchmark_all_tasks.pdf")

    # ── Figure 2: heatmap (experiments × variants) ────────────────────────────
    all_tags = []
    for e in EXPERIMENTS:
        for t in _ordered_tags(exp_data[e]):
            if t not in all_tags:
                all_tags.append(t)
    all_tags = [t for t in VARIANT_TAGS if t in all_tags] + \
               [t for t in all_tags if t not in VARIANT_TAGS]

    exp_keys  = list(EXPERIMENTS.keys())
    exp_labels= [EXPERIMENTS[e][2] for e in exp_keys]
    mat = np.full((len(exp_keys), len(all_tags)), float("nan"))
    mat_sd = np.full_like(mat, float("nan"))
    for ei, exp_key in enumerate(exp_keys):
        prim = EXPERIMENTS[exp_key][1]
        for ti, tag in enumerate(all_tags):
            mu, sd = _fold_stats(exp_data[exp_key], tag, prim)
            mat[ei, ti]    = mu
            mat_sd[ei, ti] = sd

    fig2, ax2 = plt.subplots(figsize=(max(TWO_COL_W, len(all_tags)*0.9+1.5), 4.0))
    fig2.patch.set_facecolor("white")
    vmin_h = float(np.nanmin(mat)); vmax_h = float(np.nanmax(mat))
    im = ax2.imshow(mat, cmap="RdYlGn", vmin=max(0.45, vmin_h), vmax=min(1.0, vmax_h), aspect="auto")
    plt.colorbar(im, ax=ax2, label="Metric (C-index / BAcc)", fraction=0.03, pad=0.02)
    ax2.set_xticks(range(len(all_tags)))
    ax2.set_xticklabels([VARIANT_DISPLAY.get(t,t) for t in all_tags], rotation=40, ha="right", fontsize=8)
    ax2.set_yticks(range(len(exp_keys)))
    ax2.set_yticklabels(exp_labels, fontsize=9)
    for ei in range(len(exp_keys)):
        for ti in range(len(all_tags)):
            mu = mat[ei,ti]; sd = mat_sd[ei,ti]
            if not np.isnan(mu):
                txt = f"{mu:.3f}" if np.isnan(sd) else f"{mu:.3f}\n±{sd:.3f}"
                ax2.text(ti, ei, txt, ha="center", va="center", fontsize=6.5,
                         color="white" if (mu < 0.52 or mu > 0.82) else "black")
    ax2.set_title("All endpoints — primary metric heatmap (mean over folds)",
                  fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig2, out / "fig_benchmark_heatmap_combined")
    print(f"  Saved: {out}/fig_benchmark_heatmap_combined.pdf")

    # ── Figure 3: C-index only — surv/clad/death side-by-side ────────────────
    surv_exps = ["acr_surv", "acr_alt", "clad", "death"]
    fig3, axes3 = plt.subplots(1, len(surv_exps),
                               figsize=(TWO_COL_W * 1.4, 4.0), sharey=True)
    for ax, exp_key in zip(axes3, surv_exps):
        _, dir_suf, label, color = (*EXPERIMENTS[exp_key][:1],) + EXPERIMENTS[exp_key]
        color = EXPERIMENTS[exp_key][3]
        label = EXPERIMENTS[exp_key][2]
        data  = exp_data[exp_key]
        tags  = [t for t in VARIANT_TAGS if t in data]
        if not tags:
            ax.set_title(label, fontsize=8); continue
        mus = [_fold_stats(data, t, "c_index")[0] for t in tags]
        sds = [_fold_stats(data, t, "c_index")[1] for t in tags]
        x   = np.arange(len(tags))
        ax.bar(x, mus, yerr=sds, capsize=3, color=color, alpha=0.82, width=0.6,
               edgecolor="white", lw=0.5, error_kw={"elinewidth":1.0,"ecolor":"#555"})
        ax.axhline(0.5, color="#aaa", lw=0.8, ls=":", zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_DISPLAY.get(t,t) for t in tags],
                           rotation=45, ha="right", fontsize=6.5)
        ax.set_title(label, fontsize=8.5, fontweight="bold", color=color)
        ax.set_ylim(0.45, 0.82)
        ax.spines[["top","right"]].set_visible(False)
        for xi, mu in enumerate(mus):
            if not np.isnan(mu):
                ax.text(xi, mu + 0.01, f"{mu:.3f}", ha="center", va="bottom", fontsize=6)
    axes3[0].set_ylabel("C-index", fontsize=8)
    fig3.suptitle("C-index across survival endpoints (mean ± std, test set)",
                  fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig3, out / "fig_cindex_all_surv")
    print(f"  Saved: {out}/fig_cindex_all_surv.pdf")

    # ── CSV summary ───────────────────────────────────────────────────────────
    rows = []
    for exp_key, (dir_suf, prim, label, color) in EXPERIMENTS.items():
        data = exp_data[exp_key]
        for tag in _ordered_tags(data):
            row = {"experiment": label, "variant": VARIANT_DISPLAY.get(tag,tag)}
            for metric in ["auc","bacc","mcc","c_index"]:
                mu, sd = _fold_stats(data, tag, metric)
                row[f"{metric}_mean"] = round(mu,4) if not np.isnan(mu) else None
                row[f"{metric}_std"]  = round(sd,4) if not np.isnan(sd) else None
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "benchmark_combined.csv", index=False)
    print(f"  CSV: {out}/benchmark_combined.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers (endpoint-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _load_v7_module(chicago_mil_dir: Path):
    script = chicago_mil_dir / "train_mm_abmil_v7.py"
    if not script.exists():
        raise FileNotFoundError(script)
    if str(chicago_mil_dir) not in sys.path:
        sys.path.insert(0, str(chicago_mil_dir))
    spec   = importlib.util.spec_from_file_location("tv7", str(script))
    module = importlib.util.module_from_spec(spec)
    bak, sys.argv = sys.argv[:], [str(script)]
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    finally:
        sys.argv = bak
    return module

def _tag_to_variant_kwargs(tag: str):
    if tag in ("early","early_cls","late","middle","middle_cls"):
        return tag, {}
    m = _re.match(r"crossattn(_cls)?_k(\d+)$", tag)
    if m: return ("crossattn_cls" if m.group(1) else "crossattn"), {"slot_k": int(m.group(2))}
    m = _re.match(r"crossattn_k(\d+)(_cls)?$", tag)
    if m: return ("crossattn_cls" if m.group(2) else "crossattn"), {"slot_k": int(m.group(1))}
    m = _re.match(r"crossmodal(_cls)?_k(\d+)$", tag)
    if m: return ("crossmodal_cls" if m.group(1) else "crossmodal"), {"slot_k": int(m.group(2))}
    m = _re.match(r"crossmodal_k(\d+)(_cls)?$", tag)
    if m: return ("crossmodal_cls" if m.group(2) else "crossmodal"), {"slot_k": int(m.group(1))}
    m = _re.match(r"iterative(_cls)?_r(\d+)_k(\d+)$", tag)
    if m:
        return ("iterative_cls" if m.group(1) else "iterative"), {"iter_r":int(m.group(2)),"slot_k":int(m.group(3))}
    m = _re.match(r"iterative_r(\d+)_k(\d+)(_cls)?$", tag)
    if m:
        return ("iterative_cls" if m.group(3) else "iterative"), {"iter_r":int(m.group(1)),"slot_k":int(m.group(2))}
    return None, None

def _build_and_load(tv7, tag: str, ckpt_path: Path, device):
    import torch
    variant, kwargs = _tag_to_variant_kwargs(tag)
    if variant is None:
        raise ValueError(f"Cannot map tag '{tag}' to build_model_v7")
    model = tv7.build_model_v7(variant, **kwargs)
    ckpt  = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model

def _build_stem_to_meta_endpoint(splits_csv: Path, endpoint: str) -> Tuple[dict, pd.DataFrame]:
    """Build per-stem metadata with ACR gap-time TTE + endpoint-specific TTE."""
    cfg = ENDPOINT_CFG[endpoint]
    df  = pd.read_csv(str(splits_csv))
    df["anchor_dt"] = pd.to_datetime(df["anchor_dt"])
    df["combo"]  = df.apply(_combo_from_row, axis=1)
    df["n_mods"] = df["combo"].apply(_n_mods)

    # ACR gap-time TTE
    acr_mask = df["acr_grade"].apply(
        lambda g: isinstance(g,str) and (g.startswith("A1") or g.startswith("A2"))
    )
    acr_dates: dict = {}
    for _, row in df[acr_mask].iterrows():
        acr_dates.setdefault(row["patient_id"], []).append(row["anchor_dt"])
    last_date = df.groupby("patient_id")["anchor_dt"].max().to_dict()

    meta: dict = {}
    for _, row in df.iterrows():
        stem = Path(str(row["file"])).stem
        pid  = row["patient_id"]
        t    = row["anchor_dt"]
        future = sorted([d for d in acr_dates.get(pid,[]) if d > t])
        if future:
            tte_acr, ev_acr = float((future[0]-t).days), 1
        else:
            last = last_date.get(pid, t)
            tte_acr, ev_acr = float(max((last-t).days,0)), 0
        g = str(row.get("acr_grade",""))
        lbl = (0.0 if g.startswith("A0") else
               1.0 if (g.startswith("A1") or g.startswith("A2")) else float("nan"))

        # Endpoint TTE
        try:
            ep_time  = float(row[cfg["time_col"]])
            ep_event = float(row[cfg["event_col"]])
            if math.isnan(ep_time) or ep_time < 0:
                ep_time = float("nan"); ep_event = float("nan")
        except (KeyError, TypeError, ValueError):
            ep_time = float("nan"); ep_event = float("nan")

        meta[stem] = {
            "patient_id":     pid,
            "anchor_dt":      t,
            "label":          lbl,
            "tte_next_acr":   tte_acr,
            "event_next_acr": float(ev_acr),
            "days_to_acr":    tte_acr if ev_acr == 1 else float("nan"),
            cfg["tte_key"]:   ep_time,
            cfg["ev_key"]:    ep_event,
            "days_to_event":  ep_time if (not math.isnan(ep_event) and ep_event == 1) else float("nan"),
            "combo":          row.get("combo","Unknown"),
            "n_mods":         int(row.get("n_mods",0)),
        }
    return meta, df

def _run_inference_endpoint(tv7, model, tag, stems, bag_cache, device,
                             stem_to_meta, split, fold, stem_to_datasplit,
                             tte_key, ev_key) -> list:
    import torch
    rows = []
    with torch.no_grad():
        for stem in stems:
            bags = {m: bag_cache.get(stem,{}).get(m) for m in tv7.MODALITIES}
            bags["HE_coords"] = bag_cache.get(stem,{}).get("HE_coords")
            if all(v is None for k,v in bags.items() if k != "HE_coords"):
                continue
            try:
                out = model(bags, device)
                if not isinstance(out, tuple) or len(out) < 4:
                    continue
                logit, hazard, r_cls, r_surv = out[0], out[1], out[2], out[3]
                import torch as _torch
                prob = float(_torch.sigmoid(logit.float()).item())
                haz  = float(hazard.float().item())
                meta = stem_to_meta.get(stem, {})
                dsplit = (stem_to_datasplit or {}).get(stem, "test")
                rows.append({
                    "stem":         stem,
                    "variant":      tag,
                    "patient_id":   meta.get("patient_id"),
                    "anchor_dt":    meta.get("anchor_dt"),
                    "split":        split,
                    "fold":         fold,
                    "data_split":   dsplit,
                    "cls_prob":     prob,
                    "hazard":       haz,
                    "label":        meta.get("label"),
                    "tte_next_acr": meta.get("tte_next_acr"),
                    "event_next_acr":meta.get("event_next_acr"),
                    "days_to_acr":  meta.get("days_to_acr"),
                    tte_key:        meta.get(tte_key),
                    ev_key:         meta.get(ev_key),
                    "days_to_event":meta.get("days_to_event"),
                    "combo":        meta.get("combo","Unknown"),
                    "n_mods":       meta.get("n_mods",0),
                    "rep_cls":      r_cls.detach().float().cpu().numpy(),
                    "rep_surv":     r_surv.detach().float().cpu().numpy(),
                })
            except Exception as exc:
                print(f"[infer]  {stem}: {exc}")
    return rows

def _collect_variant_data_endpoint(
    results_dir: Path, splits_csv: Path, samples_dir: Path,
    splits: List[int], folds: List[int], endpoint: str,
    device_str: str = "cpu", chicago_mil_dir: Optional[Path] = None,
) -> Optional[Dict]:
    import torch
    cfg = ENDPOINT_CFG[endpoint]
    chicago_mil = chicago_mil_dir if chicago_mil_dir else results_dir.parent.parent
    print(f"[infer] Loading tv7 from {chicago_mil} ...")
    try:
        tv7 = _load_v7_module(chicago_mil)
    except Exception as e:
        print(f"[infer] Failed: {e}"); return None

    device = torch.device(device_str if (device_str=="cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[infer] Device: {device}")

    stem_to_meta, df_csv = _build_stem_to_meta_endpoint(splits_csv, endpoint)
    all_rows: Dict[str,list] = {}

    for s in splits:
        for f in folds:
            fold_dir = results_dir / f"split{s}_fold{f}"
            if not fold_dir.exists(): continue
            split_col = f"split{s}_fold{f}"
            stem_to_datasplit = {
                Path(str(row["file"])).stem: str(row.get(split_col,""))
                for _, row in df_csv.iterrows()
                if str(row.get(split_col,"")) in ("train","val","test")
            }
            all_stems = list(stem_to_datasplit.keys())
            if not all_stems: continue
            print(f"[infer] split={s} fold={f}: preloading {len(all_stems)} bags ...")
            bag_cache = tv7.preload_bags(all_stems, str(samples_dir))

            for ckpt_dir in sorted(fold_dir.glob("ckpts_*")):
                tag = ckpt_dir.name[len("ckpts_"):]
                variant, kwargs = _tag_to_variant_kwargs(tag)
                if variant is None: continue
                ckpt_file = ckpt_dir / "best_model.pt"
                if not ckpt_file.exists():
                    print(f"[infer]   no ckpt for {tag} — skip"); continue
                print(f"[infer]   {tag} s={s} f={f}: building model ...")
                try:
                    model = _build_and_load(tv7, tag, ckpt_file, device)
                except Exception as e:
                    print(f"[infer]   build/load failed: {e}"); continue
                rows = _run_inference_endpoint(
                    tv7, model, tag, all_stems, bag_cache, device,
                    stem_to_meta, s, f, stem_to_datasplit,
                    cfg["tte_key"], cfg["ev_key"])
                all_rows.setdefault(tag, []).extend(rows)
                print(f"[infer]   {len(rows)} samples for {tag}")
                del model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            del bag_cache

    if not all_rows:
        print("[infer] No data collected."); return None

    result: Dict[str,dict] = {}
    for tag, rows in all_rows.items():
        if not rows: continue
        reps_cls  = np.stack([r.pop("rep_cls")  for r in rows])
        reps_surv = np.stack([r.pop("rep_surv") for r in rows])
        df        = pd.DataFrame(rows)
        result[tag] = {"df": df, "reps_cls": reps_cls, "reps_surv": reps_surv}
        print(f"[infer] {tag}: {len(df)} samples, dim={reps_cls.shape[1]}")
    return result

def _save_cache(variant_data: Dict, cache_path: Path):
    arrays = {}
    meta_dfs = {}
    for tag, vd in variant_data.items():
        st = tag.replace("/","_")
        arrays[f"{st}__reps_cls"]  = vd["reps_cls"]
        arrays[f"{st}__reps_surv"] = vd["reps_surv"]
        meta_dfs[st] = vd["df"]
    np.savez_compressed(str(cache_path)+"_arrays.npz", **arrays)
    with open(str(cache_path)+"_meta.pkl","wb") as f:
        pickle.dump(meta_dfs, f)
    print(f"[cache] Saved to {cache_path}_*.npz/pkl")

def _load_cache(cache_path: Path) -> Optional[Dict]:
    arr_f = str(cache_path)+"_arrays.npz"
    pkl_f = str(cache_path)+"_meta.pkl"
    if not (Path(arr_f).exists() and Path(pkl_f).exists()):
        return None
    arrays = np.load(arr_f, allow_pickle=True)
    with open(pkl_f,"rb") as f:
        meta_dfs = pickle.load(f)
    result = {}
    for st, df in meta_dfs.items():
        if f"{st}__reps_cls" in arrays:
            result[st] = {"df": df, "reps_cls": arrays[f"{st}__reps_cls"],
                          "reps_surv": arrays[f"{st}__reps_surv"]}
    print(f"[cache] Loaded {len(result)} variants from {cache_path}")
    return result

def _get_or_run_inference_endpoint(
    results_dir, splits_csv, samples_dir, splits, folds,
    endpoint, device_str, chicago_mil_dir, output_dir,
) -> Optional[Dict]:
    cache_path = output_dir / f"inference_cache_{endpoint}"
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached
    print(f"[infer] No cache — running inference for {endpoint} ...")
    vd = _collect_variant_data_endpoint(
        results_dir, splits_csv, samples_dir, splits, folds,
        endpoint, device_str, chicago_mil_dir)
    if vd:
        _save_cache(vd, cache_path)
    return vd

# ─────────────────────────────────────────────────────────────────────────────
# UMAP plot — endpoint-aware 3×3 panel
# ─────────────────────────────────────────────────────────────────────────────

def _plot_umap_split_endpoint(df: pd.DataFrame, xy: np.ndarray,
                               dsplit: str, endpoint: str,
                               title: str, out_path: Path):
    """
    3×3 UMAP for one data split:
    Row 0: [event label, modality combo, n_mods]
    Row 1: [hazard, TTE (event only), cls_prob]
    Row 2: [hexbin event density, hexbin TTE, hexbin hazard]
    """
    cfg    = ENDPOINT_CFG[endpoint]
    ep_lbl = cfg["label"]
    mask_sp = df["data_split"].values == dsplit
    if mask_sp.sum() == 0:
        print(f"[umap] No {dsplit} samples — skip"); return

    df_s = df[mask_sp].reset_index(drop=True)
    xy_s = xy[mask_sp]

    fig, axes = plt.subplots(3, 3, figsize=(TWO_COL_W*1.4, TWO_COL_W*1.3))
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ["top","right","left","bottom"]: ax.spines[sp].set_visible(False)

    ev_key  = cfg["ev_key"]
    tte_key = cfg["tte_key"]
    events  = df_s[ev_key].values.astype(float)   if ev_key  in df_s.columns else np.full(len(df_s),float("nan"))
    ttes    = df_s[tte_key].values.astype(float)   if tte_key in df_s.columns else np.full(len(df_s),float("nan"))
    haz     = df_s["hazard"].values.astype(float)
    prob    = df_s["cls_prob"].values.astype(float)
    combos  = df_s["combo"].values
    n_mods_v= df_s["n_mods"].values.astype(float)

    # ── Row 0 ─────────────────────────────────────────────────────────────────
    ax = axes[0,0]
    ev_known = ~np.isnan(events)
    if ev_known.sum() > 0:
        c_ev = np.where(events[ev_known]==1.0, cfg["color"], "#1E88E5")
        ax.scatter(xy_s[ev_known,0], xy_s[ev_known,1], c=c_ev,
                   s=10, alpha=0.75, linewidths=0, rasterized=True)
    ax.legend(handles=[Patch(color=cfg["color"],label=f"{ep_lbl} event"),
                        Patch(color="#1E88E5",label="Censored")],
              fontsize=6, loc="best", framealpha=0.8)
    ax.set_title(f"{ep_lbl} event status", fontsize=7, fontweight="bold")

    ax = axes[0,1]
    for c_name in sorted(set(combos), key=lambda x: -np.sum(combos==x)):
        mk = combos == c_name
        ax.scatter(xy_s[mk,0], xy_s[mk,1],
                   c=COMBO_COLORS.get(c_name,DEFAULT_COMBO_COLOR),
                   s=10, alpha=0.72, linewidths=0, label=c_name, rasterized=True)
    ax.legend(fontsize=5, loc="upper right", ncol=2, framealpha=0.75,
              markerscale=1.2, handlelength=1, borderpad=0.4)
    ax.set_title("Modality combination", fontsize=7, fontweight="bold")

    sc_nm = axes[0,2].scatter(xy_s[:,0], xy_s[:,1], c=n_mods_v, cmap="plasma",
                               vmin=0.5, vmax=4.5, s=10, alpha=0.75, linewidths=0, rasterized=True)
    cb = plt.colorbar(sc_nm, ax=axes[0,2], fraction=0.046, pad=0.04)
    cb.set_ticks([1,2,3,4]); cb.ax.tick_params(labelsize=6)
    axes[0,2].set_title("# modalities", fontsize=7, fontweight="bold")

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    hmin, hmax = _pct(haz)
    _scatter(axes[1,0], xy_s, haz, "Hazard score", "hot_r", hmin, hmax)

    ax = axes[1,1]
    ev_mask = (events == 1) & ~np.isnan(ttes)
    if ev_mask.sum() > 0:
        tte_ev = ttes[ev_mask]
        vmax_t = float(np.percentile(tte_ev, 98))
        sc_t = ax.scatter(xy_s[ev_mask,0], xy_s[ev_mask,1], c=tte_ev, cmap="viridis_r",
                          vmin=0, vmax=max(vmax_t,1), s=10, alpha=0.8, linewidths=0, rasterized=True)
        cb2 = plt.colorbar(sc_t, ax=ax, fraction=0.046, pad=0.04)
        cb2.ax.tick_params(labelsize=6)
    ax.set_title(f"Days to {ep_lbl} (event=1 only)", fontsize=7, fontweight="bold")

    _scatter(axes[1,2], xy_s, prob, "ACR cls probability", "RdYlGn_r", 0, 1)

    # ── Row 2 ─────────────────────────────────────────────────────────────────
    _hexbin(axes[2,0], xy_s, events,
            f"Hexbin: {ep_lbl} event density", "Reds", 0.0, 1.0, f"prop {ep_lbl}")
    vmax_hex = (float(np.nanpercentile(ttes[~np.isnan(ttes)],98))
                if (~np.isnan(ttes)).sum()>0 else 1.0)
    _hexbin(axes[2,1], xy_s, ttes,
            f"Hexbin: mean days to {ep_lbl}", "viridis_r", 0.0, vmax_hex, "mean days")
    _hexbin(axes[2,2], xy_s, haz,
            "Hexbin: mean hazard", "hot_r", hmin, hmax, "mean hazard")

    n_sp  = int(mask_sp.sum())
    n_ev  = int((events==1.0).sum())
    fig.suptitle(f"{title}  [{dsplit.upper()}  n={n_sp}  {ep_lbl}_event={n_ev}]",
                 fontsize=8, fontweight="bold", y=1.002)
    plt.tight_layout()
    _save_fig(fig, out_path)
    print(f"[umap] → {out_path}.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: UMAP for CLAD or Death
# ─────────────────────────────────────────────────────────────────────────────

def task_umap_endpoint(
    endpoint: str,
    results_dir: Path, splits_csv: Path, samples_dir: Path,
    output_dir: Path, splits: List[int], folds: List[int],
    device_str: str, chicago_mil_dir: Optional[Path],
    umap_n_neighbors: int = 30, umap_min_dist: float = 0.3,
):
    if not HAS_UMAP:
        print("[umap] umap-learn not installed — skipping."); return
    plt.rcParams.update(NATURE_RC)
    cfg = ENDPOINT_CFG[endpoint]
    out = output_dir / f"umap_{endpoint}"
    out.mkdir(parents=True, exist_ok=True)

    variant_data = _get_or_run_inference_endpoint(
        results_dir, splits_csv, samples_dir, splits, folds,
        endpoint, device_str, chicago_mil_dir, output_dir)
    if not variant_data: return

    tags_sorted = [t for t in VARIANT_TAGS if t in variant_data]
    tags_sorted += [t for t in sorted(variant_data) if t not in tags_sorted]

    stored_xy: Dict[str,dict] = {}

    for tag in tags_sorted:
        vd     = variant_data[tag]
        df     = vd["df"]
        r_cls  = vd["reps_cls"]
        r_surv = vd["reps_surv"]
        v_out  = out / tag
        v_out.mkdir(exist_ok=True)
        disp   = VARIANT_DISPLAY.get(tag, tag)

        reducer = UMAPTransform(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist,
                                 random_state=42, n_jobs=4)
        print(f"[umap] {tag}: UMAP on r_cls ({r_cls.shape}) ...")
        xy_cls = reducer.fit_transform(r_cls)
        df["umap_cls_x"] = xy_cls[:,0]; df["umap_cls_y"] = xy_cls[:,1]

        print(f"[umap] {tag}: UMAP on r_surv ({r_surv.shape}) ...")
        xy_surv = reducer.fit_transform(r_surv)
        df["umap_surv_x"] = xy_surv[:,0]; df["umap_surv_y"] = xy_surv[:,1]

        stored_xy[tag] = {"cls": xy_cls, "surv": xy_surv}

        for dsplit in ["train","val","test"]:
            _plot_umap_split_endpoint(
                df, xy_cls, dsplit, endpoint,
                f"{disp} ({endpoint.upper()}) — r_cls",
                v_out / f"fig_umap_{tag}_rcls_{dsplit}")
            _plot_umap_split_endpoint(
                df, xy_surv, dsplit, endpoint,
                f"{disp} ({endpoint.upper()}) — r_surv",
                v_out / f"fig_umap_{tag}_rsurv_{dsplit}")

        # Patient-level UMAP (test set, mean-pool per patient)
        test_mask = df["data_split"].values == "test"
        df_test   = df[test_mask].reset_index(drop=True)
        xy_test   = xy_cls[test_mask]
        pids_test = df_test["patient_id"].values
        unique_pids = list(dict.fromkeys(pids_test))

        if len(unique_pids) >= 4:
            r_test = r_cls[test_mask]
            ev_key  = cfg["ev_key"]
            tte_key = cfg["tte_key"]
            pat_reps, pat_events, pat_ttes, pat_combos = [], [], [], []
            for pid in unique_pids:
                mk = pids_test == pid
                pat_reps.append(r_test[mk].mean(axis=0))
                ev_v = df_test.loc[mk, ev_key].dropna() if ev_key in df_test.columns else pd.Series()
                pat_events.append(float(ev_v.mode()[0]) if len(ev_v)>0 else float("nan"))
                tte_v = df_test.loc[mk, tte_key].dropna() if tte_key in df_test.columns else pd.Series()
                pat_ttes.append(float(tte_v.min()) if len(tte_v)>0 else float("nan"))
                co_v = df_test.loc[mk,"combo"]
                pat_combos.append(co_v.mode()[0] if len(co_v)>0 else "Unknown")

            pat_xy = UMAPTransform(n_neighbors=min(15,len(unique_pids)-1),
                                    min_dist=0.1, random_state=42, n_jobs=4
                                    ).fit_transform(np.stack(pat_reps))
            pat_events = np.array(pat_events)
            pat_ttes   = np.array(pat_ttes)

            fig_p, ax_p = plt.subplots(1, 3, figsize=(TWO_COL_W*1.2, 3.8))
            fig_p.patch.set_facecolor("white")
            for ax in ax_p:
                ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([])

            # Panel 0: event status
            ev_k = ~np.isnan(pat_events)
            if ev_k.sum()>0:
                c_p = [cfg["color"] if e==1 else "#1E88E5" for e in pat_events[ev_k]]
                ax_p[0].scatter(pat_xy[ev_k,0], pat_xy[ev_k,1], c=c_p, s=30, alpha=0.85,
                                linewidths=0.3, edgecolors="#333")
            ax_p[0].set_title(f"{ep_lbl} event", fontsize=8, fontweight="bold")
            ax_p[0].legend(handles=[Patch(color=cfg["color"],label="Event"),
                                     Patch(color="#1E88E5",label="Censored")],
                           fontsize=7)

            # Panel 1: TTE (events only)
            ev_m = (pat_events==1) & ~np.isnan(pat_ttes)
            if ev_m.sum()>0:
                vmax_t = float(np.percentile(pat_ttes[ev_m], 95))
                sc = ax_p[1].scatter(pat_xy[ev_m,0], pat_xy[ev_m,1], c=pat_ttes[ev_m],
                                      cmap="viridis_r", vmin=0, vmax=max(vmax_t,1),
                                      s=30, alpha=0.85, linewidths=0.3, edgecolors="#333")
                plt.colorbar(sc, ax=ax_p[1], fraction=0.046, pad=0.04, label="days")
            ax_p[1].set_title(f"Days to {ep_lbl} (events)", fontsize=8, fontweight="bold")

            # Panel 2: modality combo
            for c_name in sorted(set(pat_combos),key=lambda x:-np.sum(np.array(pat_combos)==x)):
                mk = np.array(pat_combos)==c_name
                ax_p[2].scatter(pat_xy[mk,0], pat_xy[mk,1],
                                c=COMBO_COLORS.get(c_name,DEFAULT_COMBO_COLOR),
                                s=30, alpha=0.85, linewidths=0, label=c_name)
            ax_p[2].legend(fontsize=6, loc="upper right", ncol=2, framealpha=0.75)
            ax_p[2].set_title("Modality combo", fontsize=8, fontweight="bold")

            fig_p.suptitle(f"{disp} ({endpoint.upper()}) — patient-level UMAP (test)",
                           fontsize=9, fontweight="bold")
            plt.tight_layout()
            _save_fig(fig_p, v_out / f"fig_umap_{tag}_patient_test")
            print(f"[umap] → {v_out}/fig_umap_{tag}_patient_test.pdf")

    # ── Comparison figure: all variants side by side (r_surv, test only) ─────
    ep_lbl = cfg["label"]
    tags_with_xy = [t for t in tags_sorted if t in stored_xy]
    if len(tags_with_xy) >= 2:
        n_tags = len(tags_with_xy)
        fig_c, axes_c = plt.subplots(2, n_tags, figsize=(TWO_COL_W*n_tags/3, 5.5))
        if n_tags == 1: axes_c = axes_c[:, np.newaxis]
        fig_c.patch.set_facecolor("white")
        ev_key = cfg["ev_key"]
        for ti, tag in enumerate(tags_with_xy):
            vd    = variant_data[tag]
            df    = vd["df"]
            xy_s  = stored_xy[tag]["surv"]
            test_m = df["data_split"].values == "test"
            if test_m.sum() == 0: continue
            df_t  = df[test_m]
            xy_t  = xy_s[test_m]
            events= df_t[ev_key].values.astype(float) if ev_key in df_t.columns else np.full(len(df_t),float("nan"))

            ax0 = axes_c[0,ti]; ax1 = axes_c[1,ti]
            for ax in [ax0,ax1]:
                ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([])
                for sp in ["top","right","left","bottom"]: ax.spines[sp].set_visible(False)

            ev_k = ~np.isnan(events)
            if ev_k.sum()>0:
                c_ev = [cfg["color"] if e==1 else "#1E88E5" for e in events[ev_k]]
                ax0.scatter(xy_t[ev_k,0], xy_t[ev_k,1], c=c_ev, s=8, alpha=0.75, linewidths=0, rasterized=True)
            ax0.set_title(VARIANT_DISPLAY.get(tag,tag), fontsize=7, fontweight="bold")
            if ti==0: ax0.set_ylabel(f"{ep_lbl} event", fontsize=7)

            haz = df_t["hazard"].values.astype(float)
            hmin, hmax = _pct(haz)
            sc = ax1.scatter(xy_t[:,0], xy_t[:,1], c=haz, cmap="hot_r", vmin=hmin, vmax=hmax,
                             s=8, alpha=0.75, linewidths=0, rasterized=True)
            if ti == n_tags-1:
                plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label="hazard")
            if ti==0: ax1.set_ylabel("Hazard", fontsize=7)

        fig_c.suptitle(f"{ep_lbl} — r_surv UMAP comparison (test set)",
                       fontsize=9, fontweight="bold")
        plt.tight_layout()
        _save_fig(fig_c, out / f"fig_umap_{endpoint}_comparison")
        print(f"[umap] → {out}/fig_umap_{endpoint}_comparison.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 3: Per-combo C-index for CLAD/Death
# ─────────────────────────────────────────────────────────────────────────────

def task_combo_perf_endpoint(
    endpoint: str,
    results_dir: Path, splits_csv: Path, samples_dir: Path,
    output_dir: Path, splits: List[int], folds: List[int],
    device_str: str, chicago_mil_dir: Optional[Path],
    focus_variant: str = "middle", min_combo_n: int = 8,
):
    plt.rcParams.update(NATURE_RC)
    cfg    = ENDPOINT_CFG[endpoint]
    ep_lbl = cfg["label"]
    out    = output_dir / f"combo_perf_{endpoint}"
    out.mkdir(parents=True, exist_ok=True)

    variant_data = _get_or_run_inference_endpoint(
        results_dir, splits_csv, samples_dir, splits, folds,
        endpoint, device_str, chicago_mil_dir, output_dir)
    if not variant_data: return

    tags_sorted = [t for t in VARIANT_TAGS if t in variant_data]
    tags_sorted += [t for t in sorted(variant_data) if t not in tags_sorted]
    focus_tag = focus_variant if focus_variant in variant_data else (tags_sorted[0] if tags_sorted else None)
    if focus_tag is None: return

    ev_key  = cfg["ev_key"]
    tte_key = cfg["tte_key"]

    def _combo_cindex(df_sub):
        df_t = df_sub[df_sub["data_split"]=="test"] if "data_split" in df_sub.columns else df_sub
        res  = {}
        for combo, grp in df_t.groupby("combo"):
            n = len(grp)
            if ev_key not in grp.columns or tte_key not in grp.columns:
                res[combo] = {"n": n, "cindex": float("nan"), "n_events": 0}
                continue
            ev   = grp[ev_key].values.astype(float)
            tte  = grp[tte_key].values.astype(float)
            haz  = grp["hazard"].values.astype(float)
            n_ev = int((ev==1).sum())
            ci   = _safe_cindex(tte, ev, haz) if n >= min_combo_n else float("nan")
            res[combo] = {"n": n, "cindex": ci, "n_events": n_ev}
        return res

    # CSV for all variants
    rows = []
    for tag in tags_sorted:
        cm = _combo_cindex(variant_data[tag]["df"])
        for combo, vals in cm.items():
            rows.append({"variant": tag, "combo": combo, **vals})
    pd.DataFrame(rows).to_csv(out / f"combo_cindex_{endpoint}.csv", index=False)

    # Focus-variant figure
    cm_focus = _combo_cindex(variant_data[focus_tag]["df"])
    cm_focus = {c: v for c,v in cm_focus.items() if v["n"] >= min_combo_n and not np.isnan(v["cindex"])}
    combos_s = sorted(cm_focus, key=lambda c: cm_focus[c]["n"], reverse=True)
    if not combos_s:
        print(f"[combo_perf_{endpoint}] No combos with n≥{min_combo_n}"); return

    n_c  = len(combos_s)
    x_p  = np.arange(n_c)
    cols = [COMBO_COLORS.get(c, DEFAULT_COMBO_COLOR) for c in combos_s]

    fig, axes = plt.subplots(2,1, figsize=(max(TWO_COL_W, n_c*0.75+1.5), 7.0),
                              gridspec_kw={"height_ratios":[1.6,0.8]}, facecolor="white")

    ci_vals  = [cm_focus[c]["cindex"]  for c in combos_s]
    n_vals   = [cm_focus[c]["n"]       for c in combos_s]
    n_ev_vals= [cm_focus[c]["n_events"]for c in combos_s]

    axes[0].bar(x_p, ci_vals, color=cols, alpha=0.85, width=0.65, edgecolor="white", lw=0.5)
    axes[0].axhline(0.5, color="#aaa", lw=0.8, ls=":", zorder=0)
    for xi, v in enumerate(ci_vals):
        if not np.isnan(v):
            axes[0].text(xi, v+0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    axes[0].set_xticks(x_p); axes[0].set_xticklabels([], fontsize=0)
    axes[0].set_ylim(0.3,1.0); axes[0].set_ylabel("C-index", fontsize=8, fontweight="bold")
    axes[0].set_title(f"{VARIANT_DISPLAY.get(focus_tag,focus_tag)} — {ep_lbl} C-index "
                      f"per modality combo (test set)", fontsize=8, fontweight="bold")
    axes[0].spines[["top","right"]].set_visible(False)

    x_bot = np.arange(n_c)
    axes[1].bar(x_bot,       n_vals,    color=cols, alpha=0.75, width=0.65, label="n total",
                edgecolor="white", lw=0.5)
    axes[1].bar(x_bot+0.0,   n_ev_vals, color="#555", alpha=0.55, width=0.3, label=f"n events",
                edgecolor="white", lw=0.5)
    axes[1].set_xticks(x_bot)
    axes[1].set_xticklabels(combos_s, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Count", fontsize=7)
    axes[1].legend(fontsize=7)
    axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, out / f"fig_combo_cindex_{endpoint}_{focus_tag}")
    print(f"  Saved: {out}/fig_combo_cindex_{endpoint}_{focus_tag}.pdf")

    # All-variants heatmap
    all_combos = sorted({c for row in rows for c in [row["combo"]]}
                         if rows else set())
    all_combos = sorted(all_combos, key=lambda c: sum(
        1 for r in rows if r["combo"]==c and not (r.get("cindex") is None or
        (isinstance(r.get("cindex"),float) and math.isnan(r["cindex"])))))[::-1]
    mat_h = np.full((len(tags_sorted), len(all_combos)), float("nan"))
    for ti, tag in enumerate(tags_sorted):
        cm_t = {r["combo"]: r for r in rows if r["variant"]==tag}
        for ci2, combo in enumerate(all_combos):
            if combo in cm_t:
                v = cm_t[combo].get("cindex", float("nan"))
                mat_h[ti, ci2] = v if v is not None else float("nan")

    if len(all_combos) > 2:
        fig_h, ax_h = plt.subplots(figsize=(max(TWO_COL_W, len(all_combos)*0.75+1.5),
                                            len(tags_sorted)*0.55+1.5))
        fig_h.patch.set_facecolor("white")
        im = ax_h.imshow(mat_h, cmap="RdYlGn", vmin=0.4, vmax=0.8, aspect="auto")
        plt.colorbar(im, ax=ax_h, label="C-index", fraction=0.03, pad=0.02)
        ax_h.set_xticks(range(len(all_combos)))
        ax_h.set_xticklabels(all_combos, rotation=45, ha="right", fontsize=7)
        ax_h.set_yticks(range(len(tags_sorted)))
        ax_h.set_yticklabels([VARIANT_DISPLAY.get(t,t) for t in tags_sorted], fontsize=8)
        for ti in range(len(tags_sorted)):
            for ci2 in range(len(all_combos)):
                v = mat_h[ti,ci2]
                if not np.isnan(v):
                    ax_h.text(ci2,ti,f"{v:.3f}",ha="center",va="center",
                              fontsize=6,color="white" if (v<0.48 or v>0.72) else "black")
        ax_h.set_title(f"{ep_lbl} — C-index heatmap (variant × combo, test set)",
                       fontsize=9, fontweight="bold")
        plt.tight_layout()
        _save_fig(fig_h, out / f"fig_combo_heatmap_{endpoint}")
        print(f"  Saved: {out}/fig_combo_heatmap_{endpoint}.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="All-endpoint analysis for v7 models")
    p.add_argument("--tasks", nargs="+",
                   choices=["benchmark_combined","umap_clad","umap_death",
                             "combo_perf_clad","combo_perf_death","all"],
                   default=["all"])
    p.add_argument("--results_base", type=str,
                   default="/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil")
    p.add_argument("--splits_csv", type=str,
                   default="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
    p.add_argument("--samples_dir", type=str,
                   default="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples")
    p.add_argument("--output_dir", type=str,
                   default="/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/analysis/v7_endpoints_full")
    p.add_argument("--chicago_mil_dir", type=str,
                   default="/home/aih/dinesh.haridoss/chicago_mil")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--folds", nargs="+", type=int, default=[0,1,2,3])
    p.add_argument("--focus_variant", type=str, default="middle")
    p.add_argument("--min_combo_n", type=int, default=8)
    p.add_argument("--umap_n_neighbors", type=int, default=30)
    p.add_argument("--umap_min_dist", type=float, default=0.3)
    return p.parse_args()


if __name__ == "__main__":
    args     = parse_args()
    rb       = Path(args.results_base)
    sc       = Path(args.splits_csv)
    sd       = Path(args.samples_dir)
    out      = Path(args.output_dir)
    cml      = Path(args.chicago_mil_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_all = "all" in args.tasks
    tasks   = set(args.tasks)
    splits  = [0]

    print("=" * 70)
    print("  Multimodal ABMIL v7 — All-Endpoint Analysis")
    print(f"  Tasks:  {args.tasks}")
    print(f"  Output: {out}")
    print("=" * 70)

    if run_all or "benchmark_combined" in tasks:
        print("\n── benchmark_combined ──────────────────────────────────────────────")
        task_benchmark_combined(rb, out, args.folds)

    for endpoint in ["clad", "death"]:
        res_dir = rb / EXPERIMENTS[endpoint][0]
        if run_all or f"umap_{endpoint}" in tasks:
            print(f"\n── umap_{endpoint} ─────────────────────────────────────────────────")
            task_umap_endpoint(
                endpoint=endpoint,
                results_dir=res_dir,
                splits_csv=sc, samples_dir=sd, output_dir=out,
                splits=splits, folds=args.folds,
                device_str=args.device, chicago_mil_dir=cml,
                umap_n_neighbors=args.umap_n_neighbors,
                umap_min_dist=args.umap_min_dist,
            )
        if run_all or f"combo_perf_{endpoint}" in tasks:
            print(f"\n── combo_perf_{endpoint} ────────────────────────────────────────────")
            task_combo_perf_endpoint(
                endpoint=endpoint,
                results_dir=res_dir,
                splits_csv=sc, samples_dir=sd, output_dir=out,
                splits=splits, folds=args.folds,
                device_str=args.device, chicago_mil_dir=cml,
                focus_variant=args.focus_variant,
                min_combo_n=args.min_combo_n,
            )

    print(f"\n{'='*70}")
    print(f"  Done. Output: {out}")
    print(f"{'='*70}")
