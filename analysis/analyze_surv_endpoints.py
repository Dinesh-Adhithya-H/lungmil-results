#!/usr/bin/env python3
"""
analyze_surv_endpoints.py — Benchmark analysis for CLAD and Death survival tasks.

Reads metrics_*.json from results_mm_abmil_v7_clad / results_mm_abmil_v7_death,
produces C-index comparison figures and a text/CSV summary.
"""

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index as lifelines_ci
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("[warn] lifelines not installed; Cox baseline will use concordance only")

# ── Constants ─────────────────────────────────────────────────────────────────

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
ENDPOINT_COLORS = {
    "clad":  "#FB8500",
    "death": "#1B2A3B",
}
ENDPOINT_DISPLAY = {
    "clad":  "CLAD",
    "death": "Death",
}

NATURE_RC = {
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_metrics(results_dir: Path, folds: List[int]) -> Dict[str, Dict[int, dict]]:
    """data[variant_tag][fold] = test_metrics_dict"""
    data: Dict = {}
    for f in folds:
        fold_dir = results_dir / f"split0_fold{f}"
        if not fold_dir.exists():
            continue
        for mfile in sorted(fold_dir.glob("metrics_*.json")):
            tag = mfile.stem.replace("metrics_", "")
            try:
                with open(mfile) as fi:
                    m = json.load(fi)
                test = m.get("test", {})
                data.setdefault(tag, {})[f] = test
            except Exception as e:
                print(f"  [warn] {mfile}: {e}")
    return data


def _fold_stats(data: Dict, tag: str) -> Tuple[float, float, List[float]]:
    """Mean, std, and per-fold list of C-index for a variant."""
    vals = []
    for fold_data in data.get(tag, {}).values():
        v = fold_data.get("c_index")
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(float(v))
    if not vals:
        return float("nan"), float("nan"), []
    return float(np.mean(vals)), float(np.std(vals)), vals


def _ordered_tags(data: Dict) -> List[str]:
    order = [t for t in VARIANT_TAGS if t in data]
    order += [t for t in sorted(data.keys()) if t not in order]
    return order


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_analysis(results_base: Path, output_dir: Path, folds: List[int]):
    plt.rcParams.update(NATURE_RC)
    out = output_dir / "surv_endpoints"
    out.mkdir(parents=True, exist_ok=True)

    endpoints = ["clad", "death"]
    all_data: Dict[str, Dict] = {}
    for ep in endpoints:
        d = results_base / f"results_mm_abmil_v7_{ep}"
        if not d.exists():
            print(f"[warn] Results dir not found: {d}")
            all_data[ep] = {}
        else:
            all_data[ep] = _load_metrics(d, folds)
            print(f"[{ep}] loaded {len(all_data[ep])} variants")

    # ── Text table ────────────────────────────────────────────────────────────
    rows_out = []
    print("\n" + "=" * 70)
    print("  CLAD & Death Survival — C-index (mean ± std across folds)")
    print("=" * 70)
    for ep in endpoints:
        data = all_data[ep]
        if not data:
            continue
        tags = _ordered_tags(data)
        print(f"\n  Endpoint: {ENDPOINT_DISPLAY[ep]}")
        print(f"  {'Variant':<22}  {'C-index':>12}  {'Per-fold values'}")
        print("  " + "-" * 60)
        ep_rows = []
        for tag in tags:
            mu, sd, vals = _fold_stats(data, tag)
            disp = VARIANT_DISPLAY.get(tag, tag)
            fv = "  ".join(f"{v:.3f}" for v in vals) if vals else "—"
            star = " ★" if vals and mu == max(_fold_stats(data, t)[0] for t in tags
                                               if not math.isnan(_fold_stats(data, t)[0])) else ""
            print(f"  {disp:<22}  {mu:.3f}±{sd:.3f}{star:>2}  [{fv}]")
            ep_rows.append({"endpoint": ep, "variant": disp, "tag": tag,
                            "ci_mean": mu, "ci_std": sd, "n_folds": len(vals)})
        rows_out.extend(ep_rows)

    # Save CSV
    df = pd.DataFrame(rows_out)
    csv_path = out / "surv_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved → {csv_path}")

    # ── Figure 1: bar chart comparison ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    fig.suptitle("Survival Benchmark — C-index by Variant", fontsize=11, fontweight="bold")

    for ax, ep in zip(axes, endpoints):
        data = all_data[ep]
        tags = _ordered_tags(data)
        mus, sds, labels = [], [], []
        for tag in tags:
            mu, sd, _ = _fold_stats(data, tag)
            mus.append(mu); sds.append(sd)
            labels.append(VARIANT_DISPLAY.get(tag, tag))

        x = np.arange(len(labels))
        bars = ax.bar(x, mus, yerr=sds, capsize=4,
                      color=ENDPOINT_COLORS[ep], alpha=0.78, width=0.6,
                      error_kw=dict(lw=1.2, capthick=1.2))
        ax.axhline(0.5, color="#999", lw=1, linestyle="--", label="random (0.5)")
        for xi, (mu, sd) in enumerate(zip(mus, sds)):
            if not math.isnan(mu):
                ax.text(xi, mu + sd + 0.005, f"{mu:.3f}", ha="center",
                        va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("C-index"); ax.set_ylim(0.3, 0.85)
        ax.set_title(f"{ENDPOINT_DISPLAY[ep]} survival", fontweight="bold")
        ax.legend(fontsize=7)

    plt.tight_layout()
    _save(fig, out / "fig_surv_bar")

    # ── Figure 2: side-by-side C-index comparison (CLAD vs Death) ────────────
    # Use only variants present in both
    common_tags = [t for t in VARIANT_TAGS
                   if t in all_data.get("clad", {}) and t in all_data.get("death", {})]
    if not common_tags:
        common_tags = list(set(all_data.get("clad", {}).keys()) &
                           set(all_data.get("death", {}).keys()))

    if common_tags:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(common_tags))
        w = 0.35
        for i, ep in enumerate(endpoints):
            data = all_data[ep]
            mus = [_fold_stats(data, t)[0] for t in common_tags]
            sds = [_fold_stats(data, t)[1] for t in common_tags]
            ax.bar(x + (i - 0.5) * w, mus, w, yerr=sds, capsize=3,
                   color=ENDPOINT_COLORS[ep], alpha=0.80, label=ENDPOINT_DISPLAY[ep],
                   error_kw=dict(lw=1.1, capthick=1.1))
        ax.axhline(0.5, color="#999", lw=1, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_DISPLAY.get(t, t) for t in common_tags],
                            rotation=30, ha="right")
        ax.set_ylabel("C-index"); ax.set_ylim(0.3, 0.85)
        ax.set_title("CLAD vs Death — C-index by Variant", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        _save(fig, out / "fig_surv_comparison")

    # ── Figure 3: per-fold C-index strip plot ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    fig.suptitle("Per-fold C-index Distribution", fontsize=11, fontweight="bold")
    for ax, ep in zip(axes, endpoints):
        data = all_data[ep]
        tags = _ordered_tags(data)
        labels = [VARIANT_DISPLAY.get(t, t) for t in tags]
        for xi, tag in enumerate(tags):
            _, _, vals = _fold_stats(data, tag)
            jitter = np.random.uniform(-0.15, 0.15, len(vals))
            ax.scatter([xi + j for j in jitter], vals,
                       color=ENDPOINT_COLORS[ep], alpha=0.75, s=28, zorder=3)
            if vals:
                ax.plot([xi - 0.25, xi + 0.25], [np.mean(vals)] * 2,
                        color=ENDPOINT_COLORS[ep], lw=2, zorder=4)
        ax.axhline(0.5, color="#999", lw=1, linestyle="--")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("C-index"); ax.set_ylim(0.2, 0.9)
        ax.set_title(f"{ENDPOINT_DISPLAY[ep]} — per fold", fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "fig_surv_folds")

    print(f"\n  All figures saved to {out}")
    return df


def _save(fig, path_stem: Path):
    for ext, dpi in [("pdf", 200), ("png", 150)]:
        p = path_stem.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
        print(f"  Saved: {p}")
    plt.close(fig)


# ── TTE / label distribution visualizations ───────────────────────────────────

def _build_acr_records(df: pd.DataFrame, fold: int, split_col: str) -> Dict[str, list]:
    """
    Compute ACR gap-time TTE per biopsy (same logic as compute_tte_next_acr in v7).
    Returns dict split → list of {tte, event, label}.
    """
    df = df.copy()
    df["anchor_dt"] = pd.to_datetime(df["anchor_dt"])

    def _acr_label(g):
        if not isinstance(g, str): return None
        g = g.strip()
        if g.startswith("A0"): return 0
        if g.startswith("A1") or g.startswith("A2"): return 1
        return None

    df["_label"] = df["acr_grade"].apply(_acr_label)

    # Per-patient future A1/A2 dates
    pos_dates: Dict = {}
    for _, row in df[df["_label"] == 1].iterrows():
        pos_dates.setdefault(row["patient_id"], []).append(row["anchor_dt"])
    last_dt = df.groupby("patient_id")["anchor_dt"].max().to_dict()

    splits: Dict[str, list] = {"train": [], "val": [], "test": []}
    for _, row in df.iterrows():
        sp = str(row.get(split_col, ""))
        if sp not in splits:
            continue
        lbl = row["_label"]
        t   = row["anchor_dt"]
        pid = row["patient_id"]

        if lbl == 1:
            tte, ev = 0.0, 1
        elif lbl == 0:
            future = sorted([d for d in pos_dates.get(pid, []) if d > t])
            if future:
                tte, ev = float((future[0] - t).days), 1
            else:
                tte, ev = float(max((last_dt.get(pid, t) - t).days, 0)), 0
        else:
            continue  # no label → skip
        splits[sp].append({"tte": tte, "event": ev, "label": lbl})
    return splits


def _build_surv_records(df: pd.DataFrame, fold: int, split_col: str,
                        endpoint: str) -> Dict[str, list]:
    """
    Compute survival TTE per biopsy for CLAD or Death (mirrors build_splits_survival).
    Returns dict split → list of {tte, event}.
    """
    df = df.copy()
    df["anchor_dt"] = pd.to_datetime(df["anchor_dt"])
    study_end = df["anchor_dt"].max()

    time_col  = f"{endpoint}_days"    # gap time: event_date - anchor_dt
    event_col = f"{endpoint}_status"

    splits: Dict[str, list] = {"train": [], "val": [], "test": []}
    for _, row in df.iterrows():
        sp = str(row.get(split_col, ""))
        if sp not in splits:
            continue

        try:
            t = float(row.get(time_col,  float("nan")))
            e = float(row.get(event_col, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isnan(e):
            continue

        if e == 1.0:
            if math.isnan(t) or t <= 0:
                continue  # post-event biopsy or bad data
        else:  # censored
            if math.isnan(t) or t <= 0:
                # try death_days as proxy for CLAD, else study_end
                proxy = float("nan")
                if endpoint == "clad":
                    try: proxy = float(row.get("death_days", float("nan")))
                    except (TypeError, ValueError): pass
                if math.isnan(proxy) or proxy <= 0:
                    try:
                        proxy = float((study_end - row["anchor_dt"]).days)
                    except Exception:
                        proxy = float("nan")
                if math.isnan(proxy) or proxy <= 0:
                    continue
                t = proxy

        splits[sp].append({"tte": t, "event": int(e)})
    return splits


def task_tte_distributions(splits_csv: Path, output_dir: Path, folds: List[int]):
    """
    For each endpoint (ACR, CLAD, Death) × fold × split:
      - TTE histogram (event vs censored)
      - Event/label count bar

    Produces:
      fig_tte_distributions_{endpoint}.pdf   4-fold × 3-split grid of TTE histograms
      fig_label_counts.pdf                   stacked bar: events vs censored per split/fold
    """
    plt.rcParams.update(NATURE_RC)
    out = output_dir
    out.mkdir(parents=True, exist_ok=True)

    df_full = pd.read_csv(splits_csv)

    SPLITS   = ["train", "val", "test"]
    ENDPOINTS = [
        ("acr",   "ACR",   "#E63946"),
        ("clad",  "CLAD",  "#FB8500"),
        ("death", "Death", "#1B2A3B"),
    ]
    FOLD_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]
    EV_ALPHA, CEN_ALPHA = 0.75, 0.35
    CLIP = 1500   # days — clip TTE for histogram readability

    # ── Per-endpoint TTE figures (folds × splits grid) ────────────────────────
    for ep, ep_name, ep_col in ENDPOINTS:
        fig, axes = plt.subplots(
            len(folds), len(SPLITS),
            figsize=(13, 3.2 * len(folds)),
            sharex=False, sharey=False,
        )
        if len(folds) == 1:
            axes = [axes]
        fig.suptitle(f"{ep_name} — TTE distributions by fold & split",
                     fontsize=11, fontweight="bold")

        for ri, fold in enumerate(folds):
            split_col = f"split0_fold{fold}"
            if split_col not in df_full.columns:
                continue

            if ep == "acr":
                recs_by_split = _build_acr_records(df_full, fold, split_col)
            else:
                recs_by_split = _build_surv_records(df_full, fold, split_col, ep)

            for ci, sp in enumerate(SPLITS):
                ax  = axes[ri][ci]
                recs = recs_by_split.get(sp, [])

                ev_ttes  = [min(r["tte"], CLIP) for r in recs if r["event"] == 1]
                cen_ttes = [min(r["tte"], CLIP) for r in recs if r["event"] == 0]
                bins     = np.linspace(0, CLIP, 40)

                if ev_ttes:
                    ax.hist(ev_ttes,  bins=bins, color=ep_col,   alpha=EV_ALPHA,
                            label=f"event (n={len(ev_ttes)})",  density=False)
                if cen_ttes:
                    ax.hist(cen_ttes, bins=bins, color="#AAAAAA", alpha=CEN_ALPHA,
                            label=f"censored (n={len(cen_ttes)})", density=False)

                ax.set_title(f"Fold {fold} — {sp}", fontsize=8)
                ax.set_xlabel("TTE (days)", fontsize=7)
                ax.set_ylabel("Count",      fontsize=7)
                ax.tick_params(labelsize=6.5)
                ax.legend(fontsize=6, loc="upper right")
                ax.spines[["top","right"]].set_visible(False)
                if ev_ttes:
                    ax.axvline(np.median(ev_ttes), color=ep_col, lw=1.2,
                               linestyle="--", alpha=0.8)

        plt.tight_layout()
        _save(fig, out / f"fig_tte_{ep}")

    # ── Label / event count figure (all endpoints, all folds × splits) ────────
    n_ep = len(ENDPOINTS)
    fig, axes = plt.subplots(n_ep, len(SPLITS),
                              figsize=(13, 3.5 * n_ep), sharey=False)
    fig.suptitle("Event vs Censored counts by endpoint, fold & split",
                 fontsize=11, fontweight="bold")

    for ri, (ep, ep_name, ep_col) in enumerate(ENDPOINTS):
        for ci, sp in enumerate(SPLITS):
            ax = axes[ri][ci]
            n_ev_list, n_cen_list, fold_labels = [], [], []

            for fold in folds:
                split_col = f"split0_fold{fold}"
                if split_col not in df_full.columns:
                    continue
                if ep == "acr":
                    recs_by_split = _build_acr_records(df_full, fold, split_col)
                else:
                    recs_by_split = _build_surv_records(df_full, fold, split_col, ep)
                recs = recs_by_split.get(sp, [])
                n_ev_list.append(sum(1 for r in recs if r["event"] == 1))
                n_cen_list.append(sum(1 for r in recs if r["event"] == 0))
                fold_labels.append(f"F{fold}")

                # For ACR also show unlabelled count from raw df
                if ep == "acr":
                    df_sp = df_full[df_full[split_col] == sp]
                    n_none = int(df_sp["acr_grade"].apply(
                        lambda g: isinstance(g, str) and not (
                            g.strip().startswith("A0") or
                            g.strip().startswith("A1") or
                            g.strip().startswith("A2")
                        ) or not isinstance(g, str)).sum())

            x = np.arange(len(fold_labels))
            ax.bar(x, n_ev_list,  color=ep_col,   alpha=0.80, label="event",    width=0.4)
            ax.bar(x, n_cen_list, color="#CCCCCC", alpha=0.70, label="censored",
                   width=0.4, bottom=n_ev_list)
            ax.set_xticks(x); ax.set_xticklabels(fold_labels, fontsize=7)
            ax.set_title(f"{ep_name} — {sp}", fontsize=8)
            ax.set_ylabel("n samples", fontsize=7)
            ax.tick_params(labelsize=6.5)
            ax.spines[["top","right"]].set_visible(False)
            if ci == len(SPLITS) - 1:
                ax.legend(fontsize=6)

            # Annotate totals
            for xi, (ne, nc) in enumerate(zip(n_ev_list, n_cen_list)):
                ax.text(xi, ne + nc + 2, str(ne + nc), ha="center",
                        va="bottom", fontsize=5.5, color="#444")

    plt.tight_layout()
    _save(fig, out / "fig_label_counts")

    # ── ACR-specific: label (0/1) distribution ────────────────────────────────
    fig, axes = plt.subplots(len(folds), len(SPLITS),
                              figsize=(13, 3.0 * len(folds)))
    if len(folds) == 1:
        axes = [axes]
    fig.suptitle("ACR classification label distribution (0 / 1) by fold & split",
                 fontsize=11, fontweight="bold")

    for ri, fold in enumerate(folds):
        split_col = f"split0_fold{fold}"
        if split_col not in df_full.columns:
            continue
        for ci, sp in enumerate(SPLITS):
            ax = axes[ri][ci]
            df_sp = df_full[df_full[split_col] == sp].copy()
            df_sp["_lbl"] = df_sp["acr_grade"].apply(
                lambda g: (0 if (isinstance(g, str) and g.strip().startswith("A0"))
                           else 1 if (isinstance(g, str) and
                                      (g.strip().startswith("A1") or g.strip().startswith("A2")))
                           else None))
            n0    = int((df_sp["_lbl"] == 0).sum())
            n1    = int((df_sp["_lbl"] == 1).sum())
            n_nan = int(df_sp["_lbl"].isna().sum())
            bars  = ax.bar(["A0 (0)", "A1/A2 (1)", "No label"],
                           [n0, n1, n_nan],
                           color=["#3A86FF", "#E63946", "#AAAAAA"],
                           alpha=0.80, width=0.55)
            for b, v in zip(bars, [n0, n1, n_nan]):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v),
                        ha="center", va="bottom", fontsize=6.5)
            ax.set_title(f"Fold {fold} — {sp}", fontsize=8)
            ax.set_ylabel("n samples", fontsize=7)
            ax.tick_params(labelsize=6.5)
            ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, out / "fig_acr_label_dist")

    print(f"  TTE/label distribution figures saved to {out}")


def _concordance_index(times, events, scores) -> float:
    """Harrell's C-index: fraction of concordant pairs among comparable pairs."""
    times   = np.array(times,  dtype=float)
    events  = np.array(events, dtype=float)
    scores  = np.array(scores, dtype=float)
    conc = disc = tied = 0
    for i in range(len(times)):
        if events[i] != 1:
            continue
        for j in range(len(times)):
            if times[j] <= times[i]:
                continue
            if scores[i] > scores[j]:   conc += 1
            elif scores[i] < scores[j]: disc += 1
            else:                       tied += 1
    total = conc + disc + tied
    return (conc + 0.5 * tied) / total if total > 0 else float("nan")


MOD_COLS  = ["has_HE", "has_BAL", "has_CT", "has_Clinical"]
MOD_SHORT = ["HE",     "BAL",     "CT",     "Clinical"]


def _get_modality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return binary modality presence columns, filling missing as 0."""
    out = pd.DataFrame(index=df.index)
    for col in MOD_COLS:
        out[col] = df[col].fillna(0).astype(float) if col in df.columns else 0.0
    out["n_mods"] = out[MOD_COLS].sum(axis=1)
    return out


def task_modality_baseline(
    splits_csv:   Path,
    results_base: Path,
    output_dir:   Path,
    folds:        List[int],
):
    """
    Fit modality-presence Cox (survival) and logistic regression (ACR cls) baselines.

    For each endpoint × fold:
      - Train Cox/LogReg on modality presence indicators (train split)
      - Evaluate C-index / AUC on test split
      - Compare against best v7 model variant from JSON metrics

    Figures produced:
      fig_modality_baseline_ci.pdf   — C-index: baseline vs model per fold/endpoint
      fig_modality_baseline_auc.pdf  — AUC: ACR cls baseline vs model
      fig_modality_coefs.pdf         — Cox/LogReg coefficients (which mods are predictive)
      modality_baseline.csv          — all numeric results
    """
    plt.rcParams.update(NATURE_RC)
    warnings.filterwarnings("ignore")
    out = output_dir
    out.mkdir(parents=True, exist_ok=True)

    df_full = pd.read_csv(splits_csv)
    df_full["anchor_dt"] = pd.to_datetime(df_full["anchor_dt"])

    ENDPOINTS = [
        ("acr_surv",  "ACR TTE",  "#E63946", "acr"),
        ("clad",      "CLAD",     "#FB8500", "clad"),
        ("death",     "Death",    "#1B2A3B", "death"),
    ]
    ACR_CLS_COLOR = "#3A86FF"

    all_rows = []   # for CSV

    # ── Per-fold fitting ──────────────────────────────────────────────────────
    # Store: {ep_key: {fold: {"ci_base": float, "ci_model": float, "coefs": dict}}}
    ep_results:  Dict[str, Dict] = {ep[0]: {} for ep in ENDPOINTS}
    acr_cls_res: Dict[int, dict] = {}   # fold → {auc_base, auc_model, bacc_base, bacc_model}

    for fold in folds:
        split_col = f"split0_fold{fold}"
        if split_col not in df_full.columns:
            continue

        feat_all = _get_modality_features(df_full)

        # ── Survival endpoints ────────────────────────────────────────────────
        for ep_key, ep_name, ep_col, ep_raw in ENDPOINTS:
            # Build TTE records using same logic as training
            if ep_raw == "acr":
                recs = _build_acr_records(df_full, fold, split_col)
            else:
                recs = _build_surv_records(df_full, fold, split_col, ep_raw)

            # Collect train / test DataFrames with features
            def _to_df(rec_list, orig_df, split_col, split_name):
                idx = orig_df.index[orig_df[split_col] == split_name].tolist()
                # Map by position — recs are built in df row order for this split
                rows = []
                for r in rec_list:
                    rows.append(r)
                if not rows:
                    return pd.DataFrame()
                return pd.DataFrame(rows)

            train_recs = recs.get("train", [])
            test_recs  = recs.get("test",  [])
            if len(train_recs) < 10 or len(test_recs) < 5:
                continue

            # Attach modality features by re-iterating the df in the same order
            def _add_features(rec_list, split_name):
                sp_df  = df_full[df_full[split_col] == split_name].reset_index(drop=True)
                feats  = _get_modality_features(sp_df)
                # rec_list and sp_df rows correspond 1-to-1 after filtering
                # Some records are dropped (e.g. post-CLAD) so we can't assume 1:1
                # Re-build with stem as key
                stem_to_feat = {
                    str(Path(str(row["file"])).stem): feats.iloc[i].to_dict()
                    for i, (_, row) in enumerate(sp_df.iterrows())
                }
                out = []
                for i, r in enumerate(rec_list):
                    # We don't have stem in rec — use position index from split df
                    pass
                # Fallback: rebuild features from the same row filtering logic
                return feats

            # Simpler: rebuild from df directly (no stem needed)
            def _build_xy_surv(split_name):
                if ep_raw == "acr":
                    recs_sp = _build_acr_records(df_full, fold, split_col).get(split_name, [])
                else:
                    recs_sp = _build_surv_records(df_full, fold, split_col, ep_raw).get(split_name, [])

                sp_df   = df_full[df_full[split_col] == split_name].copy()
                feats_sp = _get_modality_features(sp_df).values  # (N_sp, n_feats)

                # recs_sp length <= sp_df length (some rows dropped for CLAD)
                # We need modality features for each rec, but recs lose the original index.
                # Workaround: re-iterate sp_df in same order as _build_*_records and collect.
                sp_df_iter = sp_df.reset_index(drop=True)

                if ep_raw == "acr":
                    rows_out = []
                    pos_dates: Dict = {}
                    for _, row in df_full[df_full["acr_grade"].apply(
                            lambda g: isinstance(g, str) and
                            (g.strip().startswith("A1") or g.strip().startswith("A2")))].iterrows():
                        pos_dates.setdefault(row["patient_id"], []).append(row["anchor_dt"])
                    last_dt = df_full.groupby("patient_id")["anchor_dt"].max().to_dict()

                    for i, row in sp_df_iter.iterrows():
                        g = row.get("acr_grade")
                        if isinstance(g, str) and g.strip().startswith("A0"):   lbl = 0
                        elif isinstance(g, str) and (g.strip().startswith("A1") or
                                                      g.strip().startswith("A2")): lbl = 1
                        else: continue
                        pid = row["patient_id"]; t = row["anchor_dt"]
                        if lbl == 1:
                            tte, ev = 0.0, 1
                        else:
                            future = sorted([d for d in pos_dates.get(pid, []) if d > t])
                            tte = float((future[0] - t).days) if future else float(max((last_dt.get(pid, t) - t).days, 0))
                            ev  = 1 if future else 0
                        frow = _get_modality_features(row.to_frame().T).iloc[0].to_dict()
                        rows_out.append({**frow, "tte": tte, "event": ev})
                    return pd.DataFrame(rows_out) if rows_out else pd.DataFrame()

                else:
                    time_col  = f"{ep_raw}_days"
                    event_col = f"{ep_raw}_status"
                    study_end = df_full["anchor_dt"].max()
                    rows_out  = []
                    for i, row in sp_df_iter.iterrows():
                        try:
                            t = float(row.get(time_col,  float("nan")))
                            e = float(row.get(event_col, float("nan")))
                        except (TypeError, ValueError):
                            continue
                        if math.isnan(e): continue
                        if e == 1.0:
                            if math.isnan(t) or t <= 0: continue
                        else:
                            proxy = float("nan")
                            if ep_raw == "clad":
                                try: proxy = float(row.get("death_days", float("nan")))
                                except (TypeError, ValueError): pass
                            if math.isnan(proxy) or proxy <= 0:
                                try: proxy = float((study_end - row["anchor_dt"]).days)
                                except Exception: continue
                            if math.isnan(proxy) or proxy <= 0: continue
                            t = proxy
                        frow = _get_modality_features(row.to_frame().T).iloc[0].to_dict()
                        rows_out.append({**frow, "tte": t, "event": int(e)})
                    return pd.DataFrame(rows_out) if rows_out else pd.DataFrame()

            df_tr = _build_xy_surv("train")
            df_te = _build_xy_surv("test")
            if df_tr.empty or df_te.empty or df_tr["event"].sum() < 3:
                continue

            feat_cols = MOD_COLS + ["n_mods"]
            X_tr = df_tr[feat_cols].values.astype(float)
            X_te = df_te[feat_cols].values.astype(float)

            # ── Fit Cox if lifelines available, else use linear score ────────
            coefs = {}
            if HAS_LIFELINES and df_tr["event"].sum() >= 5:
                try:
                    cph = CoxPHFitter(penalizer=0.1)
                    cph.fit(df_tr[MOD_COLS + ["tte", "event"]],
                            duration_col="tte", event_col="event")
                    scores_te = cph.predict_partial_hazard(df_te[MOD_COLS]).values
                    for c, coef in zip(MOD_COLS, cph.params_.values[:len(MOD_COLS)]):
                        coefs[c] = float(coef)
                except Exception as ex:
                    print(f"  [Cox warn] {ep_key} fold {fold}: {ex}")
                    scores_te = X_te @ np.ones(len(feat_cols))
            else:
                # Simple: weighted sum (treat each presence as +1 risk unit)
                scores_te = X_te[:, :len(MOD_COLS)].sum(axis=1)
                for c in MOD_COLS:
                    coefs[c] = 1.0

            ci_base = _concordance_index(df_te["tte"].values,
                                          df_te["event"].values, scores_te)

            # ── Load best model C-index from JSON ────────────────────────────
            if ep_raw == "acr":
                res_dir = results_base / "results_mm_abmil_v7_alt_p1"
            else:
                res_dir = results_base / f"results_mm_abmil_v7_{ep_raw}"
            ci_model = float("nan")
            if res_dir.exists():
                fold_dir = res_dir / f"split0_fold{fold}"
                best_ci  = float("nan")
                for mf in sorted(fold_dir.glob("metrics_*.json")) if fold_dir.exists() else []:
                    try:
                        m = json.load(open(mf))
                        ci_v = m.get("test", {}).get("c_index", float("nan"))
                        if not math.isnan(float(ci_v)):
                            best_ci = float(ci_v) if math.isnan(best_ci) else max(best_ci, float(ci_v))
                    except Exception:
                        pass
                ci_model = best_ci

            ep_results[ep_key][fold] = {
                "ci_base":  ci_base,
                "ci_model": ci_model,
                "coefs":    coefs,
            }
            all_rows.append({"endpoint": ep_key, "ep_name": ep_name,
                              "fold": fold, "metric": "c_index",
                              "baseline": ci_base, "model": ci_model,
                              **{f"coef_{k}": v for k, v in coefs.items()}})
            print(f"  [{ep_key}] fold {fold}: CI_base={ci_base:.3f}  CI_model={ci_model:.3f}")

        # ── ACR classification baseline ───────────────────────────────────────
        def _build_xy_cls(split_name):
            sp_df = df_full[df_full[split_col] == split_name].copy()
            rows_out = []
            for _, row in sp_df.iterrows():
                g = row.get("acr_grade")
                if isinstance(g, str) and g.strip().startswith("A0"):   lbl = 0
                elif isinstance(g, str) and (g.strip().startswith("A1") or
                                              g.strip().startswith("A2")): lbl = 1
                else: continue
                frow = _get_modality_features(row.to_frame().T).iloc[0].to_dict()
                rows_out.append({**frow, "label": lbl})
            return pd.DataFrame(rows_out) if rows_out else pd.DataFrame()

        df_tr_cls = _build_xy_cls("train")
        df_te_cls = _build_xy_cls("test")

        auc_base = bacc_base = auc_model = bacc_model = float("nan")
        cls_coefs: Dict[str, float] = {}

        if (not df_tr_cls.empty and not df_te_cls.empty and
                df_tr_cls["label"].nunique() == 2 and df_te_cls["label"].nunique() == 2):
            X_tr_c = df_tr_cls[MOD_COLS].values.astype(float)
            X_te_c = df_te_cls[MOD_COLS].values.astype(float)
            y_tr_c = df_tr_cls["label"].values
            y_te_c = df_te_cls["label"].values
            try:
                clf = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0)
                clf.fit(X_tr_c, y_tr_c)
                probs = clf.predict_proba(X_te_c)[:, 1]
                preds = clf.predict(X_te_c)
                auc_base  = float(roc_auc_score(y_te_c, probs))
                bacc_base = float(balanced_accuracy_score(y_te_c, preds))
                for c, coef in zip(MOD_COLS, clf.coef_[0]):
                    cls_coefs[c] = float(coef)
            except Exception as ex:
                print(f"  [LR warn] ACR cls fold {fold}: {ex}")

        # Load best model AUC from ACR alt results
        res_dir_acr = results_base / "results_mm_abmil_v7_alt_p1"
        if res_dir_acr.exists():
            fold_dir = res_dir_acr / f"split0_fold{fold}"
            best_auc = best_bacc = float("nan")
            for mf in sorted(fold_dir.glob("metrics_*.json")) if fold_dir.exists() else []:
                try:
                    m = json.load(open(mf))
                    te = m.get("test", {})
                    v_auc  = te.get("auc",  float("nan"))
                    v_bacc = te.get("bacc", float("nan"))
                    if not math.isnan(float(v_bacc)):
                        if math.isnan(best_bacc) or float(v_bacc) > best_bacc:
                            best_bacc = float(v_bacc); best_auc = float(v_auc)
                except Exception:
                    pass
            auc_model = best_auc; bacc_model = best_bacc

        acr_cls_res[fold] = {
            "auc_base": auc_base,   "auc_model": auc_model,
            "bacc_base": bacc_base, "bacc_model": bacc_model,
            "coefs": cls_coefs,
        }
        all_rows.append({"endpoint": "acr_cls", "ep_name": "ACR Cls",
                          "fold": fold, "metric": "auc",
                          "baseline": auc_base, "model": auc_model,
                          **{f"coef_{k}": v for k, v in cls_coefs.items()}})
        print(f"  [acr_cls] fold {fold}: AUC_base={auc_base:.3f}  AUC_model={auc_model:.3f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(all_rows)
    csv_path = out / "modality_baseline.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\n  Baseline CSV saved → {csv_path}")

    # ── Figure 1: C-index baseline vs model for all survival endpoints ────────
    surv_eps = [(k, n, c) for k, n, c, _ in ENDPOINTS]
    n_ep = len(surv_eps)

    fig, axes = plt.subplots(1, n_ep, figsize=(5 * n_ep, 5), sharey=True)
    fig.suptitle("Modality-presence Baseline vs Model — C-index (test set)",
                 fontsize=11, fontweight="bold")

    for ax, (ep_key, ep_name, ep_col) in zip(axes, surv_eps):
        res = ep_results.get(ep_key, {})
        fold_ids = sorted(res.keys())
        x = np.arange(len(fold_ids))

        base_vals  = [res[f]["ci_base"]  for f in fold_ids]
        model_vals = [res[f]["ci_model"] for f in fold_ids]

        w = 0.35
        ax.bar(x - w/2, base_vals,  w, color="#AAAAAA", alpha=0.85, label="Modality baseline")
        ax.bar(x + w/2, model_vals, w, color=ep_col,    alpha=0.85, label="Best v7 model")

        # Mean lines
        bv = [v for v in base_vals  if not math.isnan(v)]
        mv = [v for v in model_vals if not math.isnan(v)]
        if bv:  ax.axhline(np.mean(bv),  color="#888888", lw=1.5, linestyle="--")
        if mv:  ax.axhline(np.mean(mv),  color=ep_col,    lw=1.5, linestyle="--", alpha=0.7)

        ax.axhline(0.5, color="#CCCCCC", lw=1.0, linestyle=":")
        ax.set_xticks(x); ax.set_xticklabels([f"Fold {f}" for f in fold_ids], fontsize=8)
        ax.set_ylim(0.3, 0.9)
        ax.set_ylabel("C-index"); ax.set_title(ep_name, fontweight="bold")
        ax.legend(fontsize=7)

        # Annotate improvement
        for xi, (bv_, mv_) in enumerate(zip(base_vals, model_vals)):
            if not math.isnan(bv_) and not math.isnan(mv_):
                delta = mv_ - bv_
                col   = "#228833" if delta > 0 else "#CC3333"
                ax.text(xi + w/2, mv_ + 0.01, f"{delta:+.3f}",
                        ha="center", va="bottom", fontsize=6.5, color=col)

    plt.tight_layout()
    _save(fig, out / "fig_modality_baseline_ci")

    # ── Figure 2: ACR classification AUC baseline vs model ───────────────────
    fold_ids = sorted(acr_cls_res.keys())
    if fold_ids:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        fig.suptitle("Modality-presence Baseline vs Model — ACR Classification",
                     fontsize=11, fontweight="bold")
        for ax, metric, label in zip(axes, ["auc", "bacc"], ["AUC-ROC", "BAcc"]):
            x = np.arange(len(fold_ids))
            bvals = [acr_cls_res[f][f"{metric}_base"]  for f in fold_ids]
            mvals = [acr_cls_res[f][f"{metric}_model"] for f in fold_ids]
            w = 0.35
            ax.bar(x - w/2, bvals, w, color="#AAAAAA",   alpha=0.85, label="Modality baseline")
            ax.bar(x + w/2, mvals, w, color=ACR_CLS_COLOR, alpha=0.85, label="Best v7 model")
            bm = [v for v in bvals if not math.isnan(v)]
            mm = [v for v in mvals if not math.isnan(v)]
            if bm: ax.axhline(np.mean(bm), color="#888888",     lw=1.5, linestyle="--")
            if mm: ax.axhline(np.mean(mm), color=ACR_CLS_COLOR, lw=1.5, linestyle="--", alpha=0.7)
            ax.set_xticks(x); ax.set_xticklabels([f"Fold {f}" for f in fold_ids], fontsize=8)
            ax.set_ylim(0.3, 1.0); ax.set_ylabel(label); ax.set_title(f"ACR — {label}", fontweight="bold")
            ax.legend(fontsize=7)
            for xi, (b_, m_) in enumerate(zip(bvals, mvals)):
                if not math.isnan(b_) and not math.isnan(m_):
                    delta = m_ - b_
                    ax.text(xi + w/2, m_ + 0.01, f"{delta:+.3f}",
                            ha="center", va="bottom", fontsize=6.5,
                            color="#228833" if delta > 0 else "#CC3333")
        plt.tight_layout()
        _save(fig, out / "fig_modality_baseline_auc")

    # ── Figure 3: Coefficient heatmap (which modality is predictive) ─────────
    all_endpoints_for_coef = [
        ("acr_surv", "ACR TTE"),
        ("clad",     "CLAD"),
        ("death",    "Death"),
        ("acr_cls",  "ACR Cls"),
    ]
    coef_data: Dict = {}
    for ep_key, ep_name in all_endpoints_for_coef:
        if ep_key in ep_results:
            for f, res in ep_results[ep_key].items():
                for mod, val in res.get("coefs", {}).items():
                    coef_data.setdefault((ep_name, mod), []).append(val)
        elif ep_key == "acr_cls":
            for f, res in acr_cls_res.items():
                for mod, val in res.get("coefs", {}).items():
                    coef_data.setdefault((ep_name, mod), []).append(val)

    if coef_data:
        ep_names  = list(dict.fromkeys(k[0] for k in coef_data))
        mod_names = [c.replace("has_", "") for c in MOD_COLS]
        mat = np.full((len(ep_names), len(MOD_COLS)), float("nan"))
        for ei, ep_n in enumerate(ep_names):
            for mi, mc in enumerate(MOD_COLS):
                vals = coef_data.get((ep_n, mc), [])
                if vals:
                    mat[ei, mi] = float(np.mean(vals))

        fig, ax = plt.subplots(figsize=(7, 3.5))
        vmax = np.nanmax(np.abs(mat)); vmax = vmax if vmax > 0 else 1.0
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(MOD_COLS)))
        ax.set_xticklabels(mod_names, fontsize=9)
        ax.set_yticks(range(len(ep_names)))
        ax.set_yticklabels(ep_names, fontsize=9)
        plt.colorbar(im, ax=ax, label="Coefficient (mean across folds)")
        ax.set_title("Modality-presence coefficients\n(positive = higher risk / ACR+)",
                     fontsize=10, fontweight="bold")
        for ei in range(len(ep_names)):
            for mi in range(len(MOD_COLS)):
                v = mat[ei, mi]
                if not math.isnan(v):
                    ax.text(mi, ei, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if abs(v) > 0.4 * vmax else "black")
        plt.tight_layout()
        _save(fig, out / "fig_modality_coefs")

    print(f"\n  Modality baseline figures saved to {out}")


def parse_args():
    p = argparse.ArgumentParser(description="CLAD & Death survival benchmark analysis")
    p.add_argument("--results_base", type=str,
                   default="/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil")
    p.add_argument("--output_dir", type=str,
                   default="/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/analysis/v7_surv_endpoints")
    p.add_argument("--splits_csv", type=str,
                   default="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_analysis(
        results_base=Path(args.results_base),
        output_dir=out,
        folds=args.folds,
    )
    task_tte_distributions(
        splits_csv=Path(args.splits_csv),
        output_dir=out,
        folds=args.folds,
    )
    task_modality_baseline(
        splits_csv=Path(args.splits_csv),
        results_base=Path(args.results_base),
        output_dir=out,
        folds=args.folds,
    )
