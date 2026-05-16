#!/usr/bin/env python3
"""
visualize_modality_bias.py
==========================
Analyses whether modality *presence* (which tests were ordered) is itself
a biased signal of disease, independent of the modality content.

BAL/biopsy ordered preferentially when doctors suspect rejection →
has_BAL=True encodes clinical suspicion, not biology.

Produces:
  1. presence_rates.png      Modality presence rates: event vs non-event, ratio annotated
  2. odds_ratios.png         Forest plot: OR + 95% CI, each modality × endpoint
  3. temporal_trend.png      Modality presence rate in time windows before event
  4. modality_only_auc.png   AUC from presence-only logistic regression (leakage)
  5. combo_bias.png          Modality combination rates by event status
  6. presence_heatmap.png    Per-patient heatmap: which modalities at each timepoint

Usage:
    python visualize_modality_bias.py \
        --csv  /path/to/multimodal_splits_nested_cv.csv \
        --outdir ./modality_bias_viz
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

warnings.filterwarnings("ignore")

ACR_BIOPSIES_CSV = "/home/aih/dinesh.haridoss/chicago/ACR_biopsies_clean_dates_fixed (1).csv"

MODALITIES = ["HE", "BAL", "CT", "Clinical"]
MOD_COLS   = {"HE": "#1F77B4", "BAL": "#D62728", "CT": "#2CA02C", "Clinical": "#9467BD"}
EP_COLS    = {"ACR": "#D62728", "CLAD": "#FF7F0E", "Death": "#2C2C2C"}
C_NEG      = "#1F77B4"
C_CENS     = "#AAAAAA"


# ── data loading ──────────────────────────────────────────────────────────────

def load_and_label(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["anchor_dt"])
    df = df.sort_values(["patient_id", "anchor_dt"]).reset_index(drop=True)
    first = df.groupby("patient_id")["anchor_dt"].transform("min")
    df["days_from_first"] = (df["anchor_dt"] - first).dt.days

    # Transplant date → days since transplant
    try:
        acr_meta = pd.read_csv(ACR_BIOPSIES_CSV, parse_dates=["Transplant Date"])
        tx = (acr_meta.drop_duplicates("Record id")
              [["Record id", "Transplant Date"]]
              .rename(columns={"Record id": "patient_id", "Transplant Date": "transplant_dt"}))
        df = df.merge(tx, on="patient_id", how="left")
        df["days_since_transplant"] = (df["anchor_dt"] - df["transplant_dt"]).dt.days
    except Exception:
        df["transplant_dt"] = pd.NaT
        df["days_since_transplant"] = np.nan

    df["combo"] = df.apply(
        lambda r: "+".join(m for m in MODALITIES if r.get(f"has_{m}", False)) or "None",
        axis=1)

    df["acr_bin"] = (df["label"] == 1).astype(float)
    df.loc[df["label"].isna(), "acr_bin"] = np.nan

    # ACR TTE: for each sample, days to next ACR+ sample for that patient.
    # Samples already ACR+ get tte=0 / event=1 (they are the event).
    # Samples with no future ACR+ get censored at last follow-up.
    acr_tte_list, acr_event_list = [], []
    for pid, grp in df.groupby("patient_id", sort=False):
        grp = grp.sort_values("anchor_dt")
        acr_dates = grp.loc[grp["label"] == 1, "anchor_dt"].values
        for _, row in grp.iterrows():
            if row["label"] == 1:
                acr_tte_list.append(0.0)
                acr_event_list.append(1.0)
            else:
                future_acr = acr_dates[acr_dates > row["anchor_dt"]]
                if len(future_acr):
                    tte = (pd.Timestamp(future_acr.min()) - row["anchor_dt"]).days
                    acr_tte_list.append(float(tte))
                    acr_event_list.append(1.0)
                else:
                    last = grp["anchor_dt"].max()
                    tte = (last - row["anchor_dt"]).days
                    acr_tte_list.append(float(tte))
                    acr_event_list.append(0.0)
    df["acr_tte"]   = acr_tte_list
    df["acr_event"] = acr_event_list

    df["clad_bin"] = np.where(
        (df["clad_status"] == 1) & (df["clad_days"] > 0), 1.0,
        np.where((df["clad_status"] == 0) | (df["clad_days"].isna()), 0.0, np.nan))
    df["clad_tte"] = np.where(df["clad_days"] > 0, df["clad_days"], np.nan)

    df["death_bin"] = np.where(
        df["death_status"] == 1, 1.0,
        np.where(df["death_status"] == 0, 0.0, np.nan))
    df["death_tte"] = df["death_days"]
    return df


def odds_ratio(n_pos_w, n_total_pos, n_pos_wo, n_total_neg):
    a, b = n_pos_w,  n_total_pos - n_pos_w
    c, d = n_pos_wo, n_total_neg - n_pos_wo
    if any(v == 0 for v in [a, b, c, d]):
        return np.nan, np.nan, np.nan
    OR = (a/b) / (c/d)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    return OR, np.exp(np.log(OR) - 1.96*se), np.exp(np.log(OR) + 1.96*se)


# ── Figure 1: Presence rates ──────────────────────────────────────────────────

def fig_presence_rates(df: pd.DataFrame, outdir: Path):
    endpoints = [
        ("ACR",   "acr_bin",   "ACR+",          "ACR−"),
        ("CLAD",  "clad_bin",  "CLAD in future", "Censored"),
        ("Death", "death_bin", "Died",           "Censored alive"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Modality Presence Rates by Event Status\n"
        "Ratio annotated above bars  —  red bold = ratio >1.5 (strong bias)",
        fontsize=12, fontweight="bold",
    )
    for ax, (ep, col, pos_l, neg_l) in zip(axes, endpoints):
        sub = df[df[col].notna()]
        pos = sub[sub[col] == 1]
        neg = sub[sub[col] == 0]
        x, w = np.arange(len(MODALITIES)), 0.35
        rp = [pos[f"has_{m}"].mean()*100 for m in MODALITIES]
        rn = [neg[f"has_{m}"].mean()*100 for m in MODALITIES]
        ax.bar(x-w/2, rp, w, color=EP_COLS[ep], edgecolor="white", lw=0.5,
               label=f"{pos_l} (n={len(pos)})")
        ax.bar(x+w/2, rn, w, color=C_NEG,       edgecolor="white", lw=0.5, alpha=0.6,
               label=f"{neg_l} (n={len(neg)})")
        for xi, (p_, n_) in enumerate(zip(rp, rn)):
            if n_ > 0:
                r = p_/n_
                c = "#B00000" if r > 1.5 else ("k" if r > 1.1 else "#999999")
                fw = "bold" if r > 1.5 else "normal"
                ax.text(xi, max(p_, n_)+3, f"{r:.2f}×",
                        ha="center", fontsize=9, color=c, fontweight=fw)
        ax.set_xticks(x); ax.set_xticklabels(MODALITIES, fontsize=11)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Samples with modality (%)", fontsize=10)
        ax.set_title(ep, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    p = outdir / "presence_rates.pdf"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Figure 2: Odds ratio forest plot ─────────────────────────────────────────

def fig_odds_ratios(df: pd.DataFrame, outdir: Path):
    endpoints = {"ACR": "acr_bin", "CLAD": "clad_bin", "Death": "death_bin"}

    # Build (modality, endpoint) rows
    records = []
    for ep, col in endpoints.items():
        sub = df[df[col].notna()]
        pos, neg = sub[sub[col]==1], sub[sub[col]==0]
        for m in MODALITIES:
            OR, lo, hi = odds_ratio(pos[f"has_{m}"].sum(), len(pos),
                                    neg[f"has_{m}"].sum(), len(neg))
            records.append({"ep": ep, "mod": m, "OR": OR, "lo": lo, "hi": hi})
    ordf = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("#FAFAFA")
    fig.suptitle(
        "Odds Ratio: modality presence in event vs non-event samples\n"
        "OR > 1 → modality more common when disease present (selection bias)",
        fontsize=12, fontweight="bold",
    )

    n_ep   = len(endpoints)
    ep_list = list(endpoints.keys())
    pitch  = n_ep + 1.5      # vertical space per modality
    for mi, m in enumerate(MODALITIES):
        y_base = mi * pitch
        for ei, ep in enumerate(ep_list):
            row = ordf[(ordf["ep"]==ep) & (ordf["mod"]==m)]
            if len(row) == 0 or pd.isna(row.iloc[0]["OR"]):
                continue
            OR, lo, hi = float(row.iloc[0]["OR"]), float(row.iloc[0]["lo"]), float(row.iloc[0]["hi"])
            y = y_base + ei
            ax.plot([lo, hi], [y, y], color=EP_COLS[ep], lw=2, solid_capstyle="round")
            ax.scatter([OR], [y], color=EP_COLS[ep], s=55, zorder=4)
            ax.text(hi*1.05, y, f"{OR:.2f}", fontsize=9, va="center", color=EP_COLS[ep])

        # Modality label centred on its group
        y_centre = y_base + (n_ep-1)/2
        ax.text(-0.02, y_centre, m, fontsize=12, fontweight="bold", ha="right", va="center",
                transform=ax.get_yaxis_transform())
        # light separator
        if mi < len(MODALITIES)-1:
            ax.axhline(y_base + n_ep + 0.5, color="#DDDDDD", lw=0.8, zorder=0)

    ax.axvline(1.0, color="#666666", lw=1.2, ls="--", alpha=0.6, label="OR=1 (no bias)")
    ax.set_xscale("log")
    ax.set_xlabel("Odds Ratio (log scale)", fontsize=11)
    ax.set_yticks([])
    ax.grid(True, alpha=0.2, axis="x", which="both")
    legend_handles = [mpatches.Patch(color=EP_COLS[ep], label=ep) for ep in ep_list] + \
                     [Line2D([0],[0], color="#666666", ls="--", lw=1.2, label="OR=1")]
    ax.legend(handles=legend_handles, fontsize=10, loc="lower right")

    fig.tight_layout()
    p = outdir / "odds_ratios.pdf"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Figure 3: Temporal trend ──────────────────────────────────────────────────

def fig_temporal_trend(df: pd.DataFrame, outdir: Path):
    """
    For each endpoint, show modality presence rate as a function of
    time-to-event window (only event patients). If doctors order tests
    as disease approaches, the bar rises towards the 0–30d window.
    """
    windows = [(0,30,"0–30d"),(30,90,"30–90d"),(90,180,"90–180d"),
               (180,365,"180–365d"),(365,9999,"365d+")]
    # ACR TTE = days to next ACR+ sample (computed in load_and_label)
    ep_info = {
        "ACR":   ("acr_event",  "acr_tte"),
        "CLAD":  ("clad_bin",   "clad_tte"),
        "Death": ("death_bin",  "death_tte"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Modality Presence Rate vs Time to Event  (event patients only)\n"
        "Rising rate as event approaches = clinically biased ordering",
        fontsize=12, fontweight="bold",
    )
    for ax, (ep_label, (bin_col, tte_col)) in zip(axes, ep_info.items()):
        # For ACR: exclude samples that ARE the event (tte==0); keep pre-event samples
        if ep_label == "ACR":
            event_rows = df[(df[bin_col]==1) & df[tte_col].notna() & (df[tte_col]>0)].copy()
        else:
            event_rows = df[(df[bin_col]==1) & df[tte_col].notna()].copy()
        event_rows["tte"] = event_rows[tte_col]

        x_pos  = np.arange(len(windows))
        w      = 0.8 / len(MODALITIES)
        ns     = []
        for wi, (lo, hi, _) in enumerate(windows):
            ns.append(len(event_rows[(event_rows["tte"]>=lo) & (event_rows["tte"]<hi)]))

        for mi, m in enumerate(MODALITIES):
            rates = []
            for lo, hi, _ in windows:
                win = event_rows[(event_rows["tte"]>=lo) & (event_rows["tte"]<hi)]
                rates.append(win[f"has_{m}"].mean()*100 if len(win) else np.nan)
            offset = (mi - (len(MODALITIES)-1)/2) * w
            ax.bar(x_pos + offset, rates, w, color=MOD_COLS[m],
                   edgecolor="white", lw=0.5, alpha=0.85, label=m)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{lbl}\n(n={n})" for (_,_,lbl),n in zip(windows,ns)], fontsize=9)
        ax.set_ylabel("Samples with modality (%)", fontsize=10)
        ax.set_ylim(0, 105)
        ax.set_title(ep_label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time to event →", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25, axis="y")
        # arrow indicating direction of time
        ax.annotate("", xy=(0, -0.17), xytext=(len(windows)-1, -0.17),
                    xycoords=("data","axes fraction"),
                    textcoords=("data","axes fraction"),
                    arrowprops=dict(arrowstyle="<-", color="#555555", lw=1.2))

    fig.tight_layout()
    p = outdir / "temporal_trend.pdf"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Figure 4: Modality-only AUC ───────────────────────────────────────────────

def fig_modality_only_auc(df: pd.DataFrame, outdir: Path):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
    except ImportError:
        print("  [skip] sklearn not available"); return

    endpoints = {"ACR": "acr_bin", "CLAD": "clad_bin", "Death": "death_bin"}
    features  = [f"has_{m}" for m in MODALITIES]

    aucs, coefs = {}, {}
    for ep, col in endpoints.items():
        sub = df[df[col].notna() & df[features].notna().all(axis=1)]
        X = sub[features].astype(float).values
        y = sub[col].values
        if y.sum() < 5 or (y==0).sum() < 5:
            continue
        fold_aucs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X,y):
            clf = LogisticRegression(max_iter=500, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            fold_aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:,1]))
        aucs[ep] = fold_aucs
        clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(X, y)
        coefs[ep] = clf.coef_[0]

    if not aucs:
        print("  [skip] no endpoints with sufficient data"); return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Information Leakage: AUC from Modality Presence Alone\n"
        "Features = has_HE, has_BAL, has_CT, has_Clinical  (no content, no features)",
        fontsize=12, fontweight="bold",
    )

    # AUC bars
    ep_names = list(aucs.keys())
    means = [np.mean(v) for v in aucs.values()]
    stds  = [np.std(v)  for v in aucs.values()]
    bars = axes[0].bar(ep_names, means, color=[EP_COLS[k] for k in ep_names],
                       edgecolor="white", lw=0.5, width=0.45)
    axes[0].errorbar(ep_names, means, yerr=stds, fmt="none", color="#333333", capsize=6, lw=2)
    axes[0].axhline(0.5, color="#888888", lw=1.2, ls="--", label="Random AUC = 0.5")
    for b, m in zip(bars, means):
        axes[0].text(b.get_x()+b.get_width()/2, m+0.012, f"{m:.3f}",
                     ha="center", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0.3, 1.0)
    axes[0].set_ylabel("Cross-validated AUC (5-fold)", fontsize=11)
    axes[0].set_title("Prediction from presence only", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.25, axis="y")

    # Coefficients
    x = np.arange(len(MODALITIES))
    n_ep = len(coefs)
    w = 0.75 / n_ep
    for ei, (ep, c) in enumerate(coefs.items()):
        offset = (ei - (n_ep-1)/2) * w
        axes[1].bar(x+offset, c, w, label=ep, color=EP_COLS[ep],
                    edgecolor="white", lw=0.5, alpha=0.85)
    axes[1].axhline(0, color="#333333", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(MODALITIES, fontsize=11)
    axes[1].set_ylabel("Logistic regression coefficient", fontsize=10)
    axes[1].set_title("Feature coefficients\n(positive = presence predicts event)",
                      fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    p = outdir / "modality_only_auc.pdf"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Figure 5: Combo bias ─────────────────────────────────────────────────────

def fig_combo_bias(df: pd.DataFrame, outdir: Path):
    endpoints = [
        ("ACR",   "acr_bin",   "ACR+",          "ACR−"),
        ("CLAD",  "clad_bin",  "CLAD in future", "Censored"),
        ("Death", "death_bin", "Died",            "Alive"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    fig.suptitle(
        "Modality Combination Distribution by Event Status\n"
        "Shift in combo proportions between event/non-event = selection bias",
        fontsize=12, fontweight="bold",
    )
    for ax, (ep, col, pos_l, neg_l) in zip(axes, endpoints):
        sub = df[df[col].notna()]
        pos, neg = sub[sub[col]==1], sub[sub[col]==0]
        # keep combos with ≥5 samples in combined
        counts = sub["combo"].value_counts()
        combos = counts[counts >= 5].index.tolist()

        rp = [pos["combo"].value_counts().get(c,0)/max(len(pos),1)*100 for c in combos]
        rn = [neg["combo"].value_counts().get(c,0)/max(len(neg),1)*100 for c in combos]

        x = np.arange(len(combos)); w = 0.38
        ax.bar(x-w/2, rp, w, color=EP_COLS[ep], edgecolor="white", lw=0.5, alpha=0.85,
               label=f"{pos_l} (n={len(pos)})")
        ax.bar(x+w/2, rn, w, color=C_NEG,       edgecolor="white", lw=0.5, alpha=0.6,
               label=f"{neg_l} (n={len(neg)})")
        ax.set_xticks(x)
        ax.set_xticklabels(combos, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("% of samples", fontsize=10)
        ax.set_title(ep, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    p = outdir / "combo_bias.pdf"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Figure 6: Per-patient presence heatmap ───────────────────────────────────

def fig_presence_heatmap(df: pd.DataFrame, outdir: Path, max_patients: int = 40):
    """
    One patient = 4 modality rows stacked (matching patient timelines style).
    Filled dot = modality present. Events overlaid on top.
    """
    has_acr  = df.groupby("patient_id")["acr_bin"].apply(lambda x: (x==1).any())
    patients = (has_acr[has_acr].index.tolist() +
                has_acr[~has_acr].index.tolist())[:max_patients]
    n = len(patients)

    MOD_PITCH = 0.55
    SLOT_H    = len(MODALITIES) * MOD_PITCH + 0.8
    GAP_H     = 0.6
    slot      = SLOT_H + GAP_H
    total_h   = n * slot + 1.0

    fig, ax = plt.subplots(figsize=(30, max(total_h, 10)))
    ax.set_facecolor("#F7F7F7")

    y_ticks, y_labels = [], []

    for pi, pid in enumerate(reversed(patients)):
        y0  = pi * slot
        grp = df[df["patient_id"]==pid].sort_values("days_from_first")
        t_vals = grp["days_from_first"].values
        if len(t_vals) == 0: continue

        # background bar
        ax.plot([t_vals.min(), t_vals.max()], [y0, y0],
                color="#DDDDDD", lw=2.0, zorder=1)

        # CLAD event
        if (grp["clad_status"]==1).any() and grp["clad_tte"].notna().any():
            ref = grp[grp["clad_tte"].notna()].iloc[0]
            cd  = ref["days_from_first"] + ref["clad_tte"]
            ax.plot([cd,cd],
                    [y0-0.15, y0+len(MODALITIES)*MOD_PITCH+0.15],
                    color="#FF7F0E", lw=2.2, alpha=0.85, zorder=4)

        # Death event line (dotted)
        if (grp["death_bin"]==1).any() and grp["death_tte"].notna().any():
            ref = grp[grp["death_tte"].notna()].iloc[0]
            dd  = ref["days_from_first"] + ref["death_tte"]
            ax.plot([dd, dd],
                    [y0-0.3, y0+len(MODALITIES)*MOD_PITCH+0.3],
                    color="#2C2C2C", lw=2.5, ls=":", alpha=0.9, zorder=5)

        # Modality rows
        for mi, m in enumerate(MODALITIES):
            y_mod = y0 + (mi+1)*MOD_PITCH
            ax.plot([t_vals.min(), t_vals.max()], [y_mod, y_mod],
                    color="#EEEEEE", lw=1.0, zorder=1)
            t_yes = grp[grp[f"has_{m}"]==True]["days_from_first"].values
            t_no  = grp[grp[f"has_{m}"]==False]["days_from_first"].values
            if len(t_no):
                ax.scatter(t_no, [y_mod]*len(t_no),
                           color="#E0E0E0", s=18, zorder=2, linewidths=0)
            if len(t_yes):
                ax.scatter(t_yes, [y_mod]*len(t_yes),
                           color=MOD_COLS[m], s=35, zorder=3, linewidths=0)

        # ACR classification lines: red=A1/A2, blue=A0, grey=unknown/censored
        for _, row in grp.iterrows():
            lbl = row["acr_bin"]
            col = "#D62728" if lbl==1.0 else ("#1F77B4" if lbl==0.0 else "#CCCCCC")
            lw  = 2.5 if lbl==1.0 else (1.8 if lbl==0.0 else 1.2)
            ax.plot([row["days_from_first"], row["days_from_first"]],
                    [y0 - 0.2, y0 + 0.2],
                    color=col, lw=lw, solid_capstyle="round", zorder=3)

        y_ticks.append(y0 + len(MODALITIES)*MOD_PITCH/2)
        y_labels.append(pid)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Days from patient's first sample", fontsize=13)
    ax.set_ylim(-0.5, n*slot+0.5)
    ax.grid(True, alpha=0.15, axis="x")

    # Right-axis modality labels
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    if patients:
        y0_ref = (n-1)*slot
        ax2.set_yticks([y0_ref+(mi+1)*MOD_PITCH for mi in range(len(MODALITIES))]+[y0_ref])
        ax2.set_yticklabels(MODALITIES+["ACR"], fontsize=10)
        ax2.tick_params(right=False)

    legend_handles = (
        [mpatches.Patch(color=MOD_COLS[m], label=f"{m} present") for m in MODALITIES] +
        [Line2D([0],[0], color="#D62728", lw=2.5, label="ACR+ (A1/A2)"),
         Line2D([0],[0], color="#1F77B4", lw=1.8, label="ACR− (A0)"),
         Line2D([0],[0], color="#CCCCCC", lw=1.2, label="Unknown/censored"),
         Line2D([0],[0], color="#FF7F0E", lw=2.2, ls="-",  label="CLAD event"),
         Line2D([0],[0], color="#2C2C2C", lw=2.5, ls=":",  label="Death")]
    )
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right", ncol=4, framealpha=0.9)
    ax.set_title(
        f"Modality Presence × Patient Timeline  ·  {n} patients (ACR+ first)\n"
        "Coloured rows = modality present at each timepoint  ·  bottom row = ACR label",
        fontsize=10, pad=8,
    )
    fig.tight_layout()
    p = outdir / "presence_heatmap.pdf"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Summary text ──────────────────────────────────────────────────────────────

def write_summary(df: pd.DataFrame, outdir: Path):
    lines = ["="*65, "  MODALITY SELECTION BIAS SUMMARY", "="*65, ""]
    for ep, col in [("ACR","acr_bin"),("CLAD","clad_bin"),("Death","death_bin")]:
        sub = df[df[col].notna()]
        pos, neg = sub[sub[col]==1], sub[sub[col]==0]
        lines.append(f"{ep}  (event n={len(pos)}, non-event n={len(neg)})")
        for m in MODALITIES:
            rp = pos[f"has_{m}"].mean()
            rn = neg[f"has_{m}"].mean()
            OR, lo, hi = odds_ratio(pos[f"has_{m}"].sum(), len(pos),
                                    neg[f"has_{m}"].sum(), len(neg))
            flag = "  <<< STRONG BIAS" if (not np.isnan(OR) and OR>1.5) else \
                   ("  < moderate" if (not np.isnan(OR) and OR>1.1) else "")
            lines.append(f"  {m:<10}  event={rp:.2f}  non-event={rn:.2f}  "
                         f"OR={OR:.2f} [{lo:.2f}–{hi:.2f}]{flag}")
        lines.append("")
    lines += [
        "Mitigation options:",
        "  1. Include has_HE/BAL/CT/Clinical as explicit input features",
        "     so model can separate content signal from ordering signal.",
        "  2. Stratify train/val/test splits by modality combo.",
        "  3. Evaluate performance separately per modality combo.",
        "  4. For survival: use modality-presence-agnostic baselines.",
    ]
    txt = "\n".join(lines)
    print("\n"+txt)
    (outdir/"modality_bias_summary.txt").write_text(txt)
    print(f"  Saved: modality_bias_summary.txt")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",
        default="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
    ap.add_argument("--outdir",
        default="/home/aih/dinesh.haridoss/chicago_mil/visualizations/modality_bias_viz")
    ap.add_argument("--max_patients", type=int, default=40)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {outdir}")

    print("\nLoading data …")
    df = load_and_label(args.csv)
    print(f"  {len(df)} samples,  {df['patient_id'].nunique()} patients")

    print("\nFig 1: Presence rates …")
    fig_presence_rates(df, outdir)
    print("Fig 2: Odds ratios …")
    fig_odds_ratios(df, outdir)
    print("Fig 3: Temporal trend …")
    fig_temporal_trend(df, outdir)
    print("Fig 4: Modality-only AUC …")
    fig_modality_only_auc(df, outdir)
    print("Fig 5: Combo bias …")
    fig_combo_bias(df, outdir)
    print("Fig 6: Presence heatmap …")
    fig_presence_heatmap(df, outdir, max_patients=args.max_patients)
    write_summary(df, outdir)

    print(f"\nDone. All figures in {outdir}/")


if __name__ == "__main__":
    main()
