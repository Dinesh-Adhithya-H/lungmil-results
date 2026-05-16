#!/usr/bin/env python3
"""
visualize_survival_labels.py — v7 task definitions
====================================================
Visualises the ACR classification + survival labels as used in v7 multitask
training (train_mm_abmil_v7.py).

Task definitions
----------------
Classification label  : A0* → 0,  A1*/A2* → 1,  else → None (excluded)
Survival TTE          : gap-time to next A1/A2 biopsy (per patient)
  • ACR+ biopsy (A1/A2) :  tte = 0,   event = 1   (the biopsy IS the event)
  • ACR- before future ACR: tte = days_to_next_A1A2,  event = 1
  • No future ACR event :   tte = days_to_last_biopsy, event = 0  (censored)

Produces:
  1. acr_grade_distribution.pdf   Grade counts + label mapping
  2. tte_label_overview.pdf       v7 event/censored counts + label breakdown
  3. dataset_overview.pdf         Sample statistics with v7 framing
  4. tte_distributions.pdf        TTE distributions: tte=0 / pre-event / censored
  5. modality_by_endpoint.pdf     Modality rates by ACR+/ACR− label
  6. patient_timelines.pdf        Per-patient strip chart with TTE arrows
  7. sample_table.pdf             Sample-level table with v7 labels

Usage:
    python visualize_survival_labels.py \
        --csv    /path/to/multimodal_splits_nested_cv.csv \
        --outdir ./survival_label_viz \
        --max_patients 50
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
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

# ── palette ───────────────────────────────────────────────────────────────────
C_EVENT  = "#D62728"   # event / ACR+
C_PRE    = "#FF7F0E"   # pre-event (will have ACR in the future)
C_CENS   = "#AAAAAA"   # censored
C_NEG    = "#1F77B4"   # ACR- (A0)
C_UNLBL  = "#CCCCCC"   # no label (unknown grade)
C_CLAD   = "#FF7F0E"
C_DEATH  = "#2C2C2C"
MOD_COLS = {"HE": "#1F77B4", "BAL": "#D62728", "CT": "#2CA02C", "Clinical": "#9467BD"}
MODALITIES = ["HE", "BAL", "CT", "Clinical"]

MOD_PITCH = 0.55
SLOT_H    = len(MODALITIES) * MOD_PITCH + 0.8
GAP_H     = 0.6


# ── data loading + v7 label computation ──────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["anchor_dt"])
    df = df.sort_values(["patient_id", "anchor_dt"]).reset_index(drop=True)
    first = df.groupby("patient_id")["anchor_dt"].transform("min")
    df["days_from_first"] = (df["anchor_dt"] - first).dt.days
    return df


def _is_acr_event(grade: object) -> bool:
    return isinstance(grade, str) and (
        grade.upper().startswith("A1") or grade.upper().startswith("A2")
    )


def _acr_label(grade: object):
    """A0* → 0,  A1*/A2* → 1,  else → None."""
    if not isinstance(grade, str):
        return None
    g = grade.upper()
    if g.startswith("A0"):
        return 0
    if g.startswith("A1") or g.startswith("A2"):
        return 1
    return None


def compute_tte_v7(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach v7 task labels to df:
      v7_label    : 0 (A0), 1 (A1/A2), NaN (unknown)
      tte         : days to next A1/A2 biopsy; 0 for ACR+ biopsies
      event       : 1 = event, 0 = censored
      surv_status : 'acr_now' | 'pre_event' | 'censored'
    """
    df = df.copy()
    df["v7_label"] = df["acr_grade"].apply(_acr_label)

    # Collect all A1/A2 dates per patient
    acr_mask  = df["acr_grade"].apply(_is_acr_event)
    acr_dates: dict = {}
    for _, row in df[acr_mask].iterrows():
        acr_dates.setdefault(row["patient_id"], []).append(row["anchor_dt"])

    last_date: dict = df.groupby("patient_id")["anchor_dt"].max().to_dict()

    ttes, events, statuses = [], [], []
    for _, row in df.iterrows():
        pid = row["patient_id"]
        t   = row["anchor_dt"]
        future = sorted([d for d in acr_dates.get(pid, []) if d > t])

        if _is_acr_event(row.get("acr_grade")):
            tte, ev, st = 0, 1, "acr_now"
        elif future:
            tte, ev, st = (future[0] - t).days, 1, "pre_event"
        else:
            # Censored: no future ACR event for this patient.
            # TTE = days to last biopsy. Grade of THIS biopsy is irrelevant
            # — a missing/unknown grade does NOT imply ACR-negative status.
            last = last_date.get(pid, t)
            tte  = max(int((last - t).days), 0)
            ev, st = 0, "censored"

        ttes.append(float(tte))
        events.append(ev)
        statuses.append(st)

    df["tte"]         = ttes
    df["event"]       = events
    df["surv_status"] = statuses
    return df


# ── Figure 1: ACR grade distribution ─────────────────────────────────────────

def fig_acr_grade_distribution(df: pd.DataFrame, outdir: Path):
    grade_counts = df["acr_grade"].fillna("NaN/unknown").value_counts()
    top = grade_counts.head(15)

    label_map = {g: _acr_label(g) for g in top.index}
    bar_colors = [
        C_EVENT if label_map.get(g) == 1 else
        (C_NEG   if label_map.get(g) == 0 else C_UNLBL)
        for g in top.index
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ACR Grade Distribution — v7 Label Mapping", fontsize=13, fontweight="bold")

    ax = axes[0]
    bars = ax.barh(range(len(top)), top.values, color=bar_colors,
                   edgecolor="white", lw=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Sample count", fontsize=11)
    ax.set_title("Top 15 ACR grade values", fontsize=11, fontweight="bold")
    for bar, v in zip(bars, top.values):
        ax.text(v + max(top.values)*0.01, bar.get_y() + bar.get_height()/2,
                str(v), va="center", fontsize=9)
    ax.grid(True, alpha=0.25, axis="x")
    legend_h = [
        mpatches.Patch(color=C_EVENT, label="→ label = 1  (A1*/A2*)"),
        mpatches.Patch(color=C_NEG,   label="→ label = 0  (A0*)"),
        mpatches.Patch(color=C_UNLBL, label="→ label = None"),
    ]
    ax.legend(handles=legend_h, fontsize=9, loc="lower right")

    ax2 = axes[1]
    n_lbl0   = int((df["v7_label"] == 0).sum())
    n_lbl1   = int((df["v7_label"] == 1).sum())
    n_lblnan = int(df["v7_label"].isna().sum())
    names  = ["label=0\n(ACR−)", "label=1\n(ACR+)", "None\n(excluded)"]
    values = [n_lbl0, n_lbl1, n_lblnan]
    colors = [C_NEG, C_EVENT, C_UNLBL]
    bars2 = ax2.bar(names, values, color=colors, edgecolor="white", lw=0.5, width=0.5)
    for b in bars2:
        h = int(b.get_height())
        ax2.text(b.get_x() + b.get_width()/2, h + max(values)*0.02,
                 str(h), ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Samples", fontsize=11)
    ax2.set_title("v7 classification label distribution", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(values)*1.25)
    ax2.grid(True, alpha=0.25, axis="y")
    ax2.tick_params(axis="x", labelsize=10)

    fig.tight_layout()
    p = outdir / "acr_grade_distribution.pdf"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Figure 2: v7 TTE label overview ──────────────────────────────────────────

def fig_tte_label_overview(df: pd.DataFrame, outdir: Path):
    n_acr_now = int((df["surv_status"] == "acr_now").sum())
    n_pre     = int((df["surv_status"] == "pre_event").sum())
    n_cens    = int((df["surv_status"] == "censored").sum())
    total     = len(df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "v7 Survival Label Overview  —  gap-time TTE to next A1/A2 biopsy",
        fontsize=13, fontweight="bold",
    )

    # Bar 1: survival status breakdown
    ax = axes[0]
    cats = ["ACR now\n(tte=0, ev=1)", "Pre-event\n(tte>0, ev=1)", "Censored\n(ev=0)"]
    vals = [n_acr_now, n_pre, n_cens]
    cols = [C_EVENT, C_PRE, C_CENS]
    bars = ax.bar(cats, vals, color=cols, edgecolor="white", lw=0.5, width=0.5)
    for b in bars:
        h = int(b.get_height())
        ax.text(b.get_x() + b.get_width()/2, h + max(vals)*0.02,
                f"{h}\n({100*h/total:.1f}%)",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Samples", fontsize=11)
    ax.set_title("Sample breakdown by v7 survival status", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.grid(True, alpha=0.25, axis="y")
    ax.tick_params(axis="x", labelsize=9)

    # Bar 2: event rate stacked
    ax2 = axes[1]
    ev1 = n_acr_now + n_pre   # all event=1
    ev0 = n_cens               # all event=0
    ax2.bar(["Survival\n(ACR TTE)"], [ev1], color=C_EVENT, edgecolor="white", lw=0.5,
            width=0.4, label=f"Event (n={ev1})")
    ax2.bar(["Survival\n(ACR TTE)"], [ev0], bottom=[ev1], color=C_CENS, alpha=0.6,
            edgecolor="white", lw=0.5, width=0.4, label=f"Censored (n={ev0})")
    pct = 100 * ev1 / (ev1 + ev0) if (ev1 + ev0) else 0
    ax2.text(0, ev1 + ev0 + max(ev1+ev0, 1)*0.03, f"{pct:.1f}% events",
             ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Samples", fontsize=11)
    ax2.set_title("Event rate", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.25, axis="y")

    note = (
        "TTE=0:  the biopsy IS an ACR event (A1/A2) → event=1\n"
        "TTE>0, ev=1:  A0* biopsy preceding a future A1/A2 → gap-time\n"
        "TTE>0, ev=0:  no future ACR event → censored at last biopsy"
    )
    fig.text(0.5, -0.04, note, ha="center", fontsize=9,
             color="#555555", style="italic",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f9f9f9", ec="#cccccc", lw=0.8))

    fig.tight_layout()
    p = outdir / "tte_label_overview.pdf"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Figure 3: Dataset overview ────────────────────────────────────────────────

def fig_dataset_overview(df: pd.DataFrame, outdir: Path):
    acr_pos   = int((df["v7_label"] == 1).sum())
    acr_neg   = int((df["v7_label"] == 0).sum())
    acr_unlbl = int(df["v7_label"].isna().sum())
    n_event   = int((df["event"] == 1).sum())
    n_cens    = int((df["event"] == 0).sum())
    clad_ev   = int((df["clad_event"] == 1).sum()) if "clad_event" in df.columns else 0
    clad_ce   = int((df["clad_event"] == 0).sum()) if "clad_event" in df.columns else 0
    death_ev  = int((df["death_event"] == 1).sum()) if "death_event" in df.columns else 0
    death_ce  = int((df["death_event"] == 0).sum()) if "death_event" in df.columns else 0

    n_pts = df["patient_id"].nunique()

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)
    fig.suptitle(
        f"Dataset Overview  —  {len(df)} samples · {n_pts} patients  "
        "(v7 task definitions)",
        fontsize=14, fontweight="bold",
    )

    def labeled_bar(ax, labels, values, colors, title, ylabel="Samples"):
        bars = ax.bar(labels, values, color=colors, edgecolor="white", lw=0.6, width=0.55)
        for b in bars:
            h = int(b.get_height())
            if h > 0:
                ax.text(b.get_x() + b.get_width()/2, h + max(values)*0.02,
                        str(h), ha="center", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.25, axis="y")
        ax.set_ylim(0, max(values) * 1.28)
        ax.tick_params(axis="x", labelsize=9)

    labeled_bar(
        fig.add_subplot(gs[0, 0]),
        ["ACR+\n(A1/A2)", "ACR−\n(A0)", "No grade\n(excluded)"],
        [acr_pos, acr_neg, acr_unlbl],
        [C_EVENT, C_NEG, C_UNLBL],
        "Classification label (v7)",
    )
    labeled_bar(
        fig.add_subplot(gs[0, 1]),
        ["Event\n(ev=1)", "Censored\n(ev=0)"],
        [n_event, n_cens],
        [C_EVENT, C_CENS],
        "ACR survival: event/censored",
    )

    # TTE=0 vs TTE>0 breakdown
    ax_tte = fig.add_subplot(gs[0, 2])
    n_tte0    = int((df["tte"] == 0).sum())
    n_ttegt0_ev = int(((df["tte"] > 0) & (df["event"] == 1)).sum())
    n_ttegt0_ce = int(((df["tte"] > 0) & (df["event"] == 0)).sum())
    labeled_bar(
        ax_tte,
        ["tte=0\n(ACR now)", "tte>0\nev=1", "tte>0\ncensored"],
        [n_tte0, n_ttegt0_ev, n_ttegt0_ce],
        [C_EVENT, C_PRE, C_CENS],
        "TTE breakdown",
    )

    # CLAD
    if clad_ev + clad_ce > 0:
        labeled_bar(
            fig.add_subplot(gs[1, 0]),
            ["CLAD\nevent", "Censored"],
            [clad_ev, clad_ce],
            [C_CLAD, C_NEG],
            "CLAD survival",
        )

    # Death
    if death_ev + death_ce > 0:
        labeled_bar(
            fig.add_subplot(gs[1, 1]),
            ["Died\n(event)", "Alive\n(censored)"],
            [death_ev, death_ce],
            [C_DEATH, C_NEG],
            "Death survival",
        )

    # Event rate comparison
    ax_rate = fig.add_subplot(gs[1, 2])
    ep_names = ["ACR\nsurv"]
    ev_c = [n_event]; ce_c = [n_cens]
    if clad_ev + clad_ce > 0:
        ep_names.append("CLAD"); ev_c.append(clad_ev); ce_c.append(clad_ce)
    if death_ev + death_ce > 0:
        ep_names.append("Death"); ev_c.append(death_ev); ce_c.append(death_ce)
    ep_colors = [C_EVENT, C_CLAD, C_DEATH][:len(ep_names)]
    totals = [e + c for e, c in zip(ev_c, ce_c)]
    x = np.arange(len(ep_names))
    ax_rate.bar(x, ev_c, color=ep_colors, edgecolor="white", lw=0.5, width=0.45, label="Event")
    ax_rate.bar(x, ce_c, bottom=ev_c, color=C_CENS, alpha=0.55, edgecolor="white", lw=0.5, width=0.45, label="Censored")
    for xi, (e, tot) in enumerate(zip(ev_c, totals)):
        pct = 100 * e / tot if tot else 0
        ax_rate.text(xi, tot + max(totals)*0.01, f"{pct:.1f}%",
                     ha="center", fontsize=10, fontweight="bold")
    ax_rate.set_xticks(x); ax_rate.set_xticklabels(ep_names, fontsize=10)
    ax_rate.set_ylabel("Samples", fontsize=10)
    ax_rate.set_title("Class imbalance (stacked)", fontsize=11, fontweight="bold")
    ax_rate.legend(fontsize=9)
    ax_rate.grid(True, alpha=0.25, axis="y")

    p = outdir / "dataset_overview.pdf"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Figure 4: TTE schematic + distributions (ACR, CLAD, Death) ───────────────

def _hist_pair(ax, ev_vals, ce_vals, ev_col, ce_col,
               xlabel, title, ev_label, ce_label,
               include_zero_bar=False, zero_n=0, zero_col=None):
    """Shared helper: overlapping histograms for event vs censored TTE."""
    all_vals = pd.concat([ev_vals, ce_vals]).dropna()
    if len(all_vals) == 0:
        ax.set_title(f"{title}\n(no data)", fontsize=11, fontweight="bold")
        return
    hi = float(np.percentile(all_vals, 97))
    lo = 0.0
    bins = np.linspace(lo, max(hi, 1), 40)

    if len(ce_vals):
        ax.hist(ce_vals.clip(upper=hi), bins=bins, color=ce_col, alpha=0.55,
                label=f"{ce_label}  n={len(ce_vals)},  med={ce_vals.median():.0f} d",
                edgecolor="white", lw=0.3, zorder=2)
        ax.axvline(ce_vals.median(), color=ce_col, lw=1.5, ls=":",
                   alpha=0.85, zorder=3)
    if len(ev_vals):
        ax.hist(ev_vals.clip(upper=hi), bins=bins, color=ev_col, alpha=0.80,
                label=f"{ev_label}  n={len(ev_vals)},  med={ev_vals.median():.0f} d",
                edgecolor="white", lw=0.3, zorder=4)
        ax.axvline(ev_vals.median(), color=ev_col, lw=2.0, ls="--",
                   alpha=0.9, zorder=5)

    # Optional side-bar for tte=0 events (ACR now)
    if include_zero_bar and zero_n > 0:
        ax2 = ax.inset_axes([0.68, 0.45, 0.30, 0.50])
        ax2.bar([0], [zero_n], color=zero_col, edgecolor="white", lw=0.5, width=0.6)
        ax2.set_xticks([0]); ax2.set_xticklabels(["tte=0\n(ACR\nnow)"], fontsize=7)
        ax2.set_title(f"n={zero_n}", fontsize=7.5, fontweight="bold")
        ax2.tick_params(labelsize=6); ax2.grid(True, alpha=0.2, axis="y")
        for sp in ["top","right"]: ax2.spines[sp].set_visible(False)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Samples", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25, axis="y")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)


def _draw_schematic(ax):
    """
    Presentation-quality schematic of the v7 gap-time TTE definition.
    Shows a single patient with 6 biopsies (A0, A0, A1/A2, A0, A0, A0)
    and illustrates the three sample categories.
    """
    ax.set_xlim(-30, 560)
    ax.set_ylim(-0.7, 1.5)
    ax.axis("off")
    ax.set_facecolor("#FAFBFC")

    # Timeline backbone
    ax.annotate("", xy=(540, 0), xytext=(-10, 0),
                arrowprops=dict(arrowstyle="-|>", color="#888888",
                                lw=1.8, mutation_scale=14))
    ax.text(545, -0.02, "time", fontsize=10, color="#888888", va="center")

    # Biopsy positions (days from patient start)
    biopsies = [
        (30,  "A0",    "pre-event",  "#1F77B4"),
        (100, "A0",    "pre-event",  "#1F77B4"),
        (180, "A1B0",  "acr_now",    "#D62728"),
        (260, "A0",    "pre-event",  "#FF7F0E"),
        (350, "A1B0",  "acr_now",    "#D62728"),
        (460, "A0",    "censored",   "#AAAAAA"),
    ]

    # Last biopsy date (for censoring)
    last_t = max(t for t, *_ in biopsies)

    for t, grade, status, col in biopsies:
        if status == "acr_now":
            ax.scatter([t], [0], s=280, marker="*", color=col,
                       zorder=5, linewidths=0)
        elif status == "pre-event":
            ax.scatter([t], [0], s=90, marker="o", color=col,
                       zorder=4, linewidths=0)
        else:
            ax.scatter([t], [0], s=70, marker="o", color=col,
                       zorder=4, linewidths=0.5, edgecolors="#888888")
        # grade label below
        ax.text(t, -0.18, grade, ha="center", va="top",
                fontsize=8.5, color="#333333",
                fontweight="bold" if status == "acr_now" else "normal")

    # ── Arrows + TTE labels ───────────────────────────────────────────────────
    ARROW_KW = dict(lw=1.5, shrinkA=6, shrinkB=6)

    def arrow(x0, x1, y, col, label, above=True):
        sign = 1 if above else -1
        yarr = y + 0.22 * sign
        ax.annotate("", xy=(x1, yarr), xytext=(x0, yarr),
                    arrowprops=dict(arrowstyle="->", color=col, **ARROW_KW))
        ax.text((x0+x1)/2, yarr + 0.12*sign, label,
                ha="center", va="bottom" if above else "top",
                fontsize=8.5, color=col, fontweight="bold")

    # B1 → first ACR+ (t=180)
    arrow(30,  180, 0, "#1F77B4", "tte = 150 d, event = 1", above=True)
    # B2 → first ACR+
    arrow(100, 180, 0, "#1F77B4", "tte = 80 d, event = 1",  above=True)
    # B4 → second ACR+ (t=350)
    arrow(260, 350, 0, "#FF7F0E", "tte = 90 d, event = 1",  above=True)

    # ACR+ labels (tte=0)
    for t, _, status, _ in biopsies:
        if status == "acr_now":
            ax.text(t, 0.38, "tte = 0\nevent = 1", ha="center", va="bottom",
                    fontsize=8, color="#D62728", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#fff0f0",
                              ec="#D62728", lw=0.8))

    # B6 censored → last biopsy (dashed censor mark)
    ax.annotate("", xy=(last_t + 15, -0.38), xytext=(460, -0.38),
                arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=1.5,
                                linestyle="dashed", shrinkA=4, shrinkB=4))
    ax.text((460 + last_t + 15)/2, -0.55,
            f"tte = {last_t-460} d, event = 0  (censored at last biopsy)",
            ha="center", va="top", fontsize=8.5, color="#888888", style="italic")

    # Censor tick at last_t
    ax.plot([last_t, last_t], [-0.08, 0.08], color="#888888", lw=2.5,
            solid_capstyle="round", zorder=6)
    ax.text(last_t, 0.13, "last\nbiopsy", ha="center", va="bottom",
            fontsize=8, color="#888888")

    # Legend
    legend_h = [
        Line2D([0],[0], marker="*", color="w", markerfacecolor=C_EVENT,
               markersize=12, label="A1/A2 biopsy  (tte=0, event=1)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_NEG,
               markersize=9,  label="A0 before future ACR  (tte>0, event=1)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_PRE,
               markersize=9,  label="A0 before second ACR  (tte>0, event=1)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_CENS,
               markersize=9,  label="No future ACR  →  censored at last biopsy"),
    ]
    ax.legend(handles=legend_h, fontsize=9, loc="upper left",
              framealpha=0.92, edgecolor="#cccccc")
    ax.set_title(
        "v7 gap-time TTE definition  —  toy example (one patient, recurrent events)",
        fontsize=11.5, fontweight="bold", pad=10,
    )


def fig_tte_distributions(df: pd.DataFrame, outdir: Path):
    """
    Four-panel figure:
      Top row (full width): schematic showing TTE calculation
      Bottom row: ACR / CLAD / Death TTE histograms (event vs censored)
    """
    # ── Compute CLAD TTE ──────────────────────────────────────────────────────
    study_end = df["anchor_dt"].max()
    clad_ev_tte, clad_ce_tte = [], []
    for _, row in df.iterrows():
        st = row.get("clad_status")
        cd = row.get("clad_days")
        if pd.isna(st):
            continue
        if float(st) == 1.0:
            if pd.notna(cd) and float(cd) > 0:
                clad_ev_tte.append(float(cd))
        else:
            # Censored — priority: death_days, then study_end proxy
            dd = row.get("death_days")
            if pd.notna(dd) and float(dd) > 0:
                clad_ce_tte.append(float(dd))
            else:
                proxy = (study_end - row["anchor_dt"]).days
                if proxy > 0:
                    clad_ce_tte.append(float(proxy))

    # ── Compute Death TTE ─────────────────────────────────────────────────────
    death_ev_tte, death_ce_tte = [], []
    for _, row in df.iterrows():
        st = row.get("death_status")
        dd = row.get("death_days")
        if pd.isna(st):
            continue
        if float(st) == 1.0 and pd.notna(dd):
            death_ev_tte.append(float(dd))
        elif float(st) == 0.0:
            proxy = (study_end - row["anchor_dt"]).days
            if proxy > 0:
                death_ce_tte.append(float(proxy))

    clad_ev  = pd.Series(clad_ev_tte,  dtype=float)
    clad_ce  = pd.Series(clad_ce_tte,  dtype=float)
    death_ev = pd.Series(death_ev_tte, dtype=float)
    death_ce = pd.Series(death_ce_tte, dtype=float)

    # ── ACR TTE groups ────────────────────────────────────────────────────────
    acr_now = df[df["surv_status"] == "acr_now"]   # tte=0, event=1
    pre_ev  = df[df["surv_status"] == "pre_event"]["tte"].dropna()
    cens    = df[(df["surv_status"] == "censored") & (df["tte"] > 0)]["tte"].dropna()

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.52, wspace=0.38,
                            height_ratios=[1.25, 1.0])

    # Row 0: schematic spanning all 3 columns
    ax_sch = fig.add_subplot(gs[0, :])
    _draw_schematic(ax_sch)

    # Row 1 col 0: ACR TTE
    ax_acr = fig.add_subplot(gs[1, 0])
    _hist_pair(
        ax_acr, pre_ev, cens,
        ev_col=C_PRE, ce_col=C_CENS,
        xlabel="Days to next A1/A2 biopsy",
        title="ACR — gap-time TTE  (tte > 0 only)",
        ev_label="Pre-event  (ev=1)",
        ce_label="Censored  (ev=0)",
        include_zero_bar=True, zero_n=len(acr_now), zero_col=C_EVENT,
    )

    # Row 1 col 1: CLAD TTE
    ax_clad = fig.add_subplot(gs[1, 1])
    _hist_pair(
        ax_clad, clad_ev, clad_ce,
        ev_col=C_CLAD, ce_col=C_CENS,
        xlabel="Days to CLAD event",
        title="CLAD — TTE",
        ev_label="CLAD event  (ev=1)",
        ce_label="Censored  (ev=0)",
    )

    # Row 1 col 2: Death TTE
    ax_dth = fig.add_subplot(gs[1, 2])
    _hist_pair(
        ax_dth, death_ev, death_ce,
        ev_col=C_DEATH, ce_col=C_CENS,
        xlabel="Days to death / last follow-up",
        title="Death — TTE",
        ev_label="Died  (ev=1)",
        ce_label="Alive / censored  (ev=0)",
    )

    fig.suptitle(
        "Time-to-Event distributions  ·  v7 task definitions\n"
        "Censored = no event observed; TTE = days to last available follow-up",
        fontsize=13, fontweight="bold", y=1.01,
    )

    p = outdir / "tte_distributions.pdf"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Figure 5: Modality by endpoint ───────────────────────────────────────────

def fig_modality_by_endpoint(df: pd.DataFrame, outdir: Path):
    pos = df[df["v7_label"] == 1]
    neg = df[df["v7_label"] == 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Modality Availability by ACR Classification Label (v7)",
                 fontsize=13, fontweight="bold")

    x = np.arange(len(MODALITIES)); w = 0.35
    rp = [pos[f"has_{m}"].mean() * 100 for m in MODALITIES]
    rn = [neg[f"has_{m}"].mean() * 100 for m in MODALITIES]
    ax.bar(x - w/2, rp, w, label=f"ACR+ / label=1  (n={len(pos)})",
           color=C_EVENT, edgecolor="white", lw=0.5)
    ax.bar(x + w/2, rn, w, label=f"ACR− / label=0  (n={len(neg)})",
           color=C_NEG, edgecolor="white", lw=0.5, alpha=0.7)
    for xi, (p_, n_) in enumerate(zip(rp, rn)):
        if n_ > 0:
            ratio = p_ / n_
            c = "#B00000" if ratio > 1.5 else ("k" if ratio > 1.1 else "#888888")
            ax.text(xi, max(p_, n_) + 2.8, f"{ratio:.1f}×",
                    ha="center", fontsize=9, color=c, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(MODALITIES, fontsize=11)
    ax.set_ylabel("Samples with modality (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_title("Ratio >1.0 means modality over-represented in ACR+",
                 fontsize=9, color="#555")

    fig.tight_layout()
    p = outdir / "modality_by_endpoint.pdf"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Figure 6: Patient timelines ───────────────────────────────────────────────

def fig_patient_timelines(df: pd.DataFrame, outdir: Path, max_patients: int = 40):
    """
    One row per patient:
      - Modality tracks (dots per visit)
      - Event row: red ★ = ACR+ (tte=0),  orange ▲ = pre-event (tte>0),
                   grey ● = censored,  empty ○ = no label
      - Arrow from each pre-event sample → next ACR+ date
      - CLAD | and Death ✕ lines
    ACR+ patients shown first.
    """
    has_acr = df.groupby("patient_id")["v7_label"].apply(lambda x: (x == 1).any())
    patients = (has_acr[has_acr].index.tolist() +
                has_acr[~has_acr].index.tolist())[:max_patients]
    n = len(patients)

    slot     = SLOT_H + GAP_H
    total_h  = n * slot + 1.0
    fig, ax  = plt.subplots(figsize=(28, max(total_h, 10)))
    ax.set_facecolor("#F7F7F7")
    y_ticks, y_labels = [], []

    # pre-compute patient-level next ACR date
    acr_mask  = df["acr_grade"].apply(_is_acr_event)
    acr_dates_map: dict = {}
    for _, row in df[acr_mask].iterrows():
        acr_dates_map.setdefault(row["patient_id"], []).append(
            (row["days_from_first"], row["anchor_dt"])
        )
    for pid in acr_dates_map:
        acr_dates_map[pid].sort()

    for pi, pid in enumerate(reversed(patients)):
        y0  = pi * slot
        grp = df[df["patient_id"] == pid].sort_values("days_from_first")
        t_vals = grp["days_from_first"].values
        if len(t_vals) == 0:
            continue

        ax.plot([t_vals.min(), t_vals.max()], [y0, y0],
                color="#DDDDDD", lw=2.0, zorder=1, solid_capstyle="round")

        # ── CLAD event line ───────────────────────────────────────────────
        if "clad_status" in grp.columns and (grp["clad_status"] == 1).any():
            row0 = grp[grp["clad_days"].notna()].iloc[0] if grp["clad_days"].notna().any() else None
            if row0 is not None and not np.isnan(row0["clad_days"]):
                cd = row0["days_from_first"] + row0["clad_days"]
                ax.plot([cd, cd],
                        [y0 - 0.25, y0 + len(MODALITIES)*MOD_PITCH + 0.25],
                        color=C_CLAD, lw=2.2, ls="-", alpha=0.85, zorder=4)

        # ── Death line ─────────────────────────────────────────────────────
        if "death_status" in grp.columns and (grp["death_status"] == 1).any():
            row0 = grp[grp["death_days"].notna()].iloc[0] if grp["death_days"].notna().any() else None
            if row0 is not None and not np.isnan(row0["death_days"]):
                dd = row0["days_from_first"] + row0["death_days"]
                ax.plot([dd, dd],
                        [y0 - 0.3, y0 + len(MODALITIES)*MOD_PITCH + 0.3],
                        color=C_DEATH, lw=2.5, ls=":", alpha=0.9, zorder=5)

        # ── Modality tracks ────────────────────────────────────────────────
        for mi, m in enumerate(MODALITIES):
            y_mod = y0 + (mi + 1) * MOD_PITCH
            ax.plot([t_vals.min(), t_vals.max()], [y_mod, y_mod],
                    color="#EEEEEE", lw=1.0, zorder=1)
            t_pres   = grp[grp[f"has_{m}"] == True]["days_from_first"].values
            t_absent = grp[grp[f"has_{m}"] == False]["days_from_first"].values
            if len(t_absent):
                ax.scatter(t_absent, [y_mod]*len(t_absent),
                           color="#E0E0E0", s=16, zorder=2, linewidths=0)
            if len(t_pres):
                ax.scatter(t_pres, [y_mod]*len(t_pres),
                           color=MOD_COLS[m], s=32, zorder=3, linewidths=0)

        # ── Event row (main ACR row at y0) ─────────────────────────────────
        acr_times = [d for d, _ in acr_dates_map.get(pid, [])]

        for _, row in grp.iterrows():
            t   = row["days_from_first"]
            st  = row.get("surv_status", "censored")
            tte = row.get("tte", float("nan"))

            if st == "acr_now":
                # Red star — ACR event now
                ax.scatter([t], [y0], marker="*", s=160, color=C_EVENT,
                           zorder=5, linewidths=0)

            elif st == "pre_event":
                # Orange triangle — will have ACR in the future
                ax.scatter([t], [y0], marker="^", s=55, color=C_PRE,
                           zorder=4, linewidths=0)
                # Arrow to next ACR+ date
                if not np.isnan(tte):
                    t_next = t + tte
                    ax.annotate(
                        "", xy=(t_next, y0 + 0.12), xytext=(t, y0 + 0.12),
                        arrowprops=dict(
                            arrowstyle="->", color=C_EVENT,
                            lw=1.0, shrinkA=2, shrinkB=2,
                        ),
                        zorder=3,
                    )
                    ax.text((t + t_next) / 2, y0 + 0.22,
                            f"{int(tte)}d", ha="center", va="bottom",
                            fontsize=5.5, color=C_EVENT, zorder=5)

            elif st == "censored":
                ax.scatter([t], [y0], marker="o", s=28, color=C_CENS,
                           zorder=3, linewidths=0.3, edgecolors="#888888")
            else:
                ax.scatter([t], [y0], marker="o", s=28, color=C_CENS,
                           zorder=3, linewidths=0.3, edgecolors="#888888")

        # ── Patient label ──────────────────────────────────────────────────
        y_mid = y0 + len(MODALITIES)*MOD_PITCH/2
        y_ticks.append(y_mid)
        y_labels.append(str(pid))

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Days from patient's first sample", fontsize=13)
    ax.grid(True, alpha=0.15, axis="x")
    ax.set_ylim(-0.5, n * slot + 0.5)

    # Right y-axis: row labels
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    if patients:
        y0_ref = (n-1) * slot
        ax2.set_yticks(
            [y0_ref + (mi+1)*MOD_PITCH for mi in range(len(MODALITIES))] + [y0_ref]
        )
        ax2.set_yticklabels(MODALITIES + ["ACR"], fontsize=9)
        ax2.tick_params(right=False)

    legend_h = [
        Line2D([0],[0], marker="*", color="w", markerfacecolor=C_EVENT,
               markersize=10, label="ACR+ biopsy (tte=0, ev=1)"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor=C_PRE,
               markersize=8,  label="Pre-event (tte>0, ev=1) + arrow to next ACR+"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_CENS,
               markersize=7,  label="Censored (ev=0) — any grade"),
        Line2D([0],[0], color=C_CLAD,  lw=2.2, ls="-",  label="CLAD event"),
        Line2D([0],[0], color=C_DEATH, lw=2.5, ls=":",  label="Death"),
    ] + [mpatches.Patch(color=MOD_COLS[m], label=m) for m in MODALITIES]
    ax.legend(handles=legend_h, fontsize=9, loc="lower right",
              ncol=3, framealpha=0.9)

    ax.set_title(
        f"Patient timelines  ·  {n} patients (ACR+ first)\n"
        "★=ACR event (tte=0)  ▲=pre-event (arrow→next ACR+)  ●=censored",
        fontsize=12, pad=10,
    )
    fig.tight_layout()
    p = outdir / "patient_timelines.pdf"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Figure 7: Sample table ────────────────────────────────────────────────────

def fig_sample_table(df: pd.DataFrame, outdir: Path, max_rows: int = 100):
    acr_pts = df[df["v7_label"] == 1]["patient_id"].unique()
    show    = (df[df["patient_id"].isin(acr_pts)]
               .sort_values(["patient_id", "days_from_first"])
               .head(max_rows))

    col_labels = ["Patient", "Date", "Day",
                  "HE", "BAL", "CT", "Clin",
                  "Grade", "Label", "TTE (d)", "Event", "Status"]
    col_w = [0.10, 0.09, 0.05,
             0.04, 0.04, 0.04, 0.04,
             0.07, 0.05, 0.07, 0.05, 0.08]

    STATUS_COLOR = {
        "acr_now":   "#FFD5D5",
        "pre_event": "#FFF0D0",
        "censored":  "#F5F5F5",
    }

    rows, row_colors = [], []
    for _, row in show.iterrows():
        grade_str = str(row.get("acr_grade", "−")) if pd.notna(row.get("acr_grade")) else "−"
        lbl       = row.get("v7_label")
        lbl_str   = ("1 (ACR+)" if lbl == 1 else "0 (ACR−)" if lbl == 0 else "None")
        tte_str   = f"{int(row['tte'])}" if pd.notna(row.get("tte")) else "−"
        ev_str    = str(int(row["event"])) if pd.notna(row.get("event")) else "−"
        st        = row.get("surv_status", "−")
        rows.append([
            row["patient_id"],
            str(row["anchor_dt"].date()),
            str(int(row["days_from_first"])),
            "●" if row.get("has_HE",      False) else "·",
            "●" if row.get("has_BAL",     False) else "·",
            "●" if row.get("has_CT",      False) else "·",
            "●" if row.get("has_Clinical",False) else "·",
            grade_str, lbl_str, tte_str, ev_str, st,
        ])
        bg = STATUS_COLOR.get(st, "#FFFFFF")
        row_colors.append([bg] * len(col_labels))

    n   = len(rows)
    fh  = max(n * 0.24 + 1.5, 8)
    fig, ax = plt.subplots(figsize=(20, fh))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, colWidths=col_w,
                   cellColours=row_colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.25)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2C3E50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    mod_idx = {"HE": 3, "BAL": 4, "CT": 5, "Clinical": 6}
    for ri, (_, row) in enumerate(show.iterrows()):
        for m, ci in mod_idx.items():
            if row.get(f"has_{m}", False):
                tbl[(ri+1, ci)].set_facecolor(MOD_COLS[m])
                tbl[(ri+1, ci)].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "Sample table  ·  ACR patients  ·  v7 definitions\n"
        "TTE = days to next A1/A2 biopsy  ·  "
        "Status: acr_now (★) / pre_event (▲) / censored",
        fontsize=10, pad=10,
    )
    p = outdir / "sample_table.pdf"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",
        default="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
    ap.add_argument("--outdir",
        default="/home/aih/dinesh.haridoss/chicago_mil/visualizations/survival_label_viz")
    ap.add_argument("--max_patients", type=int, default=50)
    ap.add_argument("--max_table_rows", type=int, default=120)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {outdir}")

    print("\nLoading data …")
    df = load_data(args.csv)
    print(f"  {len(df)} samples,  {df['patient_id'].nunique()} patients")

    print("\nComputing v7 TTE labels …")
    df = compute_tte_v7(df)
    n_ev = int((df["event"]==1).sum())
    n_ce = int((df["event"]==0).sum())
    print(f"  event=1: {n_ev}  (tte=0: {int((df.tte==0).sum())},  "
          f"tte>0: {int(((df.tte>0)&(df.event==1)).sum())})")
    print(f"  event=0 (censored): {n_ce}")

    # add clad/death event columns if needed
    if "clad_status" in df.columns and "clad_event" not in df.columns:
        df["clad_event"] = np.where(
            (df["clad_status"]==1) & (df["clad_days"]>0), 1.0,
            np.where(df["clad_days"].isna() | (df["clad_days"]<=0), np.nan, 0.0))
    if "death_status" in df.columns and "death_event" not in df.columns:
        df["death_event"] = np.where(
            df["death_status"]==1, 1.0,
            np.where(df["death_status"]==0, 0.0, np.nan))

    print("\nFig 1: ACR grade distribution …")
    fig_acr_grade_distribution(df, outdir)
    print("Fig 2: TTE label overview …")
    fig_tte_label_overview(df, outdir)
    print("Fig 3: Dataset overview …")
    fig_dataset_overview(df, outdir)
    print("Fig 4: TTE distributions …")
    fig_tte_distributions(df, outdir)
    print("Fig 5: Modality by endpoint …")
    fig_modality_by_endpoint(df, outdir)
    print("Fig 6: Patient timelines …")
    fig_patient_timelines(df, outdir, max_patients=args.max_patients)
    print("Fig 7: Sample table …")
    fig_sample_table(df, outdir, max_rows=args.max_table_rows)

    print(f"\nDone. All figures in {outdir}/")


if __name__ == "__main__":
    main()
