#!/usr/bin/env python3
"""
ACR episode analysis.

Defines an ACR episode as a continuous run of ACR+ (A1*/A2*) biopsies
with no A0 biopsy between them. Computes:
  - Per-patient episode count, duration, inter-episode gap
  - Cohort-level stats
  - Plots: timeline for multi-episode patients, summary distributions

Known multi-episode patients from clinical team:
  2 episodes: LT035, LT038, LT084, LT157, LT169
  3 episodes: LT146, LT239
  Multiple ACR+ biopsies: LT119, LT004, LT019, LT035, LT011, LT025, LT032,
    LT041, LT081, LT084, LT092, LT101, LT146, LT168, LT169, LT173, LT239,
    LT302, LT008, LT013, LT021, LT024, LT027, LT038, LT057, LT062, LT127,
    LT130, LT157, LT175, LT176, LT206, LT238, LT244, LT252, LT305
"""
import sys
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
from lifelines import KaplanMeierFitter

warnings.filterwarnings("ignore")

CSV  = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
OUT  = Path("/home/aih/dinesh.haridoss/chicago_mil/results/acr_episode_analysis")

# Patients flagged by clinical team
KNOWN_MULTI_EPISODE = {
    "LT035": 2, "LT038": 2, "LT084": 2, "LT157": 2, "LT169": 2,
    "LT146": 3, "LT239": 3,
}
KNOWN_MULTI_POS = set("""LT119 LT004 LT019 LT035 LT011 LT025 LT032 LT041
    LT081 LT084 LT092 LT101 LT146 LT168 LT169 LT173 LT239 LT302 LT008
    LT013 LT021 LT024 LT027 LT038 LT057 LT062 LT127 LT130 LT157 LT175
    LT176 LT206 LT238 LT244 LT252 LT305""".split())


# ── Episode detection ────────────────────────────────────────────────────────

def _is_acr_pos(grade) -> bool:
    g = str(grade) if pd.notna(grade) else ""
    return g.startswith("A1") or g.startswith("A2")


def detect_episodes(pid_df: pd.DataFrame) -> list:
    """
    Episode definition (from clinical team):
      Start : first >A0 (A1/A2) biopsy
      End   :
        - A0 biopsy encountered         → episode ends at that A0 date (closed)
        - Back-to-back A1s (no A0 between):
            ≤ 90 days apart             → same episode
            > 90 days apart             → end current as unresolved, start new
        - No A0 after last A1           → unresolved / right-censored (end_date=None)
      Clinical episode labels (start/end dates) supersede biopsy-derived labels
      when available (not yet implemented — placeholder for future).
    """
    df = pid_df.sort_values("anchor_dt").reset_index(drop=True)
    episodes = []
    i = 0
    while i < len(df):
        if not _is_acr_pos(df.loc[i, "acr_grade"]):
            i += 1
            continue

        # ── start a new episode ──────────────────────────────────────────
        ep_rows      = [df.loc[i].to_dict()]
        last_pos_dt  = df.loc[i, "anchor_dt"]
        end_date     = None      # None = unresolved / right-censored
        j            = i + 1

        while j < len(df):
            curr_dt    = df.loc[j, "anchor_dt"]
            curr_grade = df.loc[j, "acr_grade"]

            if _is_acr_pos(curr_grade):
                days_gap = (curr_dt - last_pos_dt).days
                if days_gap <= 90:
                    # back-to-back within 90 d → same episode
                    ep_rows.append(df.loc[j].to_dict())
                    last_pos_dt = curr_dt
                    j += 1
                else:
                    # > 90 d gap between positives → close unresolved, new episode
                    break   # end_date stays None (unresolved)
            else:
                # A0 biopsy → episode ends here
                end_date = curr_dt
                j += 1   # advance past the closing A0
                break
        # end while

        start_date = ep_rows[0]["anchor_dt"]
        duration   = (end_date - start_date).days if end_date is not None else None
        episodes.append({
            "start_date":    start_date,
            "end_date":      end_date,          # None = unresolved
            "unresolved":    end_date is None,
            "duration_days": duration,
            "n_biopsies":    len(ep_rows),
            "n_acr_pos":     len(ep_rows),
            "start_day":     int(ep_rows[0]["days_from_tx"]),
            "end_day":       int(ep_rows[-1]["days_from_tx"]),
            "grades":        [r["acr_grade"] for r in ep_rows],
            "biopsies":      ep_rows,
        })
        i = j
    return episodes


def inter_episode_gap(episodes: list) -> list:
    """Days between end of episode N and start of episode N+1.
    Returns None for gaps where episode N is unresolved (no closing A0)."""
    return [
        (episodes[k+1]["start_date"] - episodes[k]["end_date"]).days
        if episodes[k]["end_date"] is not None else None
        for k in range(len(episodes) - 1)
    ]


# ── Load and annotate data ───────────────────────────────────────────────────

def load_data(csv: str) -> pd.DataFrame:
    df = pd.read_csv(csv, parse_dates=["anchor_dt"])
    tx = df.groupby("patient_id")["anchor_dt"].min().rename("transplant_dt")
    df = df.merge(tx, on="patient_id")
    df["days_from_tx"] = (df["anchor_dt"] - df["transplant_dt"]).dt.days
    return df


def build_episode_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, pdata in df.groupby("patient_id"):
        eps  = detect_episodes(pdata)
        gaps = inter_episode_gap(eps)
        n_acr_pos = sum(
            1 for _, r in pdata.iterrows()
            if isinstance(r["acr_grade"], str)
            and (r["acr_grade"].startswith("A1") or r["acr_grade"].startswith("A2"))
        )
        n_unresolved = sum(1 for e in eps if e.get("unresolved", False))
        rows.append({
            "patient_id":        pid,
            "n_biopsies":        len(pdata),
            "n_acr_pos":         n_acr_pos,
            "n_episodes":        len(eps),
            "n_unresolved_ep":   n_unresolved,
            "episode_durations": [e["duration_days"] for e in eps],   # None = unresolved
            "episode_sizes":     [e["n_biopsies"]    for e in eps],
            "inter_ep_gaps":     gaps,
            "first_ep_start_day": eps[0]["start_day"] if eps else None,
            "last_ep_end_day":    eps[-1]["end_day"]   if eps else None,
            "clad_status":   pdata["clad_status"].iloc[0],
            "death_status":  pdata["death_status"].iloc[0],
            "fu_days":       int(pdata["days_from_tx"].max()),
            "known_n_ep":    KNOWN_MULTI_EPISODE.get(pid, None),
            "known_multi_pos": pid in KNOWN_MULTI_POS,
        })
    return pd.DataFrame(rows)


# ── Plots ────────────────────────────────────────────────────────────────────

C = dict(
    acr0="#3A86FF", acr1="#E63946", no_label="#AAAAAA",
    ep=["#E63946","#FF9F1C","#2DC653","#7B2D8B","#1565C0"],
    gap="#999FAE", spine="#C8CDD4", bg="#FFF5F5",
)


def plot_episode_stats(ep_df: pd.DataFrame, out: Path):
    """Cohort-level episode statistics plots."""
    has_ep = ep_df[ep_df["n_episodes"] > 0]
    multi  = ep_df[ep_df["n_episodes"] > 1]

    all_durations = [d for row in has_ep["episode_durations"] for d in row if d is not None]
    all_sizes     = [s for row in has_ep["episode_sizes"]     for s in row]
    all_gaps      = [g for row in multi["inter_ep_gaps"] for g in row if g is not None]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="white")
    fig.suptitle("ACR Episode Statistics", fontsize=16, fontweight="bold")

    # 1. Episodes per patient
    ax = axes[0, 0]
    counts = ep_df["n_episodes"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="#E63946", edgecolor="white")
    ax.set_xlabel("Number of ACR episodes per patient"); ax.set_ylabel("# patients")
    ax.set_title("Episodes per patient")
    for x, y in zip(counts.index, counts.values):
        ax.text(str(x), y + 0.3, str(y), ha="center", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)

    # 2. Episode duration (days) — how long is one continuous ACR+ run
    ax = axes[0, 1]
    ax.hist(all_durations, bins=20, color="#E63946", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(all_durations), color="black", lw=2, linestyle="--",
               label=f"median={np.median(all_durations):.0f}d")
    ax.set_xlabel("Episode duration (days)"); ax.set_ylabel("# episodes")
    ax.set_title(f"Episode duration  (n={len(all_durations)} episodes)")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)

    # 3. Biopsies per episode
    ax = axes[0, 2]
    ax.hist(all_sizes, bins=range(1, max(all_sizes)+2), color="#FF9F1C",
            edgecolor="white", alpha=0.85, align="left")
    ax.set_xlabel("# ACR+ biopsies per episode"); ax.set_ylabel("# episodes")
    ax.set_title(f"Biopsies per episode  (n={len(all_sizes)})")
    ax.spines[["top","right"]].set_visible(False)

    # 4. Inter-episode gap
    ax = axes[1, 0]
    if all_gaps:
        ax.hist(all_gaps, bins=20, color="#7B2D8B", edgecolor="white", alpha=0.8)
        ax.axvline(np.median(all_gaps), color="black", lw=2, linestyle="--",
                   label=f"median={np.median(all_gaps):.0f}d")
        ax.set_xlabel("Gap between episodes (days)"); ax.set_ylabel("# gaps")
        ax.set_title(f"Inter-episode gap  (n={len(all_gaps)} gaps)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No multi-episode patients", ha="center", transform=ax.transAxes)
    ax.spines[["top","right"]].set_visible(False)

    # 5. First episode start (days from transplant)
    ax = axes[1, 1]
    starts = has_ep["first_ep_start_day"].dropna()
    ax.hist(starts, bins=25, color="#1565C0", edgecolor="white", alpha=0.8)
    ax.axvline(starts.median(), color="black", lw=2, linestyle="--",
               label=f"median={starts.median():.0f}d")
    ax.set_xlabel("Days from transplant to first episode")
    ax.set_ylabel("# patients")
    ax.set_title("Time to first ACR episode")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)

    # 6. Multi-episode: known vs detected
    ax = axes[1, 2]
    known_detected = ep_df[ep_df["known_n_ep"].notna()][["patient_id","n_episodes","known_n_ep"]]
    x = range(len(known_detected))
    ax.bar([i - 0.2 for i in x], known_detected["known_n_ep"], width=0.35,
           color="#2DC653", alpha=0.85, label="Clinical team (known)")
    ax.bar([i + 0.2 for i in x], known_detected["n_episodes"], width=0.35,
           color="#E63946", alpha=0.85, label="Detected by code")
    ax.set_xticks(list(x))
    ax.set_xticklabels(known_detected["patient_id"].tolist(), rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("# episodes"); ax.set_title("Known vs detected episodes")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out / "acr_episode_stats.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: acr_episode_stats.png")


def plot_patient_timeline(pid: str, pdata: pd.DataFrame, episodes: list,
                          out: Path):
    """Detailed timeline for one patient showing all episodes."""
    pdata = pdata.sort_values("days_from_tx").reset_index(drop=True)
    fu    = int(pdata["days_from_tx"].max())
    fig, ax = plt.subplots(figsize=(max(14, fu / 80), 5), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])

    # Spine
    x_lo = -50; x_hi = fu + 100
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(-1.2, 1.8)
    ax.plot([0, fu], [0, 0], color=C["spine"], lw=2.5, solid_capstyle="round", zorder=1)
    ax.set_xlabel("Days from transplant", fontsize=11)
    ax.set_yticks([])
    ax.spines[["top","right","left"]].set_visible(False)

    # Year gridlines
    for yr in range(1, fu // 365 + 2):
        xyr = yr * 365
        if xyr < x_hi:
            ax.axvline(xyr, color=C["spine"], lw=0.7, linestyle="--", alpha=0.4)
            ax.text(xyr, 1.65, f"yr{yr}", fontsize=7, ha="center", color=C["spine"])

    # Color episodes
    for ei, ep in enumerate(episodes):
        col = C["ep"][ei % len(C["ep"])]
        x0   = ep["start_day"]
        x1   = ep["end_day"]
        unresolved = ep.get("unresolved", False)
        x1_draw = x1 if not unresolved else min(x0 + max(fu // 5, 60), fu + 60)

        ax.fill_betweenx([-0.25, 0.25], x0 - 5, x1_draw + 5,
                         color=col, alpha=0.18, zorder=2)
        ax.fill_betweenx([-0.1, 0.1], x0 - 5, x1_draw + 5,
                         color=col, alpha=0.35, zorder=2)

        dur_str = f"{ep['duration_days']}d" if ep.get("duration_days") is not None else "unresolved"
        label   = f"Ep {ei+1}  ({dur_str},  {ep['n_biopsies']} Bx)"
        if unresolved:
            label += " →"
            # dashed right-arrow to indicate right-censored
            ax.annotate("", xy=(x1_draw + 30, 0), xytext=(x1_draw, 0),
                        arrowprops=dict(arrowstyle="->", color=col, lw=2),
                        zorder=5)

        mid = (x0 + x1_draw) / 2
        ax.text(mid, 0.45, label,
                ha="center", fontsize=9, color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.85))

    # Inter-episode gaps
    gaps = inter_episode_gap(episodes)
    for ei, gap in enumerate(gaps):
        x0 = episodes[ei]["end_day"]
        x1 = episodes[ei+1]["start_day"]
        if x0 is None:
            continue  # unresolved episode — can't draw gap arrow
        ax.annotate("", xy=(x1, -0.55), xytext=(x0, -0.55),
                    arrowprops=dict(arrowstyle="<->", color=C["gap"], lw=1.8))
        gap_label = f"gap: {gap}d" if gap is not None else "gap: ?"
        ax.text((x0 + x1) / 2, -0.72, gap_label,
                ha="center", fontsize=8.5, color=C["gap"])

    # Biopsy dots
    for _, row in pdata.iterrows():
        grade = str(row.get("acr_grade", "")) if pd.notna(row.get("acr_grade")) else ""
        is_pos = grade.startswith("A1") or grade.startswith("A2")
        col   = C["acr1"] if is_pos else C["acr0"]
        size  = 120 if is_pos else 50
        ax.scatter(float(row["days_from_tx"]), 0,
                   color=col, s=size, zorder=6,
                   edgecolors="white", linewidth=0.9, alpha=0.95)
        ax.text(float(row["days_from_tx"]), -0.3,
                str(row.get("acr_grade", ""))[:4],
                ha="center", fontsize=6.5, color=col, rotation=60)

    # CLAD / Death markers
    cs = pdata["clad_status"].iloc[0]; cd = pdata["clad_days"].iloc[0]
    ds = pdata["death_status"].iloc[0]; dd = pdata["death_days"].iloc[0]
    if cs == 1 and pd.notna(cd):
        ax.axvline(float(cd), color="#FB8500", lw=2.2, linestyle="--", alpha=0.9)
        ax.text(float(cd) + 8, 1.3, f"CLAD @ {int(cd)}d",
                fontsize=9, color="#FB8500", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8EC", ec="#FB8500", alpha=0.9))
    if ds == 1 and pd.notna(dd):
        ax.axvline(float(dd), color="#1B2A3B", lw=2, linestyle=":", alpha=0.85)
        ax.text(float(dd) + 8, -0.9, f"† {int(dd)}d",
                fontsize=9, color="#1B2A3B", fontweight="bold")

    known = KNOWN_MULTI_EPISODE.get(pid, "")
    known_str = f"  [clinical team: {known} episodes]" if known else ""
    ax.set_title(
        f"Patient {pid} — {len(episodes)} ACR episode(s){known_str}   "
        f"FU={fu}d   n={len(pdata)} biopsies",
        fontsize=12, fontweight="bold", color=C["acr1"] if episodes else "#333",
    )

    legend_items = [
        mpatches.Patch(color=C["acr1"], label="ACR+ (A1/A2)"),
        mpatches.Patch(color=C["acr0"], label="ACR− (A0)"),
    ] + [
        mpatches.Patch(color=C["ep"][i % len(C["ep"])], alpha=0.5,
                       label=f"Episode {i+1} "
                             f"({'unresolved' if ep['duration_days'] is None else str(ep['duration_days'])+'d'}"
                             f", {ep['n_biopsies']} Bx)")
        for i, ep in enumerate(episodes)
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8.5,
              framealpha=0.92, edgecolor=C["spine"])

    plt.tight_layout()
    fig.savefig(out / f"{pid}_timeline.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_episode_detail(pid: str, pdata: pd.DataFrame, episodes: list,
                        ep_idx: int, out: Path):
    """Zoomed plot of a single episode with ±60d padding, showing all annotations."""
    ep    = episodes[ep_idx]
    pdata = pdata.sort_values("days_from_tx").reset_index(drop=True)
    fu    = int(pdata["days_from_tx"].max())

    x0  = ep["start_day"]
    x1  = ep["end_day"]
    pad = max(60, (x1 - x0) // 3)
    xlo = max(-50, x0 - pad)
    xhi = min(fu + 80, x1 + pad)

    fig, ax = plt.subplots(figsize=(max(10, (xhi - xlo) / 30), 4.5), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(xlo, xhi); ax.set_ylim(-1.2, 1.8)
    ax.plot([0, fu], [0, 0], color=C["spine"], lw=2.5, solid_capstyle="round", zorder=1)
    ax.set_xlabel("Days from transplant", fontsize=11)
    ax.set_yticks([])
    ax.spines[["top","right","left"]].set_visible(False)

    # Year gridlines
    for yr in range(1, fu // 365 + 2):
        xyr = yr * 365
        if xlo < xyr < xhi:
            ax.axvline(xyr, color=C["spine"], lw=0.7, linestyle="--", alpha=0.4)
            ax.text(xyr, 1.65, f"yr{yr}", fontsize=7, ha="center", color=C["spine"])

    # Episode shading
    col        = C["ep"][ep_idx % len(C["ep"])]
    unresolved = ep.get("unresolved", False)
    x1_draw    = x1 if not unresolved else min(x0 + max(fu // 5, 60), fu + 60)
    ax.fill_betweenx([-0.25, 0.25], x0 - 5, x1_draw + 5, color=col, alpha=0.18, zorder=2)
    ax.fill_betweenx([-0.1,  0.1], x0 - 5, x1_draw + 5, color=col, alpha=0.35, zorder=2)
    dur_str = f"{ep['duration_days']}d" if ep["duration_days"] is not None else "unresolved"
    label   = f"Episode {ep_idx+1}  ({dur_str}, {ep['n_biopsies']} Bx)"
    if unresolved:
        label += " →"
        ax.annotate("", xy=(x1_draw + 30, 0), xytext=(x1_draw, 0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=2), zorder=5)
    ax.text((x0 + x1_draw) / 2, 0.45, label,
            ha="center", fontsize=10, color=col, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.85))

    # Biopsy dots (all in view window + a bit outside so context is visible)
    for _, row in pdata.iterrows():
        d = float(row["days_from_tx"])
        if d < xlo - 30 or d > xhi + 30:
            continue
        grade  = str(row.get("acr_grade", "")) if pd.notna(row.get("acr_grade")) else ""
        is_pos = grade.startswith("A1") or grade.startswith("A2")
        c_dot  = C["acr1"] if is_pos else C["acr0"]
        ax.scatter(d, 0, color=c_dot, s=130 if is_pos else 55,
                   zorder=6, edgecolors="white", linewidth=0.9, alpha=0.95)
        ax.text(d, -0.3, grade[:4], ha="center", fontsize=7, color=c_dot, rotation=60)

    # CLAD / Death
    cs = pdata["clad_status"].iloc[0]; cd = pdata["clad_days"].iloc[0]
    ds = pdata["death_status"].iloc[0]; dd = pdata["death_days"].iloc[0]
    if cs == 1 and pd.notna(cd) and xlo < float(cd) < xhi:
        ax.axvline(float(cd), color="#FB8500", lw=2.2, linestyle="--", alpha=0.9)
        ax.text(float(cd) + 8, 1.3, f"CLAD @ {int(cd)}d",
                fontsize=9, color="#FB8500", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8EC", ec="#FB8500", alpha=0.9))
    if ds == 1 and pd.notna(dd) and xlo < float(dd) < xhi:
        ax.axvline(float(dd), color="#1B2A3B", lw=2, linestyle=":", alpha=0.85)
        ax.text(float(dd) + 8, -0.9, f"† {int(dd)}d", fontsize=9, color="#1B2A3B",
                fontweight="bold")

    ax.set_title(
        f"Patient {pid} — Episode {ep_idx+1} of {len(episodes)}   "
        f"({dur_str})   FU={fu}d",
        fontsize=12, fontweight="bold", color=col,
    )
    plt.tight_layout()
    fname = out / f"{pid}_episode_{ep_idx+1}.png"
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_multi_episode_grid(df: pd.DataFrame, ep_df: pd.DataFrame, out: Path):
    """Grid of timeline thumbnails for all multi-episode patients."""
    multi_pids = ep_df[ep_df["n_episodes"] > 1].sort_values(
        "n_episodes", ascending=False)["patient_id"].tolist()
    if not multi_pids:
        print("  No multi-episode patients found"); return

    ncols = 3
    nrows = int(np.ceil(len(multi_pids) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 8, nrows * 3.5),
                             facecolor="white")
    axes = np.array(axes).flatten()
    fig.suptitle("Multi-episode ACR patients", fontsize=15, fontweight="bold")

    for idx, pid in enumerate(multi_pids):
        ax = axes[idx]
        pdata = df[df["patient_id"] == pid].sort_values("days_from_tx")
        eps   = detect_episodes(pdata)
        fu    = int(pdata["days_from_tx"].max())

        ax.set_facecolor(C["bg"])
        ax.set_xlim(-30, fu + 80); ax.set_ylim(-0.8, 1.0)
        ax.plot([0, fu], [0, 0], color=C["spine"], lw=2, solid_capstyle="round")
        ax.set_yticks([]); ax.spines[["top","right","left"]].set_visible(False)

        for ei, ep in enumerate(eps):
            col = C["ep"][ei % len(C["ep"])]
            ax.fill_betweenx([-0.2, 0.2], ep["start_day"]-3, ep["end_day"]+3,
                             color=col, alpha=0.35, zorder=2)
            dur_s = f"{ep['duration_days']}d" if ep.get("duration_days") is not None else "?"
            ax.text((ep["start_day"] + ep["end_day"]) / 2, 0.45,
                    f"Ep{ei+1}\n{dur_s}",
                    ha="center", fontsize=7, color=col, fontweight="bold")

        for _, row in pdata.iterrows():
            grade = str(row.get("acr_grade","")) if pd.notna(row.get("acr_grade")) else ""
            is_pos = grade.startswith("A1") or grade.startswith("A2")
            ax.scatter(float(row["days_from_tx"]), 0,
                       color=C["acr1"] if is_pos else C["acr0"],
                       s=60 if is_pos else 25,
                       zorder=5, edgecolors="white", linewidth=0.5)

        known = KNOWN_MULTI_EPISODE.get(pid, "")
        ep_r  = ep_df[ep_df["patient_id"] == pid].iloc[0]
        gaps  = inter_episode_gap(eps)
        gap_str = f"  gap: {gaps[0]}d" if gaps else ""
        ax.set_title(
            f"{pid}  [{len(eps)} ep, {ep_r['n_acr_pos']} ACR+ Bx]{gap_str}"
            + (f"  ✓known={known}" if known else ""),
            fontsize=8.5, fontweight="bold",
            color="#E63946" if (known and len(eps) == known) else
                  "#FF9F1C" if (known and len(eps) != known) else "#333",
        )
        ax.set_xlabel(f"FU={fu}d  n={len(pdata)}Bx", fontsize=7, color="#777")

    for ax in axes[len(multi_pids):]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(out / "multi_episode_grid.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: multi_episode_grid.png  ({len(multi_pids)} patients)")


def print_stats(ep_df: pd.DataFrame):
    n_total   = len(ep_df)
    n_acr     = (ep_df["n_acr_pos"] > 0).sum()
    n_ep1     = (ep_df["n_episodes"] == 1).sum()
    n_ep2     = (ep_df["n_episodes"] == 2).sum()
    n_ep3p    = (ep_df["n_episodes"] >= 3).sum()

    all_dur  = [d for row in ep_df["episode_durations"] for d in row if d is not None]
    all_gaps = [g for row in ep_df["inter_ep_gaps"] for g in row if g is not None]

    print(f"\n{'='*60}")
    print(f"  ACR EPISODE ANALYSIS")
    print(f"{'='*60}")
    print(f"  Total patients       : {n_total}")
    print(f"  Any ACR+ biopsy      : {n_acr}  ({100*n_acr/n_total:.1f}%)")
    print(f"  1 episode            : {n_ep1}")
    print(f"  2 episodes           : {n_ep2}")
    print(f"  3+ episodes          : {n_ep3p}")
    print(f"")
    if all_dur:
        print(f"  Episode duration (days):")
        print(f"    median={np.median(all_dur):.0f}  mean={np.mean(all_dur):.0f}"
              f"  max={max(all_dur)}  min={min(all_dur)}")
    if all_gaps:
        print(f"  Inter-episode gap (days):")
        print(f"    median={np.median(all_gaps):.0f}  mean={np.mean(all_gaps):.0f}"
              f"  max={max(all_gaps)}  min={min(all_gaps)}")

    n_unres = ep_df["n_unresolved_ep"].sum() if "n_unresolved_ep" in ep_df.columns else "?"
    print(f"  Unresolved episodes  : {n_unres}  (no closing A0)")

    print(f"\n  {'Patient':<12} {'Detected ep':>11} {'Known ep':>9} {'ACR+ Bx':>8} "
          f"{'Ep durations':>20} {'Gaps':>12} {'Match':>6}")
    print(f"  {'-'*80}")

    for _, row in ep_df[ep_df["n_episodes"] > 0].sort_values(
            "n_episodes", ascending=False).iterrows():
        pid   = row["patient_id"]
        known = row["known_n_ep"]
        known_int = int(known) if (known is not None and not (isinstance(known, float) and np.isnan(known))) else None
        match = ("✓" if known_int and int(row["n_episodes"]) == known_int
                 else "✗" if known_int else "")
        dur_str = " ".join(
            f"{d}d" if d is not None else "?" for d in row["episode_durations"]
        )
        gap_str = " ".join(f"{g}d" if g is not None else "?" for g in row["inter_ep_gaps"])
        print(f"  {pid:<12} {row['n_episodes']:>11} {str(known or ''):>9} "
              f"{row['n_acr_pos']:>8} {dur_str:>20} {gap_str:>12} {match:>6}")


def print_agreement(ep_df: pd.DataFrame):
    """Full agreement table vs clinical annotations."""
    print(f"\n{'='*70}")
    print("AGREEMENT WITH CLINICAL ANNOTATIONS")
    print(f"{'='*70}")

    cats = {
        "✓ EXACT MATCH":         [],
        "✓ MULTI CONFIRMED":     [],
        "✗ WRONG COUNT":         [],
        "? CLINICAL=MULTI WE=1": [],
        "? CLINICAL=MULTI WE=0": [],
        "? WE≥2 NOT IN LIST":    [],
    }

    for _, row in ep_df.iterrows():
        pid  = row["patient_id"]
        n    = int(row["n_episodes"])
        k    = row["known_n_ep"]
        cm   = row["known_multi_pos"]
        k_i  = int(k) if (k is not None and not (isinstance(k, float) and np.isnan(k))) else None
        durs = " / ".join(f"{d}d" if d is not None else "?" for d in row["episode_durations"])
        gaps = " / ".join(f"{g}d" for g in row["inter_ep_gaps"])
        entry = (pid, n, k_i, int(row["n_acr_pos"]), durs, gaps)
        if k_i is not None:
            cats["✓ EXACT MATCH" if n==k_i else "✗ WRONG COUNT"].append(entry)
        elif cm and n>=2: cats["✓ MULTI CONFIRMED"].append(entry)
        elif cm and n==1: cats["? CLINICAL=MULTI WE=1"].append(entry)
        elif cm and n==0: cats["? CLINICAL=MULTI WE=0"].append(entry)
        elif not cm and n>=2: cats["? WE≥2 NOT IN LIST"].append(entry)

    for label, entries in cats.items():
        if not entries: continue
        print(f"\n  {label}  [{len(entries)}]")
        print(f"  {'Patient':<10} {'Ours':>4} {'Clin':>5} {'ACR+':>5}  "
              f"{'Episode durations':<24}  Gaps between")
        print(f"  {'-'*75}")
        for pid, n, k, npos, durs, gaps in sorted(entries, key=lambda x: -x[1]):
            k_s = str(k) if k else ("multi" if label.startswith("?") else "?")
            print(f"  {pid:<10} {n:>4} {k_s:>5} {npos:>5}  {durs:<24}  {gaps}")

    # Summary line
    print(f"\n  Exact match     : {len(cats['✓ EXACT MATCH'])}/7")
    print(f"  Multi confirmed : {len(cats['✓ MULTI CONFIRMED'])}")
    print(f"  Wrong count     : {len(cats['✗ WRONG COUNT'])}  "
          f"← long unbroken runs, need treatment dates")
    print(f"  Clinical=multi, we=1 : {len(cats['? CLINICAL=MULTI WE=1'])}")
    print(f"  Missing grades  : {len(cats['? CLINICAL=MULTI WE=0'])}")


def _plot_not_in_data(pid: str, out: Path):
    """Empty placeholder timeline for a patient not present in our dataset."""
    fig, ax = plt.subplots(figsize=(10, 3), facecolor="white")
    ax.set_facecolor("#f8f8f8")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.62, f"Patient {pid}", ha="center", va="center",
            fontsize=18, fontweight="bold", color="#333333")
    ax.text(0.5, 0.42, "Not present in our dataset", ha="center", va="center",
            fontsize=13, color="#C62828")
    ax.text(0.5, 0.26, "Listed by clinical team as multi-ACR+ patient",
            ha="center", va="center", fontsize=10, color="#888888")
    fig.tight_layout()
    fig.savefig(out / f"{pid}_timeline.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df     = load_data(CSV)
    ep_df  = build_episode_table(df)

    print_stats(ep_df)
    print_agreement(ep_df)

    # Save episode table
    ep_df_out = ep_df.copy()
    for col in ["episode_durations","episode_sizes","inter_ep_gaps"]:
        ep_df_out[col] = ep_df_out[col].apply(str)
    ep_df_out.to_csv(OUT / "episode_table.csv", index=False)
    print(f"\n  Episode table → {OUT / 'episode_table.csv'}")

    print("\nGenerating plots...")
    plot_episode_stats(ep_df, OUT)
    plot_multi_episode_grid(df, ep_df, OUT)

    # ── 1. Full-timeline plot for ALL patients ──────────────────────────────
    all_dir = OUT / "all_patient_timelines"
    all_dir.mkdir(exist_ok=True)

    all_pids_in_data = sorted(df["patient_id"].unique())
    print(f"\n  Plotting all {len(all_pids_in_data)} patients → {all_dir.name}/")
    for pid in all_pids_in_data:
        pdata = df[df["patient_id"] == pid]
        eps   = detect_episodes(pdata)
        plot_patient_timeline(pid, pdata, eps, all_dir)

    # Placeholder plots for clinical-listed patients not in our dataset
    not_in_data = sorted(KNOWN_MULTI_POS - set(all_pids_in_data))
    for pid in not_in_data:
        _plot_not_in_data(pid, all_dir)

    print(f"  Saved {len(all_pids_in_data) + len(not_in_data)} full timeline PNGs")

    # ── 2. Per-episode zoom for patients with ≥2 episodes ──────────────────
    ep_detail_dir = OUT / "episode_detail"
    ep_detail_dir.mkdir(exist_ok=True)

    multi_ep_pids = ep_df[ep_df["n_episodes"] >= 2]["patient_id"].tolist()
    print(f"\n  Plotting per-episode detail for {len(multi_ep_pids)} multi-episode patients")
    for pid in multi_ep_pids:
        pdata = df[df["patient_id"] == pid]
        eps   = detect_episodes(pdata)
        pid_dir = ep_detail_dir / pid
        pid_dir.mkdir(exist_ok=True)
        for ei in range(len(eps)):
            plot_episode_detail(pid, pdata, eps, ei, pid_dir)

    n_ep_plots = sum(
        int(row["n_episodes"]) for _, row in ep_df[ep_df["n_episodes"] >= 2].iterrows()
    )
    print(f"  Saved {n_ep_plots} episode-detail PNGs → {ep_detail_dir.name}/")

    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
