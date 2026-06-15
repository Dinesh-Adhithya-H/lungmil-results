#!/usr/bin/env python3
"""
Generate all per-patient static plots for the GitHub Pages website.

Inputs  : patient_explorer/data/*.csv
          results/acr_episode_analysis/all_patient_timelines/*.png
Outputs : docs/patients/{pid}/*.png
          docs/cohort/*.png
          docs/data/patients.json
"""
from __future__ import annotations
import json, shutil, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
REPO     = Path(__file__).parent.parent
DATA_DIR = REPO / "patient_explorer" / "data"
EP_DIR    = REPO / "results" / "acr_episode_analysis" / "all_patient_timelines"
EP_DETAIL = REPO / "results" / "acr_episode_analysis" / "episode_detail"
EP_STATS  = REPO / "results" / "acr_episode_analysis"
DOCS_DIR = REPO / "docs"
PAT_DIR  = DOCS_DIR / "patients"
COH_DIR  = DOCS_DIR / "cohort"
JSON_DIR = DOCS_DIR / "data"

# ── Palette ──────────────────────────────────────────────────────────────────
CP   = "#E63946"
CN   = "#3A86FF"
CCL  = "#FB8500"
CDT  = "#1B2A3B"
CSPN = "#C8CDD4"
_BASE_PAL = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)

def pal(n):
    return [_BASE_PAL[i % len(_BASE_PAL)] for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_all():
    splits = pd.read_csv(DATA_DIR / "splits.csv", parse_dates=["anchor_dt"])
    tx_min = splits.groupby("patient_id")["anchor_dt"].transform("min")
    splits["days_from_tx"] = (splits["anchor_dt"] - tx_min).dt.days

    preds = None
    if (DATA_DIR / "predictions_all.csv").exists():
        preds = pd.read_csv(DATA_DIR / "predictions_all.csv", parse_dates=["anchor_dt"])

    ep_df = None
    if (DATA_DIR / "episodes.csv").exists():
        ep_df = pd.read_csv(DATA_DIR / "episodes.csv")

    umap = None
    if (DATA_DIR / "umap_embeddings.csv").exists():
        umap = pd.read_csv(DATA_DIR / "umap_embeddings.csv", parse_dates=["anchor_dt"])

    clinical, feat_names = None, {}
    if (DATA_DIR / "clinical_features.csv").exists():
        clinical = pd.read_csv(DATA_DIR / "clinical_features.csv", parse_dates=["anchor_dt"])
    if (DATA_DIR / "clinical_feature_names.csv").exists():
        fn = pd.read_csv(DATA_DIR / "clinical_feature_names.csv")
        feat_names = dict(zip(fn["idx"].astype(int), fn["name"].astype(str)))

    clusters = {}
    for mod in ("he", "bal", "ct"):
        p = DATA_DIR / f"{mod}_cluster_freq.csv"
        if not p.exists():
            continue
        cdf = pd.read_csv(p, parse_dates=["anchor_dt"])
        # compute days_from_tx if missing
        if "patient_id" in cdf.columns and "anchor_dt" in cdf.columns:
            tx_map = splits.groupby("patient_id")["anchor_dt"].min().to_dict()
            cdf["days_from_tx"] = cdf.apply(
                lambda r: (r["anchor_dt"] - tx_map.get(r["patient_id"], r["anchor_dt"])).days
                if pd.notna(r.get("anchor_dt")) else np.nan, axis=1
            )
        clusters[mod] = cdf

    return splits, preds, ep_df, umap, clinical, feat_names, clusters


# ══════════════════════════════════════════════════════════════════════════════
# Per-patient plots
# ══════════════════════════════════════════════════════════════════════════════

def copy_timeline(pid: str, dest_dir: Path) -> bool:
    src = EP_DIR / f"{pid}_timeline.png"
    if src.exists():
        shutil.copy(src, dest_dir / "timeline.png")
        return True
    return False


def copy_episode_details(pid: str, dest_dir: Path):
    """Copy per-episode zoom PNGs into patient folder as episode_detail_N.png."""
    pid_ep_dir = EP_DETAIL / pid
    if not pid_ep_dir.exists():
        return
    for src in sorted(pid_ep_dir.glob("*.png")):
        # src name: LT001_episode_1.png → dest: episode_detail_1.png
        parts = src.stem.split("_episode_")
        n = parts[-1] if len(parts) == 2 else src.stem
        shutil.copy(src, dest_dir / f"episode_detail_{n}.png")


def plot_clusters(pid: str, mod: str, cdf: pd.DataFrame,
                  pid_splits: pd.DataFrame, dest_dir: Path):
    rows = cdf[cdf["patient_id"] == pid].copy()
    if rows.empty:
        return

    clust_cols = sorted(
        [c for c in rows.columns if c.startswith("cluster_")],
        key=lambda x: int(x.split("_")[1])
    )
    if not clust_cols:
        return

    sort_col = "days_from_tx" if "days_from_tx" in rows.columns else "anchor_dt"
    rows = rows.sort_values(sort_col).reset_index(drop=True)

    # top-12 clusters by mean proportion; rest → "Other"
    means  = rows[clust_cols].mean()
    top12  = means.nlargest(12).index.tolist()
    other  = [c for c in clust_cols if c not in top12]
    mat    = rows[top12].values
    labels = [f"C{c.split('_')[1]}" for c in top12]
    if other:
        mat    = np.hstack([mat, rows[other].sum(axis=1).values.reshape(-1, 1)])
        labels = labels + ["Other"]

    colors = pal(len(labels))
    n_vis  = len(rows)
    fig, ax = plt.subplots(figsize=(max(8, n_vis * 0.9 + 2), 4.5), facecolor="white")
    ax.set_facecolor("white")

    x = np.arange(n_vis)
    bot = np.zeros(n_vis)
    for ci, (col, lbl) in enumerate(zip(colors, labels)):
        vals = mat[:, ci]
        ax.bar(x, vals, bottom=bot, color=col, label=lbl, width=0.82,
               edgecolor="white", linewidth=0.4)
        bot += vals

    # ACR grade triangles above bars
    acr_map = {}
    if "acr_grade" in pid_splits.columns and "days_from_tx" in pid_splits.columns:
        acr_map = pid_splits.set_index("days_from_tx")["acr_grade"].to_dict()
    for i, row in rows.iterrows():
        g = acr_map.get(row.get("days_from_tx", None), "")
        if isinstance(g, str) and (g.startswith("A1") or g.startswith("A2")):
            ax.scatter(i, 1.04, color=CP, s=50, marker="v", zorder=9, clip_on=False)

    # x-tick labels
    if "days_from_tx" in rows.columns:
        xlbls = [f"d{int(d)}" for d in rows["days_from_tx"]]
    elif "anchor_dt" in rows.columns:
        xlbls = [str(d)[:10] for d in rows["anchor_dt"]]
    else:
        xlbls = [str(i) for i in range(n_vis)]

    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Cluster proportion", fontsize=10)
    ax.set_title(
        f"{pid} · {mod.upper()} Cluster Composition  ({n_vis} visits)",
        fontsize=12, fontweight="bold"
    )
    ax.spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              fontsize=7, framealpha=0.9, ncol=1)

    plt.tight_layout()
    fig.savefig(dest_dir / f"clusters_{mod.upper()}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_clinical(pid: str, clinical_df: pd.DataFrame, feat_names: dict,
                  pid_splits: pd.DataFrame, dest_dir: Path):
    rows = clinical_df[clinical_df["patient_id"] == pid].copy()
    if rows.empty:
        return

    rows = rows.sort_values("anchor_dt").reset_index(drop=True)
    name_to_idx = {v.lower(): k for k, v in feat_names.items()}

    # Priority features
    priority = ["fvc", "fev1", "fev1_fvc", "fvcp", "fev1p",
                "albumin", "creatinine", "gfr", "glucose", "sodium"]
    selected = []
    for nm in priority:
        idx = name_to_idx.get(nm)
        if idx is None:
            idx = next((v for k, v in name_to_idx.items() if nm in k), None)
        if idx is not None:
            col = f"feat_{idx}"
            if col in rows.columns and rows[col].notna().any():
                selected.append((feat_names[idx], col))
        if len(selected) >= 6:
            break

    if not selected:
        return

    tx_dt = pid_splits["anchor_dt"].min() if not pid_splits.empty else rows["anchor_dt"].min()
    days  = (rows["anchor_dt"] - tx_dt).dt.days.values

    acr_pos_days = []
    if "acr_encoded" in pid_splits.columns:
        apd = pid_splits[pid_splits["acr_encoded"] == 1]
        acr_pos_days = ((apd["anchor_dt"] - tx_dt).dt.days).tolist()

    clad_day = None
    if "clad_status" in pid_splits.columns and pid_splits["clad_status"].iloc[0] == 1:
        if "clad_days" in pid_splits.columns:
            clad_day = float(pid_splits["clad_days"].iloc[0])

    n = len(selected)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 3.0),
                             facecolor="white", sharex=True)
    axes = np.array(axes).flatten()
    colors10 = plt.cm.tab10.colors

    for i, (name, col) in enumerate(selected):
        ax   = axes[i]
        vals = rows[col].values.astype(float)
        ok   = ~np.isnan(vals)
        ax.plot(days[ok], vals[ok], "-o", color=colors10[i % 10], lw=2,
                markersize=5, markerfacecolor="white", markeredgewidth=1.8)
        ax.fill_between(days[ok], vals[ok], alpha=0.08, color=colors10[i % 10])
        for d in acr_pos_days:
            ax.axvline(d, color=CP, lw=0.9, alpha=0.35, linestyle="--")
        if clad_day is not None:
            ax.axvline(clad_day, color=CCL, lw=1.3, alpha=0.7, linestyle=":")
        ax.set_title(name.upper(), fontsize=10, fontweight="bold",
                     color=colors10[i % 10])
        ax.set_ylabel(name, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[max(0, n - ncols):n]:
        ax.set_xlabel("Days from transplant", fontsize=8)

    fig.suptitle(f"{pid} · Clinical Features over Time",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(dest_dir / "clinical.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_umap(pid: str, umap_df: pd.DataFrame, pid_splits: pd.DataFrame,
              dest_dir: Path):
    others = umap_df[umap_df["patient_id"] != pid]
    this_p = umap_df[umap_df["patient_id"] == pid].sort_values("anchor_dt")
    if this_p.empty:
        return

    acr_map = {}
    if "acr_encoded" in pid_splits.columns:
        acr_map = pid_splits.set_index("anchor_dt")["acr_encoded"].to_dict()

    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(others["umap_x"], others["umap_y"],
               c="#CCCCCC", s=6, alpha=0.35, linewidths=0, rasterized=True, zorder=1)

    if len(this_p) > 1:
        ax.plot(this_p["umap_x"].values, this_p["umap_y"].values,
                "-", color="#555555", lw=1.5, alpha=0.5, zorder=2)

    for idx_r, row in this_p.iterrows():
        acr = acr_map.get(row["anchor_dt"], 0)
        c   = CP if acr == 1 else CN
        ax.scatter(row["umap_x"], row["umap_y"], c=c, s=90,
                   edgecolors="white", linewidths=1.5, zorder=4, alpha=0.95)

    handles = [
        mpatches.Patch(color=CP,      label=f"{pid} ACR+"),
        mpatches.Patch(color=CN,      label=f"{pid} ACR−"),
        mpatches.Patch(color="#CCCCCC", label="Other patients"),
    ]
    ax.legend(handles=handles, fontsize=9, framealpha=0.9, edgecolor=CSPN)
    ax.set_title(f"{pid} · Latent Space (UMAP)", fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(dest_dir / "umap.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(pid: str, preds_df: pd.DataFrame, pid_splits: pd.DataFrame,
                     dest_dir: Path):
    rows = preds_df[preds_df["patient_id"] == pid].copy()
    if rows.empty:
        return

    rows  = rows.sort_values("anchor_dt").reset_index(drop=True)
    tx_dt = pid_splits["anchor_dt"].min() if not pid_splits.empty else rows["anchor_dt"].min()
    days  = (rows["anchor_dt"] - tx_dt).dt.days.values

    score_defs = [
        ("pred_prob_acr",  "ACR probability",  CP),
        ("hazard_acr",     "ACR hazard",        "#FF6B6B"),
        ("hazard_clad",    "CLAD hazard",       CCL),
        ("hazard_death",   "Death hazard",      CDT),
    ]
    present = [(n, c, col) for col, n, c in
               [(d[0], d[1], d[2]) for d in score_defs]
               if col in rows.columns and rows[col].notna().any()]
    if not present:
        return

    acr_pos_days = []
    if "acr_encoded" in pid_splits.columns:
        apd = pid_splits[pid_splits["acr_encoded"] == 1]
        acr_pos_days = ((apd["anchor_dt"] - tx_dt).dt.days).tolist()

    clad_day, death_day = None, None
    if "clad_status" in pid_splits.columns and pid_splits["clad_status"].iloc[0] == 1:
        if "clad_days" in pid_splits.columns:
            clad_day = float(pid_splits["clad_days"].iloc[0])
    if "death_status" in pid_splits.columns and pid_splits["death_status"].iloc[0] == 1:
        if "death_days" in pid_splits.columns:
            death_day = float(pid_splits["death_days"].iloc[0])

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.7 + 3), 4),
                           facecolor="white")
    ax.set_facecolor("white")

    for name, color, col in present:
        vals  = rows[col].values.astype(float)
        ok    = ~np.isnan(vals)
        ax.plot(days[ok], vals[ok], "-o", color=color, lw=2.2,
                markersize=6, label=name,
                markerfacecolor="white", markeredgewidth=1.8)

    for d in acr_pos_days:
        ax.axvline(d, color=CP, lw=1.5, alpha=0.45, linestyle="--")
    if clad_day is not None:
        ax.axvline(clad_day, color=CCL, lw=2, alpha=0.8, linestyle="--",
                   label=f"CLAD @ {int(clad_day)}d")
    if death_day is not None:
        ax.axvline(death_day, color=CDT, lw=2, alpha=0.8, linestyle=":",
                   label=f"Death @ {int(death_day)}d")

    ax.set_xlabel("Days from transplant", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(f"{pid} · Model Predictions over Time", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(dest_dir / "predictions.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Cohort-level plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_cohort_umap(umap_df: pd.DataFrame, splits: pd.DataFrame, out: Path):
    """Global UMAP colored by ACR status."""
    if umap_df is None or umap_df.empty:
        return

    acr_map = {}
    if "acr_encoded" in splits.columns:
        acr_map = splits.set_index("anchor_dt").groupby("anchor_dt")["acr_encoded"].first().to_dict()

    merged = umap_df.copy()
    acr_by_stem = splits.set_index("stem")["acr_encoded"].to_dict() if "stem" in splits.columns else {}
    merged["acr"] = merged.get("stem", merged["patient_id"]).map(acr_by_stem).fillna(
        merged["anchor_dt"].map(acr_map)
    ).fillna(0).astype(int)

    clad_pids  = set(splits[splits.get("clad_status", pd.Series(0)) == 1]["patient_id"].unique()) \
                 if "clad_status" in splits.columns else set()
    death_pids = set(splits[splits.get("death_status", pd.Series(0)) == 1]["patient_id"].unique()) \
                 if "death_status" in splits.columns else set()

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), facecolor="white")
    fig.suptitle("Cohort UMAP — Latent Space", fontsize=14, fontweight="bold")

    for ax, (title, colorfn, legend) in zip(axes, [
        ("ACR Status",
         lambda r: CP if r["acr"] == 1 else CN,
         [mpatches.Patch(color=CP, label="ACR+"),
          mpatches.Patch(color=CN, label="ACR−")]),
        ("CLAD Status",
         lambda r: CCL if r["patient_id"] in clad_pids else "#CCCCCC",
         [mpatches.Patch(color=CCL, label="CLAD"), mpatches.Patch(color="#CCCCCC", label="No CLAD")]),
        ("Death Status",
         lambda r: CDT if r["patient_id"] in death_pids else "#CCCCCC",
         [mpatches.Patch(color=CDT, label="Death"), mpatches.Patch(color="#CCCCCC", label="Alive")]),
    ]):
        colors = [colorfn(r) for _, r in merged.iterrows()]
        ax.scatter(merged["umap_x"], merged["umap_y"],
                   c=colors, s=12, alpha=0.6, linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.legend(handles=legend, fontsize=9, framealpha=0.9)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out / "umap_cohort.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  cohort/umap_cohort.png")


def copy_cohort_plots(out: Path):
    """Copy episode stats plots."""
    for fname in ["acr_episode_stats.png", "multi_episode_grid.png"]:
        src = EP_STATS / fname
        if src.exists():
            shutil.copy(src, out / fname)
            print(f"  cohort/{fname}")


# ══════════════════════════════════════════════════════════════════════════════
# patients.json
# ══════════════════════════════════════════════════════════════════════════════

def build_patients_json(splits: pd.DataFrame, ep_df, preds_df) -> list:
    records = []
    for pid, grp in splits.groupby("patient_id"):
        grp = grp.sort_values("anchor_dt")
        n_bx  = len(grp)
        n_pos = int((grp["acr_encoded"] == 1).sum()) if "acr_encoded" in grp.columns else 0
        clad  = bool(grp["clad_status"].iloc[0])  if "clad_status"  in grp.columns else False
        death = bool(grp["death_status"].iloc[0]) if "death_status" in grp.columns else False
        fu    = int((grp["anchor_dt"].max() - grp["anchor_dt"].min()).days)

        n_ep = 0
        if ep_df is not None:
            ep_row = ep_df[ep_df["patient_id"] == pid]
            if not ep_row.empty:
                n_ep = int(ep_row["n_episodes"].iloc[0])

        mods = [m for m, k in [("HE","has_HE"),("BAL","has_BAL"),("CT","has_CT"),("Clinical","has_Clinical")]
                if k in grp.columns and grp[k].any()]

        pid_dir = PAT_DIR / pid
        plots   = [p.stem for p in sorted(pid_dir.glob("*.png"))] if pid_dir.exists() else []

        records.append({
            "pid":        pid,
            "n_biopsies": n_bx,
            "n_acr_pos":  n_pos,
            "n_episodes": n_ep,
            "clad":       clad,
            "death":      death,
            "fu_days":    fu,
            "modalities": mods,
            "plots":      plots,
        })

    records.sort(key=lambda r: (-r["n_acr_pos"], r["pid"]))
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    for d in [PAT_DIR, COH_DIR, JSON_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    splits, preds, ep_df, umap, clinical, feat_names, clusters = load_all()
    all_pids = sorted(splits["patient_id"].unique())
    print(f"  {len(all_pids)} patients | UMAP={'yes' if umap is not None else 'no'} | "
          f"Clinical={'yes' if clinical is not None else 'no'} | "
          f"Clusters={list(clusters.keys())}")

    # ── Cohort plots ──────────────────────────────────────────────────────────
    print("\nCohort plots...")
    copy_cohort_plots(COH_DIR)
    if umap is not None:
        plot_cohort_umap(umap, splits, COH_DIR)

    # ── Per-patient plots ─────────────────────────────────────────────────────
    print(f"\nPatient plots  (0/{len(all_pids)})...")
    n_tl, n_cl, n_clin, n_um, n_pr = 0, 0, 0, 0, 0

    for i, pid in enumerate(all_pids):
        pid_splits = splits[splits["patient_id"] == pid]
        pid_dir    = PAT_DIR / pid
        pid_dir.mkdir(exist_ok=True)

        ok = copy_timeline(pid, pid_dir);            n_tl  += ok
        copy_episode_details(pid, pid_dir)
        for mod, cdf in clusters.items():
            plot_clusters(pid, mod, cdf, pid_splits, pid_dir);   n_cl += 1
        if clinical is not None:
            plot_clinical(pid, clinical, feat_names, pid_splits, pid_dir); n_clin += 1
        if umap is not None:
            plot_umap(pid, umap, pid_splits, pid_dir);           n_um += 1
        if preds is not None:
            plot_predictions(pid, preds, pid_splits, pid_dir);   n_pr += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(all_pids):
            print(f"  {i+1}/{len(all_pids)}")

    print(f"\n  timelines={n_tl}  cluster_plots={n_cl}  clinical={n_clin}  "
          f"umap={n_um}  predictions={n_pr}")

    # ── patients.json ─────────────────────────────────────────────────────────
    records = build_patients_json(splits, ep_df, preds)
    out_json = JSON_DIR / "patients.json"
    with open(out_json, "w") as f:
        json.dump(records, f, separators=(",", ":"))
    print(f"\n  data/patients.json → {len(records)} patients")
    print(f"\nAll outputs → {DOCS_DIR}")


if __name__ == "__main__":
    main()
