"""
Per-task event/TTE distributions — one 2×2 figure per task.
  Top-left  : Patient-level event counts (none / single / multiple where applicable)
  Top-right : Time from transplant to first event (patient-level histogram, events only)
  Bottom-left: Biopsy-level TTE histogram (events vs censored), TTE >= 0 only
  Bottom-right: Modality combo counts (biopsies)

For acr_cls: top-left = ACR label distribution, top-right = same TTE, bottom-left = label bar.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = Path(__file__).resolve().parent.parent
REPS_DIR = ROOT / "results" / "mm_abmil_v8" / "biopsy_reps"
MASTER   = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
OUT_DIR  = ROOT / "figures" / "cohort"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODS      = ["has_HE", "has_BAL", "has_CT", "has_Clinical"]
MOD_NAMES = ["HE", "BAL", "CT", "Clinical"]
BG        = "#FAF6F2"
C_CENS    = "#B0BEC5"
FS_TITLE  = 15
FS_LABEL  = 13
FS_TICK   = 11
FS_ANNOT  = 10

COMBO_COLORS = [
    "#1565C0","#2E7D32","#B71C1C","#6A1B9A",
    "#E65100","#00695C","#4E342E","#37474F",
    "#F57F17","#880E4F",
]

TASK_CFG = {
    "death_surv": {
        "label":     "Post-Transplant Mortality",
        "color":     "#546E7A",
        "ev_col":    "death_status",
        "day_col":   "death_days",
        "xlabel_tte":"Days from biopsy to death / censoring",
        "xlabel_tx": "Days from transplant to death",
        "bin_w":     180,
        "multi_acr": False,
    },
    "acr_surv": {
        "label":     "ACR Survival",
        "color":     "#EF5350",
        "ev_col":    None,           # derived from biopsy labels
        "day_col":   None,
        "xlabel_tte":"Days from biopsy to first ACR / censoring",
        "xlabel_tx": "Days from first biopsy to first ACR event",
        "bin_w":     60,
        "multi_acr": True,
    },
    "clad_surv": {
        "label":     "CLAD Survival",
        "color":     "#AB47BC",
        "ev_col":    "clad_status",
        "day_col":   "clad_days",
        "xlabel_tte":"Days from biopsy to CLAD / censoring",
        "xlabel_tx": "Days from transplant to CLAD",
        "bin_w":     180,
        "multi_acr": False,
    },
    "acr_cls": {
        "label":     "ACR Classification",
        "color":     "#EF5350",
        "ev_col":    None,
        "day_col":   None,
        "xlabel_tte": None,
        "xlabel_tx":  "Days from first biopsy to first ACR event",
        "bin_w":     60,
        "multi_acr": True,
    },
}

# ── Load master ───────────────────────────────────────────────────────────────
raw = pd.read_csv(MASTER)
raw["stem"] = raw["file"].str.replace(".pt","",regex=False)
raw["combo"] = raw[MODS].apply(
    lambda r: "+".join(n for n,m in zip(MOD_NAMES, MODS) if r[m]) or "None", axis=1)
stem_to_combo = raw.set_index("stem")["combo"].to_dict()

# Patient-level master
all_pat = raw.drop_duplicates("patient_id").copy()

# ACR: labeled biopsies
lab = raw[raw["label"].notna()].copy()
lab["anchor_dt"] = pd.to_datetime(lab["anchor_dt"])
first_biopsy_dt = lab.groupby("patient_id")["anchor_dt"].min()
acr_first_dt    = lab[lab["label"]==1].groupby("patient_id")["anchor_dt"].min()
n_acr_pos_per_pat = lab[lab["label"]==1].groupby("patient_id").size()

all_pat["first_biopsy_dt"] = all_pat["patient_id"].map(first_biopsy_dt)
all_pat["first_acr_dt"]    = all_pat["patient_id"].map(acr_first_dt)
all_pat["n_acr_pos"]       = all_pat["patient_id"].map(n_acr_pos_per_pat).fillna(0).astype(int)
all_pat["has_acr"]         = all_pat["n_acr_pos"] > 0
all_pat["days_to_acr"]     = (
    pd.to_datetime(all_pat["first_acr_dt"]) -
    pd.to_datetime(all_pat["first_biopsy_dt"])
).dt.days


def load_reps(task, n_splits=5):
    records = []
    for s in range(n_splits):
        p = REPS_DIR / f"biopsy_reps_{task}_split{s}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        pids  = list(d["patient_ids"])
        stems = list(d["stems"]) if "stems" in d else [""]*len(pids)
        tte   = d["tte"].numpy()   if "tte"   in d else np.full(len(pids), np.nan)
        event = d["event"].numpy() if "event" in d else np.full(len(pids), np.nan)
        label = d["label"].numpy() if "label" in d else np.full(len(pids), np.nan)
        for i in range(len(pids)):
            records.append({
                "patient_id": pids[i],
                "stem": stems[i],
                "tte":  float(tte[i]),
                "event":float(event[i]),
                "label":float(label[i]),
                "combo":stem_to_combo.get(stems[i], "Unknown"),
            })
    return pd.DataFrame(records)


def annotate_bar(ax, bars, vals, total, fontsize=FS_ANNOT):
    for bar, v in zip(bars, vals):
        pct = 100*v/total if total else 0
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+total*0.01,
                f"n={v}\n({pct:.0f}%)", ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold")


for task, cfg in TASK_CFG.items():
    df = load_reps(task)
    if df.empty:
        print(f"[skip] {task}"); continue

    col   = cfg["color"]
    bw    = cfg["bin_w"]
    n_bio = len(df)
    n_pat = df["patient_id"].nunique()

    # filter biopsies with TTE < 0 (post-event biopsies — invalid)
    df_valid = df[df["tte"] >= 0].copy() if "tte" in df.columns else df.copy()
    df_valid = df_valid.dropna(subset=["tte"])
    n_filtered = n_bio - len(df_valid)

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), facecolor=BG)
    fig.patch.set_facecolor(BG)
    ax_ev, ax_tx, ax_tte, ax_mod = axes[0,0], axes[0,1], axes[1,0], axes[1,1]
    for ax in axes.flat:
        ax.set_facecolor(BG)

    # ── Top-left: patient event count distribution ───────────────────────────
    if cfg["multi_acr"]:
        n_none   = int((all_pat["n_acr_pos"] == 0).sum())
        n_single = int((all_pat["n_acr_pos"] == 1).sum())
        n_multi  = int((all_pat["n_acr_pos"] >  1).sum())
        cats   = ["No ACR\n(0)", "Single ACR\n(1)", "Multiple ACR\n(≥2)"]
        vals   = [n_none, n_single, n_multi]
        bcols  = [C_CENS, "#FFA726", col]
        total  = len(all_pat)
        bars = ax_ev.bar(cats, vals, color=bcols, alpha=0.88,
                         edgecolor="#333", linewidth=0.8, width=0.55, zorder=2)
        annotate_bar(ax_ev, bars, vals, total)
        ax_ev.set_ylabel("Patients", fontsize=FS_LABEL)
        ax_ev.set_ylim(0, max(vals)*1.22)
        ax_ev.set_title(f"ACR events per patient\n(n={total} patients)", fontsize=FS_TITLE, fontweight="bold")
    else:
        ev_col  = cfg["ev_col"]
        n_event = int(all_pat[ev_col].sum()) if ev_col and ev_col in all_pat else 0
        n_cens  = len(all_pat) - n_event
        cats    = [f"No event\n(censored)", f"Event"]
        vals    = [n_cens, n_event]
        bcols   = [C_CENS, col]
        total   = len(all_pat)
        bars = ax_ev.bar(cats, vals, color=bcols, alpha=0.88,
                         edgecolor="#333", linewidth=0.8, width=0.55, zorder=2)
        annotate_bar(ax_ev, bars, vals, total)
        ax_ev.set_ylabel("Patients", fontsize=FS_LABEL)
        ax_ev.set_ylim(0, max(vals)*1.22)
        ax_ev.set_title(f"Events per patient\n(n={total} patients)", fontsize=FS_TITLE, fontweight="bold")
    ax_ev.tick_params(labelsize=FS_TICK)
    ax_ev.spines[["top","right"]].set_visible(False)

    # ── Top-right: time from transplant/first biopsy to first event ──────────
    if cfg["multi_acr"]:
        tx_times = all_pat[all_pat["has_acr"]]["days_to_acr"].dropna()
        xlabel_tx = cfg["xlabel_tx"]
        n_ev_pat  = len(tx_times)
        if len(tx_times):
            bins_tx = np.arange(0, tx_times.max()+bw, bw)
            ax_tx.hist(tx_times, bins=bins_tx, color=col, alpha=0.85,
                       edgecolor="#333", linewidth=0.6, zorder=2)
            med = int(tx_times.median())
            ax_tx.axvline(med, color="#7B0000", lw=2, ls="--",
                          label=f"Median: {med} days")
            ax_tx.legend(fontsize=FS_TICK, framealpha=0.9)
            ax_tx.set_title(
                f"{xlabel_tx}\n(n={n_ev_pat} ACR+ patients, "
                f"mean={int(tx_times.mean())}d, median={med}d)",
                fontsize=FS_TITLE, fontweight="bold")
    else:
        day_col = cfg["day_col"]
        ev_col  = cfg["ev_col"]
        if day_col and ev_col and day_col in all_pat:
            tx_times = all_pat[all_pat[ev_col]==1][day_col].dropna()
            n_ev_pat = len(tx_times)
            if len(tx_times):
                bins_tx = np.arange(0, tx_times.max()+bw, bw)
                ax_tx.hist(tx_times, bins=bins_tx, color=col, alpha=0.85,
                           edgecolor="#333", linewidth=0.6, zorder=2)
                med = int(tx_times.median())
                ax_tx.axvline(med, color="#333", lw=2, ls="--",
                              label=f"Median: {med} days")
                ax_tx.legend(fontsize=FS_TICK, framealpha=0.9)
                ax_tx.set_title(
                    f"{cfg['xlabel_tx']}\n(n={n_ev_pat} patients with event, "
                    f"median={med}d)",
                    fontsize=FS_TITLE, fontweight="bold")
    ax_tx.set_xlabel("Days", fontsize=FS_LABEL)
    ax_tx.set_ylabel("Patients", fontsize=FS_LABEL)
    ax_tx.tick_params(labelsize=FS_TICK)
    ax_tx.spines[["top","right"]].set_visible(False)

    # ── Bottom-left: biopsy TTE histogram (TTE >= 0 only) ───────────────────
    if task == "acr_cls":
        n_pos = int((df["label"]==1).sum())
        n_neg = int((df["label"]==0).sum())
        cats  = ["ACR+\n(label=1)", "ACR−\n(label=0)"]
        vals  = [n_pos, n_neg]
        bars = ax_tte.bar(cats, vals, color=[col, C_CENS], alpha=0.88,
                          edgecolor="#333", linewidth=0.8, width=0.5, zorder=2)
        annotate_bar(ax_tte, bars, vals, n_bio)
        ax_tte.set_ylabel("Biopsies", fontsize=FS_LABEL)
        ax_tte.set_ylim(0, max(vals)*1.22)
        ax_tte.set_title(
            f"ACR label distribution\n{n_bio} biopsies, {n_pat} patients",
            fontsize=FS_TITLE, fontweight="bold")
    else:
        ev   = df_valid[df_valid["event"]==1]["tte"]
        cens = df_valid[df_valid["event"]==0]["tte"]
        max_tte = max(ev.max() if len(ev) else 0, cens.max() if len(cens) else 0)
        bins = np.arange(0, max_tte+bw, bw)
        ax_tte.hist(cens, bins=bins, color=C_CENS, alpha=0.60,
                    label=f"Censored (n={len(cens)})",
                    edgecolor="#888", linewidth=0.5, zorder=2)
        ax_tte.hist(ev, bins=bins, color=col, alpha=0.85,
                    label=f"Event (n={len(ev)})",
                    edgecolor="#333", linewidth=0.6, zorder=3)
        if len(ev):
            med = int(ev.median())
            ax_tte.axvline(med, color="#333", lw=2, ls="--",
                           label=f"Median event TTE: {med}d")
        pct = 100*len(ev)/len(df_valid) if len(df_valid) else 0
        note = f"  ({n_filtered} post-event biopsies excluded)" if n_filtered else ""
        ax_tte.set_title(
            f"Biopsy TTE distribution (TTE ≥ 0)\n"
            f"{len(df_valid)} biopsies, {pct:.1f}% events{note}",
            fontsize=FS_TITLE, fontweight="bold")
        ax_tte.legend(fontsize=FS_TICK, framealpha=0.9)
    ax_tte.set_xlabel(cfg["xlabel_tte"] or "Days", fontsize=FS_LABEL)
    ax_tte.set_ylabel("Biopsies", fontsize=FS_LABEL)
    ax_tte.tick_params(labelsize=FS_TICK)
    ax_tte.spines[["top","right"]].set_visible(False)

    # ── Bottom-right: modality combo counts ──────────────────────────────────
    combo_counts = df["combo"].value_counts().sort_values(ascending=True)
    colors_c = [COMBO_COLORS[i % len(COMBO_COLORS)] for i in range(len(combo_counts))]
    ys = np.arange(len(combo_counts))
    ax_mod.barh(ys, combo_counts.values, color=colors_c, alpha=0.88,
                edgecolor="#333", linewidth=0.6, zorder=2)
    for y, v in zip(ys, combo_counts.values):
        pct = 100*v/n_bio
        ax_mod.text(v + n_bio*0.005, y, f"n={v} ({pct:.1f}%)",
                    va="center", fontsize=FS_ANNOT)
    ax_mod.set_yticks(ys)
    ax_mod.set_yticklabels(combo_counts.index.tolist(), fontsize=FS_TICK)
    ax_mod.set_xlabel("Biopsies", fontsize=FS_LABEL)
    ax_mod.set_title(
        f"Modality combinations\n{n_bio} biopsies · {n_pat} patients",
        fontsize=FS_TITLE, fontweight="bold")
    ax_mod.tick_params(labelsize=FS_TICK)
    ax_mod.spines[["top","right"]].set_visible(False)

    fig.suptitle(cfg["label"], fontsize=18, fontweight="bold", y=1.01)
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(OUT_DIR/f"tte_{task}.{ext}", dpi=200,
                    bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved tte_{task}")

print("All done.")
