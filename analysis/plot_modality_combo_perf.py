"""
Per-modality-combination performance for LongMIL.

For each biopsy in the test set we know which modalities were present
(from the master table via stem key).  We group patients by the richest
modality combo they ever had across their test biopsies, then compute:
  - Survival tasks  : concordance index (sksurv) per combo
  - ACR cls         : BACC per combo

Only combos with ≥10 patients are shown individually; the rest are
pooled as "Other".

Output: figures/benchmark/modality_combo_{task}.png  (one per task)
        figures/benchmark/modality_combo_all.png      (4-panel)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT     = Path(__file__).resolve().parent.parent
REPS_DIR = ROOT / "results" / "mm_abmil_v8" / "biopsy_reps"
MASTER   = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
OUT_DIR  = ROOT / "figures" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG        = "#FAF6F2"
FS_TITLE  = 26
FS_PANEL  = 20
FS_LABEL  = 17
FS_TICK   = 15
FS_ANNOT  = 13
MIN_N_PAT = 10   # minimum patients for survival tasks
MIN_N_BIO = 20   # minimum biopsies for classification tasks

MODS = ["has_HE", "has_BAL", "has_CT", "has_Clinical"]
MOD_NAMES = ["HE", "BAL", "CT", "Clinical"]

TASKS = {
    "death_surv": {"label": "Post-Transplant Mortality", "metric": "cindex",  "metric_label": "C-index"},
    "acr_surv":   {"label": "ACR Survival",              "metric": "cindex",  "metric_label": "C-index"},
    "clad_surv":  {"label": "CLAD Survival",             "metric": "cindex",  "metric_label": "C-index"},
    "acr_cls":    {"label": "ACR Classification",        "metric": "bacc",    "metric_label": "Balanced Accuracy"},
}

# Consistent colours per combo (assigned on first encounter, then reused)
COMBO_PALETTE = [
    "#1565C0", "#2E7D32", "#B71C1C", "#6A1B9A",
    "#E65100", "#00695C", "#4E342E", "#37474F",
    "#F57F17", "#880E4F",
]
_combo_col: dict = {}

def combo_color(combo: str) -> str:
    if combo not in _combo_col:
        _combo_col[combo] = COMBO_PALETTE[len(_combo_col) % len(COMBO_PALETTE)]
    return _combo_col[combo]


def concordance_index(risk, tte, event):
    """Simple concordance index (higher risk → earlier event = concordant)."""
    try:
        from sksurv.metrics import concordance_index_censored
        ev_bool = event.astype(bool)
        if ev_bool.sum() < 2:
            return np.nan
        c, *_ = concordance_index_censored(ev_bool, tte, risk)
        return c
    except ImportError:
        # manual fallback
        n = len(risk)
        concordant = discordant = 0
        for i in range(n):
            for j in range(n):
                if event[i] == 1 and tte[i] < tte[j]:
                    if risk[i] > risk[j]:
                        concordant += 1
                    elif risk[i] < risk[j]:
                        discordant += 1
        denom = concordant + discordant
        return concordant / denom if denom > 0 else np.nan


def balanced_accuracy(pred_label, true_label):
    classes = np.unique(true_label)
    recalls = []
    for c in classes:
        mask = true_label == c
        if mask.sum() == 0:
            continue
        recalls.append((pred_label[mask] == c).mean())
    return float(np.mean(recalls)) if recalls else np.nan


def load_master():
    df = pd.read_csv(MASTER)
    df["stem"] = df["file"].str.replace(".pt", "", regex=False)
    df["combo"] = df[MODS].apply(
        lambda r: "+".join(n for n, m in zip(MOD_NAMES, MODS) if r[m]) or "None",
        axis=1
    )
    return df[["patient_id", "stem", "combo"] + MODS]


def load_reps(task, n_splits=5):
    """Load biopsy-level reps; attach per-biopsy combo from master table."""
    master = load_master()
    stem_to_combo = master.set_index("stem")["combo"].to_dict()
    records = []
    for s in range(n_splits):
        p = REPS_DIR / f"biopsy_reps_{task}_split{s}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        pids  = list(d["patient_ids"])
        stems = list(d["stems"])
        risk  = d["risk"].numpy()
        tte   = d["tte"].numpy()   if "tte"   in d else np.full(len(pids), np.nan)
        event = d["event"].numpy() if "event" in d else np.full(len(pids), np.nan)
        label = d["label"].numpy() if "label" in d else np.full(len(pids), np.nan)
        for i in range(len(pids)):
            records.append({
                "patient_id": pids[i],
                "stem":       stems[i],
                "split":      s,
                "risk":       float(risk[i]),
                "tte":        float(tte[i]),
                "event":      float(event[i]),
                "label":      float(label[i]),
                "combo":      stem_to_combo.get(stems[i], "Unknown"),
            })
    return pd.DataFrame(records)


def compute_metrics_survival(df_bio):
    """C-index per combo, biopsy-level. Returns DataFrame."""
    results = []
    combo_counts = df_bio["combo"].value_counts()
    shown = combo_counts[combo_counts >= MIN_N_BIO].index.tolist()
    other = df_bio[~df_bio["combo"].isin(shown)]

    groups = [("All biopsies", df_bio)] + \
             [(c, df_bio[df_bio["combo"] == c]) for c in shown]
    if len(other) >= MIN_N_BIO:
        groups.append(("Other", other))

    for combo, grp in groups:
        risk  = grp["risk"].values
        tte   = grp["tte"].values
        event = grp["event"].values
        mask  = ~(np.isnan(risk) | np.isnan(tte) | np.isnan(event))
        risk, tte, event = risk[mask], tte[mask], event[mask]
        if len(risk) < 3 or event.astype(bool).sum() < 2:
            continue
        val = concordance_index(risk, tte, event)
        boots, rng = [], np.random.default_rng(42)
        for _ in range(200):
            idx = rng.integers(0, len(risk), len(risk))
            v = concordance_index(risk[idx], tte[idx], event[idx])
            if not np.isnan(v):
                boots.append(v)
        results.append({"combo": combo, "n": int(mask.sum()),
                         "val": float(val) if not np.isnan(val) else np.nan,
                         "std": float(np.std(boots)) if boots else 0.0})
    return pd.DataFrame(results)


def compute_metrics_bacc(df_bio):
    """BACC per combo, biopsy-level. df_bio must have risk, label, combo columns."""
    global_median = np.nanmedian(df_bio["risk"].values)
    results = []
    combo_counts = df_bio["combo"].value_counts()
    shown = combo_counts[combo_counts >= MIN_N_BIO].index.tolist()
    other = df_bio[~df_bio["combo"].isin(shown)]

    groups = [("All biopsies", df_bio)] + \
             [(c, df_bio[df_bio["combo"] == c]) for c in shown]
    if len(other) >= MIN_N_BIO:
        groups.append(("Other", other))

    for combo, grp in groups:
        mask = ~(np.isnan(grp["risk"].values) | np.isnan(grp["label"].values))
        risk = grp["risk"].values[mask]
        lbl  = grp["label"].values[mask].astype(int)
        if len(risk) < 5 or len(np.unique(lbl)) < 2:
            continue
        pred = (risk >= global_median).astype(int)
        val  = balanced_accuracy(pred, lbl)
        boots, rng = [], np.random.default_rng(42)
        for _ in range(200):
            idx = rng.integers(0, len(risk), len(risk))
            boots.append(balanced_accuracy(pred[idx], lbl[idx]))
        results.append({"combo": combo, "n": int(mask.sum()),
                         "val": float(val), "std": float(np.std(boots))})
    return pd.DataFrame(results)


NMOD_COLORS = {
    0: "#607D8B",   # Other / unknown
    1: "#E65100",   # 1 modality
    2: "#1565C0",   # 2 modalities
    3: "#2E7D32",   # 3 modalities
    4: "#6A1B9A",   # 4 modalities
}

def _n_mods(combo):
    if combo in ("All biopsies", "Other", "Unknown", "None"):
        return -1
    return len(combo.split("+"))


def plot_task(ax, task, cfg, df_all):
    metric  = cfg["metric"]
    df_task = df_all[df_all["task"] == task].copy()
    if df_task.empty:
        ax.set_visible(False)
        return

    if metric == "bacc":
        df_metr = compute_metrics_bacc(df_task)
    else:
        df_metr = compute_metrics_survival(df_task)

    all_label = "All biopsies"

    if df_metr.empty:
        ax.set_visible(False)
        return

    df_metr = df_metr.dropna(subset=["val"])
    # sort: Other first, then by n_mods asc (1→4), All biopsies last
    other_row = df_metr[df_metr["combo"] == "Other"]
    all_row   = df_metr[df_metr["combo"] == all_label]
    mid       = df_metr[~df_metr["combo"].isin([all_label, "Other"])].copy()
    mid["_nm"] = mid["combo"].map(_n_mods)
    mid = mid.sort_values("_nm", ascending=True).drop(columns="_nm")
    df_metr = pd.concat([other_row, mid, all_row], ignore_index=True)

    ys     = np.arange(len(df_metr))
    combos = df_metr["combo"].tolist()
    vals   = df_metr["val"].values
    stds   = df_metr["std"].values
    ns     = df_metr["n"].values

    colors = []
    for c in combos:
        if c == all_label:
            colors.append("#B71C1C")
        elif c == "Other":
            colors.append("#607D8B")
        else:
            colors.append(NMOD_COLORS.get(_n_mods(c), "#607D8B"))

    ax.set_facecolor(BG)

    if all_label in combos:
        all_y = ys[combos.index(all_label)]
        ax.axhspan(all_y - 0.45, all_y + 0.45, facecolor="#FCE4EC", alpha=0.45, zorder=0)

    for i, (y, val, std, col, n) in enumerate(zip(ys, vals, stds, colors, ns)):
        ax.barh(y, val, height=0.60, color=col, alpha=0.88, zorder=2)
        ax.errorbar(val, y, xerr=std, fmt="none", color="#333",
                    capsize=4, capthick=1.2, lw=1.2, zorder=3)
        ax.text(val + std + 0.005, y, f"n={n}", va="center",
                fontsize=FS_ANNOT, color="#444")

    ax.set_yticks(ys)
    ax.set_yticklabels(combos, fontsize=FS_TICK)
    ax.set_xlabel(cfg["metric_label"], fontsize=FS_LABEL)
    title = cfg["label"]
    if task == "clad_surv":
        title += "\n(low event rate — near chance)"
    ax.set_title(title, fontsize=FS_PANEL, fontweight="bold", pad=8)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_xlim(0, 1.10)
    ax.axvline(0.5, color="#999", lw=0.8, ls="--", zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    # legend for modality count colors
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#B71C1C", label="All biopsies"),
        Patch(facecolor=NMOD_COLORS[1], label="1 modality"),
        Patch(facecolor=NMOD_COLORS[2], label="2 modalities"),
        Patch(facecolor=NMOD_COLORS[3], label="3 modalities"),
        Patch(facecolor=NMOD_COLORS[4], label="4 modalities"),
        Patch(facecolor="#607D8B", label="Other/pooled"),
    ]
    ax.legend(handles=legend_els, fontsize=max(FS_ANNOT-1, 9),
              loc="lower right", framealpha=0.88, edgecolor="#ccc")


# ── Load all tasks ────────────────────────────────────────────────────────────
print("Loading reps...")
frames = []
for task in TASKS:
    df = load_reps(task)
    df["task"] = task
    frames.append(df)
    print(f"  {task}: {len(df)} biopsies, {df['patient_id'].nunique()} patients")

df_all = pd.concat(frames, ignore_index=True)
print("Done loading.\n")

# ── Per-task figures ──────────────────────────────────────────────────────────
for task, cfg in TASKS.items():
    fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG)
    plot_task(ax, task, cfg, df_all)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"modality_combo_{task}.png", dpi=180,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved modality_combo_{task}")

# ── 4-panel all-task figure ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(56, 10), facecolor=BG)
fig.suptitle(
    "LongMIL Performance by Input Modality Combination\n"
    "(each biopsy grouped by its own modality combo at time of biopsy)",
    fontsize=FS_TITLE, fontweight="bold", y=1.02
)
for ax, (task, cfg) in zip(axes, TASKS.items()):
    plot_task(ax, task, cfg, df_all)
fig.tight_layout()
fig.savefig(OUT_DIR / "modality_combo_all.png", dpi=180,
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved modality_combo_all")
print("\nAll done.")
