"""
For each test biopsy, group by the modalities present at that visit.
Show LongMIL performance (C-index / BACC) per modality combo.
Also overlay ABMIL single-modality reference values.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT     = Path(__file__).resolve().parent.parent
REPS_DIR = ROOT / "results" / "mm_abmil_v8" / "biopsy_reps"
PRED     = ROOT / "results" / "predictions"
MASTER   = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
OUT_DIR  = ROOT / "figures" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG       = "#FAF6F2"
MODS      = ["has_HE", "has_BAL", "has_CT", "has_Clinical"]
MOD_NAMES = ["HE", "BAL", "CT", "Clinical"]
MIN_N     = 20
SPLIT_COLS = ["s0", "s1", "s2", "s3", "s4"]

TASKS = {
    "acr_cls":   {"metric": "bacc",   "mlabel": "BACC",    "comp_file": "comparison_acr_cls.csv",  "title": "ACR Classification"},
    "acr_surv":  {"metric": "cindex", "mlabel": "C-index", "comp_file": "comparison_acr_surv.csv", "title": "ACR Survival"},
    "clad_surv": {"metric": "cindex", "mlabel": "C-index", "comp_file": "comparison_clad.csv",     "title": "CLAD Survival"},
    "death_surv":{"metric": "cindex", "mlabel": "C-index", "comp_file": "comparison_death.csv",    "title": "Death Survival"},
}

# Color by number of modalities in combo
N_COLORS = {0: "#B0BEC5", 1: "#42A5F5", 2: "#1565C0", 3: "#0D3B66", 4: "#01050F"}
MOD_COLORS = {"HE": "#E64A19", "BAL": "#1565C0", "CT": "#2E7D32", "Clinical": "#7B1FA2"}
ABMIL_COLOR = "#FF6F00"   # amber for ABMIL reference dots

P1_MAP = {"P1 HE": "HE", "P1 BAL": "BAL", "P1 CT": "CT", "P1 Clinical": "Clinical"}

FS_TITLE  = 18
FS_LABEL  = 13
FS_TICK   = 12
FS_ANNOT  = 10


def load_master():
    df = pd.read_csv(MASTER)
    df["stem"]  = df["file"].str.replace(".pt", "", regex=False)
    df["combo"] = df[MODS].apply(
        lambda r: "+".join(n for n, m in zip(MOD_NAMES, MODS) if r[m]) or "None", axis=1)
    return df.set_index("stem")["combo"].to_dict()


def load_reps(task, stem_to_combo):
    rows = []
    for s in range(5):
        p = REPS_DIR / f"biopsy_reps_{task}_split{s}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        stems = list(d["stems"])
        risk  = d["risk"].numpy()
        tte   = d["tte"].numpy()   if "tte"   in d else np.full(len(stems), np.nan)
        event = d["event"].numpy() if "event" in d else np.full(len(stems), np.nan)
        label = d["label"].numpy() if "label" in d else np.full(len(stems), np.nan)
        for i, stem in enumerate(stems):
            rows.append({"stem": stem, "risk": float(risk[i]),
                         "tte": float(tte[i]), "event": float(event[i]),
                         "label": float(label[i]),
                         "combo": stem_to_combo.get(stem, "Unknown")})
    return pd.DataFrame(rows)


def concordance(risk, tte, event):
    from sksurv.metrics import concordance_index_censored
    ev = event.astype(bool)
    if ev.sum() < 2:
        return np.nan
    c, *_ = concordance_index_censored(ev, tte, risk)
    return c


def balanced_acc(risk, label, threshold):
    pred = (risk >= threshold).astype(int)
    lbl  = label.astype(int)
    recalls = [(pred[lbl == c] == c).mean() for c in np.unique(lbl) if (lbl == c).sum() > 0]
    return float(np.mean(recalls))


def load_abmil(comp_file):
    """Mean ABMIL metric per modality from comparison CSV (mean of s0-s4)."""
    path = PRED / comp_file
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        model = str(row["model"]).strip()
        if model in P1_MAP:
            vals = [float(row.get(s, np.nan)) for s in SPLIT_COLS]
            valid = [v for v in vals if not np.isnan(v)]
            if valid:
                out[P1_MAP[model]] = float(np.mean(valid))
    return out


def combo_sort_key(combo):
    if combo == "All biopsies":
        return (-1, combo)
    parts = combo.split("+")
    order = ["HE", "BAL", "CT", "Clinical"]
    idx = sum(order.index(p) if p in order else 99 for p in parts)
    return (len(parts), idx)


def compute_groups(df, metric):
    threshold = np.nanmedian(df["risk"].values) if metric == "bacc" else None
    counts = df["combo"].value_counts()
    shown  = counts[counts >= MIN_N].index.tolist()
    combos = sorted(shown, key=combo_sort_key)
    groups = [("All biopsies", df)] + [(c, df[df["combo"] == c]) for c in combos]

    results = []
    for combo, grp in groups:
        if metric == "cindex":
            mask = ~(np.isnan(grp["risk"]) | np.isnan(grp["tte"]) | np.isnan(grp["event"]))
            r, t, e = grp["risk"].values[mask], grp["tte"].values[mask], grp["event"].values[mask]
            if len(r) < 3 or e.astype(bool).sum() < 2:
                continue
            val = concordance(r, t, e)
        else:
            mask = ~(np.isnan(grp["risk"]) | np.isnan(grp["label"]))
            r, l = grp["risk"].values[mask], grp["label"].values[mask]
            if len(r) < 5 or len(np.unique(l)) < 2:
                continue
            val = balanced_acc(r, l, threshold)
        results.append({"combo": combo, "val": val, "n": int(mask.sum())})
    return results


def bar_color(combo):
    if combo == "All biopsies":
        return N_COLORS[0]
    parts = combo.split("+")
    if len(parts) == 1:
        return MOD_COLORS.get(parts[0], "#888")
    return N_COLORS[min(len(parts), 4)]


def plot_task(ax, task_key, cfg, stem_to_combo, abmil_vals):
    df = load_reps(task_key, stem_to_combo)
    if df.empty:
        ax.set_title(cfg["title"])
        return

    groups = compute_groups(df, cfg["metric"])
    if not groups:
        return

    combos = [g["combo"] for g in groups]
    vals   = [g["val"]   for g in groups]
    ns     = [g["n"]     for g in groups]
    ys     = np.arange(len(combos))
    colors = [bar_color(c) for c in combos]

    ax.set_facecolor(BG)
    ax.axvline(0.5, color="#999", linewidth=1.0, linestyle=":", alpha=0.6, zorder=1)

    # separator after "All biopsies"
    ax.axhline(0.5, color="#bbb", linewidth=1.0, linestyle="--", alpha=0.7)

    for i, (combo, val, n, col) in enumerate(zip(combos, vals, ns, colors)):
        ax.barh(ys[i], val, height=0.6, color=col, alpha=0.88, zorder=2)
        # n annotation inside bar
        ax.text(0.02, ys[i], f"n={n}", va="center", ha="left",
                fontsize=FS_ANNOT - 1, color="white", fontweight="bold", zorder=5)
        # metric value at bar end
        ax.text(val + 0.008, ys[i], f"{val:.3f}", va="center", ha="left",
                fontsize=FS_ANNOT, color="#111", zorder=5)

    # ABMIL reference dots for single-modality combos
    for i, combo in enumerate(combos):
        parts = combo.split("+")
        if len(parts) == 1 and parts[0] in abmil_vals:
            ax.scatter(abmil_vals[parts[0]], ys[i], s=60, color=ABMIL_COLOR,
                       edgecolors="#333", linewidths=0.8, zorder=6,
                       marker="D", label="ABMIL" if i == 1 else "_nolegend_")

    ax.set_yticks(ys)
    ax.set_yticklabels(combos, fontsize=FS_TICK)
    ax.set_xlabel(cfg["mlabel"], fontsize=FS_LABEL)
    ax.set_title(cfg["title"], fontsize=FS_TITLE, fontweight="bold")
    ax.set_xlim(0.0, 1.05)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", labelsize=FS_TICK)


# ── Main ─────────────────────────────────────────────────────────────────────
stem_to_combo = load_master()

legend_handles = [
    Patch(facecolor=N_COLORS[0],   label="All biopsies (reference)"),
    Patch(facecolor=MOD_COLORS["HE"],       label="HE only"),
    Patch(facecolor=MOD_COLORS["BAL"],      label="BAL only"),
    Patch(facecolor=MOD_COLORS["CT"],       label="CT only"),
    Patch(facecolor=MOD_COLORS["Clinical"], label="Clinical only"),
    Patch(facecolor=N_COLORS[2],   label="2-modality combo"),
    Patch(facecolor=N_COLORS[3],   label="3-modality combo"),
    Patch(facecolor=N_COLORS[4],   label="All 4 modalities"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=ABMIL_COLOR,
           markeredgecolor="#333", markersize=8, label="ABMIL (single-mod reference)"),
    Line2D([0], [0], color="#999", linewidth=1, linestyle=":", label="Chance (0.5)"),
]

# Per-task figures
for task_key, cfg in TASKS.items():
    abmil_vals = load_abmil(cfg["comp_file"])
    df_tmp = load_reps(task_key, stem_to_combo)
    n_groups = len(compute_groups(df_tmp, cfg["metric"])) if not df_tmp.empty else 6
    fig, ax = plt.subplots(figsize=(10, max(5, n_groups * 0.6 + 1.5)), facecolor=BG)
    fig.patch.set_facecolor(BG)
    plot_task(ax, task_key, cfg, stem_to_combo, abmil_vals)
    fig.legend(handles=legend_handles, fontsize=9, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.01),
               framealpha=0.95, edgecolor="#ccc")
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"modality_combo_input_{task_key}.{ext}",
                    dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved modality_combo_input_{task_key}")

# 2×2 combined
fig, axes = plt.subplots(2, 2, figsize=(22, 16), facecolor=BG)
fig.patch.set_facecolor(BG)
fig.suptitle("LongMIL performance by input modality combination per test biopsy",
             fontsize=16, fontweight="bold")
for ax, (task_key, cfg) in zip(axes.flat, TASKS.items()):
    abmil_vals = load_abmil(cfg["comp_file"])
    plot_task(ax, task_key, cfg, stem_to_combo, abmil_vals)
fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, 0.0), frameon=False)
fig.tight_layout(rect=[0, 0.06, 1, 0.98], h_pad=3, w_pad=3)
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"modality_combo_input_all.{ext}",
                dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved modality_combo_input_all")
print("Done.")
