"""
Per-modality-combo performance figures.

Computes actual BAcc (cls tasks) or C-index (tte tasks) per modality combination
from test-set predictions, broken down by variant.

Outputs in output_dir/combo/:
  fig_combo_{task_key}.pdf        — per-task per-variant combo bar charts
  fig_combo_summary_nmods.pdf     — score vs # modalities boxplot for all tasks
  combo_scores_{task_key}.csv     — raw per-combo scores
"""
import math
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import (
    NATURE_RC, TASKS, TASK_ENDPOINT, ENDPOINT,
    VARIANT_DISPLAY, VARIANT_TAGS, COMBO_COLORS, DEFAULT_COMBO_COLOR,
    TWO_COL_W,
)


def _save(fig, path_stem: Path):
    fig.savefig(str(path_stem) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(path_stem) + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Real metric computation ────────────────────────────────────────────────────

def _bacc(y_true, y_pred_prob, threshold=0.5):
    """Balanced accuracy from binary labels and predicted probabilities."""
    yt = np.asarray(y_true, dtype=float)
    yp = (np.asarray(y_pred_prob, dtype=float) >= threshold).astype(float)
    valid = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[valid], yp[valid]
    if len(yt) < 5 or len(np.unique(yt)) < 2:
        return float("nan"), len(yt)
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(yt.astype(int), yp.astype(int))), len(yt)


def _cindex(times, events, hazard):
    """Harrell's C-index from TTE, event flags, and hazard scores."""
    t = np.asarray(times,  dtype=float)
    e = np.asarray(events, dtype=float)
    h = np.asarray(hazard, dtype=float)
    valid = ~(np.isnan(t) | np.isnan(e) | np.isnan(h))
    t, e, h = t[valid], e[valid], h[valid]
    if len(t) < 5 or e.sum() < 2:
        return float("nan"), len(t)
    try:
        from lifelines.utils import concordance_index
        return float(concordance_index(t, -h, e)), len(t)
    except Exception:
        pass
    # fallback: manual Harrell's C
    concordant = discordant = 0
    for i in range(len(t)):
        if e[i] == 0:
            continue
        for j in range(len(t)):
            if i == j or t[j] <= t[i]:
                continue
            if h[i] > h[j]:
                concordant += 1
            elif h[i] < h[j]:
                discordant += 1
    denom = concordant + discordant
    return (float(concordant / denom) if denom > 0 else float("nan")), len(t)


def _combo_scores_cls(df: pd.DataFrame, combo_col: str = "combo") -> pd.DataFrame:
    """BAcc per modality combo for classification task."""
    rows = []
    for combo, grp in df.groupby(combo_col):
        score, n = _bacc(grp["acr_label"], grp["cls_prob"])
        rows.append({"combo": combo, "mean": score, "std": float("nan"), "n": n})
    out = pd.DataFrame(rows).set_index("combo").dropna(subset=["mean"])
    return out.sort_values("mean", ascending=False)


def _combo_scores_tte(df: pd.DataFrame, endpoint: str,
                       combo_col: str = "combo") -> pd.DataFrame:
    """C-index per modality combo for TTE task."""
    cfg = ENDPOINT[endpoint]
    tte_key = cfg["tte_key"]
    ev_key  = cfg["ev_key"]
    rows = []
    for combo, grp in df.groupby(combo_col):
        t = grp[tte_key].values if tte_key in grp.columns else np.full(len(grp), float("nan"))
        e = grp[ev_key].values  if ev_key  in grp.columns else np.full(len(grp), float("nan"))
        h = grp["hazard"].values
        score, n = _cindex(t, e, h)
        rows.append({"combo": combo, "mean": score, "std": float("nan"), "n": n})
    out = pd.DataFrame(rows).set_index("combo").dropna(subset=["mean"])
    return out.sort_values("mean", ascending=False)


# ── Bar chart helper ───────────────────────────────────────────────────────────

def _combo_bar(ax, grp: pd.DataFrame, metric_label: str, task_color: str, title: str):
    if grp.empty:
        ax.set_title(f"{title}\n(no data)", fontsize=7); return
    combos = list(grp.index)
    mus  = grp["mean"].values
    sds  = grp["std"].fillna(0).values
    ns   = grp["n"].values.astype(int)
    x    = np.arange(len(combos))
    colors = [COMBO_COLORS.get(c, DEFAULT_COMBO_COLOR) for c in combos]

    ax.bar(x, mus, yerr=np.where(np.isnan(sds), 0, sds),
           capsize=3, color=colors, alpha=0.85, width=0.7,
           edgecolor="white", lw=0.5, error_kw={"elinewidth": 1.0, "ecolor": "#555"})
    ax.axhline(0.5, color="#aaa", lw=0.8, ls=":", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(combos, ns)],
                       rotation=45, ha="right", fontsize=6)
    ax.set_ylabel(metric_label, fontsize=7)
    ax.set_title(title, fontsize=7, fontweight="bold", color=task_color)

    non_nan = mus[~np.isnan(mus)]
    if len(non_nan):
        ylo = max(0.35, float(np.nanmin(mus)) - 0.06)
        yhi = min(1.00, float(np.nanmax(mus)) + 0.10)
        ax.set_ylim(ylo, yhi)

    for xi, mu in enumerate(mus):
        if not np.isnan(mu):
            ax.text(xi, mu + 0.012, f"{mu:.3f}", ha="center", va="bottom", fontsize=5.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.5, alpha=0.4, ls="--")


# ── Main ──────────────────────────────────────────────────────────────────────

def task_combo(
    variant_data_map: Dict[str, Optional[Dict]],
    output_dir: Path,
):
    """
    variant_data_map: {task_key: variant_data_dict or None}
    variant_data_dict: {tag: {"df": pd.DataFrame, ...}}
    """
    plt.rcParams.update(NATURE_RC)
    out = output_dir / "combo"
    out.mkdir(parents=True, exist_ok=True)

    # ── Per-task figures ───────────────────────────────────────────────────────
    for task_key, (dir_suf, prim, label, color, task_type) in TASKS.items():
        vd = variant_data_map.get(task_key)
        if not vd:
            print(f"[combo] {task_key}: no data — skip"); continue

        endpoint = TASK_ENDPOINT[task_key]
        metric_label = "BAcc" if task_type == "cls" else "C-index"

        tags = [t for t in VARIANT_TAGS if t in vd] + \
               [t for t in sorted(vd) if t not in VARIANT_TAGS]

        fig, axes = plt.subplots(1, max(1, len(tags)),
                                 figsize=(max(TWO_COL_W, len(tags) * 2.8), 4.5),
                                 sharey=True)
        fig.patch.set_facecolor("white")
        if len(tags) == 1:
            axes = [axes]

        all_rows = []
        for ax, tag in zip(axes, tags):
            td   = vd[tag]
            test = td["df"][td["df"]["data_split"] == "test"].copy()
            if "combo" not in test.columns or len(test) == 0:
                ax.set_title(f"{VARIANT_DISPLAY.get(tag, tag)}\n(no data)", fontsize=7)
                continue

            if task_type == "cls":
                grp = _combo_scores_cls(test)
            else:
                grp = _combo_scores_tte(test, endpoint)

            # collect for CSV
            for combo, row in grp.iterrows():
                all_rows.append({"task": label, "variant": VARIANT_DISPLAY.get(tag, tag),
                                  "combo": combo, metric_label: row["mean"], "n": row["n"]})

            ttl = VARIANT_DISPLAY.get(tag, tag)
            _combo_bar(ax, grp, metric_label, color, ttl)

        fig.suptitle(f"{label} — {metric_label} by modality combo (test set)",
                     fontsize=8.5, fontweight="bold")
        plt.tight_layout()
        _save(fig, out / f"fig_combo_{task_key}")
        print(f"  Saved: {out}/fig_combo_{task_key}.pdf")

        if all_rows:
            pd.DataFrame(all_rows).to_csv(out / f"combo_scores_{task_key}.csv", index=False)

    # ── Summary: n_mods boxplot for each task ─────────────────────────────────
    n_tasks = len(TASKS)
    fig_s, axes_s = plt.subplots(1, n_tasks, figsize=(TWO_COL_W * 1.8, 4.5),
                                  sharey=False)
    fig_s.patch.set_facecolor("white")

    for ax, (task_key, (dir_suf, prim, label, color, task_type)) in zip(axes_s, TASKS.items()):
        vd = variant_data_map.get(task_key)
        if not vd:
            ax.set_title(label, fontsize=7, color=color); continue

        endpoint = TASK_ENDPOINT[task_key]
        metric_label = "BAcc" if task_type == "cls" else "C-index"

        # Pool test rows across all variants, compute score per (n_mods, sample)
        score_rows = []
        for tag, td in vd.items():
            test = td["df"][td["df"]["data_split"] == "test"].copy()
            if "n_mods" not in test.columns or len(test) == 0:
                continue
            for nm, grp in test.groupby("n_mods"):
                if task_type == "cls":
                    sc, n = _bacc(grp.get("acr_label", [float("nan")]*len(grp)),
                                  grp.get("cls_prob",  [float("nan")]*len(grp)))
                else:
                    cfg = ENDPOINT[endpoint]
                    t = grp[cfg["tte_key"]].values if cfg["tte_key"] in grp.columns else np.full(len(grp), float("nan"))
                    e = grp[cfg["ev_key"]].values  if cfg["ev_key"]  in grp.columns else np.full(len(grp), float("nan"))
                    h = grp["hazard"].values
                    sc, n = _cindex(t, e, h)
                if not math.isnan(sc):
                    score_rows.append({"n_mods": int(nm), "score": sc, "n": n})

        if not score_rows:
            ax.set_title(label, fontsize=7, color=color); continue

        df_sc  = pd.DataFrame(score_rows)
        groups = sorted(df_sc["n_mods"].unique())
        data_box = [df_sc[df_sc["n_mods"] == g]["score"].values for g in groups]

        bp = ax.boxplot(data_box, positions=range(len(groups)), widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops={"color": "#333", "lw": 1.5})
        for patch in bp["boxes"]:
            patch.set_facecolor(color); patch.set_alpha(0.75)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g}-mod" for g in groups], fontsize=6.5)
        ax.set_ylabel(metric_label, fontsize=7)
        ax.set_title(label, fontsize=7, fontweight="bold", color=color)
        ax.axhline(0.5, color="#aaa", lw=0.8, ls=":", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.5, alpha=0.4, ls="--")

    fig_s.suptitle(f"Performance vs # modalities (test set, per variant×fold)",
                   fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    _save(fig_s, out / "fig_combo_summary_nmods")
    print(f"  Saved: {out}/fig_combo_summary_nmods.pdf")
