"""
Aggregate Clinical feature importance across splits and plot top-20 features
per task as horizontal bar charts.

Reads:  interpretability/longitudinal_mk_interp/
            longitudinal_mk_no_alibi_split{s}_fold0_{task}/
                clinical_feature_imp_data_{task}.json
Saves:  figures/interpretability/{task}/clinical_feature_imp_{task}.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
INTERP_ROOT = REPO / "interpretability" / "longitudinal_mk_interp"
FIGS_ROOT   = REPO / "figures" / "interpretability"
N_SPLITS    = 5
VARIANT     = "longitudinal_mk_no_alibi"
TOP_N       = 20
TASKS       = ["acr_cls", "acr_surv", "clad_surv", "death_surv"]
TASK_LABELS = {
    "acr_cls":   "ACR (classification)",
    "acr_surv":  "ACR (survival)",
    "clad_surv": "CLAD (survival)",
    "death_surv":"Death (survival)",
}

for task in TASKS:
    split_deltas = []
    feature_names = None
    n_hi_list, n_lo_list = [], []

    for s in range(N_SPLITS):
        base = INTERP_ROOT / f"{VARIANT}_split{s}_fold0_{task}"
        jpath = base / f"clinical_feature_imp_data_{task}.json"
        if not jpath.exists():
            task_short = task.replace("_surv", "")
            jpath = base / f"clinical_feature_imp_data_{task_short}.json"
        if not jpath.exists():
            print(f"  [skip] clinical_feature_imp_data for {task} not found for split {s}")
            continue
        d = json.loads(jpath.read_text())
        split_deltas.append(np.array(d["delta"], dtype=np.float64))
        if feature_names is None:
            feature_names = d["names"]
        n_hi_list.append(d["n_hi"])
        n_lo_list.append(d["n_lo"])

    if len(split_deltas) < 2:
        print(f"[{task}] only {len(split_deltas)} splits found, skipping")
        continue

    # Align lengths (rare mismatch guard)
    min_len = min(a.shape[0] for a in split_deltas)
    arr = np.stack([a[:min_len] for a in split_deltas])   # (n_splits, n_features)
    delta_mean = arr.mean(0)
    delta_std  = arr.std(0)
    names = (feature_names or [str(i) for i in range(min_len)])[:min_len]

    # Pick top-20 by |delta_mean|
    order  = np.argsort(np.abs(delta_mean))[::-1][:TOP_N]
    # Sort top-20 by delta_mean value (positive → top, negative → bottom)
    order  = order[np.argsort(delta_mean[order])[::-1]]

    d_vals = delta_mean[order]
    d_errs = delta_std[order]
    labels = [names[i] for i in order]

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in d_vals]

    fig, ax = plt.subplots(figsize=(7, 0.42 * TOP_N + 1.6))
    y = np.arange(len(labels))
    ax.barh(y, d_vals, xerr=d_errs, color=colors, error_kw=dict(elinewidth=1, capsize=3),
            height=0.7, linewidth=0)
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Δ attention (hi-risk − lo-risk)", fontsize=9)
    ax.set_title(
        f"Clinical feature importance — {TASK_LABELS.get(task, task)}\n"
        f"mean±std across {len(split_deltas)} splits  "
        f"(n_hi≈{int(np.mean(n_hi_list))}, n_lo≈{int(np.mean(n_lo_list))})",
        fontsize=9,
    )
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()

    out_dir = FIGS_ROOT / task
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [("png", 200), ("pdf", 150)]:
        out_path = out_dir / f"clinical_feature_imp_{task}.{ext}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[{task}] saved figures/interpretability/{task}/clinical_feature_imp_{task}.png/pdf")
