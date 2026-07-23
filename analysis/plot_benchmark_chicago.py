#!/usr/bin/env python3
"""
Benchmark comparison: all P2 fusion methods + P1 single-modality baselines.

Reads fold0 test metrics from results/mm_abmil_v8/phase2/ and
results/mm_abmil_v8/phase1/ for all 5 splits, computes mean±std,
and produces a publication-quality comparison figure.

Usage (via SLURM -- never run directly):
  python analysis/plot_benchmark_chicago.py [--wandb] [--out-dir <path>]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT     = Path(__file__).resolve().parents[1]
RESULTS  = ROOT / "results" / "mm_abmil_v8"
PHASE2   = RESULTS / "phase2"
PHASE1   = RESULTS / "phase1"
N_SPLITS = 5

# ── Method definitions ────────────────────────────────────────────────────────
# Each entry: (display_name, reader_fn_or_tag, color)
# reader_fn receives split index and returns dict with keys:
#   bacc, acr_ci, clad_ci, death_ci  (float or nan)

def _read_mega(split, dir_tag, vtag):
    """Read metrics from a mega-task final JSON."""
    p = PHASE2 / f"split{split}_fold0" / dir_tag / f"metrics_{vtag}_final.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text()).get("test", {})
    return {
        "bacc":     d.get("bacc",            float("nan")),
        "acr_ci":   d.get("c_index",         float("nan")),
        "clad_ci":  d.get("clad_c_index",    float("nan")),
        "death_ci": d.get("death_c_index",   float("nan")),
    }

def _read_pertask(split, vtag):
    """Read per-task early/late/middle metrics and combine."""
    base = PHASE2 / f"split{split}_fold0"
    out = {}
    # ACR classification
    p = base / f"{vtag}_cls" / f"metrics_{vtag}_final.json"
    if p.exists():
        out["bacc"] = json.loads(p.read_text()).get("test", {}).get("bacc", float("nan"))
    else:
        out["bacc"] = float("nan")
    # ACR survival
    p = base / f"{vtag}_acr_surv" / f"metrics_{vtag}_final.json"
    if p.exists():
        out["acr_ci"] = json.loads(p.read_text()).get("test", {}).get("c_index", float("nan"))
    else:
        out["acr_ci"] = float("nan")
    # CLAD survival
    p = base / f"{vtag}_clad_surv" / f"metrics_{vtag}_final.json"
    if p.exists():
        out["clad_ci"] = json.loads(p.read_text()).get("test", {}).get("c_index", float("nan"))
    else:
        out["clad_ci"] = float("nan")
    # Death survival
    p = base / f"{vtag}_death_surv" / f"metrics_{vtag}_final.json"
    if p.exists():
        out["death_ci"] = json.loads(p.read_text()).get("test", {}).get("c_index", float("nan"))
    else:
        out["death_ci"] = float("nan")
    return out if any(not np.isnan(v) for v in out.values()) else None

def _read_p1(split, task_key, mod):
    """Read P1 single-modality test metrics."""
    # task_key: "acr" (for cls+surv), "acr_surv", "clad", "death"
    out = {"bacc": float("nan"), "acr_ci": float("nan"),
           "clad_ci": float("nan"), "death_ci": float("nan")}
    # Classification (acr task)
    for sub in ("final_combined", "final"):
        p = PHASE1 / f"split{split}_fold0" / "acr" / mod / sub / "metrics.json"
        if p.exists():
            d = json.loads(p.read_text()).get("test", {})
            out["bacc"] = d.get("bacc", float("nan"))
            break
    # ACR survival
    for sub in ("final_combined", "final"):
        p = PHASE1 / f"split{split}_fold0" / "acr_surv" / mod / sub / "metrics.json"
        if p.exists():
            d = json.loads(p.read_text()).get("test", {})
            out["acr_ci"] = d.get("c_index", float("nan"))
            break
    # CLAD survival
    for sub in ("final_combined", "final"):
        p = PHASE1 / f"split{split}_fold0" / "clad" / mod / sub / "metrics.json"
        if p.exists():
            d = json.loads(p.read_text()).get("test", {})
            out["clad_ci"] = d.get("c_index", float("nan"))
            break
    # Death survival
    for sub in ("final_combined", "final"):
        p = PHASE1 / f"split{split}_fold0" / "death" / mod / sub / "metrics.json"
        if p.exists():
            d = json.loads(p.read_text()).get("test", {})
            out["death_ci"] = d.get("c_index", float("nan"))
            break
    return out if any(not np.isnan(v) for v in out.values()) else None

def _read_unimodal_ablation(split, mod):
    """Read unimodal ablation from set_mil_mt_mega final metrics."""
    p = PHASE2 / f"split{split}_fold0" / "set_mil_mt_mega" / "metrics_set_mil_mt_final.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    abl = d.get("unimodal_ablation", {}).get(mod, {})
    if not abl:
        return None
    return {
        "bacc":     abl.get("bacc",            float("nan")),
        "acr_ci":   abl.get("acr_c_index",     float("nan")),
        "clad_ci":  abl.get("clad_c_index",    float("nan")),
        "death_ci": abl.get("death_c_index",   float("nan")),
    }

# Method registry: (label, reader_fn(split)->dict|None, color)
METHODS = [
    # Single-modality baselines
    ("P1 HE",          lambda s: _read_p1(s, "acr", "HE"),       "#e57373"),
    ("P1 BAL",         lambda s: _read_p1(s, "acr", "BAL"),      "#ef9a9a"),
    ("P1 CT",          lambda s: _read_p1(s, "acr", "CT"),       "#ffcdd2"),
    ("P1 Clinical",    lambda s: _read_p1(s, "acr", "Clinical"),  "#ffebee"),
    # Unimodal ablation (SetMIL multimodal model, one modality at inference)
    ("SetMIL[HE]",     lambda s: _read_unimodal_ablation(s, "HE"),       "#80cbc4"),
    ("SetMIL[BAL]",    lambda s: _read_unimodal_ablation(s, "BAL"),      "#4db6ac"),
    ("SetMIL[CT]",     lambda s: _read_unimodal_ablation(s, "CT"),       "#26a69a"),
    ("SetMIL[Clin]",   lambda s: _read_unimodal_ablation(s, "Clinical"), "#00897b"),
    # Fusion baselines
    ("Early Fusion",   lambda s: _read_pertask(s, "early"),       "#90caf9"),
    ("Late Fusion",    lambda s: _read_pertask(s, "late"),        "#64b5f6"),
    ("Middle (CMT)",   lambda s: _read_pertask(s, "middle"),      "#42a5f5"),
    # Our methods
    ("LongMK-MT",      lambda s: _read_mega(s, "longitudinal_mk_mt_mega", "longitudinal_mk_mt"), "#ffa726"),
    ("CoAttn-MT",      lambda s: _read_mega(s, "coattn_mt_mega", "coattn_mt"), "#ab47bc"),
    ("SetMIL-MT",      lambda s: _read_mega(s, "set_mil_mt_mega", "set_mil_mt"), "#f44336"),
]

METRIC_LABELS = {
    "bacc":     "ACR BACC",
    "acr_ci":   "ACR C-index",
    "clad_ci":  "CLAD C-index",
    "death_ci": "Death C-index",
}
METRICS = list(METRIC_LABELS.keys())


def collect_all():
    """Collect mean±std for each method across splits."""
    rows = []
    for label, reader, color in METHODS:
        per_split = []
        for s in range(N_SPLITS):
            res = reader(s)
            if res is not None:
                per_split.append(res)
        if not per_split:
            print(f"  [skip] {label} — no data found")
            continue
        arr = {m: [r[m] for r in per_split if not np.isnan(r[m])] for m in METRICS}
        means = {m: np.mean(v) if v else float("nan") for m, v in arr.items()}
        stds  = {m: np.std(v)  if len(v) > 1 else 0.0  for m, v in arr.items()}
        n     = {m: len(v) for m, v in arr.items()}
        rows.append({"label": label, "color": color, "means": means, "stds": stds, "n": n})
        print(f"  {label:20s}  "
              f"BACC={means['bacc']:.3f}±{stds['bacc']:.3f}(n={n['bacc']})  "
              f"ACR_CI={means['acr_ci']:.3f}±{stds['acr_ci']:.3f}  "
              f"CLAD_CI={means['clad_ci']:.3f}±{stds['clad_ci']:.3f}  "
              f"Death_CI={means['death_ci']:.3f}±{stds['death_ci']:.3f}")
    return rows


def write_tsv(rows, out_path):
    lines = ["Method\t" + "\t".join(f"{m}_mean\t{m}_std\tn" for m in METRICS)]
    for r in rows:
        vals = "\t".join(
            f"{r['means'][m]:.4f}\t{r['stds'][m]:.4f}\t{r['n'][m]}" for m in METRICS
        )
        lines.append(f"{r['label']}\t{vals}")
    out_path.write_text("\n".join(lines))
    print(f"[tsv] → {out_path}")


def plot_figure(rows, out_dir):
    fig = plt.figure(figsize=(20, 14))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    ax_bacc  = fig.add_subplot(gs[0, 0])
    ax_acr   = fig.add_subplot(gs[0, 1])
    ax_clad  = fig.add_subplot(gs[1, 0])
    ax_death = fig.add_subplot(gs[1, 1])

    axes_metrics = [
        (ax_bacc,  "bacc",     "ACR Classification (BACC)"),
        (ax_acr,   "acr_ci",   "ACR Survival (C-index)"),
        (ax_clad,  "clad_ci",  "CLAD Survival (C-index)"),
        (ax_death, "death_ci", "Death Survival (C-index)"),
    ]

    for ax, metric, title in axes_metrics:
        valid = [(r["label"], r["means"][metric], r["stds"][metric], r["color"])
                 for r in rows if not np.isnan(r["means"][metric])]
        if not valid:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=11, fontweight="bold")
            continue

        # Sort descending by mean
        valid.sort(key=lambda x: x[1], reverse=True)
        labels, means, stds, colors = zip(*valid)
        x = np.arange(len(labels))

        bars = ax.bar(x, means, color=colors, width=0.6, alpha=0.85, zorder=3)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#333", elinewidth=1.5,
                    capsize=4, capthick=1.5, zorder=4)

        # Annotate bar tops
        for xi, (m, s) in enumerate(zip(means, stds)):
            ax.text(xi, m + s + 0.005, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=6.5, color="#333")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.08)
        ax.axhline(0.5, color="#aaa", linestyle="--", linewidth=0.8, zorder=2)
        ax.grid(axis="y", alpha=0.3, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Chicago Lung Transplant — Multimodal MIL Benchmark", fontsize=14, fontweight="bold")

    for ext in ("pdf", "png"):
        p = out_dir / f"benchmark_chicago.{ext}"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"[saved] {p}")
    plt.close(fig)

    # Heatmap: methods × metrics
    fig2, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.45)))
    mat = np.array([[r["means"][m] for m in METRICS] for r in rows])
    im  = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    for i in range(len(rows)):
        for j, m in enumerate(METRICS):
            v = rows[i]["means"][m]
            s = rows[i]["stds"][m]
            txt = f"{v:.3f}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.75 else "black")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("Benchmark Heatmap (mean across splits)", fontsize=11, fontweight="bold")
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        p = out_dir / f"benchmark_chicago_heatmap.{ext}"
        fig2.savefig(p, dpi=180, bbox_inches="tight")
        print(f"[saved] {p}")
    plt.close(fig2)


def wandb_log(rows):
    try:
        import wandb
        run = wandb.init(project="chicago-mil-benchmark",
                         name="benchmark_chicago_summary",
                         config={"n_splits": N_SPLITS})
        table = wandb.Table(columns=["method"] + [f"{m}_{s}" for m in METRICS
                                                   for s in ("mean", "std", "n")])
        for r in rows:
            row_vals = [r["label"]]
            for m in METRICS:
                row_vals += [r["means"][m], r["stds"][m], r["n"][m]]
            table.add_data(*row_vals)
        run.log({"benchmark_table": table})
        # Log best method per metric
        for m in METRICS:
            valid = [(r["label"], r["means"][m]) for r in rows if not np.isnan(r["means"][m])]
            if valid:
                best_label, best_val = max(valid, key=lambda x: x[1])
                run.summary[f"best_{m}_method"] = best_label
                run.summary[f"best_{m}_value"]  = best_val
        run.finish()
        print("[wandb] logged")
    except Exception as e:
        print(f"[wandb] failed: {e}")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--out-dir", default=str(RESULTS))
    pa.add_argument("--wandb",   action="store_true")
    args = pa.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Chicago Benchmark — collecting metrics")
    print("=" * 60)
    rows = collect_all()

    if not rows:
        print("[ERROR] No results found — check paths.")
        sys.exit(1)

    write_tsv(rows, out_dir / "benchmark_chicago.tsv")
    plot_figure(rows, out_dir)

    if args.wandb:
        wandb_log(rows)

    print("Done.")


if __name__ == "__main__":
    main()
