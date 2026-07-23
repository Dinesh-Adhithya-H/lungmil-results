#!/usr/bin/env python3
"""
Benchmark comparison figures for WandB / Nature paper.

Figure 1 – benchmark_comparison.png
    All methods × all tasks, grouped bar chart (mean ± std across 5 splits).
    Methods: P1 unimodals | Unimodal Mean | Early | Late | Middle | SetMIL-MT | LongMIL-MT

Figure 2 – unimodal_ablation_setmilmt.png
    SetMIL-MT unimodal ablation vs. P1 unimodal baseline per task.
    Each modality panel: multimodal-model(mod only) vs P1(mod only).

Figure 3 – unimodal_ablation_longmilmt.png
    LongMIL-MT overall vs P1 unimodals per task (proxy ablation, no masking available).

Figure 4 – modality_gain_heatmap.png
    Heatmap: delta between SetMIL-MT unimodal-ablation and P1 unimodal (multimodal context gain).
"""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

REPO    = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil")
P1_BASE = REPO / "results/mm_abmil_v8/phase1"
P2_BASE = REPO / "results/mm_abmil_v8/phase2"
OUT_DIR = REPO / "analysis/nature_paper"

# ── Task definitions ───────────────────────────────────────────────────────────
# (display, p1_dir, p2_key, file_task_key_longmt, metric)
TASKS = [
    ("ACR Classification\n(BACC ↑)",  "acr",      "acr_cls",   "acr_cls",  "bacc"),
    ("ACR Survival\n(C-index ↑)",     "acr_surv",  "acr_surv",  "acr_surv", "c_index"),
    ("CLAD Survival\n(C-index ↑)",    "clad",      "clad_surv", "clad",     "c_index"),
    ("Death Survival\n(C-index ↑)",   "death",     "death_surv","death",    "c_index"),
]
N_T = len(TASKS)
MODALITIES  = ["HE", "BAL", "CT", "Clinical"]
MOD_COLORS  = {"HE": "#e74c3c", "BAL": "#3498db", "CT": "#2ecc71", "Clinical": "#9b59b6"}
MOD_HATCHES = {"HE": "", "BAL": "//", "CT": "xx", "Clinical": ".."}

# ── Color palette ──────────────────────────────────────────────────────────────
C_P1    = "#95a5a6"   # grey  – P1 unimodals
C_MEAN  = "#7f8c8d"   # dark grey – unimodal mean
C_EARLY = "#f39c12"   # amber – early fusion
C_LATE  = "#e67e22"   # orange – late fusion
C_MID   = "#d35400"   # burnt – middle
C_SET   = "#2980b9"   # blue  – SetMIL-MT
C_LONG  = "#1abc9c"   # teal  – LongMIL-MT


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_json(path, *keys):
    if not path.exists():
        return None
    try:
        d = json.load(open(path))
        for k in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
        return float(d) if d is not None else None
    except Exception:
        return None


def split_stats(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    return float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0


def p1_vals(p1_dir, mod, metric):
    return [load_json(P1_BASE / f"split{s}_fold0/{p1_dir}/{mod}/final_combined/metrics.json",
                      "test", metric)
            for s in range(5)]


def p2_vals(vtag, dir_suffix, metric, nested_key=None):
    out = []
    for s in range(5):
        p = P2_BASE / f"split{s}_fold0/{vtag}_{dir_suffix}/metrics_{vtag}_final.json"
        if nested_key:
            out.append(load_json(p, "test", nested_key, metric))
        else:
            out.append(load_json(p, "test", metric))
    return out


def p2_ablation_vals(vtag, dir_suffix, mod, metric):
    """unimodal_ablation[mod][metric] across splits."""
    return [load_json(P2_BASE / f"split{s}_fold0/{vtag}_{dir_suffix}/metrics_{vtag}_final.json",
                      "unimodal_ablation", mod, metric)
            for s in range(5)]


def bar_with_err(ax, x, mean, std, color, label=None, width=0.7, hatch="", alpha=0.85):
    if mean is None:
        return
    b = ax.bar(x, mean, width=width, color=color, label=label, alpha=alpha,
               hatch=hatch, edgecolor="white", linewidth=0.5)
    if std is not None and std > 0:
        ax.errorbar(x, mean, yerr=std, fmt="none", color="#333", capsize=3,
                    elinewidth=1.0, capthick=1.0)
    return b


# ── Figure 1: Method benchmark comparison ─────────────────────────────────────

def fig_benchmark(out_dir):
    # Build data: method → task_key → (mean, std)
    data = {}

    for mod in MODALITIES:
        key = f"P1-{mod}"
        data[key] = {}
        for _, p1_dir, p2_key, _, metric in TASKS:
            m, s = split_stats(p1_vals(p1_dir, mod, metric))
            data[key][p2_key] = (m, s)

    # Unimodal mean (average of 4 P1 modalities per task)
    data["Unimodal\nMean"] = {}
    for _, p1_dir, p2_key, _, metric in TASKS:
        all_vals = []
        for mod in MODALITIES:
            all_vals += [v for v in p1_vals(p1_dir, mod, metric) if v is not None]
        m, s = split_stats(all_vals)
        data["Unimodal\nMean"][p2_key] = (m, s)

    # P2 per-task (early/late/middle)
    for vtag, dir_map in [
        ("early",  {"acr_cls":"cls","acr_surv":"acr_surv","clad_surv":"clad_surv","death_surv":"death_surv"}),
        ("late",   {"acr_cls":"cls","acr_surv":"acr_surv","clad_surv":"clad_surv","death_surv":"death_surv"}),
        ("middle", {"acr_cls":"cls","acr_surv":"acr_surv","clad_surv":"clad_surv","death_surv":"death_surv"}),
    ]:
        lbl = {"early":"Early Fusion","late":"Late Fusion","middle":"Middle Fusion"}[vtag]
        data[lbl] = {}
        for _, _, p2_key, _, metric in TASKS:
            m, s = split_stats(p2_vals(vtag, dir_map[p2_key], metric))
            data[lbl][p2_key] = (m, s)

    # SetMIL-MT
    smt_dirs = {"acr_cls":"cls","acr_surv":"acr_surv","clad_surv":"clad_surv","death_surv":"death_surv"}
    data["SetMIL-MT\n(ours)"] = {}
    for _, _, p2_key, _, metric in TASKS:
        m, s = split_stats(p2_vals("set_mil_mt", smt_dirs[p2_key], metric))
        data["SetMIL-MT\n(ours)"][p2_key] = (m, s)

    # LongMIL-MT
    lmt_dirs = {"acr_cls":"cls","acr_surv":"acr_surv","clad_surv":"clad_surv","death_surv":"death_surv"}
    lmt_nested = {"acr_cls":"acr_cls","acr_surv":"acr_surv","clad_surv":"clad","death_surv":"death"}
    data["LongMIL-MT\n(ours)"] = {}
    for _, _, p2_key, nested_k, metric in TASKS:
        m, s = split_stats(p2_vals("longitudinal_mk_mt", lmt_dirs[p2_key], metric,
                                   nested_key=nested_k))
        data["LongMIL-MT\n(ours)"][p2_key] = (m, s)

    methods = list(data.keys())
    n_m = len(methods)
    colors = (
        [C_P1] * 4 + [C_MEAN]
        + [C_EARLY, C_LATE, C_MID]
        + [C_SET, C_LONG]
    )
    ours_methods = {"SetMIL-MT\n(ours)", "LongMIL-MT\n(ours)"}

    fig, axes = plt.subplots(1, N_T, figsize=(5.5 * N_T, 5.5), sharey=False)
    fig.suptitle("Multimodal MIL Benchmark: Method Comparison",
                 fontsize=13, fontweight="bold")

    xs = np.arange(n_m)
    for ti, (task_lbl, _, p2_key, _, _) in enumerate(TASKS):
        ax = axes[ti]
        for mi, (mkey, color) in enumerate(zip(methods, colors)):
            m_val, s_val = data[mkey].get(p2_key, (None, None))
            is_ours = mkey in ours_methods
            bar_with_err(ax, xs[mi], m_val, s_val, color,
                         width=0.72, alpha=0.9 if is_ours else 0.75)
            if is_ours and m_val is not None:
                ax.bar(xs[mi], m_val, width=0.72, color="none",
                       edgecolor="#1a1a1a", linewidth=1.8)

        ax.set_title(task_lbl, fontsize=10, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(methods, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("BACC" if p2_key == "acr_cls" else "C-index", fontsize=9)
        ax.set_ylim(0.35, 1.0)
        ax.axhline(0.5, color="#aaa", lw=0.8, ls="--", zorder=0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, lw=0.5)

        # Shade P1 region
        ax.axvspan(-0.5, 3.5, alpha=0.04, color="#888")
        ax.axvspan(4.5, n_m - 0.5, alpha=0.04, color="#2980b9")

    # Legend
    handles = [
        mpatches.Patch(color=C_P1,   label="P1 Unimodal"),
        mpatches.Patch(color=C_MEAN, label="Unimodal Mean"),
        mpatches.Patch(color=C_EARLY,label="Early Fusion"),
        mpatches.Patch(color=C_LATE, label="Late Fusion"),
        mpatches.Patch(color=C_MID,  label="Middle Fusion"),
        mpatches.Patch(color=C_SET,  label="SetMIL-MT (ours)"),
        mpatches.Patch(color=C_LONG, label="LongMIL-MT (ours)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, 0.97, "Mean ± std across 5 nested-CV splits", ha="center",
             fontsize=8, color="#666")

    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    p = out_dir / "benchmark_comparison.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {p}")
    return p


# ── Figure 2: SetMIL-MT unimodal ablation vs P1 baseline ─────────────────────

def fig_ablation_setmilmt(out_dir):
    smt_dirs = {"acr_cls":"cls","acr_surv":"acr_surv","clad_surv":"clad_surv","death_surv":"death_surv"}
    fig, axes = plt.subplots(1, N_T, figsize=(5.0 * N_T, 5.5), sharey=False)
    fig.suptitle("SetMIL-MT: Unimodal Ablation vs P1 Unimodal Baseline",
                 fontsize=13, fontweight="bold")

    xs = np.arange(len(MODALITIES))
    width = 0.35

    for ti, (task_lbl, p1_dir, p2_key, _, metric) in enumerate(TASKS):
        ax = axes[ti]
        for mi, mod in enumerate(MODALITIES):
            color = MOD_COLORS[mod]
            # SetMIL-MT ablation (multimodal model, single-modality input)
            abl = [v for v in p2_ablation_vals("set_mil_mt", smt_dirs[p2_key], mod, metric)
                   if v is not None]
            m_abl, s_abl = split_stats(abl) if abl else (None, None)
            # P1 unimodal baseline
            p1 = [v for v in p1_vals(p1_dir, mod, metric) if v is not None]
            m_p1, s_p1 = split_stats(p1) if p1 else (None, None)

            bar_with_err(ax, xs[mi] - width / 2, m_abl, s_abl, color,
                         width=width, hatch="", alpha=0.85)
            bar_with_err(ax, xs[mi] + width / 2, m_p1,  s_p1,  color,
                         width=width, hatch="//", alpha=0.55)

            # Delta annotation
            if m_abl is not None and m_p1 is not None:
                delta = m_abl - m_p1
                top = max(v for v in [m_abl, m_p1] if v is not None)
                sign = "+" if delta >= 0 else ""
                ax.text(xs[mi], top + 0.015, f"{sign}{delta:.3f}",
                        ha="center", va="bottom", fontsize=6.5,
                        color="#1a1a1a", fontweight="bold")

        ax.set_title(task_lbl, fontsize=10, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(MODALITIES, fontsize=9)
        ax.set_ylabel("BACC" if p2_key == "acr_cls" else "C-index", fontsize=9)
        ax.set_ylim(0.30, 1.0)
        ax.axhline(0.5, color="#aaa", lw=0.8, ls="--", zorder=0)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, lw=0.5)

    # Legend
    solid = mpatches.Patch(color="#888", alpha=0.85, label="SetMIL-MT (single modality input)")
    hatch = mpatches.Patch(color="#888", hatch="//", alpha=0.55, label="P1 Unimodal baseline")
    fig.legend(handles=[solid, hatch], loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, -0.04,
             "Δ = SetMIL-MT(mod only) − P1(mod only). Positive = multimodal context benefit.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    p = out_dir / "unimodal_ablation_setmilmt.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {p}")
    return p


# ── Figure 3: LongMIL-MT vs SetMIL-MT vs P1 unimodal ─────────────────────────

def fig_longmilmt_vs_setmilmt(out_dir):
    smt_dirs = {"acr_cls":"cls","acr_surv":"acr_surv","clad_surv":"clad_surv","death_surv":"death_surv"}
    lmt_dirs = {"acr_cls":"cls","acr_surv":"acr_surv","clad_surv":"clad_surv","death_surv":"death_surv"}
    lmt_nested = {"acr_cls":"acr_cls","acr_surv":"acr_surv","clad_surv":"clad","death_surv":"death"}

    fig, axes = plt.subplots(1, N_T, figsize=(5.0 * N_T, 5.5), sharey=False)
    fig.suptitle("LongMIL-MT vs SetMIL-MT vs P1 Unimodals",
                 fontsize=13, fontweight="bold")

    xs = np.arange(len(MODALITIES))
    width = 0.22

    for ti, (task_lbl, p1_dir, p2_key, nested_k, metric) in enumerate(TASKS):
        ax = axes[ti]

        # SetMIL-MT overall and ablation
        smt_overall = split_stats(p2_vals("set_mil_mt", smt_dirs[p2_key], metric))
        lmt_overall = split_stats(p2_vals("longitudinal_mk_mt", lmt_dirs[p2_key],
                                          metric, nested_key=nested_k))

        for mi, mod in enumerate(MODALITIES):
            color = MOD_COLORS[mod]
            # P1 unimodal
            p1 = p1_vals(p1_dir, mod, metric)
            m_p1, s_p1 = split_stats(p1)
            bar_with_err(ax, xs[mi] - width, m_p1, s_p1, color,
                         width=width, hatch="//", alpha=0.55, label=mod if ti == 0 else None)

            # SetMIL-MT ablation (mod only)
            abl_smt = p2_ablation_vals("set_mil_mt", smt_dirs[p2_key], mod, metric)
            m_smt_abl, s_smt_abl = split_stats(abl_smt)
            bar_with_err(ax, xs[mi], m_smt_abl, s_smt_abl, color,
                         width=width, hatch="", alpha=0.85)

        # SetMIL-MT overall as horizontal line
        if smt_overall[0] is not None:
            ax.axhline(smt_overall[0], color=C_SET, lw=2.0, ls="-",
                       label=f"SetMIL-MT all: {smt_overall[0]:.3f}")
            ax.axhspan(smt_overall[0] - smt_overall[1],
                       smt_overall[0] + smt_overall[1],
                       alpha=0.12, color=C_SET)

        # LongMIL-MT overall as horizontal line
        if lmt_overall[0] is not None:
            ax.axhline(lmt_overall[0], color=C_LONG, lw=2.0, ls="--",
                       label=f"LongMIL-MT all: {lmt_overall[0]:.3f}")
            ax.axhspan(lmt_overall[0] - lmt_overall[1],
                       lmt_overall[0] + lmt_overall[1],
                       alpha=0.12, color=C_LONG)

        ax.set_title(task_lbl, fontsize=10, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(MODALITIES, fontsize=9)
        ax.set_ylabel("BACC" if p2_key == "acr_cls" else "C-index", fontsize=9)
        ax.set_ylim(0.30, 1.0)
        ax.axhline(0.5, color="#aaa", lw=0.8, ls=":", zorder=0)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        ax.legend(fontsize=7, framealpha=0.7, loc="upper right")

    # Bottom legend
    handles = [
        mpatches.Patch(color="#888", hatch="//", alpha=0.55, label="P1 Unimodal"),
        mpatches.Patch(color="#888", hatch="", alpha=0.85, label="SetMIL-MT (mod only input)"),
        plt.Line2D([0],[0], color=C_SET,  lw=2, label="SetMIL-MT (all mods)"),
        plt.Line2D([0],[0], color=C_LONG, lw=2, ls="--", label="LongMIL-MT (all mods)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    p = out_dir / "unimodal_ablation_longmilmt.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {p}")
    return p


# ── Figure 4: Modality gain heatmap (SetMIL-MT context gain over P1) ──────────

def fig_modality_gain_heatmap(out_dir):
    smt_dirs = {"acr_cls":"cls","acr_surv":"acr_surv","clad_surv":"clad_surv","death_surv":"death_surv"}
    task_labels = [t[0].replace("\n", " ") for t in TASKS]
    task_keys   = [t[2] for t in TASKS]
    task_p1dirs = [t[1] for t in TASKS]
    task_metrics= [t[4] for t in TASKS]

    gain = np.full((len(MODALITIES), N_T), np.nan)

    for ti, (p2_key, p1_dir, metric) in enumerate(zip(task_keys, task_p1dirs, task_metrics)):
        for mi, mod in enumerate(MODALITIES):
            abl = [v for v in p2_ablation_vals("set_mil_mt", smt_dirs[p2_key], mod, metric)
                   if v is not None]
            p1  = [v for v in p1_vals(p1_dir, mod, metric) if v is not None]
            if abl and p1:
                gain[mi, ti] = np.mean(abl) - np.mean(p1)

    vabs = np.nanmax(np.abs(gain))
    fig, ax = plt.subplots(figsize=(N_T * 1.8 + 1, len(MODALITIES) * 1.3 + 1.2))
    im = ax.imshow(gain, cmap="RdYlGn", vmin=-vabs, vmax=vabs, aspect="auto")

    ax.set_xticks(range(N_T))
    ax.set_xticklabels(task_labels, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(MODALITIES)))
    ax.set_yticklabels(MODALITIES, fontsize=10)

    for mi in range(len(MODALITIES)):
        for ti in range(N_T):
            v = gain[mi, ti]
            if not np.isnan(v):
                sign = "+" if v >= 0 else ""
                ax.text(ti, mi, f"{sign}{v:.3f}", ha="center", va="center",
                        fontsize=9.5, fontweight="bold",
                        color="black" if abs(v) < vabs * 0.5 else "white")

    plt.colorbar(im, ax=ax, label="Δ metric (SetMIL-MT ablation − P1 unimodal)",
                 fraction=0.04, pad=0.04)
    ax.set_title("Multimodal Context Gain per Modality\n"
                 "(SetMIL-MT single-mod input vs P1 unimodal baseline)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "modality_gain_heatmap.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {p}")
    return p


# ── WandB logging ──────────────────────────────────────────────────────────────

def log_wandb(project, figures):
    try:
        import wandb
        run = wandb.init(project=project, name="benchmark_figures",
                         group="benchmark", reinit=True)
        log = {}
        captions = {
            "benchmark_comparison":      "Method comparison: all methods × all tasks",
            "unimodal_ablation_setmilmt":"SetMIL-MT unimodal ablation vs P1 baseline",
            "unimodal_ablation_longmilmt":"LongMIL-MT vs SetMIL-MT vs P1 unimodals",
            "modality_gain_heatmap":     "Multimodal context gain heatmap",
        }
        for name, path in figures.items():
            if path and path.exists():
                log[f"benchmark/{name}"] = wandb.Image(
                    str(path), caption=captions.get(name, name))
        wandb.log(log)
        run.finish()
        print(f"[wandb] {run.url}")
    except Exception as e:
        print(f"[wandb] skipped: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--out-dir", default=str(OUT_DIR))
    pa.add_argument("--wandb-project", default="chicago-mil-interpretability")
    args = pa.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    figures = {}
    print("Figure 1: Method benchmark comparison...")
    figures["benchmark_comparison"] = fig_benchmark(out)

    print("Figure 2: SetMIL-MT unimodal ablation...")
    figures["unimodal_ablation_setmilmt"] = fig_ablation_setmilmt(out)

    print("Figure 3: LongMIL-MT vs SetMIL-MT vs P1...")
    figures["unimodal_ablation_longmilmt"] = fig_longmilmt_vs_setmilmt(out)

    print("Figure 4: Modality gain heatmap...")
    figures["modality_gain_heatmap"] = fig_modality_gain_heatmap(out)

    if args.wandb_project.lower() != "none":
        log_wandb(args.wandb_project, figures)


if __name__ == "__main__":
    main()
