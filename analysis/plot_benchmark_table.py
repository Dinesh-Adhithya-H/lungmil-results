#!/usr/bin/env python3
"""
Benchmark table figure: all methods × all tasks.

Rows  : P1 unimodal (4 modalities) + P2 multimodal (early/late/middle/set_mil_mt/longitudinal_mk_mt)
Cols  : ACR-cls (BACC), ACR-surv (C-idx), CLAD-surv (C-idx), Death-surv (C-idx)
Cell  : mean ± std  /  per-split values s0–s4
Color : per-column RdYlGn heat-map (higher = greener)
Green border = best per column
"""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO    = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil")
P1_BASE = REPO / "results/mm_abmil_v8/phase1"
P2_BASE = REPO / "results/mm_abmil_v8/phase2"
OUT_DIR = REPO / "analysis/nature_paper"

# ── Task definitions ───────────────────────────────────────────────────────────
#  (display label, p1_dir, p2_key, metric_key)
TASKS = [
    ("ACR Classification\n(BACC ↑)",  "acr",      "acr_cls",   "bacc"),
    ("ACR Survival\n(C-index ↑)",     "acr_surv",  "acr_surv",  "c_index"),
    ("CLAD Survival\n(C-index ↑)",    "clad",      "clad_surv", "c_index"),
    ("Death Survival\n(C-index ↑)",   "death",     "death_surv","c_index"),
]
N_T = len(TASKS)

# ── Method catalog ─────────────────────────────────────────────────────────────
# Each entry: (group, key, display_label, is_ours)
METHODS = [
    # P1 unimodals
    ("P1 Unimodal",  "p1_HE",       "HE only (ABMIL)",             False),
    ("P1 Unimodal",  "p1_BAL",      "BAL only (ABMIL)",            False),
    ("P1 Unimodal",  "p1_CT",       "CT only (ABMIL)",             False),
    ("P1 Unimodal",  "p1_Clinical", "Clinical only (ABMIL)",       False),
    # P2 fusion
    ("P2 Multimodal", "early",              "Early Fusion",                False),
    ("P2 Multimodal", "late",               "Late Fusion",                 False),
    ("P2 Multimodal", "middle",             "Middle Fusion (CrossModal)",   False),
    ("P2 Multimodal", "set_mil_mt",         "SetMIL-MT (ours)",            True),
    ("P2 Multimodal", "longitudinal_mk_mt", "LongMIL-MT (ours)",           True),
]
MODALITIES = ["HE", "BAL", "CT", "Clinical"]


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_json_val(path, metric_key):
    if not path.exists():
        return None
    try:
        d = json.load(open(path))
        test = d.get("test", d)
        v = test.get(metric_key)
        return float(v) if v is not None else None
    except Exception:
        return None


def load_all():
    """Return dict: key → list[float|None] len=5 (one per split)."""
    data = {}

    # P1 unimodals
    for mod in MODALITIES:
        key = f"p1_{mod}"
        for task_lbl, p1_dir, _, metric_key in TASKS:
            tk = [t[2] for t in TASKS if t[1] == p1_dir][0]  # p2_key
            vals = []
            for s in range(5):
                p = P1_BASE / f"split{s}_fold0/{p1_dir}/{mod}/final_combined/metrics.json"
                vals.append(_load_json_val(p, metric_key))
            data.setdefault(key, {})[tk] = vals

    # P2 early/late/middle (per-task)
    for vtag in ("early", "late", "middle"):
        for _, _, p2_key, metric_key in TASKS:
            dir_suffix = "cls" if p2_key == "acr_cls" else p2_key
            vals = []
            for s in range(5):
                p = P2_BASE / f"split{s}_fold0/{vtag}_{dir_suffix}/metrics_{vtag}_final.json"
                vals.append(_load_json_val(p, metric_key))
            data.setdefault(vtag, {})[p2_key] = vals

    # P2 set_mil_mt (per-task)
    vtag = "set_mil_mt"
    for _, _, p2_key, metric_key in TASKS:
        dir_suffix = "cls" if p2_key == "acr_cls" else p2_key
        vals = []
        for s in range(5):
            p = P2_BASE / f"split{s}_fold0/{vtag}_{dir_suffix}/metrics_{vtag}_final.json"
            vals.append(_load_json_val(p, metric_key))
        data.setdefault(vtag, {})[p2_key] = vals

    # P2 longitudinal_mk_mt (per-task)
    # Metrics file has nested structure: test → {acr_cls, acr_surv, clad, death} → metric
    vtag = "longitudinal_mk_mt"
    task_dir_map = {
        "acr_cls":   ("cls",      "acr_cls",  "bacc"),
        "acr_surv":  ("acr_surv", "acr_surv", "c_index"),
        "clad_surv": ("clad_surv","clad",      "c_index"),
        "death_surv":("death_surv","death",    "c_index"),
    }
    for p2_key, (dir_suffix, file_task_key, metric_key) in task_dir_map.items():
        vals = []
        for s in range(5):
            p = P2_BASE / f"split{s}_fold0/{vtag}_{dir_suffix}/metrics_{vtag}_final.json"
            if not p.exists():
                vals.append(None)
                continue
            try:
                d = json.load(open(p))
                test = d.get("test", d)
                # nested: test[file_task_key][metric_key]
                sub = test.get(file_task_key, {})
                v = sub.get(metric_key) if isinstance(sub, dict) else None
                vals.append(float(v) if v is not None else None)
            except Exception:
                vals.append(None)
        data.setdefault(vtag, {})[p2_key] = vals

    return data


def split_stats(vals):
    valid = [v for v in vals if v is not None]
    if not valid:
        return None, None
    return float(np.mean(valid)), float(np.std(valid)) if len(valid) > 1 else 0.0


# ── Figure drawing ─────────────────────────────────────────────────────────────

def draw_table(all_data, out_dir, wandb_project="none"):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build value matrix  [n_methods × n_tasks]
    n_m = len(METHODS)
    means  = np.full((n_m, N_T), np.nan)
    stds   = np.full((n_m, N_T), np.nan)
    splits = [[None] * N_T for _ in range(n_m)]

    task_keys = [t[2] for t in TASKS]

    for mi, (_, mkey, _, _) in enumerate(METHODS):
        mdata = all_data.get(mkey, {})
        for ti, tk in enumerate(task_keys):
            sv = mdata.get(tk)
            if sv is not None:
                m, s = split_stats(sv)
                if m is not None:
                    means[mi, ti] = m
                    stds[mi, ti]  = s
                    splits[mi][ti] = sv

    # Per-column normalisation for background colour
    col_min = np.nanmin(means, axis=0)
    col_max = np.nanmax(means, axis=0)
    col_rng = np.where(col_max - col_min < 1e-6, 1.0, col_max - col_min)
    norm_means = (means - col_min) / col_rng

    best_idx = np.nanargmax(means, axis=0)   # index of best row per column

    # ── Layout ────────────────────────────────────────────────────────────────
    # Row structure: header | sep-P1 | 4 P1 rows | sep-P2 | 5 P2 rows
    # We track actual row indices:
    ROW_HEADER = 0
    ROW_SEP1   = 1
    P1_START   = 2        # rows 2–5
    ROW_SEP2   = 6
    P2_START   = 7        # rows 7–11

    DATA_H   = 1.10   # height per data row
    HEADER_H = 0.70
    SEP_H    = 0.30

    row_heights = (
        [HEADER_H, SEP_H]
        + [DATA_H] * 4           # P1
        + [SEP_H]
        + [DATA_H] * 5           # P2
    )
    total_rows = len(row_heights)   # 12

    # Method row mapping (METHODS list index → figure row index)
    def fig_row(mi):
        if mi < 4:   return P1_START + mi
        else:        return P2_START + (mi - 4)

    COL_W_LABEL = 2.80
    COL_W_TASK  = 2.40
    col_widths  = [COL_W_LABEL] + [COL_W_TASK] * N_T

    fig_w = sum(col_widths) + 0.5
    fig_h = sum(row_heights) + 0.55

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")

    CMAP        = cm.RdYlGn
    HDR_BG      = "#1c2833"
    HDR_FG      = "white"
    SEP_BG      = "#d5d8dc"
    SEP_FG      = "#444"
    ODD_BG      = "#f4f6f7"
    EVEN_BG     = "#ffffff"
    BEST_EDGE   = "#1e8449"
    MISSING_BG  = "#f0f0f0"
    OURS_FG     = "#154360"

    PAD_L = 0.25   # left margin (figure units)
    PAD_T = 0.30   # top margin

    def row_y(ri):
        """Top y in figure units for row ri."""
        return fig_h - PAD_T - sum(row_heights[:ri])

    def cell_rect(ri, ci):
        x = PAD_L + sum(col_widths[:ci])
        y = row_y(ri) - row_heights[ri]
        w = col_widths[ci]
        h = row_heights[ri]
        return x, y, w, h

    def add_rect(ri, ci, fc, ec="#cccccc", lw=0.6):
        x, y, w, h = cell_rect(ri, ci)
        r = mpatches.FancyBboxPatch(
            (x / fig_w, y / fig_h), w / fig_w, h / fig_h,
            boxstyle="square,pad=0",
            transform=fig.transFigure,
            facecolor=fc, edgecolor=ec, linewidth=lw, clip_on=False)
        fig.add_artist(r)

    def add_text(ri, ci, main, sub="", fs=8.8, bold=False, color="#111111",
                 halign="center"):
        x, y, w, h = cell_rect(ri, ci)
        cx = (x + w / 2) / fig_w
        cy = (y + h / 2) / fig_h
        weight = "bold" if bold else "normal"
        ha     = halign
        if sub:
            ax.text(cx, cy + 0.013, main, ha=ha, va="center",
                    fontsize=fs, fontweight=weight, color=color,
                    transform=fig.transFigure)
            ax.text(cx, cy - 0.013, sub, ha=ha, va="center",
                    fontsize=5.8, fontweight="normal", color="#666666",
                    transform=fig.transFigure,
                    fontfamily="monospace")
        else:
            ax.text(cx, cy, main, ha=ha, va="center",
                    fontsize=fs, fontweight=weight, color=color,
                    transform=fig.transFigure)

    # ── Header ────────────────────────────────────────────────────────────────
    for ci in range(N_T + 1):
        add_rect(ROW_HEADER, ci, HDR_BG, ec=HDR_BG)
    add_text(ROW_HEADER, 0, "Method", fs=10, bold=True, color=HDR_FG)
    for ti, (lbl, _, _, _) in enumerate(TASKS):
        add_text(ROW_HEADER, ti + 1, lbl, fs=8.8, bold=True, color=HDR_FG)

    # ── Section separators ────────────────────────────────────────────────────
    for ci in range(N_T + 1):
        add_rect(ROW_SEP1, ci, SEP_BG, ec=SEP_BG)
        add_rect(ROW_SEP2, ci, SEP_BG, ec=SEP_BG)
    add_text(ROW_SEP1, 0, "Unimodal baselines (Phase 1)",
             fs=8, bold=True, color=SEP_FG)
    add_text(ROW_SEP2, 0, "Multimodal fusion (Phase 2)",
             fs=8, bold=True, color=SEP_FG)

    # ── Data rows ─────────────────────────────────────────────────────────────
    for mi, (group, mkey, label, is_ours) in enumerate(METHODS):
        ri   = fig_row(mi)
        parity = mi % 2
        bg   = ODD_BG if parity == 0 else EVEN_BG

        # Method name cell
        for ci in range(N_T + 1):
            add_rect(ri, ci, bg)
        fg = OURS_FG if is_ours else "#1a1a1a"
        add_text(ri, 0, label, fs=8.8, bold=is_ours, color=fg, halign="center")

        # Task cells
        for ti in range(N_T):
            global_mi = mi   # method index into means
            is_best   = (best_idx[ti] == global_mi)
            nv        = norm_means[global_mi, ti]
            m         = means[global_mi, ti]
            s         = stds[global_mi, ti]
            sv        = splits[global_mi][ti]

            if np.isnan(nv):
                cell_bg = MISSING_BG
            else:
                rgba    = CMAP(0.12 + nv * 0.76)
                cell_bg = (*rgba[:3], 0.38)

            ec  = BEST_EDGE if is_best else "#cccccc"
            lw  = 2.0       if is_best else 0.6
            add_rect(ri, ti + 1, cell_bg, ec=ec, lw=lw)

            if np.isnan(m):
                add_text(ri, ti + 1, "—", fs=9, color="#aaaaaa")
            else:
                main = f"{m:.3f} ± {s:.3f}"
                per  = " ".join(
                    f"s{si}:{v:.3f}" if v is not None else f"s{si}:—"
                    for si, v in enumerate(sv)
                )
                add_text(ri, ti + 1, main, sub=per,
                         fs=8.8, bold=is_best, color="#111111")

    # ── Colour-bar legend ──────────────────────────────────────────────────────
    cb_ax = fig.add_axes([0.60, 0.012, 0.28, 0.016])
    cb = plt.colorbar(
        cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
        cax=cb_ax, orientation="horizontal")
    cb.set_label("Relative performance per task", fontsize=6.5)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Low", "Mid", "High"])
    cb.ax.tick_params(labelsize=6)

    # ── Marker legend ─────────────────────────────────────────────────────────
    lh = [
        mpatches.Patch(facecolor="none", edgecolor=BEST_EDGE, lw=2.0,
                       label="Best per task (green border)"),
        mpatches.Patch(facecolor=MISSING_BG, edgecolor="#ccc", lw=0.6,
                       label="Not yet available"),
    ]
    fig.legend(handles=lh, loc="lower left",
               bbox_to_anchor=(0.02, 0.010), fontsize=6.5,
               framealpha=0.9, handlelength=1.4)

    fig.suptitle(
        "Multimodal MIL Benchmark  —  All Methods × All Tasks",
        fontsize=12.5, fontweight="bold", y=0.997)

    note = ("BACC = balanced accuracy (classification). "
            "C-index = Harrell concordance (survival). "
            "Values: mean ± std / per-split (s0–s4). "
            "Our models in bold blue.")
    fig.text(0.02, 0.003, note, fontsize=6, color="#666666", va="bottom")

    # ── Save ──────────────────────────────────────────────────────────────────
    png = out_dir / "benchmark_table_full.png"
    pdf = out_dir / "benchmark_table_full.pdf"
    fig.savefig(pdf, dpi=100, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    plt.close(fig)

    # ── W&B ───────────────────────────────────────────────────────────────────
    if wandb_project.lower() != "none":
        _log_wandb(wandb_project, png, means, stds, splits, task_keys)


def _log_wandb(project, png_path, means, stds, splits, task_keys):
    try:
        import wandb
        task_short = ["ACR-cls", "ACR-surv", "CLAD-surv", "Death-surv"]
        run = wandb.init(project=project, name="benchmark_table_full",
                         group="benchmark", reinit=True)

        # Summary table: one row per method
        cols = ["Method", "Phase"] + task_short + [f"{t}_std" for t in task_short]
        rows = []
        for mi, (group, mkey, label, _) in enumerate(METHODS):
            row = [label, group]
            for ti in range(N_T):
                m = means[mi, ti]
                row.append(round(float(m), 4) if not np.isnan(m) else None)
            for ti in range(N_T):
                s = stds[mi, ti]
                row.append(round(float(s), 4) if not np.isnan(s) else None)
            rows.append(row)
        bench_tbl = wandb.Table(columns=cols, data=rows)

        # Per-split long table
        sp_cols = ["method", "task", "split", "score"]
        sp_rows = []
        for mi, (_, mkey, label, _) in enumerate(METHODS):
            for ti, t_short in enumerate(task_short):
                sv = splits[mi][ti]
                if sv is None:
                    continue
                for si, v in enumerate(sv):
                    if v is not None:
                        sp_rows.append([label, t_short, f"s{si}", round(float(v), 4)])
        split_tbl = wandb.Table(columns=sp_cols, data=sp_rows)

        # Bar-chart table
        bar_cols = ["method", "task", "score"]
        bar_rows = []
        for mi, (_, mkey, label, _) in enumerate(METHODS):
            for ti, t_short in enumerate(task_short):
                m = means[mi, ti]
                if not np.isnan(m):
                    bar_rows.append([label, t_short, round(float(m), 4)])
        bar_tbl = wandb.Table(columns=bar_cols, data=bar_rows)

        log_dict = {
            "benchmark/figure":         wandb.Image(str(png_path),
                                         caption="All methods × all tasks | mean±std"),
            "benchmark/summary_table":  bench_tbl,
            "benchmark/per_split_table": split_tbl,
            "benchmark/bar_chart":      wandb.plot.bar(
                                         bar_tbl, "method", "score",
                                         title="Method comparison (mean across 5 splits)"),
        }
        # Scalar metrics
        for mi, (_, mkey, label, _) in enumerate(METHODS):
            slug = label.replace(" ", "_").replace("(", "").replace(")", "").replace("-","_")
            for ti, t_short in enumerate(task_short):
                m = means[mi, ti]
                if not np.isnan(m):
                    log_dict[f"benchmark/{t_short}/{slug}"] = round(float(m), 4)

        wandb.log(log_dict)
        run.finish()
        print(f"[wandb] {run.url}")
    except Exception as e:
        print(f"[wandb] skipped: {e}")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--out-dir", default=str(OUT_DIR))
    pa.add_argument("--wandb-project", default="chicago-mil-interpretability")
    args = pa.parse_args()

    print("Loading metrics...")
    all_data = load_all()

    # Print data availability summary
    task_keys = [t[2] for t in TASKS]
    print("\n=== Data availability ===")
    for _, mkey, label, _ in METHODS:
        mdata = all_data.get(mkey, {})
        parts = []
        for tk in task_keys:
            sv = mdata.get(tk, [])
            n_done = sum(1 for v in sv if v is not None)
            parts.append(f"{tk[:8]}:{n_done}/5")
        print(f"  {label:<35} {' | '.join(parts)}")

    print("\nDrawing table...")
    draw_table(all_data, Path(args.out_dir), wandb_project=args.wandb_project)


if __name__ == "__main__":
    main()
