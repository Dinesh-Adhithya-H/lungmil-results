#!/usr/bin/env python3
"""
PPT-ready mechanism diagrams for MM-ABMIL v6 fusion methods.
Generates one overview PNG (2×3 grid) + 7 individual PNGs.

Run:  python plot_fusion_mechanisms.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT_DIR = Path("/home/aih/dinesh.haridoss/chicago_mil/analysis_v6_milv2/mechanism_diagrams")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "svg.fonttype":      "none",   # keep text as real SVG text (editable in Inkscape/PPT)
})

# ── Palette ──────────────────────────────────────────────────────────────────
MC = {
    "HE":       "#2E7D32",
    "BAL":      "#1565C0",
    "CT":       "#BF360C",
    "Clinical": "#4A148C",
}
MF = {m: c + "28" for m, c in MC.items()}

OP = {
    "abmil":  ("#E8F5E9", "#388E3C"),
    "xfmr":   ("#E3F2FD", "#1565C0"),
    "xattn":  ("#FFF3E0", "#E65100"),
    "slot":   ("#F3E5F5", "#7B1FA2"),
    "pool":   ("#ECEFF1", "#546E7A"),
    "out":    ("#FFEBEE", "#C62828"),
    "concat": ("#F5F5F5", "#757575"),
    "iter":   ("#FFF9C4", "#F9A825"),
    "merge":  ("#E8EAF6", "#3949AB"),
}
AC    = "#37474F"    # arrow / line colour
MODS  = ["HE", "BAL", "CT"]
YM    = [3.0, 2.0, 1.0]   # y-centres for 3 modality rows


# ── Drawing primitives ───────────────────────────────────────────────────────

def _setup(ax, title, xlim=(0, 12), ylim=(0, 4.1)):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8,
                 color="#1a1a1a", loc="center")


def rbox(ax, cx, cy, w, h, text, style="abmil", fs=7.5, fw="normal",
         lw=1.5, zorder=3, fc=None, ec=None):
    fc_ = fc if fc is not None else OP[style][0]
    ec_ = ec if ec is not None else OP[style][1]
    pad = min(w, h) * 0.06
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad={pad}",
        facecolor=fc_, edgecolor=ec_, linewidth=lw,
        zorder=zorder, clip_on=False))
    if text:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fs, fontweight=fw, color="#1a1a1a",
                zorder=zorder + 1, clip_on=False, multialignment="center",
                linespacing=1.3)


def arr(ax, x1, y1, x2, y2, color=AC, lw=1.4, ms=9):
    """Straight arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=ms,
                                connectionstyle="arc3,rad=0"),
                zorder=2, annotation_clip=False)


def bus_merge(ax, src_rxs, src_ys, tgt_lx, tgt_cy, color=AC, lw=1.5):
    """
    Right-angle comb: multiple sources → vertical bus → single target.
    src_rxs : list of right-edge x for each source
    src_ys  : list of y-centres for each source
    tgt_lx  : left-edge x of target box
    tgt_cy  : y-centre of target box
    """
    bus_x = max(src_rxs) + (tgt_lx - max(src_rxs)) * 0.32
    y_lo, y_hi = min(src_ys), max(src_ys)
    kw = dict(color=color, lw=lw, zorder=1,
              solid_capstyle="round", solid_joinstyle="round")
    for rx, ry in zip(src_rxs, src_ys):
        ax.plot([rx, bus_x], [ry, ry], "-", **kw)
    ax.plot([bus_x, bus_x], [y_lo, y_hi], "-", **kw)
    # final arrow into target
    ax.annotate("", xy=(tgt_lx, tgt_cy), xytext=(bus_x, tgt_cy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=9),
                zorder=2, annotation_clip=False)


def highlight(ax, x0, x1, y0=0.4, y1=3.6, color="#EDE7F6", alpha=0.45, zorder=0):
    ax.fill_betweenx([y0, y1], x0, x1, color=color, alpha=alpha, zorder=zorder)


def mod_inputs(ax, x=0.7, w=1.05, h=0.55, lw=2.0):
    for mod, y in zip(MODS, YM):
        rbox(ax, x, y, w, h, f"{mod}\npatches",
             fc=MF[mod], ec=MC[mod], fs=7.5, fw="bold", lw=lw)
    return x + w / 2   # right edge


# ── Phase overview ────────────────────────────────────────────────────────────

def draw_phase_overview(ax):
    _setup(ax, "Two-Phase Training Pipeline", xlim=(0, 12))
    highlight(ax, 0.3, 5.0, color="#E8F5E9", y0=0.3, y1=3.7)
    ax.text(2.65, 3.85, "Phase 1 — Single-Modal", ha="center",
            fontsize=8, color="#2E7D32", fontweight="bold")
    for mod, y in zip(MODS, YM):
        rbox(ax, 1.2, y, 1.1, 0.55, f"{mod}\npatches",
             fc=MF[mod], ec=MC[mod], fs=7.5, fw="bold", lw=2.0)
        arr(ax, 1.75, y, 2.4, y)
        rbox(ax, 3.1, y, 1.3, 0.55, "SingleModal\nABMIL", "abmil", fs=7)
        arr(ax, 3.75, y, 4.3, y)
        rbox(ax, 4.8, y, 0.9, 0.55, "P1\nlogit", "out", fs=7)
    ax.text(6.0, 2.0, "→  freeze P1\n    encoders  →", ha="center",
            fontsize=8, color="#546E7A", style="italic", va="center")
    highlight(ax, 6.9, 11.7, color="#E3F2FD", y0=0.3, y1=3.7)
    ax.text(9.3, 3.85, "Phase 2 — Multimodal Fusion", ha="center",
            fontsize=8, color="#1565C0", fontweight="bold")
    for mod, y in zip(MODS, YM):
        rbox(ax, 7.6, y, 1.0, 0.45, f"P1 enc\n({mod})",
             fc=MF[mod], ec=MC[mod], fs=6.5, fw="bold", lw=1.5)
        arr(ax, 8.1, y, 8.7, y)
    rbox(ax, 9.4, 2.0, 1.4, 2.5, "Fusion\nModule\n(6 variants)", "xfmr",
         fw="bold", fs=8, zorder=3)
    arr(ax, 10.1, 2.0, 10.8, 2.0)
    rbox(ax, 11.35, 2.0, 1.0, 0.55, "P2\nlogit", "out", fw="bold", fs=7.5)


# ── Early fusion ──────────────────────────────────────────────────────────────

def draw_early(ax):
    _setup(ax, "Early Fusion\n(concat patches → shared ABMIL)")
    x0 = mod_inputs(ax)
    highlight(ax, 2.25, 5.0, color="#E8EAF6")
    # horizontal arrows into the tall concat box at each modality's y-level
    for y in YM:
        arr(ax, x0, y, 2.45, y)
    rbox(ax, 3.0, 2.0, 1.0, 3.0, "Concat\nall\npatches",
         "concat", lw=1.5, fs=7.5, zorder=3)
    arr(ax, 3.5, 2.0, 4.6, 2.0)
    rbox(ax, 5.3, 2.0, 1.3, 0.6, "Shared\nABMIL", "abmil", fw="bold")
    arr(ax, 5.95, 2.0, 7.0, 2.0)
    rbox(ax, 7.65, 2.0, 1.1, 0.6, "Predict", "out", fw="bold")
    ax.text(3.0, 3.75, "Fusion at patch level", ha="center",
            fontsize=7, color="#3949AB", style="italic")


# ── Late fusion ───────────────────────────────────────────────────────────────

def draw_late(ax):
    _setup(ax, "Late Fusion\n(per-mod decision → weighted average)")
    x0 = mod_inputs(ax)
    for y in YM:
        arr(ax, x0, y, 2.4, y)
        rbox(ax, 3.1, y, 1.2, 0.55, "ABMIL\npool", "abmil", fw="bold")
        arr(ax, 3.7, y, 4.5, y)
        rbox(ax, 5.2, y, 1.1, 0.55, "Per-mod\nlogit", "pool")
    highlight(ax, 6.1, 8.4, color="#EDE7F6")
    # bus merge: right edges of logit boxes → weighted avg
    bus_merge(ax, [5.75] * 3, YM, tgt_lx=6.75, tgt_cy=2.0)
    rbox(ax, 7.4, 2.0, 1.3, 0.6, "Learned\nWeighted\nAvg", "merge", fw="bold", fs=7)
    arr(ax, 8.05, 2.0, 9.1, 2.0)
    rbox(ax, 9.75, 2.0, 1.1, 0.6, "Predict", "out", fw="bold")
    ax.text(7.4, 3.75, "Fusion at decision level", ha="center",
            fontsize=7, color="#3949AB", style="italic")


# ── Middle fusion ─────────────────────────────────────────────────────────────

def draw_middle(ax):
    _setup(ax, "Middle Fusion\n(ABMIL summaries → cross-modal transformer)")
    x0 = mod_inputs(ax)
    for y in YM:
        arr(ax, x0, y, 2.4, y)
        rbox(ax, 3.1, y, 1.2, 0.55, "ABMIL\npool", "abmil", fw="bold")
        arr(ax, 3.7, y, 4.4, y)
        rbox(ax, 5.0, y, 1.0, 0.55, "Summary\ntoken", "pool", fs=7)
    highlight(ax, 5.65, 8.4, color="#E3F2FD")
    # bus merge: summary token right edges → transformer
    bus_merge(ax, [5.5] * 3, YM, tgt_lx=6.5, tgt_cy=2.0)
    rbox(ax, 7.2, 2.0, 1.4, 1.8,
         "Cross-Modal\nTransformer\n(self-attn\nover tokens)", "xfmr", fw="bold", fs=7)
    arr(ax, 7.9, 2.0, 9.1, 2.0)
    rbox(ax, 9.75, 2.0, 1.1, 0.6, "Predict", "out", fw="bold")
    ax.text(7.2, 3.8, "Fusion at summary level", ha="center",
            fontsize=7, color="#1565C0", style="italic")


# ── Cross-attention fusion ────────────────────────────────────────────────────

def draw_crossattn(ax):
    _setup(ax, "Cross-Attention Fusion\n(bidir patch cross-attn → slot attn → XFmr)",
           xlim=(0, 13))
    x0 = mod_inputs(ax)
    highlight(ax, 2.25, 5.3, color="#FFF3E0")

    # input arrows into the bidir block
    for y in YM:
        arr(ax, x0, y, 2.5, y)

    # ── BIDIR CROSS-ATTENTION BLOCK ──────────────────────────────────────
    bk_cx, bk_cy, bk_w, bk_h = 3.7, 2.0, 2.3, 3.4
    rbox(ax, bk_cx, bk_cy, bk_w, bk_h, "", "xattn", lw=2.2, zorder=2)

    # block header label
    ax.text(bk_cx, bk_cy + bk_h/2 - 0.20, "Bidir Patch\nCross-Attention",
            ha="center", va="top", fontsize=7.5, fontweight="bold",
            color="#E65100", zorder=6)

    # mini modality boxes inside the block (centred horizontally)
    bw, bh = 0.52, 0.38
    cx_mb = bk_cx    # x centre of mini-boxes
    for mod, y in zip(MODS, YM):
        rbox(ax, cx_mb, y, bw, bh, mod,
             fc=MF[mod], ec=MC[mod], fs=6.5, fw="bold", lw=1.8, zorder=5)

    # ALL interaction arrows on the RIGHT flank — clear of box text
    # zorder=3 so box fill (zorder=5) renders on top if edges graze a box
    xr_dn = cx_mb + bw/2 + 0.10   # adjacent downward column
    xr_up = cx_mb + bw/2 + 0.22   # adjacent upward column
    xr_dl = cx_mb + bw/2 + 0.34   # long-range downward (HE→CT, curved)
    xr_ul = cx_mb + bw/2 + 0.46   # long-range upward  (CT→HE, curved)

    # downward (→ below)
    for (yi, yj, c, xf, rad) in [
        (YM[0], YM[1], MC["HE"],  xr_dn, 0.0),
        (YM[1], YM[2], MC["BAL"], xr_dn, 0.0),
        (YM[0], YM[2], MC["HE"],  xr_dl, 0.25),  # curves right, avoids BAL box
    ]:
        ax.annotate("", xy=(xf, yj + bh/2 + 0.04), xytext=(xf, yi - bh/2 - 0.04),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1,
                                    mutation_scale=7,
                                    connectionstyle=f"arc3,rad={rad}"),
                    zorder=3, annotation_clip=False)

    # upward (← reverse = bidirectional)
    for (yi, yj, c, xf, rad) in [
        (YM[2], YM[1], MC["CT"],  xr_up, 0.0),
        (YM[1], YM[0], MC["BAL"], xr_up, 0.0),
        (YM[2], YM[0], MC["CT"],  xr_ul, -0.25),
    ]:
        ax.annotate("", xy=(xf, yj - bh/2 - 0.04), xytext=(xf, yi + bh/2 + 0.04),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1,
                                    mutation_scale=7,
                                    connectionstyle=f"arc3,rad={rad}"),
                    zorder=3, annotation_clip=False)

    ax.text(xr_dl + 0.08, 0.63, "all pairs  ·  bidir", ha="center",
            va="center", fontsize=5.8, color="#546E7A", style="italic", zorder=5)

    # ── fan-out from block right edge → K-slot boxes ─────────────────────
    bk_rx = bk_cx + bk_w / 2
    K, sq, gap = 4, 0.20, 0.05
    slot_cx, slot_bw, slot_bh = 6.35, 1.6, 0.62
    total_sq = K * sq + (K - 1) * gap

    ax.text(slot_cx, 3.75, "Slot Attn  (K slots)",
            ha="center", va="center", fontsize=7, fontweight="bold",
            color="#7B1FA2", zorder=7)

    for mod, y in zip(MODS, YM):
        arr(ax, bk_rx, y, slot_cx - slot_bw/2, y)
        rbox(ax, slot_cx, y, slot_bw, slot_bh, "", "slot", lw=1.5, zorder=3)
        x0_sq = slot_cx - total_sq/2 + sq/2
        for k in range(K):
            sx = x0_sq + k * (sq + gap)
            ax.add_patch(FancyBboxPatch(
                (sx - sq/2, y - sq/2 + 0.06), sq, sq,
                boxstyle="round,pad=0.02",
                facecolor=MC[mod], edgecolor="white", linewidth=0.7,
                alpha=0.75, zorder=4, clip_on=False))
        ax.text(slot_cx, y - slot_bh/2 + 0.11, mod,
                ha="center", va="center", fontsize=6.0,
                color=MC[mod], fontweight="bold", zorder=5)

    # ── bus merge → XFmr → Predict ───────────────────────────────────────
    highlight(ax, 7.3, 9.7, color="#E3F2FD")
    bus_merge(ax, [slot_cx + slot_bw/2] * 3, YM, tgt_lx=8.1, tgt_cy=2.0)
    rbox(ax, 8.85, 2.0, 1.3, 1.9, "Cross-Modal\nTransformer\n(all slots)", "xfmr",
         fw="bold", fs=7)
    arr(ax, 9.5, 2.0, 10.4, 2.0)
    rbox(ax, 11.1, 2.0, 1.2, 0.55, "ABMIL\n→ Predict", "out", fw="bold", fs=7)
    ax.text(3.7, 3.95, "Patch-level interaction (all pairs)", ha="center",
            fontsize=7, color="#E65100", style="italic")
    ax.text(8.85, 3.95, "Slot-level fusion", ha="center",
            fontsize=7, color="#1565C0", style="italic")


# ── Cross-modal slots ─────────────────────────────────────────────────────────

def draw_crossmodal(ax):
    _setup(ax, "Cross-Modal Slot Fusion\n(slot compression per modality → XFmr)")
    x0 = mod_inputs(ax)
    for y in YM:
        arr(ax, x0, y, 2.4, y)
        rbox(ax, 3.1, y, 1.2, 0.55, "Slot\nAttn (K)", "slot", fw="bold", fs=7)
        arr(ax, 3.7, y, 4.5, y)
        rbox(ax, 5.1, y, 1.1, 0.55, "K compact\nslots", "pool", fs=7)
    highlight(ax, 5.75, 8.5, color="#E3F2FD")
    # bus merge: compact-slot right edges → XFmr
    bus_merge(ax, [5.65] * 3, YM, tgt_lx=6.6, tgt_cy=2.0)
    rbox(ax, 7.3, 2.0, 1.4, 1.8,
         "Cross-Modal\nTransformer\n(all slots\ncross-attend)", "xfmr",
         fw="bold", fs=7)
    arr(ax, 8.0, 2.0, 9.1, 2.0)
    rbox(ax, 9.75, 2.0, 1.2, 0.55, "ABMIL\n→ Predict", "out", fw="bold", fs=7)
    ax.text(3.1, 3.75, "Compress to K slots", ha="center",
            fontsize=7, color="#7B1FA2", style="italic")
    ax.text(7.3, 3.8, "Fusion at slot level", ha="center",
            fontsize=7, color="#1565C0", style="italic")


# ── Iterative fusion ──────────────────────────────────────────────────────────

def draw_iterative(ax):
    _setup(ax, "Iterative Fusion\n(R × patch self+cross-attn → slot attn → XFmr)",
           xlim=(0, 14.5))
    x0 = mod_inputs(ax)
    highlight(ax, 2.25, 6.55, color="#FFF9C4")
    for y in YM:
        arr(ax, x0, y, 2.5, y)

    # ── R-iteration stacking effect: shadow copies drawn first ────────────
    bk_cx, bk_cy, bk_w, bk_h = 4.35, 2.0, 3.7, 3.5
    bk_lx = bk_cx - bk_w / 2
    for (ddx, ddy, ec_c, alpha_) in [
        (0.24, 0.22, "#BDBDBD", 0.55),  # furthest back (copy R-2)
        (0.12, 0.11, "#C8B742", 0.75),  # middle (copy R-1)
    ]:
        ax.add_patch(FancyBboxPatch(
            (bk_lx + ddx, bk_cy - bk_h/2 - ddy),
            bk_w, bk_h,
            boxstyle="round,pad=0.10",
            facecolor="#FFF9C4", edgecolor=ec_c, linewidth=1.5,
            alpha=alpha_, zorder=1, clip_on=False))

    # main block (front)
    rbox(ax, bk_cx, bk_cy, bk_w, bk_h, "", "iter", lw=2.2, zorder=3)

    # "× R" corner label
    ax.text(bk_lx + 0.12, bk_cy + bk_h/2 - 0.22, "× R",
            ha="left", va="center", fontsize=11, fontweight="bold",
            color="#E65100", zorder=7)

    # ── layout parameters for the two sub-sections ───────────────────────
    bw, bh = 0.46, 0.38
    cx_sa = 3.08    # self-attn mini-boxes centre-x
    cx_ca = 5.15    # cross-attn mini-boxes centre-x

    # section headers
    for cx, lbl in [(cx_sa, "Self-Attn\n(within mod)"),
                    (cx_ca, "Cross-Attn\n(across mods)")]:
        ax.text(cx, 3.30, lbl, ha="center", va="center", fontsize=6.5,
                fontweight="bold", color="#37474F", zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#BDBDBD", lw=0.8, alpha=0.95))

    # dashed divider
    ax.plot([3.68, 3.68], [0.52, 3.08], '--', color="#BDBDBD", lw=0.9, zorder=4)

    # ── SELF-ATTN: mini boxes + self-loop arcs ────────────────────────────
    for mod, y in zip(MODS, YM):
        rbox(ax, cx_sa, y, bw, bh, mod,
             fc=MF[mod], ec=MC[mod], fs=6.5, fw="bold", lw=1.8, zorder=5)
        # self-loop arc (above each box)
        ax.annotate("",
                    xy=(cx_sa + bw * 0.28, y + bh / 2),
                    xytext=(cx_sa - bw * 0.28, y + bh / 2),
                    arrowprops=dict(arrowstyle="-|>", color=MC[mod], lw=1.2,
                                    mutation_scale=7,
                                    connectionstyle="arc3,rad=-0.65"),
                    zorder=6, annotation_clip=False)

    # ── horizontal arrows SA → CA ─────────────────────────────────────────
    for y in YM:
        ax.annotate("", xy=(cx_ca - bw/2, y), xytext=(cx_sa + bw/2, y),
                    arrowprops=dict(arrowstyle="-|>", color="#90A4AE", lw=0.9,
                                    mutation_scale=7),
                    zorder=4, annotation_clip=False)

    # ── CROSS-ATTN: mini boxes + all-pairs bidirectional arrows ──────────
    for mod, y in zip(MODS, YM):
        rbox(ax, cx_ca, y, bw, bh, mod,
             fc=MF[mod], ec=MC[mod], fs=6.5, fw="bold", lw=1.8, zorder=5)

    # ALL cross-attn arrows on RIGHT flank — away from the SA→CA connectors
    # zorder=3 so boxes (zorder=5) render on top; self-loops kept at zorder=5+
    xr_dn = cx_ca + bw/2 + 0.10   # adjacent downward
    xr_up = cx_ca + bw/2 + 0.22   # adjacent upward
    xr_dl = cx_ca + bw/2 + 0.34   # long-range downward (HE→CT)
    xr_ul = cx_ca + bw/2 + 0.46   # long-range upward  (CT→HE)

    for (yi, yj, c, xf, rad) in [
        (YM[0], YM[1], MC["HE"],  xr_dn, 0.0),
        (YM[1], YM[2], MC["BAL"], xr_dn, 0.0),
        (YM[0], YM[2], MC["HE"],  xr_dl, 0.25),
    ]:
        ax.annotate("", xy=(xf, yj + bh/2 + 0.04), xytext=(xf, yi - bh/2 - 0.04),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1,
                                    mutation_scale=7,
                                    connectionstyle=f"arc3,rad={rad}"),
                    zorder=3, annotation_clip=False)

    for (yi, yj, c, xf, rad) in [
        (YM[2], YM[1], MC["CT"],  xr_up, 0.0),
        (YM[1], YM[0], MC["BAL"], xr_up, 0.0),
        (YM[2], YM[0], MC["CT"],  xr_ul, -0.25),
    ]:
        ax.annotate("", xy=(xf, yj - bh/2 - 0.04), xytext=(xf, yi + bh/2 + 0.04),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1,
                                    mutation_scale=7,
                                    connectionstyle=f"arc3,rad={rad}"),
                    zorder=3, annotation_clip=False)

    ax.text(xr_dl + 0.08, 0.63, "Q=own  ·  KV=all others",
            ha="center", va="center", fontsize=5.8,
            color="#546E7A", style="italic", zorder=5)

    # ── K-SLOT VISUALIZATION ─────────────────────────────────────────────
    # Each modality row gets a slot-attn box containing K explicit slots
    K        = 5       # number of visible slots
    slot_cx  = 8.1     # centre-x of all slot boxes
    slot_bw  = 1.75    # slot box width
    slot_bh  = 0.65    # slot box height
    sq       = 0.21    # slot square size
    gap      = 0.05    # gap between slot squares
    total_sq = K * sq + (K - 1) * gap  # total width of K squares

    ax.text(slot_cx, 3.75, "Slot Attention  (K slots per modality)",
            ha="center", va="center", fontsize=7.5, fontweight="bold",
            color="#7B1FA2", zorder=7)

    # fan-out arrows from block right edge to slot boxes (horizontal)
    bk_rx = bk_cx + bk_w / 2
    for y in YM:
        arr(ax, bk_rx, y, slot_cx - slot_bw/2, y)

    for mod, y in zip(MODS, YM):
        # outer slot attn box
        rbox(ax, slot_cx, y, slot_bw, slot_bh, "", "slot", lw=1.5, zorder=3)
        # K coloured slot squares inside the box
        x0_sq = slot_cx - total_sq/2 + sq/2
        for k in range(K):
            sx = x0_sq + k * (sq + gap)
            ax.add_patch(FancyBboxPatch(
                (sx - sq/2, y - sq/2 + 0.06),
                sq, sq,
                boxstyle="round,pad=0.02",
                facecolor=MC[mod], edgecolor="white", linewidth=0.7,
                alpha=0.75, zorder=4, clip_on=False))
        # modality label inside, to the right of slots
        ax.text(slot_cx, y - slot_bh/2 + 0.12, mod,
                ha="center", va="center", fontsize=6.0,
                color=MC[mod], fontweight="bold", zorder=5)

    # ── bus merge → XFmr → Predict ───────────────────────────────────────
    highlight(ax, 9.1, 12.1, color="#E3F2FD")
    bus_merge(ax, [slot_cx + slot_bw/2] * 3, YM, tgt_lx=9.65, tgt_cy=2.0)
    rbox(ax, 10.5, 2.0, 1.5, 1.9, "Cross-Modal\nTransformer\n→ ABMIL", "xfmr",
         fw="bold", fs=7)
    arr(ax, 11.25, 2.0, 12.2, 2.0)
    rbox(ax, 12.85, 2.0, 1.1, 0.6, "Predict", "out", fw="bold")
    ax.text(4.35, 3.97, "Iterative patch enrichment (shared weights across R rounds)",
            ha="center", fontsize=7, color="#E65100", style="italic")


# ── Overview 2×3 panel ────────────────────────────────────────────────────────

def make_overview():
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(hspace=0.55, wspace=0.18,
                        left=0.01, right=0.99, top=0.90, bottom=0.04)

    panels = [
        (axes[0, 0], draw_early,      "1 — Early Fusion"),
        (axes[0, 1], draw_late,       "2 — Late Fusion"),
        (axes[0, 2], draw_middle,     "3 — Middle Fusion"),
        (axes[1, 0], draw_crossattn,  "4 — Cross-Attention"),
        (axes[1, 1], draw_crossmodal, "5 — Cross-Modal Slots"),
        (axes[1, 2], draw_iterative,  "6 — Iterative Fusion"),
    ]
    for ax, fn, _ in panels:
        fn(ax)

    legend_handles = [
        mpatches.Patch(fc=MF["HE"],       ec=MC["HE"],        label="H&E patches"),
        mpatches.Patch(fc=MF["BAL"],      ec=MC["BAL"],       label="BAL patches"),
        mpatches.Patch(fc=MF["CT"],       ec=MC["CT"],        label="CT patches"),
        mpatches.Patch(fc=OP["abmil"][0], ec=OP["abmil"][1],  label="ABMIL pool"),
        mpatches.Patch(fc=OP["xfmr"][0],  ec=OP["xfmr"][1],  label="Cross-Modal XFmr"),
        mpatches.Patch(fc=OP["xattn"][0], ec=OP["xattn"][1], label="Patch cross-attn"),
        mpatches.Patch(fc=OP["slot"][0],  ec=OP["slot"][1],  label="Slot attention"),
        mpatches.Patch(fc=OP["iter"][0],  ec=OP["iter"][1],  label="Iterative block"),
        mpatches.Patch(fc=OP["out"][0],   ec=OP["out"][1],   label="Prediction head"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=9,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("MM-ABMIL v6 — Multimodal Fusion Mechanism Overview",
                 fontsize=15, fontweight="bold", y=1.00)

    out = OUT_DIR / "overview_all_methods.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── Individual high-res figures ───────────────────────────────────────────────

def make_individual():
    specs = [
        ("phase_overview",    draw_phase_overview,  (12, 4.5)),
        ("early_fusion",      draw_early,            (10, 4.5)),
        ("late_fusion",       draw_late,             (10, 4.5)),
        ("middle_fusion",     draw_middle,           (10, 4.5)),
        ("crossattn_fusion",  draw_crossattn,        (12, 4.5)),
        ("crossmodal_fusion", draw_crossmodal,       (10, 4.5)),
        ("iterative_fusion",  draw_iterative,        (12, 4.5)),
    ]
    for fname, fn, fsz in specs:
        fig, ax = plt.subplots(figsize=fsz)
        fn(ax)
        fig.savefig(OUT_DIR / f"{fname}.png", dpi=250, bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{fname}.svg", format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}.png + .svg")


if __name__ == "__main__":
    print(f"\nGenerating mechanism diagrams → {OUT_DIR}\n")
    make_overview()
    make_individual()
    print("\nDone.")
