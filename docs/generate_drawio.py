#!/usr/bin/env python3
"""
Generates a draw.io XML file with all MM-ABMIL v6 fusion diagrams.
Each method is one page — fully editable (drag boxes, reroute arrows, edit text).

Open at:  https://app.diagrams.net  (File → Open from Device)
or:       draw.io desktop app

Run:  python generate_drawio.py
Out:  analysis_v6_milv2/mechanism_diagrams/fusion_mechanisms.drawio
"""
from pathlib import Path
import textwrap, html

OUT_DIR  = Path("/home/aih/dinesh.haridoss/chicago_mil/analysis_v6_milv2/mechanism_diagrams")
OUT_FILE = OUT_DIR / "fusion_mechanisms.drawio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── coordinate helpers ────────────────────────────────────────────────────────
SX = 85      # pixels per matplotlib x-unit
SY = 95      # pixels per matplotlib y-unit (slightly taller)
YMAX = 4.1   # matplotlib y maximum (top of canvas)
PAD = 20     # canvas padding in pixels

def px(x):          return round(PAD + x * SX, 1)
def py(y, h=0):     return round(PAD + (YMAX - y - h) * SY, 1)
def pw(w):          return round(w * SX, 1)
def ph(h):          return round(h * SY, 1)

# ── colour palette ────────────────────────────────────────────────────────────
# draw.io fillColor / strokeColor
C = {
    "HE":       ("#E8F5E9", "#2E7D32"),
    "BAL":      ("#E3F2FD", "#1565C0"),
    "CT":       ("#FBE9E7", "#BF360C"),
    "abmil":    ("#E8F5E9", "#388E3C"),
    "xfmr":     ("#E3F2FD", "#1565C0"),
    "slot":     ("#F3E5F5", "#7B1FA2"),
    "out":      ("#FFEBEE", "#C62828"),
    "merge":    ("#E8EAF6", "#3949AB"),
    "iter":     ("#FFF9C4", "#F9A825"),
    "xattn":    ("#FFF3E0", "#E65100"),
    "pool":     ("#ECEFF1", "#546E7A"),
    "concat":   ("#F5F5F5", "#757575"),
}

def _style(key, bold=False, fontsize=10, extra=""):
    fc, sc = C.get(key, ("#FFFFFF", "#000000"))
    fw = "fontStyle=1;" if bold else ""
    return (f"rounded=1;whiteSpace=wrap;html=1;"
            f"fillColor={fc};strokeColor={sc};"
            f"fontSize={fontsize};{fw}{extra}")

EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
        "jettySize=auto;html=1;exitX=1;exitY=0.5;exitDx=0;exitDy=0;"
        "entryX=0;entryY=0.5;entryDx=0;entryDy=0;")
EDGE_OPEN = ("edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
             "jettySize=auto;html=1;endArrow=open;endFill=0;")
LABEL_STYLE = ("text;html=1;strokeColor=none;fillColor=none;"
               "align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;")

# ── page builder ──────────────────────────────────────────────────────────────

class Page:
    """One page (tab) in a draw.io file."""

    def __init__(self, name):
        self.name = name
        self.cells = []
        self._id = 9

    def _uid(self):
        self._id += 1
        return str(self._id)

    def box(self, cx, cy, w, h, label, style_key="pool",
            bold=False, fontsize=10, extra=""):
        """Vertex rectangle. Returns cell id."""
        uid = self._uid()
        style = _style(style_key, bold=bold, fontsize=fontsize, extra=extra)
        lbl = html.escape(label).replace("\n", "&lt;br/&gt;")
        self.cells.append(
            f'<mxCell id="{uid}" value="{lbl}" style="{style}" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="{px(cx - w/2)}" y="{py(cy + h/2)}" '
            f'width="{pw(w)}" height="{ph(h)}" as="geometry" /></mxCell>')
        return uid

    def mod_box(self, mod, cx, cy, w=1.05, h=0.55):
        """Coloured modality patch-bag box."""
        fc, sc = C[mod]
        style = (f"rounded=1;whiteSpace=wrap;html=1;"
                 f"fillColor={fc};strokeColor={sc};fontColor={sc};"
                 f"fontStyle=1;fontSize=10;strokeWidth=2;")
        uid = self._uid()
        lbl = f"{mod}&lt;br/&gt;patches"
        self.cells.append(
            f'<mxCell id="{uid}" value="{lbl}" style="{style}" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="{px(cx - w/2)}" y="{py(cy + h/2)}" '
            f'width="{pw(w)}" height="{ph(h)}" as="geometry" /></mxCell>')
        return uid

    def mini_mod_box(self, mod, cx, cy, w=0.46, h=0.36):
        """Small coloured modality box for inside a block."""
        fc, sc = C[mod]
        style = (f"rounded=1;whiteSpace=wrap;html=1;"
                 f"fillColor={fc};strokeColor={sc};fontColor={sc};"
                 f"fontStyle=1;fontSize=9;strokeWidth=2;")
        uid = self._uid()
        self.cells.append(
            f'<mxCell id="{uid}" value="{mod}" style="{style}" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="{px(cx - w/2)}" y="{py(cy + h/2)}" '
            f'width="{pw(w)}" height="{ph(h)}" as="geometry" /></mxCell>')
        return uid

    def region(self, x0, x1, y0, y1, color="#EDE7F6", opacity=30):
        """Background shaded region."""
        uid = self._uid()
        style = (f"rounded=1;whiteSpace=wrap;html=1;"
                 f"fillColor={color};strokeColor=none;opacity={opacity};")
        w, h = x1 - x0, y1 - y0
        self.cells.append(
            f'<mxCell id="{uid}" value="" style="{style}" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="{px(x0)}" y="{py(y1)}" '
            f'width="{pw(w)}" height="{ph(h)}" as="geometry" /></mxCell>')

    def label(self, cx, cy, text, fontsize=8, color="#546E7A",
              bold=False, italic=False, w=2.5, h=0.35):
        """Free text label."""
        uid = self._uid()
        fw = ("1" if bold else "0")
        style = (f"{LABEL_STYLE}fontSize={fontsize};fontColor={color};"
                 f"fontStyle={fw};")
        lbl = html.escape(text).replace("\n", "&lt;br/&gt;")
        self.cells.append(
            f'<mxCell id="{uid}" value="{lbl}" style="{style}" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="{px(cx - w/2)}" y="{py(cy + h/2)}" '
            f'width="{pw(w)}" height="{ph(h)}" as="geometry" /></mxCell>')

    def edge(self, src, tgt, color="#37474F", label="", dashed=False):
        """Directed edge src → tgt (orthogonal routing)."""
        uid = self._uid()
        dash = "dashed=1;dashPattern=8 4;" if dashed else ""
        style = (f"{EDGE}strokeColor={color};{dash}")
        self.cells.append(
            f'<mxCell id="{uid}" value="{html.escape(label)}" '
            f'style="{style}" edge="1" source="{src}" target="{tgt}" '
            f'parent="1"><mxGeometry relative="1" as="geometry" /></mxCell>')
        return uid

    def bidir_edge(self, src, tgt, color="#37474F"):
        """Bidirectional arrow between two cells."""
        uid = self._uid()
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
                 f"jettySize=auto;html=1;startArrow=block;startFill=1;"
                 f"endArrow=block;endFill=1;strokeColor={color};")
        self.cells.append(
            f'<mxCell id="{uid}" value="" style="{style}" '
            f'edge="1" source="{src}" target="{tgt}" parent="1">'
            f'<mxGeometry relative="1" as="geometry" /></mxCell>')

    def self_loop(self, src, color="#37474F"):
        """Self-loop arrow (self-attention)."""
        uid = self._uid()
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
                 f"jettySize=auto;html=1;exitX=1;exitY=0;exitDx=0;exitDy=0;"
                 f"entryX=1;entryY=1;entryDx=0;entryDy=0;curved=1;"
                 f"strokeColor={color};endArrow=block;endFill=1;")
        self.cells.append(
            f'<mxCell id="{uid}" value="" style="{style}" '
            f'edge="1" source="{src}" target="{src}" parent="1">'
            f'<mxGeometry relative="1" as="geometry" /></mxCell>')

    def to_xml(self):
        cells = "\n      ".join(self.cells)
        return textwrap.dedent(f"""\
        <diagram name="{html.escape(self.name)}">
          <mxGraphModel grid="1" gridSize="10" guides="1" tooltips="1"
              connect="1" arrows="1" fold="1" page="0"
              pageScale="1" pageWidth="1654" pageHeight="1169"
              math="0" shadow="0">
            <root>
              <mxCell id="0" />
              <mxCell id="1" parent="0" />
              {cells}
            </root>
          </mxGraphModel>
        </diagram>""")


# ── modality helpers ──────────────────────────────────────────────────────────
MODS = ["HE", "BAL", "CT"]
YM   = [3.0, 2.0, 1.0]
MC   = {"HE": "#2E7D32", "BAL": "#1565C0", "CT": "#BF360C"}


def _mod_inputs(p, x=0.7):
    """Draw 3 modality boxes; return {mod: id} and right-edge x."""
    ids = {}
    for mod, y in zip(MODS, YM):
        ids[mod] = p.mod_box(mod, x, y)
    return ids, x + 0.525   # right edge


# ── per-method diagram builders ───────────────────────────────────────────────

def page_phase_overview():
    p = Page("Phase Overview")
    p.region(0.3, 5.0, 0.3, 3.7, "#E8F5E9", 25)
    p.label(2.65, 3.85, "Phase 1 — Single-Modal", fontsize=9,
            color="#2E7D32", bold=True)
    p.region(6.9, 11.7, 0.3, 3.7, "#E3F2FD", 25)
    p.label(9.3, 3.85, "Phase 2 — Multimodal Fusion", fontsize=9,
            color="#1565C0", bold=True)

    prev = {}
    for mod, y in zip(MODS, YM):
        bid  = p.mod_box(mod, 1.2, y)
        abm  = p.box(3.1, y, 1.3, 0.55, "SingleModal\nABMIL", "abmil")
        logit= p.box(4.8, y, 0.9, 0.55, "P1 logit", "out", fontsize=9)
        p.edge(bid, abm); p.edge(abm, logit)
        p.label(6.0, y, "→  freeze  →", fontsize=8, color="#546E7A", w=1.8)
        enc  = p.box(7.6, y, 1.0, 0.45, f"P1 enc\n({mod})", "pool",
                     fontsize=9, extra=f"strokeColor={MC[mod]};fillColor={C[mod][0]};")
        prev[mod] = enc

    fusion = p.box(9.4, 2.0, 1.5, 2.5, "Fusion Module\n(6 variants)", "xfmr",
                   bold=True, fontsize=11)
    p2     = p.box(11.35, 2.0, 1.0, 0.55, "P2 logit", "out", bold=True)
    for enc in prev.values():
        p.edge(enc, fusion)
    p.edge(fusion, p2)
    return p


def page_early():
    p = Page("Early Fusion")
    p.region(2.25, 5.0, 0.4, 3.6, "#E8EAF6", 30)
    p.label(4.0, 3.78, "Fusion at patch level", color="#3949AB")
    mids, x0 = _mod_inputs(p)
    concat = p.box(3.0, 2.0, 1.0, 3.0, "Concat\nall patches", "concat")
    abmil  = p.box(5.3, 2.0, 1.3, 0.6,  "Shared ABMIL", "abmil", bold=True)
    pred   = p.box(7.65, 2.0, 1.1, 0.6, "Predict", "out",  bold=True)
    for uid in mids.values():
        p.edge(uid, concat)
    p.edge(concat, abmil); p.edge(abmil, pred)
    return p


def page_late():
    p = Page("Late Fusion")
    p.region(6.1, 8.4, 0.4, 3.6, "#EDE7F6", 30)
    p.label(7.4, 3.78, "Fusion at decision level", color="#3949AB")
    mids, _ = _mod_inputs(p)
    avg  = p.box(7.4, 2.0, 1.3, 0.65, "Learned\nWeighted Avg", "merge", bold=True)
    pred = p.box(9.75, 2.0, 1.1, 0.6,  "Predict", "out", bold=True)
    for mod, y in zip(MODS, YM):
        abm   = p.box(3.1, y, 1.2, 0.55, "ABMIL pool", "abmil", bold=True)
        logit = p.box(5.2, y, 1.1, 0.55, "Per-mod logit", "pool")
        p.edge(mids[mod], abm); p.edge(abm, logit); p.edge(logit, avg)
    p.edge(avg, pred)
    return p


def page_middle():
    p = Page("Middle Fusion")
    p.region(5.65, 8.4, 0.4, 3.6, "#E3F2FD", 30)
    p.label(7.2, 3.82, "Fusion at summary level", color="#1565C0")
    mids, _ = _mod_inputs(p)
    xfmr = p.box(7.2, 2.0, 1.4, 1.9,
                 "Cross-Modal\nTransformer\n(self-attn\nover tokens)", "xfmr", bold=True)
    pred = p.box(9.75, 2.0, 1.1, 0.6, "Predict", "out", bold=True)
    for mod, y in zip(MODS, YM):
        abm  = p.box(3.1, y, 1.2, 0.55, "ABMIL pool", "abmil", bold=True)
        summ = p.box(5.0, y, 1.0, 0.55, "Summary token", "pool")
        p.edge(mids[mod], abm); p.edge(abm, summ); p.edge(summ, xfmr)
    p.edge(xfmr, pred)
    return p


def page_crossattn():
    p = Page("Cross-Attention Fusion")
    p.region(2.25, 5.3, 0.4, 3.6, "#FFF3E0", 40)
    p.label(3.7, 3.92, "Patch-level interaction (all pairs)", color="#E65100")
    p.region(7.3, 9.7, 0.4, 3.6, "#E3F2FD", 30)
    p.label(8.85, 3.92, "Slot-level fusion", color="#1565C0")

    mids, _ = _mod_inputs(p)

    # inside bidir block: mini modality boxes + bidir + self-attn
    bk = p.box(3.7, 2.0, 2.3, 3.4,
               "Bidir Patch\nCross-Attention", "xattn", bold=True, fontsize=11)
    mini = {}
    for mod, y in zip(MODS, YM):
        mini[mod] = p.mini_mod_box(mod, 3.7, y)
        p.edge(mids[mod], mini[mod])

    # bidirectional arrows inside the block
    for i, (ma, mb) in enumerate([("HE","BAL"),("BAL","CT"),("HE","CT")]):
        p.bidir_edge(mini[ma], mini[mb], color=MC[ma])

    p.label(3.7, 0.62, "all pairs  ·  shared weights",
            fontsize=7, color="#546E7A")

    # slot attn with K slots label
    p.label(6.35, 3.78, "Slot Attn  (K slots)", color="#7B1FA2", bold=True)
    slot_ids = {}
    xfmr = p.box(8.85, 2.0, 1.3, 1.9, "Cross-Modal\nTransformer\n(all slots)", "xfmr", bold=True)
    pred = p.box(11.1, 2.0, 1.2, 0.55, "ABMIL → Predict", "out", bold=True)
    for mod, y in zip(MODS, YM):
        sid = p.box(6.35, y, 1.6, 0.62,
                    f"K slots  [{mod}]", "slot", fontsize=9,
                    extra=f"strokeColor={MC[mod]};")
        slot_ids[mod] = sid
        p.edge(mini[mod], sid)
        p.edge(sid, xfmr)
    p.edge(xfmr, pred)
    return p


def page_crossmodal():
    p = Page("Cross-Modal Slot Fusion")
    p.region(5.75, 8.5, 0.4, 3.6, "#E3F2FD", 30)
    p.label(3.1, 3.78, "Compress to K slots", color="#7B1FA2")
    p.label(7.3, 3.78, "Fusion at slot level", color="#1565C0")
    mids, _ = _mod_inputs(p)
    xfmr = p.box(7.3, 2.0, 1.4, 1.9,
                 "Cross-Modal\nTransformer\n(all slots\ncross-attend)", "xfmr", bold=True)
    pred = p.box(9.75, 2.0, 1.2, 0.55, "ABMIL → Predict", "out", bold=True)
    for mod, y in zip(MODS, YM):
        slot = p.box(3.1, y, 1.2, 0.55, "Slot Attn (K)", "slot", bold=True)
        comp = p.box(5.1, y, 1.1, 0.55, f"K slots [{mod}]", "pool",
                     fontsize=9, extra=f"strokeColor={MC[mod]};")
        p.edge(mids[mod], slot); p.edge(slot, comp); p.edge(comp, xfmr)
    p.edge(xfmr, pred)
    return p


def page_iterative():
    p = Page("Iterative Fusion")
    p.region(2.25, 6.55, 0.4, 3.6, "#FFF9C4", 40)
    p.label(4.35, 3.97, "Iterative patch enrichment (shared weights, × R rounds)",
            color="#E65100", w=4.5)
    p.region(9.1, 12.1, 0.4, 3.6, "#E3F2FD", 30)
    p.label(8.1, 3.78, "Slot Attention  (K slots per modality)",
            color="#7B1FA2", bold=True, w=3.5)

    mids, _ = _mod_inputs(p)

    # SELF-ATTN section
    sa_label = p.label(3.08, 3.30, "Self-Attn\n(within mod)",
                       fontsize=8, color="#37474F", bold=True)
    sa = {}
    for mod, y in zip(MODS, YM):
        sa[mod] = p.mini_mod_box(mod, 3.08, y)
        p.edge(mids[mod], sa[mod])
        p.self_loop(sa[mod], color=MC[mod])

    # CROSS-ATTN section
    p.label(5.15, 3.30, "Cross-Attn\n(across mods)",
            fontsize=8, color="#37474F", bold=True)
    ca = {}
    for mod, y in zip(MODS, YM):
        ca[mod] = p.mini_mod_box(mod, 5.15, y)
        p.edge(sa[mod], ca[mod])

    # bidirectional arrows all-pairs
    for ma, mb in [("HE","BAL"), ("BAL","CT"), ("HE","CT")]:
        p.bidir_edge(ca[ma], ca[mb], color=MC[ma])

    p.label(5.7, 0.62, "Q=own  ·  KV=all others",
            fontsize=7, color="#546E7A", w=2.0)

    # K-slot boxes
    xfmr = p.box(10.5, 2.0, 1.5, 1.9,
                 "Cross-Modal\nTransformer\n→ ABMIL", "xfmr", bold=True)
    pred = p.box(12.85, 2.0, 1.1, 0.6, "Predict", "out", bold=True)
    for mod, y in zip(MODS, YM):
        sid = p.box(8.1, y, 1.75, 0.62,
                    f"K slots  [{mod}]", "slot", fontsize=9,
                    extra=f"strokeColor={MC[mod]};")
        p.edge(ca[mod], sid)
        p.edge(sid, xfmr)
    p.edge(xfmr, pred)
    return p


# ── assemble and write ────────────────────────────────────────────────────────

def build():
    pages = [
        page_phase_overview(),
        page_early(),
        page_late(),
        page_middle(),
        page_crossattn(),
        page_crossmodal(),
        page_iterative(),
    ]
    diagrams = "\n".join(p.to_xml() for p in pages)
    xml = f'<mxfile host="app.diagrams.net" version="21.0.0">\n{diagrams}\n</mxfile>'
    OUT_FILE.write_text(xml, encoding="utf-8")
    print(f"  Saved: {OUT_FILE.name}  ({len(pages)} pages)")
    print(f"\n  Open at:  https://app.diagrams.net  (File → Open from Device)")
    print(f"  Or:       draw.io desktop app\n")
    print(f"  In draw.io: each tab = one fusion method.")
    print(f"  All boxes and arrows are independently draggable.")
    print(f"  Export to SVG/PNG/PDF via File → Export.\n")


if __name__ == "__main__":
    print(f"\nGenerating draw.io diagram → {OUT_FILE}\n")
    build()
