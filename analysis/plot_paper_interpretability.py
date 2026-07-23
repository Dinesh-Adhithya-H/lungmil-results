"""
Publication-quality interpretability figures for SetMIL-MT.

Reads paper_interp_data.json from all 5 splits (saved by interpret_set_mil_mt.py)
and produces 4 paper figures:

  paper_fig1_BAL_celltypes.png       — BAL cell type enrichment heatmap (tasks × cell types)
  paper_fig2_clinical_features.png   — Top clinical feature importance per task
  paper_fig3_HE_CT_morphology.png    — HE tissue types & CT cluster enrichment
  paper_fig4_gate_weights.png        — Modality gate weights per task

Usage (sbatch only):
  sbatch analysis/submit_paper_interpretability.sh
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import wandb

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "mm_abmil_v8" / "phase2"
INTERP_DIR  = ROOT / "interpretability" / "set_mil_mt_interp"
OUT_DIR     = ROOT / "interpretability" / "paper_figs"

TASKS    = ["acr_cls", "acr_surv", "clad_surv", "death_surv"]
TASK_LABELS = {
    "acr_cls":    "ACR\nclassif.",
    "acr_surv":   "ACR\nsurvival",
    "clad_surv":  "CLAD\nsurvival",
    "death_surv": "Death\nsurvival",
}
TASK_COLORS = {
    "acr_cls":    "#e53935",
    "acr_surv":   "#ff7043",
    "clad_surv":  "#7e57c2",
    "death_surv": "#26a69a",
}
MOD_COLORS = {"HE": "#58a6ff", "BAL": "#3fb950", "CT": "#d4a017", "Clinical": "#d2a8ff"}

# Group label colors for BAL and Clinical (for colored tick labels)
BAL_GROUP_COLORS = {
    "TRAM":         "#E67E22",
    "MoAM":         "#16A085",
    "CXCL10 macs":  "#8E44AD",
    "T cells":      "#C0392B",
    "DCs / NK / B": "#2980B9",
    "Epithelial":   "#27AE60",
}
CLINICAL_GROUP_COLORS = {
    "PFT":       "#1565C0",
    "Chemistry": "#E65100",
    "CBC":       "#2E7D32",
    "BAL diff":  "#00838F",
    "Vitals":    "#6A1B9A",
    "Transplant":"#AD1457",
    "Other":     "#555555",
}

# BAL cell type groups for display ordering
BAL_GROUPS = {
    "TRAM":        ["TRAM-1","TRAM-2","TRAM-3","TRAM-4","TRAM-5","TRAM-6","TRAM-7","TRAM-8","TRAM-9"],
    "MoAM":        ["MoAM-1","MoAM-2","MoAM-3","MoAM-4","Profibrotic MoAM","Proliferating macrophages",
                    "Perivascular macrophages","MT macs","Monocytes"],
    "CXCL10 macs": ["CXCL10 macs-1","CXCL10 macs-2"],
    "T cells":     ["CD4 T cell-1","CD4 T cells-2","CD4 T cells-3","CD4 T naive cells","CD4 Treg",
                    "CD8 T cell-1","CD8 T cells-2","CD8 T cells-3","CD8 T cells-4","CD8 T cells-5",
                    "Proliferating T cells","gdT"],
    "DCs / NK / B":["CCR7+ DC1","DC1","DC2","pDCs","NK","B cells","Plasmablasts/Plasma cells"],
    "Epithelial":  ["AT1/2","Basaloid AT1s","Ciliated epithelial cell","Goblet cell"],
}

CLINICAL_GROUPS = {
    "PFT":        ["fvc","fev1","fev1_fvc","fev1p","fvcp","delta_fvc_from_previous",
                   "pseudoslope_fvc","delta_fev1_from_previous","pseudoslope_fev1"],
    "Chemistry":  ["ALBUMIN","TOTAL PROTEIN","SODIUM","GLUCOSE","CREATININE","CO2","CHLORIDE",
                   "CALCIUM","BLOOD UREA NITROGEN","ALKALINE PHOSPHATASE","ALT","POTASSIUM",
                   "AST","GFR","TOTAL BILIRUBIN","ANION GAP"],
    "CBC":        ["ABSOLUTE LYMPHOCYTES","ABSOLUTE NEUTROPHILS","ABSOLUTE MONOCYTES",
                   "ABSOLUTE EOSINOPHILS","ABSOLUTE BASOPHILS","WHITE BLOOD CELLS",
                   "HEMOGLOBIN","PLATELET COUNT","RED BLOOD CELL COUNT","MCV","MCH","MCHC",
                   "RDW","MPV","NEUTROPHILS %","LYMPHOCYTES %","MONOCYTES %","EOSINOPHILS %",
                   "BASOPHILS %"],
    "BAL diff":   ["EOSINOPHILS, BODY FLUID Left","EOSINOPHILS, BODY FLUID Right",
                   "LYMPHOCYTES, BODY FLUID Left","LYMPHOCYTES, BODY FLUID Right",
                   "MACROPHAGES, BODY FLUID Left","MACROPHAGES, BODY FLUID Right",
                   "MONOCYTES, BODY FLUID Left","MONOCYTES, BODY FLUID Right",
                   "NEUTROPHILS, BODY FLUID Left","NEUTROPHILS, BODY FLUID Right"],
    "Vitals":     ["BMI","BP DIASTOLIC","BP SYSTOLIC","PULSE","RESPIRATIONS","SP02","TEMPERATURE"],
    "Transplant": ["positive_dsa","dsa_antibody_number","donor_risk","cmv_donor","recipient_cmv",
                   "donor_ebv","recipient_ebv","age","age_donor","bmi_donor",
                   "pgd_t0","pgd_t24","pgd_t48","pgd_t72","prev_tx"],
}


def _load_all_splits(variant="all_splits"):
    """
    Load paper_interp_data.json from interpretability output dir.
    Tries all_splits dir first, then aggregates per-split fold0 dirs.
    Returns list of (split_tag, data_dict).
    """
    records = []

    # Try all_splits output first
    p = INTERP_DIR / f"all_splits_{variant}" / "paper_interp_data.json"
    if p.exists():
        data = json.loads(p.read_text())
        records.append(("all_splits", data))
        print(f"Loaded all_splits JSON: {p}")
        return records

    # Fall back to per-task per-split dirs
    for task in TASKS:
        task_name = {"acr_cls": "cls", "acr_surv": "acr_surv",
                     "clad_surv": "clad_surv", "death_surv": "death_surv"}.get(task, task)
        p = INTERP_DIR / f"all_splits_{task_name}" / "paper_interp_data.json"
        if p.exists():
            data = json.loads(p.read_text())
            records.append((task_name, data))
            print(f"Loaded {task_name} JSON: {p}")

    if not records:
        raise FileNotFoundError(
            f"No paper_interp_data.json found in {INTERP_DIR}. "
            "Run interpret_set_mil_mt.py --all-splits first."
        )
    return records


def _merge_records(records):
    """
    Merge multiple JSON records into a single unified structure.
    If one record covers all tasks, use it directly.
    Otherwise merge task-by-task.
    """
    if len(records) == 1:
        return records[0][1]

    merged = {"modalities": [], "n_patients": 0, "tasks": {}}
    for _, data in records:
        merged["modalities"] = data.get("modalities", [])
        for task, tdata in data.get("tasks", {}).items():
            if task not in merged["tasks"]:
                merged["tasks"][task] = tdata
    return merged


HE_CLUSTER_MAP_PATH = ROOT / "results" / "cluster_name_maps" / "HE_cluster_map.json"

def _load_he_bio_map():
    """Load HE cluster map: raw_id → biological category name."""
    try:
        return json.loads(HE_CLUSTER_MAP_PATH.read_text())
    except Exception:
        return {}

def _collapse_he_clusters(raw_names, delta_vec):
    """
    Map fine-grained HE cluster IDs (e.g. '5_1', '5_2'...) to biological categories,
    summing delta values within each category.
    Returns (bio_names, bio_delta).
    """
    bio_map = _load_he_bio_map()
    if not bio_map:
        return raw_names, delta_vec

    from collections import defaultdict, OrderedDict
    cat_delta = defaultdict(float)
    cat_order = OrderedDict()
    for nm, dv in zip(raw_names, delta_vec):
        cat = bio_map.get(str(nm), str(nm))
        cat_delta[cat] += dv
        cat_order[cat] = None

    cats   = list(cat_order.keys())
    deltas = np.array([cat_delta[c] for c in cats])
    return cats, deltas


def _delta_matrix(data, modality, tasks_to_use, n_top=None):
    """
    Returns (cluster_names, delta_matrix (n_clusters, n_tasks)) for a modality.
    delta > 0 → enriched in high-risk predictions.
    For HE, fine-grained cluster IDs are collapsed to biological categories.
    """
    # Collect all cluster names from all tasks
    all_names = None
    for task in tasks_to_use:
        ca = data["tasks"].get(task, {}).get("cluster_affinity", {}).get(modality)
        if ca:
            all_names = ca["cluster_names"]
            break
    if all_names is None:
        return None, None

    # For HE: collapse to biological categories — collect union across all tasks
    if modality == "HE":
        from collections import OrderedDict
        bio_union = OrderedDict()
        for task in tasks_to_use:
            ca0 = data["tasks"].get(task, {}).get("cluster_affinity", {}).get("HE")
            if ca0:
                names_t, _ = _collapse_he_clusters(ca0["cluster_names"],
                                                    np.array(ca0["delta"]))
                for nm in names_t:
                    bio_union[nm] = None
        if bio_union:
            all_names = list(bio_union.keys())

    n_clus = len(all_names)
    mat = np.zeros((n_clus, len(tasks_to_use)))

    for ti, task in enumerate(tasks_to_use):
        ca = data["tasks"].get(task, {}).get("cluster_affinity", {}).get(modality)
        if ca is None:
            continue
        delta = np.array(ca["delta"])
        if modality == "HE":
            bio_names_t, task_delta = _collapse_he_clusters(ca["cluster_names"], delta)
            bio_delta_map = dict(zip(bio_names_t, task_delta))
            for ci, nm in enumerate(all_names):
                if nm in bio_delta_map:
                    mat[ci, ti] = bio_delta_map[nm]
        elif len(delta) == n_clus:
            mat[:, ti] = delta
        else:
            for ci, nm in enumerate(ca["cluster_names"]):
                if nm in all_names:
                    mat[all_names.index(nm), ti] = delta[ci]

    if n_top is not None:
        scores = np.abs(mat).max(axis=1)
        top_idx = np.argsort(scores)[::-1][:n_top]
        top_idx = sorted(top_idx)
        mat = mat[top_idx]
        all_names = [all_names[i] for i in top_idx]

    return all_names, mat


def _group_order(names, group_map):
    """
    Return indices that sort `names` by group membership (group_map).
    Names not in any group go at the end.
    """
    group_rank = {}
    for gi, (grp, members) in enumerate(group_map.items()):
        for nm in members:
            group_rank[nm] = gi

    def _key(i):
        nm = names[i]
        grp = group_rank.get(nm, len(group_map))
        return (grp, nm)

    return sorted(range(len(names)), key=_key)


def _norm_mat(mat):
    """Normalize matrix to [-1, 1] by dividing by max |value|. Returns (normed, scale)."""
    scale = np.abs(mat).max()
    scale = max(scale, 1e-10)
    return mat / scale, scale


def _sci(v):
    """Format a small float as a compact scientific string, e.g. 2.3×10⁻⁵."""
    if v == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(v))))
    man = v / 10**exp
    sup = str(exp).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{man:.1f}×10{sup}"


def _heatmap_panel(ax, mat_norm, names, tasks_to_use, title, title_color="black",
                   row_fs=7, col_fs=9):
    """Draw a normalized heatmap on ax. Returns the AxesImage."""
    n_rows, n_cols = mat_norm.shape
    im = ax.imshow(mat_norm, aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks_to_use],
                       fontsize=col_fs, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(names, fontsize=row_fs)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False,
                   length=2, pad=2)
    ax.set_title(title, fontsize=col_fs, fontweight="bold", color=title_color, pad=8)
    # Light grid lines between cells
    for x in np.arange(-0.5, n_cols, 1):
        ax.axvline(x, color="white", lw=0.4)
    for y in np.arange(-0.5, n_rows, 1):
        ax.axhline(y, color="white", lw=0.4)
    return im


def _draw_bracket_panel(ax_brk, group_spans, group_colors, n_rows, fontsize=7.5):
    """
    Draw right-side bracket annotations in a narrow axes.
    ax_brk must have ylim = (n_rows - 0.5, -0.5) to match the heatmap imshow.
    Each bracket: vertical bar + top/bottom ticks + rotated label.
    """
    ax_brk.set_xlim(0, 1)
    ax_brk.set_ylim(n_rows - 0.5, -0.5)   # inverted to match imshow
    ax_brk.axis("off")

    bar_x    = 0.18   # x of the vertical bar
    tick_len = 0.15   # length of horizontal ticks
    lbl_x    = 0.55   # x of the group label text

    for grp, s, e in group_spans:
        color = group_colors.get(grp, "#333")
        y0 = s - 0.38          # top of bracket (near top of first row)
        y1 = e - 1 + 0.38      # bottom of bracket (near bottom of last row)
        mid = (y0 + y1) / 2

        # Vertical bar
        ax_brk.plot([bar_x, bar_x], [y0, y1], color=color, lw=1.8,
                    clip_on=False, solid_capstyle="round")
        # Top tick
        ax_brk.plot([bar_x - tick_len, bar_x], [y0, y0], color=color, lw=1.8,
                    clip_on=False, solid_capstyle="round")
        # Bottom tick
        ax_brk.plot([bar_x - tick_len, bar_x], [y1, y1], color=color, lw=1.8,
                    clip_on=False, solid_capstyle="round")
        # Label — rotated 90° along the bracket
        ax_brk.text(lbl_x, mid, grp, ha="left", va="center",
                    fontsize=fontsize, fontweight="bold", color=color,
                    rotation=90, rotation_mode="anchor")


def fig1_BAL(data, out_dir, tasks_to_use):
    """
    BAL cell type enrichment heatmap with right-side group bracket annotations.
    All individual cell type names shown on y-axis; bracket spans each group.
    """
    names, mat = _delta_matrix(data, "BAL", tasks_to_use, n_top=None)
    if names is None:
        print("  [fig1] BAL data missing — skip"); return

    order = _group_order(names, BAL_GROUPS)
    names = [names[i] for i in order]
    mat   = mat[order]
    mat_norm, scale = _norm_mat(mat)

    # Build group span list
    group_spans = []
    current_grp, span_start = None, 0
    for ri, nm in enumerate(names):
        grp = next((g for g, ms in BAL_GROUPS.items() if nm in ms), "Other")
        if grp != current_grp:
            if current_grp is not None:
                group_spans.append((current_grp, span_start, ri))
            current_grp, span_start = grp, ri
    if current_grp:
        group_spans.append((current_grp, span_start, len(names)))

    n_rows, n_cols = len(names), len(tasks_to_use)

    from matplotlib.gridspec import GridSpec
    fig_w = n_cols * 1.6 + 4.5
    fig_h = n_rows * 0.30 + 2.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(1, 2, width_ratios=[1, 0.12], wspace=0.01, figure=fig)
    ax     = fig.add_subplot(gs[0])
    ax_brk = fig.add_subplot(gs[1])

    im = _heatmap_panel(ax, mat_norm, names, tasks_to_use,
                        "BAL cell-type predictive enrichment", row_fs=7, col_fs=9)

    # Dashed group separators in heatmap
    for grp, s, e in group_spans:
        if s > 0:
            ax.axhline(s - 0.5, color="#888", lw=0.8, ls="--", zorder=3)

    cbar = plt.colorbar(im, ax=ax, shrink=0.45, pad=0.01, aspect=25)
    cbar.set_label(f"Normalized Δ attention\n(max = {_sci(scale)})", fontsize=7.5)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=6.5)

    _draw_bracket_panel(ax_brk, group_spans, BAL_GROUP_COLORS, n_rows, fontsize=7.5)

    fig.text(0.5, 0.0,
             "Δ = mean seed attention (high-risk) − (low-risk)  |  Normalized to [−1, 1] per modality",
             ha="center", va="top", fontsize=6.5, color="#555", style="italic")

    out = out_dir / "paper_fig1_BAL_celltypes.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "paper_fig1_BAL_celltypes.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig1] saved {out}")
    return out, out_dir / "paper_fig1_BAL_celltypes.pdf"


def fig2_clinical(data, out_dir, tasks_to_use, n_top=25):
    """
    Top clinical features by normalized Δ seed-attention.
    Features ranked by max |Δ| across tasks, then sorted by clinical category.
    """
    names, mat = _delta_matrix(data, "Clinical", tasks_to_use, n_top=None)
    if names is None:
        print("  [fig2] Clinical data missing — skip"); return

    score = np.abs(mat).max(axis=1)
    top_idx = np.argsort(score)[::-1][:n_top]
    top_idx_sorted = sorted(top_idx, key=lambda i: _group_order(
        [names[i]], CLINICAL_GROUPS)[0] * 1000 + i)
    names_top = [names[i] for i in top_idx_sorted]
    mat_top   = mat[top_idx_sorted]
    mat_norm, scale = _norm_mat(mat_top)

    # Truncate long labels
    labels = [nm if len(nm) <= 28 else nm[:26] + "…" for nm in names_top]

    # Build group spans for clinical features
    clin_group_spans = []
    prev_grp, span_s = None, 0
    for ri, nm in enumerate(names_top):
        grp = next((g for g, ms in CLINICAL_GROUPS.items() if nm in ms), "Other")
        if grp != prev_grp:
            if prev_grp is not None:
                clin_group_spans.append((prev_grp, span_s, ri))
            prev_grp, span_s = grp, ri
    if prev_grp is not None:
        clin_group_spans.append((prev_grp, span_s, len(names_top)))

    n_rows, n_cols = len(labels), len(tasks_to_use)

    from matplotlib.gridspec import GridSpec
    fig_w = n_cols * 1.6 + 5.0
    fig_h = n_rows * 0.33 + 2.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(1, 2, width_ratios=[1, 0.12], wspace=0.01, figure=fig)
    ax     = fig.add_subplot(gs[0])
    ax_brk = fig.add_subplot(gs[1])

    im = _heatmap_panel(ax, mat_norm, labels, tasks_to_use,
                        f"Top-{n_top} clinical feature predictive enrichment",
                        row_fs=7, col_fs=9)

    # Dashed group separators in heatmap
    for grp, s, e in clin_group_spans:
        if s > 0:
            ax.axhline(s - 0.5, color="#888", lw=0.8, ls="--", zorder=3)

    cbar = plt.colorbar(im, ax=ax, shrink=0.45, pad=0.01, aspect=25)
    cbar.set_label(f"Normalized Δ attention\n(max = {_sci(scale)})", fontsize=7.5)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=6.5)

    _draw_bracket_panel(ax_brk, clin_group_spans, CLINICAL_GROUP_COLORS, n_rows, fontsize=7.5)

    fig.text(0.5, 0.0,
             "Δ = mean seed attention (high-risk) − (low-risk)  |  Normalized to [−1, 1] per modality",
             ha="center", va="top", fontsize=6.5, color="#555", style="italic")

    out = out_dir / "paper_fig2_clinical_features.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "paper_fig2_clinical_features.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig2] saved {out}")
    return out, out_dir / "paper_fig2_clinical_features.pdf"


def fig3_HE_CT(data, out_dir, tasks_to_use):
    """
    HE tissue category + CT cluster enrichment (side-by-side).
    HE clusters collapsed to 7 biological categories via HE_cluster_map.json.
    CT shows top-15 clusters by max |Δ|.
    """
    he_names, he_mat = _delta_matrix(data, "HE",  tasks_to_use, n_top=None)
    ct_names, ct_mat = _delta_matrix(data, "CT",  tasks_to_use, n_top=15)

    has_he = he_names is not None
    has_ct = ct_names is not None
    if not has_he and not has_ct:
        print("  [fig3] HE and CT data missing — skip"); return

    n_panels = int(has_he) + int(has_ct)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(n_panels * (len(tasks_to_use) * 1.5 + 2.5), 7))
    if n_panels == 1:
        axes = [axes]

    panel_i = 0
    for (names, mat, mod) in [(he_names, he_mat, "HE"), (ct_names, ct_mat, "CT")]:
        if names is None:
            continue
        ax = axes[panel_i]; panel_i += 1
        mat_norm, scale = _norm_mat(mat)

        im = _heatmap_panel(ax, mat_norm, names, tasks_to_use,
                            f"{mod} tissue/cluster enrichment",
                            title_color=MOD_COLORS[mod], row_fs=7.5, col_fs=8.5)

        cbar = plt.colorbar(im, ax=ax, shrink=0.45, pad=0.02, aspect=25)
        cbar.set_label(f"Norm. Δ  (max={_sci(scale)})", fontsize=7)
        cbar.set_ticks([-1, 0, 1])
        cbar.ax.tick_params(labelsize=6.5)

    fig.suptitle(
        "Tissue/morphology predictive enrichment\n"
        "Red = enriched in high-risk seeds  |  Blue = enriched in low-risk seeds  |  "
        "Normalized per modality",
        fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = out_dir / "paper_fig3_HE_CT_morphology.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "paper_fig3_HE_CT_morphology.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig3] saved {out}")
    return out, out_dir / "paper_fig3_HE_CT_morphology.pdf"


def fig4_gates(data, out_dir, tasks_to_use):
    """
    Modality gate weights per task.

    Gate weights are sigmoid outputs of a learned per-patient modality gating
    network. Higher weight = the model relies more on that modality.
    Values close to 1 = fully open gate; values close to 0 = suppressed.
    Y-axis zoomed to show relative differences (all gates are already fairly open).
    """
    mods = data.get("modalities", ["HE", "BAL", "CT", "Clinical"])
    n_tasks = len(tasks_to_use)

    means = np.zeros((len(mods), n_tasks))
    stds  = np.zeros((len(mods), n_tasks))
    for ti, task in enumerate(tasks_to_use):
        gw = data["tasks"].get(task, {}).get("gate_weights", {})
        mn = gw.get("mean", {}); sd = gw.get("std", {})
        for mi, mod in enumerate(mods):
            means[mi, ti] = mn.get(mod, 0.0)
            stds[mi, ti]  = sd.get(mod, 0.0)

    # Y range: start at 0.55 to magnify differences; never go above 1.05
    all_vals = means[means > 0]
    ymin = max(0.55, np.min(all_vals) - np.max(stds) - 0.05) if len(all_vals) else 0
    ymax = min(1.02, np.max(means) + np.max(stds) + 0.03)

    fig, ax = plt.subplots(figsize=(n_tasks * 2.0 + 1.5, 4.5))

    x = np.arange(n_tasks)
    width = 0.8 / len(mods)
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(mods))

    for mi, (mod, off) in enumerate(zip(mods, offsets)):
        bars = ax.bar(x + off, means[mi], width=width * 0.9,
                      bottom=0,
                      color=MOD_COLORS.get(mod, "#aaa"),
                      yerr=stds[mi], capsize=3,
                      error_kw={"elinewidth": 1.2, "capthick": 1.2},
                      label=mod, alpha=0.88, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks_to_use], fontsize=10)
    ax.set_ylabel("Mean gate weight (sigmoid output)", fontsize=9)
    ax.set_ylim(ymin, ymax)
    ax.axhline(1.0, color="#aaa", lw=0.8, ls=":")
    ax.set_title("Modality gate weights per task\n(mean ± std across patients; gate = 1 → fully open)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, lw=0.5, color="#ddd", zorder=0)
    ax.set_axisbelow(True)

    fig.text(0.5, -0.04,
             "Gate weight = learned sigmoid gate applied per patient per modality.\n"
             "Higher = the model relies more on that modality for this task. "
             "Y-axis starts at 0.55 to make inter-modality differences visible.",
             ha="center", va="top", fontsize=7, color="#555", style="italic")

    fig.tight_layout()
    out = out_dir / "paper_fig4_gate_weights.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "paper_fig4_gate_weights.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig4] saved {out}")
    return out, out_dir / "paper_fig4_gate_weights.pdf"


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--variant",       default="all_splits")
    pa.add_argument("--tasks",         default=None,
                    help="Comma-separated task subset (default: all 4 tasks)")
    pa.add_argument("--n-top-clinical", type=int, default=25)
    pa.add_argument("--wandb-project", default="chicago-mil-interpretability")
    args = pa.parse_args()

    tasks_to_use = (
        [t.strip() for t in args.tasks.split(",")]
        if args.tasks else TASKS
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = _load_all_splits(variant=args.variant)
    data    = _merge_records(records)

    tasks_available = [t for t in tasks_to_use if t in data.get("tasks", {})]
    if not tasks_available:
        raise RuntimeError(
            f"No tasks found in loaded data. Available: {list(data.get('tasks', {}).keys())}"
        )
    print(f"Tasks found: {tasks_available}")

    outs = {}
    r = fig1_BAL(data, OUT_DIR, tasks_available)
    if r: outs["fig1_BAL"] = r

    r = fig2_clinical(data, OUT_DIR, tasks_available, n_top=args.n_top_clinical)
    if r: outs["fig2_clinical"] = r

    r = fig3_HE_CT(data, OUT_DIR, tasks_available)
    if r: outs["fig3_HE_CT"] = r

    r = fig4_gates(data, OUT_DIR, tasks_available)
    if r: outs["fig4_gates"] = r

    # W&B logging
    if args.wandb_project.lower() != "none" and outs:
        run = wandb.init(
            project=args.wandb_project,
            name=f"paper_interpretability_{args.variant}",
            group="paper_figs",
        )
        log_dict = {}
        for key, (png, _pdf) in outs.items():
            log_dict[f"paper/{key}"] = wandb.Image(str(png), caption=key)
        wandb.log(log_dict)
        run.finish()
        print(f"  [wandb] uploaded {len(log_dict)} figures")

    print("Done.")


if __name__ == "__main__":
    main()
