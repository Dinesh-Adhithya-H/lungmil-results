"""
Shared constants, colour maps, and utility functions used by both
interpret_set_mil_mt.py and interpret_longitudinal_mk.py.

Import with:
    from shared import (MOD_ORDER, MOD_COLORS, TASK_COLORS, TASK_LABELS,
                        HE_BIO_MAP, HE_BIO_COLORS, bio_label,
                        savefig, umap_embed,
                        seed_cluster_mass, sorted_cluster_order,
                        sort_seeds_by_diversity, noncollapsed_seed_mask)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths (shared across scripts) ─────────────────────────────────────────────
_HERE          = Path(__file__).resolve().parent
ROOT           = _HERE.parent

SPLITS_CSV     = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
SAMPLES_DIR    = "/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples/"
RESULTS_ROOT   = ROOT / "results" / "mm_abmil_v8" / "phase2"
HE_CLUSTER_MAP = ROOT / "results" / "cluster_name_maps" / "HE_cluster_map.json"

# ── Modality constants ─────────────────────────────────────────────────────────
MOD_ORDER  = ["HE", "BAL", "CT", "Clinical"]
MOD_COLORS = {"HE": "#58a6ff", "BAL": "#3fb950", "CT": "#d4a017", "Clinical": "#d2a8ff"}

# ── Task constants ─────────────────────────────────────────────────────────────
TASK_COLORS = {
    "acr_cls":   "#e53935", "acr_surv":   "#ff7043",
    "clad_surv": "#7e57c2", "death_surv": "#26a69a",
    "clad":      "#7e57c2", "death":      "#26a69a",
}
TASK_LABELS = {
    "acr_cls":   "ACR Classification", "acr_surv":   "ACR Survival",
    "clad_surv": "CLAD Survival",      "death_surv": "Death Survival",
    "clad":      "CLAD Survival",      "death":      "Death Survival",
}

PDF_DPI = 150
PNG_DPI = 120

# ── HE cluster biological-category helpers ────────────────────────────────────

def _load_he_bio_map():
    try:
        return json.loads(HE_CLUSTER_MAP.read_text())
    except Exception:
        return {}

HE_BIO_MAP = _load_he_bio_map()

HE_BIO_COLORS = {
    "Alveolar with hemorrhage and inflammation": "#e57373",
    "Alveolar with empty spaces":               "#64b5f6",
    "Alveolar":                                 "#81c784",
    "Bronchial":                                "#ce93d8",
    "Lymphocytoplasmic inflammation":           "#ffb74d",
    "Cartilage":                                "#f06292",
    "Unknown":                                  "#b0bec5",
}

_HE_BIO_ORDER = list(HE_BIO_COLORS.keys())


def bio_label(cluster_name: str) -> str:
    """Map a cluster name to its biological category string."""
    return HE_BIO_MAP.get(cluster_name, "Unknown")


# ── Figure save helper (always PDF + PNG) ─────────────────────────────────────

def savefig(fig, out_dir: Path, stem: str) -> Path:
    """Save figure as both PDF and PNG; return the PNG path (used by wandb)."""
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, dpi=PDF_DPI, bbox_inches="tight")
    fig.savefig(png, dpi=PNG_DPI, bbox_inches="tight")
    return png


# ── UMAP embedding ─────────────────────────────────────────────────────────────

def umap_embed(X, n_neighbors=30, min_dist=0.2, seed=42, metric="euclidean"):
    """Project an (N, D) array to 2D with UMAP. Returns (N, 2) float32."""
    from umap import UMAP
    return UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                metric=metric, random_state=seed,
                n_jobs=4).fit_transform(X.astype(np.float32))


# ── Seed → cluster attention mass ─────────────────────────────────────────────

def seed_cluster_mass(pma_attn_KN, cluster_labels_N, k_clus=None):
    """
    For each PMA seed k, compute total B-cos attention mass over each cluster c.

    pma_attn_KN     : (K_seeds, N) — relu(q·k)^b scores (or normalized weights)
    cluster_labels_N: (N,) integer cluster IDs for each instance
    k_clus          : number of clusters (default: max(cluster_labels)+1)

    Returns (K_seeds, k_clus) — raw summed mass per (seed, cluster) pair.
    """
    K_seeds = pma_attn_KN.shape[0]
    if k_clus is None:
        k_clus = int(cluster_labels_N.max()) + 1 if len(cluster_labels_N) > 0 else 8
    out = np.zeros((K_seeds, k_clus), dtype=np.float32)
    for c in range(k_clus):
        mask = (cluster_labels_N == c)
        if mask.any():
            out[:, c] = pma_attn_KN[:, mask].sum(axis=1)
    return out


# ── Cluster column ordering ───────────────────────────────────────────────────

def _parse_cluster_name(name: str):
    """Parse '0_1' → (macro=0, sub=1) or '2' → (2, 0) for numeric sort."""
    parts = name.split("_")
    try:
        macro = int(parts[0])
        sub   = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        macro, sub = 999, 0
    return macro, sub


def _he_sort_key(name: str):
    """Sort key for HE cluster names: (bio_category_rank, macro_id, sub_id)."""
    bio  = HE_BIO_MAP.get(name, "Unknown")
    rank = _HE_BIO_ORDER.index(bio) if bio in _HE_BIO_ORDER else len(_HE_BIO_ORDER)
    macro, sub = _parse_cluster_name(name)
    return (rank, macro, sub)


def sorted_cluster_order(nms, mod, mean_aff_raw=None, top_n=None):
    """
    Return sorted column indices for a heatmap of cluster names.

    Clinical: sort by total B-cos attention mass descending; take top_n if given.
    HE:       group by biological category → macro → sub-cluster.
    Others:   original index order.
    """
    if mod == "Clinical" and mean_aff_raw is not None:
        col_mass = mean_aff_raw.sum(axis=0)
        order = list(np.argsort(col_mass)[::-1])
        if top_n is not None:
            order = order[:top_n]
        return order
    if mod == "HE" and HE_BIO_MAP:
        return sorted(range(len(nms)), key=lambda i: _he_sort_key(nms[i]))
    return list(range(len(nms)))


# ── Seed diversity sort ───────────────────────────────────────────────────────

def sort_seeds_by_diversity(mean_aff: np.ndarray):
    """
    Reorder seed rows by hierarchical clustering so similar seeds are adjacent.
    Returns an index array compatible with mean_aff[order, :].
    """
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        if mean_aff.shape[0] < 3:
            return np.arange(mean_aff.shape[0])
        Z = linkage(mean_aff, method="ward", metric="euclidean")
        return leaves_list(Z)
    except Exception:
        return np.arange(mean_aff.shape[0])


# ── Non-collapsed seed mask ───────────────────────────────────────────────────

def noncollapsed_seed_mask(results, present_mods_main, K, keep_pct=50):
    """
    Return bool mask (total_seeds,): True = seed has high-entropy attention
    (attends broadly across instances, not collapsed to a single patch).

    Seeds in the bottom (100-keep_pct)% by mean entropy are flagged collapsed.
    """
    total   = len(present_mods_main) * K
    ent_acc = [[] for _ in range(total)]
    for r in results:
        for mod in present_mods_main:
            pa = r.get("pma_attn", {}).get(mod)
            if pa is None or pa.ndim != 2 or pa.shape[0] != K:
                continue
            s0 = present_mods_main.index(mod) * K
            a  = np.clip(pa, 1e-9, 1.0)
            h  = -np.sum(a * np.log(a), axis=1)   # entropy per seed
            for ki in range(K):
                ent_acc[s0 + ki].append(float(h[ki]))
    mean_ent = np.array([np.mean(v) if v else np.nan for v in ent_acc])
    thresh   = np.nanpercentile(mean_ent, 100 - keep_pct)
    return (mean_ent >= thresh) & ~np.isnan(mean_ent)
