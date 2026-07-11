#!/usr/bin/env python3
"""
TCGA spatial patch visualization + GeoMAE masking demo.

For a few slides per cancer type:
  Panel 1: All patches at spatial coords, colored by KMeans cluster
  Panel 2: UMAP of patch features, same cluster colors
  Panel 3: KNN graph overlay (spatial connectivity)
  Panel 4: BFS-flood contiguous masking (what GeoMAE masks)
  Panel 5: BFS depth (noise level per masked patch)
  Panel 6: Denoising order (boundary → interior)

Usage:
  python viz_tcga_spatial.py --cancers KIRC LGG --n-patients 2
"""
import sys, argparse, random
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mil.models.pretrain import build_knn_graph, bfs_distances, contiguous_region_mask

H5_DIRS = {
    "KIRC": "/lustre/groups/aih/dinesh.haridoss/mil/TCGA-KIRC",
    "BRCA": "/lustre/groups/aih/dinesh.haridoss/mil/TCGA-BRCA",
    "BLCA": "/lustre/groups/aih/dinesh.haridoss/mil/TCGA-BLCA",
    "LGG":  "/lustre/groups/aih/dinesh.haridoss/mil/TCGA-LGG",
    "GBM":  "/lustre/groups/aih/dinesh.haridoss/mil/TCGA-GBM",
}
OUT_DIR  = Path("/home/aih/dinesh.haridoss/chicago_mil/results/tcga_spatial_viz")
N_CLUSTERS = 8
KNN_K      = 8
MASK_RATIO = 0.50
MAX_PATCHES= 3000
SEED       = 42

BG   = "#0d1117"
AX   = "#161b22"
EDGE = "#30363d"

# Colorblind-friendly palette for clusters
CLUSTER_CMAP = plt.cm.tab10


def load_h5(h5_path: Path, max_patches: int):
    with h5py.File(h5_path, "r") as f:
        feats  = f["features"][0].astype(np.float32)     # (N, 1536)
        coords = f["coords_patching"][:].astype(np.float32)  # (N, 2)
    ok = ~np.isnan(feats).any(1)
    feats = feats[ok]; coords = coords[ok]
    if len(feats) > max_patches:
        idx = np.random.choice(len(feats), max_patches, replace=False)
        feats = feats[idx]; coords = coords[idx]
    return feats, coords


def cluster_patches(feats: np.ndarray, n_clusters: int = N_CLUSTERS):
    """KMeans on PCA-reduced features → cluster ids (0..k-1)."""
    pca  = PCA(n_components=min(50, feats.shape[1]), random_state=SEED)
    pca_f = pca.fit_transform(feats)
    km   = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=5)
    return km.fit_predict(pca_f)


def compute_umap(feats: np.ndarray):
    from umap import UMAP
    return UMAP(n_neighbors=15, min_dist=0.1,
                random_state=SEED, n_jobs=4).fit_transform(feats)


def run_geomae(feats: np.ndarray, coords: np.ndarray):
    t_feats  = torch.from_numpy(feats)
    t_coords = torch.from_numpy(coords)
    ei, ew   = build_knn_graph(t_coords, KNN_K)
    random.seed(SEED); torch.manual_seed(SEED)
    visible  = contiguous_region_mask(t_coords, MASK_RATIO, ei)
    dist     = bfs_distances(len(feats), ei, visible)
    alpha    = (dist.float() / max(dist.max().item(), 1)).clamp(0, 1).numpy()
    return visible.numpy(), dist.numpy(), alpha, ei.numpy(), ew.numpy()


def make_figure(cancer, slide_name, feats, coords, cluster_ids,
                umap_xy, visible, distances, alpha, ei):
    N   = len(feats)
    x   = coords[:, 0]; y = coords[:, 1]
    dot = max(2, min(6, 4000 / N))

    colors = np.array([CLUSTER_CMAP(c / N_CLUSTERS) for c in cluster_ids])

    fig = plt.figure(figsize=(28, 20), facecolor=BG)
    fig.suptitle(
        f"{cancer}  ·  {slide_name}  ·  {N:,} patches  ·  {N_CLUSTERS} clusters  "
        f"(KMeans on UNI features)",
        fontsize=13, fontweight="bold", color="white", y=0.99)

    def _ax(pos, title, sub=""):
        ax = fig.add_axes(pos)
        ax.set_facecolor(AX)
        for sp in ax.spines.values(): sp.set_color(EDGE)
        ax.tick_params(colors="#666", labelsize=7)
        ax.set_title(f"{title}\n{sub}", color="white", fontsize=9,
                     fontweight="bold", pad=5)
        return ax

    # ── Panel 1: Spatial layout colored by cluster ────────────────────────────
    ax1 = _ax((0.03, 0.53, 0.28, 0.40),
              "① Spatial layout — cluster IDs",
              "each dot = one 256×256 patch at its slide pixel position")
    for c in range(N_CLUSTERS):
        m = cluster_ids == c
        if m.sum() == 0: continue
        ax1.scatter(x[m], -y[m], c=[CLUSTER_CMAP(c/N_CLUSTERS)]*m.sum(),
                    s=dot, alpha=0.85, linewidths=0, label=f"C{c}")
    ax1.set_xlabel("x (px)", color="#666", fontsize=7)
    ax1.set_ylabel("y (px)", color="#666", fontsize=7)
    handles = [mpatches.Patch(color=CLUSTER_CMAP(c/N_CLUSTERS), label=f"Cluster {c}")
               for c in range(N_CLUSTERS)]
    ax1.legend(handles=handles, fontsize=6, loc="lower left",
               facecolor="#21262d", edgecolor=EDGE, labelcolor="white",
               ncol=2, framealpha=0.9)

    # ── Panel 2: UMAP ─────────────────────────────────────────────────────────
    ax2 = _ax((0.36, 0.53, 0.28, 0.40),
              "② Feature UMAP — same cluster colors",
              "UNI 1536-dim → UMAP(2D) — spatial proximity ≠ feature proximity")
    for c in range(N_CLUSTERS):
        m = cluster_ids == c
        if m.sum() == 0: continue
        ax2.scatter(umap_xy[m, 0], umap_xy[m, 1],
                    c=[CLUSTER_CMAP(c/N_CLUSTERS)]*m.sum(),
                    s=3, alpha=0.75, linewidths=0)
    ax2.set_xlabel("UMAP 1", color="#666", fontsize=7)
    ax2.set_ylabel("UMAP 2", color="#666", fontsize=7)

    # ── Panel 3: KNN graph ────────────────────────────────────────────────────
    ax3 = _ax((0.69, 0.53, 0.28, 0.40),
              f"③ KNN spatial graph  (k={KNN_K})",
              "edges = spatial neighbours → GeoMAE denoises along this graph")
    step = max(1, ei.shape[1] // 6000)
    src, tgt = ei[0, ::step], ei[1, ::step]
    segs = [[(x[s], -y[s]), (x[t], -y[t])] for s, t in zip(src, tgt)]
    lc   = LineCollection(segs, lw=0.2, alpha=0.2, color="#58a6ff")
    ax3.add_collection(lc)
    ax3.scatter(x, -y, c=colors, s=dot*0.8, alpha=0.7, linewidths=0, zorder=2)
    ax3.set_xlim(x.min()-200, x.max()+200)
    ax3.set_ylim(-y.max()-200, -y.min()+200)
    ax3.set_xlabel("x (px)", color="#666", fontsize=7)

    # ── Panel 4: BFS-flood masking ────────────────────────────────────────────
    ax4 = _ax((0.03, 0.05, 0.28, 0.40),
              "④ BFS-flood contiguous masking",
              f"50% masked in spatial blobs — what GeoMAE must reconstruct")
    ax4.scatter(x[~visible], -y[~visible], c="#30363d",
                s=dot*0.8, alpha=0.5, linewidths=0, zorder=1)
    for c in range(N_CLUSTERS):
        m = (cluster_ids == c) & visible
        if m.sum() == 0: continue
        ax4.scatter(x[m], -y[m], c=[CLUSTER_CMAP(c/N_CLUSTERS)]*m.sum(),
                    s=dot, alpha=0.9, linewidths=0, zorder=2)
    # Stats
    n_masked = (~visible).sum()
    ax4.text(0.02, 0.02, f"Masked: {n_masked:,} ({n_masked/N*100:.0f}%)",
             transform=ax4.transAxes, color="#8b949e", fontsize=7)
    ax4.scatter([], [], c="#30363d", s=20, label="Masked")
    ax4.scatter([], [], c="white",   s=20, label="Visible")
    ax4.legend(fontsize=7, facecolor="#21262d", edgecolor=EDGE,
               labelcolor="white", loc="lower right")
    ax4.set_xlabel("x (px)", color="#666", fontsize=7)
    ax4.set_ylabel("y (px)", color="#666", fontsize=7)

    # ── Panel 5: BFS depth (noise level) ─────────────────────────────────────
    ax5 = _ax((0.36, 0.05, 0.28, 0.40),
              "⑤ BFS depth = noise level α_t",
              "purple=boundary (easy)  →  yellow=interior (hard)")
    ax5.scatter(x[visible], -y[visible], c="white",
                s=dot*0.4, alpha=0.2, linewidths=0, zorder=1)
    sc = ax5.scatter(x[~visible], -y[~visible],
                     c=alpha[~visible], cmap="plasma",
                     s=dot*1.2, alpha=0.95, linewidths=0, zorder=2,
                     vmin=0, vmax=1)
    cb = fig.colorbar(sc, ax=ax5, fraction=0.03, pad=0.02)
    cb.set_label("α_t = dist / max_dist", color="white", fontsize=7)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
    cb.outline.set_edgecolor(EDGE)
    ax5.set_xlabel("x (px)", color="#666", fontsize=7)

    # ── Panel 6: Denoising order ──────────────────────────────────────────────
    ax6 = _ax((0.69, 0.05, 0.28, 0.40),
              "⑥ Ideal denoising order",
              "blue=reconstruct first  →  red=reconstruct last\n"
              "model must work outward→inward (causal attention)")
    ud = np.unique(distances[~visible])
    n_rings = max(len(ud), 1)
    ring_norm = np.zeros(N)
    for i, d in enumerate(ud):
        ring_norm[(~visible) & (distances == d)] = i / (n_rings - 1 + 1e-8)
    ax6.scatter(x[visible], -y[visible], c="white",
                s=dot*0.4, alpha=0.2, linewidths=0, zorder=1)
    sc2 = ax6.scatter(x[~visible], -y[~visible],
                      c=ring_norm[~visible], cmap="coolwarm",
                      s=dot*1.2, alpha=0.95, linewidths=0, zorder=2,
                      vmin=0, vmax=1)
    cb2 = fig.colorbar(sc2, ax=ax6, fraction=0.03, pad=0.02)
    cb2.set_label("Reconstruction ring", color="white", fontsize=7)
    cb2.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
    cb2.outline.set_edgecolor(EDGE)

    # Arrows showing reconstruction flow
    masked_d1 = np.where((~visible) & (distances == 1))[0]
    for _ in range(5):
        if len(masked_d1) == 0: break
        i = masked_d1[np.random.randint(len(masked_d1))]
        nbrs = ei[1][ei[0] == i]
        inner = nbrs[(distances[nbrs] > distances[i]) & (~visible[nbrs])]
        if len(inner) == 0: continue
        j = inner[0]
        ax6.annotate("", xy=(x[j], -y[j]), xytext=(x[i], -y[i]),
                     arrowprops=dict(arrowstyle="->", color="white",
                                     lw=0.8, alpha=0.6))
    ax6.set_xlabel("x (px)", color="#666", fontsize=7)

    # ── Cluster composition bar chart (small inset) ───────────────────────────
    ax_bar = fig.add_axes([0.04, 0.495, 0.26, 0.03])
    ax_bar.set_facecolor(AX)
    counts = np.bincount(cluster_ids, minlength=N_CLUSTERS)
    lefts  = np.cumsum(np.concatenate([[0], counts[:-1]])) / N
    widths = counts / N
    for c in range(N_CLUSTERS):
        ax_bar.barh(0, widths[c], left=lefts[c], height=1,
                    color=CLUSTER_CMAP(c/N_CLUSTERS), alpha=0.9)
    ax_bar.set_xlim(0, 1); ax_bar.axis("off")
    ax_bar.set_title("Cluster proportions", color="#8b949e",
                     fontsize=6, pad=2, loc="left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{cancer}_{slide_name[:20]}_spatial_geomae.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {out}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cancers",    nargs="+", default=["KIRC", "LGG"])
    p.add_argument("--n-patients", type=int,  default=2)
    p.add_argument("--max-patches",type=int,  default=MAX_PATCHES)
    p.add_argument("--n-clusters", type=int,  default=N_CLUSTERS)
    p.add_argument("--no-umap",    action="store_true",
                   help="Skip UMAP (faster)")
    args = p.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    for cancer in args.cancers:
        h5_dir = Path(H5_DIRS[cancer])
        h5s    = sorted(h5_dir.glob("*.h5"))[:args.n_patients]
        print(f"\n{'='*60}  {cancer}  {'='*60}")

        for h5_path in h5s:
            print(f"\n  {h5_path.name}")
            print(f"  Loading {args.max_patches} patches...")
            feats, coords = load_h5(h5_path, args.max_patches)
            print(f"  {len(feats):,} patches loaded")

            print(f"  Clustering (k={args.n_clusters})...")
            cluster_ids = cluster_patches(feats, args.n_clusters)

            if not args.no_umap:
                print(f"  Computing UMAP...")
                umap_xy = compute_umap(feats)
            else:
                # PCA fallback
                umap_xy = PCA(n_components=2).fit_transform(feats)

            print(f"  Running GeoMAE masking pipeline...")
            visible, distances, alpha, ei, ew = run_geomae(feats, coords)
            n_masked = (~visible).sum()
            print(f"  Masked: {n_masked:,}  Depth range: {distances[~visible].min()}-{distances[~visible].max()}")

            make_figure(cancer, h5_path.stem[:30], feats, coords,
                        cluster_ids, umap_xy, visible, distances, alpha, ei)

    print(f"\nDone → {OUT_DIR}")
    print("Download:")
    print(f"  scp 'dinesh.haridoss@ictstr01.helmholtz-munich.de:{OUT_DIR}/*.png' ~/Desktop/")


if __name__ == "__main__":
    main()
