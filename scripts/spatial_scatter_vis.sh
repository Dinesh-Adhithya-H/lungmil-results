#!/usr/bin/env bash
#SBATCH --job-name=spatial_scatter
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=150G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/spatial_scatter_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/spatial_scatter_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
"""
Per-slide spatial scatter plots using real x,y patch coordinates.
Four panels per slide:
  1. Tissue type        — safe (green/blue), warning (orange), danger (red)
  2. Patch attention    — patch_attn[k][i]  (within-cluster, before neighbourhood weighting)
  3. Neighbourhood attn — neighbourhood_attn[k] broadcast to all patches in cluster k
  4. Combined           — neighbourhood_attn[k] x patch_attn[k][i]  (product)

Figures for: 20 ACR+, 20 ACR-, up to 5 FP + 5 FN.
"""
import torch, torch.nn as nn
import numpy as np, pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

FEAT_DIR = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/slide_cluster_feats")
OUTDIR   = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil")
VIS_DIR  = OUTDIR / "spatial_scatter_plots"
VIS_DIR.mkdir(exist_ok=True)

H5AD        = "/lustre/groups/aih/dinesh.haridoss/datasets/adata_v3.h5ad"
IN_DIM, HIDDEN, DROPOUT = 1024, 256, 0.25
MAX_PATCHES = 512
N_SPLITS    = 5

# Tissue colour palette: safe=green/blue, warning=orange, danger=red
TISSUE_COLORS = {
    "Alveolar":                                       "#4CAF50",  # green  — safe
    "Alveolar with empty spaces":                     "#2196F3",  # blue   — safe
    "Bronchial":                                      "#9C27B0",  # purple — neutral
    "Cartilage":                                      "#00BCD4",  # cyan   — neutral
    "Alveolar with hemorrhage and inflammation":      "#FF9800",  # orange — warning
    "Lymphocytoplasmic inflammation":                 "#F44336",  # red    — danger
    "Unknown":                                        "#AAAAAA",
}
DRAW_ORDER = [
    "Alveolar", "Alveolar with empty spaces", "Bronchial", "Cartilage",
    "Alveolar with hemorrhage and inflammation",
    "Lymphocytoplasmic inflammation", "Unknown",
]

# ── Model ──────────────────────────────────────────────────────────────────────
class GatedAttentionPool(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(in_dim, hidden), nn.Sigmoid())
        self.w = nn.Linear(hidden, 1, bias=False)
    def forward(self, h):
        a = self.w(self.V(h) * self.U(h))
        a = torch.softmax(a, dim=1)
        return (a * h).sum(dim=1), a.squeeze(-1)

class SpatialABMIL2Level(nn.Module):
    def __init__(self, in_dim=1024, hidden=256, dropout=0.25):
        super().__init__()
        self.patch_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.patch_attn = GatedAttentionPool(hidden, hidden // 2)
        self.nbhd_proj  = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.nbhd_attn  = GatedAttentionPool(hidden, hidden // 2)
        self.head       = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, clusters):
        nbhd_reps, patch_attns = [], []
        for c in clusters:
            h = self.patch_proj(c.unsqueeze(0))
            rep, pa = self.patch_attn(h)
            nbhd_reps.append(rep); patch_attns.append(pa.squeeze(0))
        H = self.nbhd_proj(torch.stack(nbhd_reps, dim=1))
        slide_rep, na = self.nbhd_attn(H)
        return self.head(slide_rep).squeeze(-1), na.squeeze(0), patch_attns

# ── Dataset ────────────────────────────────────────────────────────────────────
class SpatialMILDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        d = torch.load(self.paths[i], map_location='cpu', weights_only=False)
        clusters = []
        for c in d['clusters']:
            f = c['feats']
            if f.shape[0] > MAX_PATCHES:
                f = f[torch.randperm(f.shape[0])[:MAX_PATCHES]]
            clusters.append(f)
        return (clusters,
                torch.tensor(d['label'], dtype=torch.float32).unsqueeze(0),
                d['slide'], d['patient'],
                [c['coords'].numpy() for c in d['clusters']],
                [c['n_patches'] for c in d['clusters']])

def collate_fn(b): return b[0]

# ── Load paths ─────────────────────────────────────────────────────────────────
all_paths = sorted(FEAT_DIR.glob("*.pt"))
labels, patients = [], []
for p in all_paths:
    d = torch.load(p, map_location='cpu', weights_only=False)
    labels.append(d['label']); patients.append(d['patient'])
labels, patients = np.array(labels), np.array(patients)
print(f"Total slides: {len(all_paths)}  ACR+={labels.sum()}", flush=True)

cv = StratifiedGroupKFold(n_splits=N_SPLITS)
fold_splits = list(cv.split(all_paths, labels, patients))

# ── Inference ─────────────────────────────────────────────────────────────────
all_results = []
for fold in [0, 1]:
    ckpt = OUTDIR / f"spatial_abmil_fold{fold}.pt"
    if not ckpt.exists():
        print(f"Fold {fold}: no checkpoint — skip", flush=True); continue

    model = SpatialABMIL2Level(IN_DIM, HIDDEN, DROPOUT)
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    model.eval()

    _, te_idx = fold_splits[fold]
    te_ds = SpatialMILDataset([all_paths[i] for i in te_idx])
    te_dl = DataLoader(te_ds, batch_size=1, shuffle=False,
                       collate_fn=collate_fn, num_workers=4)

    with torch.no_grad():
        for clusters, label, slide_name, patient, coords, n_patches in te_dl:
            logit, nbhd_attn, patch_attns = model(clusters)
            prob = torch.sigmoid(logit).item()
            na   = nbhd_attn.numpy()   # (K,)

            patch_coords_all = []
            patch_only_all   = []   # patch_attn[k][i]            — within-cluster only
            nbhd_only_all    = []   # neighbourhood_attn[k]       — broadcast per cluster
            combined_all     = []   # neighbourhood_attn[k] x patch_attn[k][i]

            for k, (coord_arr, pa, nw) in enumerate(zip(coords, patch_attns, na)):
                pa_np = pa.numpy()
                nw_f  = float(nw)
                patch_coords_all.append(coord_arr)
                patch_only_all.append(pa_np)
                nbhd_only_all.append(np.full(len(pa_np), nw_f))
                combined_all.append(pa_np * nw_f)

            all_results.append({
                'fold':       fold,
                'slide':      slide_name,
                'patient':    patient,
                'label':      int(label.item()),
                'prob':       prob,
                'coords':     patch_coords_all,
                'patch_attn': patch_only_all,
                'nbhd_attn':  nbhd_only_all,
                'combined':   combined_all,
            })

print(f"Inference done: {len(all_results)} slides", flush=True)
slide_df = pd.DataFrame([{k: v for k, v in r.items()
                           if k not in ('coords','patch_attn','nbhd_attn','combined')}
                          for r in all_results])
auc = roc_auc_score(slide_df['label'], slide_df['prob'])
print(f"OOF AUC (folds 0+1): {auc:.3f}", flush=True)

# ── Select slides ──────────────────────────────────────────────────────────────
pos = slide_df[slide_df['label'] == 1].sort_values('prob', ascending=False)
neg = slide_df[slide_df['label'] == 0].sort_values('prob', ascending=True)
fp  = slide_df[(slide_df['label'] == 0) & (slide_df['prob'] > 0.5)].sort_values('prob', ascending=False)
fn  = slide_df[(slide_df['label'] == 1) & (slide_df['prob'] < 0.5)].sort_values('prob', ascending=True)

selected = {
    'ACR_positive':  pos.head(20)['slide'].tolist(),
    'ACR_negative':  neg.head(20)['slide'].tolist(),
    'FalsePositive': fp.head(5)['slide'].tolist(),
    'FalseNegative': fn.head(5)['slide'].tolist(),
}
print(f"ACR+={len(selected['ACR_positive'])}  ACR-={len(selected['ACR_negative'])}  "
      f"FP={len(selected['FalsePositive'])}  FN={len(selected['FalseNegative'])}", flush=True)

result_map = {r['slide']: r for r in all_results}

# ── Load tissue types ──────────────────────────────────────────────────────────
print("Loading h5ad obs...", flush=True)
adata = ad.read_h5ad(H5AD, backed='r')
obs   = adata.obs[['slide_name_clean', 'chunk_left', 'chunk_top', 'tissue_type']].copy()
obs['chunk_left'] = obs['chunk_left'].astype(np.float32)
obs['chunk_top']  = obs['chunk_top'].astype(np.float32)
print(f"  Loaded {len(obs):,} patches", flush=True)

all_needed = set(sum(selected.values(), []))
obs_needed = obs[obs['slide_name_clean'].isin(all_needed)]
slide_tissue = {}
for slide, grp in obs_needed.groupby('slide_name_clean'):
    arr = grp[['chunk_left', 'chunk_top', 'tissue_type']].values
    slide_tissue[slide] = {(round(float(r[0])), round(float(r[1]))): str(r[2]) for r in arr}
print(f"  Built tissue maps for {len(slide_tissue)} slides", flush=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def _minmax(arr):
    """Raw min-max normalisation to [0,1] — no log transform."""
    a = np.array(arr, dtype=float)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

ATTN_CMAP = 'RdYlGn_r'   # green=low, yellow=mid, red=high attention

def scatter_attn(ax, x, y, vals):
    """Scatter coloured by raw attention (min-max). High-attn patches drawn on top."""
    norm  = _minmax(vals)
    order = np.argsort(norm)
    sc = ax.scatter(x[order], y[order], s=0.8, c=norm[order],
                    cmap=ATTN_CMAP, vmin=0, vmax=1,
                    alpha=0.9, linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')
    ax.spines[:].set_visible(False)
    return sc, float(np.array(vals).min()), float(np.array(vals).max())

def plot_slide_4panels(axes, res, tissue_map):
    """
    axes: (ax_tissue, ax_patch, ax_nbhd, ax_combined)
    Returns (sc_patch, sc_nbhd, sc_combined) ScalarMappables with vmin/vmax set
    to the real raw ranges of this slide (used for per-figure colorbars).
    """
    ax_t, ax_pa, ax_nb, ax_co = axes

    all_x, all_y, all_tissue = [], [], []
    all_patch, all_nbhd, all_combined = [], [], []

    for coord_arr, pa, nb, co in zip(
            res['coords'], res['patch_attn'], res['nbhd_attn'], res['combined']):
        for (x, y), pv, nv, cv in zip(coord_arr, pa, nb, co):
            all_x.append(float(x))
            all_y.append(-float(y))
            key = (round(float(x)), round(float(y)))
            all_tissue.append(tissue_map.get(key, 'Unknown'))
            all_patch.append(float(pv))
            all_nbhd.append(float(nv))
            all_combined.append(float(cv))

    all_x = np.array(all_x); all_y = np.array(all_y)

    # Panel 1: tissue type
    for tt in DRAW_ORDER:
        mask = np.array([t == tt for t in all_tissue])
        if mask.any():
            ax_t.scatter(all_x[mask], all_y[mask], s=0.8,
                         c=TISSUE_COLORS[tt], alpha=0.85,
                         linewidths=0, rasterized=True)
    ax_t.set_xticks([]); ax_t.set_yticks([])
    ax_t.set_aspect('equal')
    ax_t.spines[:].set_visible(False)

    # Panels 2-4: raw attention scores (min-max per panel per slide)
    _, pa_min, pa_max = scatter_attn(ax_pa, all_x, all_y, all_patch)
    _, nb_min, nb_max = scatter_attn(ax_nb, all_x, all_y, all_nbhd)
    _, co_min, co_max = scatter_attn(ax_co, all_x, all_y, all_combined)

    return (pa_min, pa_max), (nb_min, nb_max), (co_min, co_max)


def make_figure(slide_names, category, slides_per_row=3):
    n = len(slide_names)
    if n == 0: return

    n_rows    = int(np.ceil(n / slides_per_row))
    # Each slide: 4 panels + tiny gap; slides separated by a wider gap
    # Build column-width ratios: [1,1,1,1, 0.06] × slides_per_row (drop last gap)
    col_ratios = []
    for i in range(slides_per_row):
        col_ratios += [1, 1, 1, 1]
        if i < slides_per_row - 1:
            col_ratios += [0.08]   # inter-slide gap column
    total_cols = len(col_ratios)

    fig_w = slides_per_row * 12        # 4 panels × 3" each
    fig_h = n_rows * 3.8 + 1.4

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = fig.add_gridspec(n_rows, total_cols,
                           left=0.01, right=0.99,
                           top=0.93, bottom=0.10,
                           wspace=0.02, hspace=0.42)

    stride = 5   # 4 panels + 1 gap

    for i, slide_name in enumerate(slide_names):
        row  = i // slides_per_row
        col0 = (i % slides_per_row) * stride

        ax_t  = fig.add_subplot(gs[row, col0])
        ax_pa = fig.add_subplot(gs[row, col0 + 1])
        ax_nb = fig.add_subplot(gs[row, col0 + 2])
        ax_co = fig.add_subplot(gs[row, col0 + 3])

        if slide_name not in result_map:
            for ax in [ax_t, ax_pa, ax_nb, ax_co]:
                ax.set_visible(False)
            continue

        res  = result_map[slide_name]
        tmap = slide_tissue.get(slide_name, {})

        pa_rng, nb_rng, co_rng = plot_slide_4panels(
            (ax_t, ax_pa, ax_nb, ax_co), res, tmap)

        lbl_str  = "ACR+" if res['label'] == 1 else "ACR-"
        pred_str = f"p={res['prob']:.2f}"
        short    = slide_name.split('/')[-1][:26]
        ax_t.set_title(f"{lbl_str} {pred_str}\n{short}", fontsize=6, pad=2)
        # Show raw range in subtitle of each attn panel
        ax_pa.set_title(f"Patch attn\n[{pa_rng[0]:.4f}, {pa_rng[1]:.4f}]", fontsize=5.5, pad=2)
        ax_nb.set_title(f"Nbhd attn\n[{nb_rng[0]:.4f}, {nb_rng[1]:.4f}]",  fontsize=5.5, pad=2)
        ax_co.set_title(f"Combined\n[{co_rng[0]:.5f}, {co_rng[1]:.5f}]",    fontsize=5.5, pad=2)

    # Hide unused slots
    for i in range(n, n_rows * slides_per_row):
        row  = i // slides_per_row
        col0 = (i % slides_per_row) * stride
        for dc in range(4):
            ax = fig.add_subplot(gs[row, col0 + dc])
            ax.set_visible(False)

    # Tissue legend
    legend_items = [
        mpatches.Patch(color="#4CAF50", label="Alveolar"),
        mpatches.Patch(color="#2196F3", label="Alveolar + empty spaces"),
        mpatches.Patch(color="#9C27B0", label="Bronchial"),
        mpatches.Patch(color="#00BCD4", label="Cartilage"),
        mpatches.Patch(color="#FF9800", label="Alveolar with haemorrhage and inflammation"),
        mpatches.Patch(color="#F44336", label="Lymphocytoplasmic inflammation"),
        mpatches.Patch(color="#AAAAAA", label="Unknown"),
    ]
    fig.legend(handles=legend_items, loc='lower left', ncol=4,
               fontsize=6.5, frameon=False, bbox_to_anchor=(0.01, 0.005))

    # Three separate colorbars: patch | nbhd | combined
    sm = plt.cm.ScalarMappable(cmap=ATTN_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    for x0, label in [
        (0.55, 'Patch attn (raw, min-max)'),
        (0.70, 'Nbhd attn (raw, min-max)'),
        (0.85, 'Combined (raw, min-max)'),
    ]:
        cax = fig.add_axes([x0, 0.025, 0.12, 0.018])
        cb  = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cb.set_label(label, fontsize=5.5)
        cb.set_ticks([0, 0.5, 1])
        cb.set_ticklabels(['low', 'mid', 'high'])
        cb.ax.tick_params(labelsize=5)

    plt.suptitle(
        f'{category}  |  Tissue type  |  Patch attn  |  Nbhd attn  |  Combined (product)',
        fontsize=10, y=0.97)

    out_path = VIS_DIR / f"{category}.png"
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


# ── Generate ───────────────────────────────────────────────────────────────────
make_figure(selected['ACR_positive'],  'ACR_positive_top20',      slides_per_row=3)
make_figure(selected['ACR_negative'],  'ACR_negative_top20',      slides_per_row=3)
make_figure(selected['FalsePositive'], 'WrongPred_FalsePositive', slides_per_row=5)
make_figure(selected['FalseNegative'], 'WrongPred_FalseNegative', slides_per_row=5)

print(f"\nAll figures saved to: {VIS_DIR}", flush=True)
PYEOF
