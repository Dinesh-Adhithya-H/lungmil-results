#!/usr/bin/env bash
#SBATCH --job-name=tissue_attn_vis
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/tissue_attn_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/tissue_attn_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
"""
Side-by-side tissue-type scatter + attention scatter for:
  - 20 highest-confidence ACR+ slides
  - 20 highest-confidence ACR- slides
  - Up to 10 wrong predictions (FP + FN)

Left panel  : patches coloured by tissue type (from adata leiden annotation)
Right panel : patches coloured by per-patch attention score
              = neighbourhood_attn[k] * patch_attn_within_cluster[k][i]
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
VIS_DIR  = OUTDIR / "tissue_attn_plots"
VIS_DIR.mkdir(exist_ok=True)

H5AD    = "/lustre/groups/aih/dinesh.haridoss/datasets/adata_v3.h5ad"
IN_DIM, HIDDEN, DROPOUT = 1024, 256, 0.25
MAX_PATCHES = 512
N_SPLITS    = 5

TISSUE_COLORS = {
    "Alveolar":                              "#E53030",
    "Bronchial":                             "#1A72CC",
    "Cartilage":                             "#F5C518",
    "Alveolar with empty spaces":            "#2D8A2D",
    "Alveolar with hemorrhage and inflammation": "#FF8C00",
    "Lymphocytoplasmic inflammation":        "#CC44CC",
    "Unknown":                               "#AAAAAA",
}
DEFAULT_COLOR = "#AAAAAA"

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
        return (clusters, torch.tensor(d['label'], dtype=torch.float32).unsqueeze(0),
                d['slide'], d['patient'],
                [c['coords'].numpy() for c in d['clusters']],
                [c['n_patches'] for c in d['clusters']])

def collate_fn(b):
    return b[0]

# ── Load all slide paths ───────────────────────────────────────────────────────
all_paths = sorted(FEAT_DIR.glob("*.pt"))
labels, patients = [], []
for p in all_paths:
    d = torch.load(p, map_location='cpu', weights_only=False)
    labels.append(d['label']); patients.append(d['patient'])
labels, patients = np.array(labels), np.array(patients)
print(f"Total slides: {len(all_paths)}  ACR+={labels.sum()}", flush=True)

cv = StratifiedGroupKFold(n_splits=N_SPLITS)
fold_splits = list(cv.split(all_paths, labels, patients))

# ── Run inference for folds 0 and 1 ───────────────────────────────────────────
all_results = []   # list of dicts per slide

for fold in [0, 1]:
    ckpt = OUTDIR / f"spatial_abmil_fold{fold}.pt"
    if not ckpt.exists():
        print(f"Fold {fold}: no checkpoint — skip", flush=True); continue

    model = SpatialABMIL2Level(IN_DIM, HIDDEN, DROPOUT)
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    model.eval()

    _, te_idx = fold_splits[fold]
    te_paths  = [all_paths[i] for i in te_idx]
    te_ds     = SpatialMILDataset(te_paths)
    te_dl     = DataLoader(te_ds, batch_size=1, shuffle=False,
                           collate_fn=collate_fn, num_workers=4)

    with torch.no_grad():
        for clusters, label, slide_name, patient, coords, n_patches in te_dl:
            logit, nbhd_attn, patch_attns = model(clusters)
            prob = torch.sigmoid(logit).item()
            na   = nbhd_attn.numpy()

            # Per-patch combined attention: nbhd_attn[k] * patch_attn[k][i]
            patch_coords_all, patch_combined_attn = [], []
            for k, (coord_arr, pa, nw) in enumerate(zip(coords, patch_attns, na)):
                pa_np = pa.numpy()
                patch_coords_all.append(coord_arr)
                patch_combined_attn.append(pa_np * float(nw))

            all_results.append({
                'fold':        fold,
                'slide':       slide_name,
                'patient':     patient,
                'label':       int(label.item()),
                'prob':        prob,
                'coords':      patch_coords_all,    # list of (N_k, 2) arrays
                'attn':        patch_combined_attn, # list of (N_k,) arrays
                'nbhd_attn':   na,
                'pt_path':     FEAT_DIR / f"{slide_name.replace('/', '_')}.pt",
            })

print(f"Inference done: {len(all_results)} slides", flush=True)
slide_df = pd.DataFrame([{k: v for k, v in r.items()
                           if k not in ('coords','attn','nbhd_attn','pt_path')}
                          for r in all_results])
auc = roc_auc_score(slide_df['label'], slide_df['prob'])
print(f"OOF AUC (folds 0+1): {auc:.3f}", flush=True)

# ── Select slides to plot ──────────────────────────────────────────────────────
pos = slide_df[slide_df['label'] == 1].sort_values('prob', ascending=False)
neg = slide_df[slide_df['label'] == 0].sort_values('prob', ascending=True)
fp  = slide_df[(slide_df['label'] == 0) & (slide_df['prob'] > 0.5)].sort_values('prob', ascending=False)
fn  = slide_df[(slide_df['label'] == 1) & (slide_df['prob'] < 0.5)].sort_values('prob', ascending=True)

selected = {
    'ACR_pos':    pos.head(20)['slide'].tolist(),
    'ACR_neg':    neg.head(20)['slide'].tolist(),
    'FalsePos':   fp.head(5)['slide'].tolist(),
    'FalseNeg':   fn.head(5)['slide'].tolist(),
}
print(f"Selected — ACR+: {len(selected['ACR_pos'])}, ACR-: {len(selected['ACR_neg'])}, "
      f"FP: {len(selected['FalsePos'])}, FN: {len(selected['FalseNeg'])}", flush=True)

result_map = {r['slide']: r for r in all_results}

# ── Load tissue type from h5ad ─────────────────────────────────────────────────
print("Loading h5ad obs for tissue_type...", flush=True)
adata = ad.read_h5ad(H5AD, backed='r')
obs = adata.obs[['slide_name_clean', 'chunk_left', 'chunk_top', 'tissue_type']].copy()
obs['chunk_left'] = obs['chunk_left'].astype(np.float32)
obs['chunk_top']  = obs['chunk_top'].astype(np.float32)
print(f"  Loaded {len(obs):,} patches with tissue_type", flush=True)

all_slides_needed = set(sum(selected.values(), []))
obs_needed = obs[obs['slide_name_clean'].isin(all_slides_needed)]
slide_tissue = {}
for slide, grp in obs_needed.groupby('slide_name_clean'):
    # dict: (x, y) → tissue_type — use rounded coords for matching
    coord_map = {(round(float(r.chunk_left)), round(float(r.chunk_top))): r.tissue_type
                 for _, r in grp.iterrows()}
    slide_tissue[slide] = coord_map
print(f"  Built tissue maps for {len(slide_tissue)} slides", flush=True)

# ── Plotting function ──────────────────────────────────────────────────────────
def plot_slide_pair(ax_tissue, ax_attn, res, slide_tissue_map):
    coords_list = res['coords']
    attn_list   = res['attn']

    all_x, all_y, all_tissue, all_attn = [], [], [], []
    for coord_arr, attn_arr in zip(coords_list, attn_list):
        for (x, y), a in zip(coord_arr, attn_arr):
            all_x.append(x)
            all_y.append(-y)
            key = (round(float(x)), round(float(y)))
            tt  = slide_tissue_map.get(key, 'Unknown')
            all_tissue.append(tt)
            all_attn.append(float(a))

    all_x = np.array(all_x); all_y = np.array(all_y)
    all_attn = np.array(all_attn)

    # Left: tissue type
    for tt, col in TISSUE_COLORS.items():
        mask = [t == tt for t in all_tissue]
        if any(mask):
            ax_tissue.scatter(all_x[mask], all_y[mask],
                              s=1.5, c=col, alpha=0.8, linewidths=0, rasterized=True)

    # Right: combined attention (log scale for visibility)
    attn_plot = np.log1p(all_attn)
    vmin, vmax = attn_plot.min(), attn_plot.max()
    if vmax > vmin:
        attn_norm = (attn_plot - vmin) / (vmax - vmin)
    else:
        attn_norm = np.zeros_like(attn_plot)

    # Sort so high-attention patches are on top
    order = np.argsort(attn_norm)
    ax_attn.scatter(all_x[order], all_y[order], s=1.5,
                    c=attn_norm[order], cmap='RdYlGn_r',
                    vmin=0, vmax=1, alpha=0.85, linewidths=0, rasterized=True)

    for ax in [ax_tissue, ax_attn]:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')


def make_figure(slide_names, category, n_cols=4):
    n = len(slide_names)
    if n == 0:
        return
    n_rows = int(np.ceil(n / n_cols))
    fig_w  = n_cols * 6     # 3 per panel × 2 panels
    fig_h  = n_rows * 3.5
    fig, axes = plt.subplots(n_rows, n_cols * 2,
                              figsize=(fig_w, fig_h),
                              gridspec_kw={'wspace': 0.05, 'hspace': 0.35})
    axes = np.array(axes).reshape(n_rows, n_cols * 2)

    for i, slide_name in enumerate(slide_names):
        row, col = divmod(i, n_cols)
        ax_t = axes[row, col * 2]
        ax_a = axes[row, col * 2 + 1]

        if slide_name not in result_map:
            ax_t.set_visible(False); ax_a.set_visible(False); continue

        res  = result_map[slide_name]
        stm  = slide_tissue.get(slide_name, {})

        plot_slide_pair(ax_t, ax_a, res, stm)

        lbl_str  = "ACR+" if res['label'] == 1 else "ACR−"
        pred_str = f"p={res['prob']:.2f}"
        short    = slide_name.split('/')[-1][:30]
        ax_t.set_title(f"{lbl_str} {pred_str}\n{short}", fontsize=7, pad=2)
        ax_a.set_title("Attn", fontsize=7, pad=2)

        # Label panels
        ax_t.set_ylabel("Tissue type", fontsize=6)
        ax_a.set_ylabel("Attention", fontsize=6)

    # Hide empty axes
    for i in range(n, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col * 2].set_visible(False)
        axes[row, col * 2 + 1].set_visible(False)

    # Tissue legend (shared)
    legend_patches = [mpatches.Patch(color=c, label=t)
                      for t, c in TISSUE_COLORS.items() if t != 'Unknown']
    fig.legend(handles=legend_patches, loc='lower center',
               ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    # Attention colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:, -1], fraction=0.02, pad=0.01, shrink=0.6)
    cbar.set_label('Attention (log-norm)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    plt.suptitle(f'{category} — Tissue type (left) | Attention (right)',
                 fontsize=11, y=1.005)
    out_path = VIS_DIR / f"{category}.png"
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


# ── Generate figures ───────────────────────────────────────────────────────────
make_figure(selected['ACR_pos'],  'ACR_positive_top20', n_cols=4)
make_figure(selected['ACR_neg'],  'ACR_negative_top20', n_cols=4)
make_figure(selected['FalsePos'], 'WrongPred_FalsePositive', n_cols=3)
make_figure(selected['FalseNeg'], 'WrongPred_FalseNegative', n_cols=3)

print("\nAll figures saved to:", VIS_DIR, flush=True)
PYEOF
