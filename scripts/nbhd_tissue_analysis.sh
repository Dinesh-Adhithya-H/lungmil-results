#!/usr/bin/env bash
#SBATCH --job-name=nbhd_tissue
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=150G
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/nbhd_tissue_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/nbhd_tissue_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
"""
Which neighbourhood compositions are predictive of ACR+?

Each cluster is represented as a tissue FRACTION VECTOR
(e.g. 60% Alveolar, 30% Lymphocytic, 10% Bronchial) -- no single-type assignment.

Analyses:
  1. Attention-weighted mean tissue fraction: ACR+ vs ACR-
     For each slide, compute the attention-weighted tissue profile:
       profile_slide = sum_k [ beta_k * tissue_fraction_vector_k ]
     Then compare these slide-level profiles between ACR+ and ACR-.

  2. Per-tissue-type fraction vs neighbourhood attention correlation
     (separately for ACR+ and ACR- slides)

  3. Top-attention cluster profiles: what tissue COMBINATIONS appear in
     the highest-attention neighbourhoods of ACR+ slides?

  4. Heatmap of cluster tissue compositions, sorted by attention weight,
     separately for ACR+ and ACR- slides.
"""
import torch, torch.nn as nn
import numpy as np, pandas as pd
import anndata as ad
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

FEAT_DIR = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/slide_cluster_feats")
OUTDIR   = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil")
VIS_DIR  = OUTDIR / "nbhd_tissue_analysis"
VIS_DIR.mkdir(exist_ok=True)

H5AD        = "/lustre/groups/aih/dinesh.haridoss/datasets/adata_v3.h5ad"
IN_DIM, HIDDEN, DROPOUT = 1024, 256, 0.25
MAX_PATCHES = 512
N_SPLITS    = 5

TISSUE_TYPES = [
    "Alveolar",
    "Alveolar with empty spaces",
    "Bronchial",
    "Cartilage",
    "Alveolar with hemorrhage and inflammation",
    "Lymphocytoplasmic inflammation",
    "Unknown",
]
SHORT = {
    "Alveolar":                                       "Alveolar",
    "Alveolar with empty spaces":                     "Alv+empty",
    "Bronchial":                                      "Bronchial",
    "Cartilage":                                      "Cartilage",
    "Alveolar with hemorrhage and inflammation":      "Alv+haem",
    "Lymphocytoplasmic inflammation":                 "Lympho.inflam",
    "Unknown":                                        "Unknown",
}
TISSUE_COLORS = {
    "Alveolar":                                       "#4CAF50",
    "Alveolar with empty spaces":                     "#2196F3",
    "Bronchial":                                      "#9C27B0",
    "Cartilage":                                      "#00BCD4",
    "Alveolar with hemorrhage and inflammation":      "#FF9800",
    "Lymphocytoplasmic inflammation":                 "#F44336",
    "Unknown":                                        "#AAAAAA",
}

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

# ── Inference ──────────────────────────────────────────────────────────────────
all_paths = sorted(FEAT_DIR.glob("*.pt"))
labels, patients = [], []
for p in all_paths:
    d = torch.load(p, map_location='cpu', weights_only=False)
    labels.append(d['label']); patients.append(d['patient'])
labels, patients = np.array(labels), np.array(patients)
print(f"Total slides: {len(all_paths)}  ACR+={labels.sum()}", flush=True)

cv = StratifiedGroupKFold(n_splits=N_SPLITS)
fold_splits = list(cv.split(all_paths, labels, patients))

all_results = []
for fold in [0, 1]:
    ckpt = OUTDIR / f"spatial_abmil_fold{fold}.pt"
    if not ckpt.exists():
        print(f"Fold {fold}: skip", flush=True); continue
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
            all_results.append({
                'slide':  slide_name,
                'label':  int(label.item()),
                'prob':   torch.sigmoid(logit).item(),
                'coords': coords,
                'na':     nbhd_attn.numpy(),
            })

print(f"Inference done: {len(all_results)} slides", flush=True)
slide_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('coords','na')}
                          for r in all_results])
auc = roc_auc_score(slide_df['label'], slide_df['prob'])
print(f"OOF AUC: {auc:.3f}", flush=True)

# ── Load tissue types ──────────────────────────────────────────────────────────
print("Loading h5ad...", flush=True)
adata = ad.read_h5ad(H5AD, backed='r')
obs   = adata.obs[['slide_name_clean', 'chunk_left', 'chunk_top', 'tissue_type']].copy()
obs['chunk_left'] = obs['chunk_left'].astype(np.float32)
obs['chunk_top']  = obs['chunk_top'].astype(np.float32)

all_slides    = set(r['slide'] for r in all_results)
obs_needed    = obs[obs['slide_name_clean'].isin(all_slides)]
slide_tissue  = {}
for slide, grp in obs_needed.groupby('slide_name_clean'):
    arr = grp[['chunk_left', 'chunk_top', 'tissue_type']].values
    slide_tissue[slide] = {(round(float(r[0])), round(float(r[1]))): str(r[2]) for r in arr}
print(f"  Tissue maps for {len(slide_tissue)} slides", flush=True)

# ── Build cluster tissue fraction vectors ──────────────────────────────────────
# Each cluster → fraction of each tissue type (sums to 1)
T = len(TISSUE_TYPES)
tt_idx = {tt: i for i, tt in enumerate(TISSUE_TYPES)}

cluster_records = []   # one row per cluster
slide_profiles  = []   # one row per slide: attention-weighted tissue profile

for res in all_results:
    slide = res['slide']
    tmap  = slide_tissue.get(slide, {})
    na    = res['na']          # (K,) neighbourhood attn weights, sum=1
    K     = len(na)

    # Attention-weighted slide profile (sum over clusters of beta_k * fraction_k)
    slide_profile = np.zeros(T)

    for k, (coord_arr, nw) in enumerate(zip(res['coords'], na)):
        # Count tissue types in cluster k
        counts = np.zeros(T)
        for (x, y) in coord_arr:
            tt = tmap.get((round(float(x)), round(float(y))), 'Unknown')
            counts[tt_idx.get(tt, tt_idx['Unknown'])] += 1

        n_total = counts.sum()
        fracs   = counts / n_total if n_total > 0 else counts

        slide_profile += float(nw) * fracs   # attention-weighted contribution

        cluster_records.append({
            'slide':      slide,
            'label':      res['label'],
            'prob':       res['prob'],
            'cluster_k':  k,
            'K':          K,
            'nbhd_attn':  float(nw),
            'n_patches':  int(n_total),
            **{f'frac_{SHORT[tt]}': fracs[i] for i, tt in enumerate(TISSUE_TYPES)},
        })

    slide_profiles.append({
        'slide': slide,
        'label': res['label'],
        'prob':  res['prob'],
        **{f'wtd_{SHORT[tt]}': slide_profile[i] for i, tt in enumerate(TISSUE_TYPES)},
    })

cluster_df = pd.DataFrame(cluster_records)
slide_prof_df = pd.DataFrame(slide_profiles)
cluster_df.to_csv(VIS_DIR / "cluster_tissue_fractions.csv", index=False)
slide_prof_df.to_csv(VIS_DIR / "slide_attn_weighted_profiles.csv", index=False)
print(f"Saved CSVs: {len(cluster_df)} clusters, {len(slide_prof_df)} slides", flush=True)

# ── Analysis 1: attention-weighted tissue profile ACR+ vs ACR- ─────────────────
print("\n── Attention-weighted tissue profile (slide level) ──", flush=True)
frac_cols = [f'wtd_{SHORT[tt]}' for tt in TISSUE_TYPES]
pos_profiles = slide_prof_df[slide_prof_df['label'] == 1][frac_cols].values
neg_profiles = slide_prof_df[slide_prof_df['label'] == 0][frac_cols].values

print(f"  {'Tissue':40s}  {'ACR+ mean':>10}  {'ACR- mean':>10}  {'diff':>8}  p")
stats_rows = []
for i, tt in enumerate(TISSUE_TYPES):
    pos_v = pos_profiles[:, i]
    neg_v = neg_profiles[:, i]
    _, p  = stats.mannwhitneyu(pos_v, neg_v, alternative='two-sided')
    diff  = pos_v.mean() - neg_v.mean()
    stats_rows.append({'tissue': tt, 'short': SHORT[tt],
                        'mean_pos': pos_v.mean(), 'mean_neg': neg_v.mean(),
                        'diff': diff, 'p': p})
    print(f"  {tt:40s}  {pos_v.mean():10.5f}  {neg_v.mean():10.5f}  {diff:+8.5f}  {p:.4f}",
          flush=True)

# ── Analysis 2: cluster-level correlation — tissue fraction vs nbhd_attn ───────
print("\n── Tissue fraction vs nbhd_attn (Spearman r, per ACR group) ──", flush=True)
frac_cols_cl = [f'frac_{SHORT[tt]}' for tt in TISSUE_TYPES]
corr_rows = []
for label_val, label_name in [(1, 'ACR+'), (0, 'ACR-')]:
    sub = cluster_df[cluster_df['label'] == label_val]
    for tt in TISSUE_TYPES:
        col = f'frac_{SHORT[tt]}'
        r, p = stats.spearmanr(sub[col], sub['nbhd_attn'])
        corr_rows.append({'tissue': tt, 'short': SHORT[tt],
                          'label': label_name, 'spearman_r': r, 'p': p})
        print(f"  {label_name}  {tt:40s}  r={r:+.3f}  p={p:.4f}", flush=True)

corr_df = pd.DataFrame(corr_rows)

# ── Plots ──────────────────────────────────────────────────────────────────────

# ── Plot 1: attention-weighted profile comparison ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Attention-weighted tissue composition: ACR+ vs ACR-", fontsize=12)

ax = axes[0]
x  = np.arange(len(TISSUE_TYPES))
w  = 0.35
ax.bar(x - w/2, [r['mean_pos'] for r in stats_rows], w,
       label='ACR+', color='#E53030', alpha=0.8)
ax.bar(x + w/2, [r['mean_neg'] for r in stats_rows], w,
       label='ACR-',  color='#4477CC', alpha=0.8)
for i, r in enumerate(stats_rows):
    star = '***' if r['p'] < 0.001 else ('**' if r['p'] < 0.01 else ('*' if r['p'] < 0.05 else ''))
    if star:
        ymax = max(r['mean_pos'], r['mean_neg'])
        ax.text(x[i], ymax + 0.001, star, ha='center', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([SHORT[tt] for tt in TISSUE_TYPES], rotation=35, ha='right', fontsize=8)
ax.set_ylabel("Attention-weighted fraction\n(Σ_k β_k × tissue_frac_k)", fontsize=8)
ax.set_title("How much does each tissue type contribute\nto the model's attended regions?", fontsize=9)
ax.legend(fontsize=9); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Right: difference plot (ACR+ minus ACR-)
ax2 = axes[1]
diffs  = [r['diff'] for r in stats_rows]
colors = ['#E53030' if d > 0 else '#4477CC' for d in diffs]
ax2.barh([SHORT[tt] for tt in TISSUE_TYPES], diffs, color=colors, alpha=0.85)
for i, r in enumerate(stats_rows):
    star = '***' if r['p'] < 0.001 else ('**' if r['p'] < 0.01 else ('*' if r['p'] < 0.05 else ''))
    if star:
        xpos = r['diff'] + (0.001 if r['diff'] >= 0 else -0.001)
        ha   = 'left' if r['diff'] >= 0 else 'right'
        ax2.text(xpos, i, star, va='center', ha=ha, fontsize=9)
ax2.axvline(0, color='black', linewidth=0.8)
ax2.set_xlabel("Difference in attention-weighted fraction\n(ACR+ minus ACR-)", fontsize=8)
ax2.set_title("Tissue types the model attends more in ACR+\nvs ACR- slides", fontsize=9)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(VIS_DIR / "attn_weighted_profile.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved attn_weighted_profile.png", flush=True)

# ── Plot 2: Spearman correlation heatmap ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
pivot = corr_df.pivot(index='short', columns='label', values='spearman_r')
pivot_p = corr_df.pivot(index='short', columns='label', values='p')
# Sort by ACR+ correlation descending
pivot = pivot.loc[pivot['ACR+'].sort_values(ascending=False).index]
pivot_p = pivot_p.loc[pivot.index]

cmap = LinearSegmentedColormap.from_list('rw_b', ['#4477CC', 'white', '#E53030'])
im = ax.imshow(pivot.values, cmap=cmap, vmin=-0.3, vmax=0.3, aspect='auto')
ax.set_xticks([0, 1]); ax.set_xticklabels(['ACR+', 'ACR-'], fontsize=10)
ax.set_yticks(range(len(pivot))); ax.set_yticklabels(pivot.index, fontsize=9)
# Annotate with r value and stars
for i in range(len(pivot)):
    for j, label_name in enumerate(['ACR+', 'ACR-']):
        r_val = pivot.iloc[i, j]
        p_val = pivot_p.iloc[i, j]
        star  = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
        ax.text(j, i, f"{r_val:+.2f}{star}", ha='center', va='center', fontsize=8,
                color='white' if abs(r_val) > 0.15 else 'black')
plt.colorbar(im, ax=ax, label='Spearman r', shrink=0.8)
ax.set_title("Correlation: tissue fraction vs neighbourhood attention weight\n"
             "(sorted by ACR+ correlation, red=positive/high-attn tissues)", fontsize=10)
plt.tight_layout()
fig.savefig(VIS_DIR / "tissue_attn_correlation.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved tissue_attn_correlation.png", flush=True)

# ── Plot 3: Tissue composition heatmap of top-attention clusters ───────────────
# Take top-3 attention clusters per slide, show their tissue fraction profile
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=False)
fig.suptitle("Tissue composition of highest-attention neighbourhoods", fontsize=12)

for ax, (label_val, label_name) in zip(axes, [(1, 'ACR+'), (0, 'ACR-')]):
    sub = cluster_df[cluster_df['label'] == label_val].copy()
    # Top-3 clusters per slide by nbhd_attn
    sub['rank'] = sub.groupby('slide')['nbhd_attn'].rank(ascending=False)
    top3 = sub[sub['rank'] <= 3].copy()

    frac_mat = top3[[f'frac_{SHORT[tt]}' for tt in TISSUE_TYPES]].values
    # Sort rows by Lympho fraction (most informative expected)
    sort_idx = np.argsort(-frac_mat[:, TISSUE_TYPES.index("Lymphocytoplasmic inflammation")])
    frac_mat = frac_mat[sort_idx]

    # Subsample if too many rows
    if len(frac_mat) > 200:
        step = len(frac_mat) // 200
        frac_mat = frac_mat[::step]

    cmap2 = LinearSegmentedColormap.from_list('wh', ['white', '#1a1a2e'])
    im = ax.imshow(frac_mat.T, cmap=cmap2, vmin=0, vmax=1, aspect='auto')
    ax.set_yticks(range(T))
    ax.set_yticklabels([SHORT[tt] for tt in TISSUE_TYPES], fontsize=8)
    ax.set_xlabel("Clusters (sorted by Lympho fraction)", fontsize=8)
    ax.set_title(f"{label_name} — top-3 attention clusters\n"
                 f"({len(top3)} clusters from {top3['slide'].nunique()} slides)", fontsize=9)

plt.colorbar(im, ax=axes[1], label='Tissue fraction', shrink=0.7)
plt.tight_layout()
fig.savefig(VIS_DIR / "top_cluster_composition_heatmap.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved top_cluster_composition_heatmap.png", flush=True)

# ── Plot 4: Scatter — Lympho fraction vs nbhd_attn, coloured by ACR label ─────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Which tissue fractions drive high neighbourhood attention?", fontsize=12)

# Left: Lympho fraction vs attn
ax = axes[0]
for label_val, color, name in [(1,'#E53030','ACR+'), (0,'#4477CC','ACR-')]:
    sub = cluster_df[cluster_df['label'] == label_val]
    ax.scatter(sub['frac_Lympho.inflam'], sub['nbhd_attn'],
               s=4, alpha=0.3, color=color, label=name, rasterized=True)
ax.set_xlabel("Lymphocytoplasmic inflammation fraction in cluster", fontsize=9)
ax.set_ylabel("Neighbourhood attention weight", fontsize=9)
ax.set_title("Lympho fraction vs attention", fontsize=10)
ax.legend(fontsize=9, markerscale=3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Right: Alv+haem fraction vs attn
ax2 = axes[1]
for label_val, color, name in [(1,'#E53030','ACR+'), (0,'#4477CC','ACR-')]:
    sub = cluster_df[cluster_df['label'] == label_val]
    ax2.scatter(sub['frac_Alv+haem'], sub['nbhd_attn'],
                s=4, alpha=0.3, color=color, label=name, rasterized=True)
ax2.set_xlabel("Alv+haemorrhage+inflam fraction in cluster", fontsize=9)
ax2.set_ylabel("Neighbourhood attention weight", fontsize=9)
ax2.set_title("Haemorrhage+inflam fraction vs attention", fontsize=10)
ax2.legend(fontsize=9, markerscale=3)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(VIS_DIR / "tissue_frac_vs_attn_scatter.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved tissue_frac_vs_attn_scatter.png", flush=True)

print(f"\nAll outputs saved to: {VIS_DIR}", flush=True)
PYEOF
