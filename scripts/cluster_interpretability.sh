#!/usr/bin/env bash
#SBATCH --job-name=cluster_interp
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=150G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/cluster_interp_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/cluster_interp_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
"""
Deep interpretability of SpatialABMIL2Level high-attention clusters.

Questions answered:
  1. What tissue types are enriched in high-attention vs low-attention clusters?
     (separately for ACR+ and ACR- slides)

  2. What tissue COMBINATIONS co-occur in high-attention clusters?
     Co-occurrence matrix: how often do tissue pairs appear together?
     Do ACR+ slides have different co-occurrence patterns vs ACR-?

  3. Within the highest-attention neighbourhood, which PATCHES get the most
     patch attention (alpha_k[i])? What tissue are they?

  4. Cascade attention: beta_k x alpha_k[i]. What tissue types are at the
     very top of this combined ranking?

  5. Cluster "archetypes": cluster tissue fraction vectors → UMAP → do distinct
     neighbourhood types emerge? Which archetypes are ACR+-specific?
"""
import torch, torch.nn as nn
import numpy as np, pandas as pd
import anndata as ad
from scipy import stats
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

FEAT_DIR = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/slide_cluster_feats")
OUTDIR   = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil")
VIS_DIR  = OUTDIR / "cluster_interpretability"
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
]
SHORT = {
    "Alveolar":                                       "Alveolar",
    "Alveolar with empty spaces":                     "Alv+empty",
    "Bronchial":                                      "Bronchial",
    "Cartilage":                                      "Cartilage",
    "Alveolar with hemorrhage and inflammation":      "Alv+haem",
    "Lymphocytoplasmic inflammation":                 "Lympho",
}
COLORS = {
    "Alveolar":           "#4CAF50",
    "Alv+empty":          "#2196F3",
    "Bronchial":          "#9C27B0",
    "Cartilage":          "#00BCD4",
    "Alv+haem":           "#FF9800",
    "Lympho":             "#F44336",
}
T = len(TISSUE_TYPES)
tt_idx = {tt: i for i, tt in enumerate(TISSUE_TYPES)}

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
    def __init__(self, paths): self.paths = paths
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
labels  = np.array(labels); patients = np.array(patients)
print(f"Slides: {len(all_paths)}  ACR+={labels.sum()}", flush=True)

cv = StratifiedGroupKFold(n_splits=N_SPLITS)
fold_splits = list(cv.split(all_paths, labels, patients))

all_results = []
for fold in [0, 1]:
    ckpt = OUTDIR / f"spatial_abmil_fold{fold}.pt"
    if not ckpt.exists():
        print(f"Fold {fold}: no checkpoint — skip", flush=True); continue
    model = SpatialABMIL2Level()
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
            na   = nbhd_attn.numpy()
            all_results.append({
                'fold':        fold,
                'slide':       slide_name,
                'label':       int(label.item()),
                'prob':        prob,
                'coords':      coords,
                'na':          na,
                'patch_attns': [pa.numpy() for pa in patch_attns],
            })

print(f"Inference done: {len(all_results)} slides", flush=True)
auc = roc_auc_score([r['label'] for r in all_results],
                    [r['prob']  for r in all_results])
print(f"OOF AUC: {auc:.3f}", flush=True)

# ── Load tissue types ──────────────────────────────────────────────────────────
print("Loading h5ad...", flush=True)
adata = ad.read_h5ad(H5AD, backed='r')
obs   = adata.obs[['slide_name_clean','chunk_left','chunk_top','tissue_type']].copy()
obs['chunk_left'] = obs['chunk_left'].astype(np.float32)
obs['chunk_top']  = obs['chunk_top'].astype(np.float32)
obs_needed = obs[obs['slide_name_clean'].isin({r['slide'] for r in all_results})]
slide_tissue = {}
for slide, grp in obs_needed.groupby('slide_name_clean'):
    arr = grp[['chunk_left','chunk_top','tissue_type']].values
    slide_tissue[slide] = {(round(float(r[0])),round(float(r[1]))): str(r[2]) for r in arr}
print(f"  Tissue maps for {len(slide_tissue)} slides", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# BUILD CLUSTER RECORDS
# Each record = one cluster k in one slide.
# Contains: tissue fraction vector, nbhd_attn, patch-level attns mapped to tissue.
# ══════════════════════════════════════════════════════════════════════════════
records = []
for res in all_results:
    slide = res['slide']
    tmap  = slide_tissue.get(slide, {})
    na    = res['na']        # (K,)
    K     = len(na)
    # Rank: 1 = highest attention cluster in this slide
    ranks = K + 1 - pd.Series(na).rank(method='first').values.astype(int)

    for k, (coord_arr, pa, nw, rank) in enumerate(
            zip(res['coords'], res['patch_attns'], na, ranks)):

        # Tissue fraction vector for this cluster
        counts = np.zeros(T)
        patch_tissue = []
        for (x, y) in coord_arr:
            tt = tmap.get((round(float(x)), round(float(y))), None)
            patch_tissue.append(tt)
            if tt in tt_idx:
                counts[tt_idx[tt]] += 1

        n_total = counts.sum()
        fracs   = counts / n_total if n_total > 0 else counts

        # Patch attention by tissue type
        # For each tissue, mean patch attention weight of patches of that type
        pa_by_tissue = {tt: [] for tt in TISSUE_TYPES}
        for pt, pv in zip(patch_tissue, pa):
            if pt in pa_by_tissue:
                pa_by_tissue[pt].append(float(pv))

        mean_pa_by_tissue = {tt: np.mean(v) if v else np.nan
                              for tt, v in pa_by_tissue.items()}

        # Top-5 patches by attention: what tissue are they?
        top5_idx = np.argsort(pa)[-5:][::-1]
        top5_tissue = [patch_tissue[i] for i in top5_idx if i < len(patch_tissue)]

        rec = {
            'slide':       slide,
            'label':       res['label'],
            'prob':        res['prob'],
            'cluster_k':  k,
            'K':           K,
            'nbhd_attn':  float(nw),
            'attn_rank':  int(rank),
            'is_top1':    rank == 1,
            'is_top3':    rank <= 3,
            'is_top_half': rank <= K // 2,
            'n_patches':  int(n_total),
        }
        for i, tt in enumerate(TISSUE_TYPES):
            s = SHORT[tt]
            rec[f'frac_{s}']    = fracs[i]
            rec[f'mean_pa_{s}'] = mean_pa_by_tissue[tt]

        rec['top5_tissues'] = ','.join([t for t in top5_tissue if t])
        records.append(rec)

df = pd.DataFrame(records)
df.to_csv(VIS_DIR / "cluster_records.csv", index=False)
print(f"Built {len(df)} cluster records from {df['slide'].nunique()} slides", flush=True)

frac_cols = [f'frac_{SHORT[tt]}'    for tt in TISSUE_TYPES]
pa_cols   = [f'mean_pa_{SHORT[tt]}' for tt in TISSUE_TYPES]
shorts    = [SHORT[tt] for tt in TISSUE_TYPES]

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: What tissues are enriched in HIGH vs LOW attention clusters?
#             Separately for ACR+ and ACR- slides.
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Q1: Tissue enrichment in high-attention clusters ──", flush=True)

enrich_rows = []
for label_val, label_name in [(1,'ACR+'), (0,'ACR-')]:
    sub  = df[df['label'] == label_val]
    high = sub[sub['is_top3']]
    low  = sub[~sub['is_top_half']]
    for tt in TISSUE_TYPES:
        col = f'frac_{SHORT[tt]}'
        hv  = high[col].values
        lv  = low[col].values
        if len(hv) < 3 or len(lv) < 3: continue
        _, p = stats.mannwhitneyu(hv, lv, alternative='two-sided')
        fc   = hv.mean() / lv.mean() if lv.mean() > 0 else np.nan
        enrich_rows.append({
            'group': label_name, 'tissue': tt, 'short': SHORT[tt],
            'mean_high': hv.mean(), 'mean_low': lv.mean(),
            'log2fc': np.log2(fc) if fc > 0 else np.nan, 'p': p
        })
        print(f"  {label_name}  {tt:45s}  high={hv.mean():.4f}  low={lv.mean():.4f}  "
              f"log2FC={np.log2(fc) if fc>0 else 'nan':+.2f}  p={p:.4f}", flush=True)

enrich_df = pd.DataFrame(enrich_rows)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Tissue CO-OCCURRENCE in high-attention clusters
#             For each pair (A, B): fraction of high-attn clusters where
#             BOTH frac_A > 10% AND frac_B > 10%.
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Q2: Tissue co-occurrence in high-attention clusters ──", flush=True)
THRESH = 0.10  # at least 10% of cluster patches from each tissue type

cooc_results = {}
for label_val, label_name in [(1,'ACR+'), (0,'ACR-')]:
    sub  = df[(df['label'] == label_val) & df['is_top3']]
    mat  = np.zeros((T, T))
    for i, tti in enumerate(TISSUE_TYPES):
        for j, ttj in enumerate(TISSUE_TYPES):
            ci = f'frac_{SHORT[tti]}'; cj = f'frac_{SHORT[ttj]}'
            if i == j:
                mat[i, j] = (sub[ci] > THRESH).mean()
            else:
                mat[i, j] = ((sub[ci] > THRESH) & (sub[cj] > THRESH)).mean()
    cooc_results[label_name] = mat
    print(f"\n  {label_name} co-occurrence (fraction of top-3 clusters with both >10%):",
          flush=True)
    for i, tti in enumerate(TISSUE_TYPES):
        for j, ttj in enumerate(TISSUE_TYPES):
            if j > i and mat[i,j] > 0.05:
                print(f"    {SHORT[tti]:15s} + {SHORT[ttj]:15s}  {mat[i,j]:.3f}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Within high-attention clusters, which tissue gets highest PATCH attn?
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Q3: Patch attention by tissue within high-attention clusters ──", flush=True)
pa_rows = []
for label_val, label_name in [(1,'ACR+'), (0,'ACR-')]:
    top1 = df[(df['label'] == label_val) & df['is_top1']]
    print(f"\n  {label_name} — top-1 cluster patch attention by tissue:", flush=True)
    for tt in TISSUE_TYPES:
        col = f'mean_pa_{SHORT[tt]}'
        v   = top1[col].dropna().values
        if len(v) > 0:
            print(f"    {tt:45s}  mean_patch_attn={v.mean():.5f}  "
                  f"median={np.median(v):.5f}  n={len(v)}", flush=True)
            pa_rows.append({'group': label_name, 'tissue': tt,
                            'short': SHORT[tt], 'mean_pa': v.mean(),
                            'median_pa': np.median(v), 'n': len(v)})

pa_df = pd.DataFrame(pa_rows)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Top-5 patches in top-1 clusters — tissue type distribution
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Q4: What tissue types are the top-5 patches (by attn) in top-1 clusters? ──",
      flush=True)
top1_all = df[df['is_top1']]
for label_val, label_name in [(1,'ACR+'), (0,'ACR-')]:
    sub = top1_all[top1_all['label'] == label_val]
    all_top5 = []
    for ts in sub['top5_tissues'].dropna():
        all_top5.extend([t for t in ts.split(',') if t in TISSUE_TYPES])
    counts = pd.Series(all_top5).value_counts(normalize=True) * 100
    print(f"\n  {label_name} — tissue of top-5 most-attended patches in top-1 cluster:",
          flush=True)
    for tt, pct in counts.items():
        print(f"    {tt:45s}  {pct:.1f}%", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Plot 1: Tissue enrichment in high vs low attention clusters ───────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig.suptitle("Tissue enrichment: top-3 attention clusters vs bottom half\n"
             "log2FC > 0 means enriched in HIGH attention clusters", fontsize=12)

for ax, label_name in zip(axes, ['ACR+', 'ACR-']):
    sub = enrich_df[enrich_df['group'] == label_name].sort_values('log2fc', ascending=True)
    colors = ['#E53030' if v > 0 else '#4477CC' for v in sub['log2fc']]
    bars = ax.barh(sub['short'], sub['log2fc'], color=colors, alpha=0.85, edgecolor='white')
    for i, row in sub.reset_index(drop=True).iterrows():
        star = ('***' if row['p'] < 0.001 else '**' if row['p'] < 0.01
                else '*' if row['p'] < 0.05 else '')
        if star:
            xpos = row['log2fc'] + (0.05 if row['log2fc'] >= 0 else -0.05)
            ax.text(xpos, i, star, va='center',
                    ha='left' if row['log2fc'] >= 0 else 'right', fontsize=10)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(label_name, fontsize=11)
    ax.set_xlabel("log2FC (high-attn / low-attn clusters)", fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(VIS_DIR / "tissue_enrichment_high_vs_low_attn.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nSaved tissue_enrichment_high_vs_low_attn.png", flush=True)

# ── Plot 2: Co-occurrence heatmaps (ACR+ vs ACR-) ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Tissue co-occurrence in top-3 attention clusters (both tissues > {THRESH*100:.0f}%)",
             fontsize=12)

cmap_cooc = LinearSegmentedColormap.from_list('w2r', ['white','#E53030'])
for ax, label_name in zip(axes, ['ACR+', 'ACR-']):
    mat = cooc_results[label_name]
    im  = ax.imshow(mat, cmap=cmap_cooc, vmin=0, vmax=mat.max())
    ax.set_xticks(range(T)); ax.set_xticklabels(shorts, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(T)); ax.set_yticklabels(shorts, fontsize=8)
    for i in range(T):
        for j in range(T):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=7,
                    color='white' if v > mat.max()*0.6 else 'black')
    ax.set_title(f"{label_name} — co-occurrence", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Fraction of clusters')

plt.tight_layout()
fig.savefig(VIS_DIR / "tissue_cooccurrence_heatmap.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved tissue_cooccurrence_heatmap.png", flush=True)

# ── Plot 3: Co-occurrence DIFFERENCE (ACR+ minus ACR-) ────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
diff_mat = cooc_results['ACR+'] - cooc_results['ACR-']
cmap_diff = LinearSegmentedColormap.from_list('b2r', ['#4477CC','white','#E53030'])
im = ax.imshow(diff_mat, cmap=cmap_diff,
               vmin=-abs(diff_mat).max(), vmax=abs(diff_mat).max())
ax.set_xticks(range(T)); ax.set_xticklabels(shorts, rotation=40, ha='right', fontsize=9)
ax.set_yticks(range(T)); ax.set_yticklabels(shorts, fontsize=9)
for i in range(T):
    for j in range(T):
        v = diff_mat[i, j]
        ax.text(j, i, f"{v:+.2f}", ha='center', va='center', fontsize=8,
                color='white' if abs(v) > abs(diff_mat).max()*0.6 else 'black')
plt.colorbar(im, ax=ax, label='ACR+ minus ACR- co-occurrence')
ax.set_title("Co-occurrence difference in high-attention clusters\n"
             "Red = more frequent co-occurrence in ACR+\n"
             "Blue = more frequent co-occurrence in ACR-", fontsize=10)
plt.tight_layout()
fig.savefig(VIS_DIR / "cooccurrence_diff_ACRpos_vs_neg.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved cooccurrence_diff_ACRpos_vs_neg.png", flush=True)

# ── Plot 4: Patch attention by tissue type within top clusters ────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle("Mean patch attention by tissue type within top-1 attention cluster\n"
             "(which specific cell types does the model focus on inside the key neighbourhood?)",
             fontsize=11)

for ax, label_name in zip(axes, ['ACR+', 'ACR-']):
    sub = pa_df[pa_df['group'] == label_name].sort_values('mean_pa', ascending=True)
    bar_colors = [COLORS.get(r['short'], '#AAAAAA') for _, r in sub.iterrows()]
    ax.barh(sub['short'], sub['mean_pa'], color=bar_colors, alpha=0.85, edgecolor='white')
    ax.set_xlabel("Mean patch attention weight\n(within top-1 neighbourhood)", fontsize=9)
    ax.set_title(label_name, fontsize=11)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(row['mean_pa'] + 0.0001, i,
                f"n={row['n']}", va='center', fontsize=7)

plt.tight_layout()
fig.savefig(VIS_DIR / "patch_attn_by_tissue_in_top_cluster.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved patch_attn_by_tissue_in_top_cluster.png", flush=True)

# ── Plot 5: Tissue fraction profile of top-1 clusters, ACR+ vs ACR- ──────────
fig, ax = plt.subplots(figsize=(10, 5))
top1_pos = df[(df['label']==1) & df['is_top1']][frac_cols].mean()
top1_neg = df[(df['label']==0) & df['is_top1']][frac_cols].mean()
top3_pos = df[(df['label']==1) & df['is_top3']][frac_cols].mean()
top3_neg = df[(df['label']==0) & df['is_top3']][frac_cols].mean()
all_pos  = df[df['label']==1][frac_cols].mean()
all_neg  = df[df['label']==0][frac_cols].mean()

x  = np.arange(T)
w  = 0.13
profiles = [
    (top1_pos, '#E53030', 'ACR+ top-1 cluster'),
    (top1_neg, '#4477CC', 'ACR- top-1 cluster'),
    (top3_pos, '#FF7070', 'ACR+ top-3 clusters'),
    (top3_neg, '#7799EE', 'ACR- top-3 clusters'),
    (all_pos,  '#FFBBBB', 'ACR+ all clusters'),
    (all_neg,  '#BBCCFF', 'ACR- all clusters'),
]
for i, (vals, color, label) in enumerate(profiles):
    ax.bar(x + (i-2.5)*w, vals.values, w, label=label, color=color, alpha=0.9)

ax.set_xticks(x); ax.set_xticklabels(shorts, rotation=35, ha='right', fontsize=9)
ax.set_ylabel("Mean tissue fraction", fontsize=9)
ax.set_title("Tissue composition: top-attention clusters vs all clusters\n"
             "(ACR+ in red tones, ACR- in blue tones, darker = higher attention)", fontsize=10)
ax.legend(fontsize=7.5, frameon=False, ncol=2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(VIS_DIR / "top_cluster_tissue_profile.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved top_cluster_tissue_profile.png", flush=True)

# ── Plot 6: UMAP of cluster tissue fraction vectors ──────────────────────────
try:
    import umap
    print("\nRunning UMAP on cluster tissue fraction vectors...", flush=True)
    frac_mat  = df[frac_cols].fillna(0).values         # (N_clusters, T)
    attn_vals = df['nbhd_attn'].values
    labels_v  = df['label'].values
    is_top3   = df['is_top3'].values

    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        metric='cosine', random_state=42)
    emb = reducer.fit_transform(frac_mat)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("UMAP of cluster tissue fraction vectors\n"
                 "(each point = one cluster; shape=triangle if top-3 attention)",
                 fontsize=11)

    # Panel 1: coloured by ACR label
    ax = axes[0]
    for lv, lname, color in [(1,'ACR+','#E53030'), (0,'ACR-','#4477CC')]:
        m = labels_v == lv
        ax.scatter(emb[m & ~is_top3, 0], emb[m & ~is_top3, 1],
                   s=5, c=color, alpha=0.3, label=f'{lname} (low attn)', rasterized=True)
        ax.scatter(emb[m & is_top3,  0], emb[m & is_top3,  1],
                   s=25, c=color, alpha=0.9, marker='^',
                   label=f'{lname} (top-3 attn)', rasterized=True)
    ax.set_title("ACR label"); ax.legend(fontsize=7, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])

    # Panel 2: coloured by neighbourhood attention weight
    ax2 = axes[1]
    attn_norm = (attn_vals - attn_vals.min()) / (attn_vals.max() - attn_vals.min())
    sc = ax2.scatter(emb[:, 0], emb[:, 1], s=5, c=attn_norm,
                     cmap='RdYlGn_r', alpha=0.5, rasterized=True)
    plt.colorbar(sc, ax=ax2, label='Neighbourhood attn (norm)', shrink=0.8)
    ax2.set_title("Neighbourhood attention weight")
    ax2.set_xticks([]); ax2.set_yticks([])

    # Panel 3: coloured by dominant tissue type
    ax3 = axes[2]
    dominant = df[frac_cols].idxmax(axis=1).map(
        lambda c: SHORT[TISSUE_TYPES[[f'frac_{SHORT[tt]}' for tt in TISSUE_TYPES].index(c)]]
        if c in frac_cols else 'Unknown'
    )
    for short_name, color in COLORS.items():
        m = dominant == short_name
        if m.any():
            ax3.scatter(emb[m, 0], emb[m, 1], s=5, c=color, alpha=0.5,
                        label=short_name, rasterized=True)
    ax3.legend(fontsize=6, markerscale=2)
    ax3.set_title("Dominant tissue type")
    ax3.set_xticks([]); ax3.set_yticks([])

    plt.tight_layout()
    fig.savefig(VIS_DIR / "cluster_tissue_umap.png", dpi=140, bbox_inches='tight')
    plt.close(fig)
    print("Saved cluster_tissue_umap.png", flush=True)
except ImportError:
    print("umap not available — skipping UMAP plot", flush=True)

# ── Summary printout ──────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("SUMMARY", flush=True)
print("="*60, flush=True)
print("\nQ1 — Tissues enriched in HIGH-attention clusters:", flush=True)
for _, row in enrich_df.sort_values(['group','log2fc'], ascending=[True,False]).iterrows():
    if row['p'] < 0.05:
        dir_ = "↑ enriched" if row['log2fc'] > 0 else "↓ depleted"
        print(f"  {row['group']}  {row['short']:15s}  log2FC={row['log2fc']:+.2f}  "
              f"p={row['p']:.4f}  {dir_}", flush=True)

print("\nQ2 — Most common tissue co-occurrences in top-3 clusters:", flush=True)
for label_name in ['ACR+','ACR-']:
    mat = cooc_results[label_name]
    print(f"  {label_name}:", flush=True)
    pairs = [(mat[i,j], shorts[i], shorts[j])
             for i in range(T) for j in range(i+1,T) if mat[i,j] > 0.05]
    for v, a, b in sorted(pairs, reverse=True)[:5]:
        print(f"    {a:15s} + {b:15s}  {v:.3f}", flush=True)

print(f"\nAll outputs in: {VIS_DIR}", flush=True)
PYEOF
