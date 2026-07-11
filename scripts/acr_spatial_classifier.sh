#!/usr/bin/env bash
#SBATCH --job-name=acr_spatial_clf
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/acr_spatial/job_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/acr_spatial/job_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

OUTDIR="/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/acr_spatial"
mkdir -p "$OUTDIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, os, json
warnings.filterwarnings('ignore')

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import xgboost as xgb

OUTDIR = "/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/plots/acr_spatial"
H5AD   = "/lustre/groups/aih/dinesh.haridoss/datasets/adata_v3.h5ad"
K_NN   = 20   # spatial neighbors per patch

TISSUE_TYPES = [
    "Alveolar",
    "Alveolar with empty spaces",
    "Alveolar with hemorrhage and inflammation",
    "Bronchial",
    "Cartilage",
    "Lymphocytoplasmic inflammation",
    "Unknown",
]
T2I = {t: i for i, t in enumerate(TISSUE_TYPES)}
N_T = len(TISSUE_TYPES)

TYPE_COLORS = {
    "Alveolar":                                  "#E53030",
    "Bronchial":                                 "#1A72CC",
    "Cartilage":                                 "#F5C800",
    "Alveolar with empty spaces":                "#00BB44",
    "Alveolar with hemorrhage and inflammation": "#FF6F00",
    "Lymphocytoplasmic inflammation":            "#CC00CC",
    "Unknown":                                   "#BBBBBB",
}

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading h5ad...", flush=True)
adata = ad.read_h5ad(H5AD, backed='r')
obs = adata.obs[['chunk_left','chunk_top','tissue_type',
                  'slide_name_clean','record_id','biopsy_date',
                  'acr_status','acr_binary']].copy()
adata.file.close()

obs['chunk_left']  = obs['chunk_left'].astype(float)
obs['chunk_top']   = obs['chunk_top'].astype(float)
obs['acr_binary']  = obs['acr_binary'].astype(int)
obs['type_idx']    = obs['tissue_type'].map(T2I).fillna(N_T - 1).astype(int)

slides = obs[['slide_name_clean','record_id','acr_status','acr_binary']].drop_duplicates('slide_name_clean')
print(f"  {len(obs):,} patches | {len(slides)} slides | {obs['record_id'].nunique()} patients", flush=True)


# ── Per-slide feature extraction ───────────────────────────────────────────────
def extract_slide_features(df):
    """
    Returns a 1-D feature vector for one slide:
      - composition:     frac of each tissue type              (N_T,)
      - co-occurrence:   for each src type, frac of k-NN       (N_T × N_T,)
                         that are each dst type
      - boundary:        fraction of patches whose nearest
                         neighbor is a DIFFERENT tissue type   (1,)
    """
    n = len(df)
    coords   = df[['chunk_left','chunk_top']].values
    types    = df['type_idx'].values

    # Composition
    comp = np.zeros(N_T)
    for ti in range(N_T):
        comp[ti] = (types == ti).sum() / n

    # Spatial kNN co-occurrence
    k = min(K_NN, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree', n_jobs=4)
    nbrs.fit(coords)
    _, indices = nbrs.kneighbors(coords)
    neighbor_idx = indices[:, 1:]          # exclude self

    cooc = np.zeros((N_T, N_T))
    for ti in range(N_T):
        mask = types == ti
        if mask.sum() == 0:
            continue
        nbr_types = types[neighbor_idx[mask]].flatten()
        for tj in range(N_T):
            cooc[ti, tj] = (nbr_types == tj).sum() / (mask.sum() * k)

    # Boundary fraction: patches with at least one neighbor of different type
    same = (types[neighbor_idx] == types[:, None]).all(axis=1)
    boundary_frac = (~same).mean()

    return np.concatenate([comp, cooc.flatten(), [boundary_frac]])


# Feature column names
feat_names = (
    [f"frac_{t.replace(' ','_')[:25]}" for t in TISSUE_TYPES] +
    [f"cooc_{TISSUE_TYPES[i][:12]}_to_{TISSUE_TYPES[j][:12]}"
     for i in range(N_T) for j in range(N_T)] +
    ["boundary_frac"]
)

print(f"\nExtracting features for {len(slides)} slides...", flush=True)
slide_feats = []
for si, slide_name in enumerate(slides['slide_name_clean']):
    df = obs[obs['slide_name_clean'] == slide_name]
    feats = extract_slide_features(df)
    slide_feats.append(feats)
    if (si + 1) % 50 == 0:
        print(f"  [{si+1}/{len(slides)}]", flush=True)

X = np.array(slide_feats)
y = slides['acr_binary'].values
groups = slides['record_id'].values
acr_status = slides['acr_status'].values

print(f"\nFeature matrix: {X.shape}  |  ACR+: {y.sum()}  ACR-: {(1-y).sum()}", flush=True)


# ── Cross-validation (patient-level groups) ────────────────────────────────────
cv = StratifiedGroupKFold(n_splits=5)

models = {
    "Logistic (L1)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty='l1', solver='saga', C=0.1,
                                   max_iter=2000, class_weight='balanced'))
    ]),
    "Logistic (L2)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty='l2', C=0.1,
                                   max_iter=2000, class_weight='balanced'))
    ]),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6,
        scale_pos_weight=(1 - y).sum() / y.sum(),
        eval_metric='logloss', verbosity=0,
        random_state=42, n_jobs=8
    ),
}

results = {}
oof_preds = {m: np.zeros(len(y)) for m in models}

for mname, model in models.items():
    aucs, baccs = [], []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        pred  = (proba >= 0.5).astype(int)
        oof_preds[mname][te] = proba
        auc  = roc_auc_score(y[te], proba)
        bacc = balanced_accuracy_score(y[te], pred)
        aucs.append(auc); baccs.append(bacc)
    results[mname] = {"AUC": np.mean(aucs), "AUC_std": np.std(aucs),
                       "BACC": np.mean(baccs), "BACC_std": np.std(baccs)}
    print(f"  {mname:20s}  AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
          f"BACC={np.mean(baccs):.3f}±{np.std(baccs):.3f}", flush=True)

# Save metrics
with open(f"{OUTDIR}/cv_results.json", "w") as f:
    json.dump(results, f, indent=2)


# ── XGBoost feature importance ─────────────────────────────────────────────────
print("\nFitting final XGBoost on all data for feature importance...", flush=True)
xgb_final = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6,
    scale_pos_weight=(1 - y).sum() / y.sum(),
    eval_metric='logloss', verbosity=0,
    random_state=42, n_jobs=8
)
scaler_final = StandardScaler()
X_sc = scaler_final.fit_transform(X)
xgb_final.fit(X_sc, y)
importances = xgb_final.feature_importances_

# Top 25 features
top_idx = np.argsort(importances)[::-1][:25]


# ── Plots ──────────────────────────────────────────────────────────────────────

# 1. CV results bar chart
fig, ax = plt.subplots(figsize=(7, 4))
mnames = list(results.keys())
aucs_  = [results[m]["AUC"]  for m in mnames]
baccs_ = [results[m]["BACC"] for m in mnames]
auc_se = [results[m]["AUC_std"]  for m in mnames]
bac_se = [results[m]["BACC_std"] for m in mnames]
x_ = np.arange(len(mnames))
w  = 0.35
ax.bar(x_ - w/2, aucs_,  w, yerr=auc_se,  label='AUC',  color='#1A72CC', capsize=5)
ax.bar(x_ + w/2, baccs_, w, yerr=bac_se,  label='BACC', color='#E53030', capsize=5)
ax.set_xticks(x_); ax.set_xticklabels(mnames, fontsize=10)
ax.set_ylim(0, 1); ax.axhline(0.5, ls='--', color='grey', lw=1)
ax.set_ylabel('Score'); ax.set_title('ACR Classification — 5-Fold CV (patient-grouped)')
ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/cv_performance.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# 2. Feature importance
fig, ax = plt.subplots(figsize=(9, 7))
cols = []
for idx in top_idx:
    fn = feat_names[idx]
    if fn.startswith('frac_'):
        cols.append('#444444')
    elif fn.startswith('cooc_'):
        # colour by source tissue type
        src = fn.split('cooc_')[1].split('_to_')[0]
        match = next((t for t in TISSUE_TYPES if t.replace(' ','_')[:12] == src), None)
        cols.append(TYPE_COLORS.get(match, '#888888') if match else '#888888')
    else:
        cols.append('#999999')

ax.barh(range(25), importances[top_idx][::-1], color=cols[::-1])
ax.set_yticks(range(25))
ax.set_yticklabels([feat_names[i] for i in top_idx[::-1]], fontsize=8)
ax.set_xlabel('XGBoost Importance')
ax.set_title('Top 25 Spatial Features for ACR Prediction')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Legend for color coding
import matplotlib.patches as mpatches
leg = [mpatches.Patch(color='#444444', label='Composition (frac_)')]
for t in TISSUE_TYPES:
    if t != "Unknown":
        leg.append(mpatches.Patch(color=TYPE_COLORS[t], label=f'Co-occ src: {t}'))
ax.legend(handles=leg, fontsize=7, loc='lower right', frameon=False)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/feature_importance.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# 3. Co-occurrence heatmaps: ACR+ vs ACR- mean
short = [t.replace(' and ',' &\n').replace(' with ','\nw/ ')[:22] for t in TISSUE_TYPES]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax_, (label, mask) in zip(axes[:2], [("ACR−  (binary=0)", y==0), ("ACR+  (binary=1)", y==1)]):
    cooc_mean = X[mask, N_T:N_T+N_T*N_T].mean(axis=0).reshape(N_T, N_T)
    im = ax_.imshow(cooc_mean, cmap='YlOrRd', vmin=0)
    ax_.set_xticks(range(N_T)); ax_.set_xticklabels(short, fontsize=6, rotation=45, ha='right')
    ax_.set_yticks(range(N_T)); ax_.set_yticklabels(short, fontsize=6)
    ax_.set_title(label, fontsize=11, fontweight='bold')
    ax_.set_xlabel('Neighbor type'); ax_.set_ylabel('Source type')
    plt.colorbar(im, ax=ax_, fraction=0.046, pad=0.04)

# Difference heatmap
cooc_pos = X[y==1, N_T:N_T+N_T*N_T].mean(axis=0).reshape(N_T, N_T)
cooc_neg = X[y==0, N_T:N_T+N_T*N_T].mean(axis=0).reshape(N_T, N_T)
diff = cooc_pos - cooc_neg
vmax = np.abs(diff).max()
im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[2].set_xticks(range(N_T)); axes[2].set_xticklabels(short, fontsize=6, rotation=45, ha='right')
axes[2].set_yticks(range(N_T)); axes[2].set_yticklabels(short, fontsize=6)
axes[2].set_title('ACR+ minus ACR−\n(co-occurrence difference)', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Neighbor type'); axes[2].set_ylabel('Source type')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle('Spatial Co-occurrence of Tissue Types by ACR Status', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/cooccurrence_heatmaps.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# 4. Composition boxplots by ACR grade
fig, axes = plt.subplots(1, N_T, figsize=(16, 4), sharey=False)
acr_grades = ['A0B0', 'A1B0', 'A2B0', 'A1B1']
grade_cols  = ['#2D8A2D', '#E8A020', '#CC3333', '#7722AA']
for ti, (ax_, tt) in enumerate(zip(axes, TISSUE_TYPES)):
    data = [X[acr_status == g, ti] for g in acr_grades]
    data = [d for d in data if len(d) > 0]
    labels = [g for g, d in zip(acr_grades, [X[acr_status==g, ti] for g in acr_grades]) if len(d) > 0]
    bp = ax_.boxplot(data, patch_artist=True, widths=0.5, medianprops=dict(color='k', lw=2))
    for patch, col in zip(bp['boxes'], [grade_cols[acr_grades.index(l)] for l in labels]):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax_.set_xticks(range(1, len(labels)+1))
    ax_.set_xticklabels(labels, fontsize=7, rotation=30)
    ax_.set_title(tt.replace(' and ',' &\n').replace(' with ','\nw/ '), fontsize=7,
                  color=TYPE_COLORS[tt], fontweight='bold')
    ax_.spines['top'].set_visible(False); ax_.spines['right'].set_visible(False)

fig.suptitle('Tissue Type Fraction by ACR Grade', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(f"{OUTDIR}/composition_by_acr.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# 5. OOF ROC curve for best model
best_m = max(results, key=lambda m: results[m]['AUC'])
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y, oof_preds[best_m])
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(fpr, tpr, lw=2, color='#1A72CC',
        label=f'{best_m}\nAUC={results[best_m]["AUC"]:.3f}±{results[best_m]["AUC_std"]:.3f}')
ax.plot([0,1],[0,1],'--', color='grey', lw=1)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('OOF ROC Curve — ACR Binary'); ax.legend(fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/roc_curve.png", dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\nAll outputs in {OUTDIR}", flush=True)
print("\n=== FINAL RESULTS ===")
for m, r in results.items():
    print(f"  {m:20s}  AUC={r['AUC']:.3f}±{r['AUC_std']:.3f}  BACC={r['BACC']:.3f}±{r['BACC_std']:.3f}")
PYEOF
