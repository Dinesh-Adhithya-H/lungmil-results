#!/bin/bash
#SBATCH --job-name=results_analysis
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/results_analysis/analysis_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/results_analysis/analysis_%j.err

set -e
source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago

OUT=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/results_analysis
mkdir -p "$OUT"

python3 - << 'PYEOF'
import json, os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from scipy import stats
warnings.filterwarnings('ignore')

BASE     = "/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/results_mm_abmil_v8"
OUT      = "/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/results_analysis"
N_FOLDS  = 4

FUSIONS       = ['early', 'middle', 'late', 'self_attn', 'crossmodal_k8', 'iterative_r2_k8']
FUSION_LABELS = ['Early', 'Middle', 'Late', 'Self-Attn', 'Cross-Modal', 'Iterative']
COLORS        = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2','#937860']

# ── 1. Collect all status results ────────────────────────────────────────────
records = []
for endpoint in ['acr', 'clad', 'death']:
    base = f"{BASE}/{endpoint}"
    task_dir = [d for d in os.listdir(f"{base}/split0_fold0") if d != 'phase1'][0]
    for fold in range(N_FOLDS):
        fdir = f"{base}/split0_fold{fold}/{task_dir}"
        for sf in glob.glob(f"{fdir}/status_*.json"):
            fusion = os.path.basename(sf).replace('status_','').replace('.json','')
            if fusion not in FUSIONS: continue
            d = json.load(open(sf))
            if not d.get('completed', False): continue
            rec = {'endpoint': endpoint, 'fold': fold, 'fusion': fusion}
            if endpoint == 'acr':
                rec['bacc']    = d.get('bacc')
                rec['auc']     = d.get('auc')
                rec['sens']    = d.get('sens')
                rec['spec']    = d.get('spec')
                rec['mcc']     = d.get('mcc')
                rec['c_index'] = d.get('c_index')
            elif endpoint == 'clad':
                rec['c_index'] = d.get('c_index_clad', d.get('c_index'))
            elif endpoint == 'death':
                rec['c_index'] = d.get('c_index_death', d.get('c_index'))
            records.append(rec)

df = pd.DataFrame(records)
df.to_csv(f"{OUT}/all_fold_results.csv", index=False)
print(f"Collected {len(df)} records across {df.endpoint.nunique()} endpoints")

# ── 2. Summary table ─────────────────────────────────────────────────────────
rows = []
for endpoint in ['acr','clad','death']:
    sub = df[df.endpoint==endpoint]
    for f in FUSIONS:
        fsub = sub[sub.fusion==f]
        if len(fsub)==0: continue
        row = {'endpoint': endpoint, 'fusion': f, 'n': len(fsub)}
        for col in ['bacc','auc','sens','spec','mcc','c_index']:
            v = fsub[col].dropna().values
            if len(v):
                row[f'{col}_mean'] = round(np.mean(v),4)
                row[f'{col}_std']  = round(np.std(v),4)
        rows.append(row)
summary = pd.DataFrame(rows)
summary.to_csv(f"{OUT}/summary_table.csv", index=False)

# ── 3. Fig 1: Benchmark bar charts — one row per endpoint ─────────────────────
fig = plt.figure(figsize=(24, 14))
fig.suptitle('MM-ABMIL v8 — Full Benchmark Results', fontsize=15, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.55, wspace=0.35)

ep_metrics = {
    'acr':   [('bacc','BAcc'),('auc','AUC'),('sens','Sensitivity'),('spec','Specificity'),('mcc','MCC'),('c_index','C-index (ACR)')],
    'clad':  [('c_index','C-index')],
    'death': [('c_index','C-index')],
}
ep_row = {'acr':0,'clad':1,'death':2}
ep_title = {'acr':'ACR (Classification + Survival)','clad':'CLAD Survival','death':'Death Survival'}

for ep, metrics in ep_metrics.items():
    row = ep_row[ep]
    sub = df[df.endpoint==ep]
    n_metrics = len(metrics)
    # span all 6 cols evenly
    col_positions = np.linspace(0, 5, n_metrics, dtype=int)

    for cidx, (col, label) in zip(col_positions, metrics):
        ax = fig.add_subplot(gs[row, cidx])
        means, errs, colors = [], [], []
        for f, c in zip(FUSIONS, COLORS):
            v = sub[sub.fusion==f][col].dropna().values
            means.append(np.mean(v) if len(v) else np.nan)
            errs.append(np.std(v) if len(v) else 0)
            colors.append(c)

        valid = [(i,m,e,c) for i,(m,e,c) in enumerate(zip(means,errs,colors)) if not np.isnan(m)]
        xs = [x[0] for x in valid]
        ms = [x[1] for x in valid]
        es = [x[2] for x in valid]
        cs = [x[3] for x in valid]

        bars = ax.bar(range(len(xs)), ms, yerr=es, capsize=3, color=cs, alpha=0.85,
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([FUSION_LABELS[i] for i in xs], rotation=40, ha='right', fontsize=7)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(f'{ep_title[ep]}\n{label}' if cidx == col_positions[0] else label,
                     fontsize=8, fontweight='bold')
        lo = 0.4 if col in ['c_index','auc'] else 0.0
        ax.set_ylim(lo, 1.05)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
        for bar, m in zip(bars, ms):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=6.5)

plt.savefig(f"{OUT}/full_benchmark.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved full_benchmark.png")

# ── 4. Fig 2: Heatmap summary — best fusion per metric ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Performance Heatmap — Mean across folds', fontsize=13, fontweight='bold')

ep_cols_map = {
    'acr':   ['bacc','auc','sens','spec','mcc','c_index'],
    'clad':  ['c_index'],
    'death': ['c_index'],
}
ep_labels_map = {
    'acr':   ['BAcc','AUC','Sens','Spec','MCC','C-idx'],
    'clad':  ['C-idx'],
    'death': ['C-idx'],
}

for ax, ep in zip(axes, ['acr','clad','death']):
    sub  = df[df.endpoint==ep]
    cols = ep_cols_map[ep]
    labs = ep_labels_map[ep]
    mat  = np.full((len(FUSIONS), len(cols)), np.nan)
    for i, f in enumerate(FUSIONS):
        for j, col in enumerate(cols):
            v = sub[sub.fusion==f][col].dropna().values
            if len(v): mat[i,j] = np.mean(v)

    vmin = 0.5 if ep != 'acr' else 0.0
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=vmin, vmax=1.0)
    ax.set_xticks(range(len(cols)));  ax.set_xticklabels(labs, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(len(FUSIONS))); ax.set_yticklabels(FUSION_LABELS, fontsize=9)
    ax.set_title(f'{ep.upper()}', fontweight='bold', fontsize=11)
    for i in range(len(FUSIONS)):
        for j in range(len(cols)):
            if not np.isnan(mat[i,j]):
                txt_col = 'white' if mat[i,j] < 0.55 or mat[i,j] > 0.92 else 'black'
                ax.text(j, i, f'{mat[i,j]:.3f}', ha='center', va='center', fontsize=9, color=txt_col)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
fig.savefig(f"{OUT}/heatmap_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved heatmap_summary.png")

# ── 5. Fig 3: Per-fold scatter + mean line (ACR BAcc + CLAD/Death C-idx) ─────
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle('Per-fold Results (bars = mean, dots = individual folds)', fontsize=13, fontweight='bold')

plot_configs = [
    ('acr', 'bacc', 'ACR BAcc', 0.4),
    ('clad', 'c_index', 'CLAD C-index', 0.5),
    ('death', 'c_index', 'Death C-index', 0.5),
]
for ax, (ep, col, title, ylim_lo) in zip(axes, plot_configs):
    sub = df[df.endpoint==ep]
    for i, (f, fl, c) in enumerate(zip(FUSIONS, FUSION_LABELS, COLORS)):
        vals = sub[sub.fusion==f][col].dropna().values
        if len(vals):
            ax.hlines(np.mean(vals), i-0.35, i+0.35, color=c, linewidth=3, zorder=4)
            ax.scatter([i]*len(vals), vals, color=c, s=55, zorder=5, alpha=0.85,
                       edgecolors='black', linewidths=0.5)
            ax.text(i, np.mean(vals)+0.012, f'{np.mean(vals):.3f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(range(len(FUSIONS)))
    ax.set_xticklabels(FUSION_LABELS, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(col.upper().replace('_',' '), fontsize=10)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_ylim(ylim_lo, 1.02)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.25)

plt.tight_layout()
fig.savefig(f"{OUT}/per_fold_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved per_fold_scatter.png")

# ── 6. UMAPs: embed OOF predictions per endpoint ─────────────────────────────
try:
    from umap import UMAP
    from sklearn.preprocessing import StandardScaler

    for endpoint in ['acr', 'clad', 'death']:
        base = f"{BASE}/{endpoint}"
        task_dir = [d for d in os.listdir(f"{base}/split0_fold0") if d != 'phase1'][0]

        # collect per-slide OOF predictions from metrics_*.json (has probs + labels)
        all_probs, all_labels, all_fusions, all_folds = [], [], [], []
        for fold in range(N_FOLDS):
            fdir = f"{base}/split0_fold{fold}/{task_dir}"
            for mf in glob.glob(f"{fdir}/metrics_*.json"):
                fusion = os.path.basename(mf).replace('metrics_','').replace('.json','')
                if fusion not in FUSIONS: continue
                d = json.load(open(mf))
                # try test split first
                test_d = d.get('test', d)
                probs  = test_d.get('probs', [])
                labels = test_d.get('labels', [])
                if not probs: continue
                all_probs.extend(probs)
                all_labels.extend(labels)
                all_fusions.extend([fusion]*len(probs))
                all_folds.extend([fold]*len(probs))

        if len(all_probs) < 10:
            print(f"  {endpoint}: not enough prob data for UMAP, skipping")
            continue

        X = np.array(all_probs).reshape(-1,1)
        y = np.array(all_labels)
        fus_arr = np.array(all_fusions)

        # pad to 2D for UMAP if needed — use fold as second dim
        fold_arr = np.array(all_folds, dtype=float)
        X2 = np.column_stack([X.ravel(), fold_arr / N_FOLDS])
        X2 = StandardScaler().fit_transform(X2)

        umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        emb = umap_model.fit_transform(X2)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{endpoint.upper()} — UMAP of test predictions', fontsize=12, fontweight='bold')

        # left: by label
        ax = axes[0]
        for lbl, lname, lc in [(0,'Negative','#4C72B0'),(1,'Positive','#DD8452')]:
            mask = y == lbl
            ax.scatter(emb[mask,0], emb[mask,1], c=lc, s=20, alpha=0.6, label=lname)
        ax.set_title('Coloured by label'); ax.legend(fontsize=9)
        ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')

        # right: by fusion method
        ax = axes[1]
        for f, fl, c in zip(FUSIONS, FUSION_LABELS, COLORS):
            mask = fus_arr == f
            if mask.sum():
                ax.scatter(emb[mask,0], emb[mask,1], c=c, s=15, alpha=0.55, label=fl)
        ax.set_title('Coloured by fusion'); ax.legend(fontsize=7, ncol=2)
        ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')

        plt.tight_layout()
        fig.savefig(f"{OUT}/umap_{endpoint}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved umap_{endpoint}.png")

except ImportError:
    print("umap-learn not installed — skipping UMAP plots")

# ── 7. Fig 5: ROC curves per endpoint — best fusion + all fusions ─────────────
from sklearn.metrics import roc_curve, auc as sk_auc

for endpoint in ['acr']:  # ROC only meaningful for classification
    base = f"{BASE}/{endpoint}"
    task_dir = [d for d in os.listdir(f"{base}/split0_fold0") if d != 'phase1'][0]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0,1],[0,1],'k--',linewidth=0.8,alpha=0.5)

    for f, fl, c in zip(FUSIONS, FUSION_LABELS, COLORS):
        all_probs, all_labels = [], []
        for fold in range(N_FOLDS):
            fdir = f"{base}/split0_fold{fold}/{task_dir}"
            mf = f"{fdir}/metrics_{f}.json"
            if not os.path.exists(mf): continue
            d = json.load(open(mf))
            test_d = d.get('test', d)
            probs  = test_d.get('probs', [])
            labels = test_d.get('labels', [])
            if probs: all_probs.extend(probs); all_labels.extend(labels)
        if len(all_probs) < 5: continue
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=c, linewidth=2, label=f'{fl}  (AUC={roc_auc:.3f})')

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ACR ROC — pooled OOF predictions', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{OUT}/roc_acr.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved roc_acr.png")

# ── 8. Print final table ──────────────────────────────────────────────────────
print("\n" + "="*80)
print("FINAL RESULTS TABLE (mean ± std, 4 folds)")
print("="*80)

print("\n--- ACR ---")
print(f"{'Fusion':<20} {'BAcc':>12} {'AUC':>12} {'Sens':>12} {'Spec':>12} {'MCC':>12}")
for f, fl in zip(FUSIONS, FUSION_LABELS):
    sub = df[(df.endpoint=='acr') & (df.fusion==f)]
    def ms(col):
        v = sub[col].dropna().values
        return f"{np.mean(v):.3f}±{np.std(v):.3f}" if len(v) else "---"
    print(f"{fl:<20} {ms('bacc'):>12} {ms('auc'):>12} {ms('sens'):>12} {ms('spec'):>12} {ms('mcc'):>12}")

for ep in ['clad','death']:
    print(f"\n--- {ep.upper()} ---")
    print(f"{'Fusion':<20} {'C-index':>15}")
    for f, fl in zip(FUSIONS, FUSION_LABELS):
        sub = df[(df.endpoint==ep) & (df.fusion==f)]
        v = sub['c_index'].dropna().values
        s = f"{np.mean(v):.3f}±{np.std(v):.3f}  (n={len(v)})" if len(v) else "--- (still running)"
        print(f"{fl:<20} {s:>15}")

print(f"\nAll outputs: {OUT}")
PYEOF
