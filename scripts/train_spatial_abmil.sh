#!/usr/bin/env bash
#SBATCH --job-name=spatial_abmil
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=80G
#SBATCH --gres=gpu:1 --constraint="a100_40gb|a100_80gb|h100_80gb"
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/train_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

OUTDIR="/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil"
mkdir -p "$OUTDIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
"""
2-Level Spatial ABMIL for ACR prediction.

Architecture:
  Slide → K spatial neighbourhoods (from DBSCAN preprocessing)

  Level 1 — Patch → Neighbourhood:
    For neighbourhood i:  patches (N_i × 1024) UNI features
    → proj(1024 → 256)
    → GatedAttention over N_i patches
    → neighbourhood_rep_i (256,)  +  patch_attn_i (N_i,)

  Level 2 — Neighbourhood → Slide:
    All neighbourhood reps (K × 256)
    → GatedAttention over K neighbourhoods
    → slide_rep (256,)  +  neighbourhood_attn (K,)
    → Linear → ACR prediction

Outputs:
  - neighbourhood_attn  → which spatial regions drive rejection
  - patch_attn          → which patches within each region matter
  Both can be projected back onto WSI coordinates for visualisation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FEAT_DIR = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil/slide_cluster_feats")
OUTDIR   = Path("/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/spatial_mil")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
IN_DIM   = 1024   # UNI feature dim
HIDDEN   = 256
DROPOUT  = 0.25
LR       = 3e-4
WD       = 1e-4
EPOCHS   = 150
PATIENCE = 20
N_SPLITS = 5
MAX_PATCHES_PER_CLUSTER = 512   # cap large clusters to save GPU memory


# ── Model ──────────────────────────────────────────────────────────────────────
class GatedAttentionPool(nn.Module):
    """Gated attention pooling: (B, N, D) → (B, D), (B, N)"""
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(in_dim, hidden), nn.Sigmoid())
        self.w = nn.Linear(hidden, 1, bias=False)

    def forward(self, h):
        a = self.w(self.V(h) * self.U(h))          # (B, N, 1)
        a = torch.softmax(a, dim=1)
        return (a * h).sum(dim=1), a.squeeze(-1)   # (B, D), (B, N)


class SpatialABMIL2Level(nn.Module):
    """
    2-level hierarchical ABMIL over spatial neighbourhoods.
    Forward takes a list of patch tensors (one per neighbourhood).
    """
    def __init__(self, in_dim=1024, hidden=256, dropout=0.25):
        super().__init__()
        # Shared patch encoder (same weights applied to every neighbourhood)
        self.patch_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Level 1: patch → neighbourhood
        self.patch_attn = GatedAttentionPool(hidden, hidden // 2)

        # Level 2: neighbourhood → slide
        self.nbhd_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )
        self.nbhd_attn = GatedAttentionPool(hidden, hidden // 2)

        # Prediction head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, clusters):
        """
        clusters: list of Tensors, each (N_i, in_dim) — patches in neighbourhood i
        Returns: logit (scalar), nbhd_attn (K,), patch_attns list[(N_i,)]
        """
        nbhd_reps  = []
        patch_attns = []

        for c in clusters:
            # c: (N_i, 1024)
            h = self.patch_proj(c.unsqueeze(0))         # (1, N_i, hidden)
            rep, pa = self.patch_attn(h)                # (1, hidden), (1, N_i)
            nbhd_reps.append(rep)
            patch_attns.append(pa.squeeze(0))           # (N_i,)

        # Stack neighbourhood reps → (1, K, hidden)
        H = torch.stack(nbhd_reps, dim=1)               # (1, K, hidden)
        H = self.nbhd_proj(H)
        slide_rep, na = self.nbhd_attn(H)               # (1, hidden), (1, K)

        logit = self.head(slide_rep).squeeze(-1)         # (1,)
        return logit, na.squeeze(0), patch_attns         # scalar, (K,), list[(N_i,)]


# ── Dataset ────────────────────────────────────────────────────────────────────
class SpatialMILDataset(Dataset):
    def __init__(self, paths, max_patches=MAX_PATCHES_PER_CLUSTER):
        self.paths = paths
        self.max_p = max_patches

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        d = torch.load(self.paths[i], map_location='cpu')
        clusters = []
        for c in d['clusters']:
            feats = c['feats']                  # (N_i, 1024)
            if feats.shape[0] > self.max_p:
                idx = torch.randperm(feats.shape[0])[:self.max_p]
                feats = feats[idx]
            clusters.append(feats)
        label = torch.tensor(d['label'], dtype=torch.float32)
        return clusters, label, d['slide'], d['patient']


def collate_fn(batch):
    # batch size must be 1 — variable K and N_i across slides
    clusters, label, slide, patient = batch[0]
    return clusters, label.unsqueeze(0), slide, patient


# ── Load all paths ─────────────────────────────────────────────────────────────
all_paths = sorted(FEAT_DIR.glob("*.pt"))
labels, patients, slides = [], [], []
for p in all_paths:
    d = torch.load(p, map_location='cpu')
    labels.append(d['label'])
    patients.append(d['patient'])
    slides.append(d['slide'])
labels   = np.array(labels)
patients = np.array(patients)

print(f"Slides: {len(all_paths)}  ACR+={labels.sum()}  ACR-={(1-labels).sum()}", flush=True)

# ── 5-fold CV ──────────────────────────────────────────────────────────────────
cv = StratifiedGroupKFold(n_splits=N_SPLITS)
fold_results = []
oof_probs   = np.zeros(len(all_paths))
oof_nbhd_attn = {}   # slide → neighbourhood attention weights (K,)

for fold, (tr_idx, te_idx) in enumerate(cv.split(all_paths, labels, patients)):
    print(f"\n══ Fold {fold} ══", flush=True)

    tr_paths = [all_paths[i] for i in tr_idx]
    te_paths = [all_paths[i] for i in te_idx]

    tr_ds = SpatialMILDataset(tr_paths)
    te_ds = SpatialMILDataset(te_paths)
    tr_dl = DataLoader(tr_ds, batch_size=1, shuffle=True,  collate_fn=collate_fn, num_workers=0)
    te_dl = DataLoader(te_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model  = SpatialABMIL2Level(IN_DIM, HIDDEN, DROPOUT).to(DEVICE)
    pw     = torch.tensor([(1-labels[tr_idx]).sum() / max(labels[tr_idx].sum(), 1)]).to(DEVICE)
    crit   = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    best_auc = 0.0; best_ep = 0; no_improve = 0; best_state = None
    tr_losses = []; val_aucs = []

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train(); ep_loss = 0.0
        for clusters, label, _, _ in tr_dl:
            clusters = [c.to(DEVICE) for c in clusters]
            label    = label.to(DEVICE)
            logit, _, _ = model(clusters)
            loss = crit(logit, label)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        sched.step()
        tr_losses.append(ep_loss / len(tr_dl))

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval(); vp, vl = [], []
        with torch.no_grad():
            for clusters, label, _, _ in te_dl:
                clusters = [c.to(DEVICE) for c in clusters]
                logit, _, _ = model(clusters)
                vp.append(torch.sigmoid(logit).cpu().item())
                vl.append(label.item())
        try:    auc = roc_auc_score(vl, vp)
        except: auc = 0.5
        val_aucs.append(auc)

        if auc > best_auc:
            best_auc = auc; best_ep = epoch; no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stop ep={epoch}  best_ep={best_ep}  AUC={best_auc:.3f}", flush=True)
            break
        if (epoch+1) % 25 == 0:
            print(f"  ep={epoch+1}  loss={tr_losses[-1]:.4f}  val_AUC={auc:.3f}  best={best_auc:.3f}", flush=True)

    # ── OOF predictions + neighbourhood attention ──────────────────────────────
    model.load_state_dict(best_state); model.eval()
    fp, fl = [], []
    with torch.no_grad():
        for clusters, label, slide_name, _ in te_dl:
            clusters = [c.to(DEVICE) for c in clusters]
            logit, nbhd_attn, _ = model(clusters)
            p = torch.sigmoid(logit).cpu().item()
            fp.append(p); fl.append(label.item())
            oof_nbhd_attn[slide_name] = nbhd_attn.cpu().numpy()

    for i, idx in enumerate(te_idx):
        oof_probs[idx] = fp[i]

    fold_auc  = roc_auc_score(fl, fp)
    fold_bacc = balanced_accuracy_score(fl, [int(p>0.5) for p in fp])
    print(f"  Fold {fold}: AUC={fold_auc:.3f}  BACC={fold_bacc:.3f}  best_ep={best_ep}", flush=True)
    fold_results.append({'fold': fold, 'auc': fold_auc, 'bacc': fold_bacc,
                          'best_ep': best_ep, 'tr_losses': tr_losses, 'val_aucs': val_aucs})

    torch.save(best_state, OUTDIR / f"spatial_abmil_fold{fold}.pt")

# ── Overall results ────────────────────────────────────────────────────────────
oof_auc  = roc_auc_score(labels, oof_probs)
oof_bacc = balanced_accuracy_score(labels, (oof_probs > 0.5).astype(int))
print(f"\n{'='*50}")
print(f"OOF AUC  = {oof_auc:.3f}")
print(f"OOF BACC = {oof_bacc:.3f}")
fold_auc_strs = [f"{r['auc']:.3f}" for r in fold_results]
print(f"Per-fold: {fold_auc_strs}")

results_dict = {
    'oof_auc': float(oof_auc), 'oof_bacc': float(oof_bacc),
    'folds': [{'fold': r['fold'], 'auc': float(r['auc']),
               'bacc': float(r['bacc']), 'best_ep': r['best_ep']}
              for r in fold_results]
}
with open(OUTDIR / "spatial_abmil_results.json", 'w') as f:
    json.dump(results_dict, f, indent=2)

# Save OOF predictions + neighbourhood attention strengths
oof_df = pd.DataFrame({'slide': slides, 'label': labels,
                        'prob': oof_probs, 'patient': patients})
oof_df.to_csv(OUTDIR / "oof_predictions.csv", index=False)
torch.save(oof_nbhd_attn, OUTDIR / "oof_neighbourhood_attn.pt")

# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 1. Learning curves (last fold)
ax = axes[0]
last = fold_results[-1]
ax.plot(last['tr_losses'], color='#E53030', label='Train loss')
ax2 = ax.twinx()
ax2.plot(last['val_aucs'], color='#1A72CC', label='Val AUC')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss', color='#E53030')
ax2.set_ylabel('Val AUC', color='#1A72CC')
ax.set_title('Learning Curve (last fold)', fontsize=11)

# 2. OOF ROC
ax = axes[1]
fpr, tpr, _ = roc_curve(labels, oof_probs)
ax.plot(fpr, tpr, lw=2, color='#1A72CC', label=f'AUC = {oof_auc:.3f}')
ax.plot([0,1],[0,1],'--', color='grey', lw=1)
ax.fill_between(fpr, tpr, alpha=0.1, color='#1A72CC')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('OOF ROC — 2-Level Spatial ABMIL', fontsize=11)
ax.legend(fontsize=10)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

# 3. Per-fold AUC bar chart
ax = axes[2]
faucs = [r['auc'] for r in fold_results]
ax.bar(range(N_SPLITS), faucs, color='#1A72CC', alpha=0.8)
ax.axhline(np.mean(faucs), ls='--', color='#E53030', lw=1.5, label=f'Mean={np.mean(faucs):.3f}')
ax.axhline(0.5, ls=':', color='grey', lw=1)
ax.set_xticks(range(N_SPLITS)); ax.set_xticklabels([f'Fold {i}' for i in range(N_SPLITS)])
ax.set_ylim(0, 1); ax.set_ylabel('AUC')
ax.set_title('Per-fold AUC', fontsize=11); ax.legend()
for sp in ['top','right']: ax.spines[sp].set_visible(False)

plt.suptitle('2-Level Spatial ABMIL — ACR Prediction (UNI features)', fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(OUTDIR / "spatial_abmil_results.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Neighbourhood attention visualisation for sample slides ───────────────────
summary = pd.read_csv(OUTDIR / "slide_cluster_summary.csv")

# Plot: distribution of neighbourhood attention in ACR+ vs ACR-
fig, ax = plt.subplots(figsize=(7, 4))
for label_val, color, name in [(0, '#2D8A2D', 'ACR−'), (1, '#CC3333', 'ACR+')]:
    sl_list = summary[summary['label'] == label_val]['slide'].tolist()
    all_attn = []
    for sl in sl_list:
        if sl in oof_nbhd_attn:
            all_attn.extend(oof_nbhd_attn[sl].tolist())
    if all_attn:
        ax.hist(all_attn, bins=40, density=True, alpha=0.5, color=color, label=name)
ax.set_xlabel('Neighbourhood attention weight')
ax.set_ylabel('Density')
ax.set_title('Distribution of neighbourhood attention — ACR+ vs ACR−', fontsize=11)
ax.legend();
for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout()
fig.savefig(OUTDIR / "neighbourhood_attn_distribution.png", dpi=150, bbox_inches='tight')
plt.close(fig)

print("\nAll outputs saved to:", OUTDIR, flush=True)
PYEOF
