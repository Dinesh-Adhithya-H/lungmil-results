#!/usr/bin/env python3
"""
train_linear_models.py
Linear model baselines on nested CV splits for 4 tasks:
  1. ACR classification      → Balanced logistic regression → BACC + AUC
  2. ACR TTE (survival)      → Cox proportional hazards → C-index
  3. CLAD TTE (survival)     → Cox proportional hazards → C-index
  4. Death TTE (survival)    → Cox proportional hazards → C-index

Features (per sample, concatenated):
  - CLR-normalised cluster proportions: BAL (43), H&E tissue (6), CT (37)
  - Raw clinical features: 106 z-scored values (NaN → 0 after scaling)
  Missing modalities → zero-imputed after per-feature StandardScaler fit on train.

Folds: fold_0 … fold_3  (outer CV; val used for early stopping / C selection).
Outputs: results/linear_models/
  metrics_summary.csv        — per-fold per-task per-modality-combo test scores
  feature_importance.csv     — LASSO coefs / Cox coefs averaged over folds
  figures/                   — BACC/C-index box plots, feature importance plots

Run via sbatch — do NOT run on the login node.
"""

import gc, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ── optional lifelines for Cox ────────────────────────────────────────────
try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("lifelines not installed — survival models skipped", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
PROP_DIR   = Path("/home/aih/dinesh.haridoss/chicago_mil/results/cluster_proportions")
CLMAP      = Path("/home/aih/dinesh.haridoss/chicago_mil/results/cluster_name_maps/HE_cluster_map.json")
SPLITS_CSV = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
OUT_DIR    = Path("/home/aih/dinesh.haridoss/chicago_mil/results/linear_models")
FIG_DIR    = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8,
    "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
ACR_POS = "#E53935"; ACR_NEG = "#1E88E5"
CLAD_COL= "#FB8C00"; DEATH_COL="#5C6BC0"; GREY="#90A4AE"; DARK="#37474F"

TISSUE_MERGE = {
    "Alveolar with hemorrhage and inflammation": "Alveolar inflamed",
    "Alveolar with empty spaces":               "Alveolar (clear)",
    "Alveolar":      "Alveolar",
    "Bronchial":     "Bronchial",
    "Vascular":      "Vascular",
    "Unknown":       "Unknown",
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data …", flush=True)
splits = pd.read_csv(SPLITS_CSV)
splits["stem"] = splits["file"].str.replace(".pt", "", regex=False).str.zfill(5)
splits["acr_binary"] = splits["label"].where(splits["label"].notna())

bal_prop  = pd.read_csv(PROP_DIR / "bal_cluster_prop.csv")
he_prop   = pd.read_csv(PROP_DIR / "he_cluster_prop.csv")
ct_prop   = pd.read_csv(PROP_DIR / "ct_cluster_prop.csv")
clin_raw  = pd.read_csv(PROP_DIR / "clinical_features.csv")
he_names  = pd.read_csv(PROP_DIR / "he_cluster_names.csv")
clmap     = json.load(open(CLMAP)) if CLMAP.exists() else {}
clin_names= pd.read_csv(PROP_DIR / "clinical_feature_names.csv")

for df in [bal_prop, he_prop, ct_prop, clin_raw]:
    df["stem"] = df["stem"].astype(str).str.zfill(5)

bal_cols  = [c for c in bal_prop.columns if c.startswith("cluster_")]
he_cols   = [c for c in he_prop.columns  if c.startswith("cluster_")]
ct_cols   = [c for c in ct_prop.columns  if c.startswith("cluster_")]
clin_cols = [c for c in clin_raw.columns if c.startswith("feat_")]

# H&E tissue aggregation (54 sub → 6 tissue types)
he_sub_to_tissue = {int(r["idx"]): TISSUE_MERGE.get(clmap.get(str(r["name"]), "Unknown"), "Unknown")
                    for _, r in he_names.iterrows()}
for tt in ["Alveolar inflamed","Alveolar","Alveolar (clear)","Bronchial","Vascular","Unknown"]:
    cols_t = [c for c in he_cols if he_sub_to_tissue.get(int(c.split("_")[1]),"Unknown") == tt]
    he_prop[f"tissue_{tt}"] = he_prop[cols_t].sum(axis=1) if cols_t else 0.0
he_tissue_cols = [f"tissue_{tt}" for tt in
                  ["Alveolar inflamed","Alveolar","Alveolar (clear)","Bronchial","Vascular","Unknown"]]

# Feature names
bal_names_map  = pd.read_csv(PROP_DIR/"bal_cluster_names.csv").set_index("idx")["name"].to_dict()
ct_names_map   = pd.read_csv(PROP_DIR/"ct_cluster_names.csv").set_index("idx")["name"].to_dict()
clin_names_map = clin_names.set_index("idx")["name"].to_dict()

# Feature names — prefixed with modality tag so multimodal importance plots are unambiguous
bal_feat_names  = [f"BAL:{bal_names_map.get(int(c.split('_')[1]), c)}" for c in bal_cols]
he_feat_names   = ["HE:Alveolar inflamed","HE:Alveolar","HE:Alveolar (clear)",
                   "HE:Bronchial","HE:Vascular","HE:Unknown"]
ct_feat_names   = [f"CT:{ct_names_map.get(int(c.split('_')[1]), c)}" for c in ct_cols]
clin_feat_names = [f"Clin:{clin_names_map.get(int(c.split('_')[1]), c)}" for c in clin_cols]

# Modality tag → display color (for importance bars)
FEAT_MOD_COLORS = {
    "BAL":  "#42A5F5",
    "HE":   "#66BB6A",
    "CT":   "#FFA726",
    "Clin": "#AB47BC",
}

def feat_color(feat_name):
    prefix = feat_name.split(":")[0] if ":" in feat_name else "Clin"
    return FEAT_MOD_COLORS.get(prefix, GREY)

def feat_mod_label(feat_name):
    """Strip modality prefix for display; keep prefix as modality label."""
    if ":" in feat_name:
        mod, name = feat_name.split(":", 1)
        return name, mod
    return feat_name, "?"

print(f"  BAL={len(bal_prop)}  HE={len(he_prop)}  CT={len(ct_prop)}  Clin={len(clin_raw)}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLR TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────
def clr(df, cols, pseudo=1e-5):
    X = df[cols].values.astype(float).copy()
    X = np.where(X <= 0, pseudo, X)
    X = X / X.sum(axis=1, keepdims=True)
    X = np.where(X <= 0, pseudo, X)
    logX = np.log(X)
    return logX - logX.mean(axis=1, keepdims=True)

bal_clr = clr(bal_prop, bal_cols)
he_clr  = clr(he_prop,  he_tissue_cols)
ct_clr  = clr(ct_prop,  ct_cols)
# Clinical: NaN → 0 after StandardScaler; no CLR (not compositional)

# Build stem-indexed feature matrices
bal_feat = pd.DataFrame(bal_clr,   columns=bal_feat_names,  index=bal_prop["stem"])
he_feat  = pd.DataFrame(he_clr,    columns=he_feat_names,   index=he_prop["stem"])
ct_feat  = pd.DataFrame(ct_clr,    columns=ct_feat_names,   index=ct_prop["stem"])
clin_feat= pd.DataFrame(clin_raw[clin_cols].values, columns=clin_feat_names,
                         index=clin_raw["stem"])

all_stems = splits["stem"].values


# ─────────────────────────────────────────────────────────────────────────────
# BUILD FEATURE MATRIX (per stem, zero-impute missing modalities)
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_matrix(stems, modalities):
    """
    modalities: list of (name, feat_df) — feat_df indexed by stem.
    Returns X (n_stems × n_feats), feat_names list.
    """
    blocks, names = [], []
    for mod_name, fdf in modalities:
        aligned = fdf.reindex(stems)
        blocks.append(aligned.values)
        names.extend(fdf.columns.tolist())
    X = np.hstack(blocks)
    return X, names


# Modality sets: each modality individually + All (clinical + all image CLR features)
MODALITY_SETS = {
    "Clinical":  [("Clin", clin_feat)],
    "BAL":       [("BAL",  bal_feat)],
    "H&E":       [("HE",   he_feat)],
    "CT":        [("CT",   ct_feat)],
    "All":       [("BAL",  bal_feat), ("HE", he_feat), ("CT", ct_feat),
                  ("Clin", clin_feat)],
}


# ─────────────────────────────────────────────────────────────────────────────
# TASKS
# ─────────────────────────────────────────────────────────────────────────────
TASKS = [
    # (name,  type,    label_col,    time_col,     event_col)
    ("ACR",   "cls",   "acr_binary", None,         None),
    ("ACR_TTE","surv", None,         "acr_days",   "acr_status"),
    ("CLAD",  "surv",  None,         "clad_days",  "clad_status"),
    ("Death", "surv",  None,         "death_days", "death_status"),
]

# 5 splits × fold0 only = 5 outer evaluations per task
# For each: val used for C hyperparameter selection, then retrain on train+val, test on test
OUTER_FOLDS = [f"split{s}_fold0" for s in range(5)]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def pval_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def scale_impute(X_tr, X_val, X_te):
    """StandardScaler fit on train; NaN → 0 everywhere."""
    sc = StandardScaler()
    X_tr  = np.nan_to_num(sc.fit_transform(X_tr),  nan=0.0)
    X_val = np.nan_to_num(sc.transform(X_val),      nan=0.0)
    X_te  = np.nan_to_num(sc.transform(X_te),       nan=0.0)
    return X_tr, X_val, X_te, sc

def run_cls(X_tr, y_tr, X_val, y_val, X_te, y_te):
    """Balanced logistic regression with L1 pen.
    Select C on val BACC, then refit on train+val combined before testing.
    """
    best_C, best_bacc = 0.01, -1
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        try:
            m = LogisticRegression(C=C, penalty="l1", solver="saga",
                                   class_weight="balanced", max_iter=500,
                                   random_state=42)
            m.fit(X_tr, y_tr)
            b = balanced_accuracy_score(y_val, m.predict(X_val))
            if b > best_bacc:
                best_bacc = b; best_C = C
        except Exception:
            pass
    # Refit on train+val combined with selected C
    X_tv = np.vstack([X_tr, X_val])
    y_tv = np.concatenate([y_tr, y_val])
    m = LogisticRegression(C=best_C, penalty="l1", solver="saga",
                           class_weight="balanced", max_iter=500, random_state=42)
    m.fit(X_tv, y_tv)
    y_pred = m.predict(X_te)
    y_prob = m.predict_proba(X_te)[:, 1] if hasattr(m, "predict_proba") else y_pred
    bacc = balanced_accuracy_score(y_te, y_pred)
    try:
        auc = roc_auc_score(y_te, y_prob)
    except Exception:
        auc = float("nan")
    return bacc, auc, m.coef_.ravel(), best_C

def _clean_surv(X, df, time_col, event_col, feat_cols):
    """Build a clean DataFrame for lifelines: cast time/event to float, drop NaN/inf rows."""
    out = pd.DataFrame(X, columns=feat_cols)
    out[time_col]  = pd.to_numeric(df[time_col].values,  errors="coerce").astype(float)
    out[event_col] = pd.to_numeric(df[event_col].values, errors="coerce").astype(float)
    mask = (np.isfinite(out[time_col].values) & (out[time_col].values > 0)
            & np.isfinite(out[event_col].values))
    return out[mask].copy()

def run_cox(X_tr, df_tr, X_te, df_te, time_col, event_col):
    """CoxPH with Ridge penalisation; test C-index.
    Near-zero-variance features (many NaN-imputed zeros) are dropped before
    fitting to avoid a singular Hessian.  Newton-Raphson step_size=0.1 prevents
    coefficient explosion on the first step.
    """
    if not HAS_LIFELINES:
        return float("nan"), np.zeros(X_tr.shape[1])
    n_feats = X_tr.shape[1]

    # Drop near-constant columns (std < 0.05 after scaling); they contribute
    # only noise and make the Hessian ill-conditioned after zero-imputation.
    col_std = np.std(X_tr, axis=0)
    keep    = col_std > 0.05
    if keep.sum() < 2:
        keep = col_std > 0        # fallback: keep anything with any variance
    X_tr_k = X_tr[:, keep]
    X_te_k = X_te[:, keep]
    feat_cols = [f"f{i}" for i in range(keep.sum())]

    # Scale penalizer with retained feature count
    penalizer = max(1.0, keep.sum() / 20.0)

    try:
        tr_df = _clean_surv(X_tr_k, df_tr, time_col, event_col, feat_cols)
        if len(tr_df) < 20 or tr_df[event_col].sum() < 5:
            return float("nan"), np.zeros(n_feats)
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(tr_df, duration_col=time_col, event_col=event_col,
                show_progress=False, fit_options={"step_size": 0.1})

        te_df = _clean_surv(X_te_k, df_te, time_col, event_col, feat_cols)
        if len(te_df) < 5 or te_df[event_col].sum() < 2:
            return float("nan"), np.zeros(n_feats)
        hazards = cph.predict_partial_hazard(te_df[feat_cols])
        ci = concordance_index(te_df[time_col], -hazards, te_df[event_col])

        # Back-project kept coefs into original feature space
        coefs_full = np.zeros(n_feats)
        coefs_full[keep] = cph.params_.values
        return ci, coefs_full
    except Exception as e:
        print(f"    Cox error: {e}", flush=True)
        return float("nan"), np.zeros(n_feats)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
all_rows = []
importance_records = []   # (task, modset, fold, feat_name, coef)

for task_name, task_type, label_col, time_col, event_col in TASKS:
    print(f"\n=== Task: {task_name} ({task_type}) ===", flush=True)

    for mod_name, mod_list in MODALITY_SETS.items():
        X_full, feat_names = build_feature_matrix(all_stems, mod_list)
        X_df = pd.DataFrame(X_full, index=all_stems, columns=feat_names)

        fold_scores = []
        fold_coefs  = []

        for fold_col in OUTER_FOLDS:
            tr_idx  = splits[splits[fold_col] == "train"].index
            val_idx = splits[splits[fold_col] == "val"].index
            te_idx  = splits[splits[fold_col] == "test"].index

            tr_stems  = splits.loc[tr_idx,  "stem"].values
            val_stems = splits.loc[val_idx, "stem"].values
            te_stems  = splits.loc[te_idx,  "stem"].values

            X_tr  = X_df.reindex(tr_stems).values.astype(float)
            X_val = X_df.reindex(val_stems).values.astype(float)
            X_te  = X_df.reindex(te_stems).values.astype(float)

            X_tr, X_val, X_te, _ = scale_impute(X_tr, X_val, X_te)

            if task_type == "cls":
                y_tr  = splits.loc[tr_idx,  label_col].values
                y_val = splits.loc[val_idx, label_col].values
                y_te  = splits.loc[te_idx,  label_col].values
                # Drop NaN labels (no biopsy)
                tr_m  = ~np.isnan(y_tr.astype(float))
                val_m = ~np.isnan(y_val.astype(float))
                te_m  = ~np.isnan(y_te.astype(float))
                if tr_m.sum() < 10 or te_m.sum() < 5:
                    continue
                y_tr  = y_tr[tr_m].astype(int)
                y_val = y_val[val_m].astype(int)
                y_te  = y_te[te_m].astype(int)
                X_tr_m  = X_tr[tr_m]; X_val_m = X_val[val_m]; X_te_m = X_te[te_m]
                if y_tr.sum() < 3 or y_val.sum() < 1:
                    continue
                # run_cls: C selected on val, then refits on train+val before test
                bacc, auc, coefs, best_C = run_cls(X_tr_m, y_tr,
                                                    X_val_m, y_val,
                                                    X_te_m, y_te)
                fold_scores.append({"fold": fold_col, "bacc": bacc, "auc": auc, "C": best_C,
                                    "n_test": int(te_m.sum()),
                                    "n_pos":  int(y_te.sum()),
                                    "n_neg":  int((y_te==0).sum())})
                fold_coefs.append(coefs)
                print(f"  {mod_name} {fold_col}: BACC={bacc:.3f} AUC={auc:.3f}", flush=True)

            else:  # survival — train on train+val combined, test on test
                tv_idx   = splits[(splits[fold_col]=="train") | (splits[fold_col]=="val")].index
                tv_stems = splits.loc[tv_idx, "stem"].values
                X_tv_raw = X_df.reindex(tv_stems).values.astype(float)
                # Refit scaler on train+val
                X_tv_scaled, _, X_te_scaled, _ = scale_impute(X_tv_raw,
                                                               X_tv_raw[:1],  # dummy val
                                                               X_te)
                df_tv = splits.loc[tv_idx].copy(); df_tv["stem"] = tv_stems
                df_te = splits.loc[te_idx].copy(); df_te["stem"] = te_stems
                ci, coefs = run_cox(X_tv_scaled, df_tv, X_te_scaled, df_te,
                                    time_col, event_col)
                fold_scores.append({"fold": fold_col, "cindex": ci,
                                    "n_test": len(te_stems),
                                    "n_events": int(splits.loc[te_idx, event_col].fillna(0).sum())})
                fold_coefs.append(coefs)
                print(f"  {mod_name} {fold_col}: C-index={ci:.3f}", flush=True)

        if not fold_scores:
            continue

        # Aggregate
        score_df = pd.DataFrame(fold_scores)
        for _, row in score_df.iterrows():
            r = {"task": task_name, "task_type": task_type,
                 "modality": mod_name, "fold": row["fold"]}
            r.update(row.to_dict())
            all_rows.append(r)

        # Average feature importance
        if fold_coefs:
            mean_coef = np.nanmean([c for c in fold_coefs if len(c)==len(feat_names)], axis=0)
            for fn, cv in zip(feat_names, mean_coef):
                importance_records.append({
                    "task": task_name, "modality": mod_name,
                    "feature": fn, "coef": float(cv)
                })

# ─────────────────────────────────────────────────────────────────────────────
# SAVE METRICS
# ─────────────────────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(all_rows)
metrics_df.to_csv(OUT_DIR / "metrics_summary.csv", index=False)
print(f"\nSaved metrics → {OUT_DIR / 'metrics_summary.csv'}", flush=True)

imp_df = pd.DataFrame(importance_records)
imp_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)
print(f"Saved importance → {OUT_DIR / 'feature_importance.csv'}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating figures …", flush=True)

TASK_COLORS = {"ACR": ACR_POS, "ACR_TTE": "#D81B60",
               "CLAD": CLAD_COL, "Death": DEATH_COL}
MOD_ORDER   = ["Clinical", "BAL", "H&E", "CT", "All"]
MOD_COLORS  = {"Clinical": "#AB47BC", "BAL": "#42A5F5", "H&E": "#66BB6A",
               "CT": "#FFA726", "All": "#EF5350"}
N_EVALS = metrics_df["fold"].nunique()

def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {name}.png/.pdf", flush=True)


# ── Summary table: mean ± std across all splits × folds ─────────────────────
cls_grp  = metrics_df[metrics_df["task_type"]=="cls"].groupby(["task","modality"])["bacc"]
surv_grp = metrics_df[metrics_df["task_type"]=="surv"].groupby(["task","modality"])["cindex"]
cls_tbl  = cls_grp.agg(["mean","std","count"]).round(3).reset_index()
surv_tbl = surv_grp.agg(["mean","std","count"]).round(3).reset_index()
cls_tbl["metric"]  = "BACC"
surv_tbl["metric"] = "C-index"
summary_tbl = pd.concat([cls_tbl, surv_tbl], ignore_index=True)
summary_tbl["mean±std"] = (summary_tbl["mean"].map(lambda x: f"{x:.3f}") + " ± " +
                            summary_tbl["std"].map(lambda x: f"{x:.3f}"))
summary_tbl.to_csv(OUT_DIR / "metrics_mean_std.csv", index=False)
print(f"Saved mean±std table → {OUT_DIR / 'metrics_mean_std.csv'}", flush=True)
print(summary_tbl[["task","modality","metric","mean±std","count"]].to_string(index=False))

# ── Fig 0: Mean ± std summary table as heatmap figure ───────────────────────
TASK_ORDER = ["ACR","ACR_TTE","CLAD","Death"]
METRIC_MAP = {"ACR": "BACC", "ACR_TTE": "C-index", "CLAD": "C-index", "Death": "C-index"}

pivot_mean = pd.DataFrame(index=MOD_ORDER, columns=TASK_ORDER, dtype=float)
pivot_std  = pd.DataFrame(index=MOD_ORDER, columns=TASK_ORDER, dtype=float)
for task in TASK_ORDER:
    met = "bacc" if task == "ACR" else "cindex"
    sub = metrics_df[metrics_df["task"]==task].groupby("modality")[met].agg(["mean","std"])
    for mod in MOD_ORDER:
        if mod in sub.index:
            pivot_mean.loc[mod, task] = sub.loc[mod, "mean"]
            pivot_std.loc[mod, task]  = sub.loc[mod, "std"]

fig_tbl, ax_tbl = plt.subplots(figsize=(10, 5))
ax_tbl.axis("off")
fig_tbl.suptitle(
    f"Linear Model Test Performance — Mean ± Std over {N_EVALS} evaluations "
    f"(5 splits × 4 folds)\n"
    "Features: CLR cluster proportions (BAL/H&E/CT) + raw clinical z-scores",
    fontsize=10, fontweight="bold", y=1.02)

cell_text, cell_colors = [], []
col_labels = [f"{t}\n({METRIC_MAP[t]})" for t in TASK_ORDER]

import matplotlib.cm as cm
cmap_perf = cm.RdYlGn

for mod in MOD_ORDER:
    row_text, row_col = [], []
    for task in TASK_ORDER:
        m = pivot_mean.loc[mod, task]
        s = pivot_std.loc[mod, task]
        if pd.isna(m):
            row_text.append("—"); row_col.append("#F5F5F5")
        else:
            row_text.append(f"{m:.3f}\n±{s:.3f}")
            norm_val = (m - 0.45) / 0.3   # 0.45→red, 0.75→green
            norm_val = float(np.clip(norm_val, 0, 1))
            rgba = cmap_perf(norm_val)
            # lighten
            r, g, b, _ = rgba
            row_col.append((r*0.4+0.6, g*0.4+0.6, b*0.4+0.6, 1.0))
    cell_text.append(row_text)
    cell_colors.append(row_col)

tbl = ax_tbl.table(
    cellText=cell_text, cellColours=cell_colors,
    rowLabels=MOD_ORDER, colLabels=col_labels,
    cellLoc="center", loc="center",
    bbox=[0, 0, 1, 1]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#CFD8DC")
    if r == 0:
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)
    elif c == -1:
        cell.set_facecolor(MOD_COLORS.get(MOD_ORDER[r-1], GREY))
        cell.set_text_props(color="white", fontweight="bold")
    cell.set_height(0.16)

save_fig(fig_tbl, "fig0_summary_table")
gc.collect()


# ── Fig A: BACC/C-index box plot per task × modality ────────────────────────
fig_a, axes_a = plt.subplots(1, 4, figsize=(20, 6))
fig_a.suptitle(f"Linear Model Test Performance — {N_EVALS} evaluations (5 splits × 4 folds)\n"
               "Features: CLR cluster proportions + raw clinical; "
               "Classification: balanced logistic (L1); Survival: CoxPH (ridge)",
               fontsize=10, fontweight="bold")

for ai, (task_name, task_type, *_) in enumerate(TASKS):
    ax = axes_a[ai]
    task_df = metrics_df[metrics_df["task"] == task_name]
    metric  = "bacc" if task_type == "cls" else "cindex"
    baseline= 0.5
    xs, ys, cols = [], [], []
    for xi, mod in enumerate([m for m in MOD_ORDER if m in task_df["modality"].unique()]):
        vals = task_df[task_df["modality"] == mod][metric].dropna().values
        for v in vals:
            xs.append(xi); ys.append(v); cols.append(MOD_COLORS.get(mod, GREY))
        if len(vals):
            ax.boxplot([vals], positions=[xi], widths=0.5,
                       patch_artist=True,
                       boxprops=dict(facecolor=MOD_COLORS.get(mod, GREY), alpha=0.6),
                       medianprops=dict(color=DARK, linewidth=2),
                       whiskerprops=dict(color=DARK), capprops=dict(color=DARK),
                       flierprops=dict(marker="o", markersize=4,
                                       color=MOD_COLORS.get(mod, GREY), alpha=0.7))
    np.random.seed(42)
    ax.scatter([x+np.random.uniform(-0.12,0.12) for x in xs], ys,
               c=cols, s=20, alpha=0.8, zorder=5, linewidths=0)
    ax.axhline(baseline, color=GREY, lw=0.9, ls="--", alpha=0.7,
               label="chance (0.5)")
    mod_present = [m for m in MOD_ORDER if m in task_df["modality"].unique()]
    ax.set_xticks(range(len(mod_present)))
    ax.set_xticklabels(mod_present, rotation=30, ha="right", fontsize=7)
    ylabel = "BACC" if task_type == "cls" else "C-index"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{task_name}\n({ylabel})", color=TASK_COLORS.get(task_name, DARK))
    ax.set_ylim(0.35, 1.0)
    ax.axhline(0.5, color=GREY, lw=0.8, ls="--", alpha=0.5)

plt.tight_layout()
save_fig(fig_a, "figA_model_performance")
gc.collect()


# ── Fig B: Feature importance — All modality, top 25 per task, colored by modality ─
if len(imp_df):
    fig_b, axes_b = plt.subplots(1, 4, figsize=(24, 14))
    fig_b.suptitle(
        "Multimodal Feature Importance (top 25 by |coef|, averaged over 5 splits)\n"
        "Bar color = modality origin   |   Direction: + → higher risk / ACR+",
        fontsize=10, fontweight="bold")
    for bi, (task_name, task_type, *_) in enumerate(TASKS):
        ax_b = axes_b[bi]
        sub = imp_df[(imp_df["task"] == task_name) & (imp_df["modality"] == "All")].copy()
        if len(sub) == 0:
            ax_b.text(0.5, 0.5, "No data", ha="center", va="center",
                      transform=ax_b.transAxes); continue
        sub = sub.reindex(sub["coef"].abs().sort_values(ascending=False).index).head(25)
        ys_b  = np.arange(len(sub))
        # Color by modality (from prefix), shade by direction
        bar_cols = []
        for feat, coef in zip(sub["feature"].values, sub["coef"].values):
            base = feat_color(feat)
            # darken if negative coef (protective)
            import matplotlib.colors as mc
            rgb = mc.to_rgb(base)
            if coef < 0:
                rgb = tuple(max(0, c - 0.18) for c in rgb)
            bar_cols.append(rgb)
        ax_b.barh(ys_b, sub["coef"].values, color=bar_cols, alpha=0.88, linewidth=0)
        ax_b.set_yticks(ys_b)
        # Display name without prefix, prefix shown via color
        display_names = [feat_mod_label(f)[0][:28] for f in sub["feature"].values]
        ax_b.set_yticklabels(display_names, fontsize=6.5)
        ax_b.axvline(0, color=GREY, lw=0.8)
        ax_b.set_xlabel("Coef  (+ → risk / ACR+)", fontsize=8)
        ax_b.set_title(f"{task_name}", color=TASK_COLORS.get(task_name, DARK),
                       fontsize=10, fontweight="bold")
        ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)
        # Modality legend inside each panel
        legend_handles = [mpatches.Patch(color=c, label=m)
                          for m, c in FEAT_MOD_COLORS.items()]
        ax_b.legend(handles=legend_handles, fontsize=6.5, frameon=False,
                    loc="lower right", title="Modality", title_fontsize=6.5)
    plt.tight_layout()
    save_fig(fig_b, "figB_feature_importance_top25")
    gc.collect()


# ── Fig C: Unimodal vs multimodal performance summary ───────────────────────
if len(metrics_df):
    fig_c, ax_c = plt.subplots(figsize=(14, 6))
    fig_c.suptitle("Unimodal vs Multimodal Performance — Mean ± SD across folds",
                   fontsize=10, fontweight="bold")
    x_pos, x_labels, x_colors = [], [], []
    y_means, y_sds = [], []
    xi = 0
    for task_name, task_type, *_ in TASKS:
        task_df = metrics_df[metrics_df["task"] == task_name]
        metric  = "bacc" if task_type == "cls" else "cindex"
        for mod in MOD_ORDER:
            vals = task_df[task_df["modality"] == mod][metric].dropna().values
            if len(vals) == 0: continue
            x_pos.append(xi); xi += 1
            x_labels.append(f"{task_name}\n{mod}")
            x_colors.append(MOD_COLORS.get(mod, GREY))
            y_means.append(np.mean(vals))
            y_sds.append(np.std(vals))
        xi += 0.5  # gap between tasks

    bars = ax_c.bar(x_pos, y_means, color=x_colors, alpha=0.8,
                    yerr=y_sds, capsize=3,
                    error_kw={"ecolor": DARK, "elinewidth": 1.0})
    ax_c.axhline(0.5, color=GREY, lw=0.9, ls="--", alpha=0.7, label="chance")
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(x_labels, rotation=40, ha="right", fontsize=6)
    ax_c.set_ylabel("BACC / C-index")
    ax_c.set_ylim(0.35, 1.0)
    handles = [mpatches.Patch(color=MOD_COLORS[m], label=m) for m in MOD_ORDER
               if m in MOD_COLORS]
    ax_c.legend(handles=handles, frameon=False, fontsize=7,
                loc="upper right", ncol=2)
    plt.tight_layout()
    save_fig(fig_c, "figC_unimodal_vs_multimodal")
    gc.collect()


# ── Fig D: Per-task feature importance — separate modality-group panels ──────
if len(imp_df):
    for task_name, task_type, *_ in TASKS:
        fig_d, axes_d = plt.subplots(1, len(MODALITY_SETS), figsize=(4*len(MODALITY_SETS), 14))
        coef_lbl = ("LASSO coef (balanced logistic)" if task_type=="cls"
                    else "Cox coef (+→ shorter TTE / higher risk)")
        fig_d.suptitle(
            f"Feature Importance: {task_name} — per modality\n"
            f"{coef_lbl};  averaged over 5 splits  |  "
            f"'All' panel: bar color = modality origin",
            fontsize=10, fontweight="bold")
        for di, (mod_name, _) in enumerate(MODALITY_SETS.items()):
            ax_d = axes_d[di]
            sub_d = imp_df[(imp_df["task"]==task_name) &
                           (imp_df["modality"]==mod_name)].copy()
            if len(sub_d) == 0:
                ax_d.text(0.5,0.5,"No data",ha="center",va="center",
                          transform=ax_d.transAxes); continue
            n_show = min(20, len(sub_d))
            sub_d = sub_d.reindex(
                sub_d["coef"].abs().sort_values(ascending=False).index).head(n_show)
            ys_d = np.arange(len(sub_d))

            if mod_name == "All":
                # Color by modality origin
                import matplotlib.colors as mc2
                bar_cols_d = []
                for feat, coef in zip(sub_d["feature"].values, sub_d["coef"].values):
                    base = feat_color(feat)
                    rgb  = mc2.to_rgb(base)
                    if coef < 0:
                        rgb = tuple(max(0, c - 0.18) for c in rgb)
                    bar_cols_d.append(rgb)
                display_d = [feat_mod_label(f)[0][:25] for f in sub_d["feature"].values]
                # Modality legend
                leg_h = [mpatches.Patch(color=c, label=m)
                         for m, c in FEAT_MOD_COLORS.items()]
                ax_d.legend(handles=leg_h, fontsize=6, frameon=False,
                            loc="lower right", title="Modality", title_fontsize=6)
            else:
                # Single modality: use the modality's own color, shade by direction
                import matplotlib.colors as mc3
                base_col = MOD_COLORS.get(mod_name, GREY)
                bar_cols_d = []
                for coef in sub_d["coef"].values:
                    rgb = mc3.to_rgb(base_col)
                    if coef < 0:
                        rgb = tuple(max(0, c - 0.18) for c in rgb)
                    bar_cols_d.append(rgb)
                display_d = [feat_mod_label(f)[0][:25] for f in sub_d["feature"].values]

            ax_d.barh(ys_d, sub_d["coef"].values, color=bar_cols_d, alpha=0.88,
                      linewidth=0)
            ax_d.set_yticks(ys_d)
            ax_d.set_yticklabels(display_d, fontsize=6.5)
            ax_d.axvline(0, color=GREY, lw=0.8)
            ax_d.set_title(mod_name, fontsize=9, fontweight="bold",
                           color=MOD_COLORS.get(mod_name, DARK) if mod_name != "All" else DARK)
            ax_d.set_xlabel("Coef (+→ event/ACR+)" if task_type=="cls"
                            else "Cox coef (+→ shorter TTE)")
            ax_d.spines["top"].set_visible(False); ax_d.spines["right"].set_visible(False)
        plt.tight_layout()
        save_fig(fig_d, f"figD_importance_{task_name.lower().replace('_','')}")
        gc.collect()


# ── Fig E: P-value forest plot — clinical features vs each task ──────────────
print("\nGenerating p-value forest plots …", flush=True)
from scipy.stats import mannwhitneyu as mwu

# Build a combined clinical + outcome DataFrame for forest plots
clin = clin_raw[["stem"] + clin_cols].copy()
clin = clin.merge(
    splits[["stem","acr_binary","acr_status","acr_days",
            "clad_status","clad_days","death_status","death_days"]].drop_duplicates("stem"),
    on="stem", how="left"
)

for task_name, task_type, label_col, time_col, event_col in TASKS:
    fig_e, axes_e = plt.subplots(1, 2, figsize=(16, max(8, 26)),
                                  gridspec_kw={"width_ratios": [2, 1]})
    fig_e.suptitle(f"Clinical Feature Associations: {task_name}\n"
                   f"Mann-Whitney U + BH FDR; effect size = |ΔCLR| / |Δ mean|",
                   fontsize=10, fontweight="bold")
    ax_f = axes_e[0]; ax_p = axes_e[1]

    if task_type == "cls":
        tmp  = clin.copy()
        gcol = "acr_binary"
        pos  = tmp[tmp[gcol]==1]; neg = tmp[tmp[gcol]==0]
        pos_lbl, neg_lbl = "ACR+", "ACR−"
    else:
        tmp = clin.copy()
        pos = tmp[tmp[event_col].fillna(0).astype(int)==1]
        neg = tmp[tmp[event_col].fillna(0).astype(int)==0]
        pos_lbl = f"{task_name}+"
        neg_lbl = f"{task_name}−"

    results_e = []
    for ci2, cn in zip(clin_cols, clin_feat_names):
        pv = pos[ci2].dropna().values; nv = neg[ci2].dropna().values
        if len(pv)<3 or len(nv)<3: continue
        try:
            _, p = mwu(pv, nv, alternative="two-sided")
            effect = pv.mean() - nv.mean()
            results_e.append({"feature": cn, "effect": effect, "p": p,
                               "mean_pos": pv.mean(), "mean_neg": nv.mean(),
                               "n_pos": len(pv), "n_neg": len(nv)})
        except Exception:
            pass
    if not results_e:
        plt.close(fig_e); continue
    res_df = pd.DataFrame(results_e)
    _, res_df["q"], _, _ = multipletests(res_df["p"], method="fdr_bh")
    res_df = res_df.sort_values("effect", ascending=False).reset_index(drop=True)
    n_show_e = min(40, len(res_df))
    top_e = pd.concat([res_df.head(n_show_e//2), res_df.tail(n_show_e//2)])
    top_e = top_e.drop_duplicates("feature").reset_index(drop=True)

    ys_e = np.arange(len(top_e))
    cols_e = [ACR_POS if v>0 else ACR_NEG for v in top_e["effect"].values]
    ax_f.barh(ys_e, top_e["effect"].values, color=cols_e, alpha=0.8, linewidth=0)
    for yi_e, (_, rw) in enumerate(top_e.iterrows()):
        if rw["q"] < 0.05:
            ax_f.text(rw["effect"]+(0.003 if rw["effect"]>=0 else -0.003),
                      yi_e, pval_stars(rw["p"]), va="center", fontsize=6,
                      color=DARK, ha="left" if rw["effect"]>=0 else "right",
                      fontweight="bold")
    ax_f.set_yticks(ys_e)
    ax_f.set_yticklabels(top_e["feature"].str[:28].values, fontsize=6)
    ax_f.axvline(0, color=GREY, lw=0.8)
    ax_f.set_xlabel(f"Mean difference ({pos_lbl} − {neg_lbl})")
    ax_f.set_title(f"Effect size (top/bottom {n_show_e//2})")

    # Dot plot of -log10(q)
    nlq_e = -np.log10(res_df["q"].clip(1e-10))
    sort_e = nlq_e.sort_values(ascending=False).head(30)
    ax_p.scatter(sort_e.values, np.arange(len(sort_e)),
                 c=[ACR_POS if res_df.loc[i,"effect"]>0 else ACR_NEG
                    for i in sort_e.index],
                 s=np.where(res_df.loc[sort_e.index,"q"]<0.05,40,15),
                 alpha=0.85, linewidths=0)
    ax_p.set_yticks(np.arange(len(sort_e)))
    ax_p.set_yticklabels(res_df.loc[sort_e.index,"feature"].str[:22].values, fontsize=6)
    ax_p.axvline(-np.log10(0.05), color=GREY, lw=0.8, ls="--", alpha=0.7)
    ax_p.set_xlabel("−log₁₀(FDR q)")
    ax_p.set_title(f"Significance (top 30)")
    n_sig_e = int((res_df["q"]<0.05).sum())
    ax_p.text(0.98,0.01,f"{n_sig_e}/{len(res_df)} FDR<5%",
              transform=ax_p.transAxes,ha="right",va="bottom",fontsize=7,color=ACR_POS)
    ax_p.spines["top"].set_visible(False); ax_p.spines["right"].set_visible(False)

    plt.tight_layout()
    save_fig(fig_e, f"figE_clinical_pvals_{task_name.lower()}")
    gc.collect()


print("\n=== Training complete ===", flush=True)
