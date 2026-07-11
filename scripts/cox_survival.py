"""
cox_survival.py — Cox PH multimodal survival analysis on lung-transplant data.

SEC 10: Univariate + multivariate Cox, KM stratification, time-dependent AUC,
         random survival forest feature importances.
Outputs:
  cox_results.csv, survival_risk_scores.csv
  fig10_cox_clad_univariate.png, fig10_cox_death_univariate.png,
  fig10_cox_km_clad.png, fig10_cox_km_death.png,
  fig10_cox_tdauc.png, fig10_cox_rsf.png
"""

import warnings, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

OUT_DIR = Path("/home/aih/dinesh.haridoss/chicago_mil/analysis/nature_paper")
PCA_CSV = OUT_DIR / "pca_scores.csv"

NATURE_STYLE = {
    "font.family": "sans-serif", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
}
plt.rcParams.update(NATURE_STYLE)

def savefig(fig, name):
    p = OUT_DIR / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved → {p.name}", flush=True)

# ── load PCA scores ────────────────────────────────────────────────────────────
print("="*65)
print("COX PH MULTIMODAL SURVIVAL ANALYSIS")
print("="*65, flush=True)

if not PCA_CSV.exists():
    print(f"  ERROR: {PCA_CSV} not found — run nature_analysis.py first", flush=True)
    raise SystemExit(1)

pca = pd.read_csv(PCA_CSV)
print(f"  Loaded PCA scores: {pca.shape}", flush=True)
print(f"  Columns: {list(pca.columns[:20])}", flush=True)

# ── per-patient summary (take first timepoint for survival endpoints) ──────────
# Outcomes: use first non-null clad_days/clad_binary per patient
def first_valid(series):
    v = series.dropna()
    return v.iloc[0] if len(v) > 0 else np.nan

pc_cols = [c for c in pca.columns if c.startswith("PC")]
if not pc_cols:
    # try to find any factor-like columns
    pc_cols = [c for c in pca.columns if any(k in c for k in ["PC","Factor","factor"])]

feature_cols = pc_cols[:min(10, len(pc_cols))]
print(f"  Feature columns: {feature_cols}", flush=True)

# Aggregate per patient: mean PC scores, first survival endpoint
group_cols = ["patient_id"] if "patient_id" in pca.columns else ["stem"]
pt_agg = {}
for pid, grp in pca.groupby(group_cols[0]):
    row = {"patient_id": pid}
    for fc in feature_cols:
        if fc in grp.columns:
            row[fc] = grp[fc].mean()
    for col in ["clad_binary","clad_days","death_binary","death_days","acr_binary"]:
        if col in grp.columns:
            row[col] = first_valid(grp[col])
    pt_agg[pid] = row

pt_df = pd.DataFrame(list(pt_agg.values()))
print(f"  Patient-level rows: {len(pt_df)}", flush=True)

# ── lifelines Cox PH ──────────────────────────────────────────────────────────
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter

def run_cox_univariate(pt_df, feature_cols, dur_col, event_col, outcome_name):
    results = []
    for fc in feature_cols:
        sub = pt_df[[fc, dur_col, event_col]].dropna()
        if len(sub) < 10 or sub[event_col].sum() < 3:
            continue
        # normalise feature
        sub = sub.copy()
        sub[fc] = (sub[fc] - sub[fc].mean()) / (sub[fc].std() + 1e-8)
        try:
            cph = CoxPHFitter()
            cph.fit(sub, duration_col=dur_col, event_col=event_col, show_progress=False)
            s = cph.summary
            results.append({
                "feature":  fc,
                "outcome":  outcome_name,
                "HR":       float(np.exp(s.loc[fc, "coef"])),
                "HR_lower": float(np.exp(s.loc[fc, "coef lower 95%"])),
                "HR_upper": float(np.exp(s.loc[fc, "coef upper 95%"])),
                "pval":     float(s.loc[fc, "p"]),
                "c_index":  float(cph.concordance_index_),
            })
        except Exception as e:
            print(f"    Cox {fc}/{outcome_name}: {e}", flush=True)
    return pd.DataFrame(results)

all_cox = []
for dur_col, event_col, name in [
    ("clad_days",  "clad_binary",  "CLAD"),
    ("death_days", "death_binary", "Death"),
]:
    if dur_col not in pt_df.columns or event_col not in pt_df.columns:
        continue
    sub_df = pt_df[pt_df[dur_col].notna() & pt_df[event_col].notna()].copy()
    sub_df[dur_col]   = sub_df[dur_col].astype(float)
    sub_df[event_col] = sub_df[event_col].astype(float)
    sub_df = sub_df[sub_df[dur_col] > 0]
    res = run_cox_univariate(sub_df, feature_cols, dur_col, event_col, name)
    all_cox.append(res)
    print(f"  Univariate Cox {name}: {len(res)} features fitted", flush=True)

cox_df = pd.concat(all_cox, ignore_index=True) if all_cox else pd.DataFrame()
cox_df.to_csv(OUT_DIR / "cox_results.csv", index=False)
print("  cox_results.csv saved", flush=True)

# ── Forest plots ───────────────────────────────────────────────────────────────
def forest_plot(res_df, title, fname):
    if res_df.empty:
        return
    res_df = res_df.sort_values("HR", ascending=True).reset_index(drop=True)
    n = len(res_df)
    fig, ax = plt.subplots(figsize=(6, max(3, n * 0.35 + 1)))
    y = np.arange(n)
    ax.scatter(res_df["HR"], y, color="#D32F2F", s=30, zorder=3)
    for i, row in res_df.iterrows():
        ax.plot([row["HR_lower"], row["HR_upper"]], [i, i],
                color="#D32F2F", lw=1.5, zorder=2)
    ax.axvline(1.0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(res_df["feature"].str.replace("_", " "), fontsize=7)
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title(title)
    # significance markers
    for i, row in res_df.iterrows():
        if row["pval"] < 0.05:
            ax.text(row["HR_upper"] + 0.02, i, "*", fontsize=8, va="center", color="black")
    savefig(fig, fname)

if not cox_df.empty:
    for name, fname in [("CLAD", "fig10_cox_clad_univariate"), ("Death", "fig10_cox_death_univariate")]:
        sub = cox_df[cox_df["outcome"] == name]
        forest_plot(sub, f"Figure 10: Univariate Cox HR — {name}", fname)

# ── KM stratification by median risk ──────────────────────────────────────────
risk_rows = []
for dur_col, event_col, name, fname in [
    ("clad_days",  "clad_binary",  "CLAD",  "fig10_cox_km_clad"),
    ("death_days", "death_binary", "Death", "fig10_cox_km_death"),
]:
    if dur_col not in pt_df.columns or event_col not in pt_df.columns:
        continue
    sub_df = pt_df[pt_df[dur_col].notna() & pt_df[event_col].notna()].copy()
    sub_df[dur_col]   = sub_df[dur_col].astype(float)
    sub_df[event_col] = sub_df[event_col].astype(float)
    sub_df = sub_df[sub_df[dur_col] > 0]
    valid_feats = [f for f in feature_cols if f in sub_df.columns and sub_df[f].notna().sum() > 10]
    if not valid_feats:
        continue

    # Multivariate linear predictor
    mat = sub_df[valid_feats].fillna(0).values
    mat = (mat - mat.mean(0)) / (mat.std(0) + 1e-8)
    try:
        from sklearn.decomposition import PCA as _PCA
        lp = _PCA(n_components=1, random_state=42).fit_transform(mat).ravel()
    except Exception:
        lp = mat[:, 0]

    sub_df["risk_score"] = lp
    sub_df["risk_group"] = (lp >= np.median(lp)).astype(int)

    # Store risk scores
    for _, row in sub_df.iterrows():
        risk_rows.append({
            "patient_id": row.get("patient_id", ""),
            "outcome": name,
            "risk_score": row["risk_score"],
            "risk_group": "High" if row["risk_group"] == 1 else "Low",
        })

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"High": "#D32F2F", "Low": "#1976D2"}
    for grp_val, grp_label, color in [(1, "High risk", "#D32F2F"), (0, "Low risk", "#1976D2")]:
        mask = sub_df["risk_group"] == grp_val
        kmf = KaplanMeierFitter()
        if mask.sum() < 3:
            continue
        kmf.fit(sub_df.loc[mask, dur_col], sub_df.loc[mask, event_col], label=grp_label)
        kmf.plot_survival_function(ax=ax, color=color, ci_show=True)
    ax.set_xlabel("Days post-transplant")
    ax.set_ylabel("Event-free probability")
    ax.set_title(f"Figure 10: KM — {name} by multimodal risk score")
    ax.legend(fontsize=7)

    # log-rank test
    try:
        from lifelines.statistics import logrank_test
        hi = sub_df[sub_df["risk_group"] == 1]
        lo = sub_df[sub_df["risk_group"] == 0]
        lr = logrank_test(hi[dur_col], lo[dur_col], hi[event_col], lo[event_col])
        ax.text(0.65, 0.05, f"Log-rank p={lr.p_value:.3f}", transform=ax.transAxes, fontsize=7)
    except Exception:
        pass
    savefig(fig, fname)

if risk_rows:
    pd.DataFrame(risk_rows).to_csv(OUT_DIR / "survival_risk_scores.csv", index=False)
    print("  survival_risk_scores.csv saved", flush=True)

# ── Time-dependent AUC ────────────────────────────────────────────────────────
print("\nComputing time-dependent AUC ...", flush=True)
try:
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv

    fig, ax = plt.subplots(figsize=(6, 4))
    time_points = np.array([90, 180, 365, 730, 1095])
    plotted = False
    for dur_col, event_col, name, color in [
        ("clad_days",  "clad_binary",  "CLAD",  "#F44336"),
        ("death_days", "death_binary", "Death", "#9C27B0"),
    ]:
        if dur_col not in pt_df.columns:
            continue
        sub_df = pt_df[pt_df[dur_col].notna() & pt_df[event_col].notna()].copy()
        sub_df = sub_df[sub_df[dur_col] > 0]
        valid_feats = [f for f in feature_cols if f in sub_df.columns and sub_df[f].notna().sum() > 10]
        if not valid_feats or len(sub_df) < 20:
            continue
        scores = sub_df[valid_feats].fillna(0).mean(axis=1).values
        y = Surv.from_arrays(sub_df[event_col].astype(bool), sub_df[dur_col].astype(float))
        tps = time_points[time_points < sub_df[dur_col].max() * 0.9]
        if len(tps) < 2:
            continue
        try:
            aucs, mean_auc = cumulative_dynamic_auc(y, y, scores, tps)
            ax.plot(tps, aucs, marker="o", label=f"{name} (mean={mean_auc:.2f})", color=color)
            plotted = True
        except Exception as e:
            print(f"    td-AUC {name}: {e}", flush=True)
    if plotted:
        ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="Random")
        ax.set_xlabel("Days"); ax.set_ylabel("Time-dependent AUC")
        ax.set_title("Figure 10e: Time-dependent AUC — multimodal risk score")
        ax.legend(fontsize=7)
        savefig(fig, "fig10_cox_tdauc")
    else:
        plt.close(fig)
except Exception as e:
    print(f"  td-AUC skipped: {e}", flush=True)

# ── Random Survival Forest ─────────────────────────────────────────────────────
print("\nRandom Survival Forest ...", flush=True)
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv

    dur_col, event_col, name = "clad_days", "clad_binary", "CLAD"
    if dur_col in pt_df.columns:
        sub_df = pt_df[pt_df[dur_col].notna() & pt_df[event_col].notna()].copy()
        sub_df = sub_df[sub_df[dur_col] > 0]
        valid_feats = [f for f in feature_cols if f in sub_df.columns and sub_df[f].notna().sum() > 5]
        X_rsf = sub_df[valid_feats].fillna(0)
        y_rsf = Surv.from_arrays(sub_df[event_col].astype(bool).values,
                                  sub_df[dur_col].astype(float).values)
        if len(X_rsf) >= 20 and y_rsf["Status"].sum() >= 5:
            rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=5,
                                       random_state=42, n_jobs=4)
            rsf.fit(X_rsf, y_rsf)
            importances = rsf.feature_importances_
            order = np.argsort(importances)[::-1]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(range(len(order)), importances[order],
                    color=plt.cm.viridis(np.linspace(0, 0.8, len(order))))
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([valid_feats[i].replace("_", " ") for i in order], fontsize=7)
            ax.set_xlabel("Feature importance")
            ax.set_title("Figure 10f: Random Survival Forest — feature importances (CLAD)")
            savefig(fig, "fig10_cox_rsf")
except Exception as e:
    print(f"  RSF skipped: {e}", flush=True)

print("\nCox survival analysis COMPLETE", flush=True)
