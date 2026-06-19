# pages/6_Benchmark.py
"""Page 6 — Model Performance Benchmark across all tasks, folds, and modalities."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Benchmark", page_icon="📊", layout="wide")

from utils.styles import (
    card_css, metric_card,
    BG, BG2, TEXT, MUTED, ACCENT, BORDER, CARD,
    TASK_COLORS, MOD_COLORS, PLOTLY_THEME,
)
from utils.data_loader import DATA_DIR

st.markdown(card_css(), unsafe_allow_html=True)
st.markdown(f"<h2 style='color:{TEXT}'>📊 Model Performance Benchmark</h2>", unsafe_allow_html=True)

# ── Load benchmark data ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_benchmark() -> pd.DataFrame:
    p = DATA_DIR / "benchmark_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_baselines() -> dict:
    p = DATA_DIR.parent / "results" / "mm_abmil_v8" / "baselines_summary.json"
    alt = DATA_DIR / "baselines_summary.json"
    for path in [p, alt]:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_fold_metrics() -> pd.DataFrame:
    """Parse all per-fold metrics_*.json files into a single DataFrame."""
    results_dir = DATA_DIR.parent / "results" / "mm_abmil_v8"
    if not results_dir.exists():
        return pd.DataFrame()
    rows = []
    for jf in results_dir.glob("metrics_split*_fold*_*.json"):
        parts = jf.stem.split("_")
        try:
            fold_idx = next(i for i, p in enumerate(parts) if p.startswith("fold"))
            fold = int(parts[fold_idx][4:])
            variant_task = "_".join(parts[fold_idx + 1:])
            # variant_task like "late_cls" or "middle_acr_surv"
            variant_parts = variant_task.split("_")
            variant = variant_parts[0]
            task = "_".join(variant_parts[1:]) if len(variant_parts) > 1 else "unknown"
        except Exception:
            continue
        with open(jf) as f:
            d = json.load(f)
        td = d.get("test", d)
        row = {"fold": fold, "variant": variant, "task": task,
               "model": f"{variant}_{task}"}
        row["auc"]     = td.get("auc",     np.nan)
        row["auprc"]   = td.get("auprc",   np.nan)
        row["bacc"]    = td.get("bacc",    np.nan)
        row["c_index"] = td.get("c_index", np.nan)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


bench  = load_benchmark()
folds  = load_fold_metrics()
blines = load_baselines()

# ── Build unified metrics table ───────────────────────────────────────────────
if not folds.empty:
    mean_df = (folds.groupby(["variant","task"])
               [["auc","auprc","bacc","c_index"]]
               .mean().reset_index())
    mean_df["model"] = mean_df["variant"] + " + " + mean_df["task"]
else:
    mean_df = bench.copy() if not bench.empty else pd.DataFrame()

# Baselines rows
baseline_rows = []
if blines:
    for name, vals in blines.items():
        if isinstance(vals, dict):
            baseline_rows.append({
                "model": name, "variant": "classical", "task": "multi",
                "auc":     vals.get("test_auc",  np.nan),
                "bacc":    vals.get("test_bacc", np.nan),
                "c_index": vals.get("test_ci_acr", np.nan),
                "auprc":   np.nan,
            })
baseline_df = pd.DataFrame(baseline_rows) if baseline_rows else pd.DataFrame()

# ── Summary metric cards ──────────────────────────────────────────────────────
st.markdown(f"<p class='section-title'>Best Performance</p>", unsafe_allow_html=True)

if not mean_df.empty:
    best_auc   = mean_df["auc"].max()
    best_model = mean_df.loc[mean_df["auc"].idxmax(), "model"]
    best_bacc  = mean_df["bacc"].max()
    best_ci    = mean_df["c_index"].max()
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(metric_card("Best AUROC", f"{best_auc:.3f}", best_model), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Best BACC",  f"{best_bacc:.3f}"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Best C-index", f"{best_ci:.3f}"), unsafe_allow_html=True)
    with c4: st.markdown(metric_card("# Models evaluated", str(len(mean_df))), unsafe_allow_html=True)
else:
    st.info("No benchmark data found. Results will appear after training completes.")

st.divider()

# ── AUROC heatmap ─────────────────────────────────────────────────────────────
st.markdown(f"<p class='section-title'>AUROC Heatmap (model × task)</p>", unsafe_allow_html=True)

if not mean_df.empty and "task" in mean_df.columns and "variant" in mean_df.columns:
    tasks    = mean_df["task"].unique().tolist()
    variants = mean_df["variant"].unique().tolist()
    z = np.full((len(variants), len(tasks)), np.nan)
    for i, v in enumerate(variants):
        for j, t in enumerate(tasks):
            row = mean_df[(mean_df["variant"]==v) & (mean_df["task"]==t)]
            if len(row):
                z[i, j] = row["auc"].values[0]
    fig_hm = go.Figure(go.Heatmap(
        z=z, x=tasks, y=variants,
        colorscale="RdYlGn",
        zmid=0.6, zmin=0.45, zmax=0.85,
        text=np.where(np.isnan(z), "", np.round(z, 3).astype(str)),
        texttemplate="%{text}",
        colorbar=dict(title="AUROC"),
        hovertemplate="Model: %{y}<br>Task: %{x}<br>AUROC: %{z:.3f}<extra></extra>",
    ))
    fig_hm.update_layout(**PLOTLY_THEME, height=300 + 30*len(variants),
                         xaxis_title="Task", yaxis_title="Fusion variant")
    st.plotly_chart(fig_hm, use_container_width=True)
else:
    st.info("Fold metrics not yet available. Run training to generate results.")

st.divider()

# ── Unimodal vs Multimodal comparison ────────────────────────────────────────
st.markdown(f"<p class='section-title'>Unimodal vs Multimodal — ACR Classification</p>", unsafe_allow_html=True)

# Pull unimodal baselines from baselines JSON
unimodal_data = []
for key in ["HE_acr","BAL_acr","CT_acr","Clinical_acr"]:
    if blines.get("unimodal_baselines", {}).get(key):
        v = blines["unimodal_baselines"][key]
        unimodal_data.append({"model": key.replace("_acr",""), "auc": v.get("auc",np.nan),
                               "type": "Unimodal"})
# Pull unimodal_ablation from fold metrics json if available
fold0_cls = DATA_DIR.parent / "results" / "mm_abmil_v8" / "metrics_split1_fold0_late_cls.json"
if fold0_cls.exists():
    with open(fold0_cls) as f:
        d0 = json.load(f)
    for mod, vals in d0.get("unimodal_ablation", {}).items():
        unimodal_data.append({"model": f"{mod} (ablation)", "auc": vals.get("auc",np.nan),
                               "type": "Unimodal (ablation)"})

# Add multimodal
if not mean_df.empty:
    for _, row in mean_df[mean_df["task"].str.contains("cls", na=False)].iterrows():
        unimodal_data.append({"model": row["variant"]+" fusion",
                               "auc": row["auc"], "type": "Multimodal"})

if unimodal_data:
    um_df = pd.DataFrame(unimodal_data).dropna(subset=["auc"])
    color_map = {"Unimodal": MUTED, "Unimodal (ablation)": MOD_COLORS["HE"],
                 "Multimodal": ACCENT}
    fig_um = go.Figure()
    for typ in um_df["type"].unique():
        sub = um_df[um_df["type"]==typ]
        fig_um.add_trace(go.Bar(
            name=typ, x=sub["model"], y=sub["auc"],
            marker_color=color_map.get(typ, ACCENT),
            text=sub["auc"].round(3), textposition="outside",
        ))
    fig_um.add_hline(y=0.5, line_dash="dash", line_color=MUTED, annotation_text="Random")
    fig_um.update_layout(**PLOTLY_THEME, barmode="group",
                         yaxis_title="AUROC", yaxis_range=[0.4, 0.9],
                         title="ACR Classification: Modality Comparison", height=380)
    st.plotly_chart(fig_um, use_container_width=True)
else:
    st.info("Unimodal comparison data not yet available.")

st.divider()

# ── Fold-level variance (violin) ─────────────────────────────────────────────
st.markdown(f"<p class='section-title'>Cross-fold Variance</p>", unsafe_allow_html=True)

if not folds.empty:
    metric_sel = st.selectbox("Metric", ["auc", "bacc", "c_index", "auprc"], key="violin_metric")
    task_sel   = st.multiselect("Tasks", folds["task"].unique().tolist(),
                                 default=folds["task"].unique().tolist()[:4],
                                 key="violin_tasks")
    folds_sub = folds[folds["task"].isin(task_sel)] if task_sel else folds

    fig_viol = go.Figure()
    for var in folds_sub["variant"].unique():
        sub = folds_sub[(folds_sub["variant"]==var)][metric_sel].dropna()
        if len(sub):
            fig_viol.add_trace(go.Violin(
                y=sub.values, name=var,
                box_visible=True, meanline_visible=True,
                points="all", jitter=0.3,
                opacity=0.8,
            ))
    fig_viol.update_layout(**PLOTLY_THEME, yaxis_title=metric_sel.upper(),
                           title=f"Cross-fold {metric_sel.upper()} Distribution",
                           height=380)
    st.plotly_chart(fig_viol, use_container_width=True)

st.divider()

# ── Full metrics table ────────────────────────────────────────────────────────
st.markdown(f"<p class='section-title'>All Results Table</p>", unsafe_allow_html=True)

all_rows = []
if not mean_df.empty:
    all_rows.append(mean_df[["model","task","auc","auprc","bacc","c_index"]])
if not baseline_df.empty:
    all_rows.append(baseline_df[["model","task","auc","auprc","bacc","c_index"]])

if all_rows:
    all_df = pd.concat(all_rows, ignore_index=True)
    sort_col = st.selectbox("Sort by", ["auc","auprc","bacc","c_index"], key="sort_metric")
    all_df = all_df.sort_values(sort_col, ascending=False)
    for col in ["auc","auprc","bacc","c_index"]:
        if col in all_df.columns:
            all_df[col] = all_df[col].round(4)
    st.dataframe(
        all_df.style.background_gradient(subset=["auc"], cmap="RdYlGn", vmin=0.4, vmax=0.85),
        use_container_width=True, height=450,
    )
else:
    st.info("No metrics data found yet.")
