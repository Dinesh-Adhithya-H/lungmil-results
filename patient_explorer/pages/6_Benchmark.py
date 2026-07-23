# pages/6_Benchmark.py
"""Page 6 — Model Performance Benchmark: P1 unimodal + P2 multimodal results."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Benchmark", page_icon="📊", layout="wide")

from utils.styles import (
    card_css, metric_card,
    BG, BG2, TEXT, MUTED, ACCENT, BORDER, CARD,
    TASK_COLORS, MOD_COLORS, PLOTLY_THEME,
)
from utils.data_loader import load_benchmark_results

st.markdown(card_css(), unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df_all = load_benchmark_results()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:{TEXT}'>📊 Benchmark</h3>", unsafe_allow_html=True)
    st.divider()

    tasks_avail = sorted(df_all["task"].unique().tolist()) if not df_all.empty else []
    task_sel = st.selectbox("Task", tasks_avail, index=0 if tasks_avail else 0)

    metrics_avail = sorted(df_all["metric"].unique().tolist()) if not df_all.empty else []
    metric_sel = st.selectbox("Metric", metrics_avail, index=0 if metrics_avail else 0)

    show_splits = st.checkbox("Show per-split points", value=True)

st.markdown(f"<h2 style='color:{TEXT}'>📊 Model Performance Benchmark</h2>", unsafe_allow_html=True)

if df_all.empty:
    st.warning("benchmark_results.csv not found.")
    st.stop()

# ── Filter to selected task + metric ─────────────────────────────────────────
df = df_all[(df_all["task"] == task_sel) & (df_all["metric"] == metric_sel)].copy()
df = df.dropna(subset=["mean"])

# ── Summary metric cards ──────────────────────────────────────────────────────
if not df.empty:
    best_row = df.loc[df["mean"].idxmax()]
    p1_df = df[df["phase"] == "P1"]
    p2_df = df[df["phase"] == "P2"]
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Best model", best_row["model"], f"{best_row['mean']:.3f}"), unsafe_allow_html=True)
    c2.markdown(metric_card(f"Best {metric_sel}", f"{best_row['mean']:.3f}"), unsafe_allow_html=True)
    c3.markdown(metric_card("P1 models", str(len(p1_df))), unsafe_allow_html=True)
    c4.markdown(metric_card("P2 models", str(len(p2_df))), unsafe_allow_html=True)

st.divider()

# ── Bar chart: all models for selected task+metric ────────────────────────────
st.markdown(f"<p class='section-title'>{metric_sel} — {task_sel} (all models)</p>", unsafe_allow_html=True)

PHASE_COLORS = {"P1": MOD_COLORS["HE"], "P2": ACCENT}
MODEL_BASE_COLORS = {
    "unimodal_HE": MOD_COLORS["HE"],
    "unimodal_BAL": MOD_COLORS["BAL"],
    "unimodal_CT": MOD_COLORS["CT"],
    "unimodal_Clinical": MOD_COLORS["Clinical"],
    "early": "#58a6ff",
    "late": "#7c83ff",
    "middle": "#bc8cff",
    "set_mil_mt": "#f78166",
    "longitudinal_mk_mt": "#3fb950",
    "longitudinal_set_mil_mt": "#3fb950",
}

MODEL_DISPLAY = {
    "set_mil_mt": "SetMIL-MT",
    "longitudinal_mk_mt": "LongitudinalSetMIL",
    "longitudinal_set_mil_mt": "LongitudinalSetMIL",
    "early": "Early fusion",
    "late": "Late fusion",
    "middle": "Middle fusion",
    "unimodal_HE": "HE (unimodal)",
    "unimodal_BAL": "BAL (unimodal)",
    "unimodal_CT": "CT (unimodal)",
    "unimodal_Clinical": "Clinical (unimodal)",
}

if not df.empty:
    fig = go.Figure()

    for phase in ["P1", "P2"]:
        sub = df[df["phase"] == phase].sort_values("mean", ascending=False).copy()
        if sub.empty:
            continue
        sub["display_model"] = sub["model"].map(lambda m: MODEL_DISPLAY.get(m, m))
        bar_colors = [MODEL_BASE_COLORS.get(m, MUTED) for m in sub["model"]]
        fig.add_trace(go.Bar(
            name=f"Phase {phase}",
            x=sub["display_model"],
            y=sub["mean"],
            marker_color=bar_colors,
            error_y=dict(type="data", array=sub["std"].fillna(0).tolist(), visible=True,
                         color=TEXT, thickness=1.5, width=4),
            text=sub["mean"].round(3),
            textposition="outside",
            textfont=dict(color=TEXT, size=11),
            hovertemplate="<b>%{x}</b><br>" + metric_sel + ": %{y:.3f}<extra>Phase " + phase + "</extra>",
        ))

        # Per-split scatter overlay
        if show_splits:
            split_cols = [c for c in ["s0","s1","s2","s3","s4"] if c in sub.columns]
            for _, row in sub.iterrows():
                vals = [row[c] for c in split_cols if pd.notna(row.get(c))]
                if vals:
                    fig.add_trace(go.Scatter(
                        x=[row["display_model"]] * len(vals),
                        y=vals,
                        mode="markers",
                        marker=dict(color="white", size=5, opacity=0.7,
                                    line=dict(color=BORDER, width=1)),
                        showlegend=False,
                        hovertemplate=f"{row['display_model']}<br>split: %{{y:.3f}}<extra></extra>",
                    ))

    # Reference line at 0.5
    fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED, line_width=1,
                  annotation_text="0.5", annotation_font_color=MUTED)

    ymax = df["mean"].max()
    fig.update_layout(
        **PLOTLY_THEME,
        height=420,
        barmode="group",
        xaxis_title="Model",
        yaxis_title=metric_sel,
        yaxis=dict(range=[max(0, df["mean"].min() - 0.1), min(1.0, ymax + 0.12)]),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig)
else:
    st.info(f"No data for task={task_sel}, metric={metric_sel}.")

st.divider()

# ── P1 vs P2 grouped heatmap: all tasks × models ──────────────────────────────
st.markdown(f"<p class='section-title'>Overview heatmap — {metric_sel} across all tasks</p>", unsafe_allow_html=True)

df_metric = df_all[df_all["metric"] == metric_sel].dropna(subset=["mean"])
if not df_metric.empty:
    tasks_all   = sorted(df_metric["task"].unique())
    models_all  = df_metric.groupby("model")["mean"].mean().sort_values(ascending=False).index.tolist()
    display_models = [MODEL_DISPLAY.get(m, m) for m in models_all]
    z = np.full((len(models_all), len(tasks_all)), np.nan)
    text_z = [[""] * len(tasks_all) for _ in range(len(models_all))]
    for i, m in enumerate(models_all):
        for j, t in enumerate(tasks_all):
            row = df_metric[(df_metric["model"] == m) & (df_metric["task"] == t)]
            if not row.empty:
                v = row["mean"].values[0]
                z[i, j] = v
                text_z[i][j] = f"{v:.3f}"

    fig_hm = go.Figure(go.Heatmap(
        z=z, x=tasks_all, y=display_models,
        colorscale="RdYlGn", zmid=0.5, zmin=0.3, zmax=0.85,
        text=text_z, texttemplate="%{text}",
        colorbar=dict(title=metric_sel, tickfont=dict(color=TEXT)),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>",
    ))
    fig_hm.update_layout(
        **PLOTLY_THEME,
        height=80 + 35 * len(models_all),
        xaxis_title="Task",
        yaxis_title="Model",
    )
    st.plotly_chart(fig_hm)

st.divider()

# ── Full results table ────────────────────────────────────────────────────────
st.markdown(f"<p class='section-title'>Full results table</p>", unsafe_allow_html=True)

display_df = df_all.copy()
split_cols = [c for c in ["s0","s1","s2","s3","s4"] if c in display_df.columns]
for c in ["mean","std"] + split_cols:
    if c in display_df.columns:
        display_df[c] = display_df[c].round(4)

col_filter = st.multiselect("Filter tasks", sorted(df_all["task"].unique()),
                             default=sorted(df_all["task"].unique()))
if col_filter:
    display_df = display_df[display_df["task"].isin(col_filter)]

display_df = display_df.sort_values(["phase","task","mean"], ascending=[True,True,False])
st.dataframe(
    display_df.style.background_gradient(subset=["mean"], cmap="RdYlGn", vmin=0.3, vmax=0.85),
    width="stretch",
    height=500,
    hide_index=True,
)
