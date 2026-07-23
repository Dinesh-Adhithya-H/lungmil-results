"""Page 2 — Cluster Composition (HE / BAL / CT)."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Cluster Composition", page_icon="🔬", layout="wide")

from utils.styles import card_css, BG, BG2, TEXT, MUTED, ACCENT, BORDER
from utils.data_loader import (
    patient_splits, load_cluster_freq, cohort_cluster_freq_mean,
)
from utils.plots import cluster_freq_figure

st.markdown(card_css(), unsafe_allow_html=True)

pid = st.session_state.get("selected_patient")
if not pid:
    st.warning("Select a patient from the home page.")
    st.stop()

st.markdown(f"<h2 style='color:{TEXT}'>🔬 Cluster Composition — {pid}</h2>", unsafe_allow_html=True)

splits = patient_splits(pid)


def _render_mod(mod: str, col_label: str):
    df = load_cluster_freq(mod)
    if df.empty:
        st.info(f"No {mod} cluster frequency data. Run `export_data.py` to generate it.")
        return

    cluster_cols = sorted([c for c in df.columns if c.startswith("cluster_")],
                          key=lambda x: int(x.split("_")[1]))
    if not cluster_cols:
        st.warning(f"No cluster columns found in {mod} data.")
        return

    K = len(cluster_cols)
    # cluster names from column header or generic
    cluster_names = [c.replace("cluster_", f"C") for c in cluster_cols]
    if f"{mod}_cluster_names" in df.columns:
        cluster_names = df[f"{mod}_cluster_names"].iloc[0] \
            if isinstance(df[f"{mod}_cluster_names"].iloc[0], list) else cluster_names

    # patient row(s) — pick closest timepoint if multiple
    if "patient_id" in df.columns:
        pt_rows = df[df["patient_id"] == pid]
    else:
        pt_rows = pd.DataFrame()

    # timepoint selector
    if "anchor_dt" in pt_rows.columns and len(pt_rows) > 1:
        options = pt_rows["anchor_dt"].dt.strftime("%Y-%m-%d").tolist()
        sel_dt  = st.selectbox(f"{mod} — Select timepoint", options,
                               key=f"tp_{mod}")
        pt_row = pt_rows[pt_rows["anchor_dt"].dt.strftime("%Y-%m-%d") == sel_dt]
    else:
        pt_row = pt_rows

    patient_freq = pt_row[cluster_cols].iloc[0] if len(pt_row) > 0 else None

    # cohort means per ACR group
    cohort_means = {}
    if "acr_encoded" in df.columns:
        for grp, sub in df.groupby("acr_encoded"):
            label = "ACR+" if grp == 1 else "ACR−"
            cohort_means[label] = sub[cluster_cols].mean()

    fig = cluster_freq_figure(patient_freq, cohort_means, cluster_names, mod)
    st.plotly_chart(fig)

    # stacked bar: all timepoints for this patient
    if len(pt_rows) > 1 and "anchor_dt" in pt_rows.columns:
        st.markdown(f"<p class='section-title'>{mod} — All timepoints for {pid}</p>",
                    unsafe_allow_html=True)
        fig2 = go.Figure()
        dates = pt_rows["anchor_dt"].dt.strftime("%Y-%m-%d").tolist()
        palette = _rainbow(K)
        for i, (col, name) in enumerate(zip(cluster_cols, cluster_names)):
            fig2.add_trace(go.Bar(
                name=name,
                x=dates,
                y=pt_rows[col].values * 100,
                marker_color=palette[i],
                hovertemplate=f"<b>{name}</b>: %{{y:.1f}}%<br>%{{x}}<extra></extra>",
            ))
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG2,
            barmode="stack", height=320,
            margin=dict(l=20, r=20, t=20, b=60),
            xaxis=dict(tickangle=-30, gridcolor=BORDER),
            yaxis=dict(title="% patches", gridcolor=BORDER),
            showlegend=False,
        )
        st.plotly_chart(fig2)


def _rainbow(n: int):
    import plotly.express as px
    return px.colors.sample_colorscale("Turbo", [i / max(n - 1, 1) for i in range(n)])


# ── Tabs per modality ──────────────────────────────────────────────────────
tab_he, tab_bal, tab_ct = st.tabs(["🧫 HE (Histology)", "💧 BAL (Cell Types)", "🫁 CT (Radiomics)"])

with tab_he:
    _render_mod("HE", "HE")
with tab_bal:
    _render_mod("BAL", "BAL")
with tab_ct:
    _render_mod("CT", "CT")

# ── Radar chart: all modalities combined ──────────────────────────────────
st.divider()
st.markdown(f"<p class='section-title'>Multi-Modal Profile (top clusters)</p>",
            unsafe_allow_html=True)

# Use top 3 clusters per modality for a combined radar
TOP = 4
radar_cats, radar_vals = [], []
for mod in ["HE", "BAL", "CT"]:
    df = load_cluster_freq(mod)
    if df.empty:
        continue
    cc = sorted([c for c in df.columns if c.startswith("cluster_")],
                key=lambda x: int(x.split("_")[1]))
    if "patient_id" in df.columns:
        pt = df[df["patient_id"] == pid]
    else:
        pt = pd.DataFrame()
    if pt.empty:
        continue
    vals = pt[cc].mean()  # mean over timepoints
    top_idx = vals.nlargest(TOP).index
    for idx in top_idx:
        radar_cats.append(f"{mod}·{idx.replace('cluster_','C')}")
        radar_vals.append(float(vals[idx]) * 100)

if radar_cats:
    fig_radar = go.Figure(go.Scatterpolar(
        r=radar_vals + [radar_vals[0]],
        theta=radar_cats + [radar_cats[0]],
        fill="toself",
        fillcolor=f"rgba(124,131,255,0.25)",
        line=dict(color=ACCENT, width=2),
        name=pid,
    ))
    fig_radar.update_layout(
        template="plotly_dark", paper_bgcolor=BG,
        polar=dict(
            bgcolor=BG2,
            radialaxis=dict(color=MUTED, gridcolor=BORDER),
            angularaxis=dict(color=TEXT, gridcolor=BORDER),
        ),
        height=380,
        margin=dict(l=40, r=40, t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_radar)
else:
    st.info("Cluster frequency data not yet available. Run `export_data.py`.")
