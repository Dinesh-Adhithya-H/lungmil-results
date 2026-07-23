# pages/7_MIL_Patches.py
"""Page 7 — MIL Patch Analysis: patch counts, cluster composition, attention."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="MIL Patches", page_icon="🔬", layout="wide")

from utils.styles import (
    card_css, metric_card,
    BG, BG2, TEXT, MUTED, ACCENT, BORDER, CARD,
    ACR_COLORS, MOD_COLORS, PLOTLY_THEME,
)
from utils.data_loader import (
    load_splits, load_cluster_freq, patient_list, DATA_DIR,
)

st.markdown(card_css(), unsafe_allow_html=True)
st.markdown(f"<h2 style='color:{TEXT}'>🔬 MIL Patch Analysis</h2>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{MUTED}'>Each patient sample is a bag of patch embeddings. "
    f"HE histology patches (~1 024-d UNI), CT scan patches (~1 024-d), "
    f"BAL scRNA cells (~10-d). Cluster counts summarise patch-level composition.</p>",
    unsafe_allow_html=True,
)

splits = load_splits()

# ── Sidebar patient selector ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<p class='section-title'>Patient</p>", unsafe_allow_html=True)
    patients = patient_list()
    query = st.text_input("Search", placeholder="LT…", label_visibility="collapsed")
    filtered = [p for p in patients if query.upper() in p.upper()] if query else patients
    pid = st.selectbox("Patient", filtered, label_visibility="collapsed")

# ── Patch count distribution (cohort) ────────────────────────────────────────
st.markdown(f"<p class='section-title'>Patch Count Distribution (Cohort)</p>", unsafe_allow_html=True)

# We proxy patch counts from splits columns (has_HE, has_CT, has_BAL availability)
# and use the cluster freq tables for composition analysis.
col1, col2 = st.columns(2)

with col1:
    # Availability by ACR
    if "acr_encoded" in splits.columns:
        mod_avail_cols = [c for c in ["has_HE","has_CT","has_BAL","has_Clinical"] if c in splits.columns]
        acr_groups = {0: "ACR−", 1: "ACR+"}
        fig = go.Figure()
        for enc, lbl in acr_groups.items():
            sub = splits[splits["acr_encoded"]==enc]
            pcts = [100*sub[c].mean() for c in mod_avail_cols]
            fig.add_trace(go.Bar(
                name=lbl,
                x=[c.replace("has_","") for c in mod_avail_cols],
                y=pcts,
                marker_color=ACR_COLORS["ACR+"] if enc==1 else ACR_COLORS["ACR-"],
                opacity=0.85,
            ))
        fig.update_layout(**PLOTLY_THEME, barmode="group",
                          title="Modality Availability by ACR Status (%)",
                          yaxis_title="% of samples", height=320)
        st.plotly_chart(fig)

with col2:
    # Days since transplant: has HE vs no HE
    if "days_since_tx" in splits.columns and "has_HE" in splits.columns:
        fig2 = go.Figure()
        for has, lbl, col_hex in [(True,"HE available", MOD_COLORS["HE"]),
                                   (False,"HE missing",   MUTED)]:
            sub = splits[splits["has_HE"]==has]["days_since_tx"].dropna()
            if len(sub):
                fig2.add_trace(go.Violin(
                    y=sub.values, name=lbl,
                    box_visible=True, meanline_visible=True,
                    line_color=col_hex, fillcolor=col_hex, opacity=0.5, points=False,
                ))
        fig2.update_layout(**PLOTLY_THEME,
                           title="Days Since Transplant: HE Availability",
                           yaxis_title="Days", height=320)
        st.plotly_chart(fig2)

st.divider()

# ── Cluster composition heatmap (all patients, HE) ────────────────────────────
st.markdown(f"<p class='section-title'>HE Cluster Composition — All Patients</p>", unsafe_allow_html=True)

he_freq = load_cluster_freq("HE")
if he_freq.empty:
    st.info("HE cluster frequency data not yet available. Run `export_data.py` to generate it.")
else:
    cluster_cols = sorted([c for c in he_freq.columns if c.startswith("cluster_")],
                          key=lambda x: int(x.split("_")[1]))
    if cluster_cols and "acr_encoded" in he_freq.columns:
        # sort rows by ACR status, then mean of cluster_0
        he_sorted = he_freq.sort_values(["acr_encoded", cluster_cols[0]], ascending=[False, False])
        z = he_sorted[cluster_cols].values
        acr_labels = he_sorted["acr_encoded"].map({0:"ACR−",1:"ACR+"}).tolist() if "acr_encoded" in he_sorted.columns else [""] * len(he_sorted)
        pid_labels = he_sorted["patient_id"].tolist() if "patient_id" in he_sorted.columns else list(range(len(he_sorted)))

        # load cluster names if available
        names_p = DATA_DIR / "he_cluster_names.csv"
        col_labels = cluster_cols
        if names_p.exists():
            nm = pd.read_csv(names_p)
            if "name" in nm.columns:
                col_labels = nm["name"].tolist()[:len(cluster_cols)]

        fig3 = go.Figure(go.Heatmap(
            z=z, x=col_labels,
            y=[f"{p} ({a})" for p, a in zip(pid_labels, acr_labels)],
            colorscale="Viridis",
            colorbar=dict(title="Proportion"),
            hovertemplate="Sample: %{y}<br>Cluster: %{x}<br>Proportion: %{z:.3f}<extra></extra>",
        ))
        fig3.update_layout(**PLOTLY_THEME,
                           title=f"HE Cluster Composition (n={len(he_sorted)} samples, sorted by ACR)",
                           xaxis_title="HE Cluster", yaxis_title="Sample",
                           height=max(400, min(900, 12*len(he_sorted))),
                           yaxis=dict(showticklabels=len(he_sorted) < 80))
        st.plotly_chart(fig3)
    else:
        st.info("Cluster columns or ACR labels not found in HE frequency data.")

st.divider()

# ── Per-patient cluster composition ──────────────────────────────────────────
st.markdown(f"<p class='section-title'>Patient — Cluster Composition Over Time</p>", unsafe_allow_html=True)

_RAINBOW = [
    "#58a6ff","#3fb950","#f78166","#d2a8ff","#f2cc60","#ffa657","#79c0ff",
    "#56d364","#ff7b72","#a5d6ff","#e3b341","#ff9800","#26c6da","#ef5350",
    "#ab47bc",
]


def render_patient_mod(mod: str):
    df = load_cluster_freq(mod)
    if df.empty:
        st.info(f"No {mod} cluster data available.")
        return
    cluster_cols = sorted([c for c in df.columns if c.startswith("cluster_")],
                          key=lambda x: int(x.split("_")[1]))
    if not cluster_cols:
        return
    pid_col = "patient_id" if "patient_id" in df.columns else None
    if pid_col is None:
        st.info(f"No patient_id column in {mod} data.")
        return
    pt_rows = df[df[pid_col]==pid].copy()
    if pt_rows.empty:
        st.info(f"No {mod} data for {pid}.")
        return
    date_col = "anchor_dt" if "anchor_dt" in pt_rows.columns else None
    if date_col:
        pt_rows = pt_rows.sort_values(date_col)
        x_labels = pt_rows[date_col].astype(str).tolist()
    else:
        x_labels = [f"T{i+1}" for i in range(len(pt_rows))]

    names_p = DATA_DIR / f"{mod.lower()}_cluster_names.csv"
    col_labels = cluster_cols
    if names_p.exists():
        nm = pd.read_csv(names_p)
        if "name" in nm.columns:
            col_labels = nm["name"].tolist()[:len(cluster_cols)]
    else:
        col_labels = [c.replace("cluster_", f"{mod} C") for c in cluster_cols]

    fig = go.Figure()
    for j, (col, lbl) in enumerate(zip(cluster_cols, col_labels)):
        fig.add_trace(go.Bar(
            name=lbl, x=x_labels, y=pt_rows[col].values,
            marker_color=_RAINBOW[j % len(_RAINBOW)],
        ))
    fig.update_layout(**PLOTLY_THEME, barmode="stack",
                      title=f"{pid} — {mod} Cluster Composition Over Time",
                      xaxis_title="Timepoint", yaxis_title="Proportion",
                      yaxis_range=[0, 1], height=350,
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig)


tabs = st.tabs(["HE", "CT", "BAL"])
with tabs[0]: render_patient_mod("HE")
with tabs[1]: render_patient_mod("CT")
with tabs[2]: render_patient_mod("BAL")

st.divider()

# ── Modality availability timeline for patient ────────────────────────────────
st.markdown(f"<p class='section-title'>{pid} — Data Availability Timeline</p>", unsafe_allow_html=True)

pt_splits = splits[splits["patient_id"]==pid].sort_values("anchor_dt").copy()
if pt_splits.empty:
    st.info(f"No splits data for {pid}.")
else:
    mod_cols_avail = [c for c in ["has_Clinical","has_HE","has_CT","has_BAL"] if c in pt_splits.columns]
    dates = pt_splits["anchor_dt"].astype(str).tolist()
    fig_avail = go.Figure()
    for i, mc in enumerate(mod_cols_avail):
        lbl = mc.replace("has_","")
        y_vals = pt_splits[mc].astype(int).values
        fig_avail.add_trace(go.Scatter(
            x=dates, y=[i + v*0.5 for v in y_vals],
            mode="markers+lines",
            name=lbl,
            marker=dict(
                color=[MOD_COLORS.get(lbl, MUTED) if v else "#333" for v in y_vals],
                size=12,
                symbol=["circle" if v else "x" for v in y_vals],
            ),
            line=dict(color=MOD_COLORS.get(lbl, MUTED), width=1, dash="dot"),
        ))
    # ACR annotations
    for _, row in pt_splits.iterrows():
        if row.get("acr_encoded") == 1:
            fig_avail.add_vline(x=str(row["anchor_dt"]),
                                line_dash="dash", line_color=ACR_COLORS["ACR+"],
                                annotation_text="ACR+", annotation_position="top")
    fig_avail.update_layout(**PLOTLY_THEME,
                            title=f"{pid} — Modality Availability per Visit",
                            xaxis_title="Date",
                            yaxis=dict(tickvals=list(range(len(mod_cols_avail))),
                                       ticktext=[c.replace("has_","") for c in mod_cols_avail]),
                            height=300)
    st.plotly_chart(fig_avail)
