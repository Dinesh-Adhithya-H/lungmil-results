"""Page 10 — Multi-modal Factor Analysis: PCA, cross-modal correlation, loadings."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Factor Analysis", page_icon="📊", layout="wide")

from utils.styles import card_css, BG, BG2, TEXT, MUTED, ACCENT, BORDER, CARD, PLOTLY_THEME
from utils.data_loader import (
    load_pca_scores, load_cross_modal_corr, load_splits,
    patient_list, patient_splits,
)

st.markdown(card_css(), unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"<h3 style='color:{TEXT}'>📊 Factor Analysis</h3>", unsafe_allow_html=True)
    patients = patient_list()
    default_idx = patients.index(st.session_state.get("selected_patient", patients[0])) \
        if st.session_state.get("selected_patient") in patients else 0
    pid = st.selectbox("Highlight patient", patients, index=default_idx)
    st.session_state["selected_patient"] = pid
    st.divider()
    color_by = st.selectbox("Colour PC scatter by",
                             ["ACR status","CLAD status","Mortality","Days post-transplant"])

st.markdown(f"<h2 style='color:{TEXT}'>📊 Multi-modal Factor Analysis</h2>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{MUTED}'>Joint PCA of clinical + HE + CT + radiomics features across all samples.</p>",
            unsafe_allow_html=True)

pca_df  = load_pca_scores()
corr_df = load_cross_modal_corr()
all_splits = load_splits()

# ── PCA scatter ───────────────────────────────────────────────────────────
if pca_df.empty:
    st.info("PCA scores not yet available. Run `nature_analysis.sh` to generate `pca_scores.csv`.")
else:
    # merge outcome cols
    merge_cols = ["patient_id","anchor_dt","days_since_tx","acr_encoded","clad_status","death_status"] \
        if all(c in all_splits.columns for c in ["patient_id","days_since_tx"]) \
        else [c for c in ["patient_id","acr_encoded","clad_status","death_status"] if c in all_splits.columns]

    all_splits2 = all_splits.copy()
    all_splits2["days_since_tx"] = (
        all_splits2["anchor_dt"] - all_splits2.groupby("patient_id")["anchor_dt"].transform("min")
    ).dt.days

    pca_m = pca_df.merge(
        all_splits2[list(set(merge_cols))].drop_duplicates(subset=["patient_id"] +
            (["anchor_dt"] if "anchor_dt" in merge_cols else [])),
        on=[c for c in ["patient_id","anchor_dt"] if c in pca_df.columns and c in all_splits2.columns],
        how="left",
    ) if "patient_id" in pca_df.columns else pca_df.copy()

    pc_cols = [c for c in pca_m.columns if c.startswith("PC")]
    if len(pc_cols) < 2:
        st.warning("Need at least PC1 and PC2 in pca_scores.csv.")
    else:
        col_map = {
            "ACR status":            ("acr_binary",   "ACR",   {"0":"#58A6FF","1":"#F85149"}),
            "CLAD status":           ("clad_binary",  "CLAD",  {"0":"#3FB950","1":"#F78166"}),
            "Mortality":             ("death_binary", "Death", {"0":"#8BC34A","1":"#9C27B0"}),
            "Days post-transplant":  ("days_since_tx","Days",  None),
        }
        col_key, col_name, cat_map = col_map[color_by]

        # PC axis selectors
        c1, c2 = st.columns(2)
        with c1: x_pc = st.selectbox("X axis", pc_cols, index=0)
        with c2: y_pc = st.selectbox("Y axis", pc_cols, index=min(1, len(pc_cols)-1))

        fig_sc = go.Figure()
        if cat_map is not None and col_key in pca_m.columns:
            for val, color in cat_map.items():
                mask = pca_m[col_key].astype(str) == val
                m = pca_m[mask]
                is_pid = m["patient_id"] == pid if "patient_id" in m.columns else pd.Series([False]*len(m))
                fig_sc.add_trace(go.Scatter(
                    x=m.loc[~is_pid, x_pc], y=m.loc[~is_pid, y_pc],
                    mode="markers", name=f"{col_name}={val}",
                    marker=dict(color=color, size=5, opacity=0.6, line=dict(width=0)),
                    hovertemplate=f"<b>%{{text}}</b><br>{x_pc}: %{{x:.2f}}<br>{y_pc}: %{{y:.2f}}<extra></extra>",
                    text=m.loc[~is_pid,"patient_id"].values if "patient_id" in m.columns else None,
                ))
                # selected patient highlighted
                if is_pid.any():
                    fig_sc.add_trace(go.Scatter(
                        x=m.loc[is_pid, x_pc], y=m.loc[is_pid, y_pc],
                        mode="markers+text", name=pid,
                        marker=dict(color="#FFD700", size=12, symbol="star",
                                    line=dict(color="white", width=1)),
                        text=[pid]*is_pid.sum(), textposition="top center",
                    ))
        elif col_key in pca_m.columns:
            fig_sc.add_trace(go.Scatter(
                x=pca_m[x_pc], y=pca_m[y_pc], mode="markers",
                marker=dict(color=pca_m[col_key].values, colorscale="Viridis",
                            size=5, showscale=True,
                            colorbar=dict(title=col_name)),
                hovertemplate=f"{x_pc}: %{{x:.2f}}<br>{y_pc}: %{{y:.2f}}<extra></extra>",
            ))

        fig_sc.update_layout(**PLOTLY_THEME, height=480,
                             xaxis_title=x_pc, yaxis_title=y_pc,
                             legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1))
        st.plotly_chart(fig_sc, use_container_width=True)

        # PC–outcome correlation heatmap
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<p class='section-title'>PC–Outcome Correlations (Spearman ρ)</p>",
                        unsafe_allow_html=True)
            outcome_cols = [c for c in ["acr_binary","clad_binary","death_binary"]
                            if c in pca_m.columns]
            use_pcs = pc_cols[:min(8, len(pc_cols))]
            if outcome_cols and use_pcs:
                from scipy.stats import spearmanr
                rho_mat = np.zeros((len(use_pcs), len(outcome_cols)))
                for pi, pc in enumerate(use_pcs):
                    for oi, oc in enumerate(outcome_cols):
                        valid = pca_m[[pc, oc]].dropna()
                        if len(valid) > 5:
                            rho_mat[pi, oi], _ = spearmanr(valid[pc], valid[oc])
                fig_corr = go.Figure(go.Heatmap(
                    z=rho_mat, x=[c.replace("_binary","") for c in outcome_cols],
                    y=use_pcs, colorscale="RdBu_r", zmin=-0.5, zmax=0.5,
                    colorbar=dict(title="ρ"),
                    text=np.round(rho_mat, 2),
                    texttemplate="%{text}",
                    hovertemplate="%{y} × %{x}: ρ=%{z:.3f}<extra></extra>",
                ))
                fig_corr.update_layout(**PLOTLY_THEME, height=320)
                st.plotly_chart(fig_corr, use_container_width=True)

        with c2:
            st.markdown(f"<p class='section-title'>Patient Trajectory in PC Space — {pid}</p>",
                        unsafe_allow_html=True)
            if "patient_id" in pca_m.columns:
                pt_pca = pca_m[pca_m["patient_id"]==pid].copy()
                if "anchor_dt" in pca_m.columns:
                    pt_pca = pt_pca.sort_values("anchor_dt")
                if len(pt_pca) >= 2:
                    acr_col = "acr_binary" if "acr_binary" in pt_pca.columns else None
                    colors_pt = (["#F85149" if v==1 else "#58A6FF"
                                  for v in pt_pca[acr_col]]
                                 if acr_col else [ACCENT]*len(pt_pca))
                    fig_traj = go.Figure()
                    fig_traj.add_trace(go.Scatter(
                        x=pt_pca[x_pc], y=pt_pca[y_pc], mode="lines+markers+text",
                        line=dict(color=MUTED, width=1.5, dash="dot"),
                        marker=dict(color=colors_pt, size=10, line=dict(color="white",width=1)),
                        text=[str(d)[:10] if pd.notna(d) else ""
                              for d in pt_pca.get("anchor_dt", [""]*len(pt_pca))],
                        textposition="top center", textfont=dict(size=8),
                        hovertemplate=f"{x_pc}: %{{x:.2f}}<br>{y_pc}: %{{y:.2f}}<extra></extra>",
                    ))
                    # mark start
                    fig_traj.add_trace(go.Scatter(
                        x=[pt_pca[x_pc].iloc[0]], y=[pt_pca[y_pc].iloc[0]],
                        mode="markers", name="Start",
                        marker=dict(symbol="circle-open", size=14, color="#FFD700",
                                    line=dict(width=2, color="#FFD700")),
                    ))
                    fig_traj.update_layout(**PLOTLY_THEME, height=320,
                                          xaxis_title=x_pc, yaxis_title=y_pc,
                                          showlegend=False)
                    st.plotly_chart(fig_traj, use_container_width=True)
                else:
                    st.info(f"{pid} has <2 PCA timepoints.")
            else:
                st.info("patient_id not in PCA scores.")

# ── Cross-modal correlation ───────────────────────────────────────────────
if not corr_df.empty:
    st.divider()
    st.markdown(f"<p class='section-title'>Cross-modal Correlation (PC1 per modality)</p>",
                unsafe_allow_html=True)
    fig_xm = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns.tolist(),
        y=corr_df.index.tolist(),
        colorscale="RdBu_r", zmin=-1, zmax=1,
        colorbar=dict(title="Spearman ρ"),
        text=np.round(corr_df.values, 2),
        texttemplate="%{text}",
        hovertemplate="%{y} × %{x}: ρ=%{z:.3f}<extra></extra>",
    ))
    fig_xm.update_layout(**PLOTLY_THEME, height=350)
    st.plotly_chart(fig_xm, use_container_width=True)
else:
    st.info("Cross-modal correlation not yet computed. Run `nature_analysis.sh`.")
