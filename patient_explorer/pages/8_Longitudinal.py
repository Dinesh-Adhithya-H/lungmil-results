"""Page 8 — Longitudinal Trajectories: survival curves, time-series risk, serial biopsies."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Longitudinal", page_icon="📉", layout="wide")

from utils.styles import card_css, BG, BG2, TEXT, MUTED, ACCENT, BORDER, CARD, PLOTLY_THEME, MOD_COLORS
from utils.data_loader import (
    load_splits, load_predictions, patient_splits, patient_predictions,
    patient_list, load_cluster_freq,
)

st.markdown(card_css(), unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"<h3 style='color:{TEXT}'>📉 Longitudinal</h3>", unsafe_allow_html=True)
    patients = patient_list()
    default_idx = patients.index(st.session_state.get("selected_patient", patients[0])) \
        if st.session_state.get("selected_patient") in patients else 0
    pid = st.selectbox("Patient", patients, index=default_idx)
    st.session_state["selected_patient"] = pid
    st.divider()
    show_km    = st.checkbox("Kaplan-Meier", value=True)
    show_risk  = st.checkbox("Risk trajectory", value=True)
    show_clust = st.checkbox("Cluster trajectory", value=True)

st.markdown(f"<h2 style='color:{TEXT}'>📉 Longitudinal Trajectories</h2>", unsafe_allow_html=True)

all_splits = load_splits()
all_splits["days_since_tx"] = (
    all_splits["anchor_dt"] - all_splits.groupby("patient_id")["anchor_dt"].transform("min")
).dt.days

# ── Kaplan-Meier ──────────────────────────────────────────────────────────
if show_km and "clad_status" in all_splits.columns:
    st.markdown(f"<p class='section-title'>Kaplan-Meier Survival Curves (Cohort)</p>", unsafe_allow_html=True)

    pt_summary = (all_splits.sort_values("anchor_dt")
                  .groupby("patient_id")[["clad_status","clad_days","death_status","death_days"]]
                  .first().reset_index())

    fig_km = go.Figure()
    for stat_col, days_col, color, name in [
        ("clad_status", "clad_days", "#F78166", "CLAD-free"),
        ("death_status","death_days","#D2A8FF", "Overall survival"),
    ]:
        sub = pt_summary[[stat_col, days_col]].dropna()
        if len(sub) < 3:
            continue
        t = sub[days_col].values.astype(float)
        e = sub[stat_col].values.astype(float)
        order = np.argsort(t); t, e = t[order], e[order]
        n_risk = len(t); surv = 1.0
        xs, ys = [0], [100.0]
        for ti, ei in zip(t, e):
            if ei == 1:
                surv *= (1 - 1/n_risk)
                xs.append(ti); ys.append(surv*100)
            n_risk -= 1
        fig_km.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name,
                                    line=dict(color=color, width=2, shape="hv"),
                                    hovertemplate="Day %{x}: %{y:.1f}%<extra>"+name+"</extra>"))

    fig_km.update_layout(**PLOTLY_THEME, height=340,
                         xaxis_title="Days post-transplant",
                         yaxis_title="Event-free (%)", yaxis=dict(range=[0,105]),
                         legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1))
    st.plotly_chart(fig_km, use_container_width=True)

# ── Risk score trajectory ─────────────────────────────────────────────────
if show_risk:
    preds  = patient_predictions(pid)
    splits = patient_splits(pid)
    st.divider()
    st.markdown(f"<p class='section-title'>Risk Score Trajectory — {pid}</p>", unsafe_allow_html=True)
    if preds.empty:
        st.info("No model predictions available for this patient.")
    else:
        prob_cols = [c for c in preds.columns if any(k in c for k in ["prob","score","pred","risk"])]
        date_col  = "anchor_dt" if "anchor_dt" in preds.columns else None
        if prob_cols and date_col:
            fig_r = go.Figure()
            colors_t = [ACCENT, "#3FB950", "#F78166", "#D2A8FF"]
            for col, cc in zip(prob_cols[:4], colors_t):
                v = preds[[date_col, col]].dropna()
                if v.empty: continue
                fig_r.add_trace(go.Scatter(x=v[date_col], y=v[col], mode="lines+markers",
                                           name=col.replace("_prob","").replace("_score",""),
                                           line=dict(color=cc, width=2), marker=dict(size=6),
                                           hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>"))
            if "acr_encoded" in splits.columns:
                acr_pos = splits[splits["acr_encoded"] == 1]
                if not acr_pos.empty:
                    fig_r.add_trace(go.Scatter(x=acr_pos["anchor_dt"],
                                               y=[0.95]*len(acr_pos), mode="markers",
                                               name="ACR event",
                                               marker=dict(color="#F85149", size=10, symbol="x"),
                                               hovertemplate="%{x|%Y-%m-%d}: ACR<extra></extra>"))
            fig_r.update_layout(**PLOTLY_THEME, height=300,
                                yaxis_title="Risk score", yaxis=dict(range=[0,1]),
                                legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1))
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("Prediction columns not in expected format.")

# ── Cluster trajectory ────────────────────────────────────────────────────
if show_clust:
    st.divider()
    st.markdown(f"<p class='section-title'>Cluster Composition Over Time — {pid}</p>", unsafe_allow_html=True)
    tabs = st.tabs(["H&E", "CT", "BAL"])
    for tab, mod in zip(tabs, ["HE","CT","BAL"]):
        with tab:
            df_f = load_cluster_freq(mod)
            if df_f.empty:
                st.info(f"No {mod} data."); continue
            ccols = sorted([c for c in df_f.columns if c.startswith("cluster_")],
                           key=lambda x: int(x.split("_")[1]))
            if not ccols: continue
            pt_f = df_f[df_f["patient_id"]==pid].copy() if "patient_id" in df_f.columns else pd.DataFrame()
            if pt_f.empty or len(pt_f) < 2:
                st.info(f"Need ≥2 timepoints for {pid} in {mod}."); continue
            if "anchor_dt" in pt_f.columns:
                pt_f = pt_f.sort_values("anchor_dt")
                xs = pt_f["anchor_dt"].dt.strftime("%Y-%m-%d").tolist()
            else:
                xs = [f"T{i+1}" for i in range(len(pt_f))]
            fig_t = go.Figure()
            pal = px.colors.qualitative.Plotly
            for ci, col in enumerate(ccols):
                fig_t.add_trace(go.Bar(name=f"C{ci}", x=xs, y=pt_f[col].values,
                                       marker_color=pal[ci%len(pal)],
                                       hovertemplate=f"C{ci}: %{{y:.2f}}<extra></extra>"))
            fig_t.update_layout(**PLOTLY_THEME, barmode="stack", height=280,
                                yaxis_title="Proportion",
                                legend=dict(bgcolor=CARD, bordercolor=BORDER,
                                            orientation="h", y=-0.4))
            st.plotly_chart(fig_t, use_container_width=True)

# ── Cohort risk heatmap ────────────────────────────────────────────────────
st.divider()
st.markdown(f"<p class='section-title'>Cohort ACR Risk Heatmap (Time Post-Transplant)</p>",
            unsafe_allow_html=True)
all_preds = load_predictions()
if not all_preds.empty:
    m = all_splits[["patient_id","anchor_dt","days_since_tx","acr_encoded"]].copy()
    ap2 = all_preds.merge(m, on=["patient_id","anchor_dt"], how="left")
    prob_col = next((c for c in ap2.columns if "prob" in c or "score" in c), None)
    if prob_col and "days_since_tx" in ap2.columns:
        bins   = [0,90,180,365,730,1095,9999]
        labels = ["0-90d","90-180d","6m-1y","1-2y","2-3y",">3y"]
        ap2["tbin"] = pd.cut(ap2["days_since_tx"], bins=bins, labels=labels)
        agg = ap2.groupby(["patient_id","tbin"])[prob_col].mean().unstack("tbin").reindex(columns=labels)
        agg = agg.dropna(how="all")
        if len(agg) > 0:
            highlighted = pid if pid in agg.index else None
            agg_sorted = agg.copy()
            # sort by mean risk, put selected patient first
            if highlighted:
                pt_row = agg_sorted.loc[[highlighted]]
                rest   = agg_sorted.drop(highlighted).sort_values(labels[0], na_position="last")
                agg_sorted = pd.concat([pt_row, rest])
            else:
                agg_sorted = agg_sorted.sort_values(labels[0], na_position="last")

            fig_h = go.Figure(go.Heatmap(
                z=agg_sorted.values, x=labels, y=agg_sorted.index.tolist(),
                colorscale="RdYlBu_r", zmin=0, zmax=1,
                colorbar=dict(title="Risk"),
                hovertemplate="Patient %{y}<br>%{x}: %{z:.3f}<extra></extra>",
            ))
            if highlighted and highlighted in agg_sorted.index:
                idx = agg_sorted.index.tolist().index(highlighted)
                fig_h.add_hline(y=idx, line_color="#FFD700", line_width=2, annotation_text=highlighted)
            fig_h.update_layout(**PLOTLY_THEME,
                                height=max(400, min(900, len(agg)*9)),
                                xaxis_title="Post-transplant period",
                                yaxis=dict(showticklabels=len(agg_sorted)<80))
            st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("No probability column in predictions.")
else:
    st.info("No predictions available. Run export_data.py.")
