"""Page 8 — Longitudinal Trajectories: SetMIL-MT risk scores, KM curves, patient summaries."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Longitudinal", page_icon="📉", layout="wide")

from utils.styles import card_css, metric_card, BG, BG2, TEXT, MUTED, ACCENT, BORDER, CARD, PLOTLY_THEME, MOD_COLORS
from utils.data_loader import (
    load_setmilmt, patient_list, patient_setmilmt,
    setmilmt_summary_png, longitudinal_summary_png,
)

st.markdown(card_css(), unsafe_allow_html=True)

TASK_COLORS = {
    "score_acr_cls":  "#7c83ff",
    "pct_acr_surv":   "#58a6ff",
    "pct_clad_surv":  "#8E24AA",
    "pct_death_surv": "#00897B",
}
TASK_LABELS = {
    "score_acr_cls":  "ACR classif.",
    "pct_acr_surv":   "ACR survival",
    "pct_clad_surv":  "CLAD survival",
    "pct_death_surv": "Death survival",
}
MOD_ORDER = ["HE", "BAL", "CT", "Clinical"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:{TEXT}'>📉 Longitudinal</h3>", unsafe_allow_html=True)
    patients = patient_list()
    default_idx = patients.index(st.session_state.get("selected_patient", patients[0])) \
        if st.session_state.get("selected_patient") in patients else 0
    pid = st.selectbox("Patient", patients, index=default_idx)
    st.session_state["selected_patient"] = pid
    st.divider()
    show_km      = st.checkbox("Cohort KM curves", value=True)
    show_traj    = st.checkbox("Risk trajectory", value=True)
    show_mods    = st.checkbox("Modality availability", value=True)
    show_setmil  = st.checkbox("SetMIL-MT summary panel", value=True)
    show_longi   = st.checkbox("Longitudinal model panel", value=True)
    st.divider()
    show_tasks = st.multiselect(
        "Tasks to plot",
        list(TASK_LABELS.keys()),
        default=list(TASK_LABELS.keys()),
        format_func=lambda x: TASK_LABELS[x],
    )

st.markdown(f"<h2 style='color:{TEXT}'>📉 Longitudinal — {pid}</h2>", unsafe_allow_html=True)

df_all = load_setmilmt()
df_pt  = patient_setmilmt(pid)

# ── Cohort Kaplan-Meier from SetMIL-MT data ────────────────────────────────────
if show_km and not df_all.empty:
    st.markdown(f"<p class='section-title'>Cohort Kaplan-Meier (SetMIL-MT data)</p>", unsafe_allow_html=True)

    # Use first visit per patient for event/TTE
    pt_first = df_all.sort_values("anchor_dt").groupby("patient_id").first().reset_index()

    fig_km = go.Figure()
    for ev_col, tte_col, color, name in [
        ("event_clad",  "tte_clad",  "#8E24AA", "CLAD-free survival"),
        ("event_death", "tte_death", "#00897B", "Overall survival"),
    ]:
        sub = pt_first[[ev_col, tte_col]].dropna()
        if len(sub) < 5:
            continue
        t = sub[tte_col].values.astype(float)
        e = sub[ev_col].values.astype(float)
        order = np.argsort(t)
        t, e = t[order], e[order]
        n_risk = len(t)
        surv = 1.0
        xs, ys = [0.0], [100.0]
        for ti, ei in zip(t, e):
            if ei == 1:
                surv *= (1 - 1.0 / n_risk)
                xs.append(float(ti))
                ys.append(surv * 100)
            n_risk -= 1
        fig_km.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=name,
            line=dict(color=color, width=2.5, shape="hv"),
            hovertemplate=f"Day %{{x:.0f}}: %{{y:.1f}}%<extra>{name}</extra>",
        ))

    fig_km.update_layout(
        **PLOTLY_THEME, height=340,
        xaxis_title="Days post-transplant",
        yaxis_title="Event-free (%)",
        yaxis=dict(range=[0, 105]),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
    )
    st.plotly_chart(fig_km)
    st.divider()

if df_pt.empty:
    st.warning(f"No SetMIL-MT data for {pid}.")
    st.stop()

# Compute event absolute days
t0 = df_pt["anchor_dt"].min()
df_pt = df_pt.copy()
df_pt["days"] = (df_pt["anchor_dt"] - t0).dt.days

def _event_day(df, ev_col, tte_col):
    rows = df[df[ev_col] == 1]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return row["days"] + float(row[tte_col]) if pd.notna(row.get(tte_col)) else None

clad_day  = _event_day(df_pt, "event_clad",  "tte_clad")
death_day = _event_day(df_pt, "event_death", "tte_death")
n_visits  = len(df_pt)
is_flagged = df_pt["flagged"].any()

# Metric row
c1, c2, c3, c4 = st.columns(4)
c1.markdown(metric_card("Visits", str(n_visits)), unsafe_allow_html=True)
c2.markdown(metric_card("Follow-up", f"{df_pt['days'].max()}d"), unsafe_allow_html=True)
c3.markdown(metric_card("CLAD", "event" if clad_day else "censored"), unsafe_allow_html=True)
c4.markdown(metric_card("Death", "event" if death_day else "censored"), unsafe_allow_html=True)

# ── Risk trajectory ───────────────────────────────────────────────────────────
if show_traj:
    st.divider()
    st.markdown(f"<p class='section-title'>SetMIL-MT Risk Trajectory — {pid}</p>", unsafe_allow_html=True)

    fig = go.Figure()
    for task in show_tasks:
        vals = df_pt[task].values.astype(float)
        if np.all(np.isnan(vals)):
            continue
        fig.add_trace(go.Scatter(
            x=df_pt["days"], y=vals,
            mode="lines+markers",
            name=TASK_LABELS[task],
            line=dict(color=TASK_COLORS[task], width=2.2),
            marker=dict(size=6, color=TASK_COLORS[task]),
            hovertemplate=(
                f"<b>{TASK_LABELS[task]}</b><br>"
                "Day %{x}: %{y:.3f}<br>%{customdata}<extra></extra>"
            ),
            customdata=df_pt["anchor_dt"].dt.strftime("%Y-%m-%d"),
        ))

    if clad_day is not None:
        fig.add_vline(x=clad_day, line_color="#8E24AA", line_width=2,
                      annotation_text="CLAD", annotation_font_color="#8E24AA",
                      annotation_position="top left")
    if death_day is not None:
        fig.add_vline(x=death_day, line_color="#00897B", line_width=2,
                      annotation_text="Death", annotation_font_color="#00897B",
                      annotation_position="top right")
    for _, row in df_pt.iterrows():
        ev = row.get("event_acr")
        if pd.notna(ev):
            fig.add_vline(x=row["days"], line_color="#E53935" if ev == 1 else "#43A047",
                          line_width=1, line_dash="dot", opacity=0.4)

    fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED, line_width=1)
    fig.update_layout(
        **PLOTLY_THEME, height=380,
        xaxis_title="Days from transplant",
        yaxis_title="Score (0=low risk, 1=high risk)",
        yaxis=dict(range=[-0.05, 1.05]),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig)

# ── Modality availability ─────────────────────────────────────────────────────
if show_mods:
    st.divider()
    st.markdown(f"<p class='section-title'>Modality Availability — {pid}</p>", unsafe_allow_html=True)

    fig_mod = go.Figure()
    for mi, mod in enumerate(MOD_ORDER):
        present_days, absent_days = [], []
        for _, row in df_pt.iterrows():
            mods = set(str(row["present_mods"]).split(",")) if pd.notna(row["present_mods"]) else set()
            (present_days if mod in mods else absent_days).append(row["days"])
        if present_days:
            fig_mod.add_trace(go.Scatter(
                x=present_days, y=[mi]*len(present_days), mode="markers", name=mod,
                marker=dict(symbol="square", size=10, color=MOD_COLORS[mod]),
                hovertemplate=f"{mod}<br>Day %{{x}}<extra></extra>",
            ))
        if absent_days:
            fig_mod.add_trace(go.Scatter(
                x=absent_days, y=[mi]*len(absent_days), mode="markers",
                name=f"{mod} absent", showlegend=False,
                marker=dict(symbol="x", size=7, color=MUTED),
                hovertemplate=f"{mod} absent<br>Day %{{x}}<extra></extra>",
            ))
    if clad_day:
        fig_mod.add_vline(x=clad_day, line_color="#8E24AA", line_width=2)
    if death_day:
        fig_mod.add_vline(x=death_day, line_color="#00897B", line_width=2)
    fig_mod.update_layout(
        **PLOTLY_THEME, height=220,
        xaxis_title="Days from transplant",
        yaxis=dict(tickvals=list(range(len(MOD_ORDER))), ticktext=MOD_ORDER, showgrid=False),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
    )
    st.plotly_chart(fig_mod)

# ── Patient summary PNGs ──────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

if show_setmil:
    with col_a:
        st.divider()
        st.markdown(f"<p class='section-title'>SetMIL-MT Summary Panel</p>", unsafe_allow_html=True)
        found = False
        for _, row in df_pt.sort_values("days", ascending=False).iterrows():
            s = str(row["stem"]).zfill(5) if str(row["stem"]).isdigit() else str(row["stem"])
            p = setmilmt_summary_png(s)
            if p and p.exists():
                st.image(Image.open(p),
                         caption=f"SetMIL-MT — {pid} (stem {s})")
                found = True
                break
        if not found:
            st.info("SetMIL-MT summary PNG not found for this patient.")

if show_longi:
    with col_b:
        st.divider()
        st.markdown(f"<p class='section-title'>Longitudinal Model Summary Panel</p>", unsafe_allow_html=True)
        p = longitudinal_summary_png(pid)
        if p and p.exists():
            st.image(Image.open(p),
                     caption=f"Longitudinal SetMIL — {pid}")
        else:
            st.info("Longitudinal model summary not found for this patient.")
