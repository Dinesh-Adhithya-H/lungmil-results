"""Page 8 — Longitudinal Trajectories: best model per task, modality timeline, summary panels."""

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
    setmilmt_summary_png, longitudinal_summary_png, longitudinal_seed_timeline_png,
    load_longi_preds, patient_longi_preds,
)

st.markdown(card_css(), unsafe_allow_html=True)

# Best model per task (benchmark-confirmed)
BEST_MODEL = {
    "score_acr_cls":  "SetMIL-MT (no SAB)",   # BACC 0.623
    "pct_acr_surv":   "LongMK",               # C-index 0.679
    "pct_clad_surv":  "SetMIL-MT",            # C-index 0.563
    "pct_death_surv": "LongMK",               # C-index 0.771
}
TASK_METRIC = {
    "score_acr_cls":  "BACC 0.623",
    "pct_acr_surv":   "C-index 0.679",
    "pct_clad_surv":  "C-index 0.563",
    "pct_death_surv": "C-index 0.771",
}
TASK_LABELS = {
    "score_acr_cls":  "ACR Classification",
    "pct_acr_surv":   "ACR Survival",
    "pct_clad_surv":  "CLAD Survival",
    "pct_death_surv": "Death Survival",
}
TASK_COLORS = {
    "score_acr_cls":  "#7c83ff",
    "pct_acr_surv":   "#58a6ff",
    "pct_clad_surv":  "#f78166",
    "pct_death_surv": "#3fb950",
}
# Source: which data frame to pull each column from
TASK_SOURCE = {
    "score_acr_cls":  "setmil",  # SetMIL-MT (no SAB)
    "pct_acr_surv":   "longi",   # LongMK
    "pct_clad_surv":  "setmil",  # SetMIL-MT
    "pct_death_surv": "longi",   # LongMK
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
    show_km     = st.checkbox("Cohort KM curves", value=True)
    show_traj   = st.checkbox("Best-model trajectory", value=True)
    show_all    = st.checkbox("All-model comparison", value=False)
    show_mods   = st.checkbox("Modality availability", value=True)
    show_setmil = st.checkbox("SetMIL-MT summary panel", value=True)
    show_longi  = st.checkbox("LongMK summary panel", value=True)

st.markdown(f"<h2 style='color:{TEXT}'>📉 Patient Trajectories — {pid}</h2>", unsafe_allow_html=True)

# ── Best-model summary banner ────────────────────────────────────────────────
banner_html = "<div style='display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;'>"
for task, model in BEST_MODEL.items():
    color = TASK_COLORS[task]
    metric = TASK_METRIC[task]
    label = TASK_LABELS[task]
    banner_html += f"""
    <div style='background:{CARD};border:1px solid {BORDER};border-left:4px solid {color};
                border-radius:8px;padding:10px 16px;min-width:200px;flex:1'>
      <div style='color:{MUTED};font-size:0.72rem;text-transform:uppercase;letter-spacing:.05em'>{label}</div>
      <div style='color:{TEXT};font-size:1rem;font-weight:700;margin-top:4px'>{model}</div>
      <div style='color:{MUTED};font-size:0.78rem;margin-top:2px'>{metric}</div>
    </div>"""
banner_html += "</div>"
st.markdown(banner_html, unsafe_allow_html=True)

df_setmil = load_setmilmt()
df_pt_sm  = patient_setmilmt(pid)
df_longi  = patient_longi_preds(pid)

# ── Cohort Kaplan-Meier ───────────────────────────────────────────────────────
if show_km and not df_setmil.empty:
    st.markdown(f"<p class='section-title'>Cohort Kaplan-Meier (all patients, n≈350)</p>", unsafe_allow_html=True)
    pt_first = df_setmil.sort_values("anchor_dt").groupby("patient_id").first().reset_index()
    fig_km = go.Figure()
    for ev_col, tte_col, color, name in [
        ("event_clad",  "tte_clad",  "#f78166", "CLAD-free survival"),
        ("event_death", "tte_death", "#3fb950", "Overall survival"),
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
        **PLOTLY_THEME, height=300,
        xaxis_title="Days post-transplant",
        yaxis_title="Event-free (%)",
        yaxis=dict(range=[0, 105]),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
    )
    st.plotly_chart(fig_km, use_container_width=True)
    st.divider()

if df_pt_sm.empty:
    st.warning(f"No SetMIL-MT data for {pid}.")
    st.stop()

t0 = df_pt_sm["anchor_dt"].min()
df_pt_sm = df_pt_sm.copy()
df_pt_sm["days"] = (df_pt_sm["anchor_dt"] - t0).dt.days

if not df_longi.empty:
    df_longi = df_longi.copy()
    df_longi["days"] = (df_longi["anchor_dt"] - df_longi["anchor_dt"].min()).dt.days

def _event_day(df, ev_col, tte_col):
    rows = df[df[ev_col] == 1]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return row["days"] + float(row[tte_col]) if pd.notna(row.get(tte_col)) else None

clad_day  = _event_day(df_pt_sm, "event_clad",  "tte_clad")
death_day = _event_day(df_pt_sm, "event_death", "tte_death")
n_visits  = len(df_pt_sm)

# Metric row
c1, c2, c3, c4 = st.columns(4)
c1.markdown(metric_card("Biopsies", str(n_visits)), unsafe_allow_html=True)
c2.markdown(metric_card("Follow-up", f"{df_pt_sm['days'].max()}d"), unsafe_allow_html=True)
c3.markdown(metric_card("CLAD", "event" if clad_day else "censored"), unsafe_allow_html=True)
c4.markdown(metric_card("Death", "event" if death_day else "censored"), unsafe_allow_html=True)


def _add_event_lines(fig):
    if clad_day is not None:
        fig.add_vline(x=clad_day, line_color="#f78166", line_width=2,
                      annotation_text="CLAD event", annotation_font_color="#f78166",
                      annotation_position="top left")
    if death_day is not None:
        fig.add_vline(x=death_day, line_color="#3fb950", line_width=2,
                      annotation_text="Death", annotation_font_color="#3fb950",
                      annotation_position="top right")
    # ACR biopsy events
    for _, row in df_pt_sm.iterrows():
        ev = row.get("event_acr")
        if pd.notna(ev) and ev == 1:
            fig.add_vline(x=row["days"], line_color="#E53935",
                          line_width=1, line_dash="dot", opacity=0.45)
    fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED, line_width=1,
                  annotation_text="0.5", annotation_font_color=MUTED)


# ── Best-model trajectory ────────────────────────────────────────────────────
if show_traj:
    st.markdown(f"<p class='section-title'>Best-Model Risk Trajectory — {pid}</p>", unsafe_allow_html=True)
    fig_best = go.Figure()

    for task, source in TASK_SOURCE.items():
        color = TASK_COLORS[task]
        label = TASK_LABELS[task]
        model = BEST_MODEL[task]
        if source == "setmil":
            df_src = df_pt_sm
        else:
            df_src = df_longi
        if df_src.empty or task not in df_src.columns:
            continue
        vals = df_src[task].values.astype(float)
        if np.all(np.isnan(vals)):
            continue
        sym = "circle" if source == "longi" else "square"
        fig_best.add_trace(go.Scatter(
            x=df_src["days"], y=vals,
            mode="lines+markers",
            name=f"{label} ({model})",
            line=dict(color=color, width=2.5,
                      dash="dash" if source == "longi" else "solid"),
            marker=dict(size=7, symbol=sym, color=color),
            hovertemplate=(
                f"<b>{label}</b> [{model}]<br>"
                "Day %{x}: %{y:.3f}<extra></extra>"
            ),
        ))

    _add_event_lines(fig_best)
    fig_best.update_layout(
        **PLOTLY_THEME, height=420,
        xaxis_title="Days from transplant",
        yaxis_title="Risk score (0 = low risk, 1 = high risk)",
        yaxis=dict(range=[-0.05, 1.05]),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_best, use_container_width=True)
    st.caption(
        "Solid line = SetMIL-MT (ACR cls, CLAD surv); Dashed = LongMK (ACR surv, Death surv). "
        "Vertical red dotted = ACR+ biopsy. ● = set-based model point; ◆ = longitudinal model point."
    )
    st.divider()

# ── All-model comparison (collapsible) ───────────────────────────────────────
if show_all:
    st.markdown(f"<p class='section-title'>All-Model Comparison (SetMIL-MT vs LongMK)</p>", unsafe_allow_html=True)
    col_sm, col_lg = st.columns(2)

    with col_sm:
        st.markdown("**SetMIL-MT**", unsafe_allow_html=False)
        fig_sm = go.Figure()
        for task, color in TASK_COLORS.items():
            if task not in df_pt_sm.columns:
                continue
            vals = df_pt_sm[task].values.astype(float)
            if np.all(np.isnan(vals)):
                continue
            fig_sm.add_trace(go.Scatter(
                x=df_pt_sm["days"], y=vals, mode="lines+markers",
                name=TASK_LABELS[task],
                line=dict(color=color, width=2),
                marker=dict(size=5, color=color),
                hovertemplate=f"<b>{TASK_LABELS[task]}</b><br>Day %{{x}}: %{{y:.3f}}<extra></extra>",
            ))
        _add_event_lines(fig_sm)
        fig_sm.update_layout(**PLOTLY_THEME, height=360,
                             xaxis_title="Days", yaxis_title="Score",
                             yaxis=dict(range=[-0.05, 1.05]),
                             legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
                             hovermode="x unified")
        st.plotly_chart(fig_sm, use_container_width=True)

    with col_lg:
        st.markdown("**LongMK**", unsafe_allow_html=False)
        fig_lk = go.Figure()
        for task, color in TASK_COLORS.items():
            if df_longi.empty or task not in df_longi.columns:
                continue
            vals = df_longi[task].values.astype(float)
            if np.all(np.isnan(vals)):
                continue
            fig_lk.add_trace(go.Scatter(
                x=df_longi["days"], y=vals, mode="lines+markers",
                name=TASK_LABELS[task],
                line=dict(color=color, width=2, dash="dash"),
                marker=dict(size=5, symbol="diamond", color=color),
                hovertemplate=f"<b>{TASK_LABELS[task]} (LongMK)</b><br>Day %{{x}}: %{{y:.3f}}<extra></extra>",
            ))
        _add_event_lines(fig_lk)
        fig_lk.update_layout(**PLOTLY_THEME, height=360,
                              xaxis_title="Days", yaxis_title="Score",
                              yaxis=dict(range=[-0.05, 1.05]),
                              legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
                              hovermode="x unified")
        st.plotly_chart(fig_lk, use_container_width=True)
    st.divider()

# ── Modality availability ─────────────────────────────────────────────────────
if show_mods:
    st.markdown(f"<p class='section-title'>Modality Availability — {pid}</p>", unsafe_allow_html=True)
    fig_mod = go.Figure()
    for mi, mod in enumerate(MOD_ORDER):
        present_days, absent_days = [], []
        for _, row in df_pt_sm.iterrows():
            mods = set(str(row["present_mods"]).split(",")) if pd.notna(row.get("present_mods")) else set()
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
        fig_mod.add_vline(x=clad_day, line_color="#f78166", line_width=2)
    if death_day:
        fig_mod.add_vline(x=death_day, line_color="#3fb950", line_width=2)
    fig_mod.update_layout(
        **PLOTLY_THEME, height=220,
        xaxis_title="Days from transplant",
        yaxis=dict(tickvals=list(range(len(MOD_ORDER))), ticktext=MOD_ORDER, showgrid=False),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
    )
    st.plotly_chart(fig_mod, use_container_width=True)
    st.divider()

# ── Patient summary PNGs ──────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

if show_setmil:
    with col_a:
        st.markdown(f"<p class='section-title'>SetMIL-MT Summary Panel</p>", unsafe_allow_html=True)
        found = False
        for _, row in df_pt_sm.sort_values("days", ascending=False).iterrows():
            s = str(row["stem"]).zfill(5) if str(row["stem"]).isdigit() else str(row["stem"])
            p = setmilmt_summary_png(s)
            if p and p.exists():
                st.image(Image.open(p), caption=f"SetMIL-MT — {pid} (stem {s})", use_container_width=True)
                found = True
                break
        if not found:
            st.info("SetMIL-MT summary PNG not found for this patient.")

if show_longi:
    with col_b:
        st.markdown(f"<p class='section-title'>LongMK Summary Panel</p>", unsafe_allow_html=True)
        p0 = longitudinal_summary_png(pid)
        p1 = longitudinal_seed_timeline_png(pid)
        if p0 and p0.exists():
            st.image(Image.open(p0), caption=f"LongMK — L0 Summary — {pid}", use_container_width=True)
        else:
            st.info("LongMK summary (L0) not found for this patient.")
        if p1 and p1.exists():
            st.image(Image.open(p1), caption=f"LongMK — L1 Seed Timeline — {pid}", use_container_width=True)
