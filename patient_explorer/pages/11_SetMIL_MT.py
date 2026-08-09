"""Page 11 — SetMIL-MT: best model for ACR classification & CLAD survival."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="SetMIL-MT", page_icon="🧬", layout="wide")

from utils.styles import card_css, metric_card, BG, BG2, TEXT, MUTED, ACCENT, BORDER, CARD, PLOTLY_THEME
from utils.data_loader import (
    load_setmilmt, patient_list, patient_setmilmt, setmilmt_summary_png,
)

st.markdown(card_css(), unsafe_allow_html=True)

TASK_COLORS = {
    "score_acr_cls":  "#7c83ff",
    "pct_acr_surv":   "#58a6ff",
    "pct_clad_surv":  "#f78166",
    "pct_death_surv": "#3fb950",
}
TASK_LABELS = {
    "score_acr_cls":  "ACR classification P(ACR+)",
    "pct_acr_surv":   "ACR survival risk",
    "pct_clad_surv":  "CLAD survival risk",
    "pct_death_surv": "Death survival risk",
}
BEST_TASKS = {"score_acr_cls", "pct_clad_surv"}
MOD_COLORS = {"HE": "#a5d6ff", "BAL": "#ffa657", "CT": "#7ee787", "Clinical": "#f2cc60"}
MOD_ORDER = ["HE", "BAL", "CT", "Clinical"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:#9C27B0'>🧬 SetMIL-MT</h3>", unsafe_allow_html=True)
    df_all = load_setmilmt()
    if df_all.empty:
        st.error("setmilmt_preds.csv not found.")
        st.stop()

    st.divider()
    filter_clad  = st.checkbox("Has CLAD event", value=False)
    filter_death = st.checkbox("Has death event", value=False)

    pt_events = df_all.groupby("patient_id").agg(
        ev_clad=("event_clad", "max"),
        ev_death=("event_death", "max"),
    ).reset_index()
    mask = pd.Series([True] * len(pt_events), index=pt_events.index)
    if filter_clad:
        mask &= pt_events["ev_clad"] > 0
    if filter_death:
        mask &= pt_events["ev_death"] > 0
    patients = sorted(pt_events.loc[mask, "patient_id"].tolist())
    if not patients:
        st.warning("No patients match filters.")
        st.stop()
    st.caption(f"{len(patients)} patient(s)")

    default_pid = st.session_state.get("selected_patient", patients[0])
    default_idx = patients.index(default_pid) if default_pid in patients else 0
    pid = st.selectbox("Patient", patients, index=default_idx)
    st.session_state["selected_patient"] = pid

    st.divider()
    show_png = st.checkbox("Show summary panel", value=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"<h2 style='color:{TEXT}'>🧬 SetMIL-MT — {pid}</h2>", unsafe_allow_html=True)

# Best model banner
st.markdown(
    f"""<div style='background:{CARD};border:1px solid {BORDER};border-left:4px solid #9C27B0;
        border-radius:8px;padding:12px 18px;margin-bottom:16px;display:flex;gap:40px'>
      <div>
        <div style='color:{MUTED};font-size:0.72rem;text-transform:uppercase'>Best for ACR Classification</div>
        <div style='color:#9C27B0;font-size:1rem;font-weight:700'>SetMIL-MT (no SAB)</div>
        <div style='color:{TEXT};font-size:0.9rem'>BACC 0.623 ± 0.034</div>
      </div>
      <div>
        <div style='color:{MUTED};font-size:0.72rem;text-transform:uppercase'>Best for CLAD Survival</div>
        <div style='color:#9C27B0;font-size:1rem;font-weight:700'>SetMIL-MT</div>
        <div style='color:{TEXT};font-size:0.9rem'>C-index 0.563 ± 0.080</div>
      </div>
    </div>""",
    unsafe_allow_html=True,
)

# ── Load patient data ─────────────────────────────────────────────────────────
df = patient_setmilmt(pid)
if df.empty:
    st.error(f"No SetMIL-MT data for {pid}.")
    st.stop()

t0 = df["anchor_dt"].min()
df = df.copy()
df["days"] = (df["anchor_dt"] - t0).dt.days

def _event_day(df, ev_col, tte_col):
    rows = df[df[ev_col] == 1]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return row["days"] + float(row[tte_col]) if not pd.isna(row[tte_col]) else None

clad_day  = _event_day(df, "event_clad",  "tte_clad")
death_day = _event_day(df, "event_death", "tte_death")

# ── Metric row ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown(metric_card("Biopsies", str(len(df))), unsafe_allow_html=True)
c2.markdown(metric_card("Follow-up", f"{df['days'].max()}d"), unsafe_allow_html=True)
c3.markdown(metric_card("CLAD", "event" if clad_day else "censored"), unsafe_allow_html=True)
c4.markdown(metric_card("Death", "event" if death_day else "censored"), unsafe_allow_html=True)

st.divider()

# ── Risk trajectory ───────────────────────────────────────────────────────────
st.markdown(f"<p class='section-title'>Risk Score Trajectory — {pid}</p>", unsafe_allow_html=True)
st.caption("Bold lines = tasks where SetMIL-MT is the best model (ACR cls, CLAD surv)")

fig = go.Figure()
for task, color in TASK_COLORS.items():
    if task not in df.columns:
        continue
    vals = df[task].values
    if np.all(np.isnan(vals.astype(float))):
        continue
    is_best = task in BEST_TASKS
    fig.add_trace(go.Scatter(
        x=df["days"], y=vals,
        mode="lines+markers",
        name=f"{'★ ' if is_best else ''}{TASK_LABELS[task]}",
        line=dict(color=color, width=3 if is_best else 1.5,
                  dash="solid" if is_best else "dot"),
        marker=dict(size=7 if is_best else 4, color=color),
        opacity=1.0 if is_best else 0.55,
        hovertemplate=f"<b>{TASK_LABELS[task]}</b><br>Day %{{x}}: %{{y:.3f}}<extra></extra>",
    ))

for _, row in df.iterrows():
    ev = row.get("event_acr")
    if pd.notna(ev) and ev == 1:
        fig.add_vline(x=row["days"], line_color="#E53935", line_width=1.2,
                      line_dash="dash", opacity=0.5)

if clad_day is not None:
    fig.add_vline(x=clad_day, line_color="#f78166", line_width=2.5,
                  annotation_text="CLAD event", annotation_font_color="#f78166",
                  annotation_position="top left")
if death_day is not None:
    fig.add_vline(x=death_day, line_color="#3fb950", line_width=2.5,
                  annotation_text="Death", annotation_font_color="#3fb950",
                  annotation_position="top right")

fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED, line_width=1)
fig.update_layout(
    **PLOTLY_THEME, height=400,
    xaxis_title="Days from transplant",
    yaxis_title="Score (0 = low risk, 1 = high risk)",
    yaxis=dict(range=[-0.05, 1.05]),
    legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Cohort boxplots: score distribution by outcome ────────────────────────────
st.markdown(f"<p class='section-title'>Cohort: Score Distribution by Outcome</p>", unsafe_allow_html=True)

pt_summary = (
    df_all.sort_values("anchor_dt")
    .groupby("patient_id")
    .agg(
        score_acr_cls=("score_acr_cls", "mean"),
        pct_clad_surv=("pct_clad_surv", "mean"),
        event_clad=("event_clad", "max"),
        event_death=("event_death", "max"),
    )
    .reset_index()
)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(f"**ACR Classification — mean score by CLAD outcome**")
    fig_box1 = go.Figure()
    for ev, label, color in [(0, "CLAD-free", "#3fb950"), (1, "CLAD event", "#f78166")]:
        sub = pt_summary[pt_summary["event_clad"] == ev]["score_acr_cls"].dropna()
        fig_box1.add_trace(go.Box(
            y=sub.values, name=label,
            marker_color=color, boxmean="sd",
            hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
        ))
    fig_box1.add_hline(y=0.5, line_dash="dot", line_color=MUTED, line_width=1)
    fig_box1.update_layout(**PLOTLY_THEME, height=320,
                            yaxis_title="Mean score_acr_cls",
                            yaxis=dict(range=[0, 1.05]))
    st.plotly_chart(fig_box1, use_container_width=True)

with col_b:
    st.markdown(f"**CLAD Survival — mean risk by death outcome**")
    fig_box2 = go.Figure()
    for ev, label, color in [(0, "Alive", "#3fb950"), (1, "Died", "#C62828")]:
        sub = pt_summary[pt_summary["event_death"] == ev]["pct_clad_surv"].dropna()
        fig_box2.add_trace(go.Box(
            y=sub.values, name=label,
            marker_color=color, boxmean="sd",
            hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
        ))
    fig_box2.add_hline(y=0.5, line_dash="dot", line_color=MUTED, line_width=1)
    fig_box2.update_layout(**PLOTLY_THEME, height=320,
                            yaxis_title="Mean pct_clad_surv",
                            yaxis=dict(range=[0, 1.05]))
    st.plotly_chart(fig_box2, use_container_width=True)

# ── Modality availability ─────────────────────────────────────────────────────
st.divider()
st.markdown(f"<p class='section-title'>Modality Availability — {pid}</p>", unsafe_allow_html=True)

fig_mod = go.Figure()
for mi, mod in enumerate(MOD_ORDER):
    present_days, absent_days = [], []
    for _, row in df.iterrows():
        mods = set(row["present_mods"].split(",")) if isinstance(row.get("present_mods"), str) else set()
        (present_days if mod in mods else absent_days).append(row["days"])
    if present_days:
        fig_mod.add_trace(go.Scatter(
            x=present_days, y=[mi] * len(present_days), mode="markers", name=mod,
            marker=dict(symbol="square", size=10, color=MOD_COLORS[mod]),
            hovertemplate=f"{mod}<br>Day %{{x}}<extra></extra>",
        ))
    if absent_days:
        fig_mod.add_trace(go.Scatter(
            x=absent_days, y=[mi] * len(absent_days), mode="markers",
            name=f"{mod} (absent)", showlegend=False,
            marker=dict(symbol="x", size=7, color=MUTED),
        ))
if clad_day is not None:
    fig_mod.add_vline(x=clad_day, line_color="#f78166", line_width=2)
if death_day is not None:
    fig_mod.add_vline(x=death_day, line_color="#3fb950", line_width=2)
fig_mod.update_layout(
    **PLOTLY_THEME, height=220,
    xaxis_title="Days from transplant",
    yaxis=dict(tickvals=list(range(len(MOD_ORDER))), ticktext=MOD_ORDER, showgrid=False),
    legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
)
st.plotly_chart(fig_mod, use_container_width=True)

# ── Summary PNG ───────────────────────────────────────────────────────────────
if show_png:
    st.divider()
    st.markdown(f"<p class='section-title'>SetMIL-MT Summary Panel</p>", unsafe_allow_html=True)
    found = False
    for _, row in df.sort_values("days", ascending=False).iterrows():
        s = str(row["stem"]).zfill(5) if str(row["stem"]).isdigit() else str(row["stem"])
        p = setmilmt_summary_png(s)
        if p and p.exists():
            st.image(Image.open(p), caption=f"SetMIL-MT summary — {pid} (stem {s})",
                     use_container_width=True)
            found = True
            break
    if not found:
        st.info("Summary PNG not found for this patient.")

# ── Event context table ───────────────────────────────────────────────────────
st.divider()
st.markdown(f"<p class='section-title'>Event Context</p>", unsafe_allow_html=True)
last = df.iloc[-1]
tte_rows = []
for ev_col, tte_col, label in [
    ("event_acr", "tte_acr", "ACR"),
    ("event_clad", "tte_clad", "CLAD"),
    ("event_death", "tte_death", "Death"),
]:
    ev  = last.get(ev_col)
    tte = last.get(tte_col)
    status = "event" if ev == 1 else ("censored" if ev == 0 else "?")
    tte_str = f"{tte:.0f} days" if pd.notna(tte) else "?"
    tte_rows.append({"Outcome": label, "Status": status, "TTE": tte_str})
st.dataframe(pd.DataFrame(tte_rows), hide_index=True, use_container_width=True)
