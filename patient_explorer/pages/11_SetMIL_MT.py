"""Page 11 — SetMIL-MT Patient Trajectories."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="SetMIL-MT Trajectories", page_icon="🧬", layout="wide")

from utils.styles import card_css, metric_card, BG, BG2, TEXT, MUTED, ACCENT, BORDER, CARD, PLOTLY_THEME
from utils.data_loader import (
    load_setmilmt, patient_list, patient_setmilmt, setmilmt_summary_png,
)

st.markdown(card_css(), unsafe_allow_html=True)

# ── Colors ────────────────────────────────────────────────────────────────────
TASK_COLORS = {
    "score_acr_cls":  "#7c83ff",
    "pct_acr_surv":   "#58a6ff",
    "pct_clad_surv":  "#8E24AA",
    "pct_death_surv": "#00897B",
}
TASK_LABELS = {
    "score_acr_cls":  "ACR classif. P(ACR+)",
    "pct_acr_surv":   "ACR survival (pctile)",
    "pct_clad_surv":  "CLAD survival (pctile)",
    "pct_death_surv": "Death survival (pctile)",
}
MOD_COLORS = {
    "HE":       "#a5d6ff",
    "BAL":      "#ffa657",
    "CT":       "#7ee787",
    "Clinical": "#f2cc60",
}
MOD_ORDER = ["HE", "BAL", "CT", "Clinical"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:{TEXT}'>🧬 SetMIL-MT</h3>", unsafe_allow_html=True)

    df_all = load_setmilmt()
    if df_all.empty:
        st.error("setmilmt_preds.csv not found. Run export_setmilmt_preds.sh first.")
        st.stop()

    st.divider()
    show_flagged_only = st.checkbox("⚠ Flagged patients only", value=False)
    filter_clad  = st.checkbox("Has CLAD event", value=False)
    filter_death = st.checkbox("Has death event", value=False)

    # Build filtered patient list — use max() per patient so ANY visit with event=1 counts
    pt_events = df_all.groupby("patient_id").agg(
        ev_clad=("event_clad", "max"),
        ev_death=("event_death", "max"),
        ev_flag=("flagged", "max"),
    ).reset_index()
    mask = pd.Series([True] * len(pt_events), index=pt_events.index)
    if show_flagged_only:
        mask &= pt_events["ev_flag"].astype(bool)
    if filter_clad:
        mask &= pt_events["ev_clad"] > 0
    if filter_death:
        mask &= pt_events["ev_death"] > 0
    patients = sorted(pt_events.loc[mask, "patient_id"].tolist())
    if not patients:
        st.warning("No patients match filters.")
        st.stop()
    st.caption(f"{len(patients)} patient(s) shown")

    default_pid = st.session_state.get("selected_patient", patients[0])
    default_idx = patients.index(default_pid) if default_pid in patients else 0
    pid = st.selectbox("Patient", patients, index=default_idx, key="setmil_pid")
    st.session_state["selected_patient"] = pid

    st.divider()
    show_tasks = st.multiselect(
        "Show tasks",
        list(TASK_LABELS.keys()),
        default=list(TASK_LABELS.keys()),
        format_func=lambda x: TASK_LABELS[x],
    )
    show_png = st.checkbox("Show full summary panel", value=True)

# ── Load patient data ─────────────────────────────────────────────────────────
df = patient_setmilmt(pid)
if df.empty:
    st.error(f"No SetMIL-MT data for {pid}.")
    st.stop()

# Compute days from first visit
t0 = df["anchor_dt"].min()
df = df.copy()
df["days"] = (df["anchor_dt"] - t0).dt.days

# Event absolute days (first visit where event=1)
def _event_day(df, ev_col, tte_col):
    rows = df[df[ev_col] == 1]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return row["days"] + float(row[tte_col]) if not pd.isna(row[tte_col]) else None

clad_day  = _event_day(df, "event_clad",  "tte_clad")
death_day = _event_day(df, "event_death", "tte_death")

n_visits = len(df)
is_flagged = df["flagged"].any()

# ── Header ────────────────────────────────────────────────────────────────────
flag_badge = "  ⚠ DISCORDANCE" if is_flagged else ""
st.markdown(
    f"<h2 style='color:{'#f78166' if is_flagged else TEXT}'>"
    f"🧬 SetMIL-MT — {pid}{flag_badge}</h2>",
    unsafe_allow_html=True,
)

# ── Metric row ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(metric_card("Visits", str(n_visits)), unsafe_allow_html=True)
c2.markdown(metric_card("Follow-up", f"{df['days'].max()}d"), unsafe_allow_html=True)
c3.markdown(metric_card("CLAD", "event" if clad_day else "censored"), unsafe_allow_html=True)
c4.markdown(metric_card("Death", "event" if death_day else "censored"), unsafe_allow_html=True)
c5.markdown(metric_card("Flagged", "yes ⚠" if is_flagged else "no"), unsafe_allow_html=True)

st.divider()

# ── Trajectory plot ───────────────────────────────────────────────────────────
st.markdown(f"<p class='section-title'>Risk Score Trajectory</p>", unsafe_allow_html=True)

fig = go.Figure()

for task in show_tasks:
    vals = df[task].values
    if np.all(np.isnan(vals.astype(float))):
        continue
    fig.add_trace(go.Scatter(
        x=df["days"], y=vals,
        mode="lines+markers",
        name=TASK_LABELS[task],
        line=dict(color=TASK_COLORS[task], width=2),
        marker=dict(size=6, color=TASK_COLORS[task]),
        hovertemplate=(
            f"<b>{TASK_LABELS[task]}</b><br>"
            "Day %{x}: %{y:.3f}<br>"
            "%{customdata}<extra></extra>"
        ),
        customdata=df["anchor_dt"].dt.strftime("%Y-%m-%d"),
    ))

# ACR biopsy vlines — 3 colors: ACR+ red, ACR− green, unknown/None grey
for _, row in df.iterrows():
    lbl = row.get("event_acr")
    day = row["days"]
    date_str = row["anchor_dt"].strftime("%Y-%m-%d") if pd.notna(row.get("anchor_dt")) else ""
    if pd.isna(lbl):
        color, dash, label = "#6c7199", "dot", "ACR unknown"
    elif lbl == 1:
        color, dash, label = "#E53935", "dash", "ACR+"
    else:
        color, dash, label = "#43A047", "dash", "ACR−"
    fig.add_vline(x=day, line_color=color, line_width=1.2,
                  line_dash=dash, opacity=0.6)

# CLAD and death event vlines with date annotation
if clad_day is not None:
    clad_date = (df["anchor_dt"].min() + pd.Timedelta(days=int(clad_day))).strftime("%Y-%m-%d") if pd.notna(df["anchor_dt"].min()) else ""
    fig.add_vline(x=clad_day, line_color="#8E24AA", line_width=2.5,
                  annotation_text=f"CLAD<br>{clad_date}",
                  annotation_font_color="#8E24AA", annotation_font_size=10,
                  annotation_position="top left")
if death_day is not None:
    death_date = (df["anchor_dt"].min() + pd.Timedelta(days=int(death_day))).strftime("%Y-%m-%d") if pd.notna(df["anchor_dt"].min()) else ""
    fig.add_vline(x=death_day, line_color="#00897B", line_width=2.5,
                  annotation_text=f"Death<br>{death_date}",
                  annotation_font_color="#00897B", annotation_font_size=10,
                  annotation_position="top right")

fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED, line_width=1)
# Invisible legend entries for ACR biopsy colors
for acr_label, acr_color in [("ACR+ biopsy", "#E53935"), ("ACR− biopsy", "#43A047"), ("ACR unknown", "#6c7199"),
                               ("CLAD event", "#8E24AA"), ("Death event", "#00897B")]:
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             name=acr_label, line=dict(color=acr_color, width=2, dash="dash"),
                             showlegend=True))
fig.update_layout(
    **PLOTLY_THEME,
    height=400,
    xaxis_title="Days from transplant",
    yaxis_title="Score (0 = low risk, 1 = high risk)",
    yaxis=dict(range=[-0.05, 1.05]),
    legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ── Modality availability ─────────────────────────────────────────────────────
st.markdown(f"<p class='section-title'>Modality Availability</p>", unsafe_allow_html=True)

fig_mod = go.Figure()
for mi, mod in enumerate(MOD_ORDER):
    present_days, absent_days = [], []
    for _, row in df.iterrows():
        mods = set(row["present_mods"].split(",")) if isinstance(row["present_mods"], str) else set()
        if mod in mods:
            present_days.append(row["days"])
        else:
            absent_days.append(row["days"])
    if present_days:
        fig_mod.add_trace(go.Scatter(
            x=present_days, y=[mi] * len(present_days),
            mode="markers", name=mod,
            marker=dict(symbol="square", size=10, color=MOD_COLORS[mod]),
            showlegend=True,
            hovertemplate=f"{mod}<br>Day %{{x}}<extra></extra>",
        ))
    if absent_days:
        fig_mod.add_trace(go.Scatter(
            x=absent_days, y=[mi] * len(absent_days),
            mode="markers", name=f"{mod} (absent)",
            marker=dict(symbol="x", size=7, color=MUTED),
            showlegend=False,
            hovertemplate=f"{mod} absent<br>Day %{{x}}<extra></extra>",
        ))

# ACR biopsy markers on modality timeline
for _, row in df.iterrows():
    lbl = row.get("event_acr")
    day = row["days"]
    if pd.isna(lbl):
        color = "#6c7199"
    elif lbl == 1:
        color = "#E53935"
    else:
        color = "#43A047"
    fig_mod.add_vline(x=day, line_color=color, line_width=1.0, line_dash="dash", opacity=0.5)

# CLAD / death event vlines with date labels
if clad_day is not None:
    clad_date = (df["anchor_dt"].min() + pd.Timedelta(days=int(clad_day))).strftime("%Y-%m-%d") if pd.notna(df["anchor_dt"].min()) else ""
    fig_mod.add_vline(x=clad_day, line_color="#8E24AA", line_width=2.5,
                      annotation_text=f"CLAD {clad_date}", annotation_font_color="#8E24AA",
                      annotation_font_size=9, annotation_position="top left")
if death_day is not None:
    death_date = (df["anchor_dt"].min() + pd.Timedelta(days=int(death_day))).strftime("%Y-%m-%d") if pd.notna(df["anchor_dt"].min()) else ""
    fig_mod.add_vline(x=death_day, line_color="#00897B", line_width=2.5,
                      annotation_text=f"Death {death_date}", annotation_font_color="#00897B",
                      annotation_font_size=9, annotation_position="top right")

fig_mod.update_layout(
    **PLOTLY_THEME,
    height=220,
    xaxis_title="Days from transplant",
    yaxis=dict(tickvals=list(range(len(MOD_ORDER))), ticktext=MOD_ORDER, showgrid=False),
    showlegend=True,
    legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
)
st.plotly_chart(fig_mod, use_container_width=True)

# ── TTE summary table ─────────────────────────────────────────────────────────
st.divider()
st.markdown(f"<p class='section-title'>Event Context (last visit)</p>", unsafe_allow_html=True)
last = df.iloc[-1]
tte_rows = []
for ev_col, tte_col, label in [
    ("event_acr",   "tte_acr",   "ACR"),
    ("event_clad",  "tte_clad",  "CLAD"),
    ("event_death", "tte_death", "Death"),
]:
    ev  = last.get(ev_col)
    tte = last.get(tte_col)
    status = "event" if ev == 1 else ("censored" if ev == 0 else "?")
    tte_str = f"{tte:.0f}d" if pd.notna(tte) else "?"
    tte_rows.append({"Outcome": label, "Status": status, "TTE": tte_str})
st.dataframe(pd.DataFrame(tte_rows), use_container_width=True, hide_index=True)

# ── Full summary PNG ──────────────────────────────────────────────────────────
if show_png:
    st.divider()
    st.markdown(f"<p class='section-title'>Full Summary Panel</p>", unsafe_allow_html=True)
    # Use the last visit's stem (most recent biopsy summary)
    last_stem = str(last["stem"]).zfill(5) if str(last["stem"]).isdigit() else str(last["stem"])
    png_path = setmilmt_summary_png(last_stem)
    if png_path and png_path.exists():
        img = Image.open(png_path)
        st.image(img, use_container_width=True,
                 caption=f"SetMIL-MT summary — {pid} (stem {last_stem})")
    else:
        # Try finding any stem for this patient
        found = False
        for _, row in df.sort_values("days", ascending=False).iterrows():
            s = str(row["stem"]).zfill(5) if str(row["stem"]).isdigit() else str(row["stem"])
            p = setmilmt_summary_png(s)
            if p and p.exists():
                img = Image.open(p)
                st.image(img, use_container_width=True,
                         caption=f"SetMIL-MT summary — {pid} (stem {s})")
                found = True
                break
        if not found:
            st.info("Summary PNG not found for this patient.")
