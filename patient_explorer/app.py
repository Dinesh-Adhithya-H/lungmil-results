"""Patient Explorer — main page (overview + patient selector)."""

import os
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Lung Transplant · Patient Explorer",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Password gate ──────────────────────────────────────────────────────────
_CORRECT_PWD = os.environ.get("EXPLORER_PASSWORD", "lungmil2024")

if not st.session_state.get("_authenticated"):
    st.markdown("## 🫁 Lung Transplant · Patient Explorer")
    st.markdown("This tool contains research data. Please enter the access password.")
    pwd = st.text_input("Password", type="password", key="_pwd_input")
    if st.button("Enter"):
        if pwd == _CORRECT_PWD:
            st.session_state["_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

from utils.styles import card_css, metric_card, BG, TEXT, MUTED, ACCENT
from utils.data_loader import (
    load_splits, load_predictions, load_episodes,
    patient_list, patient_splits, patient_predictions, patient_episode,
    available_data,
)

st.markdown(card_css(), unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"<h2 style='color:{TEXT};margin-bottom:4px'>🫁 Patient Explorer</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};font-size:0.82rem'>Lung Transplant · MiddleFusionMIL</p>", unsafe_allow_html=True)
    st.divider()

    patients = patient_list()
    if not patients:
        st.error("No data found. Run `export_data.py` first.")
        st.stop()

    query = st.text_input("Search patient", placeholder="LT001…", label_visibility="collapsed")
    filtered = [p for p in patients if query.upper() in p.upper()] if query else patients

    selected = st.selectbox("Patient", filtered, label_visibility="collapsed")
    st.session_state["selected_patient"] = selected

    st.divider()
    st.markdown(f"<p class='section-title' style='color:{MUTED};font-size:.7rem;text-transform:uppercase;letter-spacing:.08em'>Patient views</p>", unsafe_allow_html=True)
    st.page_link("pages/1_Timeline.py",       label="📅  Timeline & Episodes")
    st.page_link("pages/2_Clusters.py",       label="🔬  Cluster Composition")
    st.page_link("pages/3_Latent_Space.py",   label="🗺  Latent Space")
    st.page_link("pages/4_Clinical.py",       label="📈  Clinical Features")
    st.page_link("pages/7_MIL_Patches.py",    label="🧩  MIL Patches")
    st.page_link("pages/8_Longitudinal.py",   label="📉  Longitudinal Trajectories")
    st.divider()
    st.markdown(f"<p class='section-title' style='color:{MUTED};font-size:.7rem;text-transform:uppercase;letter-spacing:.08em'>Cohort analysis</p>", unsafe_allow_html=True)
    st.page_link("pages/5_Cohort_Overview.py",  label="🌐  Cohort Overview")
    st.page_link("pages/6_Benchmark.py",         label="🏆  Model Benchmark")
    st.page_link("pages/9_Differential.py",      label="🔥  Differential Abundance")
    st.page_link("pages/10_Factor_Analysis.py",  label="📊  Factor Analysis")

# ── Main: patient overview ─────────────────────────────────────────────────

pid = selected
splits = patient_splits(pid)
preds  = patient_predictions(pid)
ep     = patient_episode(pid)
avail  = available_data(pid)

if splits.empty:
    st.error(f"No records found for {pid}.")
    st.stop()

# header
row0 = splits.iloc[0]
tx_date  = splits["anchor_dt"].min().strftime("%Y-%m-%d")
last_dt  = splits["anchor_dt"].max().strftime("%Y-%m-%d")
n_biops  = len(splits)
n_acr_pos = int((splits["acr_encoded"] == 1).sum()) if "acr_encoded" in splits.columns else "?"
clad_evt  = bool(splits["clad_status"].iloc[0])  if "clad_status" in splits.columns else False
death_evt = bool(splits["death_status"].iloc[0]) if "death_status" in splits.columns else False

modalities = " · ".join(
    m for m, k in [("HE","has_HE"),("BAL","has_BAL"),("CT","has_CT"),("Clinical","has_Clinical")]
    if (k in splits.columns and splits[k].any())
)

st.markdown(f"<h1 style='color:{TEXT};margin-bottom:0'>{pid}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{MUTED};margin-top:0'>{modalities}</p>", unsafe_allow_html=True)

# metric cards
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.markdown(metric_card("Biopsies", str(n_biops), f"{tx_date} → {last_dt}"), unsafe_allow_html=True)
with c2: st.markdown(metric_card("ACR+ events", str(n_acr_pos)), unsafe_allow_html=True)
with c3:
    episodes = ep["n_episodes"] if ep is not None and "n_episodes" in ep.index else "?"
    st.markdown(metric_card("ACR episodes", str(episodes)), unsafe_allow_html=True)
with c4:
    clad_str = f"✓ day {int(splits['clad_days'].iloc[0])}" if clad_evt else "No"
    st.markdown(metric_card("CLAD", clad_str), unsafe_allow_html=True)
with c5:
    death_str = f"✓ day {int(splits['death_days'].iloc[0])}" if death_evt else "No"
    st.markdown(metric_card("Death", death_str), unsafe_allow_html=True)

st.divider()

# ACR grade timeline (compact)
st.markdown(f"<p class='section-title'>ACR Grade Timeline</p>", unsafe_allow_html=True)
from utils.plots import timeline_figure
fig = timeline_figure(splits, preds, ep)
st.plotly_chart(fig, use_container_width=True)

# Data availability badges
st.divider()
st.markdown(f"<p class='section-title'>Data Availability</p>", unsafe_allow_html=True)
cols = st.columns(7)
badges = [
    ("HE",       avail.get("he_freq", False) or splits["has_HE"].any() if "has_HE" in splits.columns else False),
    ("BAL",      avail.get("bal_freq", False) or splits["has_BAL"].any() if "has_BAL" in splits.columns else False),
    ("CT",       avail.get("ct_freq", False) or splits["has_CT"].any() if "has_CT" in splits.columns else False),
    ("Clinical", "has_Clinical" in splits.columns and splits["has_Clinical"].any()),
    ("UMAP",     avail.get("umap", False)),
    ("Preds",    avail.get("predictions", False)),
    ("Episodes", avail.get("timeline", False)),
]
for col, (label, ok) in zip(cols, badges):
    icon  = "✅" if ok else "⬜"
    color = "#3fb950" if ok else MUTED
    col.markdown(f"<div style='text-align:center;color:{color};font-size:0.85rem'>{icon}<br>{label}</div>",
                 unsafe_allow_html=True)

# Cohort snapshot
st.divider()
st.markdown(f"<p class='section-title'>Cohort Snapshot</p>", unsafe_allow_html=True)
all_splits = load_splits()
cc1, cc2 = st.columns(2)
with cc1:
    from utils.plots import grade_pie
    st.plotly_chart(grade_pie(all_splits, pid), use_container_width=True)
with cc2:
    n_pat = all_splits["patient_id"].nunique()
    n_tp  = len(all_splits)
    n_clad = int(all_splits.groupby("patient_id")["clad_status"].first().sum()) \
             if "clad_status" in all_splits.columns else "?"
    n_death = int(all_splits.groupby("patient_id")["death_status"].first().sum()) \
              if "death_status" in all_splits.columns else "?"
    st.markdown(
        f"<div style='padding:16px'>"
        f"<p style='color:{MUTED};font-size:.75rem;text-transform:uppercase;letter-spacing:.06em'>Cohort</p>"
        f"<p style='color:{TEXT};font-size:1.1rem'><b>{n_pat}</b> patients · <b>{n_tp}</b> timepoints</p>"
        f"<p style='color:{TEXT}'>CLAD events: <b>{n_clad}</b></p>"
        f"<p style='color:{TEXT}'>Deaths: <b>{n_death}</b></p>"
        f"</div>",
        unsafe_allow_html=True,
    )
