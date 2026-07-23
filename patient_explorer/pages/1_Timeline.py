"""Page 1 — Patient Timeline & ACR Episodes."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Timeline", page_icon="📅", layout="wide")

from utils.styles import card_css, metric_card, BG, BG2, TEXT, MUTED, ACCENT, BORDER, CARD, ACR_COLORS
from utils.data_loader import (
    patient_splits, patient_predictions, patient_episode, load_splits,
)
from utils.plots import timeline_figure, acr_grade_color

st.markdown(card_css(), unsafe_allow_html=True)

pid = st.session_state.get("selected_patient")
if not pid:
    st.warning("Select a patient from the home page.")
    st.stop()

splits = patient_splits(pid)
preds  = patient_predictions(pid)
ep     = patient_episode(pid)

if splits.empty:
    st.error(f"No records for {pid}")
    st.stop()

st.markdown(f"<h2 style='color:{TEXT}'>📅 Timeline — {pid}</h2>", unsafe_allow_html=True)

# ── Full timeline figure ───────────────────────────────────────────────────
fig = timeline_figure(splits, preds, ep)
fig.update_layout(height=520)
st.plotly_chart(fig, width="stretch")

# ── Episode summary table ─────────────────────────────────────────────────
st.divider()
st.markdown(f"<p class='section-title'>ACR Episodes</p>", unsafe_allow_html=True)

if ep is not None:
    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card("Total episodes", str(ep.get("n_episodes", "?"))), unsafe_allow_html=True)
    c2.markdown(metric_card("ACR+ biopsies",  str(ep.get("n_acr_pos",  "?"))), unsafe_allow_html=True)
    c3.markdown(metric_card("Total biopsies", str(ep.get("n_biopsies", "?"))), unsafe_allow_html=True)

    # Episode bar chart (per-episode duration/size)
    ep_sizes = ep.get("episode_sizes", [])
    ep_durs  = ep.get("episode_durations", [])
    gaps     = ep.get("inter_ep_gaps", [])
    if len(ep_sizes) > 0:
        fig2 = go.Figure()
        eps = list(range(1, len(ep_sizes) + 1))
        fig2.add_trace(go.Bar(
            name="Biopsies in episode",
            x=[f"Ep {i}" for i in eps],
            y=ep_sizes,
            marker_color=ACCENT,
            hovertemplate="Episode %{x}: %{y} biopsy/biopsies<extra></extra>",
        ))
        if ep_durs and any(d > 0 for d in ep_durs):
            fig2.add_trace(go.Bar(
                name="Duration (days)",
                x=[f"Ep {i}" for i in eps],
                y=ep_durs,
                marker_color="#f78166",
                hovertemplate="Episode %{x}: %{y} days<extra></extra>",
            ))
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG2,
            height=260, margin=dict(l=20, r=20, t=20, b=20),
            barmode="group",
            xaxis=dict(gridcolor=BORDER),
            yaxis=dict(gridcolor=BORDER),
            legend=dict(orientation="h", x=0, y=-0.2, font_size=10),
        )
        st.plotly_chart(fig2, width="stretch")

# ── Biopsy-level table ─────────────────────────────────────────────────────
st.divider()
st.markdown(f"<p class='section-title'>All Biopsies</p>", unsafe_allow_html=True)

cols_show = ["anchor_dt", "acr_grade", "acr_encoded", "days_since_tx",
             "has_HE", "has_BAL", "has_CT", "has_Clinical",
             "clad_days", "death_days"]
cols_show = [c for c in cols_show if c in splits.columns]
tbl = splits[cols_show].copy()
if "anchor_dt" in tbl.columns:
    tbl["anchor_dt"] = tbl["anchor_dt"].dt.strftime("%Y-%m-%d")

# append predictions if available
if not preds.empty:
    pred_cols = ["anchor_dt", "pred_prob_acr", "hazard_acr", "hazard_clad", "hazard_death"]
    pred_cols = [c for c in pred_cols if c in preds.columns]
    p = preds[pred_cols].copy()
    if "anchor_dt" in p.columns:
        p["anchor_dt"] = p["anchor_dt"].dt.strftime("%Y-%m-%d")
    tbl = tbl.merge(p, on="anchor_dt", how="left")

def _highlight_acr(row):
    grade = row.get("acr_grade", "") or ""
    if "A1" in str(grade) or "A2" in str(grade) or "A3" in str(grade):
        return ["background-color: rgba(224,92,75,0.18)"] * len(row)
    return [""] * len(row)

tbl = tbl.reset_index(drop=True)
float_cols = [c for c in tbl.columns if tbl[c].dtype == float]
fmt = {c: "{:.3f}" for c in float_cols if "prob" in c or "hazard" in c}
st.dataframe(
    tbl.style.apply(_highlight_acr, axis=1).format(fmt, na_rep="—"),
    width="stretch",
    height=min(42 * len(tbl) + 40, 480),
)

# download button
csv = tbl.to_csv(index=False)
st.download_button("⬇ Download CSV", data=csv,
                   file_name=f"{pid}_timeline.csv", mime="text/csv")
