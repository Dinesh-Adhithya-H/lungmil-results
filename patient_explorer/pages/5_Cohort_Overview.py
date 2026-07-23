# pages/5_Cohort_Overview.py
"""Page 5 — Cohort Overview: population-level statistics across all patients."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Cohort Overview", page_icon="🌍", layout="wide")

from utils.styles import (
    card_css, metric_card,
    BG, BG2, TEXT, MUTED, ACCENT, BORDER, CARD,
    ACR_COLORS, MOD_COLORS, PLOTLY_THEME,
)
from utils.data_loader import load_splits, load_setmilmt, DATA_DIR

st.markdown(card_css(), unsafe_allow_html=True)
st.markdown(f"<h2 style='color:{TEXT}'>🌍 Cohort Overview</h2>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{MUTED}'>Population-level analysis across all patients and timepoints.</p>", unsafe_allow_html=True)

# ── Data ────────────────────────────────────────────────────────────────────
splits = load_splits()

if splits.empty:
    st.error("No data found. Run `export_data.py` first.")
    st.stop()

# derived
splits = splits.copy()
splits["acr_label"] = splits["acr_encoded"].map(lambda x: "ACR+" if x == 1 else "ACR−")
splits["month"] = pd.to_datetime(splits["anchor_dt"]).dt.to_period("M").astype(str)

patients = splits["patient_id"].nunique()
n_samples = len(splits)
acr_events = int((splits["acr_encoded"] == 1).sum())

# CLAD / Death — one row per patient (first row)
pt_first = splits.sort_values("anchor_dt").groupby("patient_id").first().reset_index()
clad_events = int(pt_first["clad_status"].eq(1.0).sum()) if "clad_status" in pt_first.columns else 0
death_events = int(pt_first["death_status"].eq(1.0).sum()) if "death_status" in pt_first.columns else 0

# optional cohort_summary.json
summary_path = DATA_DIR / "cohort_summary.json"
cohort_json = {}
if summary_path.exists():
    with open(summary_path) as f:
        cohort_json = json.load(f)

# ── Metric cards ─────────────────────────────────────────────────────────────
st.markdown(f"<p class='section-title'>Summary</p>", unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6 = st.columns(6)
spp = splits.groupby("patient_id").size()
with c1: st.markdown(metric_card("Patients", str(patients)), unsafe_allow_html=True)
with c2: st.markdown(metric_card("Samples", str(n_samples), f"~{n_samples/patients:.1f}/patient"), unsafe_allow_html=True)
with c3: st.markdown(metric_card("ACR events", str(acr_events), f"{100*acr_events/n_samples:.0f}%"), unsafe_allow_html=True)
with c4: st.markdown(metric_card("CLAD events", str(clad_events), f"{100*clad_events/patients:.0f}% of pts"), unsafe_allow_html=True)
with c5: st.markdown(metric_card("Deaths", str(death_events), f"{100*death_events/patients:.0f}% of pts"), unsafe_allow_html=True)
with c6: st.markdown(metric_card("Serial patients", str(int((spp > 1).sum())), f"max {spp.max()} samples"), unsafe_allow_html=True)

st.divider()

# ── Row 1: ACR grade pie + CLAD/Death bars ───────────────────────────────────
st.markdown(f"<p class='section-title'>Outcome Distributions</p>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    grade_counts = splits["acr_grade"].value_counts()
    colors_pie = [ACR_COLORS.get(g, MUTED) for g in grade_counts.index]
    fig = go.Figure(go.Pie(
        labels=grade_counts.index, values=grade_counts.values,
        marker=dict(colors=colors_pie, line=dict(color=BG, width=1)),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        textinfo="label+percent",
    ))
    fig.update_layout(**PLOTLY_THEME, title="ACR Grade Distribution", height=320,
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # CLAD by ACR grade at first visit
    if "clad_status" in pt_first.columns and "acr_encoded" in pt_first.columns:
        clad_by_acr = pt_first.groupby("acr_encoded")["clad_status"].mean().reset_index()
        clad_by_acr["acr_label"] = clad_by_acr["acr_encoded"].map({0: "ACR−", 1: "ACR+"})
        fig2 = go.Figure(go.Bar(
            x=clad_by_acr["acr_label"], y=clad_by_acr["clad_status"],
            marker_color=[ACR_COLORS["ACR-"], ACR_COLORS["ACR+"]],
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        ))
        fig2.update_layout(**PLOTLY_THEME, title="CLAD Rate by ACR Status",
                           yaxis_tickformat=".0%", height=320)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("CLAD data not available.")

with col3:
    if "death_status" in pt_first.columns and "clad_status" in pt_first.columns:
        outcomes = pd.DataFrame({
            "Outcome": ["CLAD", "Death", "CLAD+Death"],
            "Count": [
                clad_events,
                death_events,
                int(pt_first[pt_first["clad_status"].eq(1.0) & pt_first["death_status"].eq(1.0)].shape[0]),
            ]
        })
        fig3 = go.Figure(go.Bar(
            x=outcomes["Outcome"], y=outcomes["Count"],
            marker_color=[MOD_COLORS["CT"], "#ff7b72", "#bf3a2e"],
            text=outcomes["Count"], textposition="outside",
        ))
        fig3.update_layout(**PLOTLY_THEME, title="Adverse Event Counts", height=320)
        st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Row 2: Modality availability ─────────────────────────────────────────────
st.markdown(f"<p class='section-title'>Modality Availability</p>", unsafe_allow_html=True)
col4, col5 = st.columns([1.4, 1])

with col4:
    mod_cols = [c for c in ["has_CT", "has_HE", "has_BAL", "has_Clinical"] if c in splits.columns]
    if mod_cols:
        mod_labels = [c.replace("has_", "") for c in mod_cols]
        acr_groups = splits["acr_label"].unique()
        fig4 = go.Figure()
        colors_mod = [MOD_COLORS.get(m, MUTED) for m in mod_labels]
        for mod_col, mod_lbl, col_hex in zip(mod_cols, mod_labels, colors_mod):
            acr_n = splits.groupby("acr_label")[mod_col].sum()
            acr_pct = acr_n / splits.groupby("acr_label").size() * 100
            for i, grp in enumerate(["ACR−", "ACR+"]):
                if grp in acr_pct.index:
                    fig4.add_trace(go.Bar(
                        name=f"{mod_lbl} {grp}",
                        x=[mod_lbl],
                        y=[acr_pct[grp]],
                        marker_color=col_hex,
                        opacity=0.6 if grp == "ACR−" else 1.0,
                        offsetgroup=grp,
                        legendgroup=grp,
                        showlegend=(mod_lbl == mod_labels[0]),
                    ))
        fig4.update_layout(**PLOTLY_THEME, barmode="group",
                           title="Modality Availability by ACR Status (%)",
                           yaxis_title="% of group samples", height=350)
        st.plotly_chart(fig4, use_container_width=True)

with col5:
    # Co-availability heatmap
    mod_cols_avail = [c for c in ["has_Clinical", "has_HE", "has_CT", "has_BAL"] if c in splits.columns]
    if len(mod_cols_avail) >= 2:
        labels_avail = [c.replace("has_", "") for c in mod_cols_avail]
        n = len(mod_cols_avail)
        mat = np.zeros((n, n), dtype=int)
        for i, a in enumerate(mod_cols_avail):
            for j, b in enumerate(mod_cols_avail):
                mat[i, j] = int((splits[a] & splits[b]).sum())
        fig5 = go.Figure(go.Heatmap(
            z=mat, x=labels_avail, y=labels_avail,
            colorscale="Blues",
            text=mat, texttemplate="%{text}",
            hovertemplate="Both %{x}+%{y}: %{z} samples<extra></extra>",
        ))
        fig5.update_layout(**PLOTLY_THEME, title="Co-availability (# samples with both)",
                           height=350)
        st.plotly_chart(fig5, use_container_width=True)

st.divider()

# ── Row 3: Temporal views ─────────────────────────────────────────────────────
st.markdown(f"<p class='section-title'>Temporal Analysis</p>", unsafe_allow_html=True)
col6, col7 = st.columns(2)

with col6:
    # Cohort timeline: samples per month
    monthly = splits.groupby("month").size().reset_index(name="n_samples")
    acr_monthly = splits[splits["acr_label"]=="ACR+"].groupby("month").size().reset_index(name="acr_n")
    monthly = monthly.merge(acr_monthly, on="month", how="left").fillna(0)
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(x=monthly["month"], y=monthly["n_samples"],
                          name="All samples", marker_color=ACCENT, opacity=0.7))
    fig6.add_trace(go.Bar(x=monthly["month"], y=monthly["acr_n"],
                          name="ACR+ samples", marker_color=ACR_COLORS["ACR+"]))
    fig6.update_layout(**PLOTLY_THEME, barmode="overlay",
                       title="Samples per Month", xaxis_title="Month",
                       yaxis_title="# Samples", height=320,
                       xaxis_tickangle=-45)
    st.plotly_chart(fig6, use_container_width=True)

with col7:
    # Samples per patient histogram
    fig7 = go.Figure(go.Histogram(
        x=spp.values, nbinsx=20,
        marker_color=MOD_COLORS["Clinical"],
        opacity=0.85,
    ))
    fig7.update_layout(**PLOTLY_THEME, title="Samples per Patient Distribution",
                       xaxis_title="# Samples", yaxis_title="# Patients", height=320)
    st.plotly_chart(fig7, use_container_width=True)

st.divider()

# ── Row 4: Days since transplant by modality ──────────────────────────────────
st.markdown(f"<p class='section-title'>Post-transplant Sampling Distribution</p>", unsafe_allow_html=True)

if "days_since_tx" in splits.columns:
    mod_cols_dxt = [c for c in ["has_Clinical","has_HE","has_CT","has_BAL"] if c in splits.columns]
    if mod_cols_dxt:
        fig8 = go.Figure()
        for mod_col in mod_cols_dxt:
            lbl = mod_col.replace("has_", "")
            sub = splits[splits[mod_col].astype(bool)]["days_since_tx"].dropna()
            fig8.add_trace(go.Violin(
                y=sub.values, name=lbl,
                box_visible=True, meanline_visible=True,
                line_color=MOD_COLORS.get(lbl, MUTED),
                fillcolor=MOD_COLORS.get(lbl, MUTED),
                opacity=0.6,
                points=False,
            ))
        fig8.update_layout(**PLOTLY_THEME,
                           title="Days Since Transplant at Sampling (by Modality)",
                           yaxis_title="Days since transplant", height=380)
        st.plotly_chart(fig8, use_container_width=True)
else:
    st.info("days_since_tx column not available.")

st.divider()

# ── Row 5: CLAD & Death survival distributions ────────────────────────────────
st.markdown(f"<p class='section-title'>Outcome Survival Distributions</p>", unsafe_allow_html=True)

df_smt = load_setmilmt()
if not df_smt.empty:
    # Per-patient: first visit (days_from_tx=0) has tte = time-to-event from transplant
    pt_surv = (
        df_smt.sort_values("days_from_tx")
        .groupby("patient_id")
        .first()
        .reset_index()[["patient_id", "event_clad", "tte_clad", "event_death", "tte_death"]]
    )

    def km_curve(times, events):
        """Kaplan-Meier estimator."""
        df_km = pd.DataFrame({"time": times, "event": events}).sort_values("time").reset_index(drop=True)
        at_risk = len(df_km)
        T, S = [0], [1.0]
        for _, row in df_km.iterrows():
            if row["event"] > 0:
                S.append(S[-1] * (1 - 1 / max(at_risk, 1)))
                T.append(row["time"] / 365.25)  # convert days → years
            at_risk -= 1
        return T, S

    col_c, col_d = st.columns(2)
    with col_c:
        times_c = pt_surv["tte_clad"].dropna().values
        events_c = pt_surv["event_clad"].fillna(0).values[:len(times_c)]
        T_c, S_c = km_curve(times_c, events_c)
        n_clad_ev = int((pt_surv["event_clad"] > 0).sum())
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(
            x=T_c, y=S_c, mode="lines", name="CLAD-free survival",
            line=dict(color="#8E24AA", width=2.5, shape="hv"),
            fill="tozeroy", fillcolor="rgba(142,36,170,0.1)",
        ))
        fig_c.update_layout(
            **PLOTLY_THEME,
            title=f"CLAD-free Survival (n={n_clad_ev} events)",
            xaxis_title="Years from transplant",
            yaxis_title="Survival probability",
            yaxis=dict(range=[0, 1.05]),
            height=340,
        )
        st.plotly_chart(fig_c, use_container_width=True)

        # TTE histogram
        fig_ch = go.Figure()
        clad_ev = pt_surv[pt_surv["event_clad"] > 0]["tte_clad"].dropna() / 365.25
        clad_cx = pt_surv[pt_surv["event_clad"] <= 0]["tte_clad"].dropna() / 365.25
        fig_ch.add_trace(go.Histogram(x=clad_ev, name="CLAD event", nbinsx=20,
                                       marker_color="#8E24AA", opacity=0.85))
        fig_ch.add_trace(go.Histogram(x=clad_cx, name="Censored", nbinsx=20,
                                       marker_color=MUTED, opacity=0.5))
        fig_ch.update_layout(**PLOTLY_THEME, barmode="overlay",
                              title="Time to CLAD (years)", xaxis_title="Years",
                              yaxis_title="# Patients", height=260)
        st.plotly_chart(fig_ch, use_container_width=True)

    with col_d:
        times_d = pt_surv["tte_death"].dropna().values
        events_d = pt_surv["event_death"].fillna(0).values[:len(times_d)]
        T_d, S_d = km_curve(times_d, events_d)
        n_death_ev = int((pt_surv["event_death"] > 0).sum())
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(
            x=T_d, y=S_d, mode="lines", name="Overall survival",
            line=dict(color="#00897B", width=2.5, shape="hv"),
            fill="tozeroy", fillcolor="rgba(0,137,123,0.1)",
        ))
        fig_d.update_layout(
            **PLOTLY_THEME,
            title=f"Overall Survival (n={n_death_ev} events)",
            xaxis_title="Years from transplant",
            yaxis_title="Survival probability",
            yaxis=dict(range=[0, 1.05]),
            height=340,
        )
        st.plotly_chart(fig_d, use_container_width=True)

        fig_dh = go.Figure()
        death_ev = pt_surv[pt_surv["event_death"] > 0]["tte_death"].dropna() / 365.25
        death_cx = pt_surv[pt_surv["event_death"] <= 0]["tte_death"].dropna() / 365.25
        fig_dh.add_trace(go.Histogram(x=death_ev, name="Death event", nbinsx=20,
                                       marker_color="#00897B", opacity=0.85))
        fig_dh.add_trace(go.Histogram(x=death_cx, name="Censored", nbinsx=20,
                                       marker_color=MUTED, opacity=0.5))
        fig_dh.update_layout(**PLOTLY_THEME, barmode="overlay",
                              title="Time to Death (years)", xaxis_title="Years",
                              yaxis_title="# Patients", height=260)
        st.plotly_chart(fig_dh, use_container_width=True)
else:
    st.info("Survival data not available (setmilmt_preds.csv missing).")
