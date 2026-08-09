"""Page 9 — Illustrative Patient Cases: LongMK risk trajectories for clinical presentation."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Illustrative Cases", page_icon="🔍", layout="wide")

from utils.styles import card_css, metric_card, BG, BG2, TEXT, MUTED, ACCENT, BORDER, CARD, PLOTLY_THEME
from utils.data_loader import load_longi_preds, patient_longi_preds

st.markdown(card_css(), unsafe_allow_html=True)

DATA = Path(__file__).resolve().parent.parent / "data"

TASK_COLORS = {
    "pct_death_surv": "#C62828",
    "pct_clad_surv":  "#8E24AA",
    "pct_acr_surv":   "#1565C0",
    "score_acr_cls":  "#7c83ff",
}
TASK_LABELS = {
    "pct_death_surv": "Death risk",
    "pct_clad_surv":  "CLAD risk",
    "pct_acr_surv":   "ACR survival risk",
    "score_acr_cls":  "ACR+ probability",
}

# Three hand-picked illustrative patients
CASES = [
    {
        "pid":     "LT070",
        "title":   "Case A — Early warning: rising mortality risk",
        "icon":    "⚠️",
        "outcome": "Died (+ CLAD)",
        "story":   (
            "Initial post-transplant risk was very low (0.6%). "
            "The model detected a progressive rise in mortality risk across 16 serial biopsies "
            "before the terminal outcome. Risk exceeded 90% more than 6 months before death, "
            "providing a potential window for intensified monitoring or intervention."
        ),
        "highlight_task": "pct_death_surv",
        "color_outcome": "#C62828",
    },
    {
        "pid":     "LT073",
        "title":   "Case B — Stable low risk: consistent long-term survivor",
        "icon":    "✅",
        "outcome": "Alive (no events)",
        "story":   (
            "Across 21 biopsies over the full follow-up period, the model consistently assigned "
            "very low mortality risk (0.4%–4.5%). Death and CLAD risk remained flat throughout, "
            "correctly identifying this patient as a stable long-term survivor."
        ),
        "highlight_task": "pct_death_surv",
        "color_outcome": "#2E7D32",
    },
    {
        "pid":     "LT038",
        "title":   "Case C — Gradual deterioration: slow rise over years",
        "icon":    "📈",
        "outcome": "Died",
        "story":   (
            "Over 36 biopsies — the longest follow-up in the cohort — the model tracked a "
            "gradual, sustained increase in mortality risk from 37.7% to 98.7%. "
            "This trajectory illustrates how the longitudinal model captures slow deterioration "
            "that may not be apparent from single time-point assessments."
        ),
        "highlight_task": "pct_death_surv",
        "color_outcome": "#C62828",
    },
]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f"<h2 style='color:{TEXT}'>🔍 Illustrative Patient Cases</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#888;font-size:0.92em;'>Three representative patients from the test set "
    "illustrating how LongMK tracks risk over serial biopsies. "
    "All patients are from held-out cross-validation folds.</p>",
    unsafe_allow_html=True,
)
st.divider()

sm = pd.read_csv(DATA / "setmilmt_preds.csv", parse_dates=["anchor_dt"])
lk = load_longi_preds()


def _get_outcome(pid):
    pt = sm[sm["patient_id"] == pid]
    died  = pt["event_death"].max() == 1
    clad  = pt["event_clad"].max() == 1
    tte_d = pt["tte_death"].dropna().values
    tte_c = pt["tte_clad"].dropna().values
    return {
        "died":   died,
        "clad":   clad,
        "tte_death": float(tte_d[0]) if len(tte_d) else None,
        "tte_clad":  float(tte_c[0]) if len(tte_c) else None,
    }


def _trajectory_fig(pid, highlight_task, color_outcome):
    df = patient_longi_preds(pid)
    if df.empty:
        return None

    df = df.copy().sort_values("anchor_dt")
    t0 = df["anchor_dt"].min()
    df["days"] = (df["anchor_dt"] - t0).dt.days

    out = _get_outcome(pid)

    fig = go.Figure()

    # All 4 task risk scores
    for col, color in TASK_COLORS.items():
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        if np.all(np.isnan(vals)):
            continue
        lw = 3.5 if col == highlight_task else 1.5
        opacity = 1.0 if col == highlight_task else 0.45
        fig.add_trace(go.Scatter(
            x=df["days"], y=vals,
            mode="lines+markers",
            name=TASK_LABELS[col],
            line=dict(color=color, width=lw, dash="solid" if col == highlight_task else "dot"),
            marker=dict(size=8 if col == highlight_task else 4, color=color),
            opacity=opacity,
            hovertemplate=(
                f"<b>{TASK_LABELS[col]}</b><br>"
                "Day %{x}: %{y:.1%}<extra></extra>"
            ),
        ))

    # Event lines
    if out["tte_death"] and out["died"]:
        fig.add_vline(
            x=float(out["tte_death"]), line_color="#C62828", line_width=2.5,
            annotation_text="Death", annotation_font_color="#C62828",
            annotation_position="top left",
        )
    if out["tte_clad"] and out["clad"]:
        fig.add_vline(
            x=float(out["tte_clad"]), line_color="#8E24AA", line_width=2,
            line_dash="dash",
            annotation_text="CLAD onset", annotation_font_color="#8E24AA",
            annotation_position="top right",
        )

    # 50% reference
    fig.add_hline(y=0.5, line_dash="dot", line_color="#999", line_width=1,
                  annotation_text="50%", annotation_font_size=10)

    fig.update_layout(
        **PLOTLY_THEME, height=380,
        xaxis_title="Days from transplant",
        yaxis_title="Predicted risk (0 = low, 1 = high)",
        yaxis=dict(range=[-0.05, 1.05], tickformat=".0%"),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1, font=dict(size=12)),
        hovermode="x unified",
        margin=dict(t=20, b=50),
    )
    return fig


# ── Render each case ──────────────────────────────────────────────────────────
for case in CASES:
    pid   = case["pid"]
    out   = _get_outcome(pid)
    df_lk = patient_longi_preds(pid)
    n_vis = len(df_lk) if not df_lk.empty else "?"

    # Case header
    st.markdown(
        f"<h3 style='color:{TEXT}'>{case['icon']}  {case['title']}</h3>",
        unsafe_allow_html=True,
    )

    # Metric row
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Patient ID", pid), unsafe_allow_html=True)
    c2.markdown(metric_card("Outcome", case["outcome"]), unsafe_allow_html=True)
    c3.markdown(metric_card("Serial biopsies", str(n_vis)), unsafe_allow_html=True)
    tte_str = f"{int(out['tte_death'])} days" if out.get("tte_death") else "censored"
    c4.markdown(metric_card("Follow-up / TTE", tte_str), unsafe_allow_html=True)

    # Clinical story
    st.info(case["story"])

    # Trajectory plot
    fig = _trajectory_fig(pid, case["highlight_task"], case["color_outcome"])
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No LongMK prediction data available for {pid}.")

    st.divider()

st.caption(
    "LongMK: Longitudinal Multi-Key attention model. Predictions shown are from the held-out "
    "test fold for each patient (out-of-sample). Risk scores are percentile-normalised hazard "
    "estimates from the survival head."
)
