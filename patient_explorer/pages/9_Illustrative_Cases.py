"""Page 9 — Illustrative Patient Cases: best-model risk trajectories for clinical presentation."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Illustrative Cases", page_icon="🔍", layout="wide")

from utils.styles import card_css, metric_card, BG, BG2, TEXT, MUTED, ACCENT, BORDER, CARD, PLOTLY_THEME
from utils.data_loader import load_longi_preds, patient_longi_preds, load_setmilmt, patient_setmilmt

st.markdown(card_css(), unsafe_allow_html=True)

# Best model per task
BEST_MODEL_SOURCE = {
    "pct_death_surv": "longi",    # LongMK  C-index 0.771
    "pct_acr_surv":   "longi",    # LongMK  C-index 0.679
    "pct_clad_surv":  "setmil",   # SetMIL-MT C-index 0.563
    "score_acr_cls":  "setmil",   # SetMIL-MT (no SAB) BACC 0.623
}
BEST_MODEL_LABEL = {
    "pct_death_surv": "LongMK",
    "pct_acr_surv":   "LongMK",
    "pct_clad_surv":  "SetMIL-MT",
    "score_acr_cls":  "SetMIL-MT (no SAB)",
}
TASK_COLORS = {
    "pct_death_surv": "#3fb950",
    "pct_clad_surv":  "#f78166",
    "pct_acr_surv":   "#58a6ff",
    "score_acr_cls":  "#7c83ff",
}
TASK_LABELS = {
    "pct_death_surv": "Death risk",
    "pct_clad_surv":  "CLAD risk",
    "pct_acr_surv":   "ACR survival risk",
    "score_acr_cls":  "ACR+ probability",
}

CASES = [
    {
        "pid":     "LT070",
        "title":   "Case A — Early warning: rising mortality risk",
        "icon":    "⚠️",
        "outcome": "Died (+ CLAD)",
        "story":   (
            "Initial post-transplant risk was very low (0.6%). "
            "LongMK detected a progressive rise in mortality risk across 16 serial biopsies "
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
            "Across 21 biopsies over the full follow-up period, LongMK consistently assigned "
            "very low mortality risk (0.4%–4.5%). Death and CLAD risk remained flat throughout, "
            "correctly identifying this patient as a stable long-term survivor. "
            "SetMIL-MT confirms low ACR+ probability at every visit."
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
            "Over 36 biopsies — the longest follow-up in the cohort — LongMK tracked a "
            "gradual, sustained increase in mortality risk from 37.7% to 98.7%. "
            "This trajectory illustrates how the longitudinal model captures slow deterioration "
            "that may not be apparent from single time-point assessments. "
            "CLAD risk (SetMIL-MT) rose in parallel, corroborating the deterioration signal."
        ),
        "highlight_task": "pct_death_surv",
        "color_outcome": "#C62828",
    },
]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"<h2 style='color:{TEXT}'>🔍 Illustrative Patient Cases</h2>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{MUTED};font-size:0.92em'>"
    "Three representative patients from held-out CV folds. "
    "Each task uses its best model: "
    "<b style='color:#C62828'>LongMK</b> for Death risk & ACR survival; "
    "<b style='color:#9C27B0'>SetMIL-MT</b> for CLAD risk & ACR classification."
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

lk  = load_longi_preds()
smt = load_setmilmt()


def _get_outcome(pid):
    pt = smt[smt["patient_id"] == pid] if not smt.empty else pd.DataFrame()
    if pt.empty:
        return {"died": False, "clad": False, "tte_death": None, "tte_clad": None}
    died  = pt["event_death"].max() == 1
    clad  = pt["event_clad"].max() == 1
    tte_d = pt["tte_death"].dropna().values
    tte_c = pt["tte_clad"].dropna().values
    return {
        "died":     died,
        "clad":     clad,
        "tte_death": float(tte_d[0]) if len(tte_d) else None,
        "tte_clad":  float(tte_c[0]) if len(tte_c) else None,
    }


def _trajectory_fig(pid, highlight_task, color_outcome):
    df_lk  = patient_longi_preds(pid)
    df_smt = patient_setmilmt(pid)
    out    = _get_outcome(pid)

    if df_lk.empty and df_smt.empty:
        return None

    # Build a unified time axis
    all_days = set()
    if not df_lk.empty:
        df_lk = df_lk.copy().sort_values("anchor_dt")
        t0_lk = df_lk["anchor_dt"].min()
        df_lk["days"] = (df_lk["anchor_dt"] - t0_lk).dt.days
        all_days.update(df_lk["days"].tolist())
    if not df_smt.empty:
        df_smt = df_smt.copy().sort_values("anchor_dt")
        t0_smt = df_smt["anchor_dt"].min()
        df_smt["days"] = (df_smt["anchor_dt"] - t0_smt).dt.days
        all_days.update(df_smt["days"].tolist())

    fig = go.Figure()

    for task in ["pct_death_surv", "pct_acr_surv", "pct_clad_surv", "score_acr_cls"]:
        source = BEST_MODEL_SOURCE[task]
        color  = TASK_COLORS[task]
        model  = BEST_MODEL_LABEL[task]
        is_hl  = (task == highlight_task)

        if source == "longi" and not df_lk.empty and task in df_lk.columns:
            df_src = df_lk
        elif source == "setmil" and not df_smt.empty and task in df_smt.columns:
            df_src = df_smt
        else:
            continue

        vals = df_src[task].values.astype(float)
        if np.all(np.isnan(vals)):
            continue

        fig.add_trace(go.Scatter(
            x=df_src["days"], y=vals,
            mode="lines+markers",
            name=f"{'★ ' if is_hl else ''}{TASK_LABELS[task]}  [{model}]",
            line=dict(color=color, width=3.5 if is_hl else 1.5,
                      dash="solid" if is_hl else "dot"),
            marker=dict(size=8 if is_hl else 4, color=color,
                        symbol="circle" if source == "setmil" else "diamond"),
            opacity=1.0 if is_hl else 0.45,
            hovertemplate=(
                f"<b>{TASK_LABELS[task]}</b> [{model}]<br>"
                "Day %{x}: %{y:.1%}<extra></extra>"
            ),
        ))

    # Event lines
    if out["tte_death"] and out["died"]:
        fig.add_vline(x=float(out["tte_death"]), line_color="#3fb950", line_width=2.5,
                      annotation_text="Death", annotation_font_color="#3fb950",
                      annotation_position="top right")
    if out["tte_clad"] and out["clad"]:
        fig.add_vline(x=float(out["tte_clad"]), line_color="#f78166", line_width=2,
                      line_dash="dash",
                      annotation_text="CLAD onset", annotation_font_color="#f78166",
                      annotation_position="top left")
    fig.add_hline(y=0.5, line_dash="dot", line_color="#888", line_width=1,
                  annotation_text="50%", annotation_font_size=10)

    fig.update_layout(
        **PLOTLY_THEME, height=400,
        xaxis_title="Days from transplant",
        yaxis_title="Predicted risk (0 = low, 1 = high)",
        yaxis=dict(range=[-0.05, 1.05], tickformat=".0%"),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1, font=dict(size=11)),
        hovermode="x unified",
        margin=dict(t=20, b=50),
    )
    return fig


# ── Render each case ──────────────────────────────────────────────────────────
for case in CASES:
    pid  = case["pid"]
    out  = _get_outcome(pid)
    df_lk_pt = patient_longi_preds(pid)
    n_vis = len(df_lk_pt) if not df_lk_pt.empty else "?"

    st.markdown(
        f"<h3 style='color:{TEXT}'>{case['icon']}  {case['title']}</h3>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Patient ID", pid), unsafe_allow_html=True)
    c2.markdown(metric_card("Outcome", case["outcome"]), unsafe_allow_html=True)
    c3.markdown(metric_card("LongMK biopsies", str(n_vis)), unsafe_allow_html=True)
    tte_str = f"{int(out['tte_death'])} days" if out.get("tte_death") else "censored"
    c4.markdown(metric_card("Follow-up / TTE", tte_str), unsafe_allow_html=True)

    st.info(case["story"])

    fig = _trajectory_fig(pid, case["highlight_task"], case["color_outcome"])
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No prediction data available for {pid}.")

    st.divider()

st.caption(
    "★ = highlighted/best task per case. ● = SetMIL-MT point; ◆ = LongMK point. "
    "Solid line = highlighted task; dotted = supporting tasks. "
    "All predictions are from held-out test folds (out-of-sample). "
    "n = 263 patients · 5-fold CV · Helmholtz Munich."
)
