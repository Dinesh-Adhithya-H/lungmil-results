"""Page 10 — Biopsy Representation Space: per-patient UMAP from LongMK biopsy embeddings."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Biopsy Space", page_icon="🔬", layout="wide")

from utils.styles import card_css, metric_card, BG, BG2, TEXT, MUTED, ACCENT, BORDER, CARD, PLOTLY_THEME
from utils.data_loader import load_biopsy_umap, patient_list, patient_biopsy_umap

st.markdown(card_css(), unsafe_allow_html=True)

TASK_LABELS = {
    "acr_cls":   "ACR Classification (LongMK reps)",
    "acr_surv":  "ACR Survival (LongMK reps)",
    "clad_surv": "CLAD Survival (LongMK reps)",
    "death_surv":"Death Survival (LongMK reps)",
}
TASK_METRIC = {
    "acr_cls":   "Best: SetMIL-MT (no SAB) · BACC 0.623",
    "acr_surv":  "Best: LongMK · C-index 0.679",
    "clad_surv": "Best: SetMIL-MT · C-index 0.563",
    "death_surv":"Best: LongMK · C-index 0.771",
}
COLOR_BY_OPTS = ["risk", "event", "patient_highlighted", "biopsy_days"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:{TEXT}'>🔬 Biopsy Space</h3>", unsafe_allow_html=True)
    patients = patient_list()
    default_idx = patients.index(st.session_state.get("selected_patient", patients[0])) \
        if st.session_state.get("selected_patient") in patients else 0
    pid = st.selectbox("Patient", patients, index=default_idx)
    st.session_state["selected_patient"] = pid
    st.divider()
    task = st.selectbox("Task", list(TASK_LABELS.keys()), format_func=lambda x: TASK_LABELS[x])
    color_by = st.selectbox(
        "Color by",
        COLOR_BY_OPTS,
        format_func=lambda x: {
            "risk": "Predicted risk",
            "event": "Event status (0/1)",
            "patient_highlighted": "Patient highlighted",
            "biopsy_days": "Days from transplant",
        }[x]
    )
    min_splits = st.slider("Min splits to include biopsy", 1, 5, 1,
                           help="Only show biopsies present in ≥ N splits")

st.markdown(
    f"<h2 style='color:{TEXT}'>🔬 Biopsy Representation Space — {pid}</h2>",
    unsafe_allow_html=True,
)
st.caption(
    "Each point = one biopsy, embedded in LongMK's 256-dim representation space (UMAP projection). "
    "Biopsy reps are extracted from the best-performing longitudinal multimodal model."
)

df_all = load_biopsy_umap(task)

if df_all.empty:
    st.warning(
        f"Biopsy UMAP data not yet generated for **{task}**. "
        "Submit the SLURM export job first:\n\n"
        "```bash\nsbatch patient_explorer/submit_export_biopsy_umap.sh\n```"
    )
    st.stop()

# Filter by min_splits: keep biopsies seen in ≥ min_splits splits
if "split" in df_all.columns and min_splits > 1:
    counts = df_all.groupby(["patient_id", "stem"])["split"].nunique().reset_index(name="n_splits")
    df_all = df_all.merge(counts, on=["patient_id", "stem"])
    df_all = df_all[df_all["n_splits"] >= min_splits].drop_duplicates(subset=["patient_id", "stem"])
else:
    df_all = df_all.drop_duplicates(subset=["patient_id", "stem"])

df_pt = df_all[df_all["patient_id"] == pid].copy()

# Metric cards
n_total    = len(df_all)
n_pts      = df_all["patient_id"].nunique()
n_pt_bio   = len(df_pt)
c1, c2, c3, c4 = st.columns(4)
c1.markdown(metric_card("Total biopsies", str(n_total)), unsafe_allow_html=True)
c2.markdown(metric_card("Patients", str(n_pts)), unsafe_allow_html=True)
c3.markdown(metric_card(f"{pid} biopsies", str(n_pt_bio)), unsafe_allow_html=True)
c4.markdown(metric_card("Task", task.replace("_", " ")), unsafe_allow_html=True)

st.divider()

# ── Main UMAP plot ────────────────────────────────────────────────────────────
st.markdown(f"<p class='section-title'>Global Biopsy UMAP — {TASK_LABELS[task]}</p>",
            unsafe_allow_html=True)
st.caption(TASK_METRIC[task])

df_bg = df_all[df_all["patient_id"] != pid].copy()

fig = go.Figure()

if color_by == "patient_highlighted":
    # Background: all other patients in grey
    fig.add_trace(go.Scattergl(
        x=df_bg["umap_x"], y=df_bg["umap_y"],
        mode="markers",
        marker=dict(size=3, color="#3a3f5c", opacity=0.4),
        name="Other patients",
        hovertemplate="<b>Other patient</b><br>%{customdata}<extra></extra>",
        customdata=df_bg["patient_id"],
    ))
    # Foreground: patient's biopsies, coloured by risk
    if not df_pt.empty:
        risk_vals = df_pt["risk"].values.astype(float)
        fig.add_trace(go.Scattergl(
            x=df_pt["umap_x"], y=df_pt["umap_y"],
            mode="markers",
            marker=dict(
                size=11,
                color=risk_vals,
                colorscale="RdYlGn_r",
                cmin=0, cmax=1,
                colorbar=dict(title="Risk", thickness=12, len=0.5),
                line=dict(width=1.5, color="white"),
            ),
            name=pid,
            hovertemplate=(
                f"<b>{pid}</b><br>"
                "Day %{customdata[0]:.0f}<br>"
                "Risk: %{customdata[1]:.3f}<br>"
                "Event: %{customdata[2]}<extra></extra>"
            ),
            customdata=np.stack([
                df_pt["biopsy_days"].values,
                df_pt["risk"].values,
                df_pt["event"].values,
            ], axis=1),
        ))

elif color_by == "risk":
    risk_vals = df_all["risk"].values.astype(float)
    fig.add_trace(go.Scattergl(
        x=df_all["umap_x"], y=df_all["umap_y"],
        mode="markers",
        marker=dict(
            size=4,
            color=risk_vals,
            colorscale="RdYlGn_r",
            cmin=0, cmax=1,
            colorbar=dict(title="Risk", thickness=12, len=0.5),
            opacity=0.7,
        ),
        name="All biopsies",
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Risk: %{customdata[1]:.3f}<br>"
            "Day: %{customdata[2]:.0f}<extra></extra>"
        ),
        customdata=np.stack([
            df_all["patient_id"].values,
            df_all["risk"].values,
            df_all["biopsy_days"].values,
        ], axis=1),
    ))
    # Highlight selected patient
    if not df_pt.empty:
        fig.add_trace(go.Scattergl(
            x=df_pt["umap_x"], y=df_pt["umap_y"],
            mode="markers",
            marker=dict(size=12, color="white", symbol="star",
                        line=dict(width=1.5, color="#7c83ff")),
            name=f"★ {pid}",
            hovertemplate=f"<b>{pid}</b><br>Day %{{customdata}}<extra></extra>",
            customdata=df_pt["biopsy_days"].values,
        ))

elif color_by == "event":
    event_colors = df_all["event"].fillna(0).astype(float).map(
        lambda v: "#e05c4b" if v == 1 else "#3fb950"
    ).tolist()
    fig.add_trace(go.Scattergl(
        x=df_all["umap_x"], y=df_all["umap_y"],
        mode="markers",
        marker=dict(size=4, color=event_colors, opacity=0.65),
        name="All biopsies",
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Event: %{customdata[1]:.0f}<extra></extra>"
        ),
        customdata=np.stack([
            df_all["patient_id"].values,
            df_all["event"].fillna(0).values,
        ], axis=1),
    ))
    if not df_pt.empty:
        fig.add_trace(go.Scattergl(
            x=df_pt["umap_x"], y=df_pt["umap_y"],
            mode="markers",
            marker=dict(size=12, color="white", symbol="star",
                        line=dict(width=1.5, color="#7c83ff")),
            name=f"★ {pid}",
        ))

else:  # biopsy_days
    day_vals = df_all["biopsy_days"].values.astype(float)
    fig.add_trace(go.Scattergl(
        x=df_all["umap_x"], y=df_all["umap_y"],
        mode="markers",
        marker=dict(
            size=4,
            color=day_vals,
            colorscale="Viridis",
            colorbar=dict(title="Days from Tx", thickness=12, len=0.5),
            opacity=0.65,
        ),
        name="All biopsies",
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Day: %{customdata[1]:.0f}<extra></extra>"
        ),
        customdata=np.stack([
            df_all["patient_id"].values,
            df_all["biopsy_days"].values,
        ], axis=1),
    ))
    if not df_pt.empty:
        fig.add_trace(go.Scattergl(
            x=df_pt["umap_x"], y=df_pt["umap_y"],
            mode="markers",
            marker=dict(size=12, color="white", symbol="star",
                        line=dict(width=1.5, color="#7c83ff")),
            name=f"★ {pid}",
        ))

fig.update_layout(
    **PLOTLY_THEME, height=560,
    xaxis=dict(showticklabels=False, title="UMAP 1"),
    yaxis=dict(showticklabels=False, title="UMAP 2"),
    legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
    margin=dict(l=40, r=40, t=40, b=40),
)
st.plotly_chart(fig, use_container_width=True)

# ── Per-patient biopsy timeline ───────────────────────────────────────────────
if not df_pt.empty:
    st.divider()
    st.markdown(f"<p class='section-title'>Biopsy Risk Timeline — {pid}</p>", unsafe_allow_html=True)

    df_pt_sorted = df_pt.sort_values("biopsy_days")
    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(
        x=df_pt_sorted["biopsy_days"],
        y=df_pt_sorted["risk"],
        mode="lines+markers",
        marker=dict(
            size=9,
            color=df_pt_sorted["risk"].values,
            colorscale="RdYlGn_r",
            cmin=0, cmax=1,
            colorbar=dict(title="Risk", thickness=12, len=0.4),
            line=dict(width=1, color="white"),
        ),
        line=dict(color=ACCENT, width=2),
        hovertemplate="Day %{x:.0f}<br>Risk: %{y:.3f}<extra></extra>",
    ))
    fig_tl.add_hline(y=0.5, line_dash="dot", line_color=MUTED, line_width=1,
                     annotation_text="0.5", annotation_font_color=MUTED)
    fig_tl.update_layout(
        **PLOTLY_THEME, height=280,
        xaxis_title="Days from transplant",
        yaxis_title="Predicted risk (LongMK)",
        yaxis=dict(range=[-0.05, 1.05]),
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    # Biopsy table
    with st.expander("Biopsy details table"):
        show_cols = [c for c in ["biopsy_days", "risk", "event", "tte", "label", "split"]
                     if c in df_pt_sorted.columns]
        st.dataframe(
            df_pt_sorted[show_cols].rename(columns={
                "biopsy_days": "Days from Tx",
                "risk": "Predicted risk",
                "event": "Event (0/1)",
                "tte": "TTE (days)",
                "label": "Label",
                "split": "CV split",
            }).reset_index(drop=True),
            use_container_width=True,
        )
