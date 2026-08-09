"""Page 3 — Latent Space: LongMK biopsy UMAP (interactive) + SetMIL-MT instance UMAP (static)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Latent Space", page_icon="🗺", layout="wide")

from utils.styles import card_css, metric_card, BG, BG2, TEXT, MUTED, ACCENT, BORDER, CARD, PLOTLY_THEME
from utils.data_loader import load_biopsy_umap, patient_list

st.markdown(card_css(), unsafe_allow_html=True)
st.markdown(f"<h2 style='color:{TEXT}'>🗺 Latent Space</h2>", unsafe_allow_html=True)

INTERP_ROOT = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/interpretability/set_mil_mt_interp")

TASK_LABELS = {
    "acr_cls":   "ACR Classification",
    "acr_surv":  "ACR Survival",
    "clad_surv": "CLAD Survival",
    "death_surv":"Death Survival",
}
TASK_BEST = {
    "acr_cls":   "Best: SetMIL-MT (no SAB) · BACC 0.623",
    "acr_surv":  "Best: LongMK · C-index 0.679",
    "clad_surv": "Best: SetMIL-MT · C-index 0.563",
    "death_surv":"Best: LongMK · C-index 0.771",
}

tab_biopsy, tab_instance = st.tabs(["🔴 Biopsy Space (LongMK)", "🟣 Instance Space (SetMIL-MT)"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Interactive biopsy-level UMAP from LongMK representations
# ═══════════════════════════════════════════════════════════════════════════════
with tab_biopsy:
    st.markdown(
        f"<p style='color:{MUTED}'>Each point = one biopsy embedded in LongMK's 256-dim representation "
        f"space (UMAP projection). Select a patient to highlight their biopsies.</p>",
        unsafe_allow_html=True,
    )

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.5, 1, 1])
    with col_ctrl1:
        patients = patient_list()
        default_idx = patients.index(st.session_state.get("selected_patient", patients[0])) \
            if st.session_state.get("selected_patient") in patients else 0
        pid = st.selectbox("Patient", patients, index=default_idx, key="ls_pid")
        st.session_state["selected_patient"] = pid
    with col_ctrl2:
        task = st.selectbox("Task", list(TASK_LABELS.keys()),
                            format_func=lambda x: TASK_LABELS[x], key="ls_task")
    with col_ctrl3:
        color_by = st.selectbox("Color by", ["patient_highlighted", "risk", "event", "biopsy_days"],
                                format_func=lambda x: {
                                    "patient_highlighted": "Patient highlighted",
                                    "risk": "Predicted risk",
                                    "event": "Event status",
                                    "biopsy_days": "Days from transplant",
                                }[x], key="ls_color")

    df_all = load_biopsy_umap(task)

    if df_all.empty:
        st.warning(
            f"Biopsy UMAP not yet generated for **{task}**. "
            "Run: `sbatch patient_explorer/submit_export_biopsy_umap.sh`"
        )
    else:
        # Normalize risk: raw log-hazard → percentile rank [0,1]
        from scipy.stats import rankdata
        if "risk" in df_all.columns:
            r = df_all["risk"].values.astype(float)
            valid = ~np.isnan(r)
            pct = np.zeros_like(r)
            if valid.sum() > 1:
                pct[valid] = (rankdata(r[valid]) - 1) / (valid.sum() - 1)
            df_all = df_all.copy()
            df_all["risk"] = pct

        df_all = df_all.drop_duplicates(subset=["patient_id", "stem"])
        df_pt = df_all[df_all["patient_id"] == pid].copy()
        df_bg = df_all[df_all["patient_id"] != pid].copy()

        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("Total biopsies", str(len(df_all))), unsafe_allow_html=True)
        c2.markdown(metric_card("Patients", str(df_all["patient_id"].nunique())), unsafe_allow_html=True)
        c3.markdown(metric_card(f"{pid} biopsies", str(len(df_pt))), unsafe_allow_html=True)

        st.caption(TASK_BEST[task])

        fig = go.Figure()

        if color_by == "patient_highlighted":
            fig.add_trace(go.Scattergl(
                x=df_bg["umap_x"], y=df_bg["umap_y"], mode="markers",
                marker=dict(size=3, color="#3a3f5c", opacity=0.4),
                name="Other patients",
                hovertemplate="<b>%{customdata}</b><extra></extra>",
                customdata=df_bg["patient_id"],
            ))
            if not df_pt.empty:
                fig.add_trace(go.Scattergl(
                    x=df_pt["umap_x"], y=df_pt["umap_y"], mode="markers",
                    marker=dict(size=11, color=df_pt["risk"].values,
                                colorscale="RdYlGn_r", cmin=0, cmax=1,
                                colorbar=dict(title="Risk %ile", thickness=12, len=0.5),
                                line=dict(width=1.5, color="white")),
                    name=pid,
                    hovertemplate=(
                        f"<b>{pid}</b><br>Day %{{customdata[0]:.0f}}<br>"
                        "Risk: %{customdata[1]:.2f}<extra></extra>"
                    ),
                    customdata=np.stack([df_pt["biopsy_days"].values, df_pt["risk"].values], axis=1),
                ))

        elif color_by == "risk":
            fig.add_trace(go.Scattergl(
                x=df_all["umap_x"], y=df_all["umap_y"], mode="markers",
                marker=dict(size=4, color=df_all["risk"].values,
                            colorscale="RdYlGn_r", cmin=0, cmax=1,
                            colorbar=dict(title="Risk %ile", thickness=12, len=0.5), opacity=0.7),
                name="All biopsies",
                hovertemplate="<b>%{customdata[0]}</b><br>Risk: %{customdata[1]:.2f}<extra></extra>",
                customdata=np.stack([df_all["patient_id"].values, df_all["risk"].values], axis=1),
            ))
            if not df_pt.empty:
                fig.add_trace(go.Scattergl(
                    x=df_pt["umap_x"], y=df_pt["umap_y"], mode="markers",
                    marker=dict(size=13, color="white", symbol="star",
                                line=dict(width=1.5, color="#7c83ff")),
                    name=f"★ {pid}",
                ))

        elif color_by == "event":
            ev_colors = df_all["event"].fillna(0).astype(float).map(
                lambda v: "#e05c4b" if v == 1 else "#3fb950").tolist()
            fig.add_trace(go.Scattergl(
                x=df_all["umap_x"], y=df_all["umap_y"], mode="markers",
                marker=dict(size=4, color=ev_colors, opacity=0.65),
                name="All biopsies",
                hovertemplate="<b>%{customdata[0]}</b><br>Event: %{customdata[1]:.0f}<extra></extra>",
                customdata=np.stack([df_all["patient_id"].values,
                                     df_all["event"].fillna(0).values], axis=1),
            ))
            if not df_pt.empty:
                fig.add_trace(go.Scattergl(
                    x=df_pt["umap_x"], y=df_pt["umap_y"], mode="markers",
                    marker=dict(size=13, color="white", symbol="star",
                                line=dict(width=1.5, color="#7c83ff")),
                    name=f"★ {pid}",
                ))

        else:  # biopsy_days
            fig.add_trace(go.Scattergl(
                x=df_all["umap_x"], y=df_all["umap_y"], mode="markers",
                marker=dict(size=4, color=df_all["biopsy_days"].values,
                            colorscale="Viridis",
                            colorbar=dict(title="Days from Tx", thickness=12, len=0.5), opacity=0.65),
                name="All biopsies",
                hovertemplate="<b>%{customdata[0]}</b><br>Day: %{customdata[1]:.0f}<extra></extra>",
                customdata=np.stack([df_all["patient_id"].values,
                                     df_all["biopsy_days"].values], axis=1),
            ))
            if not df_pt.empty:
                fig.add_trace(go.Scattergl(
                    x=df_pt["umap_x"], y=df_pt["umap_y"], mode="markers",
                    marker=dict(size=13, color="white", symbol="star",
                                line=dict(width=1.5, color="#7c83ff")),
                    name=f"★ {pid}",
                ))

        fig.update_layout(
            **PLOTLY_THEME, height=580,
            xaxis=dict(showticklabels=False, title="UMAP 1"),
            yaxis=dict(showticklabels=False, title="UMAP 2"),
            legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
            margin=dict(l=40, r=40, t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Per-patient risk timeline
        if not df_pt.empty:
            st.markdown(f"<p class='section-title'>Biopsy Risk Timeline — {pid}</p>", unsafe_allow_html=True)
            df_pt_s = df_pt.sort_values("biopsy_days")
            fig_tl = go.Figure()
            fig_tl.add_trace(go.Scatter(
                x=df_pt_s["biopsy_days"], y=df_pt_s["risk"],
                mode="lines+markers",
                marker=dict(size=9, color=df_pt_s["risk"].values,
                            colorscale="RdYlGn_r", cmin=0, cmax=1,
                            colorbar=dict(title="Risk %ile", thickness=12, len=0.4),
                            line=dict(width=1, color="white")),
                line=dict(color=ACCENT, width=2),
                hovertemplate="Day %{x:.0f}<br>Risk %ile: %{y:.2f}<extra></extra>",
            ))
            fig_tl.add_hline(y=0.5, line_dash="dot", line_color=MUTED, line_width=1)
            fig_tl.update_layout(
                **PLOTLY_THEME, height=260,
                xaxis_title="Days from transplant",
                yaxis_title="Risk percentile (LongMK)",
                yaxis=dict(range=[-0.05, 1.05]),
            )
            st.plotly_chart(fig_tl, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Static SetMIL-MT instance-level UMAP (pre-rendered PNGs)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_instance:
    st.markdown(
        f"<p style='color:{MUTED}'>Instance-level UMAP from SetMIL-MT patch/cell representations, "
        f"aggregated across all 5 CV splits. Each point is a patch or cell coloured by task outcome.</p>",
        unsafe_allow_html=True,
    )

    TASK_DIRS = {
        "acr_cls":   INTERP_ROOT / "all_splits_cls",
        "acr_surv":  INTERP_ROOT / "all_splits_acr_surv",
        "clad_surv": INTERP_ROOT / "all_splits_clad_surv",
        "death_surv":INTERP_ROOT / "all_splits_death_surv",
    }

    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        task2 = st.selectbox("Task", list(TASK_LABELS.keys()),
                             format_func=lambda x: TASK_LABELS[x], key="ls_task2")
        umap_type = st.radio("UMAP metric", ["Cosine (recommended)", "Euclidean"],
                             label_visibility="collapsed")

    fname = "A_instance_reps_cosine.png" if "Cosine" in umap_type else "A_instance_reps.png"
    task_dir = TASK_DIRS.get(task2)

    if task_dir and task_dir.exists():
        p = task_dir / fname
        if p.exists():
            st.image(Image.open(p), use_container_width=True,
                     caption=f"SetMIL-MT instance UMAP — {TASK_LABELS[task2]} ({umap_type})")
        else:
            alt = task_dir / "A_instance_reps.png"
            if alt.exists():
                st.image(Image.open(alt), use_container_width=True,
                         caption=f"SetMIL-MT instance UMAP — {TASK_LABELS[task2]} (Euclidean)")
            else:
                st.info(f"No UMAP figure found in {task_dir}")
    else:
        st.info(f"Directory not found: {task_dir}")
