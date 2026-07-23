"""Page 12 — SetMIL-MT Interpretability: gate weights, seed importance, panel figures."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Interpretability", page_icon="🔬", layout="wide")

from utils.styles import (
    card_css, metric_card,
    BG, BG2, TEXT, MUTED, ACCENT, BORDER, CARD, PLOTLY_THEME, MOD_COLORS,
)
from utils.data_loader import load_paper_interp, PANEL_FIG_DIR

st.markdown(card_css(), unsafe_allow_html=True)

TASK_LABELS = {
    "acr_cls":   "ACR Classification",
    "acr_surv":  "ACR Survival",
    "clad_surv": "CLAD Survival",
    "death_surv":"Death Survival",
}
TASK_COLORS = {
    "acr_cls":   "#7c83ff",
    "acr_surv":  "#58a6ff",
    "clad_surv": "#f78166",
    "death_surv":"#ff7b72",
}
MOD_ORDER = ["HE", "BAL", "CT", "Clinical"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:{TEXT}'>🔬 Interpretability</h3>", unsafe_allow_html=True)
    st.divider()
    task = st.selectbox("Task", list(TASK_LABELS.keys()),
                        format_func=lambda x: TASK_LABELS[x])
    st.divider()
    show_gates  = st.checkbox("Gate weights", value=True)
    show_seeds  = st.checkbox("Seed importance (Δα)", value=True)
    show_panels = st.checkbox("Panel figures", value=True)

st.markdown(f"<h2 style='color:{TEXT}'>🔬 SetMIL-MT Interpretability</h2>", unsafe_allow_html=True)

interp = load_paper_interp()
if not interp:
    st.warning("paper_interp_data.json not found.")
    st.stop()

task_data = interp.get("tasks", {}).get(task, {})
n_patients = task_data.get("n_patients", interp.get("n_patients", 0))
n_hi = task_data.get("n_hi", 0)
n_lo = task_data.get("n_lo", 0)

# ── Summary cards ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown(metric_card("Patients", str(n_patients)), unsafe_allow_html=True)
c2.markdown(metric_card("High risk", str(n_hi)), unsafe_allow_html=True)
c3.markdown(metric_card("Low risk", str(n_lo)), unsafe_allow_html=True)
c4.markdown(metric_card("Task", TASK_LABELS[task]), unsafe_allow_html=True)

# ── Gate weights ──────────────────────────────────────────────────────────────
if show_gates:
    st.divider()
    st.markdown(f"<p class='section-title'>Modality Gate Weights — {TASK_LABELS[task]}</p>",
                unsafe_allow_html=True)
    st.caption("Mean gate weight (σ) per modality across all patients. Higher = modality more used.")

    gw = task_data.get("gate_weights", {})
    means = gw.get("mean", {})
    stds  = gw.get("std", {})

    if means:
        mods   = [m for m in MOD_ORDER if m in means]
        y_vals = [means[m] for m in mods]
        e_vals = [stds.get(m, 0) for m in mods]
        colors = [MOD_COLORS.get(m, MUTED) for m in mods]

        fig_gw = go.Figure(go.Bar(
            x=y_vals, y=mods,
            orientation="h",
            marker_color=colors,
            error_x=dict(type="data", array=e_vals, visible=True,
                         color=TEXT, thickness=1.5, width=4),
            text=[f"{v:.3f}" for v in y_vals],
            textposition="outside",
            textfont=dict(color=TEXT),
            hovertemplate="<b>%{y}</b><br>Gate weight: %{x:.3f}<extra></extra>",
        ))
        fig_gw.update_layout(
            **PLOTLY_THEME, height=280,
            xaxis_title="Mean gate weight",
            xaxis=dict(range=[0, max(y_vals) * 1.25]),
            yaxis=dict(categoryorder="array", categoryarray=list(reversed(mods))),
        )
        st.plotly_chart(fig_gw, use_container_width=True)
    else:
        st.info("Gate weight data not available for this task.")

# ── Seed importance (Δα) ──────────────────────────────────────────────────────
if show_seeds:
    st.divider()
    st.markdown(f"<p class='section-title'>Seed Importance (Δα) — {TASK_LABELS[task]}</p>",
                unsafe_allow_html=True)
    st.caption(
        "Δα = α − 1/N_seeds. Positive seeds attract more attention than chance; "
        "negative seeds are suppressed. Shows difference between high-risk and low-risk patients."
    )

    sda = task_data.get("seed_delta_alpha", {})
    mods_with_data = [m for m in MOD_ORDER if m in sda and sda[m]]

    if mods_with_data:
        tabs = st.tabs(mods_with_data)
        for tab, mod in zip(tabs, mods_with_data):
            with tab:
                vals = sda[mod]
                if not vals:
                    st.info(f"No seed data for {mod}.")
                    continue
                n_seeds = len(vals)
                seed_idx = list(range(1, n_seeds + 1))
                colors = [MOD_COLORS.get(mod, ACCENT) if v >= 0 else MUTED for v in vals]
                fig_sa = go.Figure(go.Bar(
                    x=seed_idx, y=vals,
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in vals],
                    textposition="outside",
                    textfont=dict(color=TEXT, size=9),
                    hovertemplate="Seed %{x}: Δα=%{y:.4f}<extra></extra>",
                ))
                fig_sa.add_hline(y=0, line_color=TEXT, line_width=1, line_dash="dot")
                yabs = max(abs(v) for v in vals) if vals else 0.1
                fig_sa.update_layout(
                    **PLOTLY_THEME, height=300,
                    xaxis_title="Seed index",
                    xaxis=dict(tickvals=seed_idx, ticktext=[str(i) for i in seed_idx]),
                    yaxis_title="Δα",
                    yaxis=dict(range=[-(yabs * 1.3), yabs * 1.3]),
                    title=f"{mod} — {n_seeds} seeds",
                    title_font=dict(color=MOD_COLORS.get(mod, TEXT)),
                )
                st.plotly_chart(fig_sa, use_container_width=True)

        # Summary: mean |Δα| per modality
        st.markdown(f"<p class='section-title'>Mean |Δα| per modality (seed importance summary)</p>",
                    unsafe_allow_html=True)
        summary_mods, summary_vals = [], []
        for m in MOD_ORDER:
            if m in sda and sda[m]:
                summary_mods.append(m)
                summary_vals.append(float(np.mean(np.abs(sda[m]))))

        if summary_mods:
            fig_sum = go.Figure(go.Bar(
                x=summary_mods, y=summary_vals,
                marker_color=[MOD_COLORS.get(m, MUTED) for m in summary_mods],
                text=[f"{v:.4f}" for v in summary_vals],
                textposition="outside",
                textfont=dict(color=TEXT),
                hovertemplate="<b>%{x}</b><br>Mean |Δα|: %{y:.4f}<extra></extra>",
            ))
            fig_sum.update_layout(
                **PLOTLY_THEME, height=260,
                xaxis_title="Modality",
                yaxis_title="Mean |Δα|",
            )
            st.plotly_chart(fig_sum, use_container_width=True)
    else:
        st.info("Seed importance data not available.")

# ── Panel figures ─────────────────────────────────────────────────────────────
if show_panels:
    st.divider()
    st.markdown(f"<p class='section-title'>Pre-computed Panel Figures</p>", unsafe_allow_html=True)

    # Panel C — cross-modal attention
    st.markdown("**Panel C — Cross-modal SAB attention**")
    for fname in ["C_sab_crossmodal_attn.png", "C_sab_significant.png"]:
        p = PANEL_FIG_DIR / fname
        if p.exists():
            st.image(Image.open(p), use_container_width=True, caption=fname.replace(".png","").replace("_"," "))

    st.divider()

    # Panel D — seed importance bar
    st.markdown("**Panel D — ABMIL seed importance (Δα)**")
    p_d = PANEL_FIG_DIR / "D_abmil_seed_importance.png"
    if p_d.exists():
        st.image(Image.open(p_d), use_container_width=True, caption="Seed importance Δα")
    else:
        st.info("Panel D PNG not found.")

    st.divider()

    # Panel G — representation hexbins per task
    st.markdown("**Panel G — Patient representation space (hexbin)**")
    g_tasks = ["acr_cls", "acr_surv", "clad_surv", "death_surv"]
    g_cols = st.columns(2)
    for i, t in enumerate(g_tasks):
        fname = f"G_final_rep_hexbin_{t}.png"
        p_g = PANEL_FIG_DIR / fname
        with g_cols[i % 2]:
            if p_g.exists():
                st.image(Image.open(p_g), use_container_width=True,
                         caption=TASK_LABELS.get(t, t))
            else:
                st.info(f"{fname} not found.")
