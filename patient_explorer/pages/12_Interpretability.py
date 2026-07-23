"""Page 12 — Interpretability: SetMIL-MT + LongitudinalSetMIL panel figures."""

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

# Map task → per-task figure directory and filename suffixes
INTERP_ROOT = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/interpretability/set_mil_mt_interp")
LONGI_INTERP_ROOT = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/interpretability/longitudinal_mk_interp")
TASK_DIR = {
    "acr_cls":   INTERP_ROOT / "all_splits_cls",
    "acr_surv":  INTERP_ROOT / "all_splits_acr_surv",
    "clad_surv": INTERP_ROOT / "all_splits_clad_surv",
    "death_surv": INTERP_ROOT / "all_splits_death_surv",
}
G_FNAME = {
    "acr_cls":   "G_final_rep_hexbin_acr_cls.png",
    "acr_surv":  "G_final_rep_hexbin_acr_surv.png",
    "clad_surv": "G_final_rep_hexbin_clad.png",
    "death_surv":"G_final_rep_hexbin_death.png",
}
K_FNAME = {
    "acr_cls":   "K_multisplit_seed_attribution_acr_cls.png",
    "acr_surv":  "K_multisplit_seed_attribution_acr_surv.png",
    "clad_surv": "K_multisplit_seed_attribution_clad.png",
    "death_surv":"K_multisplit_seed_attribution_death.png",
}


def show_panel(path: Path, caption: str):
    if path.exists():
        st.image(Image.open(path), caption=caption)
    else:
        st.caption(f"_{caption} not yet generated._")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:{TEXT}'>🔬 Interpretability</h3>", unsafe_allow_html=True)
    st.divider()
    model_sel = st.radio("Model", ["SetMIL-MT", "LongitudinalSetMIL"], index=0)
    st.divider()
    if model_sel == "SetMIL-MT":
        task = st.selectbox("Task", list(TASK_LABELS.keys()),
                            format_func=lambda x: TASK_LABELS[x])
        tdir = TASK_DIR[task]
        sections = st.multiselect(
            "Sections",
            ["Gate weights", "UMAP reps", "Seed stratification",
             "Cluster affinity", "Co-activation", "Representation space",
             "Cross-split attribution"],
            default=["Gate weights", "Seed stratification", "Representation space",
                     "Cross-split attribution"],
        )

if model_sel == "SetMIL-MT":
    st.markdown(f"<h2 style='color:{TASK_COLORS[task]}'>🔬 {TASK_LABELS[task]} — SetMIL-MT</h2>",
                unsafe_allow_html=True)

if model_sel == "SetMIL-MT":
    interp = load_paper_interp()
    task_data = interp.get("tasks", {}).get(task, {}) if interp else {}
    n_hi = task_data.get("n_hi", "?")
    n_lo = task_data.get("n_lo", "?")
    n_pts = task_data.get("n_patients", interp.get("n_patients", "?") if interp else "?")

    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card("Patients", str(n_pts)), unsafe_allow_html=True)
    c2.markdown(metric_card("High risk", str(n_hi)), unsafe_allow_html=True)
    c3.markdown(metric_card("Low risk",  str(n_lo)), unsafe_allow_html=True)

    # ── E: Gate weights ───────────────────────────────────────────────────────────
    if "Gate weights" in sections:
        st.divider()
        st.markdown(f"<p class='section-title'>Modality Gate Weights</p>", unsafe_allow_html=True)
        st.caption("Mean σ(gate) per modality across all patients. Scores are close but HE consistently leads.")

        col_fig, col_img = st.columns([1, 1])
        with col_fig:
            gw = task_data.get("gate_weights", {})
            means = gw.get("mean", {})
            stds  = gw.get("std", {})
            if means:
                mods   = [m for m in MOD_ORDER if m in means]
                y_vals = [means[m] for m in mods]
                e_vals = [stds.get(m, 0) for m in mods]
                colors = [MOD_COLORS.get(m, MUTED) for m in mods]
                fig_gw = go.Figure(go.Bar(
                    x=y_vals, y=mods, orientation="h",
                    marker_color=colors,
                    error_x=dict(type="data", array=e_vals, visible=True,
                                 color=TEXT, thickness=1.5, width=5),
                    text=[f"{v:.3f} ± {e:.3f}" for v, e in zip(y_vals, e_vals)],
                    textposition="outside", textfont=dict(color=TEXT, size=11),
                    hovertemplate="<b>%{y}</b><br>%{x:.3f}<extra></extra>",
                ))
                fig_gw.update_layout(
                    **PLOTLY_THEME, height=260,
                    xaxis_title="Mean gate weight",
                    xaxis=dict(range=[0, max(y_vals) * 1.3]),
                    yaxis=dict(categoryorder="array", categoryarray=list(reversed(mods))),
                    margin=dict(l=10, r=80, t=20, b=40),
                )
                st.plotly_chart(fig_gw)
            else:
                st.info("Gate weight data not available.")

        with col_img:
            show_panel(tdir / "E_task_modal_gate.png", "Panel E — TaskModalGate heatmap")

    # ── A: UMAP instance reps ──────────────────────────────────────────────────────
    if "UMAP reps" in sections:
        st.divider()
        st.markdown(f"<p class='section-title'>Instance Representations (UMAP)</p>", unsafe_allow_html=True)
        st.caption("UMAP of per-modality instance embeddings post-ModalFFNEncoder. Top: by ACR label. Bottom: by pre-computed cluster.")
        show_panel(tdir / "A_instance_reps_cosine.png", "Panel A — UMAP (cosine metric)")

    # ── I: Seed risk stratification ────────────────────────────────────────────────
    if "Seed stratification" in sections:
        st.divider()
        st.markdown(f"<p class='section-title'>Seed Risk Stratification (Δα)</p>", unsafe_allow_html=True)
        st.caption("Top discriminative seeds: ACR+ vs ACR−. Red seeds are higher in high-risk patients. Right panel: all 64 seeds ranked by Δα.")

        col_int, col_img = st.columns([1, 1])
        with col_int:
            sda = task_data.get("seed_delta_alpha", {})
            mods_avail = [m for m in MOD_ORDER if m in sda and sda[m]]
            if mods_avail:
                tabs = st.tabs(mods_avail)
                for tab, mod in zip(tabs, mods_avail):
                    with tab:
                        vals = np.array(sda[mod])
                        n_seeds = len(vals)
                        colors = [MOD_COLORS.get(mod, ACCENT) if v >= 0 else "#e05c4b" for v in vals]
                        fig_sa = go.Figure(go.Bar(
                            x=list(range(1, n_seeds + 1)), y=vals,
                            marker_color=colors,
                            hovertemplate="Seed %{x}: Δα=%{y:.4f}<extra></extra>",
                        ))
                        fig_sa.add_hline(y=0, line_color=TEXT, line_width=1, line_dash="dot")
                        yabs = max(abs(vals.max()), abs(vals.min()), 0.001)
                        fig_sa.update_layout(
                            **PLOTLY_THEME, height=280,
                            xaxis_title="Seed", yaxis_title="Δα",
                            yaxis=dict(range=[-(yabs * 1.35), yabs * 1.35]),
                            margin=dict(t=10, b=40),
                        )
                        st.plotly_chart(fig_sa)
            else:
                st.info("Seed Δα data not available.")

        with col_img:
            if task == "acr_cls":
                show_panel(tdir / "I_seed_risk_stratification.png", "Panel I — Seed risk stratification")
            else:
                st.caption("_Panel I only generated for ACR Classification._")

    # ── H: Cluster affinity ────────────────────────────────────────────────────────
    if "Cluster affinity" in sections:
        st.divider()
        st.markdown(f"<p class='section-title'>Seed → Cluster Affinity (Panel H)</p>", unsafe_allow_html=True)
        st.caption("Each seed's cosine-mass distribution over biological clusters, per modality. Reveals which tissue biology each seed specialises in.")
        if task == "acr_cls":
            show_panel(tdir / "H_seed_cluster_permod.png", "Panel H — Seed cluster affinity per modality")
            show_panel(tdir / "H_information_pathway.png", "Panel H — Information pathway (all modalities combined)")
        else:
            st.caption("_Panel H only generated for ACR Classification._")

    # ── J: Seed co-activation ──────────────────────────────────────────────────────
    if "Co-activation" in sections:
        st.divider()
        st.markdown(f"<p class='section-title'>Seed Co-activation Correlation (Panel J)</p>", unsafe_allow_html=True)
        st.caption("Pearson r of ABMIL α across patients. Dark red = seeds co-activate together; dark blue = competitive. Seeds reordered by hierarchical clustering.")
        show_panel(tdir / "J_seed_coactivation_corr.png", "Panel J — Seed co-activation")

    # ── G: Representation hexbin ───────────────────────────────────────────────────
    if "Representation space" in sections:
        st.divider()
        st.markdown(f"<p class='section-title'>Patient Representation Space (Panel G)</p>", unsafe_allow_html=True)
        st.caption("Hexbin UMAP of final patient representations. Coloured by ACR label, P(ACR+), TTE, modality combo, risk vs TTE, KM stratification, and CV split.")
        show_panel(PANEL_FIG_DIR / G_FNAME[task], f"Panel G — {TASK_LABELS[task]}")

    # ── K: Cross-split attribution ─────────────────────────────────────────────────
    if "Cross-split attribution" in sections:
        st.divider()
        st.markdown(f"<p class='section-title'>Cross-Split Seed Attribution (Panel K)</p>", unsafe_allow_html=True)
        st.caption("Per-split breakdown: mean ABMIL α for ACR− (left) and ACR+ (centre), and Δα (right). Confirms which seeds are consistently discriminative across all 5 CV splits.")
        show_panel(tdir / K_FNAME[task], f"Panel K — {TASK_LABELS[task]}")

else:
    # ── LongitudinalSetMIL interpretability ───────────────────────────────────
    st.markdown(f"<h2 style='color:#3fb950'>🔬 LongitudinalSetMIL — Interpretability</h2>",
                unsafe_allow_html=True)

    # Find available dirs: split*_fold0 (legacy no-suffix) + split*_fold0_{task}
    _LONGI_TASKS = ["acr_cls", "acr_surv", "clad_surv", "death_surv"]
    _LONGI_TASK_LABELS = {
        "acr_cls": "ACR Classification",
        "acr_surv": "ACR Survival",
        "clad_surv": "CLAD Survival",
        "death_surv": "Death Survival",
    }

    # Build available (split, task) combos
    _avail = {}  # task -> list of split_dir paths
    for _task in _LONGI_TASKS:
        _avail[_task] = []
        for _s in range(5):
            _d = LONGI_INTERP_ROOT / f"split{_s}_fold0_{_task}"
            if _d.exists():
                _avail[_task].append(_d)
        # legacy no-suffix dir counts for any task
        _legacy = LONGI_INTERP_ROOT / "split0_fold0"
        if _legacy.exists() and not _avail[_task]:
            _avail[_task].append(_legacy)

    _tasks_avail = [t for t in _LONGI_TASKS if _avail[t]]
    _total_dirs = sum(len(v) for v in _avail.values())

    # Sidebar selectors
    with st.sidebar:
        if _tasks_avail:
            longi_task_sel = st.selectbox(
                "Task", _tasks_avail,
                format_func=lambda x: _LONGI_TASK_LABELS[x]
            )
            _split_dirs_for_task = _avail[longi_task_sel]
            _split_opts = [d.name for d in _split_dirs_for_task]
            longi_split_label = st.selectbox("Split", _split_opts, index=0)
            longi_split_dir = LONGI_INTERP_ROOT / longi_split_label
        else:
            longi_task_sel = "acr_cls"
            longi_split_dir = LONGI_INTERP_ROOT / "split0_fold0"
            st.caption("No longitudinal interp dirs found yet.")

    _n_task_splits = {t: len(v) for t, v in _avail.items()}
    st.caption(
        f"Available: " +
        ", ".join(f"{_LONGI_TASK_LABELS[t]}: {_n_task_splits[t]}/5" for t in _LONGI_TASKS) +
        ". Per-patient panels (L0, L1) on the **Longitudinal** page."
    )

    st.divider()
    st.markdown(f"<p class='section-title'>Model Properties (L_global) — {longi_split_dir.name}</p>", unsafe_allow_html=True)
    st.caption("ALiBi head slopes, per-task recency γ, and temporal decay curves.")
    show_panel(longi_split_dir / "L_global_model_properties.png", "L_global — Model properties")

    st.divider()
    st.markdown(f"<p class='section-title'>Population: Seed Trends (Lpop) — {longi_split_dir.name}</p>", unsafe_allow_html=True)
    st.caption("PMA seed activations across biopsy positions at population level.")
    show_panel(longi_split_dir / "Lpop_seed_trends.png", "Lpop — Seed trends across biopsies")

    st.divider()
    st.markdown(f"<p class='section-title'>Population: Recency ABMIL α (Lpop) — {longi_split_dir.name}</p>", unsafe_allow_html=True)
    st.caption("Which biopsy position dominates per task — recency vs history bias.")
    show_panel(longi_split_dir / "Lpop_alpha_recency.png", "Lpop — Recency α distribution")

    if _total_dirs < 20:
        missing = 20 - _total_dirs
        st.divider()
        st.info(f"{_total_dirs}/20 task×split combinations done ({missing} pending). Per-patient panels accumulate as jobs finish.")
