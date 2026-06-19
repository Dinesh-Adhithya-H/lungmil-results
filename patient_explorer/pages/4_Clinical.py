"""Page 4 — Clinical Features & Model Attention."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Clinical Features", page_icon="📈", layout="wide")

from utils.styles import card_css, BG, BG2, TEXT, MUTED, ACCENT, BORDER
from utils.data_loader import (
    load_clinical, load_feature_names, patient_splits, patient_predictions,
)
from utils.plots import clinical_trend_figure

st.markdown(card_css(), unsafe_allow_html=True)

pid = st.session_state.get("selected_patient")
if not pid:
    st.warning("Select a patient from the home page.")
    st.stop()

st.markdown(f"<h2 style='color:{TEXT}'>📈 Clinical Features — {pid}</h2>", unsafe_allow_html=True)

clin_df   = load_clinical()
feat_names = load_feature_names()
splits    = patient_splits(pid)
preds     = patient_predictions(pid)


# ── Clinical time series ──────────────────────────────────────────────────
st.markdown(f"<p class='section-title'>Longitudinal Clinical Features</p>",
            unsafe_allow_html=True)

if clin_df.empty:
    st.info("Clinical feature data not yet available. Run `export_data.py` to generate it.")
else:
    # filter to this patient
    if "patient_id" in clin_df.columns:
        pt_clin = clin_df[clin_df["patient_id"] == pid].copy()
    elif "record_id" in clin_df.columns:
        pt_clin = clin_df[clin_df["record_id"] == pid].copy()
    else:
        pt_clin = pd.DataFrame()

    if pt_clin.empty:
        st.info(f"No clinical feature data for {pid}.")
    else:
        feat_cols = sorted([c for c in pt_clin.columns if c.startswith("feat_")],
                           key=lambda x: int(x.split("_")[1]))
        if feat_cols:
            # feature selector
            n_avail = len(feat_cols)
            display_names = {
                c: feat_names.get(int(c.split("_")[1]), c)
                for c in feat_cols
            }
            rev_map = {v: k for k, v in display_names.items()}

            # top features by variance
            variances = pt_clin[feat_cols].var().sort_values(ascending=False)
            top_feats = variances.head(12).index.tolist()
            default_names = [display_names[c] for c in top_feats]

            sel_names = st.multiselect(
                "Features (top by variance pre-selected)",
                options=list(display_names.values()),
                default=default_names[:8],
                max_selections=16,
            )
            sel_cols = [rev_map[n] for n in sel_names if n in rev_map]

            if sel_cols:
                date_col = "anchor_dt" if "anchor_dt" in pt_clin.columns else \
                           "spiro_date" if "spiro_date" in pt_clin.columns else None

                fig = go.Figure()
                palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
                for i, col in enumerate(sel_cols):
                    name = display_names[col]
                    x = pt_clin[date_col] if date_col else pt_clin.index
                    fig.add_trace(go.Scatter(
                        x=x, y=pt_clin[col],
                        mode="lines+markers", name=name,
                        line=dict(color=palette[i % len(palette)], width=2),
                        marker=dict(size=7),
                        hovertemplate=f"<b>{name}</b>: %{{y:.3f}}<br>%{{x}}<extra></extra>",
                    ))

                fig.update_layout(
                    template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG2,
                    height=380, margin=dict(l=20, r=20, t=20, b=40),
                    xaxis=dict(gridcolor=BORDER),
                    yaxis=dict(title="Normalised value", gridcolor=BORDER),
                    legend=dict(orientation="h", x=0, y=-0.22, font_size=9),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

# ── Prediction trajectories ───────────────────────────────────────────────
st.divider()
st.markdown(f"<p class='section-title'>Model Prediction Trajectories</p>",
            unsafe_allow_html=True)

if preds.empty:
    st.info("Model predictions not yet available. Run `export_data.py`.")
else:
    task_cols = {
        "pred_prob_acr":  ("ACR classification risk",   "#7c83ff"),
        "hazard_acr":     ("ACR TTE hazard",            "#58a6ff"),
        "hazard_clad":    ("CLAD TTE hazard",           "#f78166"),
        "hazard_death":   ("Death TTE hazard",          "#ff7b72"),
    }
    fig2 = go.Figure()
    for col, (name, color) in task_cols.items():
        sub = preds[["anchor_dt", col]].dropna()
        if sub.empty:
            continue
        fig2.add_trace(go.Scatter(
            x=sub["anchor_dt"], y=sub[col],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2),
            marker=dict(size=7),
            hovertemplate=f"<b>{name}</b>: %{{y:.3f}}<br>%{{x|%Y-%m-%d}}<extra></extra>",
        ))

    # add ACR grade as vertical lines
    acr_rows = splits[splits.get("acr_encoded", pd.Series(dtype=float)) == 1] \
               if "acr_encoded" in splits.columns else pd.DataFrame()
    for _, row in acr_rows.iterrows():
        fig2.add_vline(
            x=row["anchor_dt"],
            line_dash="dot", line_color="rgba(224,92,75,0.4)",
            annotation_text="ACR+", annotation_font_size=8,
            annotation_font_color="rgba(224,92,75,0.8)",
        )

    fig2.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG2,
        height=360, margin=dict(l=20, r=20, t=20, b=40),
        xaxis=dict(title="Date", gridcolor=BORDER),
        yaxis=dict(title="Score", gridcolor=BORDER),
        legend=dict(orientation="h", x=0, y=-0.22, font_size=10),
        hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Clinical feature importance (from attention) ──────────────────────────
st.divider()
st.markdown(f"<p class='section-title'>Clinical Feature Attention (cohort model)</p>",
            unsafe_allow_html=True)

# Try loading per-patient attention if exported
ATTN_CSV = (
    __import__("pathlib").Path(__file__).parent.parent / "data" / "clinical_attn.csv"
)
if ATTN_CSV.exists():
    attn_df = pd.read_csv(ATTN_CSV)
    if "patient_id" in attn_df.columns:
        pt_attn = attn_df[attn_df["patient_id"] == pid]
    else:
        pt_attn = pd.DataFrame()

    if not pt_attn.empty:
        feat_cols_a = sorted([c for c in pt_attn.columns if c.startswith("feat_")],
                              key=lambda x: int(x.split("_")[1]))
        if feat_cols_a:
            row = pt_attn.iloc[0]
            vals  = row[feat_cols_a].values.astype(float)
            names = [feat_names.get(int(c.split("_")[1]), c) for c in feat_cols_a]
            # top 20 by absolute attention
            top_idx = np.argsort(np.abs(vals))[::-1][:20]
            y_vals  = [vals[i] for i in top_idx]
            y_names = [names[i] for i in top_idx]
            colors  = ["#e05c4b" if v > 0 else "#3fb950" for v in y_vals]

            fig3 = go.Figure(go.Bar(
                x=y_vals, y=y_names,
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig3.update_layout(
                template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG2,
                height=max(320, len(y_names) * 22 + 60),
                margin=dict(l=160, r=20, t=20, b=20),
                xaxis=dict(title="Attention weight", gridcolor=BORDER),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No feature columns in clinical attention file.")
    else:
        st.info(f"No clinical attention data for {pid}.")
else:
    st.info("Per-patient clinical attention not yet exported. Run `export_data.py`.")

# ── Raw data table ────────────────────────────────────────────────────────
with st.expander("Raw clinical feature values"):
    if not clin_df.empty and "patient_id" in clin_df.columns:
        pt_c = clin_df[clin_df["patient_id"] == pid]
    elif not clin_df.empty and "record_id" in clin_df.columns:
        pt_c = clin_df[clin_df["record_id"] == pid]
    else:
        pt_c = pd.DataFrame()
    if not pt_c.empty:
        # rename feat_N → actual name
        rename = {c: feat_names.get(int(c.split("_")[1]), c)
                  for c in pt_c.columns if c.startswith("feat_")}
        st.dataframe(pt_c.rename(columns=rename), use_container_width=True)
    else:
        st.write("No data.")
