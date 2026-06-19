"""Page 3 — Latent Space / UMAP."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Latent Space", page_icon="🗺", layout="wide")

from utils.styles import card_css, BG, BG2, TEXT, MUTED, ACCENT, BORDER, TASK_COLORS
from utils.data_loader import load_umap, load_splits, patient_splits
from utils.plots import umap_figure

st.markdown(card_css(), unsafe_allow_html=True)

pid = st.session_state.get("selected_patient")
if not pid:
    st.warning("Select a patient from the home page.")
    st.stop()

st.markdown(f"<h2 style='color:{TEXT}'>🗺 Latent Space — {pid}</h2>", unsafe_allow_html=True)

umap_df  = load_umap()
splits   = load_splits()
pt_splits = patient_splits(pid)

if umap_df.empty:
    st.info("UMAP data not yet available. Run `export_data.py` to generate it.")
    st.stop()

# merge splits metadata into umap for coloring
if "patient_id" in umap_df.columns and "acr_encoded" not in umap_df.columns:
    umap_df = umap_df.merge(
        splits[["patient_id", "anchor_dt", "acr_encoded", "clad_status", "death_status"]].drop_duplicates(),
        on=["patient_id", "anchor_dt"] if "anchor_dt" in umap_df.columns else ["patient_id"],
        how="left",
    )

# ── Controls ───────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    color_opts = {
        "ACR label":     "acr_encoded",
        "CLAD status":   "clad_status",
        "Death status":  "death_status",
        "Task":          "task",
    }
    color_choice = st.selectbox("Colour by", list(color_opts.keys()))
    color_col = color_opts[color_choice]

with c2:
    task_opts = ["All tasks"] + sorted(umap_df["task"].unique().tolist()) \
                if "task" in umap_df.columns else ["All tasks"]
    sel_task = st.selectbox("Task", task_opts)

with c3:
    show_traj = st.checkbox("Show trajectory", value=True)

# filter by task
plot_df = umap_df if sel_task == "All tasks" else umap_df[umap_df["task"] == sel_task]

# ── Main UMAP scatter ──────────────────────────────────────────────────────
if color_col not in plot_df.columns:
    st.warning(f"Column '{color_col}' not available in UMAP data.")
    color_col = "acr_encoded" if "acr_encoded" in plot_df.columns else plot_df.columns[0]

fig = umap_figure(plot_df, pid, color_col=color_col,
                  title=f"UMAP – coloured by {color_choice}")

if not show_traj:
    fig.update_layout(annotations=[])

st.plotly_chart(fig, use_container_width=True)

# ── Per-task facet ─────────────────────────────────────────────────────────
if "task" in umap_df.columns and len(umap_df["task"].unique()) > 1:
    st.divider()
    st.markdown(f"<p class='section-title'>All Tasks</p>", unsafe_allow_html=True)

    tasks = umap_df["task"].unique().tolist()
    n_cols = min(4, len(tasks))
    cols   = st.columns(n_cols)

    for i, task in enumerate(tasks):
        sub = umap_df[umap_df["task"] == task]
        me  = sub[sub["patient_id"] == pid]
        with cols[i % n_cols]:
            f = go.Figure()
            col = "acr_encoded" if "acr_encoded" in sub.columns else None
            color_map = {0: "#3fb950", 1: "#e05c4b", 0.0: "#3fb950", 1.0: "#e05c4b"}
            if col:
                for grp, gsub in sub[sub["patient_id"] != pid].groupby(col):
                    f.add_trace(go.Scatter(
                        x=gsub["umap_x"], y=gsub["umap_y"],
                        mode="markers",
                        marker=dict(color=color_map.get(grp, MUTED), size=4, opacity=0.3),
                        showlegend=False, hoverinfo="skip",
                    ))
            else:
                others = sub[sub["patient_id"] != pid]
                f.add_trace(go.Scatter(
                    x=others["umap_x"], y=others["umap_y"],
                    mode="markers",
                    marker=dict(color=MUTED, size=4, opacity=0.3),
                    showlegend=False, hoverinfo="skip",
                ))
            if len(me) > 0:
                f.add_trace(go.Scatter(
                    x=me["umap_x"], y=me["umap_y"],
                    mode="markers", name=pid,
                    marker=dict(color=ACCENT, size=12, symbol="star",
                                line=dict(color="white", width=1)),
                    showlegend=False,
                    hovertemplate=f"<b>{pid}</b><extra>{task}</extra>",
                ))
            f.update_layout(
                template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG2,
                title=dict(text=task.replace("_", " ").upper(), font_color=TEXT, font_size=11, x=0.05),
                height=220,
                margin=dict(l=5, r=5, t=30, b=5),
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            )
            st.plotly_chart(f, use_container_width=True)

# ── Nearest neighbours ─────────────────────────────────────────────────────
st.divider()
st.markdown(f"<p class='section-title'>Nearest Neighbours in Latent Space</p>",
            unsafe_allow_html=True)

me_rows = umap_df[(umap_df["patient_id"] == pid)]
if len(me_rows) > 0:
    # centroid of this patient
    cx = me_rows["umap_x"].mean()
    cy = me_rows["umap_y"].mean()
    others = umap_df[umap_df["patient_id"] != pid].copy()
    if not others.empty:
        others["dist"] = np.sqrt((others["umap_x"] - cx)**2 + (others["umap_y"] - cy)**2)
        # aggregate to patient level
        nn = others.groupby("patient_id")["dist"].min().nsmallest(10).reset_index()
        nn.columns = ["patient_id", "distance"]
        if "acr_encoded" in others.columns:
            meta = others.drop_duplicates("patient_id")[["patient_id","acr_encoded"]]
            nn = nn.merge(meta, on="patient_id", how="left")
            nn["ACR"] = nn["acr_encoded"].map({0: "ACR−", 1: "ACR+", 0.0: "ACR−", 1.0: "ACR+"})
        st.dataframe(nn.reset_index(drop=True), use_container_width=True, height=260)
else:
    st.info(f"{pid} not found in UMAP embeddings.")
