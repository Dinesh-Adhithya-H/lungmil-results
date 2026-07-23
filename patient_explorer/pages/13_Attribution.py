"""Page 13 — Feature Attribution: what does the model look at per modality?"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Feature Attribution", page_icon="🔍", layout="wide")

from utils.styles import card_css, TEXT, MUTED, BG, BG2, BORDER, PLOTLY_THEME, CARD

st.markdown(card_css(), unsafe_allow_html=True)
st.markdown(f"<h2 style='color:{TEXT}'>🔍 Feature Attribution</h2>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{MUTED}'>What does the model attend to when predicting each outcome? "
    f"<b>Δ-attention = high-risk attention − low-risk attention</b>. "
    f"Positive (red) → feature attended more in high-risk patients. "
    f"Negative (blue) → feature attended more in low-risk patients.</p>",
    unsafe_allow_html=True,
)

# ── Data ──────────────────────────────────────────────────────────────────────
INTERP_ROOT = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/interpretability/set_mil_mt_interp")

TASK_MAP = {
    "Death survival":  ("all_splits_death_surv", "death_surv"),
    "CLAD survival":   ("all_splits_clad_surv",  "clad_surv"),
    "ACR classification": ("all_splits_cls",     "acr_cls"),
    "ACR survival":    ("all_splits_acr_surv",   "acr_surv"),
}

MOD_COLORS = {
    "HE":       ("#a5d6ff", "#1f6feb"),
    "BAL":      ("#ffa657", "#d1651a"),
    "CT":       ("#7ee787", "#2ea043"),
    "Clinical": ("#f2cc60", "#b08800"),
}
MOD_LABELS = {
    "HE":       "HE Histology clusters",
    "BAL":      "BAL scRNA cell types",
    "CT":       "CT morphology clusters",
    "Clinical": "Clinical features",
}

@st.cache_data(show_spinner=False)
def load_task_data(task_dir: str, task_key: str):
    jf = INTERP_ROOT / task_dir / "paper_interp_data.json"
    if not jf.exists():
        return None
    with open(jf) as f:
        d = json.load(f)
    task = d["tasks"].get(task_key)
    if task is None:
        return None
    return task

def attribution_df(task_data, modality: str) -> pd.DataFrame:
    ca = task_data["cluster_affinity"][modality]
    names  = ca["cluster_names"]
    hi     = np.array(ca["hi_score"])
    lo     = np.array(ca["lo_score"])
    delta  = np.array(ca["delta"])
    return pd.DataFrame({
        "cluster": names,
        "hi_score": hi,
        "lo_score": lo,
        "delta": delta,
    }).sort_values("delta", ascending=False).reset_index(drop=True)

def bar_chart(df: pd.DataFrame, mod: str, top_n: int, task_label: str) -> go.Figure:
    col_pos, col_neg = MOD_COLORS[mod]
    # take top_n positive + top_n negative
    pos = df.head(top_n)
    neg = df.tail(top_n).sort_values("delta")
    combined = pd.concat([pos, neg]).drop_duplicates("cluster")
    combined = combined.sort_values("delta")

    colors = [col_pos if v >= 0 else col_neg for v in combined["delta"]]

    fig = go.Figure(go.Bar(
        x=combined["delta"],
        y=combined["cluster"],
        orientation="h",
        marker_color=colors,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Δ-attention: %{x:.5f}<br>"
            "<extra></extra>"
        ),
        text=[f"{v:+.4f}" for v in combined["delta"]],
        textposition="outside",
        textfont=dict(size=9, color=TEXT),
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text=f"{MOD_LABELS[mod]} — {task_label}", font_color=TEXT, font_size=13),
        xaxis=dict(
            title="Δ attention (high-risk − low-risk)",
            zeroline=True, zerolinecolor=BORDER, zerolinewidth=1.5,
        ),
        yaxis=dict(tickfont=dict(size=10)),
        height=max(300, len(combined) * 22 + 80),
        margin=dict(l=10, r=60, t=40, b=30),
        bargap=0.25,
    )
    return fig

def gate_gauge(task_data) -> go.Figure:
    gw = task_data["gate_weights"]["mean"]
    gw_std = task_data["gate_weights"].get("std", {m: 0 for m in gw})
    mods = list(gw.keys())
    vals = [gw[m] for m in mods]
    errs = [gw_std.get(m, 0) for m in mods]
    colors = [MOD_COLORS[m][0] for m in mods]

    fig = go.Figure(go.Bar(
        x=mods, y=vals,
        error_y=dict(type="data", array=errs, visible=True, color=MUTED),
        marker_color=colors,
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        hovertemplate="<b>%{x}</b>: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title="Modality Gate Weights (mean ± std across splits)",
        yaxis=dict(range=[0.7, 0.95], title="Gate weight"),
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

# ── Controls ──────────────────────────────────────────────────────────────────
col_t, col_n = st.columns([3, 1])
with col_t:
    task_label = st.selectbox("Outcome / task", list(TASK_MAP.keys()), index=0)
with col_n:
    top_n = st.slider("Top N features per modality", 10, 40, 20)

task_dir, task_key = TASK_MAP[task_label]
task_data = load_task_data(task_dir, task_key)

if task_data is None:
    st.error(f"No interpretability data found for {task_label}. Check that `paper_interp_data.json` exists.")
    st.stop()

st.divider()

# ── Modality gate weights ─────────────────────────────────────────────────────
st.markdown(f"<p class='section-title'>Modality Importance (Gate Weights)</p>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{MUTED};font-size:0.82rem'>Each modality gate weight reflects how much the model "
    f"relies on that modality when making predictions. All modalities are heavily used (>0.8).</p>",
    unsafe_allow_html=True,
)
st.plotly_chart(gate_gauge(task_data), width="stretch")

st.divider()

# ── Per-modality attribution ──────────────────────────────────────────────────
st.markdown(
    f"<p class='section-title'>Cluster-level Feature Attribution — {task_label}</p>",
    unsafe_allow_html=True,
)

tab_bal, tab_clinical, tab_he, tab_ct = st.tabs(["🦠 BAL cell types", "📋 Clinical features", "🔬 HE clusters", "🖥 CT clusters"])

with tab_bal:
    df = attribution_df(task_data, "BAL")
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.82rem'>"
        f"43 BAL scRNA cell-type clusters. Top positive = cell types whose attention is higher in "
        f"high-risk patients. TRAM = Tissue-Resident Alveolar Macrophages.</p>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**High-risk enriched (top 20)**")
        top20 = df.head(20)[["cluster", "hi_score", "lo_score", "delta"]].copy()
        top20.columns = ["Cell type", "Hi-risk attn", "Lo-risk attn", "Δ-attn"]
        st.dataframe(
            top20.style.background_gradient(subset=["Δ-attn"], cmap="Reds"),
            height=480, hide_index=True,
        )
    with c2:
        st.markdown("**Low-risk enriched (bottom 20)**")
        bot20 = df.tail(20).sort_values("delta")[["cluster", "hi_score", "lo_score", "delta"]].copy()
        bot20.columns = ["Cell type", "Hi-risk attn", "Lo-risk attn", "Δ-attn"]
        st.dataframe(
            bot20.style.background_gradient(subset=["Δ-attn"], cmap="Blues_r"),
            height=480, hide_index=True,
        )
    st.plotly_chart(bar_chart(df, "BAL", top_n, task_label), width="stretch")

with tab_clinical:
    df = attribution_df(task_data, "Clinical")
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.82rem'>"
        f"106 clinical variables (lung function tests, blood labs, donor/recipient metadata). "
        f"Top positive = variables where model attends more in high-risk patients.</p>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**High-risk enriched (top 20)**")
        top20 = df.head(20)[["cluster", "hi_score", "lo_score", "delta"]].copy()
        top20.columns = ["Feature", "Hi-risk attn", "Lo-risk attn", "Δ-attn"]
        st.dataframe(
            top20.style.background_gradient(subset=["Δ-attn"], cmap="Reds"),
            height=480, hide_index=True,
        )
    with c2:
        st.markdown("**Low-risk enriched (bottom 20)**")
        bot20 = df.tail(20).sort_values("delta")[["cluster", "hi_score", "lo_score", "delta"]].copy()
        bot20.columns = ["Feature", "Hi-risk attn", "Lo-risk attn", "Δ-attn"]
        st.dataframe(
            bot20.style.background_gradient(subset=["Δ-attn"], cmap="Blues_r"),
            height=480, hide_index=True,
        )
    st.plotly_chart(bar_chart(df, "Clinical", top_n, task_label), width="stretch")

with tab_he:
    df = attribution_df(task_data, "HE")
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.82rem'>"
        f"54 HE histology patch clusters (hierarchical, e.g. 0_1 = sub-cluster 1 of cluster 0). "
        f"See HE morphology figure in paper for cluster annotation.</p>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**High-risk enriched (top 20)**")
        top20 = df.head(20)[["cluster", "hi_score", "lo_score", "delta"]].copy()
        top20.columns = ["Cluster", "Hi-risk attn", "Lo-risk attn", "Δ-attn"]
        st.dataframe(
            top20.style.background_gradient(subset=["Δ-attn"], cmap="Reds"),
            height=480, hide_index=True,
        )
    with c2:
        st.markdown("**Low-risk enriched (bottom 20)**")
        bot20 = df.tail(20).sort_values("delta")[["cluster", "hi_score", "lo_score", "delta"]].copy()
        bot20.columns = ["Cluster", "Hi-risk attn", "Lo-risk attn", "Δ-attn"]
        st.dataframe(
            bot20.style.background_gradient(subset=["Δ-attn"], cmap="Blues_r"),
            height=480, hide_index=True,
        )
    st.plotly_chart(bar_chart(df, "HE", top_n, task_label), width="stretch")

with tab_ct:
    df = attribution_df(task_data, "CT")
    n_ct = len(df)
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.82rem'>"
        f"{n_ct} CT morphology patch clusters (numeric IDs). "
        f"See CT morphology figure in paper for representative patches per cluster.</p>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**High-risk enriched (top 20)**")
        top20 = df.head(20)[["cluster", "hi_score", "lo_score", "delta"]].copy()
        top20.columns = ["Cluster", "Hi-risk attn", "Lo-risk attn", "Δ-attn"]
        st.dataframe(
            top20.style.background_gradient(subset=["Δ-attn"], cmap="Reds"),
            height=480, hide_index=True,
        )
    with c2:
        st.markdown("**Low-risk enriched (bottom 20)**")
        bot20 = df.tail(20).sort_values("delta")[["cluster", "hi_score", "lo_score", "delta"]].copy()
        bot20.columns = ["Cluster", "Hi-risk attn", "Lo-risk attn", "Δ-attn"]
        st.dataframe(
            bot20.style.background_gradient(subset=["Δ-attn"], cmap="Blues_r"),
            height=480, hide_index=True,
        )
    st.plotly_chart(bar_chart(df, "CT", top_n, task_label), width="stretch")

st.divider()

# ── Cross-task summary heatmap ─────────────────────────────────────────────────
st.markdown(f"<p class='section-title'>Cross-Task Comparison</p>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{MUTED};font-size:0.82rem'>Δ-attention for top features across all tasks. "
    f"Rows = features, columns = tasks. Red = high-risk enriched.</p>",
    unsafe_allow_html=True,
)

sel_mod = st.selectbox("Modality for cross-task heatmap", ["BAL", "Clinical", "CT", "HE"], index=0)

# collect top features across all tasks for the selected modality
all_dfs = {}
for tl, (td, tk) in TASK_MAP.items():
    td_data = load_task_data(td, tk)
    if td_data:
        all_dfs[tl] = attribution_df(td_data, sel_mod)

if all_dfs:
    # union of top 20 features from each task
    top_feats = set()
    for df in all_dfs.values():
        top_feats.update(df.head(20)["cluster"].tolist())
        top_feats.update(df.tail(10)["cluster"].tolist())

    tasks_list = list(all_dfs.keys())
    feat_list = sorted(top_feats)

    mat = np.zeros((len(feat_list), len(tasks_list)))
    for j, tl in enumerate(tasks_list):
        df = all_dfs[tl].set_index("cluster")["delta"]
        for i, feat in enumerate(feat_list):
            mat[i, j] = df.get(feat, 0.0)

    # sort rows by max absolute delta
    row_order = np.argsort(-np.abs(mat).max(axis=1))
    mat = mat[row_order]
    feat_list = [feat_list[i] for i in row_order]

    fig_ht = go.Figure(go.Heatmap(
        z=mat,
        x=tasks_list,
        y=feat_list,
        colorscale="RdBu_r",
        zmid=0,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.5f}<extra></extra>",
        colorbar=dict(title="Δ-attn", thickness=12),
    ))
    fig_ht.update_layout(
        **PLOTLY_THEME,
        title=f"{MOD_LABELS[sel_mod]} — Δ-attention across tasks",
        height=max(400, len(feat_list) * 18 + 100),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(tickfont=dict(size=9)),
    )
    st.plotly_chart(fig_ht, width="stretch")
