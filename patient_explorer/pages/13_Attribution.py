"""Page 13 — Feature Attribution: what does the model look at per modality?"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Feature Attribution", page_icon="🔍", layout="wide")

from utils.styles import card_css, TEXT, MUTED, BG, BG2, BORDER, PLOTLY_THEME, CARD

# ── HE cluster biological name map ────────────────────────────────────────────
_HE_MAP_PATH = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/cluster_name_maps/HE_cluster_map.json")

@st.cache_data(show_spinner=False)
def _load_he_map():
    if _HE_MAP_PATH.exists():
        return json.loads(_HE_MAP_PATH.read_text())
    return {}

HE_BIO_MAP = _load_he_map()

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
    # Normalize to relative enrichment: % of max |delta| in this modality
    max_abs = np.abs(delta).max()
    rel = (delta / max_abs * 100) if max_abs > 0 else delta * 0
    df = pd.DataFrame({
        "cluster": names,
        "hi_score": hi,
        "lo_score": lo,
        "delta_raw": delta,
        "delta": rel,          # relative enrichment in %
    })
    if modality == "HE" and HE_BIO_MAP:
        df["bio_category"] = df["cluster"].map(lambda c: HE_BIO_MAP.get(c, "Unknown"))
    return df.sort_values("delta", ascending=False).reset_index(drop=True)

def bar_chart(df: pd.DataFrame, mod: str, top_n: int, task_label: str) -> go.Figure:
    col_pos, col_neg = MOD_COLORS[mod]
    # Separate genuinely positive from genuinely negative deltas
    pos_df = df[df["delta"] > 0].head(top_n)
    neg_df = df[df["delta"] < 0].tail(top_n).sort_values("delta")
    combined = pd.concat([neg_df, pos_df]).drop_duplicates("cluster").sort_values("delta")

    if combined.empty:
        combined = df.head(top_n).sort_values("delta")

    colors = [col_pos if v >= 0 else col_neg for v in combined["delta"]]

    # For HE: append bio category to y-axis labels
    has_bio = "bio_category" in combined.columns
    y_labels = (
        [f"{c} ({b})" for c, b in zip(combined["cluster"], combined["bio_category"])]
        if has_bio else combined["cluster"].tolist()
    )

    # customdata: [delta_raw, bio_category (or "")]
    if has_bio:
        customdata = np.column_stack([combined["delta_raw"], combined["bio_category"]])
        hover = (
            "<b>%{customdata[1]}</b> (%{y})<br>"
            "Relative enrichment: %{x:.1f}%<br>"
            "Δ-attention (raw): %{customdata[0]:.2e}<br>"
            "<extra></extra>"
        )
    else:
        customdata = combined["delta_raw"].values
        hover = (
            "<b>%{y}</b><br>"
            "Relative enrichment: %{x:.1f}%<br>"
            "Δ-attention (raw): %{customdata:.2e}<br>"
            "<extra></extra>"
        )

    fig = go.Figure(go.Bar(
        x=combined["delta"],
        y=y_labels,
        orientation="h",
        marker_color=colors,
        customdata=customdata,
        hovertemplate=hover,
        text=[f"{v:+.0f}%" for v in combined["delta"]],
        textposition="outside",
        textfont=dict(size=9, color=TEXT),
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text=f"{MOD_LABELS[mod]} — {task_label}", font_color=TEXT, font_size=13),
        xaxis=dict(
            title="Relative Δ-attention (% of max, high-risk − low-risk)",
            zeroline=True, zerolinecolor=BORDER, zerolinewidth=1.5,
        ),
        yaxis=dict(tickfont=dict(size=9 if has_bio else 10)),
        height=max(300, len(combined) * 24 + 80),
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
st.plotly_chart(gate_gauge(task_data))

st.divider()

# ── Per-modality attribution ──────────────────────────────────────────────────
st.markdown(
    f"<p class='section-title'>Cluster-level Feature Attribution — {task_label}</p>",
    unsafe_allow_html=True,
)

tab_bal, tab_clinical, tab_he, tab_ct = st.tabs(["🦠 BAL cell types", "📋 Clinical features", "🔬 HE clusters", "🖥 CT clusters"])

def _render_mod_tab(df, col_label, description, mod, top_n, task_label):
    """Render the two-column table + bar chart for one modality."""
    delta_max = df["delta"].max()
    delta_min = df["delta"].min()
    all_negative = delta_max <= 0
    all_positive = delta_min >= 0

    direction_note = ""
    if all_negative:
        direction_note = (
            f" ⚠ All Δ values are negative for this modality/task: the model attends "
            f"<b>more to low-risk patients</b> across all clusters. Table is sorted from "
            f"least-negative (relatively more in high-risk) to most-negative."
        )
    elif all_positive:
        direction_note = " The model attends more to high-risk patients across all clusters."

    st.markdown(
        f"<p style='color:{MUTED};font-size:0.82rem'>{description}{direction_note}</p>",
        unsafe_allow_html=True,
    )

    has_bio = "bio_category" in df.columns

    def fmt_tbl(sub, cmap):
        if has_bio:
            out = sub[["cluster", "bio_category", "delta_raw", "delta"]].copy()
            out.columns = [col_label, "Biological category", "Δ-attention (raw)", "Rel. enrichment (%)"]
        else:
            out = sub[["cluster", "delta_raw", "delta"]].copy()
            out.columns = [col_label, "Δ-attention (raw)", "Rel. enrichment (%)"]
        out["Δ-attention (raw)"] = out["Δ-attention (raw)"].map(lambda x: f"{x:.2e}")
        return out.style.background_gradient(subset=["Rel. enrichment (%)"], cmap=cmap)

    if all_negative:
        # Show top (least-negative = most relatively enriched in high-risk) and bottom
        st.markdown("**Relatively more attended in high-risk (least-negative)**")
        st.dataframe(fmt_tbl(df.head(20), "YlOrRd"), height=480, hide_index=True)
        st.markdown("**Relatively more attended in low-risk (most-negative)**")
        st.dataframe(fmt_tbl(df.tail(20).sort_values("delta"), "Blues"), height=480, hide_index=True)
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**High-risk enriched**")
            st.dataframe(fmt_tbl(df[df["delta"] > 0].head(20), "Reds"), height=480, hide_index=True)
        with c2:
            st.markdown("**Low-risk enriched**")
            st.dataframe(fmt_tbl(df[df["delta"] < 0].sort_values("delta").head(20), "Blues_r"), height=480, hide_index=True)

    st.plotly_chart(bar_chart(df, mod, top_n, task_label))

with tab_bal:
    df = attribution_df(task_data, "BAL")
    _render_mod_tab(df, "Cell type",
        "43 BAL scRNA cell-type clusters. TRAM = Tissue-Resident Alveolar Macrophages; "
        "MoAM = Monocyte-derived Alveolar Macrophages.",
        "BAL", top_n, task_label)

with tab_clinical:
    df = attribution_df(task_data, "Clinical")
    _render_mod_tab(df, "Feature",
        "106 clinical variables: lung function (FVC, FEV1), blood labs, PGD score, "
        "donor/recipient metadata (CMV, EBV, DSA).",
        "Clinical", top_n, task_label)

with tab_he:
    df = attribution_df(task_data, "HE")
    _render_mod_tab(df, "Cluster",
        "54 HE histology patch clusters. Cluster ID format: macro_sub (e.g. 5_1 = sub-cluster 1 of macro "
        "cluster 5). Biological category shows the tissue morphology type (Alveolar, Bronchial, etc.). "
        "Bar chart y-axis shows cluster ID with category in parentheses.",
        "HE", top_n, task_label)

with tab_ct:
    df = attribution_df(task_data, "CT")
    _render_mod_tab(df, "Cluster",
        f"{len(df)} CT morphology patch clusters (numeric IDs). "
        "See paper CT morphology figure for representative patches.",
        "CT", top_n, task_label)

with st.expander("ℹ️ How to read these charts"):
    st.markdown(
        """
**Relative enrichment (%)** = (Δ-attention / max |Δ-attention| in this modality) × 100.

Δ-attention = mean attention weight for that cluster in **high-risk** patients minus **low-risk** patients,
averaged across all splits. Attention weights are softmax-normalised across all clusters within a modality,
so absolute values are small (~1/N where N is number of clusters). The relative enrichment score makes
cross-modality comparisons readable without changing the ranking.

**Raw Δ-attention** is shown in scientific notation in the tables and on hover in the charts.

Note: if all bars point in one direction (e.g. HE for death survival), it means that modality is
used more for one risk group overall — not that individual clusters are uninformative. The ranking
still tells you which clusters contribute most.
        """,
        unsafe_allow_html=False,
    )

# ── Keep a divider before the cross-task section ─────────────────────────────
    with c2:
        st.markdown("**Low-risk enriched (bottom 20)**")
        bot20 = df.tail(20).sort_values("delta")[["cluster", "hi_score", "lo_score", "delta"]].copy()
        bot20.columns = ["Cluster", "Hi-risk attn", "Lo-risk attn", "Δ-attn"]
        st.dataframe(
            bot20.style.background_gradient(subset=["Δ-attn"], cmap="Blues_r"),
            height=480, hide_index=True,
        )
    st.plotly_chart(bar_chart(df, "CT", top_n, task_label))

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
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Rel. Δ (%)", thickness=12),
    ))
    fig_ht.update_layout(
        **PLOTLY_THEME,
        title=f"{MOD_LABELS[sel_mod]} — Δ-attention across tasks",
        height=max(400, len(feat_list) * 18 + 100),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(tickfont=dict(size=9)),
    )
    st.plotly_chart(fig_ht)
