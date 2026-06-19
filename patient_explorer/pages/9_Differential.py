"""Page 9 — Differential Abundance: volcano plots, top features, heatmaps."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Differential Abundance", page_icon="🔥", layout="wide")

from utils.styles import card_css, BG, BG2, TEXT, MUTED, ACCENT, BORDER, CARD, PLOTLY_THEME
from utils.data_loader import load_differential, load_cluster_freq, load_splits

st.markdown(card_css(), unsafe_allow_html=True)

st.markdown(f"<h2 style='color:{TEXT}'>🔥 Differential Abundance Analysis</h2>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{MUTED}'>Feature-level comparisons between outcome groups across all modalities.</p>", unsafe_allow_html=True)

diff_df = load_differential()

if diff_df.empty:
    st.info("Differential abundance analysis not yet run. Submit `nature_analysis.sh` to generate `differential_abundance.csv`.")
    st.stop()

# ── Controls ───────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    outcomes_avail = sorted(diff_df["outcome"].unique())
    outcome = st.selectbox("Outcome", outcomes_avail)
with c2:
    feats_avail = sorted(diff_df[diff_df["outcome"]==outcome]["feat_label"].unique())
    feat_label = st.selectbox("Feature type", feats_avail)
with c3:
    fdr_thr = st.selectbox("FDR threshold", [0.05, 0.10, 0.20], format_func=lambda x: f"q < {x}")

sub = diff_df[(diff_df["outcome"]==outcome) & (diff_df["feat_label"]==feat_label)].copy()
sig = sub[sub["qval"] < fdr_thr] if "qval" in sub.columns else pd.DataFrame()

# ── Summary metrics ────────────────────────────────────────────────────────
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Total features", len(sub))
mc2.metric("Significant (q<0.05)", int((sub["qval"]<0.05).sum()) if "qval" in sub.columns else "—")
mc3.metric("Up-regulated", int(((sub["qval"]<fdr_thr)&(sub["log2fc"]>0)).sum()) if "qval" in sub.columns else "—")
mc4.metric("Down-regulated", int(((sub["qval"]<fdr_thr)&(sub["log2fc"]<0)).sum()) if "qval" in sub.columns else "—")

# ── Volcano plot ───────────────────────────────────────────────────────────
st.divider()
st.markdown(f"<p class='section-title'>Volcano Plot — {outcome} × {feat_label}</p>", unsafe_allow_html=True)

if "log2fc" in sub.columns and "pval" in sub.columns:
    sub["neg_log10p"] = -np.log10(sub["pval"].clip(lower=1e-10))
    sub["is_sig"]     = (sub.get("qval", pd.Series([1]*len(sub))) < fdr_thr)
    sub["color"]      = sub.apply(
        lambda r: ("#F85149" if r["log2fc"]>0 else "#58A6FF") if r.get("is_sig") else "#7D8590", axis=1)

    fig_v = go.Figure()
    for is_sig, label, opacity in [(False,"Not significant",0.4),(True,"Significant",0.9)]:
        m = sub[sub["is_sig"]==is_sig]
        if m.empty: continue
        fig_v.add_trace(go.Scatter(
            x=m["log2fc"], y=m["neg_log10p"],
            mode="markers",
            name=label,
            marker=dict(color=m["color"].tolist(), size=7 if is_sig else 5,
                        opacity=opacity, line=dict(width=0.5, color="#30363D")),
            text=m["feature"],
            hovertemplate="<b>%{text}</b><br>log₂FC: %{x:.3f}<br>−log₁₀p: %{y:.2f}<extra></extra>",
            customdata=m[["mean_pos","mean_neg","qval"]].values if "mean_pos" in m.columns else None,
        ))

    # label top hits
    if not sig.empty:
        top = sub[sub["is_sig"]].nlargest(min(10, len(sig)), "neg_log10p")
        for _, r in top.iterrows():
            fig_v.add_annotation(x=r["log2fc"], y=r["neg_log10p"], text=r["feature"],
                                 showarrow=True, arrowhead=2, arrowsize=0.8,
                                 font=dict(size=8, color=TEXT), arrowcolor=MUTED,
                                 bgcolor=BG2, bordercolor=BORDER, borderwidth=1)

    fig_v.add_vline(x=0, line_color=MUTED, line_width=0.8, line_dash="dash")
    fig_v.add_hline(y=-np.log10(0.05), line_color="#D29922", line_width=0.8,
                    line_dash="dash", annotation_text="p=0.05", annotation_position="right")

    fig_v.update_layout(**PLOTLY_THEME, height=500,
                        xaxis_title="log₂ fold-change (outcome+ vs outcome−)",
                        yaxis_title="−log₁₀ p-value",
                        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1))
    st.plotly_chart(fig_v, use_container_width=True)

# ── Top significant features ───────────────────────────────────────────────
if not sig.empty:
    st.divider()
    st.markdown(f"<p class='section-title'>Top Significant Features (q < {fdr_thr})</p>", unsafe_allow_html=True)
    top20 = sig.reindex(sig["log2fc"].abs().sort_values(ascending=False).index).head(20)
    fig_bar = go.Figure(go.Bar(
        x=top20["log2fc"].values,
        y=top20["feature"].values,
        orientation="h",
        marker=dict(
            color=["#F85149" if v>0 else "#58A6FF" for v in top20["log2fc"].values],
            line=dict(width=0)
        ),
        hovertemplate="<b>%{y}</b><br>log₂FC: %{x:.3f}<extra></extra>",
    ))
    fig_bar.add_vline(x=0, line_color=MUTED, line_width=0.8)
    fig_bar.update_layout(**PLOTLY_THEME, height=max(300, 20*len(top20)),
                          xaxis_title="log₂ fold-change",
                          yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Full table ─────────────────────────────────────────────────────────────
st.divider()
with st.expander("Full results table"):
    show_cols = [c for c in ["feature","feat_label","log2fc","pval","qval","sig",
                              "mean_pos","mean_neg"] if c in sub.columns]
    st.dataframe(
        sub[show_cols].sort_values("qval" if "qval" in sub.columns else "pval")
                      .style.background_gradient(subset=["log2fc"], cmap="RdBu_r", vmin=-2, vmax=2)
                      .format({c: "{:.4f}" for c in ["log2fc","pval","qval","mean_pos","mean_neg"]
                                if c in show_cols}),
        use_container_width=True, height=400,
    )
