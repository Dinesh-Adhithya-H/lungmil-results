"""Reusable Plotly figure builders."""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.styles import (
    BG, BG2, TEXT, MUTED, ACCENT, BORDER, CARD,
    ACR_COLORS, TASK_COLORS, MOD_COLORS, PLOTLY_THEME,
)

_BASE = dict(**PLOTLY_THEME)


# ---------------------------------------------------------------------------
# Patient timeline
# ---------------------------------------------------------------------------

def acr_grade_color(grade: str) -> str:
    if pd.isna(grade):
        return MUTED
    g = str(grade).strip().upper()
    if g in ACR_COLORS:
        return ACR_COLORS[g]
    if "A0" in g and "B0" in g:
        return ACR_COLORS["A0B0"]
    if "A1" in g or "A2" in g or "A3" in g:
        return ACR_COLORS["A3"]
    return MUTED


def timeline_figure(
    splits: pd.DataFrame,
    preds: pd.DataFrame,
    episodes: Optional[pd.Series] = None,
) -> go.Figure:
    """Biopsy timeline with ACR grades and model risk scores."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Biopsy Timeline", "Model Risk Scores"],
    )

    # ── row 1: biopsy dots on timeline ────────────────────────────────────
    # ACR episode spans
    if episodes is not None and len(episodes.get("episode_durations", [])) > 0:
        import ast
        starts = episodes.get("episode_durations", [])
        sizes  = episodes.get("episode_sizes", [])
        # we stored inter_ep_gaps but not start days directly; skip if absent
        # Use biopsy dates from splits to reconstruct episode spans
        acr_rows = splits[splits["acr_encoded"] == 1].sort_values("anchor_dt")
        if len(acr_rows) > 0:
            # group consecutive ACR+ days into episodes
            dates = acr_rows["anchor_dt"].sort_values().reset_index(drop=True)
            ep_starts, ep_ends = [], []
            ep_s = dates.iloc[0]
            ep_e = dates.iloc[0]
            for d in dates.iloc[1:]:
                if (d - ep_e).days <= 90:
                    ep_e = d
                else:
                    ep_starts.append(ep_s); ep_ends.append(ep_e)
                    ep_s = d; ep_e = d
            ep_starts.append(ep_s); ep_ends.append(ep_e)
            for s, e in zip(ep_starts, ep_ends):
                fig.add_vrect(
                    x0=s, x1=max(e, s + pd.Timedelta(days=3)),
                    fillcolor="rgba(224,92,75,0.13)", line_width=0,
                    row=1, col=1,
                )

    # Endpoint lines
    tx_date = splits["anchor_dt"].min()
    clad_days  = splits["clad_days"].iloc[0]  if "clad_days"  in splits.columns else None
    death_days = splits["death_days"].iloc[0] if "death_days" in splits.columns else None
    clad_evt   = splits["clad_status"].iloc[0]  if "clad_status"  in splits.columns else 0
    death_evt  = splits["death_status"].iloc[0] if "death_status" in splits.columns else 0

    if clad_days and float(clad_days) > 0 and float(clad_evt) == 1:
        clad_date = tx_date + pd.Timedelta(days=float(clad_days))
        fig.add_vline(x=clad_date, line_dash="dash", line_color="#f78166",
                      annotation_text="CLAD", annotation_font_color="#f78166",
                      row=1, col=1)
    if death_days and float(death_days) > 0 and float(death_evt) == 1:
        d_date = tx_date + pd.Timedelta(days=float(death_days))
        fig.add_vline(x=d_date, line_dash="dot", line_color="#ff7b72",
                      annotation_text="Death", annotation_font_color="#ff7b72",
                      row=1, col=1)

    # Biopsy points
    for _, row in splits.iterrows():
        color = acr_grade_color(row.get("acr_grade", ""))
        grade  = row.get("acr_grade", "?")
        days   = row.get("days_since_tx", 0)
        has_HE  = "✓" if row.get("has_HE") else "✗"
        has_BAL = "✓" if row.get("has_BAL") else "✗"
        has_CT  = "✓" if row.get("has_CT") else "✗"
        hover = (
            f"<b>{row['anchor_dt'].strftime('%Y-%m-%d')}</b><br>"
            f"ACR grade: {grade}<br>"
            f"Days since Tx: {days}<br>"
            f"HE {has_HE} · BAL {has_BAL} · CT {has_CT}"
        )
        fig.add_trace(go.Scatter(
            x=[row["anchor_dt"]], y=[0],
            mode="markers",
            marker=dict(color=color, size=13, symbol="circle",
                        line=dict(color="white", width=1.5)),
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

    # ── row 2: model risk scores over time ─────────────────────────────────
    if not preds.empty:
        task_map = {
            "pred_prob_acr":  ("ACR risk",   TASK_COLORS["acr_cls"]),
            "hazard_acr":     ("ACR hazard", TASK_COLORS["acr_surv"]),
            "hazard_clad":    ("CLAD hazard",TASK_COLORS["clad_surv"]),
            "hazard_death":   ("Death hazard",TASK_COLORS["death_surv"]),
        }
        for col, (name, color) in task_map.items():
            sub = preds[["anchor_dt", col]].dropna()
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["anchor_dt"], y=sub[col],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{name}</b>: %{{y:.3f}}<br>%{{x|%Y-%m-%d}}<extra></extra>",
            ), row=2, col=1)

    # ── layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        **_BASE,
        height=480,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", x=0, y=-0.15, font_size=11),
        hovermode="x unified",
    )
    fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor=BORDER)
    fig.update_yaxes(title_text="Score", gridcolor=BORDER, row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# Cluster frequency bar chart
# ---------------------------------------------------------------------------

def cluster_freq_figure(
    patient_freq: pd.Series,          # cluster_0 … cluster_K
    cohort_means: dict[str, pd.Series],  # {group_label: cluster_series}
    cluster_names: list[str],
    mod: str,
    title: str = "",
) -> go.Figure:
    K = len(cluster_names)
    x = list(range(K))
    fig = go.Figure()

    bar_w = max(0.18, min(0.35, 6 / max(K, 1)))

    # cohort means per group
    grp_colors = ["#3fb950", "#e05c4b", "#58a6ff", "#f2cc60"]
    for i, (lbl, series) in enumerate(cohort_means.items()):
        fig.add_trace(go.Bar(
            name=f"Group {lbl} mean",
            x=x, y=series.values * 100,
            width=bar_w, opacity=0.55,
            marker_color=grp_colors[i % len(grp_colors)],
            hovertemplate="%{x}: %{y:.1f}%<extra>" + str(lbl) + " mean</extra>",
        ))

    # this patient
    if patient_freq is not None and len(patient_freq) == K:
        fig.add_trace(go.Bar(
            name="This patient",
            x=x, y=patient_freq.values * 100,
            width=bar_w * 0.7, opacity=0.95,
            marker_color=ACCENT,
            hovertemplate="%{x}: %{y:.1f}%<extra>this patient</extra>",
        ))

    fig.update_layout(
        **_BASE,
        title=dict(text=title or f"{mod} Cluster Composition", font_color=TEXT, x=0),
        barmode="group",
        height=360,
        margin=dict(l=20, r=20, t=50, b=80),
        xaxis=dict(
            tickmode="array", tickvals=x, ticktext=cluster_names,
            tickangle=-45, tickfont_size=9, gridcolor=BORDER,
        ),
        yaxis=dict(title="% patches", gridcolor=BORDER),
        legend=dict(orientation="h", x=0, y=-0.3, font_size=10),
    )
    return fig


# ---------------------------------------------------------------------------
# UMAP scatter with patient highlighted
# ---------------------------------------------------------------------------

def umap_figure(
    umap_df: pd.DataFrame,
    highlight_pid: str,
    color_col: str = "acr_encoded",
    title: str = "Latent Space (UMAP)",
) -> go.Figure:
    fig = go.Figure()

    color_map = {0: "#3fb950", 1: "#e05c4b", "0": "#3fb950", "1": "#e05c4b"}

    # background population
    others = umap_df[umap_df["patient_id"] != highlight_pid]
    grp_labels = {0: "ACR−", 1: "ACR+", "0": "ACR−", "1": "ACR+"}
    for grp, sub in others.groupby(color_col):
        fig.add_trace(go.Scatter(
            x=sub["umap_x"], y=sub["umap_y"],
            mode="markers",
            name=grp_labels.get(grp, str(grp)),
            marker=dict(color=color_map.get(grp, MUTED), size=5, opacity=0.35),
            hoverinfo="skip",
        ))

    # highlighted patient
    me = umap_df[umap_df["patient_id"] == highlight_pid].sort_values("anchor_dt") \
         if "anchor_dt" in umap_df.columns \
         else umap_df[umap_df["patient_id"] == highlight_pid]

    if len(me) > 1:
        # trajectory arrows
        for i in range(len(me) - 1):
            r0, r1 = me.iloc[i], me.iloc[i + 1]
            fig.add_annotation(
                ax=r0["umap_x"], ay=r0["umap_y"],
                x=r1["umap_x"],  y=r1["umap_y"],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.2,
                arrowwidth=2, arrowcolor=ACCENT,
            )

    if len(me) > 0:
        dt_col = "anchor_dt" if "anchor_dt" in me.columns else None
        hover  = "<b>" + highlight_pid + "</b><br>" + \
                 (me["anchor_dt"].dt.strftime("%Y-%m-%d").astype(str) if dt_col else me.index.astype(str))
        fig.add_trace(go.Scatter(
            x=me["umap_x"], y=me["umap_y"],
            mode="markers",
            name=highlight_pid,
            marker=dict(
                color=ACCENT, size=14, symbol="star",
                line=dict(color="white", width=1.5),
            ),
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
        ))

    fig.update_layout(
        **_BASE,
        title=dict(text=title, font_color=TEXT, x=0),
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        legend=dict(orientation="h", x=0, y=-0.08, font_size=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Clinical feature trends
# ---------------------------------------------------------------------------

def clinical_trend_figure(
    clin_df: pd.DataFrame,
    feat_names: dict[int, str],
    top_n: int = 8,
    title: str = "Clinical Features Over Time",
) -> go.Figure:
    feat_cols = [c for c in clin_df.columns if c.startswith("feat_")]
    if not feat_cols:
        return go.Figure()

    # pick top_n features with highest variance
    sel = clin_df[feat_cols].std().nlargest(top_n).index.tolist()
    date_col = "anchor_dt" if "anchor_dt" in clin_df.columns else \
               "spiro_date" if "spiro_date" in clin_df.columns else None

    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for i, col in enumerate(sel):
        idx = int(col.split("_")[1])
        name = feat_names.get(idx, col)
        x = clin_df[date_col] if date_col else clin_df.index
        fig.add_trace(go.Scatter(
            x=x, y=clin_df[col],
            mode="lines+markers", name=name,
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{name}</b>: %{{y:.2f}}<br>%{{x}}<extra></extra>",
        ))

    fig.update_layout(
        **_BASE,
        title=dict(text=title, font_color=TEXT, x=0),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor=BORDER),
        yaxis=dict(title="Normalised value", gridcolor=BORDER),
        legend=dict(orientation="h", x=0, y=-0.2, font_size=9),
    )
    return fig


# ---------------------------------------------------------------------------
# Cohort ACR grade composition pie
# ---------------------------------------------------------------------------

def grade_pie(splits: pd.DataFrame, pid: str) -> go.Figure:
    counts = splits["acr_grade"].value_counts().reset_index()
    counts.columns = ["grade", "count"]
    colors = [acr_grade_color(g) for g in counts["grade"]]
    patient_grade = splits.query("patient_id == @pid")["acr_grade"].mode()
    fig = go.Figure(go.Pie(
        labels=counts["grade"], values=counts["count"],
        marker_colors=colors, hole=0.45,
        textinfo="label+percent", textfont_size=11,
    ))
    fig.update_layout(
        **_BASE,
        title=dict(text="ACR Grade Distribution (cohort)", font_color=TEXT, x=0),
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )
    return fig
