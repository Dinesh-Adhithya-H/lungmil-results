# pages/6_Benchmark.py
"""Page 6 — Model Performance Benchmark: pre-rendered figures + best-model summary."""

from pathlib import Path
from PIL import Image
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Benchmark", page_icon="📊", layout="wide")

from utils.styles import card_css, metric_card, BG, TEXT, MUTED, BORDER, CARD, PLOTLY_THEME

st.markdown(card_css(), unsafe_allow_html=True)

FIG_DIR = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/figures/benchmark")

BEST_MODELS = {
    "acr_cls":   {"model": "SetMIL-MT (no SAB)", "metric": "BACC",    "value": 0.623, "std": 0.034, "model_color": "#9C27B0", "task_color": "#7c83ff"},
    "acr_surv":  {"model": "LongMK",             "metric": "C-index", "value": 0.679, "std": 0.064, "model_color": "#C62828", "task_color": "#58a6ff"},
    "clad_surv": {"model": "SetMIL-MT",          "metric": "C-index", "value": 0.563, "std": 0.080, "model_color": "#9C27B0", "task_color": "#f78166"},
    "death_surv":{"model": "LongMK",             "metric": "C-index", "value": 0.771, "std": 0.056, "model_color": "#C62828", "task_color": "#3fb950"},
}

TASK_LABELS = {
    "acr_cls":   "ACR Classification",
    "acr_surv":  "ACR Survival",
    "clad_surv": "CLAD Survival",
    "death_surv":"Death Survival",
    "all":       "All tasks (combined)",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:{TEXT}'>📊 Benchmark</h3>", unsafe_allow_html=True)
    st.divider()
    task_options = list(TASK_LABELS.keys())
    task_sel = st.selectbox("Task", task_options, format_func=lambda x: TASK_LABELS[x], index=4)

st.markdown(f"<h2 style='color:{TEXT}'>📊 Model Performance Benchmark</h2>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{MUTED}'>All models, all tasks · n=263 patients · 5-fold cross-validation · "
    f"Helmholtz Munich. ★ = best multimodal model per task.</p>",
    unsafe_allow_html=True,
)

# ── Best model summary cards ──────────────────────────────────────────────────
st.markdown(f"<p class='section-title'>Best Model per Task</p>", unsafe_allow_html=True)
cols = st.columns(4)
for col, (task_key, bm) in zip(cols, BEST_MODELS.items()):
    with col:
        col.markdown(
            f"""<div style='background:{CARD};border:1px solid {BORDER};
                border-left:4px solid {bm["task_color"]};border-radius:8px;
                padding:12px 16px'>
              <div style='color:{MUTED};font-size:0.7rem;text-transform:uppercase;letter-spacing:.05em'>
                {TASK_LABELS[task_key]}</div>
              <div style='color:{bm["model_color"]};font-size:0.95rem;font-weight:700;margin-top:4px'>
                {bm["model"]}</div>
              <div style='color:{TEXT};font-size:1.3rem;font-weight:800'>
                {bm["value"]:.3f}</div>
              <div style='color:{MUTED};font-size:0.78rem'>
                {bm["metric"]} ± {bm["std"]:.3f}</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.divider()

# ── Figure display ────────────────────────────────────────────────────────────
if task_sel == "all":
    fig_path = FIG_DIR / "benchmark_all.png"
    st.markdown(f"<p class='section-title'>Combined — all tasks</p>", unsafe_allow_html=True)
    if fig_path.exists():
        st.image(Image.open(fig_path), use_container_width=True,
                 caption="All models × all tasks. Bars = mean ± s.d. across 5 splits. ○ = per-split values.")
    else:
        st.warning(f"Figure not found: {fig_path}")
else:
    bm = BEST_MODELS[task_sel]
    fig_path = FIG_DIR / f"benchmark_{task_sel}.png"
    model_name  = bm["model"]
    model_color = bm["model_color"]
    metric_str  = f"{bm['metric']} {bm['value']:.3f} ± {bm['std']:.3f}"
    task_label  = TASK_LABELS[task_sel]
    st.markdown(
        f"<p class='section-title'>{task_label} — "
        f"<span style='color:{model_color}'>{model_name}</span> "
        f"{metric_str}</p>",
        unsafe_allow_html=True,
    )
    if fig_path.exists():
        st.image(Image.open(fig_path), use_container_width=True,
                 caption=f"{TASK_LABELS[task_sel]}: bars = mean ± s.d., ○ = per-split, ★ = best multimodal model.")
    else:
        st.warning(f"Figure not found: {fig_path}")

st.divider()

# ── Hardcoded results table ───────────────────────────────────────────────────
st.markdown(f"<p class='section-title'>Summary Table</p>", unsafe_allow_html=True)

MODEL_GROUPS = {
    "Linear HE":          "Unimodal Linear",
    "Linear BAL":         "Unimodal Linear",
    "Linear CT":          "Unimodal Linear",
    "Linear Clinical":    "Unimodal Linear",
    "Wtd. avg. Linear":   "Unimodal Linear",
    "ABMIL HE":           "Unimodal ABMIL",
    "ABMIL BAL":          "Unimodal ABMIL",
    "ABMIL CT":           "Unimodal ABMIL",
    "ABMIL Clinical":     "Unimodal ABMIL",
    "Wtd. avg. ABMIL":    "Unimodal ABMIL",
    "Early fusion":       "Multimodal Fusion",
    "Middle fusion":      "Multimodal Fusion",
    "Late fusion":        "Multimodal Fusion",
    "SetMIL":             "Set-based MIL",
    "SetMIL-MT":          "Set-based MIL",
    "SetMIL-MT (no SAB)": "Set-based MIL",
    "LongMK-MT":          "Longitudinal",
    "LongMK":             "Longitudinal",
}

rows = [
    # ACR Classification (BACC)
    {"Group": "Set-based MIL",    "Model": "SetMIL-MT (no SAB)", "Task": "ACR cls",   "Metric": "BACC",    "Mean": 0.623, "Std": 0.034, "Best": "★"},
    {"Group": "Set-based MIL",    "Model": "SetMIL-MT",          "Task": "ACR cls",   "Metric": "BACC",    "Mean": 0.606, "Std": 0.038, "Best": ""},
    {"Group": "Longitudinal",     "Model": "LongMK",             "Task": "ACR cls",   "Metric": "BACC",    "Mean": 0.598, "Std": 0.041, "Best": ""},
    # ACR Survival (C-index)
    {"Group": "Longitudinal",     "Model": "LongMK",             "Task": "ACR surv",  "Metric": "C-index", "Mean": 0.679, "Std": 0.064, "Best": "★"},
    {"Group": "Longitudinal",     "Model": "LongMK-MT",          "Task": "ACR surv",  "Metric": "C-index", "Mean": 0.651, "Std": 0.059, "Best": ""},
    {"Group": "Set-based MIL",    "Model": "SetMIL-MT",          "Task": "ACR surv",  "Metric": "C-index", "Mean": 0.628, "Std": 0.055, "Best": ""},
    # CLAD Survival (C-index)
    {"Group": "Set-based MIL",    "Model": "SetMIL-MT",          "Task": "CLAD surv", "Metric": "C-index", "Mean": 0.563, "Std": 0.080, "Best": "★"},
    {"Group": "Longitudinal",     "Model": "LongMK",             "Task": "CLAD surv", "Metric": "C-index", "Mean": 0.549, "Std": 0.072, "Best": ""},
    # Death Survival (C-index)
    {"Group": "Longitudinal",     "Model": "LongMK",             "Task": "Death surv","Metric": "C-index", "Mean": 0.771, "Std": 0.056, "Best": "★"},
    {"Group": "Longitudinal",     "Model": "LongMK-MT",          "Task": "Death surv","Metric": "C-index", "Mean": 0.748, "Std": 0.062, "Best": ""},
    {"Group": "Set-based MIL",    "Model": "SetMIL-MT",          "Task": "Death surv","Metric": "C-index", "Mean": 0.720, "Std": 0.058, "Best": ""},
]

df_table = pd.DataFrame(rows)
st.dataframe(
    df_table.style.background_gradient(subset=["Mean"], cmap="RdYlGn", vmin=0.5, vmax=0.85)
            .map(lambda v: "font-weight:bold;color:#FFD700" if v == "★" else "", subset=["Best"]),
    hide_index=True,
    use_container_width=True,
)
st.caption(
    "★ = best model for that task. Full benchmark with all 18 models available in the figure above. "
    "n = 263 patients · 5-fold CV · mean ± s.d. across folds."
)
