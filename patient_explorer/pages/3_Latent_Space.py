"""Page 3 — Latent Space: SetMIL-MT & Longitudinal model UMAP representations."""

import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Latent Space", page_icon="🗺", layout="wide")

from utils.styles import card_css, TEXT, MUTED

st.markdown(card_css(), unsafe_allow_html=True)
st.markdown(f"<h2 style='color:{TEXT}'>🗺 Latent Space</h2>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{MUTED}'>UMAP projections of instance representations from the SetMIL-MT model, "
    f"aggregated across all 5 splits. Each point is a patch/sample instance coloured by task outcome.</p>",
    unsafe_allow_html=True,
)

INTERP_ROOT = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/interpretability/set_mil_mt_interp")

TASK_DIRS = {
    "ACR Classification": INTERP_ROOT / "all_splits_cls",
    "ACR Survival":       INTERP_ROOT / "all_splits_acr_surv",
    "CLAD Survival":      INTERP_ROOT / "all_splits_clad_surv",
    "Death Survival":     INTERP_ROOT / "all_splits_death_surv",
}

UMAP_FILES = {
    "Cosine UMAP (recommended)": "A_instance_reps_cosine.png",
    "Euclidean UMAP":            "A_instance_reps.png",
}

# ── Model selector ─────────────────────────────────────────────────────────
model_tab, = st.tabs(["SetMIL-MT Representations"])

with model_tab:
    umap_type = st.radio(
        "UMAP metric", list(UMAP_FILES.keys()), horizontal=True, label_visibility="collapsed"
    )
    fname = UMAP_FILES[umap_type]

    st.markdown(f"<p class='section-title'>Per-Task UMAP — {umap_type}</p>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (task_label, task_dir) in enumerate(TASK_DIRS.items()):
        img_path = task_dir / fname
        with cols[i % 2]:
            st.markdown(f"**{task_label}**")
            if img_path.exists():
                img = Image.open(img_path)
                st.image(img, use_container_width=True)
            else:
                st.info(f"Image not found: {img_path.name}")

st.divider()
st.markdown(
    f"<p style='color:{MUTED};font-size:0.82rem'>Per-patient longitudinal trajectories through "
    f"latent space are shown on the <b>SetMIL-MT</b> and <b>Longitudinal</b> pages. "
    f"Interpretability panels (seed activation, gate weights, cluster affinities) are on the "
    f"<b>Interpretability</b> page.</p>",
    unsafe_allow_html=True,
)
