"""Patient Explorer — password gate + landing redirect to Cohort Overview."""

import os
import streamlit as st

st.set_page_config(
    page_title="Lung Transplant · Multimodal AI Patient Explorer",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

PASSWORD = os.environ.get("EXPLORER_PASSWORD", "")

def _check_password() -> bool:
    if st.session_state.get("_auth_ok"):
        return True
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none}
        body { background: #0e1117; color: #e8eaf0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h1 style='color:#e8eaf0;margin-bottom:4px'>🫁 Lung Transplant · Multimodal AI Patient Explorer</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#6c7199;font-size:1.05em;max-width:680px'>"
        "A research tool for exploring AI-predicted clinical trajectories in lung transplant patients "
        "(n=263, Helmholtz Munich). Integrates <b style='color:#C62828'>LongMK</b> "
        "(longitudinal multimodal attention) and <b style='color:#9C27B0'>SetMIL-MT</b> "
        "(set-based multimodal MIL) predictions across four clinical tasks: "
        "ACR classification, ACR survival, CLAD survival, and post-transplant mortality."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#6c7199;font-size:0.9em'>This is a private research tool. Enter the access password to continue.</p>",
        unsafe_allow_html=True,
    )
    pwd = st.text_input("Password", type="password", key="_pwd_input")
    if st.button("Enter", type="primary"):
        if pwd == PASSWORD:
            st.session_state["_auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

if _check_password():
    st.switch_page("pages/5_Cohort_Overview.py")
