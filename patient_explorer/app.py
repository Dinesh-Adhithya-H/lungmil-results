"""Patient Explorer — password gate + landing redirect to Cohort Overview."""

import os
import streamlit as st

st.set_page_config(
    page_title="Lung Transplant · Patient Explorer",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

PASSWORD = os.environ.get("EXPLORER_PASSWORD", "lungmil2024")

def _check_password() -> bool:
    if st.session_state.get("_auth_ok"):
        return True
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("## 🫁 Lung Transplant Patient Explorer")
    st.markdown("This is a private research tool. Enter the access password to continue.")
    pwd = st.text_input("Password", type="password", key="_pwd_input")
    if st.button("Enter"):
        if pwd == PASSWORD:
            st.session_state["_auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

if _check_password():
    st.switch_page("pages/5_Cohort_Overview.py")
