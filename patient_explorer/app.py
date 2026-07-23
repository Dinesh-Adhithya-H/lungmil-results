"""Patient Explorer — landing redirect to Cohort Overview."""

import streamlit as st

st.set_page_config(
    page_title="Lung Transplant · Patient Explorer",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.switch_page("pages/5_Cohort_Overview.py")
