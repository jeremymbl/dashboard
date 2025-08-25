import streamlit as st
from src.ui_theme import apply_theme

st.set_page_config(page_title="Auditoo – Dashboard", layout="wide")

if not st.session_state.get("authentication_status"):
    st.switch_page("pages/Login.py")
    st.stop()

apply_theme()

st.write("Bienvenue sur le dashboard Auditoo !")
st.sidebar.success("Sélectionnez une page ci-dessus.")