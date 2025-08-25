# app.py

import streamlit as st
from src.ui_theme import apply_theme
from src.auth_guard import require_login
require_login()

st.set_page_config(page_title="Auditoo – Dashboard", layout="wide")

apply_theme()

st.write("Bienvenue sur le dashboard Auditoo !")
st.sidebar.success("Sélectionnez une page ci-dessus.")