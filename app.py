import streamlit as st
from pages import Home, Prompts
from src.ui_theme import apply_theme
                                              
st.set_page_config(page_title="Auditoo – Dashboards", layout="wide")

apply_theme()

# Streamlit multipage
st.switch_page("pages/Home.py")
