import streamlit as st
from pages import Home
from src.ui_theme import apply_theme

st.set_page_config(page_title="Auditoo – Dashboard", layout="wide")

apply_theme()

st.switch_page("pages/Home.py")