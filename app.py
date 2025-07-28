import streamlit as st
from pages import Home, Prompts

st.set_page_config(page_title="Auditoo – Dashboards", layout="wide")

# Streamlit multipage
st.switch_page("pages/Home.py")
