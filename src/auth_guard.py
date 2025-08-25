import streamlit as st

def require_login():
    """Redirige vers Login si l’utilisateur n’est pas authentifié."""
    if not st.session_state.get("authentication_status"):
        # Empêche le rendu partiel de la page courante
        st.switch_page("pages/Login.py")
        st.stop()