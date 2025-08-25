# src/auth_guard.py

import streamlit as st
import streamlit_authenticator as stauth
from src.auth import get_authenticator

def require_login() -> None:
    """
    Garantit qu’un utilisateur est authentifié.
    Redirige vers la page de Login si nécessaire.
    """
    authenticator = get_authenticator()
    
    # Appel du widget de login invisible. Celui-ci lira le cookie
    # et mettra à jour le session_state lors d'un re-run.
    authenticator.login(location="main", key="main_login")

    # Si après cet appel, le statut n'est pas True, on redirige.
    if not st.session_state.get("authentication_status"):
        st.switch_page("pages/Login.py")
        st.stop()