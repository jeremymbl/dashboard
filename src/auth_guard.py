# src/auth_guard.py

import streamlit as st
import streamlit_authenticator as stauth
from src.auth import get_auth_config  # On importe la nouvelle fonction

def require_login() -> None:
    """
    Garantit qu’un utilisateur est authentifié.
    """
    config = get_auth_config()
    authenticator = stauth.Authenticate(**config) # On crée l'objet ici

    if "authentication_status" not in st.session_state:
        authenticator.login(
            location="cookie-check",
            fields={},
            key="cookie-check",
            clear_on_submit=False,
        )

    if not st.session_state.get("authentication_status"):
        st.switch_page("pages/Login.py")
        st.stop()