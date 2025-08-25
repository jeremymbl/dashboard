# src/auth_guard.py

import streamlit as st
import streamlit_authenticator as stauth
from src.auth import get_authenticator

def require_login() -> None:
    """
    Garantit qu’un utilisateur est authentifié.
    """
    authenticator = get_authenticator()

    # Le check silencieux du cookie au démarrage
    if "authentication_status" not in st.session_state:
        authenticator.login(
            location="cookie-check",
            fields={},
            key="cookie-check",
            clear_on_submit=False,
        )

    # Si le statut n'est toujours pas bon (pas de cookie ou cookie invalide),
    # on redirige vers la page de login.
    if not st.session_state.get("authentication_status"):
        st.switch_page("pages/Login.py")
        st.stop()