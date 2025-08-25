# pages/Login.py

import streamlit as st
import streamlit_authenticator as stauth
from src.auth import get_auth_config # On importe la nouvelle fonction

# On récupère la config et on instancie l'objet
config = get_auth_config()
authenticator = stauth.Authenticate(**config)

# ---------- Login ----------
authenticator.login(
    location="main",
    fields={"Form name": "🔐 Connexion"},
    key="auditoo_login",
)

if st.session_state.get("authentication_status"):
    st.session_state["user"] = st.session_state.get("name") or "Utilisateur"
    st.switch_page("pages/Home.py")
elif st.session_state.get("authentication_status") is False:
    st.error("Login ou mot de passe incorrect.")
else:
    st.info("Entrez vos identifiants.")