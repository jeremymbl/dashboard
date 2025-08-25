# pages/Login.py

import streamlit as st
from src.auth import get_authenticator

# On récupère l'instance unique depuis le session_state
authenticator = get_authenticator()

# On affiche le formulaire de login. La méthode .login() est assez intelligente
# pour afficher le formulaire uniquement si on n'est pas déjà connecté par cookie.
authenticator.login(
    location="main",
    fields={"Form name": "🔐 Connexion"},
    key="form_login", # Une clé différente du garde
)

if st.session_state.get("authentication_status"):
    st.session_state["user"] = st.session_state.get("name") or "Utilisateur"
    # Redirige vers la page Home APRES le login réussi
    st.switch_page("pages/Home.py")
elif st.session_state.get("authentication_status") is False:
    st.error("Login ou mot de passe incorrect.")
else:
    st.info("Entrez vos identifiants.")