# src/auth.py

import streamlit as st
from collections.abc import Mapping

def _to_dict(obj):
    if isinstance(obj, Mapping):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj

@st.cache_data  # On peut utiliser cache_data car on ne retourne que des données simples (dict)
def get_auth_config():
    """
    Lit les secrets et retourne un dictionnaire de configuration pour l'authentificateur.
    Cette partie peut être mise en cache en toute sécurité.
    """
    try:
        creds_config = st.secrets["credentials"]
        cookie_key = st.secrets["COOKIE_SECRET_KEY"]
    except KeyError as e:
        st.error(f"❌ La clé '{e.args[0]}' est manquante dans vos secrets.")
        st.stop()

    creds = _to_dict(creds_config)
    
    return {
        "credentials": creds,
        "cookie_name": "auditoo_dashboard",
        "cookie_key": cookie_key,
        "cookie_expiry_days": 30,
    }