import hashlib
from datetime import datetime, timedelta

import streamlit as st
from loguru import logger
from streamlit_local_storage import LocalStorage

app_prefix = "auditoo_dashboard"
expiry_timedelta = timedelta(days=30)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def verify_password(stored_hash: str, input_password: str) -> bool:
    return stored_hash == hash_password(input_password)

def verify_username(input_username: str) -> bool:
    return input_username == st.secrets["username"]

# Hash du mot de passe depuis les secrets
current_pw_hash = hash_password(st.secrets["password"])

def store_session_cookie(ls: LocalStorage, stored_hash: str):
    now = datetime.now()
    expires_at = now + expiry_timedelta
    ls.setItem(f"{app_prefix}_pw_hash", stored_hash, "pw_hash")
    ls.setItem(f"{app_prefix}_pw_expires_at", expires_at.isoformat(), "pw_expires_at")
    logger.info("Session Cookie : stored")

def clear_session_cookie(ls: LocalStorage):
    ls.deleteItem(f"{app_prefix}_pw_hash", "pw_hash")
    ls.deleteItem(f"{app_prefix}_pw_expires_at", "pw_expires_at")
    logger.info("Session Cookie : cleared")

def load_session_cookie(ls: LocalStorage) -> bool:
    cookie = ls.getAll()

    if f"{app_prefix}_pw_hash" not in cookie or f"{app_prefix}_pw_expires_at" not in cookie:
        logger.info("Session Cookie : not found")
        return False

    if current_pw_hash != cookie[f"{app_prefix}_pw_hash"]:
        logger.warning("Session Cookie : invalid password")
        return False

    expires_at = datetime.fromisoformat(cookie[f"{app_prefix}_pw_expires_at"])
    if expires_at < datetime.now():
        logger.warning("Session Cookie : expired")
        clear_session_cookie(ls)
        return False

    logger.success("Session Cookie : valid session")
    return True

def validate(ls: LocalStorage):
    """Validate user authentication from password or cookie.
    If the user is not authenticated stops execution."""

    def validate_credentials():
        """Checks whether username and password entered by the user are correct."""
        username_valid = "username" in st.session_state and verify_username(st.session_state["username"])
        password_valid = "password" in st.session_state and verify_password(current_pw_hash, st.session_state["password"])
        
        if username_valid and password_valid:
            logger.success("Credentials are valid")
            st.session_state["logged_in"] = True
            store_session_cookie(ls, current_pw_hash)
            del st.session_state["password"]  # Don't store the password.
            del st.session_state["username"]  # Don't store the username.
        else:
            logger.error("Invalid credentials")
            st.session_state["logged_in"] = False

    def logout():
        if "logged_in" in st.session_state:
            del st.session_state["logged_in"]
        if "password" in st.session_state:
            del st.session_state["password"]
        clear_session_cookie(ls)
        st.cache_data.clear()
        
    if load_session_cookie(ls):
        st.session_state["logged_in"] = True

    with st.sidebar:
        # If user is logged-in, show the logout button
        if st.session_state.get("logged_in", False):
            st.button("🚪 Se déconnecter", on_click=logout, use_container_width=True)
            logger.success("User authenticated")
            return True

        # If user is not logged-in, show the inputs for username and password
        st.text_input(
            "👤 Nom d'utilisateur", on_change=validate_credentials, key="username"
        )
        st.text_input(
            "🔐 Mot de passe", type="password", on_change=validate_credentials, key="password"
        )

        # The user is not logged-in, but has attempt to log-in because the key "logged_in" is in the session state
        if "logged_in" in st.session_state:
            st.error("😕 Identifiants incorrects")

        logger.error("User not authenticated")
        return False

def is_authenticated() -> bool:
    """Vérifie si l'utilisateur est authentifié."""
    return st.session_state.get("logged_in", False) is True

def require_login():
    """Fonction pour les pages qui nécessitent une authentification."""
    ls = LocalStorage()
    if not validate(ls):
        st.stop()
