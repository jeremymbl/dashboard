import streamlit as st
import streamlit_authenticator as stauth
from collections.abc import Mapping

def _to_dict(obj):
    if isinstance(obj, Mapping):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj

creds = _to_dict(st.secrets["credentials"])

# --- DEBUG PRINT 1: Check the credentials dictionary ---
print("\n[DEBUG] Credentials loaded from secrets.toml:")
print(creds)
print("-" * 50)

auth = stauth.Authenticate(
    creds,
    cookie_name="auditoo_dashboard",
    key="super_secret_key",     # 🔒 change en prod
    cookie_expiry_days=30,
)

# ---------- Login (API ≥ 0.3) ----------
# This single call renders the login form in the main page area
# and updates session_state with the authentication status.
auth.login(
    "main",
    fields={"Form name": "🔐 Connexion"},
    key="auditoo_login",
)

# --- DEBUG PRINT 2: Check the session state after the login attempt ---
print("\n[DEBUG] Session state after login attempt:")
print(st.session_state)
print("-" * 50)

# ---------- Post-login ----------
# This block checks the session_state populated by the auth.login() call.
if st.session_state.get("authentication_status"):
    st.session_state["user"] = st.session_state.get("name") or "Utilisateur"
    st.switch_page("pages/Home.py")
elif st.session_state.get("authentication_status") is False:
    st.error("Login ou mot de passe incorrect.")
elif st.session_state.get("authentication_status") is None:
    st.info("Entrez vos identifiants.")