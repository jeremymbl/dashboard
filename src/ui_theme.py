import streamlit as st

_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>
html, body, [class*="st-"]  { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4              { letter-spacing: -0.5px; }

section.main > div          { padding-top: 1rem; }

.stApp [data-testid="metric-container"] {
    background: var(--secondary-background-color);
    border-radius: 1rem;
    padding: 1.2rem 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}

div[data-baseweb="table"]   { border-radius: 0.75rem; overflow: hidden; }
.stDataFrameContainer       { border: none; box-shadow: 0 2px 10px rgba(0,0,0,0.3); }

::-webkit-scrollbar         { height: 8px; width: 8px; }
::-webkit-scrollbar-thumb   { background: #444; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #666; }
</style>
"""

def apply_theme() -> None:
    """Injecte CSS custom une seule fois."""
    if "_auditoo_theme" not in st.session_state:
        st.session_state["_auditoo_theme"] = True
        st.markdown(_CSS, unsafe_allow_html=True)
