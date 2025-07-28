import streamlit as st
import plotly.express as px

from src.home_helpers import load_prompts_df, get_weekly_metrics, daily_prompt_series, daily_active_users, weekly_active_users

st.title("Home dashboard (demo CSV)")

df = load_prompts_df()
if df.empty:
    st.error("CSV d’exemple introuvable ou vide.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# T1 — KPIs semaine en cours
# ─────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("T 1.1  •  Semaine en cours")
    metrics = get_weekly_metrics(df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("WAU", metrics["wau"])
    col2.metric("Projets", "N/A")
    col3.metric("Export Liciel", "N/A")
    col4.metric("Prompts", metrics["prompts"])

st.divider()

# ─────────────────────────────────────────────────────────────
# T2 — Prompts par jour (30 j)
# ─────────────────────────────────────────────────────────────
st.subheader("T 2  •  Nombre de prompts (30 derniers jours)")

ser = daily_prompt_series(df)
fig = px.line(
    ser,
    x="date",
    y=["prompts", "failed"],
    labels={"value": "n prompts", "date": "jour", "variable": ""},
)
fig.update_layout(legend=dict(orientation="h", y=-0.25))
st.plotly_chart(fig, use_container_width=True)

st.caption("Données demo issues du CSV local – pas de projets ni d’exports pour l’instant.")

# ─────────────────────────────────────────────────────────────
# T3 — Daily active users (DAU)
# ─────────────────────────────────────────────────────────────
st.subheader("T 3  •  Utilisateurs actifs / jour (30 j)")

dau_df = daily_active_users(df)
fig_dau = px.line(
    dau_df,
    x="date",
    y="dau",
    labels={"dau": "n utilisateurs actifs", "date": "jour"},
)
fig_dau.update_traces(mode="lines+markers")     # petits points
st.plotly_chart(fig_dau, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# T4 — Weekly active users (WAU)
# ─────────────────────────────────────────────────────────────
st.subheader("T 4  •  Utilisateurs actifs / semaine (8 semaines)")

wau_df = weekly_active_users(df)
fig_wau = px.line(
    wau_df,
    x="week",
    y="wau",
    labels={"wau": "n utilisateurs actifs", "week": "semaine (lundi)"},
)
fig_wau.update_traces(mode="lines+markers")
st.plotly_chart(fig_wau, use_container_width=True)