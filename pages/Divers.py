import streamlit as st
from src.auth_guard import require_login 
require_login() #
import plotly.express as px
import pandas as pd
import numpy as np
import datetime as _dt

from src.home_helpers import load_prompts_df

st.title("Divers")

# ------------------------------------------------------------------
# Chargement des données brutes
# ------------------------------------------------------------------
df = load_prompts_df()
if df.empty:
    st.error("Aucune donnée disponible.")
    st.stop()

# ------------------------------------------------------------------
# Filtres minimalistes (mêmes valeurs par défaut que Home)
# ------------------------------------------------------------------
earliest_dt  = _dt.date(2025, 6, 1)
default_end  = df["timestamp"].max().date()
default_start = earliest_dt

col1, col2 = st.columns(2)
with col1:
    start_date, end_date = st.date_input(
        "Période",
        value=(default_start, default_end),
        min_value=earliest_dt,
        max_value=default_end,
        format="DD/MM/YYYY",
    )
with col2:
    user_filter = st.text_input("Email contient…")

mask = (
    (df["timestamp"].dt.date >= start_date)
    & (df["timestamp"].dt.date <= end_date)
)
if user_filter:
    mask &= df["email utilisateur"].str.contains(user_filter, case=False, na=False)

filt_df = df[mask].copy()

# ------------------------------------------------------------------
# T 2 bis : Heatmap intensité (jour × heure, 30 jours)
# ------------------------------------------------------------------
st.subheader("T 2 bis • Intensité d’usage (jour × heure)")

# 1. Préparation
if filt_df["timestamp"].dt.tz is None:
    filt_df["timestamp"] = filt_df["timestamp"].dt.tz_localize("Europe/Paris")
else:
    filt_df["timestamp"] = filt_df["timestamp"].dt.tz_convert("Europe/Paris")

filt_df["dow"]  = filt_df["timestamp"].dt.dayofweek   # 0 = lundi
filt_df["hour"] = filt_df["timestamp"].dt.hour

# 2. Pivot (30 derniers jours)
cutoff = pd.Timestamp.now(tz="Europe/Paris") - pd.Timedelta(days=30)
counts = (
    filt_df[filt_df["timestamp"] >= cutoff]
        .groupby(["dow", "hour"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(0, 7), fill_value=0)
        .sort_index()
)

# 3. Labels jour
counts.index = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

# 4. Plotly heatmap
fig_heat = px.imshow(
    counts,
    labels=dict(x="Heure", y="Jour", color="n prompts"),
    aspect="auto",
    color_continuous_scale="Blues",
)
fig_heat.update_xaxes(side="top")
fig_heat.update_layout(margin=dict(l=0, r=0, t=30, b=0))

st.plotly_chart(fig_heat, use_container_width=True)
