import streamlit as st
from src.auth_guard import require_login 
import plotly.express as px
from zoneinfo import ZoneInfo
import pandas as pd
import datetime as _dt

from src.home_helpers import load_prompts_df
require_login()


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
earliest_dt   = _dt.date(2025, 6, 1)
today_local   = _dt.datetime.now(ZoneInfo("Europe/Paris")).date()
default_start = earliest_dt
default_end   = today_local
calendar_max  = today_local

def _normalize_date_range(
    sel,
    *,
    prev: tuple[_dt.date, _dt.date] | None,
    default_start: _dt.date,
    default_end: _dt.date,
    min_value: _dt.date,
    max_value: _dt.date,
) -> tuple[_dt.date, _dt.date]:
    def _to_date(x):
        if isinstance(x, _dt.datetime):
            return x.date()
        if isinstance(x, _dt.date):
            return x
        return None

    if isinstance(sel, (list, tuple)):
        items = []
        for it in sel:
            if isinstance(it, (list, tuple)):
                items.extend(list(it))
            else:
                items.append(it)
    else:
        items = [sel]

    a = _to_date(items[0]) if len(items) >= 1 else None
    b = _to_date(items[1]) if len(items) >= 2 else None

    if a is None and b is None:
        a, b = default_start, default_end
    elif a is None:
        a = prev[0] if prev else default_start
    elif b is None:
        b = prev[1] if prev else a

    minv = _to_date(min_value) or default_start
    maxv = _to_date(max_value) or default_end

    a = max(minv, min(maxv, _to_date(a) or default_end))
    b = max(minv, min(maxv, _to_date(b) or default_end))

    if b < a:
        a, b = b, a
    return a, b

col1, col2 = st.columns(2)
with col1:
    raw_sel = st.date_input(
        "Période",
        value=(default_start, default_end),
        min_value=default_start,
        max_value=calendar_max,
        format="DD/MM/YYYY",
        key="divers_period",
    )
    start_date, end_date = _normalize_date_range(
        raw_sel,
        prev=st.session_state.get("_divers_period_last"),
        default_start=default_start,
        default_end=default_end,
        min_value=default_start,
        max_value=calendar_max,
    )
    st.session_state["_divers_period_last"] = (start_date, end_date)

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
