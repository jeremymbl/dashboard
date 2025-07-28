import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ------------------------------------------------------------------
# 1. Source de données : API quand dispo, CSV local sinon
# ------------------------------------------------------------------
try:
    from src.data_sources import fetch_logfire_events      # prod (token)
except (ImportError, KeyError):
    from src.data_sources_csv import fetch_logfire_events  # fallback

raw = fetch_logfire_events(limit=5_000)
if not raw:
    st.error("Aucune donnée disponible (CSV manquant ou vide).")
    st.stop()

df = pd.DataFrame(raw)

# ------------------------------------------------------------------
# 2. Filtres globaux
# ------------------------------------------------------------------
st.title("Prompts")

col1, col2, col3, col4 = st.columns(4)

with col1:
    days = st.slider("Période (j)", 1, 30, 30)

with col2:
    user_filter = st.text_input("Email contient…")

with col3:
    scope_filter = st.selectbox(
        "Scope",
        options=["Tous"] + sorted(df["scope"].dropna().unique().tolist()),
        index=0,
    )

with col4:
    status_filter = st.selectbox("Statut", ["Tous", "Succès", "Échec"], 0)

# ------------------------------------------------------------------
# 3. Application des filtres
# ------------------------------------------------------------------
now = pd.Timestamp.utcnow().tz_localize(None)
mask = df["timestamp"] >= (now - pd.Timedelta(days=days))

if user_filter:
    mask &= df["email utilisateur"].str.contains(user_filter, case=False, na=False)
if scope_filter != "Tous":
    mask &= df["scope"] == scope_filter
if status_filter != "Tous":
    mask &= df["statut"] == status_filter

filt_df = df[mask].copy()

# ------------------------------------------------------------------
# T5 – Tableau complet
# ------------------------------------------------------------------
st.subheader(f"T5  •  Tous les prompts ({len(filt_df)})")
st.dataframe(
    filt_df[
        [
            "timestamp",
            "email utilisateur",
            "prompt",
            "scope",
            "duree traitement",
            "statut",
            "trace_id",
            "id projet",
        ]
    ].sort_values("timestamp", ascending=False),
    use_container_width=True,
)

st.divider()

# ------------------------------------------------------------------
# T6 – Prompts lents (> 7 s)  +  T7 – Camembert temps de réponse
# ------------------------------------------------------------------
slow_df = filt_df[filt_df["duree traitement"].gt(7)]
left, right = st.columns([2, 1])

with left:
    st.subheader(f"T6  •  Prompts > 7 s ({len(slow_df)})")
    st.dataframe(
        slow_df[
            [
                "timestamp",
                "email utilisateur",
                "prompt",
                "scope",
                "duree traitement",
                "statut",
            ]
        ].sort_values("timestamp", ascending=False),
        height=300,
        use_container_width=True,
    )

with right:
    st.subheader("T7  •  Répartition par temps")
    bins = [0, 3, 7, np.inf]
    labels = ["0–3 s", "3–7 s", "7 s et +"]
    cat = pd.cut(filt_df["duree traitement"], bins=bins, labels=labels, right=False)
    pie = cat.value_counts().reindex(labels, fill_value=0).reset_index()
    pie.columns = ["tranche", "count"]
    color_map = {
    "0–3 s":  "#2ecc71",   # vert
    "3–7 s": "#f5b041",    # orange
    "7 s et +": "#e74c3c", # rouge
    }

    fig = px.pie(
    pie,
    names="tranche",
    values="count",
    hole=0.4,
    color="tranche",
    color_discrete_map=color_map,
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption("Données issues du CSV local — sections Projets / Exports arriveront lorsque les tables seront disponibles.")

# ------------------------------------------------------------------
# T8 – Prompts par projet
# ------------------------------------------------------------------
st.divider()
st.subheader("T8  •  Prompts par projets")

if "id projet" not in filt_df.columns or filt_df["id projet"].isna().all():
    st.info("Pas encore de données projet dans le CSV — attendons Supabase/Logfire.")
else:
    proj = (
        filt_df.assign(date=filt_df["timestamp"].dt.date)
        .groupby("id projet")
        .agg(
            date_crea=("date", "min"),
            user=("email utilisateur", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
            nb_prompts=("prompt", "count"),
            first_ts=("timestamp", "min"),
            last_ts=("timestamp", "max"),
        )
        .reset_index()
    )
    proj["durée (j)"] = (proj["last_ts"] - proj["first_ts"]).dt.days
    proj = proj.sort_values("last_ts", ascending=False)

    st.dataframe(
        proj[
            [
                "date_crea",
                "user",
                "nb_prompts",
                "durée (j)",
                "id projet",
            ]
        ],
        use_container_width=True,
    )
    st.caption("Colonnes Excel/Liciel à venir quand les logs seront branchés.")


# ------------------------------------------------------------------
# T9 – Prompts par utilisateur : 4 semaines glissantes
# ------------------------------------------------------------------
st.divider()
st.subheader("T9  •  Prompts par utilisateur – 4 semaines glissantes")

latest_monday = now.normalize() - pd.Timedelta(days=now.weekday())
cutoff        = latest_monday - pd.Timedelta(days=28)      # 4 semaines
last4_df      = filt_df[filt_df["timestamp"] >= cutoff].copy()

if last4_df.empty:
    st.info("Aucun prompt sur les 4 semaines glissantes sélectionnées.")
else:
    # index de semaine (0 = semaine courante, 1 = -1, etc.)
    last4_df["week_idx"] = (
        (latest_monday - last4_df["timestamp"].dt.normalize()) //
        pd.Timedelta(days=7)
    ).astype(int)

    pivot = (
        last4_df
        .groupby(["email utilisateur", "week_idx"])["prompt"]
        .count()
        .unstack(fill_value=0)                 # lignes : user, colonnes : week_idx
        .reindex(columns=range(0, 4), fill_value=0)  # force les semaines 0-3
        .sort_index(axis=1)                    # 0 | 1 | 2 | 3
    )

    pivot.columns = [f"Semaine -{i}" for i in pivot.columns]   # jolis entêtes
    st.dataframe(pivot, use_container_width=True)
    st.caption("Clique l’icône ↗︎ pour ouvrir le tableau plein écran.")