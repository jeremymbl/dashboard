import streamlit as st
from src.auth_guard import require_login

require_login()
import plotly.express as px
import pandas as pd
import numpy as np
import datetime as _dt 
from streamlit import column_config as cc

from src.data_sources import fetch_logfire_events
from src.home_helpers import (
    load_prompts_df,
    get_weekly_metrics,
    daily_prompt_series,
    daily_active_users,
    weekly_active_users,
    weekly_liciel_exports,
)

st.title("Home dashboard (données live logfire)")

df = load_prompts_df()

# ————————————————————————————————
# Toggle global dans la barre latérale
# ————————————————————————————————

exclude_test = st.sidebar.toggle(
    "Exclure données de test (test@test.com et @auditoo.eco)",
    value=True,
)

if exclude_test:
    emails = df["email utilisateur"].str.lower().fillna("")
    mask_excl = emails.eq("test@test.com") | emails.str.endswith("@auditoo.eco")
    df = df[~mask_excl]

# ─────────────────────────────────────────────────────────────
# T1 — KPIs semaine en cours
# ─────────────────────────────────────────────────────────────
monday     = df["timestamp"].max().normalize() - pd.Timedelta(days=df["timestamp"].max().weekday())
last_monday = monday - pd.Timedelta(days=7)
last_sunday = monday - pd.Timedelta(seconds=1)

def kpis_between(start, end):
    period = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    return {
        "wau":       period["email utilisateur"].nunique(),
        "prompts":   len(period),
        "projects":  period["id projet"].nunique(),
        # valeur provisoire, on l’écrasera juste après
        "exports":   0,
    }

curr = kpis_between(monday, df["timestamp"].max())
prev = kpis_between(last_monday, last_sunday)

exp_week = (
    weekly_liciel_exports(df)          # ‹week, exports›
      .set_index("week")["exports"]    # Series indexée par lundi
)

curr["exports"] = exp_week.get(monday.date(), 0)
prev["exports"] = exp_week.get(last_monday.date(), 0)

# — panneau semaine en cours
col_left, col_right = st.columns(2)

with col_left:
    with st.container(border=True):
        st.subheader("T 1.1  •  Semaine en cours")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("WAU", curr["wau"])
        c2.metric("Projets", curr["projects"])
        c3.metric("Export Liciel", curr["exports"])
        c4.metric("Prompts", curr["prompts"])
        st.caption("Depuis lundi")

with col_right:
    with st.container(border=True):
        st.subheader("T 1.2  •  Semaine dernière")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("WAU", prev["wau"])
        c2.metric("Projets", prev["projects"])
        c3.metric("Export Liciel", prev["exports"])
        c4.metric("Prompts", prev["prompts"])
        st.caption("Lundi-Dimanche précédents")

st.divider()

# ─────────────────────────────────────────────────────────────
# T2 — Prompts par jour (30 j)
# ─────────────────────────────────────────────────────────────
# — palette Figma —
BLUE = "#1f77b4"
RED  = "#e74c3c"

st.subheader("T 2  •  Nombre de prompts (historique complet)")

# 1. DataFrame « success / failed »
ser = daily_prompt_series(df)   # colonnes: date, success, failed

# 2. Passage au format « long »
stack_df = (
    ser.melt(id_vars="date", value_vars=["success", "failed"],
             var_name="statut", value_name="count")
        .replace({"statut": {"success": "succès", "failed": "échec"}})
)

# 3. Graphe barres groupées
fig = px.bar(
    stack_df, x="date", y="count", color="statut",
    barmode="group",
    color_discrete_map={"succès": BLUE, "échec": RED},
    labels={"count": "n prompts", "date": "jour", "statut": ""}
)
fig.update_layout(
    legend=dict(orientation="h", y=-0.25),
)
fig.update_xaxes(
    dtick="D1",               # ✅ un tick par jour
    tickformat="%d %b",       # ex. « 25 jun »
    tickangle=-45             # labels inclinés
)
st.plotly_chart(fig, use_container_width=True)

st.caption("Données live Logfire – sections Projets / Exports arriveront lorsque les tables seront disponibles.")


# ─────────────────────────────────────────────────────────────
# T3 — Daily active users (DAU)
# ─────────────────────────────────────────────────────────────
st.subheader("T 3  •  Utilisateurs actifs / jour (historique complet)")

dau_df = daily_active_users(df)
fig_dau = px.line(
    dau_df,
    x="date",
    y="dau",
    labels={"dau": "n utilisateurs actifs", "date": "jour"},
)

fig_dau.update_traces(mode="lines+markers")
fig_dau.update_yaxes(dtick=1, tickformat=".0f", rangemode="tozero")
fig_dau.update_xaxes(
    dtick="D1",
    tickformat="%d %b",
    tickangle=-45
)

st.plotly_chart(fig_dau, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# T4 — Weekly active users (WAU)
# ─────────────────────────────────────────────────────────────
st.subheader("T 4  •  Utilisateurs actifs / semaine (historique complet)")

wau_df = weekly_active_users(df)
fig_wau = px.line(
    wau_df,
    x="week",
    y="wau",
    labels={"wau": "n utilisateurs actifs", "week": "semaine (lundi)"},
)
fig_wau.update_traces(mode="lines+markers")
fig_wau.update_xaxes(tickformat="%d %b")

st.plotly_chart(fig_wau, use_container_width=True)


# ------------------------------------------------------------------
# 1. Source de données
# ------------------------------------------------------------------
today       = _dt.datetime.utcnow().date()
earliest_dt = _dt.date(2025, 6, 1)
days_back   = (today - earliest_dt).days + 1

raw = fetch_logfire_events(lookback_days=days_back, limit=20000)

if not raw:
    st.error("Aucune donnée disponible depuis Logfire.")
    st.stop()
st.caption("Données live Logfire")
df = pd.DataFrame(raw)
df["timestamp"] = pd.to_datetime(df["timestamp"],
                                 format="%d/%m/%Y %H:%M:%S",
                                 errors="coerce")

# ------------------------------------------------------------------
# 2. Filtres globaux
# ------------------------------------------------------------------
st.title("Prompts")

col1, col2, col3, col4 = st.columns(4)

default_end   = df["timestamp"].max().date()
default_start = earliest_dt

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
mask = (
    (df["timestamp"].dt.date >= start_date) &
    (df["timestamp"].dt.date <= end_date)
)

if user_filter:
    mask &= df["email utilisateur"].str.contains(user_filter, case=False, na=False)
if scope_filter != "Tous":
    mask &= df["scope"] == scope_filter
if status_filter != "Tous":
    mask &= df["statut"] == status_filter

filt_df = df[mask].copy()

if exclude_test:
    # retire toutes les lignes dont l’email est "test@test.com"
    # ou se termine par "@auditoo.eco" (insensible à la casse)
    emails     = filt_df["email utilisateur"].str.lower().fillna("")
    mask_excl  = emails.eq("test@test.com") | emails.str.endswith("@auditoo.eco")
    filt_df    = filt_df[~mask_excl]

# ------------------------------------------------------------------
# T5 – Tableau complet
# ------------------------------------------------------------------
st.subheader(f"T5  •  Tous les prompts ({len(filt_df)})")

# ── 1. Préparation du DataFrame ──────────────────────────────────────────────
COLS = [
    "timestamp",
    "email utilisateur",
    "prompt",
    "scope",
    "duree traitement",
    "statut",
    "trace_id",
    "id projet",
]

BASE_LOGFIRE = st.secrets["LOGFIRE_PROJECT_URL"].rstrip("/")

def make_logfire_url(tid: str, window: str = "30d") -> str:
    """
    Construit le lien Logfire à partir d'un trace_id.
    """
    if pd.isna(tid) or not tid:
        return ""
    return (
        f"{BASE_LOGFIRE}"
        f"?q=trace_id%3D%27{tid}%27"     # filtre URL-encodé
        f"&traceId={tid}"                # trace pré-sélectionnée
        f"&last={window}"                # fenêtre temporelle
    )

df_table = (
    filt_df[COLS]
        .assign(
            trace_url=lambda d: d["trace_id"].apply(make_logfire_url),
            trace_short=lambda d: d["trace_id"].str.slice(0, 8) + "…",
        )
        .sort_values("timestamp", ascending=False)
)

st.dataframe(
    df_table,
    column_order=[
        "statut","timestamp", "email utilisateur", "prompt", "scope",
        "duree traitement", 
        "trace_short",   # affiché au lieu du long ID
        "trace_url",     # la vraie URL (colonne lien)
        "id projet",
    ],
    column_config={
        "trace_short": cc.TextColumn(width="small", label="trace_id"),
        "trace_url":   cc.LinkColumn(display_text="ouvrir"),
    },
    hide_index=True,
    use_container_width=True,
    height=700,
)


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

st.caption("Données live Logfire — sections Projets / Exports arriveront lorsque les tables seront disponibles.")

# ------------------------------------------------------------------
# T8 – Prompts par projet
# ------------------------------------------------------------------
st.divider()
st.subheader("T8  •  Prompts par projets")

def _fmt_timedelta(delta: pd.Timedelta) -> str:
    """dd → '6 j 4 h 23 min'  /  hh:mm → '2 h 11 min'  /  <1h → '15 min'."""
    total = int(delta.total_seconds())
    days, rem = divmod(total, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, _ = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days} j")
    if hours:
        parts.append(f"{hours} h")
    parts.append(f"{minutes} min")
    return " ".join(parts)

if "id projet" not in filt_df.columns or filt_df["id projet"].isna().all():
    st.info("Pas encore de données projet dans le CSV — attendons Supabase/Logfire.")
else:
    proj = (
        filt_df
        .groupby("id projet")
        .agg(
            date_premier_prompt=("timestamp", "min"),
            date_dernier_prompt=("timestamp", "max"),
            user=("email utilisateur", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
            nb_prompts=("prompt", "count"),
        )
        .reset_index()
    )
    proj["durée projet"] = (proj["date_dernier_prompt"] - proj["date_premier_prompt"]).apply(_fmt_timedelta)
    proj = proj.sort_values("date_dernier_prompt", ascending=False)
    st.dataframe(
        proj[
            [
                "user",
                "date_premier_prompt",
                "date_dernier_prompt",
                "durée projet",
                "nb_prompts",
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
st.subheader("T9  •  Prompts par utilisateur – semaines glissantes")

# Slider de profondeur
nb_semaines = st.slider(
    "Nombre de semaines à afficher",
    min_value=1,
    max_value=12,
    value=4,
    step=1,
    help="Fenêtre glissante depuis la semaine courante (lundi).",
)

latest_monday = now.normalize() - pd.Timedelta(days=now.weekday())
cutoff        = latest_monday - pd.Timedelta(weeks=nb_semaines)
last_df       = filt_df[filt_df["timestamp"] >= cutoff].copy()

if last_df.empty:
    st.info("Aucun prompt sur les dernières semaines sélectionnées.")
else:
    # index de semaine (0 = semaine courante, 1 = -1, etc.)
    # 1. Lundi de la semaine du prompt
    monday_of_ts = (
    last_df["timestamp"].dt.normalize()
    - pd.to_timedelta(last_df["timestamp"].dt.weekday, unit="d")
    )

    # 2. Index semaine (0 = en cours, 1 = précéd.)
    last_df["week_idx"] = (
    (latest_monday - monday_of_ts).dt.days // 7
    ).astype(int)


    pivot = (
        last_df
        .groupby(["email utilisateur", "week_idx"])
        .size()                            # ↩︎ compte même si « prompt » NaN
        .unstack(fill_value=0)
        .reindex(columns=range(0, nb_semaines), fill_value=0)
        .sort_index(axis=1)
    )

    pivot.columns = [f"Semaine -{i}" for i in pivot.columns]
    st.dataframe(pivot, use_container_width=True)
    st.caption("Clique l’icône ↗︎ pour ouvrir le tableau plein écran.")


# ------------------------------------------------------------------
# T8 bis – Projets par utilisateur : semaines glissantes
# ------------------------------------------------------------------
st.divider()
st.subheader("T8 bis  •  Projets par utilisateur – semaines glissantes")

# Slider indépendant pour T8 bis
nb_semaines_proj = st.slider(
    "Nombre de semaines à afficher (T8 bis)",
    min_value=1,
    max_value=12,
    value=4,
    step=1,
    help="Fenêtre glissante depuis la semaine courante (lundi).",
)

if filt_df["id projet"].isna().all():
    st.info("Pas encore de données projet dans le CSV — attendons Supabase/Logfire.")
else:
    latest_monday = now.normalize() - pd.Timedelta(days=now.weekday())
    cutoff        = latest_monday - pd.Timedelta(weeks=nb_semaines_proj)
    last_proj_df  = filt_df[
        (filt_df["timestamp"] >= cutoff) &
        (~filt_df["id projet"].isna())
    ].copy()

    if last_proj_df.empty:
        st.info("Aucun projet sur les dernières semaines sélectionnées.")
    else:
        monday_of_ts = (
            last_proj_df["timestamp"].dt.normalize()
            - pd.to_timedelta(last_proj_df["timestamp"].dt.weekday, unit="d")
        )
        last_proj_df["week_idx"] = (
            (latest_monday - monday_of_ts).dt.days // 7
        ).astype(int)

        pivot_proj = (
            last_proj_df
            .groupby(["email utilisateur", "week_idx"])["id projet"]
            .nunique()
            .unstack(fill_value=0)
            .reindex(columns=range(0, nb_semaines_proj), fill_value=0)
            .sort_index(axis=1)
        )

        pivot_proj.columns = [f"Semaine -{i}" for i in pivot_proj.columns]
        st.dataframe(pivot_proj, use_container_width=True)
        st.caption("Clique l’icône ↗︎ pour ouvrir le tableau plein écran.")


# ------------------------------------------------------------------
# T10 – Exports Liciel par utilisateur : semaines glissantes
# ------------------------------------------------------------------
st.divider()
st.subheader("T10  •  Exports Liciel par utilisateur – semaines glissantes")

# Slider indépendant pour T10
nb_semaines_exp = st.slider(
    "Nombre de semaines à afficher (T10)",
    min_value=1,
    max_value=12,
    value=4,
    step=1,
    help="Fenêtre glissante depuis la semaine courante (lundi).",
)

# 1. Filtre « exports Liciel »
_LICIEL_ROUTE = r"GET /project/.+/liciel"
liciel_df = filt_df[filt_df["span_name"].str.contains(_LICIEL_ROUTE, na=False, regex=True)].copy()

if liciel_df.empty:
    st.info("Aucun export Liciel trouvé sur la période sélectionnée.")
else:
    # 2. Semaine ISO (lundi) du timestamp
    monday_of_ts = (
        liciel_df["timestamp"].dt.normalize()
        - pd.to_timedelta(liciel_df["timestamp"].dt.weekday, unit="d")
    )
    latest_monday = now.normalize() - pd.Timedelta(days=now.weekday())

    # 3. Index semaine glissante (0 = semaine courante, 1 = précédente…)
    liciel_df["week_idx"] = ((latest_monday - monday_of_ts).dt.days // 7).astype(int)

    # 4. Pivot : lignes = e-mail, colonnes = Semaine -i, valeur = nb exports
    pivot_exp = (
        liciel_df
        .groupby(["email utilisateur", "week_idx"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(0, nb_semaines_exp), fill_value=0)
        .sort_index(axis=1)
    )
    pivot_exp.columns = [f"Semaine -{i}" for i in pivot_exp.columns]

    # 5. Affichage
    st.dataframe(pivot_exp, use_container_width=True)
    st.caption("Exports Liciel comptés quand `span_name` contient « GET /project/*/liciel ».")
