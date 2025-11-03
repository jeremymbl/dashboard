import streamlit as st
from src.auth_guard import require_login
import plotly.express as px
import pandas as pd
import numpy as np
import datetime as _dt
from streamlit import column_config as cc
from zoneinfo import ZoneInfo
from src.data_sources import clear_cache, fetch_aggregated_dashboard_data, fetch_weekly_project_counts, fetch_weekly_active_users
from src.home_helpers import load_prompts_df, DASHBOARD_ROW_LIMIT

require_login()



def create_dynamic_title(base_title: str, df_view: pd.DataFrame, original_df: pd.DataFrame) -> str:
    """Creates a descriptive title for a table based on its data."""
    if df_view.empty:
        return f"{base_title} (0)"

    count = len(df_view)
    min_date = df_view["timestamp"].min().strftime('%d %b')
    max_date = df_view["timestamp"].max().strftime('%d %b')
    date_range_str = f"du {min_date} au {max_date}" if min_date != max_date else f"le {min_date}"

    limit_hit = len(original_df) == DASHBOARD_ROW_LIMIT
    if limit_hit:
        return f"{base_title} ({count} plus récents {date_range_str}, limite atteinte)"
    else:
        return f"{base_title} ({count} {date_range_str})"


st.title("Home dashboard (données live logfire)")

# --- NEW: Fetch aggregated data for the dashboard ---
# This is fast and powers T1 through T4.
agg_df = fetch_aggregated_dashboard_data(lookback_days=90)  # Load 90 days for historical charts

# --- NEW: Fetch weekly project counts and WAU for the T1 KPI ---
project_counts_df = fetch_weekly_project_counts(lookback_days=90)
wau_df = fetch_weekly_active_users(lookback_days=90)
# --- END NEW ---

# ————————————————————————————————
# Toggle global dans la barre latérale (partagé avec Images.py)
# ————————————————————————————————

# Initialize the toggle's state in session_state if it's not already present
if 'exclude_test' not in st.session_state:
    st.session_state.exclude_test = True

# Create the toggle. The `key` will now manage its state entirely.
# The `value` parameter is removed to resolve the warning.
st.sidebar.toggle(
    "Exclure données de test (test@test.com et @auditoo.eco)",
    key='exclude_test'
)

# The widget's state is already bound to st.session_state.exclude_test,
# so we just use that directly.
exclude_test = st.session_state.exclude_test

# ─────────────────────────────────────────────────────────────
# T1 — KPIs semaine en cours (Refactored to include project counts)
# ─────────────────────────────────────────────────────────────
today = pd.Timestamp.now(tz="Europe/Paris").date()
monday = today - pd.Timedelta(days=today.weekday())
last_monday = monday - pd.Timedelta(days=7)

# Helper functions to safely get values from aggregated dataframes
def get_project_count_for_week(df, week_start_date):
    if df.empty or 'week_start_date' not in df.columns:
        return 0
    # Ensure date types match for comparison
    if not isinstance(week_start_date, _dt.date):
        week_start_date = week_start_date.date()

    row = df[df['week_start_date'] == week_start_date]
    return row['unique_project_count'].iloc[0] if not row.empty else 0

def get_wau_for_week(df, week_start_date):
    if df.empty or 'week_start_date' not in df.columns:
        return 0
    # Ensure date types match for comparison
    if not isinstance(week_start_date, _dt.date):
        week_start_date = week_start_date.date()

    row = df[df['week_start_date'] == week_start_date]
    return int(row['wau'].iloc[0]) if not row.empty else 0

# Current Week KPIs from agg_df
curr = {"peak_dau": 0, "wau": 0, "prompts": 0, "exports": 0}
if not agg_df.empty:
    curr_week_df = agg_df[agg_df['date'] >= monday]
    curr['peak_dau'] = curr_week_df['dau'].max() if not curr_week_df.empty else 0
    curr['prompts'] = curr_week_df['total_prompts'].sum()
    curr['exports'] = curr_week_df['liciel_exports'].sum()

# Previous Week KPIs from agg_df
prev = {"peak_dau": 0, "wau": 0, "prompts": 0, "exports": 0}
if not agg_df.empty:
    prev_week_df = agg_df[(agg_df['date'] >= last_monday) & (agg_df['date'] < monday)]
    prev['peak_dau'] = prev_week_df['dau'].max() if not prev_week_df.empty else 0
    prev['prompts'] = prev_week_df['total_prompts'].sum()
    prev['exports'] = prev_week_df['liciel_exports'].sum()

# Add the accurate counts from dedicated aggregation queries (excluding test users)
curr["projects"] = get_project_count_for_week(project_counts_df, monday)
prev["projects"] = get_project_count_for_week(project_counts_df, last_monday)
curr["wau"] = get_wau_for_week(wau_df, monday)
prev["wau"] = get_wau_for_week(wau_df, last_monday)

# — panneau semaine en cours
col_left, col_right = st.columns(2)

with col_left:
    with st.container(border=True):
        st.subheader("T 1.1  •  Semaine en cours")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("WAU", curr["wau"], help="Utilisateurs uniques actifs cette semaine (sans test)")
        c2.metric("Pic DAU", curr["peak_dau"], help="Pic d'utilisateurs quotidiens")
        c3.metric("Projets", curr["projects"], help="Projets uniques cette semaine (sans test)")
        c4.metric("Exports", curr["exports"], help="Exports Liciel")
        c5.metric("Prompts", curr["prompts"], help="Total prompts")
        st.caption("Depuis lundi")

with col_right:
    with st.container(border=True):
        st.subheader("T 1.2  •  Semaine dernière")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("WAU", prev["wau"], help="Utilisateurs uniques actifs cette semaine (sans test)")
        c2.metric("Pic DAU", prev["peak_dau"], help="Pic d'utilisateurs quotidiens")
        c3.metric("Projets", prev["projects"], help="Projets uniques cette semaine (sans test)")
        c4.metric("Exports", prev["exports"], help="Exports Liciel")
        c5.metric("Prompts", prev["prompts"], help="Total prompts")
        st.caption("Lundi-Dimanche précédents")

st.divider()

# ─────────────────────────────────────────────────────────────
# T2 — Prompts par jour (Refactored to use agg_df)
# ─────────────────────────────────────────────────────────────
# — palette Figma —
BLUE = "#1f77b4"
RED  = "#e74c3c"

st.subheader("T 2  •  Nombre de prompts (historique complet)")

if not agg_df.empty:
    # 1. Rename columns for melting
    plot_df = agg_df[['date', 'successful_prompts', 'failed_prompts']].rename(columns={
        'successful_prompts': 'succès',
        'failed_prompts': 'échec'
    })

    # 2. Passage au format « long »
    stack_df = plot_df.melt(id_vars="date", value_vars=["succès", "échec"],
                 var_name="statut", value_name="count")

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
else:
    st.info("No aggregated prompt data available.")

st.caption("Données live Logfire – sections Projets / Exports arriveront lorsque les tables seront disponibles.")


# ─────────────────────────────────────────────────────────────
# T3 — Daily active users (DAU) (Refactored to use agg_df)
# ─────────────────────────────────────────────────────────────
st.subheader("T 3  •  Utilisateurs actifs / jour (historique complet)")

if not agg_df.empty:
    fig_dau = px.line(
        agg_df,
        x="date",
        y="dau",
        labels={"dau": "n utilisateurs actifs", "date": "jour"},
    )

    fig_dau.update_traces(mode="lines+markers")
    fig_dau.update_yaxes(dtick=1, tickformat=".0f", range=[0, None])
    fig_dau.update_xaxes(
        dtick="D1",
        tickformat="%d %b",
        tickangle=-45
    )

    st.plotly_chart(fig_dau, use_container_width=True)
else:
    st.info("No aggregated user data available.")

# ─────────────────────────────────────────────────────────────
# T4 — Weekly active users (WAU) (Refactored to use agg_df)
# ─────────────────────────────────────────────────────────────
st.subheader("T 4  •  Utilisateurs actifs / semaine (historique complet)")

if not agg_df.empty:
    wau_df = agg_df.copy()
    wau_df['week'] = pd.to_datetime(wau_df['date']).dt.to_period('W-MON').apply(lambda p: p.start_time.date())
    weekly_summary = wau_df.groupby('week').agg(wau=('dau', 'max')).reset_index()

    fig_wau = px.line(
        weekly_summary,
        x="week",
        y="wau",
        labels={"wau": "n utilisateurs actifs", "week": "semaine (lundi)"},
    )
    fig_wau.update_traces(mode="lines+markers")
    fig_wau.update_yaxes(dtick=1, tickformat=".0f", range=[0, None])
    fig_wau.update_xaxes(tickformat="%d %b")

    st.plotly_chart(fig_wau, use_container_width=True)
else:
    st.info("No aggregated user data available.")

st.divider()

# ------------------------------------------------------------------
# Expander for slow data and detailed tables
# ------------------------------------------------------------------

with st.expander("🔍 Afficher les tableaux de données détaillées et les filtres", expanded=False):

    # Load slow data only when the expander is opened
    with st.spinner("Chargement des données détaillées..."):
        # Default to 7 days for initial load
        initial_lookback_days = 7
        df = load_prompts_df(lookback_days=initial_lookback_days)

    # Apply the exclude_test filter to df
    if exclude_test:
        emails = df["email utilisateur"].str.lower().fillna("")
        mask_excl = emails.eq("test@test.com") | emails.str.endswith("@auditoo.eco")
        df = df[~mask_excl]

    # ------------------------------------------------------------------
    # Filtres globaux
    # ------------------------------------------------------------------

    col_refresh, col_title = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Actualiser", help="Force le rafraîchissement des données (ignore le cache)"):
            clear_cache()
            st.rerun()

    with col_title:
        st.title("Prompts")

    col1, col2, col3, col4 = st.columns(4)

    today_local   = _dt.datetime.now(ZoneInfo("Europe/Paris")).date()
    default_start = today_local - _dt.timedelta(days=6)  # 7-day range (6 days back + today = 7 days)
    default_end   = today_local                 # 👈 ancre l'ouverture sur le mois courant
    calendar_max  = today_local                 # borne sup = aujourd'hui
    calendar_min  = today_local - _dt.timedelta(days=180)  # 6 months historical data

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

        # Aplatir une éventuelle structure imbriquée : ((date,date),) → [date, date]
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

        # Remplissage des bornes manquantes
        if a is None and b is None:
            a, b = default_start, default_end
        elif a is None:
            a = prev[0] if prev else default_start
        elif b is None:
            b = prev[1] if prev else a

        # Clamp dans [min_value, max_value] avec coercition
        minv = _to_date(min_value) or default_start
        maxv = _to_date(max_value) or default_end

        a = max(minv, min(maxv, _to_date(a) or default_end))
        b = max(minv, min(maxv, _to_date(b) or default_end))

        # Réordonner si nécessaire
        if b < a:
            a, b = b, a
        return a, b

    with col1:
        raw_sel = st.date_input(
            "Période",
            value=(default_start, default_end),
            min_value=calendar_min,
            max_value=calendar_max,
            format="DD/MM/YYYY",
            key="home_period",
        )
        start_date, end_date = _normalize_date_range(
            raw_sel,
            prev=st.session_state.get("_home_period_last"),
            default_start=default_start,
            default_end=default_end,
            min_value=calendar_min,
            max_value=calendar_max,
        )
        st.session_state["_home_period_last"] = (start_date, end_date)

        # ——— Dynamic data loading: detect if we need more historical data ———
        required_lookback_days = (today_local - start_date).days + 1

        if required_lookback_days > initial_lookback_days:
            # User selected dates outside current range - need to reload with more data
            with st.spinner(f"⏳ Chargement de {required_lookback_days} jours d'historique en cours..."):
                df = load_prompts_df(lookback_days=required_lookback_days)
                # Re-apply the exclude_test filter after reloading
                if exclude_test:
                    emails = df["email utilisateur"].str.lower().fillna("")
                    mask_excl = emails.eq("test@test.com") | emails.str.endswith("@auditoo.eco")
                    df = df[~mask_excl]

        # Optionnel : feedback si l'utilisateur n'a fixé qu'une borne
        if isinstance(raw_sel, _dt.date):
            st.info("Plage incomplète détectée : j'ai fixé une plage d'un jour.", icon="ℹ️")

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
        # retire toutes les lignes dont l'email est "test@test.com"
        # ou se termine par "@auditoo.eco" (insensible à la casse)
        emails     = filt_df["email utilisateur"].str.lower().fillna("")
        mask_excl  = emails.eq("test@test.com") | emails.str.endswith("@auditoo.eco")
        filt_df    = filt_df[~mask_excl]

    # ------------------------------------------------------------------
    # T5 – Tableau complet
    # ------------------------------------------------------------------
    # Filtrer pour ne garder que les vrais prompts (POST message), pas les exports Liciel
    _MESSAGE_ROUTE = r"POST /projects/.+/(message|prompts/chat)"  # Matches both old and new prompt routes
    prompts_only_df = filt_df[filt_df["span_name"].str.contains(_MESSAGE_ROUTE, na=False, regex=True)].copy()

    st.subheader(create_dynamic_title("T5  •  Tous les prompts", prompts_only_df, df))

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
        prompts_only_df[COLS]
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
    # T5 bis – Tableau des exports Liciel
    # ------------------------------------------------------------------
    # Filtrer pour ne garder que les exports Liciel
    _LICIEL_ROUTE = r"GET /projects/.+/liciel"  # Corrigé: projects au pluriel
    exports_only_df = filt_df[filt_df["span_name"].str.contains(_LICIEL_ROUTE, na=False, regex=True)].copy()

    st.subheader(create_dynamic_title("T5 bis  •  Tous les exports Liciel", exports_only_df, df))

    if exports_only_df.empty:
        st.info("Aucun export Liciel trouvé sur la période sélectionnée.")
    else:
        # Colonnes spécifiques aux exports (pas de prompt ni scope)
        EXPORT_COLS = [
            "timestamp",
            "email utilisateur",
            "duree traitement",
            "statut",
            "trace_id",
            "id projet",
        ]

        df_exports_table = (
            exports_only_df[EXPORT_COLS]
                .assign(
                    trace_url=lambda d: d["trace_id"].apply(make_logfire_url),
                    trace_short=lambda d: d["trace_id"].str.slice(0, 8) + "…",
                )
                .sort_values("timestamp", ascending=False)
        )

        st.dataframe(
            df_exports_table,
            column_order=[
                "statut", "timestamp", "email utilisateur",
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
            height=400,  # Plus petit que T5 car généralement moins d'exports
        )

    st.divider()

    # ------------------------------------------------------------------
    # T6 – Prompts lents (> 7 s)  +  T7 – Camembert temps de réponse
    # ------------------------------------------------------------------
    slow_df = prompts_only_df[prompts_only_df["duree traitement"].gt(7)]
    left, right = st.columns([2, 1])

    with left:
        st.subheader(create_dynamic_title("T6  •  Prompts > 7 s", slow_df, df))
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
        if len(df) == DASHBOARD_ROW_LIMIT:
            st.caption("_(Basé sur les données limitées affichées dans T5)_")
        bins = [0, 3, 7, np.inf]
        labels = ["0–3 s", "3–7 s", "7 s et +"]
        cat = pd.cut(prompts_only_df["duree traitement"], bins=bins, labels=labels, right=False)
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
    st.subheader(create_dynamic_title("T8  •  Prompts par projets", prompts_only_df, df))

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

    if "id projet" not in prompts_only_df.columns or prompts_only_df["id projet"].isna().all():
        st.info("Pas encore de données projet dans le CSV — attendons Supabase/Logfire.")
    else:
        proj = (
            prompts_only_df
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
    last_df       = prompts_only_df[prompts_only_df["timestamp"] >= cutoff].copy()

    st.subheader(create_dynamic_title("T9  •  Prompts par utilisateur – semaines glissantes", last_df, df))

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
        st.caption("Clique l'icône ↗︎ pour ouvrir le tableau plein écran.")


    # ------------------------------------------------------------------
    # T8 bis – Projets par utilisateur : semaines glissantes
    # ------------------------------------------------------------------
    st.divider()

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
        st.subheader("T8 bis  •  Projets par utilisateur – semaines glissantes")
        st.info("Pas encore de données projet dans le CSV — attendons Supabase/Logfire.")
    else:
        latest_monday = now.normalize() - pd.Timedelta(days=now.weekday())
        cutoff        = latest_monday - pd.Timedelta(weeks=nb_semaines_proj)
        last_proj_df  = filt_df[
            (filt_df["timestamp"] >= cutoff) &
            (~filt_df["id projet"].isna())
        ].copy()

        st.subheader(create_dynamic_title("T8 bis  •  Projets par utilisateur – semaines glissantes", last_proj_df, df))

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
    _LICIEL_ROUTE = r"GET /projects/.+/liciel"  # Corrigé: projects au pluriel
    liciel_df = filt_df[filt_df["span_name"].str.contains(_LICIEL_ROUTE, na=False, regex=True)].copy()

    st.subheader(create_dynamic_title("T10  •  Exports Liciel par utilisateur – semaines glissantes", liciel_df, df))

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

