import datetime as _dt
from difflib import SequenceMatcher
from typing import List, Optional
from zoneinfo import ZoneInfo

import streamlit as st
from src.auth_guard import require_login
import pandas as pd
import numpy as np
import plotly.express as px

from src.data_sources import get_supabase
require_login()


st.title("Transcriptions – comparatif des services STT")

# ------------------------------------------------------------------
# 0. Utilitaire commun : première colonne existante dans `df`
# ------------------------------------------------------------------

def _find(cands: List[str], df: pd.DataFrame) -> Optional[str]:
    """Retourne la première colonne trouvée parmi *cands* ou None."""
    return next((c for c in cands if c in df.columns), None)

# ------------------------------------------------------------------
# 1. Chargement Supabase ➜ DataFrame unifié (req + results)
# ------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_data(schema: str = "auditoo") -> pd.DataFrame:
    sb = get_supabase()

    req_df = pd.DataFrame(
        sb.schema(schema).table("transcription_requests").select("*").execute().data or []
    )
    res_df = pd.DataFrame(
        sb.schema(schema).table("transcription_jobs").select("*").execute().data or []
    )
    resp_df = pd.DataFrame(
        sb.schema(schema).table("transcription_responses").select("*").execute().data or []
    )

    if req_df.empty or res_df.empty or resp_df.empty:
        return pd.DataFrame()

    # ————— ne garder que les résultats gagnants —————
    res_id  = _find(["id", "result_id", "uuid"], res_df) or "id"
    link_id = _find(["transcription_result_id", "result_id", "res_id"], resp_df)

    if link_id and res_id in res_df.columns:
        winners = resp_df[link_id].dropna().unique()
        res_df  = res_df[res_df[res_id].isin(winners)]

    # —— identifie timestamps et PK ——
    req_ts = _find(["requested_at", "created_at", "inserted_at", "request_timestamp"], req_df)
    res_ts = _find(["responded_at", "completed_at", "response_timestamp"],                res_df)

    pk_req = _find(["id", "request_id", "pk", "uuid"], req_df) or "id"
    pk_res = _find(["transcription_request_id", "request_id", "req_id"], res_df) or "transcription_request_id"

    # fallback : si timestamp manquant ➜ premier datetime64_* rencontré
    def _first_datetime(df: pd.DataFrame) -> Optional[str]:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return c
        return None

    if req_ts is None:
        req_ts = _first_datetime(req_df)
    if res_ts is None:
        res_ts = _first_datetime(res_df)

    if (req_ts is None) or (res_ts is None) or (pk_req not in req_df.columns) or (pk_res not in res_df.columns):
        return pd.DataFrame()

    # —— standardisation colonnes ——
    req_df = req_df.rename(columns={req_ts: "requested_at", pk_req: "req_id"})
    res_df = res_df.rename(columns={res_ts: "responded_at", pk_res: "req_id"})

    req_df["requested_at"]  = pd.to_datetime(req_df["requested_at"],  errors="coerce")
    res_df["responded_at"] = pd.to_datetime(res_df["responded_at"], errors="coerce")

    df = res_df.merge(
        req_df[["req_id", "requested_at", "audio_duration"]],
        on="req_id", how="left",
    )
    return df

# ------------------------------------------------------------------
# 2. Colonnes auxiliaires (engine, succès, latence)
# ------------------------------------------------------------------

df = load_data()
if df.empty:
    st.warning("Aucune donnée exploitable (timestamps ou clés manquants).")
    st.stop()

# —— détection colonnes timestamps APRES merge ——
# Note: After merge, requested_at becomes requested_at_x (from jobs) and requested_at_y (from requests)
# We want to use requested_at_x (from jobs) to match the behavior of racing analysis
REQ_TS = _find(["requested_at_x", "requested_at", "created_at", "inserted_at"], df)
RES_TS = _find(["responded_at", "completed_at"], df)
if (REQ_TS is None) or (RES_TS is None):
    st.warning("Impossible de localiser les colonnes timestamp dans les données fusionnées.")
    st.stop()


# —— Engine ——
engine_col = _find(["model", "engine"], df) or "model"
df["engine"]   = df[engine_col].fillna("unknown").str.lower()
df["provider"] = df["engine"].str.split(":").str[0]

# —— Latence (s) ——
for _c in (REQ_TS, RES_TS):
    if not pd.api.types.is_datetime64_any_dtype(df[_c]):
        df[_c] = pd.to_datetime(df[_c], errors="coerce")

with pd.option_context("mode.use_inf_as_na", True):
    df["latency_s"] = (df[RES_TS] - df[REQ_TS]).dt.total_seconds()


# —— Succès ——
status_col = _find(["status", "success", "state"], df)
if status_col:
    df["is_success"] = df[status_col].astype(str).str.lower().eq("success")
else:
    df["is_success"] = np.nan

# ------------------------------------------------------------------
# 3. Filtres UI (période + moteur)
# ------------------------------------------------------------------
min_date = df[REQ_TS].min().date()
max_date = df[REQ_TS].max().date()
calendar_max = max(max_date, _dt.datetime.now(ZoneInfo("Europe/Paris")).date())

today_local  = _dt.datetime.now(ZoneInfo("Europe/Paris")).date()
default_start = min_date
default_end   = max(today_local, max_date)  # ouvre sur mois courant si possible
calendar_max  = max(today_local, max_date)

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

c1, c2 = st.columns(2)
with c1:
    raw_sel = st.date_input(
        "Période",
        value=(default_start, default_end),
        min_value=default_start,
        max_value=calendar_max,
        format="DD/MM/YYYY",
        key="stt_period",
    )
    start_date, end_date = _normalize_date_range(
        raw_sel,
        prev=st.session_state.get("_stt_period_last"),
        default_start=default_start,
        default_end=default_end,
        min_value=default_start,
        max_value=calendar_max,
    )
    st.session_state["_stt_period_last"] = (start_date, end_date)

with c2:
    eng_sel = st.selectbox(
        "Moteur STT",
        options=["Tous"] + sorted(df["engine"].unique().tolist()),
        index=0,
    )

mask = (
    (df[REQ_TS].dt.date >= start_date) &
    (df[REQ_TS].dt.date <= end_date)
)
if eng_sel != "Tous":
    mask &= df["engine"] == eng_sel
sub = df[mask]

if sub.empty:
    st.info("Aucune donnée trouvée pour les filtres sélectionnés.")
    st.stop()

# ------------------------------------------------------------------
# 4. Agrégation par moteur
# ------------------------------------------------------------------
agg = (
    sub.groupby("engine")
       .agg(
           n_calls = ("req_id", "size"),
           avg_lat = ("latency_s", "mean"),
           med_lat = ("latency_s", "median"),
           p90_lat = ("latency_s", lambda s: np.nanpercentile(s.dropna(), 90) if s.notna().any() else np.nan),
       )
       .sort_values("n_calls", ascending=False)
)

# # ------------------------------------------------------------------------
# # Mises à jour de l’affichage
# # ------------------------------------------------------------------------
# st.subheader("Tableau comparatif des moteurs")
# st.dataframe(
#     agg.style.format({
#         "avg_lat": "{:.2f} s",
#         "med_lat": "{:.2f} s",
#         "p90_lat": "{:.2f} s",
#     }),
#     use_container_width=True,
# )


# st.subheader("Volume des requêtes par moteur")
# fig_calls = px.bar(
#     agg.reset_index(),
#     x="engine", y="n_calls", text_auto=True,
#     labels={"engine": "Moteur", "n_calls": "Requêtes"},
# )
# fig_calls.update_layout(margin=dict(l=0, r=0, t=40, b=0))
# st.plotly_chart(fig_calls, use_container_width=True)

st.subheader("Distribution des latences")


fig_lat = px.box(
    sub, x="engine", y="latency_s",
    labels={"engine": "Moteur", "latency_s": "Latence (s)"},
)
fig_lat.update_layout(margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig_lat, use_container_width=True)

# ------------------------------------------------------------------
# 5. Detailed comparison table: all models per prompt
# ------------------------------------------------------------------
st.subheader("Comparaison détaillée par prompt")

def fetch_all_rows(sb, schema: str, table_name: str, select: str = "*") -> list:
    """Fetch all rows from a table using pagination to avoid 1000-row limit."""
    all_data = []
    offset = 0
    batch_size = 1000

    while True:
        batch = sb.schema(schema).table(table_name).select(select).range(offset, offset + batch_size - 1).execute().data
        if not batch:
            break
        all_data.extend(batch)
        if len(batch) < batch_size:
            break
        offset += batch_size

    return all_data

@st.cache_data(ttl=300)
def load_detailed_comparison(schema: str = "auditoo") -> pd.DataFrame:
    """Load all jobs with request and response info for detailed comparison."""
    sb = get_supabase()

    # Get all tables with pagination
    requests_df = pd.DataFrame(fetch_all_rows(sb, schema, "transcription_requests"))
    jobs_df = pd.DataFrame(fetch_all_rows(sb, schema, "transcription_jobs"))
    responses_df = pd.DataFrame(fetch_all_rows(sb, schema, "transcription_responses"))
    files_df = pd.DataFrame(fetch_all_rows(sb, schema, "project_files"))

    if requests_df.empty or jobs_df.empty:
        return pd.DataFrame()

    # Merge to get winning job IDs
    if not responses_df.empty:
        winner_map = responses_df.set_index('transcription_request_id')['transcription_result_id'].to_dict()
    else:
        winner_map = {}

    # Get user info
    users_df = pd.DataFrame(fetch_all_rows(sb, schema, 'users', 'id, email'))

    # Merge jobs with requests
    comparison_df = jobs_df.merge(
        requests_df[['id', 'received_at', 'audio_duration', 'file_id', 'user_id']],
        left_on='transcription_request_id',
        right_on='id',
        how='left',
        suffixes=('', '_request')
    )

    # Merge with users to get email
    if not users_df.empty:
        comparison_df = comparison_df.merge(
            users_df[['id', 'email']],
            left_on='user_id',
            right_on='id',
            how='left',
            suffixes=('', '_user')
        )

    # Merge with files to get storage path
    if not files_df.empty:
        comparison_df = comparison_df.merge(
            files_df[['id', 'storage_path']],
            left_on='file_id',
            right_on='id',
            how='left',
            suffixes=('', '_file')
        )

    # Add winner indicator
    comparison_df['is_winner'] = comparison_df.apply(
        lambda row: row['id'] == winner_map.get(row['transcription_request_id'], None),
        axis=1
    )

    # Parse timestamps
    comparison_df['received_at'] = pd.to_datetime(comparison_df['received_at'], errors='coerce')
    comparison_df['requested_at'] = pd.to_datetime(comparison_df['requested_at'], errors='coerce')
    comparison_df['responded_at'] = pd.to_datetime(comparison_df['responded_at'], errors='coerce')


    return comparison_df

detailed_df = load_detailed_comparison()

# Build storage_paths globally for sidebar access
storage_paths = {}

if not detailed_df.empty:
    # Apply the same date filter
    detail_mask = (
        (detailed_df['received_at'].dt.date >= start_date) &
        (detailed_df['received_at'].dt.date <= end_date)
    )
    if eng_sel != "Tous":
        detail_mask &= detailed_df['model'].str.lower() == eng_sel

    detail_sub = detailed_df[detail_mask].copy()

    # NOTE: We keep ALL jobs here (winners and non-winners) because:
    # - Racing analysis needs all jobs per request to compare fastest vs winner
    # - GPT-4o comparison needs both models for each request
    # We'll filter to winners only in the final stats display section

    if not detail_sub.empty:
        # Build storage_paths from detail_sub
        for req_id, group in detail_sub.groupby('transcription_request_id'):
            first_row = group.iloc[0]
            storage_path = first_row.get('storage_path')
            if pd.notna(storage_path):
                storage_paths[str(req_id)] = storage_path

# ------------------------------------------------------------------
# Sidebar Audio Player
# ------------------------------------------------------------------
if storage_paths:
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔊 Audio Player")

        # Use session state to auto-load from "Charger" buttons
        default_req_id = st.session_state.get("loaded_request_id", "")

        sidebar_req_id = st.text_input(
            "Request ID:",
            value=default_req_id,
            placeholder="Collez un Request ID ici",
            key="sidebar_audio_input"
        )

        if sidebar_req_id.strip() and sidebar_req_id.strip() in storage_paths:
            storage_path = storage_paths[sidebar_req_id.strip()]
            try:
                sb = get_supabase()
                signed_url_data = sb.storage.from_('project-files').create_signed_url(storage_path, 3600)
                audio_url = signed_url_data.get('signedURL') or signed_url_data.get('signedUrl')
                if audio_url:
                    st.audio(audio_url)
                else:
                    st.error("Impossible de générer l'URL audio")
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
        elif sidebar_req_id.strip():
            st.warning("Request ID non trouvé")

# ------------------------------------------------------------------
# 5. Detailed comparison table: all models per prompt (continued)
# ------------------------------------------------------------------
if not detailed_df.empty and not detail_sub.empty:
    if not detail_sub.empty:
        # Get all unique models
        all_models = sorted(detail_sub['model'].unique())

        # Create display table (without generating signed URLs upfront)
        display_data = []

        for req_id, group in detail_sub.groupby('transcription_request_id'):
            # Get basic info from first row
            first_row = group.iloc[0]

            row_data = {
                'Request ID': str(req_id),
                'Author': first_row.get('email', '') or '',
                'Request Time': first_row['received_at'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(first_row['received_at']) else '',
                'Audio Duration': f"{first_row['audio_duration']:.2f}s" if pd.notna(first_row['audio_duration']) else '',
            }

            # Create a mapping of model -> (text, duration, is_winner)
            model_data = {}
            for _, job in group.iterrows():
                model_name = job['model']
                duration = job['duration']
                text = job['text'] or ''
                is_winner = job['is_winner']

                # Use full text (no truncation)
                text_display = text
                model_data[model_name] = (text_display, duration, is_winner)

            # Add columns for each model
            for model in all_models:
                if model in model_data:
                    text_display, duration, is_winner = model_data[model]
                    row_data[model] = (f"[{duration:.2f}s] {text_display}", is_winner)
                else:
                    row_data[model] = ('', False)

            display_data.append(row_data)

        if display_data:
            # Separate data and styling info
            table_data = {}
            winner_info = {}

            for col in display_data[0].keys():
                if col in ['Request ID', 'Author', 'Request Time', 'Audio Duration']:
                    table_data[col] = [row[col] for row in display_data]
                else:
                    table_data[col] = [row[col][0] if isinstance(row[col], tuple) else row[col] for row in display_data]
                    winner_info[col] = [row[col][1] if isinstance(row[col], tuple) else False for row in display_data]

            comparison_table = pd.DataFrame(table_data)

            # Apply styling
            def highlight_winners(row):
                styles = [''] * len(row)
                for idx, col in enumerate(comparison_table.columns):
                    if col in winner_info and winner_info[col][row.name]:
                        styles[idx] = 'background-color: #4CAF50; color: white'
                return styles

            styled_table = comparison_table.style.apply(highlight_winners, axis=1)

            st.dataframe(
                styled_table,
                use_container_width=True,
                height=600,
            )

            st.caption("Les cellules en vert indiquent le modèle sélectionné pour la réponse finale. Format: [durée] transcription")
    else:
        st.info("Aucune donnée de comparaison disponible pour les filtres sélectionnés.")
else:
    st.warning("Impossible de charger les données de comparaison détaillée.")

# ------------------------------------------------------------------
# 6. Racing Strategy Analysis: Is waiting for "better" models worth it?
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("🏁 Analyse de la stratégie de racing")

if not detailed_df.empty and not detail_sub.empty:
    # For each request, find fastest job vs winning job
    racing_analysis = []

    for req_id, group in detail_sub.groupby('transcription_request_id'):
        if len(group) < 2:  # Need at least 2 models to compare
            continue

        # Calculate duration for each job
        group['job_duration'] = (group['responded_at'] - group['requested_at']).dt.total_seconds()

        # Find fastest and winner
        fastest_idx = group['job_duration'].idxmin()
        fastest_job = group.loc[fastest_idx]

        winner_job = group[group['is_winner']]
        if winner_job.empty:
            continue
        winner_job = winner_job.iloc[0]

        time_wasted = winner_job['job_duration'] - fastest_job['job_duration']

        # Calculate similarity between fastest and winner
        fastest_text = fastest_job['text'] or ''
        winner_text = winner_job['text'] or ''
        similarity_ratio = SequenceMatcher(None, fastest_text, winner_text).ratio()

        racing_analysis.append({
            'request_id': req_id,
            'fastest_model': fastest_job['model'],
            'fastest_duration': fastest_job['job_duration'],
            'fastest_text': fastest_text,
            'winner_model': winner_job['model'],
            'winner_duration': winner_job['job_duration'],
            'winner_text': winner_text,
            'time_wasted': time_wasted,
            'winner_is_fastest': fastest_job['model'] == winner_job['model'],
            'similarity_ratio': similarity_ratio,
            'texts_identical': fastest_text == winner_text,
            'char_diff': abs(len(fastest_text) - len(winner_text)),
        })

    if racing_analysis:
        race_df = pd.DataFrame(racing_analysis)

        # Calculate key metrics
        pct_different = (100 * (~race_df['winner_is_fastest']).sum() / len(race_df))
        cases_where_waited = race_df[~race_df['winner_is_fastest']]
        median_wasted = cases_where_waited['time_wasted'].median() if not cases_where_waited.empty else 0
        total_wasted = cases_where_waited['time_wasted'].sum() if not cases_where_waited.empty else 0

        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Stratégie utilisée",
                f"{pct_different:.1f}%",
                help="Pourcentage de cas où on a attendu un modèle plus lent que le plus rapide"
            )
        with col2:
            st.metric(
                "Temps médian perdu",
                f"{median_wasted:.2f}s",
                help="Temps médian d'attente supplémentaire quand on choisit un modèle plus lent"
            )
        # with col3:
        #     st.metric(
        #         "Temps total perdu",
        #         f"{total_wasted:.1f}s",
        #         help="Temps total qui aurait pu être économisé en prenant toujours le plus rapide"
        #     )

        # # Time wasted distribution
        # if not cases_where_waited.empty:
        #     st.markdown("### Distribution du temps perdu")
        #     fig_wasted = px.box(
        #         cases_where_waited,
        #         y='time_wasted',
        #         labels={'time_wasted': 'Temps perdu (s)'},
        #         title=f"Quand la stratégie est utilisée ({len(cases_where_waited)} cas)"
        #     )
        #     fig_wasted.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        #     st.plotly_chart(fig_wasted, use_container_width=True)

        # Winner vs Fastest breakdown
        st.markdown("### Quel modèle gagne vs. quel modèle est le plus rapide ?")

        col1, col2 = st.columns(2)
        with col1:
            fastest_counts = race_df['fastest_model'].value_counts().reset_index()
            fastest_counts.columns = ['Modèle', 'Nombre de fois le plus rapide']
            fig_fastest = px.bar(
                fastest_counts,
                x='Modèle', y='Nombre de fois le plus rapide',
                text_auto=True,
                title="Modèles les plus rapides"
            )
            fig_fastest.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_fastest, use_container_width=True)

        with col2:
            winner_counts = race_df['winner_model'].value_counts().reset_index()
            winner_counts.columns = ['Modèle', 'Nombre de victoires']
            fig_winner = px.bar(
                winner_counts,
                x='Modèle', y='Nombre de victoires',
                text_auto=True,
                title="Modèles gagnants"
            )
            fig_winner.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_winner, use_container_width=True)

        # Quality comparison samples
        st.markdown("### Exemples de comparaison qualité (quand le gagnant ≠ le plus rapide)")

        if not cases_where_waited.empty:
            # Global metrics for quality comparison
            st.markdown("#### 📊 Métriques globales")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Échantillons",
                    len(cases_where_waited),
                    help="Nombre de cas où on a attendu un modèle plus lent"
                )

            with col2:
                highly_similar_pct = 100 * (cases_where_waited['similarity_ratio'] >= 0.95).sum() / len(cases_where_waited)
                st.metric(
                    "Très similaires",
                    f"{highly_similar_pct:.1f}%",
                    help="Pourcentage de textes avec >95% de similarité (fuzzy matching)"
                )

            with col3:
                avg_similarity = 100 * cases_where_waited['similarity_ratio'].mean()
                st.metric(
                    "Similarité moyenne",
                    f"{avg_similarity:.1f}%",
                    help="Score moyen de similarité fuzzy entre le plus rapide et le gagnant"
                )

            with col4:
                avg_time_wasted = cases_where_waited['time_wasted'].mean()
                st.metric(
                    "Temps perdu moyen",
                    f"{avg_time_wasted:.2f}s",
                    help="Temps moyen perdu en attendant le gagnant au lieu du plus rapide"
                )

            # Second row: Additional details
            st.caption(f"📝 Textes exactement identiques: {100 * cases_where_waited['texts_identical'].sum() / len(cases_where_waited):.1f}% | "
                       f"Diff. caractères moyenne: {cases_where_waited['char_diff'].mean():.1f}")

            st.markdown("---")


            # Show top 10 cases by time wasted
            sample_cases = cases_where_waited.nlargest(min(10, len(cases_where_waited)), 'time_wasted')

            for idx, row in sample_cases.iterrows():
                full_req_id = str(row['request_id'])

                # Calculate similarity badge
                similarity_pct = row['similarity_ratio'] * 100
                if row['texts_identical']:
                    similarity_badge = "✅ IDENTIQUE (100%)"
                elif similarity_pct >= 95:
                    similarity_badge = f"🟢 TRÈS SIMILAIRE ({similarity_pct:.1f}%)"
                elif similarity_pct >= 90:
                    similarity_badge = f"🟡 SIMILAIRE ({similarity_pct:.1f}%)"
                else:
                    similarity_badge = f"🔴 DIFFÉRENT ({similarity_pct:.1f}%)"

                latency_info = (
                    f"⚡ {row['fastest_model']} ({row['fastest_duration']:.2f}s) vs "
                    f"🏆 {row['winner_model']} ({row['winner_duration']:.2f}s) | "
                    f"⏱️ +{row['time_wasted']:.2f}s"
                )

                with st.expander(
                    f"{similarity_badge} | Request {full_req_id[:8]}... | {latency_info}"
                ):
                    # Display Request ID with "Charger" button
                    col_id, col_btn = st.columns([3, 1])
                    with col_id:
                        st.caption(f"**Request ID:** `{full_req_id}`")
                    with col_btn:
                        if st.button("🔊 Charger", key=f"load_audio_racing_{idx}"):
                            st.session_state.loaded_request_id = full_req_id
                            st.rerun()

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**⚡ Plus rapide: {row['fastest_model']}** ({row['fastest_duration']:.2f}s)")
                        st.text_area(
                            "Transcription",
                            value=row['fastest_text'],
                            height=200,
                            key=f"fastest_{idx}",
                            disabled=True
                        )
                        st.caption(f"Longueur: {len(row['fastest_text'])} caractères")
                    with col2:
                        st.markdown(f"**🏆 Gagnant: {row['winner_model']}** ({row['winner_duration']:.2f}s)")
                        st.text_area(
                            "Transcription",
                            value=row['winner_text'],
                            height=200,
                            key=f"winner_{idx}",
                            disabled=True
                        )
                        st.caption(f"Longueur: {len(row['winner_text'])} caractères")

                    # Show similarity info
                    st.markdown("**Analyse:**")
                    st.caption(f"• Similarité fuzzy: {similarity_pct:.1f}%")
                    st.caption(f"• Différence de caractères: {row['char_diff']}")
                    st.caption(f"• Temps perdu en attendant le gagnant: {row['time_wasted']:.2f}s")

        # # Recommendation summary
        # st.markdown("### 📊 Conclusion")

        # if pct_different < 10:
        #     st.success(
        #         f"✅ **Le modèle le plus rapide gagne {100-pct_different:.1f}% du temps.** "
        #         f"La stratégie de racing n'est probablement pas utile."
        #     )
        # elif pct_different < 30:
        #     st.info(
        #         f"🤔 **La stratégie est utilisée dans {pct_different:.1f}% des cas.** "
        #         f"Temps médian perdu: {median_wasted:.2f}s. "
        #         f"Vérifiez si les différences de qualité justifient cette latence."
        #     )
        # else:
        #     st.warning(
        #         f"⚠️ **La stratégie est utilisée dans {pct_different:.1f}% des cas.** "
        #         f"Temps médian perdu: {median_wasted:.2f}s. "
        #         f"Si les différences de qualité sont minimes, considérez utiliser uniquement le modèle le plus rapide."
        #     )

        # # Additional insight
        # if not cases_where_waited.empty:
        #     avg_wasted = cases_where_waited['time_wasted'].mean()
        #     st.caption(
        #         f"💡 En moyenne, quand on attend un meilleur modèle, on perd {avg_wasted:.2f}s par requête. "
        #         f"Sur {len(cases_where_waited)} cas, cela représente {total_wasted:.1f}s au total."
        #     )
    else:
        st.info("Pas assez de données pour analyser la stratégie de racing.")
else:
    st.info("Aucune donnée disponible pour l'analyse de racing.")

# ------------------------------------------------------------------
# 7. GPT-4o vs GPT-4o-mini Quality Comparison Tool
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Comparaison qualité: GPT-4o vs GPT-4o-mini")

if not detailed_df.empty and not detail_sub.empty:
    # Find requests where both GPT-4o and GPT-4o-mini completed successfully
    comparison_pairs = []

    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate fuzzy similarity ratio between two texts (0.0 to 1.0)"""
        return SequenceMatcher(None, text1, text2).ratio()

    for req_id, group in detail_sub.groupby('transcription_request_id'):
        # Look for both models in this request
        # NOTE: We compare ALL attempts (winners and non-winners) because:
        # - Each request typically has only ONE winner
        # - We want to compare how both models perform on the SAME audio
        # - This shows the quality comparison across all transcriptions, not just production
        gpt4o = group[group['model'] == 'openai:gpt-4o-transcribe']
        gpt4o_mini = group[group['model'] == 'openai:gpt-4o-mini-transcribe']

        if not gpt4o.empty and not gpt4o_mini.empty:
            gpt4o_row = gpt4o.iloc[0]
            mini_row = gpt4o_mini.iloc[0]

            # Calculate duration
            gpt4o_duration = (gpt4o_row['responded_at'] - gpt4o_row['requested_at']).total_seconds()
            mini_duration = (mini_row['responded_at'] - mini_row['requested_at']).total_seconds()

            gpt4o_text = gpt4o_row['text'] or ''
            mini_text = mini_row['text'] or ''

            # Calculate fuzzy similarity
            similarity_ratio = calculate_similarity(gpt4o_text, mini_text)

            comparison_pairs.append({
                'request_id': req_id,
                'gpt4o_text': gpt4o_text,
                'mini_text': mini_text,
                'gpt4o_duration': gpt4o_duration,
                'mini_duration': mini_duration,
                'latency_diff': gpt4o_duration - mini_duration,
                'char_diff': abs(len(gpt4o_text) - len(mini_text)),
                'texts_identical': gpt4o_text == mini_text,
                'similarity_ratio': similarity_ratio,
                'highly_similar': similarity_ratio >= 0.95,  # Consider 95%+ as highly similar
            })

    if comparison_pairs:
        comp_df = pd.DataFrame(comparison_pairs)

        # Overall metrics
        st.markdown("### 📊 Métriques globales")

        # First row: Main metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Échantillons",
                len(comp_df),
                help="Nombre de requêtes où les deux modèles ont répondu"
            )

        with col2:
            highly_similar_pct = 100 * comp_df['highly_similar'].sum() / len(comp_df)
            st.metric(
                "Très similaires",
                f"{highly_similar_pct:.1f}%",
                help="Pourcentage de textes avec >95% de similarité (fuzzy matching)"
            )

        with col3:
            avg_similarity = 100 * comp_df['similarity_ratio'].mean()
            st.metric(
                "Similarité moyenne",
                f"{avg_similarity:.1f}%",
                help="Score moyen de similarité fuzzy entre les transcriptions"
            )

        with col4:
            avg_latency_saved = comp_df['latency_diff'].mean()
            st.metric(
                "Gain de latence moyen",
                f"{avg_latency_saved:.2f}s",
                help="Temps économisé avec GPT-4o-mini en moyenne",
                delta=f"{avg_latency_saved:.2f}s"
            )

        # Second row: Additional details
        st.caption(f"📝 Textes exactement identiques: {100 * comp_df['texts_identical'].sum() / len(comp_df):.1f}% | "
                   f"Diff. caractères moyenne: {comp_df['char_diff'].mean():.1f}")

        # # Show distributions side by side
        # col1, col2 = st.columns(2)

        # with col1:
        #     st.markdown("### ⚡ Distribution du gain de latence")
        #     fig_latency = px.histogram(
        #         comp_df,
        #         x='latency_diff',
        #         nbins=30,
        #         labels={'latency_diff': 'Gain de latence (s)', 'count': 'Nombre'},
        #     )
        #     fig_latency.add_vline(
        #         x=comp_df['latency_diff'].mean(),
        #         line_dash="dash",
        #         line_color="red",
        #         annotation_text=f"Moy: {comp_df['latency_diff'].mean():.2f}s"
        #     )
        #     fig_latency.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        #     st.plotly_chart(fig_latency, use_container_width=True)

        # with col2:
        #     st.markdown("### 🎯 Distribution de la similarité")
        #     fig_similarity = px.histogram(
        #         comp_df,
        #         x=comp_df['similarity_ratio'] * 100,  # Convert to percentage
        #         nbins=30,
        #         labels={'x': 'Similarité (%)', 'count': 'Nombre'},
        #     )
        #     fig_similarity.add_vline(
        #         x=95,
        #         line_dash="dash",
        #         line_color="green",
        #         annotation_text="Seuil: 95%"
        #     )
        #     fig_similarity.add_vline(
        #         x=comp_df['similarity_ratio'].mean() * 100,
        #         line_dash="dash",
        #         line_color="red",
        #         annotation_text=f"Moy: {comp_df['similarity_ratio'].mean()*100:.1f}%"
        #     )
        #     fig_similarity.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        #     st.plotly_chart(fig_similarity, use_container_width=True)

        # Sample comparisons
        st.markdown("### 🔍 Exemples de comparaisons (échantillon aléatoire)")

        # Select random sample (max 20)
        sample_size = min(20, len(comp_df))
        sample_df = comp_df.sample(n=sample_size, random_state=42)

        for idx, row in sample_df.iterrows():
            full_req_id = str(row['request_id'])
            similarity_pct = row['similarity_ratio'] * 100
            if row['texts_identical']:
                similarity_badge = "✅ IDENTIQUE (100%)"
            elif similarity_pct >= 95:
                similarity_badge = f"🟢 TRÈS SIMILAIRE ({similarity_pct:.1f}%)"
            elif similarity_pct >= 90:
                similarity_badge = f"🟡 SIMILAIRE ({similarity_pct:.1f}%)"
            else:
                similarity_badge = f"🔴 DIFFÉRENT ({similarity_pct:.1f}%)"

            latency_info = f"⚡ mini: {row['mini_duration']:.2f}s | 4o: {row['gpt4o_duration']:.2f}s (gain: {row['latency_diff']:.2f}s)"

            with st.expander(
                f"{similarity_badge} | Request {full_req_id[:8]}... | {latency_info}"
            ):
                # Display Request ID with "Charger" button
                col_id, col_btn = st.columns([3, 1])
                with col_id:
                    st.caption(f"**Request ID:** `{full_req_id}`")
                with col_btn:
                    if st.button("🔊 Charger", key=f"load_audio_sample_{idx}"):
                        st.session_state.loaded_request_id = full_req_id
                        st.rerun()

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**🏆 GPT-4o** ({row['gpt4o_duration']:.2f}s)")
                    st.text_area(
                        "Transcription",
                        value=row['gpt4o_text'],
                        height=200,
                        key=f"gpt4o_{idx}",
                        disabled=True
                    )
                    st.caption(f"Longueur: {len(row['gpt4o_text'])} caractères")

                with col2:
                    st.markdown(f"**⚡ GPT-4o-mini** ({row['mini_duration']:.2f}s)")
                    st.text_area(
                        "Transcription",
                        value=row['mini_text'],
                        height=200,
                        key=f"mini_{idx}",
                        disabled=True
                    )
                    st.caption(f"Longueur: {len(row['mini_text'])} caractères")

                # Show similarity info
                st.markdown("**Analyse:**")
                st.caption(f"• Similarité fuzzy: {similarity_pct:.1f}%")
                st.caption(f"• Différence de caractères: {row['char_diff']}")
                st.caption(f"• Gain de latence: {row['latency_diff']:.2f}s")

        # # Decision helper
        # st.markdown("### 🎯 Aide à la décision")

        # # Calculate recommendation scores
        # identical_pct = 100 * comp_df['texts_identical'].sum() / len(comp_df)
        # avg_similarity = 100 * comp_df['similarity_ratio'].mean()
        # speed_gain = avg_latency_saved
        # speedup_pct = 100 * speed_gain / comp_df['gpt4o_duration'].mean()

        # if highly_similar_pct >= 80 and speed_gain >= 0.15:
        #     st.success(
        #         f"✅ **Recommandation: PASSER À GPT-4o-mini**\n\n"
        #         f"• {highly_similar_pct:.1f}% des textes sont très similaires (>95% similarité)\n"
        #         f"• Similarité moyenne: {avg_similarity:.1f}%\n"
        #         f"• Gain de latence moyen: {speed_gain:.2f}s ({speedup_pct:.0f}% plus rapide)\n"
        #         f"• Les différences sont majoritairement cosmétiques (ponctuation, formatage)\n\n"
        #         f"**Impact utilisateur**: Expérience {speedup_pct:.0f}% plus rapide avec qualité quasi-identique"
        #     )
        # elif avg_similarity >= 90 and speed_gain >= 0.10:
        #     st.info(
        #         f"🤔 **Recommandation: TEST EN PRODUCTION RECOMMANDÉ**\n\n"
        #         f"• Similarité moyenne: {avg_similarity:.1f}%\n"
        #         f"• {highly_similar_pct:.1f}% des textes >95% similaires\n"
        #         f"• Gain de latence moyen: {speed_gain:.2f}s ({speedup_pct:.0f}%)\n"
        #         f"• Différences mineures à valider avec utilisateurs\n\n"
        #         f"**Suggestion**: Déployer GPT-4o-mini pour 50% du trafic et collecter feedback"
        #     )
        # elif avg_similarity >= 85:
        #     st.warning(
        #         f"⚠️ **Recommandation: EXAMINER LES DIFFÉRENCES**\n\n"
        #         f"• Similarité moyenne: {avg_similarity:.1f}%\n"
        #         f"• Seulement {highly_similar_pct:.1f}% des textes >95% similaires\n"
        #         f"• Gain de latence: {speed_gain:.2f}s ({speedup_pct:.0f}%)\n\n"
        #         f"**Suggestion**: Examiner manuellement les exemples ci-dessus pour évaluer l'impact qualité"
        #     )
        # else:
        #     st.error(
        #         f"❌ **Recommandation: RESTER SUR GPT-4o OU TESTER DEEPGRAM**\n\n"
        #         f"• Similarité moyenne: {avg_similarity:.1f}% (insuffisante)\n"
        #         f"• Gain de latence: {speed_gain:.2f}s ({speedup_pct:.0f}%)\n\n"
        #         f"**Alternative**: Si la vitesse est prioritaire, considérer Deepgram Nova-2 pour un gain plus significatif"
        #     )

        # # Show detailed breakdown
        # st.caption(f"📊 Détails: {identical_pct:.1f}% identiques caractère par caractère | "
        #            f"Médiane similarité: {100*comp_df['similarity_ratio'].median():.1f}% | "
        #            f"P10 similarité: {100*comp_df['similarity_ratio'].quantile(0.1):.1f}%")

        # Additional insights
        with st.expander("📈 Statistiques détaillées"):
            st.markdown(f"""
            **⚠️ Note:** Ces statistiques incluent TOUTES les tentatives (gagnantes et non-gagnantes) car nous comparons les performances des deux modèles sur les MÊMES audios.
            Cela diffère du box plot en haut de page qui ne montre que les transcriptions gagnantes (production).

            **Distribution des latences (toutes tentatives):**
            - GPT-4o: min={comp_df['gpt4o_duration'].min():.2f}s, max={comp_df['gpt4o_duration'].max():.2f}s, médiane={comp_df['gpt4o_duration'].median():.2f}s
            - GPT-4o-mini: min={comp_df['mini_duration'].min():.2f}s, max={comp_df['mini_duration'].max():.2f}s, médiane={comp_df['mini_duration'].median():.2f}s

            **Distribution de similarité:**
            - Moyenne: {100*comp_df['similarity_ratio'].mean():.1f}%
            - Médiane: {100*comp_df['similarity_ratio'].median():.1f}%
            - P10 (10% pires cas): {100*comp_df['similarity_ratio'].quantile(0.1):.1f}%
            - P90: {100*comp_df['similarity_ratio'].quantile(0.9):.1f}%
            - Textes >95% similaires: {100*comp_df['highly_similar'].sum()/len(comp_df):.1f}%
            - Textes >90% similaires: {100*(comp_df['similarity_ratio']>=0.9).sum()/len(comp_df):.1f}%
            - Textes exactement identiques: {100*comp_df['texts_identical'].sum()/len(comp_df):.1f}%

            **Distribution des gains:**
            - Gain de latence: P50={comp_df['latency_diff'].quantile(0.5):.2f}s, P90={comp_df['latency_diff'].quantile(0.9):.2f}s, P95={comp_df['latency_diff'].quantile(0.95):.2f}s
            - Différence de caractères: P50={comp_df['char_diff'].quantile(0.5):.0f}, P90={comp_df['char_diff'].quantile(0.9):.0f}, P95={comp_df['char_diff'].quantile(0.95):.0f}

            **Impact sur {len(comp_df)} requêtes:**
            - Temps total économisé avec mini: {comp_df['latency_diff'].sum():.1f}s ({comp_df['latency_diff'].sum()/60:.1f} minutes)
            - Amélioration moyenne: {100*avg_latency_saved/comp_df['gpt4o_duration'].mean():.1f}%
            """)

    else:
        st.info("Aucune requête où GPT-4o et GPT-4o-mini ont tous deux répondu.")
else:
    st.info("Aucune donnée disponible pour la comparaison de qualité.")
