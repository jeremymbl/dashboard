#!/usr/bin/env python3
"""
data_sources.py
===============
Utility to access Supabase and Logfire analytics for the Auditoo Streamlit dashboard.

"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import streamlit as st
from supabase import Client as SupabaseClient, create_client

from logfire.query_client import LogfireQueryClient, QueryExecutionError

_LF_READ_TOKEN: str = st.secrets["LOGFIRE_TOKEN"]
_PROJECT_GUID = st.secrets["_PROJECT_GUID"]


_PROMPT_SQL_TEMPLATE = """
WITH success_spans AS (
    SELECT DISTINCT trace_id
    FROM   records
    WHERE  span_name = 'Agent run succeeded'
)
SELECT
  DATE_FORMAT(r.start_timestamp AT TIME ZONE 'Europe/Paris',
              '%d/%m/%Y %H:%M:%S')           AS "timestamp",
  COALESCE(
      r.attributes -> 'fastapi.arguments.values' -> 'auth_info' ->> 'email',
      r.attributes -> 'user_email' ->> 0
  )                                         AS "email utilisateur",
  JSON_GET(
      JSON_GET(r.attributes, 'fastapi.arguments.values'),
      'user_id'
  )                                         AS "user_id",
  COALESCE(
      r.attributes -> 'fastapi.arguments.values' ->> 'content',
      r.attributes -> 'fastapi.arguments.values' -> 'prompt' ->> 'content'
  )                                         AS "prompt",
  COALESCE(
      r.attributes -> 'fastapi.arguments.values' ->> 'scope',
      r.attributes -> 'fastapi.arguments.values' -> 'prompt' ->> 'scope'
  )                                         AS "scope",
  ROUND(r.duration, 2)                      AS "duree traitement",
  CASE
    WHEN s.trace_id IS NOT NULL THEN 'Succès'
    ELSE 'Échec'
  END                                       AS "statut",
  r.trace_id                                AS "trace_id",
  JSON_GET(
      JSON_GET(r.attributes, 'fastapi.arguments.values'),
      'project_id'
  )                                         AS "id projet",
  r.span_name                               AS "span_name",
  r.attributes -> 'fastapi.arguments.values' AS "_raw_values"
FROM   records r
LEFT JOIN success_spans s USING(trace_id)
WHERE  (
          (r.span_name ILIKE 'POST /projects/%/message'
           AND r.start_timestamp >= CURRENT_DATE - INTERVAL {lookback_days} DAY)
       OR (r.span_name ILIKE 'POST /projects/%/prompts/chat'
           AND r.start_timestamp >= CURRENT_DATE - INTERVAL {lookback_days} DAY)
       OR (r.span_name ILIKE 'POST /projects/%/transcribe'
           AND r.start_timestamp >= CURRENT_DATE - INTERVAL {lookback_days} DAY)
       OR (r.span_name ILIKE 'GET /projects/%/liciel%'
           AND r.start_timestamp >= CURRENT_DATE - INTERVAL {lookback_days} DAY)
      )
  AND  r.project_id = '{project_guid}'
ORDER  BY r.start_timestamp DESC
LIMIT  {limit} OFFSET {offset}
"""

_AGGREGATED_METRICS_SQL_TEMPLATE = """
WITH relevant_spans AS (
    SELECT
        trace_id,
        start_timestamp,
        span_name,
        COALESCE(
            attributes -> 'fastapi.arguments.values' -> 'auth_info' ->> 'email',
            attributes -> 'user_email' ->> 0
        ) AS email
    FROM records
    WHERE (
            span_name ILIKE 'POST /projects/%/message' OR
            span_name ILIKE 'POST /projects/%/prompts/chat' OR
            span_name ILIKE 'GET /projects/%/liciel%'
          )
      AND start_timestamp >= CURRENT_DATE - INTERVAL {lookback_days} DAY
      AND project_id = '{project_guid}'
),
success_spans AS (
    SELECT DISTINCT trace_id
    FROM records
    WHERE span_name = 'Agent run succeeded'
      AND start_timestamp >= CURRENT_DATE - INTERVAL {lookback_days} DAY
      AND project_id = '{project_guid}'
)
SELECT
    CAST((start_timestamp AT TIME ZONE 'Europe/Paris') AS DATE) AS "date",
    COUNT(DISTINCT email) AS "dau",
    SUM(CASE WHEN span_name ILIKE 'POST /projects/%/message' OR span_name ILIKE 'POST /projects/%/prompts/chat' THEN 1 ELSE 0 END) AS "total_prompts",
    SUM(CASE WHEN (span_name ILIKE 'POST /projects/%/message' OR span_name ILIKE 'POST /projects/%/prompts/chat') AND s.trace_id IS NOT NULL THEN 1 ELSE 0 END) AS "successful_prompts",
    SUM(CASE WHEN span_name ILIKE 'GET /projects/%/liciel%' THEN 1 ELSE 0 END) AS "liciel_exports"
FROM relevant_spans r
LEFT JOIN success_spans s USING(trace_id)
GROUP BY 1
ORDER BY 1 ASC;
"""

_WEEKLY_PROJECTS_SQL_TEMPLATE = """
SELECT
    DATE_TRUNC('week', start_timestamp AT TIME ZONE 'Europe/Paris') AS "week_start_date",
    COUNT(DISTINCT JSON_GET(
        JSON_GET(attributes, 'fastapi.arguments.values'),
        'project_id'
    )) AS "unique_project_count"
FROM records
WHERE
    start_timestamp >= CURRENT_DATE - INTERVAL {lookback_days} DAY
    AND project_id = '{project_guid}'
    AND JSON_GET(JSON_GET(attributes, 'fastapi.arguments.values'), 'project_id') IS NOT NULL
    -- Exclude test users to get accurate project counts
    AND LOWER(COALESCE(
        attributes -> 'fastapi.arguments.values' -> 'auth_info' ->> 'email',
        attributes -> 'user_email' ->> 0,
        ''
    )) NOT LIKE '%@auditoo.eco'
    AND LOWER(COALESCE(
        attributes -> 'fastapi.arguments.values' -> 'auth_info' ->> 'email',
        attributes -> 'user_email' ->> 0,
        ''
    )) != 'test@test.com'
GROUP BY 1
ORDER BY 1 DESC;
"""

_WEEKLY_ACTIVE_USERS_SQL_TEMPLATE = """
SELECT
    DATE_TRUNC('week', start_timestamp AT TIME ZONE 'Europe/Paris') AS "week_start_date",
    COUNT(DISTINCT COALESCE(
        attributes -> 'fastapi.arguments.values' -> 'auth_info' ->> 'email',
        attributes -> 'user_email' ->> 0
    )) AS "wau"
FROM records
WHERE
    start_timestamp >= CURRENT_DATE - INTERVAL {lookback_days} DAY
    AND project_id = '{project_guid}'
    AND COALESCE(
        attributes -> 'fastapi.arguments.values' -> 'auth_info' ->> 'email',
        attributes -> 'user_email' ->> 0
    ) IS NOT NULL
    -- Exclude test users for accurate user counts
    AND LOWER(COALESCE(
        attributes -> 'fastapi.arguments.values' -> 'auth_info' ->> 'email',
        attributes -> 'user_email' ->> 0,
        ''
    )) NOT LIKE '%@auditoo.eco'
    AND LOWER(COALESCE(
        attributes -> 'fastapi.arguments.values' -> 'auth_info' ->> 'email',
        attributes -> 'user_email' ->> 0,
        ''
    )) != 'test@test.com'
GROUP BY 1
ORDER BY 1 DESC;
"""

# ---------------------------------------------------------------------------
# Project Inspection SQL Template (for debugging user interactions)
# ---------------------------------------------------------------------------

_INSPECTION_SQL_TEMPLATE = """
WITH agent_run_traces AS (
    SELECT DISTINCT trace_id
    FROM records
    WHERE span_name = 'agent run'
),
success_spans AS (
    SELECT DISTINCT trace_id
    FROM records
    WHERE span_name = 'Agent run succeeded'
)
SELECT
    r.trace_id,
    r.span_id,
    r.span_name,
    r.start_timestamp AT TIME ZONE 'Europe/Paris' AS "timestamp",
    r.duration,
    r.attributes,
    CASE
        WHEN s.trace_id IS NOT NULL THEN 'success'
        ELSE 'failure'
    END AS status
FROM records r
INNER JOIN agent_run_traces a ON r.trace_id = a.trace_id
LEFT JOIN success_spans s ON r.trace_id = s.trace_id
WHERE r.project_id = '{logfire_project_id}'
    AND r.span_name LIKE 'POST /projects/%'
    AND r.start_timestamp >= NOW() - INTERVAL '{lookback_days} days'
    AND (
        r.attributes -> 'fastapi.arguments.values' ->> 'project_id' = '{user_project_id}'
        OR r.attributes -> 'fastapi.arguments.values' LIKE '%"project_id":"{user_project_id}"%'
    )
ORDER BY r.start_timestamp DESC
LIMIT {limit}
"""

# ---------------------------------------------------------------------------
# Supabase configuration
# ---------------------------------------------------------------------------

_SB_URL: str = st.secrets["SUPABASE_API_URL"]
_SB_KEY: str = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY") or st.secrets.get("SUPABASE_ANON_KEY")

if not _SB_KEY:
    raise RuntimeError(
        "No SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY found in secrets.toml"
    )

_supabase: Optional[SupabaseClient] = None


def get_supabase() -> SupabaseClient:
    """Return a lazily‑instantiated singleton Supabase client."""
    global _supabase
    if _supabase is None:
        _supabase = create_client(_SB_URL, _SB_KEY)
    return _supabase


def fetch_events(
    table: str = "events",
    *,
    from_ts: int | None = None,
    to_ts: int | None = None,
    limit: int = 10_000,
) -> List[Dict[str, Any]]:
    """Return a list of event rows ordered chronologically."""
    sb = get_supabase()
    query = (
        sb.table(table)
        .select("*")
        .order("timestamp", desc=False)
        .limit(limit)
    )

    if from_ts is not None:
        query = query.gte("timestamp", from_ts)
    if to_ts is not None:
        query = query.lte("timestamp", to_ts)

    res = query.execute()
    return res.data or []


# ---------------------------------------------------------------------------
# Logfire configuration
# ---------------------------------------------------------------------------

_LF_BASE_URL: str = st.secrets.get("LOGFIRE_BASE_URL", "https://logfire-us.pydantic.dev").rstrip("/")
_LF_PROJECT_URL: str = st.secrets["LOGFIRE_PROJECT_URL"].rstrip("/")
_LF_TOKEN: str = st.secrets["LOGFIRE_TOKEN"]

_LF_PROJECT_PATH = _LF_PROJECT_URL.replace(_LF_BASE_URL + "/", "")

_http = httpx.Client(
    base_url=_LF_BASE_URL,
    headers={"Authorization": f"Bearer {_LF_TOKEN}"},
    timeout=5.0,  # Réduit de 10s à 5s pour des réponses plus rapides
)

# ---------------------------------------------------------------------------
# Logfire query tuning
# ---------------------------------------------------------------------------

LOGFIRE_BATCH_SIZE = 500  # Exposed for diagnostics/scripts
_LOGFIRE_RATE_LIMIT_MAX_RETRIES = 4
_LOGFIRE_RATE_LIMIT_BACKOFF_BASE = 1.5  # seconds, exponential backoff
_LOGFIRE_POST_BATCH_DELAY = 0.2  # seconds, light throttling between batches

# ---------------------------------------------------------------------------
# Simple in‑memory TTL cache for expensive Logfire queries
# ---------------------------------------------------------------------------

_TTL_SECONDS = 60  # 1 min pour des données plus fraîches (était 5 min)
_cache: Dict[str, tuple[float, Any]] = {}

# Cache séparé pour les mappings user/email avec TTL plus long
_MAPPING_TTL_SECONDS = 300  # 5 min pour les mappings (changent moins souvent)
_mapping_cache: Dict[str, tuple[float, Any]] = {}


def _cache_get(key: str) -> Any | None:
    item = _cache.get(key)
    if item and time.time() - item[0] < _TTL_SECONDS:
        return item[1]
    return None


def _cache_set(key: str, value: Any) -> None:
    _cache[key] = (time.time(), value)


def _mapping_cache_get(key: str) -> Any | None:
    item = _mapping_cache.get(key)
    if item and time.time() - item[0] < _MAPPING_TTL_SECONDS:
        return item[1]
    return None


def _mapping_cache_set(key: str, value: Any) -> None:
    _mapping_cache[key] = (time.time(), value)


def clear_cache() -> None:
    """Force la suppression du cache pour obtenir des données fraîches."""
    global _cache, _mapping_cache
    _cache.clear()
    _mapping_cache.clear()


def fetch_logfire_metrics(
    *,
    status: str = "All",
    resolution: str = "6h",
    lookback_days: int = 14,
) -> List[Dict[str, Any]]:
    """Query aggregated metrics from Logfire dashboard API."""
    cache_key = f"metrics:{status}:{resolution}:{lookback_days}"
    if (cached := _cache_get(cache_key)) is not None:
        return cached

    params = {
        "env": "prod",
        "var-filterStatus": status,
        "var-resolution": resolution,
        "start": f"{lookback_days}d",
    }
    resp = _http.get(f"/{_LF_PROJECT_PATH}/metrics", params=params)
    resp.raise_for_status()
    data = resp.json()
    _cache_set(cache_key, data)
    return data


# ---------------------------------------------------------------------------
# Higher‑level helpers
# ---------------------------------------------------------------------------

def get_usage_funnel(
    *,
    event_table: str = "events",
    limit: int = 10_000,
) -> List[Dict[str, Any]]:
    """Build a very coarse funnel: Sessions ➜ Prompts ➜ Exports."""
    counts = {"session_start": 0, "prompt_call": 0, "export": 0}

    for event in fetch_events(table=event_table, limit=limit):
        etype = event.get("event_type")
        if etype in counts:
            counts[etype] += 1

    return [
        {"step": "Sessions", "count": counts["session_start"]},
        {"step": "Prompts", "count": counts["prompt_call"]},
        {"step": "Exports", "count": counts["export"]},
    ]


# ---------------------------------------------------------------------------
#  Enrichissement e-mails via Supabase (avec cache optimisé)
# ---------------------------------------------------------------------------
_user_to_email: dict[str, str] | None = None
_proj_to_user : dict[str, str] | None = None

def _get_user_to_email() -> dict[str, str]:
    """user_id → email  (auditoo.users) avec cache optimisé."""
    global _user_to_email

    # Vérifier le cache d'abord
    if (cached := _mapping_cache_get("user_to_email")) is not None:
        print("   ✓ user_to_email mapping loaded from cache")
        return cached

    if _user_to_email is None:
        print("   ⏱️  Loading user_to_email mapping from Supabase (auditoo.users)...")
        _start = time.time()
        sb   = get_supabase()
        rows = (
            sb.schema("auditoo")
              .table("users")
              .select("id,email")
              .execute()
              .data
            or []
        )
        _user_to_email = {str(r["id"]): r["email"] for r in rows if r.get("email")}
        print(f"   ✅ Loaded {len(_user_to_email)} users in {time.time()-_start:.2f}s")

    # Mettre en cache avec TTL plus long
    _mapping_cache_set("user_to_email", _user_to_email)
    return _user_to_email

def _get_proj_to_user() -> dict[str, str]:
    """project_id → user_id  (auditoo.user_prompts) avec cache optimisé."""
    global _proj_to_user

    # Vérifier le cache d'abord
    if (cached := _mapping_cache_get("proj_to_user")) is not None:
        print("   ✓ proj_to_user mapping loaded from cache")
        return cached

    if _proj_to_user is None:
        print("   ⏱️  Loading proj_to_user mapping from Supabase (auditoo.user_prompts)...")
        _start = time.time()
        sb   = get_supabase()
        rows = (
            sb.schema("auditoo")
              .table("user_prompts")
              .select("project_id,user_id")
              .execute()
              .data
            or []
        )
        # on garde le premier user_id rencontré pour un project_id donné
        _proj_to_user = {}
        for r in rows:
            pid = str(r.get("project_id"))
            uid = str(r.get("user_id"))
            if pid and uid and pid not in _proj_to_user:
                _proj_to_user[pid] = uid
        print(f"   ✅ Loaded {len(_proj_to_user)} project mappings in {time.time()-_start:.2f}s")

    # Mettre en cache avec TTL plus long
    _mapping_cache_set("proj_to_user", _proj_to_user)
    return _proj_to_user

def _parse_new_route_data(raw_values_json: Any) -> dict[str, Any]:
    """
    Parse data from the new route POST /projects/%/prompts/chat.

    Handles two cases:
    1. Already-parsed dict (Logfire returns pre-parsed JSON for some records)
    2. Double-JSON-encoded string (for other records)

    The user_prompts array contains ALL historical prompts - we want the LAST one.
    Returns dict with: email, prompt, scope, project_id
    """
    import json
    import re

    try:
        result = {}

        # Case 1: Already a parsed dict
        if isinstance(raw_values_json, dict):
            # Direct access to fields
            auth_info = raw_values_json.get('auth_info', {})
            if auth_info and isinstance(auth_info, dict):
                email = auth_info.get('email')
                if email:
                    result["email utilisateur"] = email

            # Check top-level prompt field (current prompt)
            prompt = raw_values_json.get('prompt', {})
            if prompt and isinstance(prompt, dict):
                text_message = prompt.get('text_message')
                if text_message:
                    result["prompt"] = text_message

                context = prompt.get('context', {})
                if context and isinstance(context, dict):
                    scope = context.get('scope')
                    if scope:
                        result["scope"] = scope

            # Project ID
            project_id = raw_values_json.get('project_id')
            if project_id:
                result["id projet"] = str(project_id)

            return result

        # Case 2: String (double-encoded JSON)
        if isinstance(raw_values_json, str):
            # First level: decode the JSON array value to get the string
            parsed = json.loads(raw_values_json)

            # If still a string, it's double-encoded
            if isinstance(parsed, str):
                # Extract fields using regex (JSON may be truncated)
                # Email from auth_info
                email_match = re.search(r'"email":\s*"([^"]+)"', parsed)
                if email_match:
                    result["email utilisateur"] = email_match.group(1)

                # The user_prompts array contains ALL previous prompts.
                # We want the LAST text_message (most recent prompt).
                text_matches = list(re.finditer(r'"text_message":\s*"([^"]+)"', parsed))
                if text_matches:
                    # Take the last match
                    last_match = text_matches[-1]
                    # Unescape unicode
                    result["prompt"] = last_match.group(1).encode().decode('unicode_escape')

                # Find the last scope
                scope_matches = list(re.finditer(r'"scope":\s*"([^"]+)"', parsed))
                if scope_matches:
                    result["scope"] = scope_matches[-1].group(1)

                # Project ID
                proj_match = re.search(r'"project_id":\s*"([0-9a-f-]+)"', parsed)
                if proj_match:
                    result["id projet"] = proj_match.group(1)

            elif isinstance(parsed, dict):
                # Already parsed after one json.loads - treat as Case 1
                return _parse_new_route_data(parsed)

        return result
    except Exception:
        return {}


def fetch_logfire_events(*, lookback_days: int = 90, limit: int = 20_000, force_refresh: bool = False) -> list[dict]:
    """
    Lit jusqu'à `limit` prompts dans la fenêtre temporelle spécifiée.

    Args:
        lookback_days: Nombre de jours d'historique à récupérer (par défaut 90)
        limit: Limite du nombre d'événements
        force_refresh: Force le rafraîchissement du cache
    """
    _func_start = time.time()
    cache_key = f"events:{lookback_days}:{limit}"  # Cache key includes lookback_days for proper caching

    if force_refresh:
        print("   🔄 Force refresh requested - clearing cache")
        clear_cache()
    elif (cached := _cache_get(cache_key)) is not None:
        print(f"   ✅ Cache HIT for fetch_logfire_events (returning {len(cached)} rows instantly)")
        return cached

    print(f"   ⚠️  Cache MISS - fetching from Logfire (lookback_days={lookback_days}, limit={limit})")

    batch_size   = LOGFIRE_BATCH_SIZE  # Ajusté car Logfire semble limiter à cette valeur
    collected    = []
    offset       = 0

    # Pré-charger les mappings une seule fois
    print("   📊 Pre-loading Supabase mappings...")
    _mappings_start = time.time()
    email_by_user = _get_user_to_email()     # user_id → email
    user_by_proj = _get_proj_to_user()       # project_id → user_id
    print(f"   ✅ Mappings loaded in {time.time()-_mappings_start:.2f}s")

    print("   🔍 Starting Logfire query batches...")
    with LogfireQueryClient(read_token=_LF_READ_TOKEN) as client:
        max_iterations = (limit // batch_size) + 2  # Dynamic based on limit (+2 for safety)
        iteration = 0

        while len(collected) < limit and iteration < max_iterations:
            _batch_start = time.time()
            sql = _PROMPT_SQL_TEMPLATE.format(
                project_guid=_PROJECT_GUID,
                lookback_days=lookback_days,
                limit=batch_size,
                offset=offset,
            )
            retries = 0
            while True:
                try:
                    res = client.query_json_rows(sql=sql)
                    break
                except AssertionError as exc:
                    if "Rate limit exceeded" in str(exc) and retries < _LOGFIRE_RATE_LIMIT_MAX_RETRIES:
                        wait_seconds = _LOGFIRE_RATE_LIMIT_BACKOFF_BASE * (2 ** retries)
                        print(
                            f"   ⏳ Rate limit hit on batch {iteration+1} "
                            f"(retry {retries+1}/{_LOGFIRE_RATE_LIMIT_MAX_RETRIES}) – sleeping {wait_seconds:.2f}s"
                        )
                        time.sleep(wait_seconds)
                        retries += 1
                        continue
                    raise

            batch = res.get("rows", res)

            # Si aucun résultat, on arrête
            if not batch:
                print(f"   ✓ Batch {iteration+1}: No more results")
                break

            print(f"   ✓ Batch {iteration+1}: Fetched {len(batch)} rows in {time.time()-_batch_start:.2f}s")

            # —— complète les e-mails manquants (optimisé) ——
            for row in batch:
                # Pour la nouvelle route /prompts/chat, parser les données si manquantes
                span_name = row.get("span_name", "")
                if "prompts/chat" in span_name and not row.get("prompt"):
                    raw_values = row.get("_raw_values")
                    if raw_values:
                        parsed_data = _parse_new_route_data(raw_values)
                        # Mettre à jour uniquement les champs manquants
                        for key, value in parsed_data.items():
                            if not row.get(key):
                                row[key] = value

                # Enrichissement email si toujours manquant
                if not row.get("email utilisateur"):
                    # 1️⃣ tentative via user_id direct (s'il existe toujours)
                    uid = row.get("user_id")
                    if uid and (email := email_by_user.get(str(uid))):
                        row["email utilisateur"] = email
                    else:
                        # 2️⃣ fallback  project_id → user_id → email
                        pid = row.get("id projet")
                        uid = user_by_proj.get(str(pid)) if pid else None
                        if uid and (email := email_by_user.get(str(uid))):
                            row["email utilisateur"] = email

                # Nettoyage des champs internes
                row.pop("user_id", None)
                row.pop("_raw_values", None)

            collected.extend(batch)

            # Continuer tant qu'on reçoit des résultats (même si moins que batch_size)
            # On s'arrête seulement si on n'a aucun résultat
            offset += len(batch)  # Utiliser len(batch) au lieu de batch_size
            iteration += 1

            if _LOGFIRE_POST_BATCH_DELAY:
                time.sleep(_LOGFIRE_POST_BATCH_DELAY)

    _cache_set(cache_key, collected[:limit])
    print(f"   ✅ fetch_logfire_events COMPLETE: {len(collected[:limit])} rows in {time.time()-_func_start:.2f}s total")
    return collected[:limit]


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_aggregated_dashboard_data(*, lookback_days: int = 30) -> pd.DataFrame:
    """
    Fetches pre-aggregated daily metrics from Logfire for the main dashboard.
    This is much faster than fetching raw events.
    """
    print(f"   🚀 Fetching AGGREGATED dashboard data for the last {lookback_days} days...")
    sql = _AGGREGATED_METRICS_SQL_TEMPLATE.format(
        project_guid=_PROJECT_GUID,
        lookback_days=lookback_days,
    )
    try:
        with LogfireQueryClient(read_token=_LF_READ_TOKEN) as client:
            res = client.query_json_rows(sql=sql)

        df = pd.DataFrame(res.get("rows", res))
        if df.empty:
            return pd.DataFrame(columns=["date", "dau", "total_prompts", "successful_prompts", "liciel_exports"])

        df['date'] = pd.to_datetime(df['date']).dt.date
        df['failed_prompts'] = df['total_prompts'] - df['successful_prompts']
        print(f"   ✅ AGGREGATED data fetched successfully ({len(df)} rows).")
        return df
    except Exception as e:
        st.error(f"Failed to fetch aggregated dashboard data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_weekly_project_counts(*, lookback_days: int = 90) -> pd.DataFrame:
    """
    Fetches the count of unique active projects per week from Logfire.
    Excludes test users (@auditoo.eco and test@test.com).
    """
    print(f"   🚀 Fetching weekly unique project counts for the last {lookback_days} days...")
    sql = _WEEKLY_PROJECTS_SQL_TEMPLATE.format(
        project_guid=_PROJECT_GUID,
        lookback_days=lookback_days,
    )
    try:
        with LogfireQueryClient(read_token=_LF_READ_TOKEN) as client:
            res = client.query_json_rows(sql=sql)

        df = pd.DataFrame(res.get("rows", res))
        if df.empty:
            return pd.DataFrame(columns=["week_start_date", "unique_project_count"])

        # Handle date conversion - week_start_date comes as ISO 8601 string
        if 'week_start_date' in df.columns:
            # Parse ISO 8601 timestamps with timezone
            df['week_start_date'] = pd.to_datetime(df['week_start_date'], format='ISO8601', utc=True)
            # Convert to Europe/Paris timezone
            df['week_start_date'] = df['week_start_date'].dt.tz_convert('Europe/Paris')
            # Extract just the date part
            df['week_start_date'] = df['week_start_date'].dt.date

        print(f"   ✅ Weekly project counts fetched successfully ({len(df)} rows).")
        return df
    except Exception as e:
        st.error(f"Failed to fetch weekly project counts: {e}")
        print(f"   ❌ Error details: {e}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame()


def fetch_weekly_active_users(*, lookback_days: int = 90) -> pd.DataFrame:
    """
    Fetches the true WAU (Weekly Active Users) - total unique users per week from Logfire.
    Excludes test users (@auditoo.eco and test@test.com).
    """
    print(f"   🚀 Fetching true WAU counts for the last {lookback_days} days...")
    sql = _WEEKLY_ACTIVE_USERS_SQL_TEMPLATE.format(
        project_guid=_PROJECT_GUID,
        lookback_days=lookback_days,
    )
    try:
        with LogfireQueryClient(read_token=_LF_READ_TOKEN) as client:
            res = client.query_json_rows(sql=sql)

        df = pd.DataFrame(res.get("rows", res))
        if df.empty:
            return pd.DataFrame(columns=["week_start_date", "wau"])

        # Handle date conversion - week_start_date comes as ISO 8601 string
        if 'week_start_date' in df.columns:
            # Parse ISO 8601 timestamps with timezone
            df['week_start_date'] = pd.to_datetime(df['week_start_date'], format='ISO8601', utc=True)
            # Convert to Europe/Paris timezone
            df['week_start_date'] = df['week_start_date'].dt.tz_convert('Europe/Paris')
            # Extract just the date part
            df['week_start_date'] = df['week_start_date'].dt.date

        print(f"   ✅ Weekly active users fetched successfully ({len(df)} rows).")
        return df
    except Exception as e:
        st.error(f"Failed to fetch weekly active users: {e}")
        print(f"   ❌ Error details: {e}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Project Inspection Functions (for debugging user interactions)
# ---------------------------------------------------------------------------

def construct_logfire_url(trace_id: str, span_id: Optional[str] = None, window: str = "30d") -> str:
    """
    Construct logfire URL to view trace (same format as Home.py make_logfire_url).

    Args:
        trace_id: Logfire trace ID
        span_id: Optional span ID (kept for compatibility, not used)
        window: Time window for trace view (default: 30d)

    Returns:
        URL string to logfire UI
    """
    if not trace_id:
        return ""
    return (
        f"{_LF_PROJECT_URL}"
        f"?q=trace_id%3D%27{trace_id}%27"  # URL-encoded filter
        f"&traceId={trace_id}"              # Pre-selected trace
        f"&last={window}"                   # Time window
    )


def _fetch_trace_spans(trace_id: str) -> list[dict]:
    """
    Fetch all spans for a trace.

    Uses cache to avoid repeated queries for same trace.
    """
    cache_key = f"trace_spans_{trace_id}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    query = f"""
    SELECT
        trace_id,
        span_id,
        parent_span_id,
        span_name,
        start_timestamp AT TIME ZONE 'Europe/Paris' AS "timestamp",
        duration,
        attributes,
        is_exception
    FROM records
    WHERE trace_id = '{trace_id}'
    ORDER BY start_timestamp ASC
    """

    try:
        with LogfireQueryClient(read_token=_LF_READ_TOKEN) as client:
            result = client.query_json_rows(sql=query)
            spans = result.get("rows", result)
            _cache_set(cache_key, spans)
            return spans
    except QueryExecutionError as e:
        st.error(f"Error fetching trace spans: {e}")
        return []


def extract_run_messages(trace_id: str) -> Optional[list[dict]]:
    """
    Extract run_messages from agent run span.

    Returns the pydantic_ai.all_messages list or None if not found.
    """
    spans = _fetch_trace_spans(trace_id)

    # Find the 'agent run' span
    for span in spans:
        if span.get('span_name') == 'agent run':
            attrs = span.get('attributes', {})
            return attrs.get('pydantic_ai.all_messages')

    return None


def extract_agent_response(trace_id: str) -> Optional[dict]:
    """
    Extract agent response details from agent run span.

    Returns dict with: final_result, duration, model_name, tokens
    """
    spans = _fetch_trace_spans(trace_id)

    for span in spans:
        if span.get('span_name') == 'agent run':
            attrs = span.get('attributes', {})
            return {
                'final_result': attrs.get('final_result'),
                'duration': span.get('duration'),
                'model_name': attrs.get('model_name'),
                'input_tokens': attrs.get('gen_ai.usage.input_tokens'),
                'output_tokens': attrs.get('gen_ai.usage.output_tokens'),
            }

    return None


def fetch_project_interactions(
    user_project_id: str,
    *,
    lookback_days: int = 30,
    limit: int = 100
) -> list[dict]:
    """
    Fetch all interactions for a specific user project.

    Args:
        user_project_id: The user's project ID (not logfire project_id)
        lookback_days: Number of days to look back
        limit: Maximum number of interactions to fetch

    Returns:
        List of interaction dicts, each containing:
        - trace_id
        - timestamp
        - email
        - scope
        - prompt_text
        - run_messages (list)
        - agent_response (dict)
        - status ('success' or 'failure')
        - trace_url
        - duration
    """
    # Check cache first
    cache_key = f"project_interactions_{user_project_id}_{lookback_days}_{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Query for prompts
    sql = _INSPECTION_SQL_TEMPLATE.format(
        logfire_project_id=_PROJECT_GUID,
        user_project_id=user_project_id,
        lookback_days=lookback_days,
        limit=limit
    )

    try:
        with LogfireQueryClient(read_token=_LF_READ_TOKEN) as client:
            result = client.query_json_rows(sql=sql)
            prompt_rows = result.get("rows", result)
    except QueryExecutionError as e:
        st.error(f"Error fetching interactions: {e}")
        return []

    # Process each prompt
    interactions = []

    for row in prompt_rows:
        trace_id = row.get('trace_id')
        span_id = row.get('span_id')
        attrs = row.get('attributes', {})

        # Extract user prompt using existing parser
        request_data = attrs.get('fastapi.arguments.values', {})

        # Parse based on route type
        email = None
        prompt_text = None
        scope = None

        if row.get('span_name') == 'POST /projects/{project_id}/message':
            # Old route - direct access
            if isinstance(request_data, dict):
                email = request_data.get('email')
                prompt_text = request_data.get('content')
                scope = request_data.get('scope')
        elif 'transcribe' in row.get('span_name', ''):
            # Transcribe route
            if isinstance(request_data, dict):
                auth_info = request_data.get('auth_info', {})
                context = request_data.get('context', {})
                email = auth_info.get('email') if isinstance(auth_info, dict) else None
                scope = context.get('scope') if isinstance(context, dict) else None
                prompt_text = "[Audio transcription]"
        else:
            # New route - use existing parser
            parsed = _parse_new_route_data(request_data)
            email = parsed.get('email utilisateur')
            prompt_text = parsed.get('prompt')
            scope = parsed.get('scope')

        # Extract run messages and response
        run_messages = extract_run_messages(trace_id)
        agent_response = extract_agent_response(trace_id)

        interaction = {
            'trace_id': trace_id,
            'timestamp': row.get('timestamp'),
            'email': email,
            'scope': scope,
            'prompt_text': prompt_text,
            'run_messages': run_messages or [],
            'agent_response': agent_response or {},
            'status': row.get('status', 'failure'),
            'trace_url': construct_logfire_url(trace_id),
            'duration': row.get('duration', 0),
        }

        interactions.append(interaction)

    # Cache the results
    _cache_set(cache_key, interactions)

    return interactions
