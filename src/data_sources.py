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
           AND r.start_timestamp >= '2025-10-13')
       OR (r.span_name ILIKE 'POST /projects/%/transcribe'
           AND r.start_timestamp >= '2025-10-13')
       OR (r.span_name ILIKE 'GET /projects/%/liciel%'
           AND r.start_timestamp >= CURRENT_DATE - INTERVAL {lookback_days} DAY)
      )
  AND  r.project_id = '{project_guid}'
ORDER  BY r.start_timestamp DESC
LIMIT  {limit} OFFSET {offset}
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
        return cached
    
    if _user_to_email is None:
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
    
    # Mettre en cache avec TTL plus long
    _mapping_cache_set("user_to_email", _user_to_email)
    return _user_to_email

def _get_proj_to_user() -> dict[str, str]:
    """project_id → user_id  (auditoo.user_prompts) avec cache optimisé."""
    global _proj_to_user
    
    # Vérifier le cache d'abord
    if (cached := _mapping_cache_get("proj_to_user")) is not None:
        return cached
    
    if _proj_to_user is None:
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
    cache_key = f"events:{lookback_days}:{limit}"  # Cache key includes lookback_days for proper caching

    if force_refresh:
        clear_cache()
    elif (cached := _cache_get(cache_key)) is not None:
        return cached

    batch_size   = 500  # Ajusté à 500 car Logfire semble limiter à cette valeur
    collected    = []
    offset       = 0

    # Pré-charger les mappings une seule fois
    email_by_user = _get_user_to_email()     # user_id → email
    user_by_proj = _get_proj_to_user()       # project_id → user_id

    with LogfireQueryClient(read_token=_LF_READ_TOKEN) as client:
        max_iterations = 10  # Protection contre les boucles infinies
        iteration = 0

        while len(collected) < limit and iteration < max_iterations:
            sql = _PROMPT_SQL_TEMPLATE.format(
                project_guid=_PROJECT_GUID,
                lookback_days=lookback_days,
                limit=batch_size,
                offset=offset,
            )
            res   = client.query_json_rows(sql=sql)
            batch = res.get("rows", res)

            # Si aucun résultat, on arrête
            if not batch:
                break

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
    _cache_set(cache_key, collected[:limit])
    return collected[:limit]
