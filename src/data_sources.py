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
    timeout=10.0,
)

# ---------------------------------------------------------------------------
# Simple in‑memory TTL cache for expensive Logfire queries
# ---------------------------------------------------------------------------

_TTL_SECONDS = 300  # 5 min
_cache: Dict[str, tuple[float, Any]] = {}


def _cache_get(key: str) -> Any | None:
    item = _cache.get(key)
    if item and time.time() - item[0] < _TTL_SECONDS:
        return item[1]
    return None


def _cache_set(key: str, value: Any) -> None:
    _cache[key] = (time.time(), value)


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
