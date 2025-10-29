#!/usr/bin/env python3
"""
Instrument the data loading code to understand how many external API calls are
made when the Home page pulls Logfire data.

Run with:
    python scripts/analyze_api_usage.py
    python scripts/analyze_api_usage.py --limit 20000  # stress test
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from logfire.query_client import LogfireQueryClient

import src.data_sources as data_sources
from src.home_helpers import LOGFIRE_ROW_LIMIT

HTTP_CALLS: List[Dict[str, Any]] = []
LOGFIRE_BATCHES: List[Dict[str, Any]] = []


def _fallback_request_url(client: httpx.Client, url: Any, params: Any) -> str:
    base = getattr(client, "base_url", None)
    base_str = str(base) if base else ""
    if base_str and not base_str.endswith("/"):
        base_str += "/"
    url_str = str(url)
    if url_str.startswith(("http://", "https://")):
        full_url = url_str
    else:
        full_url = f"{base_str}{url_str.lstrip('/')}"
    if params:
        query = str(httpx.QueryParams(params))
        separator = "&" if "?" in full_url else "?"
        full_url = f"{full_url}{separator}{query}"
    return full_url


def _extract_int(sql: str, keyword: str) -> Optional[int]:
    pattern = rf"{keyword}\s+(\d+)"
    match = re.search(pattern, sql, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _print_run_summary(*, lookback_days: int, limit: int, outcome: str, rows_returned: Optional[int]) -> None:
    print("\n=== Run summary ===")
    print(f"Outcome: {outcome}")
    if rows_returned is not None:
        print(f"Rows returned: {rows_returned}")
    batch_size = data_sources.LOGFIRE_BATCH_SIZE
    expected_batches = math.ceil(limit / batch_size)
    print(f"Configured lookback_days={lookback_days}, limit={limit}, batch_size={batch_size}")
    print(f"Code allows up to {expected_batches} batches (+2 safety iterations) for this limit.")


def _print_http_summary() -> None:
    print("\n=== HTTP call breakdown ===")
    if not HTTP_CALLS:
        print("No HTTP calls captured.")
        return

    print(f"Total HTTP requests made: {len(HTTP_CALLS)}")
    host_summary: Dict[Optional[str], List[Dict[str, Any]]] = defaultdict(list)
    for entry in HTTP_CALLS:
        host_summary[entry["host"]].append(entry)

    for host, entries in sorted(host_summary.items(), key=lambda kv: len(kv[1]), reverse=True):
        host_label = host or "<unknown>"
        print(f"\nHost: {host_label} ({len(entries)} calls)")
        for idx, entry in enumerate(entries, 1):
            status = entry["status"] if entry["status"] is not None else "?"
            line = f"  {idx:02d}. {entry['method']} {entry['url']} -> {status} ({entry['duration']:.2f}s)"
            if entry["status"] and entry["status"] >= 400:
                line += "  ⚠️"
            print(line)
            if entry["error"]:
                print(f"      error: {entry['error']}")


def _print_logfire_summary() -> None:
    print("\n=== Logfire query batches ===")
    if not LOGFIRE_BATCHES:
        print("No Logfire query batches recorded.")
        return

    for idx, batch in enumerate(LOGFIRE_BATCHES, 1):
        rows = batch["rows"] if batch["rows"] is not None else "?"
        offset = batch["offset"] if batch["offset"] is not None else "?"
        limit = batch["limit"] if batch["limit"] is not None else "?"
        status_bits = []
        if batch["http_status"] is not None:
            status_bits.append(f"http_status={batch['http_status']}")
        if batch["error"]:
            status_bits.append(f"error={batch['error']}")
        status_text = ", ".join(status_bits) if status_bits else "ok"
        print(
            f"  Batch {idx:02d}: offset={offset} limit={limit} rows={rows} "
            f"duration={batch['duration']:.2f}s status={status_text}"
        )
        if batch["http_url"]:
            print(f"      URL: {batch['http_url']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse the API usage during Logfire fetches.")
    parser.add_argument("--lookback-days", type=int, default=7, help="Number of days to fetch from Logfire.")
    parser.add_argument(
        "--limit",
        type=int,
        default=LOGFIRE_ROW_LIMIT,
        help="Maximum number of rows to request from Logfire.",
    )
    args = parser.parse_args()

    lookback_days = args.lookback_days
    limit = args.limit

    # Ensure we actually hit the external services instead of returning cached data.
    data_sources.clear_cache()
    data_sources._user_to_email = None  # type: ignore[attr-defined]
    data_sources._proj_to_user = None   # type: ignore[attr-defined]

    original_http_request = httpx.Client.request
    original_logfire_query = LogfireQueryClient.query_json_rows

    HTTP_CALLS.clear()
    LOGFIRE_BATCHES.clear()

    def instrument_request(self, method: str, url: Any, *args: Any, **kwargs: Any) -> httpx.Response:
        start = time.time()
        response: Optional[httpx.Response] = None
        error_repr: Optional[str] = None
        try:
            response = original_http_request(self, method, url, *args, **kwargs)
            return response
        except Exception as exc:  # pragma: no cover - diagnostic script
            error_repr = repr(exc)
            raise
        finally:
            duration = time.time() - start
            if response is not None:
                request_url = str(response.request.url)
                host = response.request.url.host
                status = response.status_code
            else:
                request_url = _fallback_request_url(self, url, kwargs.get("params"))
                host = urlparse(request_url).netloc or None
                status = None
            HTTP_CALLS.append(
                {
                    "method": method.upper(),
                    "url": request_url,
                    "host": host,
                    "status": status,
                    "duration": duration,
                    "error": error_repr,
                }
            )

    def instrument_query(self, *args: Any, **kwargs: Any) -> Any:
        sql = kwargs.get("sql") or (args[0] if args else "")
        offset = _extract_int(sql, "OFFSET")
        limit_value = _extract_int(sql, "LIMIT")
        http_record_start = len(HTTP_CALLS)
        start = time.time()
        error_repr: Optional[str] = None
        rows_observed: Optional[int] = None

        try:
            result = original_logfire_query(self, *args, **kwargs)
            rows = result.get("rows", result)
            if isinstance(rows, list):
                rows_observed = len(rows)
            return result
        except Exception as exc:  # pragma: no cover - diagnostic script
            error_repr = repr(exc)
            raise
        finally:
            elapsed = time.time() - start
            related_calls = HTTP_CALLS[http_record_start:]
            http_status = related_calls[-1]["status"] if related_calls else None
            http_url = related_calls[-1]["url"] if related_calls else None
            LOGFIRE_BATCHES.append(
                {
                    "offset": offset,
                    "limit": limit_value,
                    "rows": rows_observed,
                    "duration": elapsed,
                    "http_status": http_status,
                    "http_url": http_url,
                    "error": error_repr,
                }
            )

    httpx.Client.request = instrument_request  # type: ignore[assignment]
    LogfireQueryClient.query_json_rows = instrument_query  # type: ignore[assignment]

    outcome = "success"
    rows_returned: Optional[int] = None
    try:
        result = data_sources.fetch_logfire_events(
            lookback_days=lookback_days,
            limit=limit,
            force_refresh=True,
        )
        rows_returned = len(result)
    except Exception as exc:  # pragma: no cover - diagnostic script
        outcome = f"error: {exc}"
        traceback.print_exc()
    finally:
        httpx.Client.request = original_http_request  # type: ignore[assignment]
        LogfireQueryClient.query_json_rows = original_logfire_query  # type: ignore[assignment]

    _print_run_summary(lookback_days=lookback_days, limit=limit, outcome=outcome, rows_returned=rows_returned)
    _print_http_summary()
    _print_logfire_summary()


if __name__ == "__main__":
    main()
