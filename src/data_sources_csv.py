"""
data_sources_csv.py
-------------------
Fallback local : lit le CSV exporté depuis Logfire (séparateur « ; »),
et expose la même API que data_sources.fetch_logfire_events.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import pandas as pd
import streamlit as st

_SAMPLE_CSV = Path("T3_prompts_sample.csv")


def load_local_csv(limit: int | None = None) -> List[Dict]:
    """Charge le CSV, nettoie les entêtes, parse la colonne timestamp."""
    if not _SAMPLE_CSV.exists():
        st.warning(f"CSV d'exemple introuvable : {_SAMPLE_CSV}")
        return []

    # CSV Numbers → séparateur « ; »
    df = pd.read_csv(_SAMPLE_CSV, sep=";", header=0)
    df.columns = df.columns.str.strip().str.lower()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")

    if limit is not None:
        df = df.head(limit)

    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# API publique identique à la future version Logfire
# ---------------------------------------------------------------------------
def fetch_logfire_events(
    *,
    lookback_days: int = 14,   # ignoré pour le CSV, gardé pour compat
    limit: int = 5000,
) -> List[Dict]:
    """Renvoie les événements depuis le CSV local."""
    return load_local_csv(limit=limit)
