#!/usr/bin/env python3
"""
liciel_exports.py
-----------------
Liste tous les exports Liciel effectués entre le **lundi 21 juillet 2025 à 00 h 00 (Europe/Paris)** et aujourd’hui.

Usage :
    $ python liciel_exports.py

Le script interroge Logfire via la fonction `fetch_logfire_events`,
filtre les lignes dont `span_name` matche le pattern « GET /project/*/liciel »
et affiche les colonnes principales dans l’ordre chronologique.
"""

from __future__ import annotations

import datetime as dt
import sys
from zoneinfo import ZoneInfo

import pandas as pd

from src.data_sources import fetch_logfire_events
from src.home_helpers import _LICIEL_ROUTE  # regex « GET /project/.+/liciel »

# Fuseau horaire des exports (identique au dashboard)
TZ = ZoneInfo("Europe/Paris")

START = dt.datetime(2025, 7, 21, tzinfo=TZ)  # lundi 21/07/2025 00:00
NOW   = dt.datetime.now(tz=TZ)

# Nombre de jours à remonter depuis START jusqu’à aujourd’hui inclus
lookback_days = (NOW.date() - START.date()).days + 1

try:
    rows = fetch_logfire_events(lookback_days=lookback_days, limit=50_000)
except Exception as exc:
    sys.exit(f"Erreur lors de la requête Logfire : {exc}")

if not rows:
    sys.exit("Aucune ligne Logfire récupérée.")

df = pd.DataFrame(rows)
# Normalise la colonne timestamp → datetime aware Europe/Paris
if "timestamp" not in df.columns:
    sys.exit("Colonne 'timestamp' absente dans les données Logfire.")

df["timestamp"] = (
    pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
      .dt.tz_localize(TZ, nonexistent="shift_forward")
)

# Filtre exports Liciel dans la fenêtre temporelle
liciel_df = (
    df[df["span_name"].str.contains(_LICIEL_ROUTE, na=False, regex=True)]
      .loc[lambda d: d["timestamp"] >= START]
      .sort_values("timestamp")
)

if liciel_df.empty:
    print("Aucun export Liciel trouvé sur la période demandée.")
    sys.exit(0)

# Colonnes d’intérêt (en ajouter / retirer selon besoin)
cols = [
    "timestamp",
    "email utilisateur",
    "id projet",
    "trace_id",
]
print(
    f"\n★ {len(liciel_df)} export(s) Liciel du {START:%d/%m/%Y} au {NOW:%d/%m/%Y} ★\n"
)
print(liciel_df[cols].to_string(index=False))
