#!/usr/bin/env python3
"""
Vérifie que les exports Liciel sont bien récupérés et comptés
après la modification de _PROMPT_SQL_TEMPLATE.
"""

import pandas as pd
from src.data_sources import fetch_logfire_events
from src.home_helpers import weekly_liciel_exports

df = pd.DataFrame(fetch_logfire_events(lookback_days=30, limit=5000))

# 🔄 assure qu’on a bien un datetime pour éviter toute surprise
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)

print(f"Total logfire rows : {len(df)}")
print("Colonnes disponibles :", list(df.columns))

liciel_df = weekly_liciel_exports(df)
print("\nExports Liciel par semaine")
print(liciel_df)

if liciel_df['exports'].sum() == 0:
    print("❌  Toujours zéro — pattern regex à revoir.")
else:
    print("✅  Compte exports Liciel OK :", liciel_df['exports'].sum())
