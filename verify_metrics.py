#!/usr/bin/env python3
"""
Independent verification of dashboard metrics using raw Logfire data.
This script calculates last week's metrics from scratch without using
the dashboard's aggregation functions.
"""

import pandas as pd
from src.data_sources import fetch_logfire_events

# Calculate last week's date range
today = pd.Timestamp.now(tz="Europe/Paris").date()
monday = today - pd.Timedelta(days=today.weekday())
last_monday = monday - pd.Timedelta(days=7)
last_sunday = monday - pd.Timedelta(days=1)

print("=" * 70)
print("VERIFICATION INDEPENDANTE - DONNEES RAW LOGFIRE")
print("=" * 70)
print(f"\nSemaine dernière: {last_monday} à {last_sunday}")
print()

# Fetch RAW events from Logfire (not using the aggregated function)
print("📊 Chargement des événements RAW depuis Logfire...")
print("   (limite: 10000 événements, 90 jours lookback)\n")

# Fetch raw events
rows = fetch_logfire_events(lookback_days=90, limit=10000)
df = pd.DataFrame(rows)

if df.empty:
    print("❌ Pas de données!")
    exit(1)

# Parse timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
print(f"✅ {len(df)} événements chargés\n")

# Filter for last week ONLY
last_week_df = df[(df["timestamp"].dt.date >= last_monday) &
                   (df["timestamp"].dt.date <= last_sunday)].copy()

print(f"🔍 Événements de la semaine dernière: {len(last_week_df)}\n")

# METRIC 1: WAU (Weekly Active Users)
print("=" * 70)
print("CALCUL 1: WAU (Weekly Active Users)")
print("=" * 70)
unique_users = last_week_df["email utilisateur"].dropna().unique()
print(f"\nUtilisateurs uniques de la semaine:")
for i, user in enumerate(sorted(unique_users), 1):
    user_events = len(last_week_df[last_week_df["email utilisateur"] == user])
    print(f"  {i:2}. {user:40} ({user_events} événements)")

wau = len(unique_users)
print(f"\n✅ WAU = {wau} utilisateurs uniques")

# METRIC 2: Prompts (count all prompt events)
print("\n" + "=" * 70)
print("CALCUL 2: PROMPTS (événements de type prompt)")
print("=" * 70)
# Prompts are identified by span_name containing "POST /projects/.../message" or "prompts/chat"
_MESSAGE_ROUTE = r"POST /projects/.+/(message|prompts/chat)"
prompts_df = last_week_df[last_week_df["span_name"].str.contains(_MESSAGE_ROUTE, na=False, regex=True)]

print(f"\nNombre total de prompts: {len(prompts_df)}")
print("\nRépartition par jour:")
daily_prompts = prompts_df.groupby(prompts_df["timestamp"].dt.date).size()
for date, count in daily_prompts.items():
    print(f"  {date}: {count:4} prompts")

total_prompts = len(prompts_df)
print(f"\n✅ Total Prompts = {total_prompts}")

# METRIC 3: Export Liciel
print("\n" + "=" * 70)
print("CALCUL 3: EXPORT LICIEL")
print("=" * 70)
_LICIEL_ROUTE = r"GET /projects/.+/liciel"
exports_df = last_week_df[last_week_df["span_name"].str.contains(_LICIEL_ROUTE, na=False, regex=True)]

print(f"\nNombre total d'exports Liciel: {len(exports_df)}")
print("\nRépartition par jour:")
daily_exports = exports_df.groupby(exports_df["timestamp"].dt.date).size()
for date, count in daily_exports.items():
    print(f"  {date}: {count:4} exports")

total_exports = len(exports_df)
print(f"\n✅ Total Export Liciel = {total_exports}")

# METRIC 4: Projets (unique project IDs)
print("\n" + "=" * 70)
print("CALCUL 4: PROJETS UNIQUES")
print("=" * 70)
unique_projects = last_week_df["id projet"].dropna().unique()
print(f"\nNombre de projets uniques: {len(unique_projects)}")
if len(unique_projects) < 50:  # Only print if reasonable number
    print("\nPremiers 20 IDs de projets:")
    for i, proj_id in enumerate(sorted(unique_projects)[:20], 1):
        print(f"  {i:3}. {proj_id}")
    if len(unique_projects) > 20:
        print(f"  ... et {len(unique_projects) - 20} autres projets")

total_projects = len(unique_projects)
print(f"\n✅ Total Projets = {total_projects}")

# FINAL COMPARISON
print("\n" + "=" * 70)
print("COMPARAISON FINALE - CALCUL INDEPENDANT vs DASHBOARD")
print("=" * 70)

expected = {
    'WAU': 13,
    'Projets': 135,
    'Export Liciel': 71,
    'Prompts': 1592
}

actual = {
    'WAU': wau,
    'Projets': total_projects,
    'Export Liciel': total_exports,
    'Prompts': total_prompts
}

print()
for metric, expected_val in expected.items():
    actual_val = actual[metric]
    diff = actual_val - expected_val
    match = "✅ MATCH" if actual_val == expected_val else f"❌ DIFF ({diff:+d})"
    print(f"{metric:20} | Dashboard: {expected_val:6} | Raw calc: {actual_val:6} | {match}")

print("\n" + "=" * 70)
