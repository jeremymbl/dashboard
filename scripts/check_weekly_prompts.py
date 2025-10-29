#!/usr/bin/env python3
"""
Diagnostic script to check actual weekly prompt counts
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import datetime as dt
from src.data_sources import fetch_logfire_events

print("=" * 80)
print("WEEKLY PROMPT COUNT DIAGNOSTIC")
print("=" * 80)

# Load MORE data to ensure we capture everything
lookback_days = 21  # 3 weeks to be safe
limit = 10000  # Much higher limit

print(f"\n📥 Loading {lookback_days} days of data with limit={limit}...")
print("(This may take a moment...)\n")

rows = fetch_logfire_events(lookback_days=lookback_days, limit=limit)
print(f"✅ Loaded {len(rows)} total events")

# Convert to DataFrame
df = pd.DataFrame(rows)
if df.empty:
    print("❌ No data loaded!")
    sys.exit(1)

df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")

# Filter for actual prompts (not Liciel exports)
MESSAGE_ROUTE = r"POST /projects/.+/(message|prompts/chat)"
prompts_df = df[df["span_name"].str.contains(MESSAGE_ROUTE, na=False, regex=True)].copy()
print(f"📊 Found {len(prompts_df)} prompts (excluding Liciel exports)")

# Exclude test data
emails = prompts_df["email utilisateur"].str.lower().fillna("")
mask_excl = emails.eq("test@test.com") | emails.str.endswith("@auditoo.eco")
prompts_df = prompts_df[~mask_excl]
print(f"🔍 After excluding test data: {len(prompts_df)} prompts\n")

# Calculate week boundaries
today = pd.Timestamp.now().normalize()
# Find Monday of current week
current_monday = today - pd.Timedelta(days=today.weekday())
last_monday = current_monday - pd.Timedelta(days=7)
last_sunday = current_monday - pd.Timedelta(seconds=1)
two_weeks_ago_monday = last_monday - pd.Timedelta(days=7)
two_weeks_ago_sunday = last_monday - pd.Timedelta(seconds=1)

print("📅 WEEK DEFINITIONS:")
print(f"   Today: {today.date()}")
print(f"   Current week Monday: {current_monday.date()}")
print(f"   Last week: {last_monday.date()} to {last_sunday.date()}")
print(f"   Two weeks ago: {two_weeks_ago_monday.date()} to {two_weeks_ago_sunday.date()}\n")

# Count prompts for each period
current_week = prompts_df[prompts_df["timestamp"] >= current_monday]
last_week = prompts_df[(prompts_df["timestamp"] >= last_monday) & (prompts_df["timestamp"] <= last_sunday)]
two_weeks_ago = prompts_df[(prompts_df["timestamp"] >= two_weeks_ago_monday) & (prompts_df["timestamp"] <= two_weeks_ago_sunday)]

print("=" * 80)
print("📊 PROMPT COUNTS BY WEEK:")
print("=" * 80)
print(f"Current week (since {current_monday.date()}):")
print(f"   🔢 {len(current_week)} prompts")
print(f"   👥 {current_week['email utilisateur'].nunique()} unique users (WAU)")
print()
print(f"Last week ({last_monday.date()} to {last_sunday.date()}):")
print(f"   🔢 {len(last_week)} prompts  ← THIS IS WHAT T.1.2 SHOULD SHOW")
print(f"   👥 {last_week['email utilisateur'].nunique()} unique users (WAU)")
print()
print(f"Two weeks ago ({two_weeks_ago_monday.date()} to {two_weeks_ago_sunday.date()}):")
print(f"   🔢 {len(two_weeks_ago)} prompts")
print(f"   👥 {two_weeks_ago['email utilisateur'].nunique()} unique users (WAU)")
print()

# Daily breakdown for last week
print("=" * 80)
print(f"📅 DAILY BREAKDOWN FOR LAST WEEK ({last_monday.date()} to {last_sunday.date()}):")
print("=" * 80)
last_week_copy = last_week.copy()
last_week_copy["date"] = last_week_copy["timestamp"].dt.date
daily_counts = last_week_copy.groupby("date").size().sort_index()
for date, count in daily_counts.items():
    day_name = pd.Timestamp(date).strftime("%A")
    print(f"   {date} ({day_name}): {count} prompts")

if len(daily_counts) > 0:
    print(f"\n   TOTAL: {daily_counts.sum()} prompts")
else:
    print("   ⚠️  No data for last week!")

# Check if we hit the limit
print("\n" + "=" * 80)
print("⚠️  DATA COMPLETENESS CHECK:")
print("=" * 80)
if len(rows) >= limit * 0.95:
    print(f"❌ WARNING: You loaded {len(rows)} rows, close to the limit of {limit}")
    print("   This suggests there may be MORE data that wasn't loaded!")
    print("   The actual count could be HIGHER than reported above.")
else:
    print(f"✅ OK: Loaded {len(rows)} rows, well below the limit of {limit}")
    print("   The counts above should be accurate.")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print(f"The T.1.2 table should show: {len(last_week)} prompts for last week")
print("If it shows 653, there's a data loading issue (lookback_days or limit too small)")
print("=" * 80)
