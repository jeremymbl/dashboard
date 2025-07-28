"""
home_helpers.py
---------------
Stats basées sur le CSV fallback (T3_prompts_sample.csv).
Aucune dépendance réseau.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict

import pandas as pd

_CSV = Path("T3_prompts_sample.csv")

def load_prompts_df() -> pd.DataFrame:
    df = pd.read_csv(_CSV, sep=";")
    df.columns = df.columns.str.strip().str.lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df

def get_weekly_metrics(df: pd.DataFrame) -> Dict[str, int]:
    monday = df["timestamp"].max().normalize() - pd.Timedelta(days=df["timestamp"].max().weekday())
    current_week = df[df["timestamp"] >= monday]
    wau = current_week["email utilisateur"].nunique()
    prompts = len(current_week)
    return {"wau": wau, "prompts": prompts}

def daily_prompt_series(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["date"] = tmp["timestamp"].dt.date
    tmp["is_fail"] = tmp["statut"].str.contains("échec", case=False, na=False)
    grp = tmp.groupby("date").agg(
        prompts=("prompt", "count"),
        failed=("is_fail", "sum"),
    )
    return grp.tail(30).reset_index()          # dernier mois

def daily_active_users(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne un DataFrame ‹date, dau› sur les 30 derniers jours.
    Active = ≥ 1 prompt le jour J.
    """
    tmp = df.copy()
    tmp["date"] = tmp["timestamp"].dt.date
    dau = (
        tmp.groupby("date")["email utilisateur"]
        .nunique()                 # n utilisateurs actifs
        .tail(30)                  # fenêtre glissante 30 j
        .reset_index(name="dau")
    )
    return dau


def weekly_active_users(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne ‹week_start, wau› (lundi) sur ~8 semaines.
    WAU = n utilisateurs uniques dans la semaine calendaire.
    """
    tmp = df.copy()
    # Grouper par semaine ISO qui commence le lundi
    tmp["week"] = tmp["timestamp"].dt.to_period("W-MON").apply(lambda p: p.start_time.date())
    wau = (
        tmp.groupby("week")["email utilisateur"]
        .nunique()
        .tail(8)                   # 8 semaines ~ 2 mois
        .reset_index(name="wau")
    )
    return wau