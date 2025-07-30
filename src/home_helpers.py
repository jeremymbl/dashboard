"""
home_helpers.py
---------------
Stats basées sur le CSV fallback (T3_prompts_sample.csv).
Aucune dépendance réseau.
"""

from __future__ import annotations
from typing import Dict
from src.data_sources import fetch_logfire_events
import pandas as pd
import datetime as _dt

_EARLIEST = _dt.date(2025, 6, 1)


def load_prompts_df() -> pd.DataFrame:
    days_back = (_dt.datetime.utcnow().date() - _EARLIEST).days + 1
    rows      = fetch_logfire_events(lookback_days=days_back, limit=20000)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
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

    # échec = tout sauf statut contenant « succès » (plus robuste)
    tmp["is_fail"] = ~tmp["statut"].str.contains("succès", case=False, na=False)

    grp = tmp.groupby("date").agg(
        total=("prompt", "count"),      # total prompts
        failed=("is_fail", "sum"),      # nb échecs
    )
    grp["success"] = grp["total"] - grp["failed"]  # ✅ succès

    # on ne garde que les colonnes tracées
    return (
        grp[["success", "failed"]]
           .tail(30)                    # fenêtre 30 j
           .reset_index()
    )

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

# ------------------------------------------------------------------
#  Exports Liciel / semaine
# ------------------------------------------------------------------
_LICIEL_ROUTE = r"GET /project/.+/liciel"          # pattern trouvé dans Logfire

def weekly_liciel_exports(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne ‹week_start, exports› sur ~8 semaines pour les exports Liciel.
    La fonction s’auto-protège : si « span_name » ou « timestamp » manquent
    ou ne sont pas au bon format, elle renvoie un DataFrame vide.
    """
    if "span_name" not in df.columns or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["week", "exports"])

    # ⤵︎ assure que `timestamp` est un datetime64
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.assign(
            timestamp=pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)
        )

    tmp = df[df["span_name"].str.contains(_LICIEL_ROUTE, na=False, regex=True)].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["week", "exports"])

    tmp["week"] = (
        tmp["timestamp"]
        .dt.to_period("W-SUN")
        .apply(lambda p: p.start_time.date())
    )
    exports = (
        tmp.groupby("week")
           .size()
           .tail(8)
           .reset_index(name="exports")
    )
    return exports


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