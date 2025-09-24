import json
import datetime as _dt
from typing import List, Optional
from zoneinfo import ZoneInfo

import streamlit as st
from src.auth_guard import require_login
require_login()
import pandas as pd
import numpy as np
import plotly.express as px

from src.data_sources import get_supabase

st.title("Transcriptions – comparatif des services STT")

# ------------------------------------------------------------------
# 0. Utilitaire commun : première colonne existante dans `df`
# ------------------------------------------------------------------

def _find(cands: List[str], df: pd.DataFrame) -> Optional[str]:
    """Retourne la première colonne trouvée parmi *cands* ou None."""
    return next((c for c in cands if c in df.columns), None)

# ------------------------------------------------------------------
# 1. Chargement Supabase ➜ DataFrame unifié (req + results)
# ------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_data(schema: str = "auditoo") -> pd.DataFrame:
    sb = get_supabase()

    req_df = pd.DataFrame(
        sb.schema(schema).table("transcription_requests").select("*").execute().data or []
    )
    res_df = pd.DataFrame(
        sb.schema(schema).table("transcription_results").select("*").execute().data or []
    )
    resp_df = pd.DataFrame(
        sb.schema(schema).table("transcription_responses").select("*").execute().data or []
    )

    if req_df.empty or res_df.empty or resp_df.empty:
        return pd.DataFrame()

    # ————— ne garder que les résultats gagnants —————
    res_id  = _find(["id", "result_id", "uuid"], res_df) or "id"
    link_id = _find(["transcription_result_id", "result_id", "res_id"], resp_df)

    if link_id and res_id in res_df.columns:
        winners = resp_df[link_id].dropna().unique()
        res_df  = res_df[res_df[res_id].isin(winners)]

    # —— identifie timestamps et PK ——
    req_ts = _find(["requested_at", "created_at", "inserted_at", "request_timestamp"], req_df)
    res_ts = _find(["responded_at", "completed_at", "response_timestamp"],                res_df)

    pk_req = _find(["id", "request_id", "pk", "uuid"], req_df) or "id"
    pk_res = _find(["transcription_request_id", "request_id", "req_id"], res_df) or "transcription_request_id"

    # fallback : si timestamp manquant ➜ premier datetime64_* rencontré
    def _first_datetime(df: pd.DataFrame) -> Optional[str]:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return c
        return None

    if req_ts is None:
        req_ts = _first_datetime(req_df)
    if res_ts is None:
        res_ts = _first_datetime(res_df)

    if (req_ts is None) or (res_ts is None) or (pk_req not in req_df.columns) or (pk_res not in res_df.columns):
        return pd.DataFrame()

    # —— standardisation colonnes ——
    req_df = req_df.rename(columns={req_ts: "requested_at", pk_req: "req_id"})
    res_df = res_df.rename(columns={res_ts: "responded_at", pk_res: "req_id"})

    req_df["requested_at"]  = pd.to_datetime(req_df["requested_at"],  errors="coerce")
    res_df["responded_at"] = pd.to_datetime(res_df["responded_at"], errors="coerce")

    df = res_df.merge(
        req_df[["req_id", "requested_at", "audio_duration"]],
        on="req_id", how="left",
    )
    return df

# ------------------------------------------------------------------
# 2. Colonnes auxiliaires (engine, succès, latence)
# ------------------------------------------------------------------

df = load_data()
if df.empty:
    st.warning("Aucune donnée exploitable (timestamps ou clés manquants).")
    st.stop()

# —— détection colonnes timestamps APRES merge ——
REQ_TS = _find(["requested_at", "created_at", "inserted_at"], df)
RES_TS = _find(["responded_at", "completed_at"], df)
if (REQ_TS is None) or (RES_TS is None):
    st.warning("Impossible de localiser les colonnes timestamp dans les données fusionnées.")
    st.stop()

# —— Engine ——
engine_col = _find(["model", "engine"], df) or "model"
df["engine"]   = df[engine_col].fillna("unknown").str.lower()
df["provider"] = df["engine"].str.split(":").str[0]

# —— Latence (s) ——
for _c in (REQ_TS, RES_TS):
    if not pd.api.types.is_datetime64_any_dtype(df[_c]):
        df[_c] = pd.to_datetime(df[_c], errors="coerce")

with pd.option_context("mode.use_inf_as_na", True):
    df["latency_s"] = (df[RES_TS] - df[REQ_TS]).dt.total_seconds()

# —— Succès ——
status_col = _find(["status", "success", "state"], df)
if status_col:
    df["is_success"] = df[status_col].astype(str).str.lower().eq("success")
else:
    df["is_success"] = np.nan

# ------------------------------------------------------------------
# 3. Filtres UI (période + moteur)
# ------------------------------------------------------------------
min_date = df[REQ_TS].min().date()
max_date = df[REQ_TS].max().date()
calendar_max = max(max_date, _dt.datetime.now(ZoneInfo("Europe/Paris")).date())

today_local  = _dt.datetime.now(ZoneInfo("Europe/Paris")).date()
default_start = min_date
default_end   = max(today_local, max_date)  # ouvre sur mois courant si possible
calendar_max  = max(today_local, max_date)

def _normalize_date_range(
    sel,
    *,
    prev: tuple[_dt.date, _dt.date] | None,
    default_start: _dt.date,
    default_end: _dt.date,
    min_value: _dt.date,
    max_value: _dt.date,
) -> tuple[_dt.date, _dt.date]:
    def _to_date(x):
        if isinstance(x, _dt.datetime):
            return x.date()
        if isinstance(x, _dt.date):
            return x
        return None

    if isinstance(sel, (list, tuple)):
        items = []
        for it in sel:
            if isinstance(it, (list, tuple)):
                items.extend(list(it))
            else:
                items.append(it)
    else:
        items = [sel]

    a = _to_date(items[0]) if len(items) >= 1 else None
    b = _to_date(items[1]) if len(items) >= 2 else None

    if a is None and b is None:
        a, b = default_start, default_end
    elif a is None:
        a = prev[0] if prev else default_start
    elif b is None:
        b = prev[1] if prev else a

    minv = _to_date(min_value) or default_start
    maxv = _to_date(max_value) or default_end

    a = max(minv, min(maxv, _to_date(a) or default_end))
    b = max(minv, min(maxv, _to_date(b) or default_end))

    if b < a:
        a, b = b, a
    return a, b

c1, c2 = st.columns(2)
with c1:
    raw_sel = st.date_input(
        "Période",
        value=(default_start, default_end),
        min_value=default_start,
        max_value=calendar_max,
        format="DD/MM/YYYY",
        key="stt_period",
    )
    start_date, end_date = _normalize_date_range(
        raw_sel,
        prev=st.session_state.get("_stt_period_last"),
        default_start=default_start,
        default_end=default_end,
        min_value=default_start,
        max_value=calendar_max,
    )
    st.session_state["_stt_period_last"] = (start_date, end_date)

with c2:
    eng_sel = st.selectbox(
        "Moteur STT",
        options=["Tous"] + sorted(df["engine"].unique().tolist()),
        index=0,
    )

mask = (
    (df[REQ_TS].dt.date >= start_date) &
    (df[REQ_TS].dt.date <= end_date)
)
if eng_sel != "Tous":
    mask &= df["engine"] == eng_sel
sub = df[mask]

if sub.empty:
    st.info("Aucune donnée trouvée pour les filtres sélectionnés.")
    st.stop()

# ------------------------------------------------------------------
# 4. Agrégation par moteur
# ------------------------------------------------------------------
agg = (
    sub.groupby("engine")
       .agg(
           n_calls = ("req_id", "size"),
           avg_lat = ("latency_s", "mean"),
           med_lat = ("latency_s", "median"),
           p90_lat = ("latency_s", lambda s: np.nanpercentile(s.dropna(), 90) if s.notna().any() else np.nan),
       )
       .sort_values("n_calls", ascending=False)
)

# ------------------------------------------------------------------------
# Mises à jour de l’affichage
# ------------------------------------------------------------------------
st.subheader("Tableau comparatif des moteurs")
st.dataframe(
    agg.style.format({
        "avg_lat": "{:.2f} s",
        "med_lat": "{:.2f} s",
        "p90_lat": "{:.2f} s",
    }),
    use_container_width=True,
)


st.subheader("Volume des requêtes par moteur")
fig_calls = px.bar(
    agg.reset_index(),
    x="engine", y="n_calls", text_auto=True,
    labels={"engine": "Moteur", "n_calls": "Requêtes"},
)
fig_calls.update_layout(margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig_calls, use_container_width=True)

st.subheader("Distribution des latences")
fig_lat = px.box(
    sub, x="engine", y="latency_s",
    labels={"engine": "Moteur", "latency_s": "Latence (s)"},
)
fig_lat.update_layout(margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig_lat, use_container_width=True)
