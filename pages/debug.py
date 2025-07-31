import streamlit as st, json, os, httpx

st.write("Clés détectées dans st.secrets :")
st.json(list(st.secrets.keys()))     # pas les valeurs, juste la liste

url = st.secrets.get("SUPABASE_API_URL")
st.write("URL Supabase :", url)

if url:
    try:
        r = httpx.get(url, timeout=5)
        st.write("Ping status code :", r.status_code)
    except Exception as e:
        st.error(f"Connexion impossible : {e}")
