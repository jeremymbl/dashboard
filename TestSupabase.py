import streamlit as st
import pandas as pd

from src.data_sources import get_supabase

st.title("Test connexion Supabase")

sb = get_supabase()

schema_name = st.text_input("Schéma", "auditoo")
table_name  = st.text_input("Nom de la table", "addresses")
limit       = st.number_input("Nb de lignes", min_value=1, max_value=1000, value=10, step=1)

if st.button("Lancer la requête"):
    with st.status("🔄 Interrogation Supabase…", expanded=False):
        try:
            res = (
                sb.schema(schema_name)
                  .table(table_name)
                  .select("*")
                  .limit(limit)
                  .execute()
            )
            rows = res.data or []
            st.success(f"{len(rows)} ligne(s) récupérée(s) dans {schema_name}.{table_name}.")
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception as exc:
            st.error(f"Erreur Supabase : {exc}")
