#!/usr/bin/env python3
"""
Script de debug avancé pour examiner la requête SQL Logfire directement
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from logfire.query_client import LogfireQueryClient
from datetime import datetime, timedelta

def debug_raw_logfire():
    print("🔍 Debug avancé Logfire - Requête SQL directe")
    print("=" * 60)
    
    _LF_READ_TOKEN = st.secrets["LOGFIRE_TOKEN"]
    _PROJECT_GUID = st.secrets["_PROJECT_GUID"]
    
    # 1. Test de la requête la plus simple possible
    print("1. Test de requête basique...")
    
    simple_sql = f"""
    SELECT 
        DATE_FORMAT(r.start_timestamp AT TIME ZONE 'Europe/Paris', '%d/%m/%Y %H:%M:%S') AS timestamp,
        r.span_name,
        r.project_id
    FROM records r
    WHERE r.project_id = '{_PROJECT_GUID}'
    ORDER BY r.start_timestamp DESC
    LIMIT 10
    """
    
    with LogfireQueryClient(read_token=_LF_READ_TOKEN) as client:
        print("   → Exécution de la requête basique...")
        try:
            result = client.query_json_rows(sql=simple_sql)
            rows = result.get("rows", result)
            print(f"   → {len(rows)} lignes récupérées")
            
            if rows:
                print("   📅 Les 10 événements les plus récents (tous types):")
                for row in rows[:10]:
                    print(f"     • {row.get('timestamp')} | {row.get('span_name')}")
            else:
                print("   ❌ Aucune donnée récupérée")
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    # 2. Test avec filtre de date récente
    print("\n2. Test avec filtre de date récente (depuis le 20 septembre)...")
    
    recent_sql = f"""
    SELECT 
        DATE_FORMAT(r.start_timestamp AT TIME ZONE 'Europe/Paris', '%d/%m/%Y %H:%M:%S') AS timestamp,
        r.span_name,
        COALESCE(
            r.attributes -> 'fastapi.arguments.values' -> 'auth_info' ->> 'email',
            r.attributes -> 'user_email' ->> 0
        ) AS email
    FROM records r
    WHERE r.project_id = '{_PROJECT_GUID}'
      AND r.start_timestamp >= '2025-09-20'
    ORDER BY r.start_timestamp DESC
    LIMIT 50
    """
    
    with LogfireQueryClient(read_token=_LF_READ_TOKEN) as client:
        print("   → Exécution de la requête avec filtre récent...")
        try:
            result = client.query_json_rows(sql=recent_sql)
            rows = result.get("rows", result)
            print(f"   → {len(rows)} lignes récupérées depuis le 20 septembre")
            
            if rows:
                print("   📅 Événements depuis le 20 septembre:")
                for row in rows:
                    email = row.get('email') or 'N/A'
                    print(f"     • {row.get('timestamp')} | {email} | {row.get('span_name')}")
            else:
                print("   ℹ️ Aucun événement depuis le 20 septembre")
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    # 3. Test avec une plage de dates plus large
    print("\n3. Test avec plage étendue (depuis le 10 septembre)...")
    
    extended_sql = f"""
    SELECT 
        DATE_FORMAT(r.start_timestamp AT TIME ZONE 'Europe/Paris', '%d/%m/%Y %H:%M:%S') AS timestamp,
        r.span_name,
        COALESCE(
            r.attributes -> 'fastapi.arguments.values' -> 'auth_info' ->> 'email',
            r.attributes -> 'user_email' ->> 0
        ) AS email
    FROM records r
    WHERE r.project_id = '{_PROJECT_GUID}'
      AND r.start_timestamp >= '2025-09-10'
      AND (r.span_name ILIKE 'POST /project/%/message' OR r.span_name ILIKE 'GET /project/%/liciel%')
    ORDER BY r.start_timestamp DESC
    LIMIT 100
    """
    
    with LogfireQueryClient(read_token=_LF_READ_TOKEN) as client:
        print("   → Exécution de la requête étendue...")
        try:
            result = client.query_json_rows(sql=extended_sql)
            rows = result.get("rows", result)
            print(f"   → {len(rows)} lignes récupérées depuis le 10 septembre")
            
            if rows:
                # Grouper par date
                dates = {}
                for row in rows:
                    date = row.get('timestamp', '').split(' ')[0]
                    if date not in dates:
                        dates[date] = 0
                    dates[date] += 1
                
                print("   📊 Répartition par date:")
                for date in sorted(dates.keys(), reverse=True):
                    print(f"     • {date}: {dates[date]} événements")
                    
                # Afficher les plus récents
                print("\n   📅 Les 5 plus récents:")
                for row in rows[:5]:
                    email = row.get('email') or 'N/A'
                    print(f"     • {row.get('timestamp')} | {email} | {row.get('span_name')}")
            else:
                print("   ℹ️ Aucun événement trouvé")
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    # 4. Informations sur le projet
    print(f"\n4. Informations de configuration:")
    print(f"   → PROJECT_GUID: {_PROJECT_GUID}")
    print(f"   → Heure actuelle: {datetime.now()}")

if __name__ == "__main__":
    try:
        debug_raw_logfire()
        print(f"\n✅ Debug avancé terminé.")
    except Exception as e:
        print(f"\n❌ Erreur lors du debug avancé: {e}")
        import traceback
        traceback.print_exc()
