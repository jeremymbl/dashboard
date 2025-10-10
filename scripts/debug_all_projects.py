#!/usr/bin/env python3
"""
Script pour chercher des prompts récents dans TOUS les projets Logfire
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from logfire.query_client import LogfireQueryClient
from datetime import datetime, timedelta

def debug_all_projects():
    print("🔍 Recherche de prompts récents dans TOUS les projets")
    print("=" * 60)
    
    _LF_READ_TOKEN = st.secrets["LOGFIRE_TOKEN"]
    _PROJECT_GUID = st.secrets["_PROJECT_GUID"]
    
    # 1. Chercher tous les prompts récents (tous projets)
    print("1. Recherche de prompts récents (tous projets)...")
    
    all_prompts_sql = """
    SELECT 
        DATE_FORMAT(r.start_timestamp AT TIME ZONE 'Europe/Paris', '%d/%m/%Y %H:%M:%S') AS timestamp,
        r.project_id,
        r.span_name,
        COALESCE(
            r.attributes -> 'fastapi.arguments.values' -> 'auth_info' ->> 'email',
            r.attributes -> 'user_email' ->> 0
        ) AS email,
        COALESCE(
            r.attributes -> 'fastapi.arguments.values' ->> 'content',
            r.attributes -> 'fastapi.arguments.values' -> 'prompt' ->> 'content'
        ) AS prompt_content
    FROM records r
    WHERE r.span_name ILIKE 'POST /project/%/message'
      AND r.start_timestamp >= '2025-09-20'
    ORDER BY r.start_timestamp DESC
    LIMIT 50
    """
    
    with LogfireQueryClient(read_token=_LF_READ_TOKEN) as client:
        print("   → Recherche dans tous les projets depuis le 20 septembre...")
        try:
            result = client.query_json_rows(sql=all_prompts_sql)
            rows = result.get("rows", result)
            print(f"   → {len(rows)} prompts trouvés depuis le 20 septembre")
            
            if rows:
                print("   📅 Prompts récents trouvés:")
                for row in rows:
                    email = row.get('email') or 'N/A'
                    project = row.get('project_id') or 'N/A'
                    content = (row.get('prompt_content') or 'N/A')[:50] + "..." if len(str(row.get('prompt_content') or '')) > 50 else (row.get('prompt_content') or 'N/A')
                    is_current_project = "✅" if project == _PROJECT_GUID else "❌"
                    print(f"     {is_current_project} {row.get('timestamp')} | {email} | {project[:8]}... | {content}")
            else:
                print("   ℹ️ Aucun prompt trouvé depuis le 20 septembre dans AUCUN projet")
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    # 2. Chercher spécifiquement les emails @auditoo.eco récents
    print("\n2. Recherche spécifique des emails @auditoo.eco récents...")
    
    auditoo_sql = """
    SELECT 
        DATE_FORMAT(r.start_timestamp AT TIME ZONE 'Europe/Paris', '%d/%m/%Y %H:%M:%S') AS timestamp,
        r.project_id,
        r.span_name,
        COALESCE(
            r.attributes -> 'fastapi.arguments.values' -> 'auth_info' ->> 'email',
            r.attributes -> 'user_email' ->> 0
        ) AS email
    FROM records r
    WHERE (
        r.attributes -> 'fastapi.arguments.values' -> 'auth_info' ->> 'email' ILIKE '%@auditoo.eco'
        OR r.attributes -> 'user_email' ->> 0 ILIKE '%@auditoo.eco'
    )
    AND r.start_timestamp >= '2025-09-20'
    ORDER BY r.start_timestamp DESC
    LIMIT 50
    """
    
    with LogfireQueryClient(read_token=_LF_READ_TOKEN) as client:
        print("   → Recherche des activités @auditoo.eco depuis le 20 septembre...")
        try:
            result = client.query_json_rows(sql=auditoo_sql)
            rows = result.get("rows", result)
            print(f"   → {len(rows)} activités @auditoo.eco trouvées")
            
            if rows:
                print("   📧 Activités @auditoo.eco récentes:")
                for row in rows:
                    email = row.get('email') or 'N/A'
                    project = row.get('project_id') or 'N/A'
                    span = row.get('span_name') or 'N/A'
                    is_current_project = "✅" if project == _PROJECT_GUID else "❌"
                    print(f"     {is_current_project} {row.get('timestamp')} | {email} | {span}")
            else:
                print("   ℹ️ Aucune activité @auditoo.eco depuis le 20 septembre")
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    # 3. Vérifier les projets actifs
    print("\n3. Liste des projets avec activité récente...")
    
    projects_sql = """
    SELECT 
        r.project_id,
        COUNT(*) as event_count,
        MAX(DATE_FORMAT(r.start_timestamp AT TIME ZONE 'Europe/Paris', '%d/%m/%Y %H:%M:%S')) as last_activity
    FROM records r
    WHERE r.start_timestamp >= '2025-09-20'
    GROUP BY r.project_id
    ORDER BY MAX(r.start_timestamp) DESC
    LIMIT 10
    """
    
    with LogfireQueryClient(read_token=_LF_READ_TOKEN) as client:
        print("   → Analyse des projets actifs...")
        try:
            result = client.query_json_rows(sql=projects_sql)
            rows = result.get("rows", result)
            print(f"   → {len(rows)} projets avec activité récente")
            
            if rows:
                print("   📊 Projets actifs depuis le 20 septembre:")
                for row in rows:
                    project = row.get('project_id') or 'N/A'
                    count = row.get('event_count') or 0
                    last = row.get('last_activity') or 'N/A'
                    is_current = "✅ ACTUEL" if project == _PROJECT_GUID else "  "
                    print(f"     {is_current} {project} | {count} événements | Dernier: {last}")
            else:
                print("   ℹ️ Aucun projet actif trouvé")
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    print(f"\n4. Configuration actuelle:")
    print(f"   → Projet configuré: {_PROJECT_GUID}")

if __name__ == "__main__":
    try:
        debug_all_projects()
        print(f"\n✅ Recherche terminée.")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
