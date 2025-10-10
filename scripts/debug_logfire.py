#!/usr/bin/env python3
"""
Script de debug pour examiner les données Logfire les plus récentes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_sources import fetch_logfire_events, clear_cache
import pandas as pd
from datetime import datetime, timezone
import json

def debug_logfire_data():
    print("🔍 Debug des données Logfire")
    print("=" * 50)
    
    # 1. Forcer le rafraîchissement du cache
    print("1. Nettoyage du cache...")
    clear_cache()
    
    # 2. Récupérer les données avec différentes limites
    print("\n2. Récupération des données...")
    
    # Test avec une petite limite d'abord
    print("   - Test avec limit=100...")
    data_small = fetch_logfire_events(limit=100, force_refresh=True)
    print(f"   → {len(data_small)} événements récupérés")
    
    if data_small:
        df_small = pd.DataFrame(data_small)
        df_small["timestamp"] = pd.to_datetime(df_small["timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        
        print(f"   → Date la plus récente: {df_small['timestamp'].max()}")
        print(f"   → Date la plus ancienne: {df_small['timestamp'].min()}")
        
        # Afficher les 5 plus récents
        print("\n   📅 Les 5 événements les plus récents:")
        recent = df_small.nlargest(5, 'timestamp')[['timestamp', 'email utilisateur', 'prompt', 'statut']]
        for idx, row in recent.iterrows():
            email = row['email utilisateur'] or 'N/A'
            prompt = (row['prompt'] or 'N/A')[:50] + "..." if len(str(row['prompt'] or '')) > 50 else (row['prompt'] or 'N/A')
            print(f"     • {row['timestamp']} | {email} | {row['statut']} | {prompt}")
    
    # Test avec une limite plus grande
    print("\n   - Test avec limit=1000...")
    data_large = fetch_logfire_events(limit=1000, force_refresh=True)
    print(f"   → {len(data_large)} événements récupérés")
    
    if data_large:
        df_large = pd.DataFrame(data_large)
        df_large["timestamp"] = pd.to_datetime(df_large["timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        
        print(f"   → Date la plus récente: {df_large['timestamp'].max()}")
        print(f"   → Date la plus ancienne: {df_large['timestamp'].min()}")
        
        # Statistiques par jour pour les 10 derniers jours
        print("\n   📊 Événements par jour (10 derniers jours):")
        df_large['date'] = df_large['timestamp'].dt.date
        daily_counts = df_large['date'].value_counts().sort_index().tail(10)
        for date, count in daily_counts.items():
            print(f"     • {date}: {count} événements")
    
    # 3. Vérifier les emails @auditoo.eco spécifiquement
    print("\n3. Vérification des emails @auditoo.eco...")
    if data_large:
        auditoo_emails = df_large[df_large['email utilisateur'].str.contains('@auditoo.eco', na=False)]
        print(f"   → {len(auditoo_emails)} événements avec @auditoo.eco trouvés")
        
        if len(auditoo_emails) > 0:
            print("   📧 Derniers événements @auditoo.eco:")
            recent_auditoo = auditoo_emails.nlargest(3, 'timestamp')[['timestamp', 'email utilisateur', 'statut']]
            for idx, row in recent_auditoo.iterrows():
                print(f"     • {row['timestamp']} | {row['email utilisateur']} | {row['statut']}")
    
    # 4. Informations système
    print(f"\n4. Informations système:")
    print(f"   → Heure actuelle: {datetime.now()}")
    print(f"   → Timezone: {datetime.now().astimezone().tzinfo}")
    
    return data_large

if __name__ == "__main__":
    try:
        data = debug_logfire_data()
        print(f"\n✅ Debug terminé. {len(data) if data else 0} événements au total.")
    except Exception as e:
        print(f"\n❌ Erreur lors du debug: {e}")
        import traceback
        traceback.print_exc()
