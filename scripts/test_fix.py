#!/usr/bin/env python3
"""
Test rapide pour vérifier que les données récentes apparaissent maintenant
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_sources import fetch_logfire_events, clear_cache
import pandas as pd
from datetime import datetime

def test_recent_data():
    print("🧪 Test des données récentes après correction")
    print("=" * 50)
    
    # Forcer le rafraîchissement du cache
    clear_cache()
    
    # Récupérer les données avec la correction
    print("Récupération des données...")
    data = fetch_logfire_events(limit=100, force_refresh=True)
    
    if not data:
        print("❌ Aucune donnée récupérée")
        return
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    
    print(f"✅ {len(data)} événements récupérés")
    print(f"Date la plus récente: {df['timestamp'].max()}")
    print(f"Date la plus ancienne: {df['timestamp'].min()}")
    
    # Vérifier les données du 25 septembre
    today_data = df[df['timestamp'].dt.date == datetime.now().date()]
    print(f"\n📅 Données du 25 septembre: {len(today_data)} événements")
    
    if len(today_data) > 0:
        print("✅ SUCCÈS: Les données récentes apparaissent maintenant!")
        print("\nDerniers événements:")
        recent = df.nlargest(5, 'timestamp')[['timestamp', 'email utilisateur', 'span_name']]
        for idx, row in recent.iterrows():
            email = row['email utilisateur'] or 'N/A'
            print(f"  • {row['timestamp']} | {email} | {row['span_name']}")
    else:
        print("❌ Pas encore de données pour aujourd'hui")
    
    # Vérifier les emails @auditoo.eco
    auditoo_data = df[df['email utilisateur'].str.contains('@auditoo.eco', na=False)]
    print(f"\n📧 Données @auditoo.eco: {len(auditoo_data)} événements")
    
    if len(auditoo_data) > 0:
        print("✅ Les données @auditoo.eco sont visibles")
        recent_auditoo = auditoo_data.nlargest(3, 'timestamp')[['timestamp', 'email utilisateur']]
        for idx, row in recent_auditoo.iterrows():
            print(f"  • {row['timestamp']} | {row['email utilisateur']}")

if __name__ == "__main__":
    test_recent_data()
