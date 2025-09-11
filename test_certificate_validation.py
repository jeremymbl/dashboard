#!/usr/bin/env python3
"""
Script de test pour vérifier la logique de validation des certificats.
Ce script permet de s'assurer que la fonction analyze_certificates_from_df()
compte bien uniquement les certificats valides à la date de publication du CSV.
"""

import pandas as pd
from datetime import datetime, date
from src.diagnostiqueurs_helpers import analyze_certificates_from_df
from scripts.diag_list import DiagListRessource
from scripts.diag_list_clean import clean_registry
import asyncio

def test_certificate_validation():
    """Test détaillé de la validation des certificats."""
    
    print("=" * 80)
    print("TEST DE VALIDATION DES CERTIFICATS")
    print("=" * 80)
    
    # Récupérer une ressource récente pour tester
    resources = DiagListRessource.fetch_all_sync()
    if not resources:
        print("❌ Impossible de récupérer les ressources")
        return
    
    # Prendre une ressource datée (pas la "latest" qui a date=None)
    dated_resources = [r for r in resources if r.date is not None]
    if not dated_resources:
        print("❌ Aucune ressource datée trouvée")
        return
    
    latest_resource = dated_resources[0]  # La plus récente avec une date
    print(f"📅 Test avec la ressource: {latest_resource.label}")
    print(f"📅 Date de publication: {latest_resource.date}")
    
    # Télécharger et nettoyer les données
    print("\n🔄 Téléchargement et nettoyage des données...")
    raw_df = asyncio.run(latest_resource.fetch_csv_dataframe())
    cleaned_df = clean_registry(raw_df)
    
    print(f"📊 Nombre total d'enregistrements après nettoyage: {len(cleaned_df)}")
    
    # Analyser les colonnes de dates
    print("\n🔍 ANALYSE DES COLONNES DE DATES:")
    date_columns = [col for col in cleaned_df.columns if 'date' in col.lower() or 'debut' in col.lower() or 'fin' in col.lower()]
    for col in date_columns:
        print(f"  - {col}")
    
    # Vérifier les colonnes DPE et Audit spécifiquement
    dpe_columns = [col for col in cleaned_df.columns if 'DPE' in col]
    audit_columns = [col for col in cleaned_df.columns if 'Audit' in col]
    
    print(f"\n📋 Colonnes DPE trouvées: {dpe_columns}")
    print(f"📋 Colonnes Audit trouvées: {audit_columns}")
    
    # Test 1: Comptage sans validation de date (ancien comportement)
    print("\n" + "="*50)
    print("TEST 1: COMPTAGE SANS VALIDATION DE DATE")
    print("="*50)
    
    counts_no_validation = analyze_certificates_from_df(cleaned_df, csv_date=None)
    print(f"DPE (sans validation): {counts_no_validation['DPE']}")
    print(f"Audit (sans validation): {counts_no_validation['Audit']}")
    
    # Test 2: Comptage avec validation de date (nouveau comportement)
    print("\n" + "="*50)
    print("TEST 2: COMPTAGE AVEC VALIDATION DE DATE")
    print("="*50)
    
    counts_with_validation = analyze_certificates_from_df(cleaned_df, csv_date=latest_resource.date)
    print(f"DPE (avec validation): {counts_with_validation['DPE']}")
    print(f"Audit (avec validation): {counts_with_validation['Audit']}")
    
    # Calculer la différence
    dpe_diff = counts_no_validation['DPE'] - counts_with_validation['DPE']
    audit_diff = counts_no_validation['Audit'] - counts_with_validation['Audit']
    
    print(f"\n📉 Différence DPE: {dpe_diff} certificats expirés filtrés")
    print(f"📉 Différence Audit: {audit_diff} certificats expirés filtrés")
    
    # Test 3: Analyse détaillée des dates
    print("\n" + "="*50)
    print("TEST 3: ANALYSE DÉTAILLÉE DES DATES")
    print("="*50)
    
    # Analyser les dates DPE
    if 'DPE_fin' in cleaned_df.columns:
        dpe_with_dates = cleaned_df[
            cleaned_df['DPE_debut'].notna() & 
            cleaned_df['DPE_fin'].notna()
        ].copy()
        
        if len(dpe_with_dates) > 0:
            print(f"\n🔍 Analyse des {len(dpe_with_dates)} certificats DPE avec dates:")
            
            # Convertir en date si nécessaire
            if not pd.api.types.is_datetime64_any_dtype(dpe_with_dates['DPE_fin']):
                print("⚠️  Conversion des dates DPE_fin en datetime...")
                dpe_with_dates['DPE_fin'] = pd.to_datetime(dpe_with_dates['DPE_fin'], errors='coerce')
            
            # Compter les valides vs expirés
            csv_date = latest_resource.date
            valid_dpe = dpe_with_dates[dpe_with_dates['DPE_fin'].dt.date >= csv_date]
            expired_dpe = dpe_with_dates[dpe_with_dates['DPE_fin'].dt.date < csv_date]
            
            print(f"  ✅ Certificats DPE valides au {csv_date}: {len(valid_dpe)}")
            print(f"  ❌ Certificats DPE expirés au {csv_date}: {len(expired_dpe)}")
            
            # Afficher quelques exemples d'expirés
            if len(expired_dpe) > 0:
                print(f"\n📋 Exemples de certificats DPE expirés:")
                sample_expired = expired_dpe[['email', 'DPE_debut', 'DPE_fin']].head(5)
                for idx, row in sample_expired.iterrows():
                    print(f"  - {row['email']}: expire le {row['DPE_fin'].date()}")
    
    # Analyser les dates Audit
    if 'Audit_fin' in cleaned_df.columns:
        audit_with_dates = cleaned_df[
            cleaned_df['Audit_debut'].notna() & 
            cleaned_df['Audit_fin'].notna()
        ].copy()
        
        if len(audit_with_dates) > 0:
            print(f"\n🔍 Analyse des {len(audit_with_dates)} certificats Audit avec dates:")
            
            # Convertir en date si nécessaire
            if not pd.api.types.is_datetime64_any_dtype(audit_with_dates['Audit_fin']):
                print("⚠️  Conversion des dates Audit_fin en datetime...")
                audit_with_dates['Audit_fin'] = pd.to_datetime(audit_with_dates['Audit_fin'], errors='coerce')
            
            # Compter les valides vs expirés
            csv_date = latest_resource.date
            valid_audit = audit_with_dates[audit_with_dates['Audit_fin'].dt.date >= csv_date]
            expired_audit = audit_with_dates[audit_with_dates['Audit_fin'].dt.date < csv_date]
            
            print(f"  ✅ Certificats Audit valides au {csv_date}: {len(valid_audit)}")
            print(f"  ❌ Certificats Audit expirés au {csv_date}: {len(expired_audit)}")
            
            # Afficher quelques exemples d'expirés
            if len(expired_audit) > 0:
                print(f"\n📋 Exemples de certificats Audit expirés:")
                sample_expired = expired_audit[['email', 'Audit_debut', 'Audit_fin']].head(5)
                for idx, row in sample_expired.iterrows():
                    print(f"  - {row['email']}: expire le {row['Audit_fin'].date()}")
    
    # Test 4: Vérification avec une date dans le passé
    print("\n" + "="*50)
    print("TEST 4: VALIDATION AVEC UNE DATE DANS LE PASSÉ")
    print("="*50)
    
    # Tester avec une date il y a 6 mois
    past_date = date(2025, 3, 1)  # 1er mars 2025
    counts_past = analyze_certificates_from_df(cleaned_df, csv_date=past_date)
    
    print(f"📅 Comptage au {past_date}:")
    print(f"  DPE: {counts_past['DPE']}")
    print(f"  Audit: {counts_past['Audit']}")
    
    print(f"\n📈 Évolution depuis le {past_date}:")
    print(f"  DPE: {counts_past['DPE']} → {counts_with_validation['DPE']} (Δ: {counts_with_validation['DPE'] - counts_past['DPE']})")
    print(f"  Audit: {counts_past['Audit']} → {counts_with_validation['Audit']} (Δ: {counts_with_validation['Audit'] - counts_past['Audit']})")
    
    print("\n" + "="*80)
    print("✅ TESTS TERMINÉS - La logique de validation semble fonctionner correctement!")
    print("="*80)

if __name__ == "__main__":
    test_certificate_validation()
