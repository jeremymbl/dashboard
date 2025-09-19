#!/usr/bin/env python3
import streamlit as st
from src.data_sources import get_supabase
import sys

# Configuration des secrets (simuler l'environnement Streamlit)
import os
os.environ['SUPABASE_API_URL'] = 'https://htbveqilhnfkktmkwrxw.supabase.co'
os.environ['SUPABASE_SERVICE_ROLE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh0YnZlcWlsaG5ma2t0bWt3cnh3Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MjMxNTQzNiwiZXhwIjoyMDU3ODkxNDM2fQ.O94wdVm6jcMoDy8dtXtXsTz488FtoC5XmuYUBSnvjbE'

# Mock streamlit secrets
class MockSecrets:
    def __getitem__(self, key):
        return os.environ.get(key)
    def get(self, key, default=None):
        return os.environ.get(key, default)

st.secrets = MockSecrets()

# Paramètres utilisateur
if len(sys.argv) != 5:
    print("Usage: python add_user.py <email> <password> <prenom> <nom>")
    print("Exemple: python add_user.py john@test.com motdepasse123 John Doe")
    sys.exit(1)

email = sys.argv[1]
password = sys.argv[2]
prenom = sys.argv[3]
nom = sys.argv[4]

try:
    sb = get_supabase()
    
    print(f"Création de l'utilisateur: {email}")
    
    # 1. Créer l'utilisateur dans auth.users
    auth_response = sb.auth.admin.create_user({
        "email": email,
        "password": password,
        "email_confirm": True  # Confirmer l'email automatiquement
    })
    
    user_id = auth_response.user.id
    print(f"✅ Utilisateur créé dans auth.users avec ID: {user_id}")
    
    # 2. Ajouter les infos dans auditoo.users (avec le bon schéma)
    auditoo_response = sb.schema('auditoo').table('users').insert({
        "id": user_id,
        "email": email,
        "first_name": prenom,
        "last_name": nom,
        "role": "user"
    }).execute()
    
    print(f"✅ Informations ajoutées dans auditoo.users")
    print(f"Email: {email}")
    print(f"Prénom: {prenom}")
    print(f"Nom: {nom}")
    print(f"Rôle: user")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
