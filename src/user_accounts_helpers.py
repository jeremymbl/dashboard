#!/usr/bin/env python3
"""
user_accounts_helpers.py
========================
Helpers pour la gestion des comptes utilisateurs avec Supabase Auth et auditoo.users
"""

from typing import List, Dict, Any, Optional
import streamlit as st
from src.data_sources import get_supabase
import pandas as pd


def create_user_account(email: str, password: str, first_name: str, last_name: str) -> Dict[str, Any]:
    """
    Créer un nouveau compte utilisateur dans auth.users et auditoo.users
    
    Args:
        email: Email de l'utilisateur
        password: Mot de passe en clair
        first_name: Prénom
        last_name: Nom de famille
        
    Returns:
        Dict avec le résultat de l'opération
    """
    try:
        sb = get_supabase()
        
        # 1. Créer l'utilisateur dans auth.users avec display_name
        auth_response = sb.auth.admin.create_user({
            "email": email,
            "password": password,
            "email_confirm": True,  # Confirmer l'email automatiquement
            "user_metadata": {
                "first_name": first_name,
                "last_name": last_name,
                "display_name": f"{first_name} {last_name}"
            }
        })
        
        user_id = auth_response.user.id
        
        # 2. Toujours essayer d'insérer dans auditoo.users (ou mettre à jour si existe)
        try:
            # Essayer d'insérer avec le mot de passe en clair
            auditoo_response = sb.schema('auditoo').table('users').insert({
                "id": user_id,
                "email": email,
                "first_name": first_name,
                "last_name": last_name,
                "role": "user",
                "plain_password": password  # Stocker le mot de passe en clair (non recommandé mais demandé)
            }).execute()
        except Exception as insert_error:
            # Si l'insertion échoue (utilisateur existe déjà), faire une mise à jour
            if "duplicate key" in str(insert_error):
                auditoo_response = sb.schema('auditoo').table('users').update({
                    "email": email,
                    "first_name": first_name,
                    "last_name": last_name,
                    "role": "user",
                    "plain_password": password
                }).eq('id', user_id).execute()
            else:
                raise insert_error
        
        return {
            "success": True,
            "user_id": user_id,
            "email": email,
            "first_name": first_name,
            "last_name": last_name,
            "message": "Utilisateur créé avec succès"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Erreur lors de la création: {str(e)}"
        }


def get_all_user_accounts() -> pd.DataFrame:
    """
    Récupérer tous les comptes utilisateurs avec JOIN entre auth.users et auditoo.users
    
    Returns:
        DataFrame avec les informations des utilisateurs
    """
    try:
        sb = get_supabase()
        
        # Récupérer les utilisateurs de auth.users
        auth_users_response = sb.auth.admin.list_users()
        auth_users = auth_users_response
        
        # Récupérer les utilisateurs de auditoo.users
        auditoo_users_response = sb.schema('auditoo').table('users').select('*').execute()
        auditoo_users = auditoo_users_response.data or []
        
        # Créer un dictionnaire pour les infos auditoo
        auditoo_dict = {user['id']: user for user in auditoo_users}
        
        # Construire la liste des utilisateurs combinés
        combined_users = []
        for auth_user in auth_users:
            user_id = auth_user.id
            auditoo_info = auditoo_dict.get(user_id, {})
            
            combined_users.append({
                'id': user_id,
                'email': auth_user.email,
                'password': '***masqué***',  # Masquer le mot de passe pour la sécurité
                'first_name': auditoo_info.get('first_name', ''),
                'last_name': auditoo_info.get('last_name', ''),
                'role': auditoo_info.get('role', 'user'),
                'created_at': auth_user.created_at
            })
        
        return pd.DataFrame(combined_users)
        
    except Exception as e:
        st.error(f"Erreur lors de la récupération des utilisateurs: {e}")
        return pd.DataFrame()


def delete_user_account(user_id: str) -> Dict[str, Any]:
    """
    Supprimer un compte utilisateur de auth.users (cascade vers auditoo.users)
    
    Args:
        user_id: ID de l'utilisateur à supprimer
        
    Returns:
        Dict avec le résultat de l'opération
    """
    try:
        sb = get_supabase()
        
        # Supprimer de auth.users (cascade automatique vers auditoo.users)
        sb.auth.admin.delete_user(user_id)
        
        return {
            "success": True,
            "message": "Utilisateur supprimé avec succès"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Erreur lors de la suppression: {str(e)}"
        }


def get_user_passwords() -> Dict[str, str]:
    """
    Note: Cette fonction est un placeholder car Supabase Auth ne permet pas
    de récupérer les mots de passe en clair pour des raisons de sécurité.
    
    Dans un vrai système, les mots de passe seraient hachés et non récupérables.
    Pour les besoins du CEO, on pourrait stocker les mots de passe en clair
    dans une colonne séparée (non recommandé pour la sécurité).
    
    Returns:
        Dict vide pour l'instant
    """
    return {}
