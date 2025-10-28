#!/usr/bin/env python3
"""
bucket_helpers.py
=================
Fonctions pour interagir avec le bucket Supabase Storage.
"""

import requests
import streamlit as st
from typing import List, Dict, Any
from datetime import datetime
import base64


@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_bucket_images(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Récupère toutes les images du bucket project-files.
    
    Returns:
        List[Dict]: Liste des images avec leurs métadonnées
    """
    try:
        SUPABASE_URL = st.secrets["SUPABASE_API_URL"]
        SERVICE_ROLE_KEY = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
    except:
        return []
    
    BUCKET_NAME = "project-files"
    all_images = []
    
    # 1. Récupère la liste des projets
    projects = _list_bucket_objects(SUPABASE_URL, SERVICE_ROLE_KEY, BUCKET_NAME)
    
    for project in projects:
        project_id = project.get('name', '')
        if not project_id:
            continue
        
        # 2. Explore le dossier input/image/ de chaque projet
        images_path = f"{project_id}/input/image/"
        images = _list_bucket_objects(SUPABASE_URL, SERVICE_ROLE_KEY, BUCKET_NAME, images_path)
        
        for image in images:
            if image.get('metadata') and _is_image_file(image.get('name', '')):
                # Enrichit avec les infos du projet
                image['project_id'] = project_id
                # Le chemin complet doit inclure project_id/input/image/filename
                image_filename = image.get('name', '').split('/')[-1]  # Extrait juste le nom du fichier
                full_path = f"{project_id}/input/image/{image_filename}"
                image['image_url'] = _get_image_url(SUPABASE_URL, BUCKET_NAME, full_path)
                all_images.append(image)
    
    # Trie par date de création (plus récent en premier)
    all_images.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return all_images[:limit]


def _list_bucket_objects(supabase_url: str, service_key: str, bucket: str, prefix: str = "") -> List[Dict[str, Any]]:
    """Liste les objets d'un bucket avec un préfixe donné."""
    url = f"{supabase_url}/storage/v1/object/list/{bucket}"
    
    headers = {
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "limit": 1000,
        "offset": 0,
        "prefix": prefix
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except:
        return []


def _is_image_file(filename: str) -> bool:
    """Vérifie si un fichier est une image basé sur son extension."""
    if not filename:
        return False
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'}
    return any(filename.lower().endswith(ext) for ext in image_extensions)


def _get_image_url(supabase_url: str, bucket: str, object_path: str) -> str:
    """Télécharge l'image et la convertit en data URL base64."""
    try:
        SERVICE_ROLE_KEY = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
        
        # URL de téléchargement direct
        download_url = f"{supabase_url}/storage/v1/object/{bucket}/{object_path}"
        
        headers = {
            "Authorization": f"Bearer {SERVICE_ROLE_KEY}"
        }
        
        # Télécharge l'image
        response = requests.get(download_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Détermine le type MIME
        content_type = response.headers.get('content-type', 'image/jpeg')
        
        # Convertit en base64
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        
        # Retourne une data URL
        return f"data:{content_type};base64,{image_base64}"
            
    except Exception as e:
        # Debug: afficher l'erreur dans les logs Streamlit
        st.error(f"Erreur téléchargement image {object_path}: {str(e)}")
        print(f"DEBUG - Erreur image {object_path}: {str(e)}")
        print(f"DEBUG - URL: {download_url}")
        # En cas d'erreur, retourne une URL d'image placeholder
        return "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkVycmV1cjwvdGV4dD48L3N2Zz4="


def format_file_size(size_bytes: int) -> str:
    """Formate la taille d'un fichier en unité lisible."""
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB']
    i = 0
    size = float(size_bytes)
    
    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1
    
    return f"{size:.1f} {units[i]}"


def format_datetime(date_str: str) -> str:
    """Formate une date ISO en format lisible."""
    if not date_str:
        return "N/A"
    
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime('%d/%m/%Y %H:%M')
    except:
        return date_str


@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_image_author(project_id: str) -> str:
    """
    Récupère l'auteur d'une image basé sur le project_id.
    Utilise la même logique que dans data_sources.py pour faire le lien
    project_id → user_id → email.
    
    Args:
        project_id: L'ID du projet associé à l'image
        
    Returns:
        str: L'email de l'auteur ou "N/A" si non trouvé
    """
    if not project_id:
        return "N/A"
    
    try:
        # Import ici pour éviter les imports circulaires
        from src.data_sources import get_supabase
        
        sb = get_supabase()
        
        # 1. Récupère le user_id depuis auditoo.user_prompts
        user_prompt_result = (
            sb.schema("auditoo")
              .table("user_prompts")
              .select("user_id")
              .eq("project_id", project_id)
              .limit(1)
              .execute()
        )
        
        if not user_prompt_result.data:
            return "N/A"
        
        user_id = user_prompt_result.data[0].get("user_id")
        if not user_id:
            return "N/A"
        
        # 2. Récupère l'email depuis auditoo.users
        user_result = (
            sb.schema("auditoo")
              .table("users")
              .select("email")
              .eq("id", user_id)
              .limit(1)
              .execute()
        )
        
        if not user_result.data:
            return "N/A"
        
        email = user_result.data[0].get("email")
        return email if email else "N/A"
        
    except Exception as e:
        # En cas d'erreur, retourne N/A silencieusement
        return "N/A"
