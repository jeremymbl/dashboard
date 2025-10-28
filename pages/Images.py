import streamlit as st
from src.auth_guard import require_login

require_login()

from src.bucket_helpers import get_bucket_images, format_file_size, format_datetime, get_image_author

st.title("Galerie d'images")

# Récupérer le filtre exclude_test depuis session_state (partagé avec Home.py)
exclude_test = st.session_state.get('exclude_test', True)

st.sidebar.info("💡 Le filtre de données de test est géré depuis la page Home")

# Récupération des images
with st.spinner("Chargement des images..."):
    images = get_bucket_images(limit=20)

# Application du filtre d'exclusion des données de test
if exclude_test and images:
    filtered_images = []
    for image in images:
        project_id = image.get('project_id', '')
        author = get_image_author(project_id)

        # Exclure si l'auteur est test@test.com ou se termine par @auditoo.eco
        if author and author.lower() != "n/a":
            author_lower = author.lower()
            if author_lower == "test@test.com" or author_lower.endswith("@auditoo.eco"):
                continue  # Exclure cette image

        filtered_images.append(image)

    images = filtered_images

if not images:
    st.info("Aucune image trouvée dans le bucket." + (" (après exclusion des données de test)" if exclude_test else ""))
else:
    # En-tête avec statistiques
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Images", len(images))
    with col_info2:
        unique_projects = len(set(img.get('project_id', '') for img in images))
        st.metric("Projets", unique_projects)
    with col_info3:
        total_size = sum(img.get('metadata', {}).get('size', 0) for img in images)
        st.metric("Taille totale", format_file_size(total_size))

    st.markdown("---")

    # Slider pour le nombre d'images par ligne
    cols_per_row = st.slider(
        "Images par ligne",
        min_value=2,
        max_value=5,
        value=3,
        step=1,
        help="Nombre d'images à afficher par ligne dans la galerie."
    )

    # CSS personnalisé pour les cartes d'images avec hauteur fixe
    st.markdown("""
    <style>
    .image-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .image-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    .image-metadata {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 8px;
        margin-top: 8px;
        font-size: 0.85em;
        line-height: 1.4;
    }
    .project-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .metadata-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 4px 0;
    }
    .metadata-label {
        color: #666;
        font-weight: 500;
    }
    .metadata-value {
        color: #333;
        font-weight: 400;
    }
    /* Cibler UNIQUEMENT les images dans la section galerie (après le titre T11) */
    .main .block-container > div:last-child div[data-testid="stImage"] > img {
        height: 250px !important;
        width: 100% !important;
        object-fit: cover !important;
        border-radius: 8px;
    }
    .main .block-container > div:last-child div[data-testid="stImage"] {
        height: 250px !important;
        overflow: hidden;
        border-radius: 8px;
    }
    /* Préserver l'affichage normal pour les vues modales/agrandies */
    div[data-testid="stImageViewer"] img,
    div[data-testid="stModal"] img,
    div[data-testid="stImageModal"] img,
    .stModal img {
        height: auto !important;
        width:  auto !important;
        object-fit: contain !important;
        max-width: 100% !important;
        max-height: 100% !important;
    }

    /* ——— Lightbox full-screen sans JS ——— */
    .lightbox {
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.85);
        display: none;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        cursor: zoom-out;
    }
    .lightbox:target {
        display: flex;
    }
    .lightbox img {
        max-width: 90vw;
        max-height: 90vh;
        border-radius: 12px;
        box-shadow: 0 2px 24px rgba(0,0,0,0.4);
    }

    /* ——— Lightbox full-screen sans JS ——— */
    .lightbox {
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.85);
        display: none;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    }
    .lightbox:target {
        display: flex;
    }
    .lightbox img {
        max-width: 90vw;
        max-height: 90vh;
        border-radius: 12px;
        box-shadow: 0 2px 24px rgba(0,0,0,0.4);
    }
    </style>
    """, unsafe_allow_html=True)

    # Affichage en grille avec cartes stylisées
    for i in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)

        for j, col in enumerate(cols):
            if i + j < len(images):
                image = images[i + j]

                with col:
                    # Container avec style personnalisé
                    with st.container():
                        try:
                            # Badge du projet avec tooltip
                            project_id = image.get('project_id', 'N/A')
                            project_short = project_id[:12] + "..." if len(project_id) > 12 else project_id

                            st.markdown(f'<div class="project-badge" title="{project_id}">{project_short}</div>',
                                      unsafe_allow_html=True)

                            # Image principale avec lightbox
                            # 1. Identifiant unique pour chaque lightbox
                            img_id = f"img-{i+j}"

                            # 2. Miniature → ancre #img-X (ouvre la lightbox)
                            # 3. Lightbox cachée (#img-X) : clique n'importe où pour fermer (href="")
                            st.markdown(
                                f'''
                                <a href="#{img_id}" target="_self">
                                    <img src="{image['image_url']}"
                                         style="width:100%; height:250px; object-fit:cover; border-radius:8px; cursor:zoom-in;" />
                                </a>

                                <a href="#" id="{img_id}" class="lightbox" target="_self">
                                    <img src="{image['image_url']}" />
                                </a>
                                ''',
                                unsafe_allow_html=True
                            )

                            # Métadonnées dans un container stylisé
                            metadata = image.get('metadata', {})
                            file_name = image.get('name', 'N/A').split('/')[-1]
                            file_size = format_file_size(metadata.get('size', 0))
                            created_at = format_datetime(image.get('created_at', ''))

                            # Récupération de l'auteur de l'image
                            author = get_image_author(project_id)

                            # Raccourcir le nom de fichier si trop long
                            if len(file_name) > 25:
                                file_name = file_name[:22] + "..."

                            # Raccourcir l'email de l'auteur si trop long
                            author_display = author
                            if author != "N/A" and len(author) > 20:
                                author_display = author[:17] + "..."

                            metadata_html = f"""
                            <div class="image-metadata">
                                <div class="metadata-row">
                                    <span class="metadata-label">Auteur:</span>
                                    <span class="metadata-value" title="{author}">{author_display}</span>
                                </div>
                                <div class="metadata-row">
                                    <span class="metadata-label">Fichier:</span>
                                    <span class="metadata-value">{file_name}</span>
                                </div>
                                <div class="metadata-row">
                                    <span class="metadata-label">Taille:</span>
                                    <span class="metadata-value">{file_size}</span>
                                </div>
                                <div class="metadata-row">
                                    <span class="metadata-label">Créé:</span>
                                    <span class="metadata-value">{created_at}</span>
                                </div>
                            </div>
                            """

                            st.markdown(metadata_html, unsafe_allow_html=True)

                        except Exception as e:
                            st.error("Erreur de chargement")
                            st.caption(f"Détails: {str(e)[:50]}...")

    # Footer avec informations
    st.markdown("---")
    st.caption("Images triées par date de création (plus récentes en premier) • Bucket privé Supabase")
