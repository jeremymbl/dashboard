import streamlit as st
from src.auth_guard import require_login
import pandas as pd
from src.user_accounts_helpers import create_user_account, get_all_user_accounts, delete_user_account

require_login()



st.title("Comptes utilisateurs")

# ─────────────────────────────────────────────────────────────
# Section 1: Création d'un nouveau compte
# ─────────────────────────────────────────────────────────────

st.subheader("Créer un nouveau compte")

# Initialiser les variables de session pour les messages et le compteur de formulaire
if 'creation_success' not in st.session_state:
    st.session_state.creation_success = None
if 'creation_info' not in st.session_state:
    st.session_state.creation_info = None
if 'deletion_success' not in st.session_state:
    st.session_state.deletion_success = None
if 'deletion_info' not in st.session_state:
    st.session_state.deletion_info = None
if 'form_counter' not in st.session_state:
    st.session_state.form_counter = 0

# Afficher les messages persistants de création s'ils existent
if st.session_state.creation_success:
    st.success(st.session_state.creation_success)
    if st.session_state.creation_info:
        st.info(st.session_state.creation_info)

# Utiliser le compteur pour créer une clé unique qui force le vidage des champs
form_key = f"create_account_form_{st.session_state.form_counter}"

with st.form(form_key):
    col1, col2 = st.columns(2)
    
    with col1:
        email = st.text_input("Email *", placeholder="exemple@domaine.com")
        prenom = st.text_input("Prénom *", placeholder="Jean")
    
    with col2:
        password = st.text_input("Mot de passe *", type="password", placeholder="Mot de passe sécurisé")
        nom = st.text_input("Nom *", placeholder="Dupont")
    
    submitted = st.form_submit_button("Créer le compte", use_container_width=True)
    
    if submitted:
        # Réinitialiser les messages précédents
        st.session_state.creation_success = None
        st.session_state.creation_info = None
        st.session_state.deletion_success = None
        st.session_state.deletion_info = None
        
        # Validation des champs
        if not all([email, password, prenom, nom]):
            st.error("Tous les champs sont obligatoires")
        elif "@" not in email:
            st.error("Format d'email invalide")
        else:
            # Créer le compte
            with st.spinner("Création du compte en cours..."):
                result = create_user_account(email, password, prenom, nom)
                
                if result["success"]:
                    # Stocker les messages dans la session
                    st.session_state.creation_success = f"✅ {result['message']}"
                    st.session_state.creation_info = f"Utilisateur créé: {prenom} {nom} ({email})"
                    # Incrémenter le compteur pour forcer un nouveau formulaire vide
                    st.session_state.form_counter += 1
                    # Recharger la page
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")

st.divider()

# ─────────────────────────────────────────────────────────────
# Section 2: Tableau des comptes existants
# ─────────────────────────────────────────────────────────────

st.subheader("Comptes utilisateurs existants")

# Bouton de rafraîchissement
col_refresh, col_info = st.columns([1, 4])
with col_refresh:
    if st.button("🔄 Actualiser", help="Rafraîchir la liste des utilisateurs"):
        st.rerun()

# Récupérer les données
with st.spinner("Chargement des utilisateurs..."):
    users_df = get_all_user_accounts()

if users_df.empty:
    st.info("Aucun utilisateur trouvé")
else:
    with col_info:
        st.info(f"Total: {len(users_df)} utilisateur(s)")
    
    # Préparer les données pour l'affichage
    display_df = users_df.copy()
    
    # Formater la date de création
    if 'created_at' in display_df.columns:
        display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%d/%m/%Y %H:%M')
    
    # Réorganiser les colonnes
    column_order = ['email', 'password', 'first_name', 'last_name', 'role', 'created_at']
    display_df = display_df[column_order]
    
    # Renommer les colonnes pour l'affichage
    display_df.columns = ['Email', 'Mot de passe', 'Prénom', 'Nom', 'Rôle', 'Créé le']
    
    # Afficher le tableau avec configuration personnalisée
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Email": st.column_config.TextColumn(
                "Email",
                help="Adresse email de l'utilisateur",
                width="medium"
            ),
            "Mot de passe": st.column_config.TextColumn(
                "Mot de passe",
                help="Mot de passe (masqué pour la sécurité)",
                width="small"
            ),
            "Prénom": st.column_config.TextColumn(
                "Prénom",
                width="small"
            ),
            "Nom": st.column_config.TextColumn(
                "Nom",
                width="small"
            ),
            "Rôle": st.column_config.TextColumn(
                "Rôle",
                width="small"
            ),
            "Créé le": st.column_config.TextColumn(
                "Créé le",
                help="Date et heure de création du compte",
                width="medium"
            )
        }
    )
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # Section 3: Suppression d'utilisateurs
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Supprimer un utilisateur")
    
    # Afficher les messages persistants de suppression s'ils existent
    if st.session_state.deletion_success:
        st.success(st.session_state.deletion_success)
        if st.session_state.deletion_info:
            st.info(st.session_state.deletion_info)
    
    # Sélecteur d'utilisateur à supprimer
    user_options = {}
    for _, user in users_df.iterrows():
        display_name = f"{user['first_name']} {user['last_name']} ({user['email']})"
        user_options[display_name] = user['id']
    
    if user_options:
        selected_user = st.selectbox(
            "Sélectionner un utilisateur à supprimer:",
            options=list(user_options.keys()),
            index=None,
            placeholder="Choisir un utilisateur..."
        )
        
        if selected_user:
            user_id = user_options[selected_user]
            
            # Confirmation de suppression
            st.warning(f"⚠️ Vous êtes sur le point de supprimer: **{selected_user}**")
            st.error("Cette action est irréversible !")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("🗑️ Confirmer la suppression", type="primary"):
                    with st.spinner("Suppression en cours..."):
                        result = delete_user_account(user_id)
                        
                        if result["success"]:
                            # Réinitialiser les messages de création
                            st.session_state.creation_success = None
                            st.session_state.creation_info = None
                            # Stocker les messages de suppression dans la session
                            st.session_state.deletion_success = f"✅ {result['message']}"
                            st.session_state.deletion_info = f"Utilisateur supprimé: {selected_user}"
                            # Recharger la page
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")
            
            with col2:
                if st.button("❌ Annuler"):
                    st.rerun()



st.caption("Page de gestion des comptes utilisateurs • Auditoo Dashboard")
