#!/usr/bin/env python3
"""
diag_list_clean.py
------------------
Nettoie l'annuaire des diagnostiqueurs.

•  Mode local :  --input  chemin/vers/CSV
•  Mode latest  :  --latest  (télécharge le dernier CSV depuis data.gouv.fr)

Exemples
--------
# CSV déjà sur disque
python scripts/diag_list_clean.py -i 20250701-annuaire-diagnostiqueurs.csv -o clean.csv

# Téléchargement + nettoyage en un coup
python scripts/diag_list_clean.py --latest -o clean.csv
"""

from __future__ import annotations

import argparse
import unicodedata as _ud  # pour neutraliser les accents
from pathlib import Path

import pandas as pd

# Importation relative pour l'exécution directe du script
try:
    from scripts.diag_list import print_diag_list_ressources
except ModuleNotFoundError:
    # Fallback pour l'exécution directe du script
    from diag_list import print_diag_list_ressources

# ─────────────────────────────────────────────
# 1)  Logique de nettoyage
# ─────────────────────────────────────────────
ID_KEEP = ["Nom", "Prenom", "Societe", "Adresse", "CP", "Ville", "Tel1", "Tel2", "email"]


def clean_registry(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et agrège un DataFrame issu de l'annuaire."""
    # Hygiène basique
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip().str.replace(r"\s+", " ", regex=True)
    df["email"] = df["email"].str.lower()

    # Fiche la plus complète par email
    df["non_null"] = df[ID_KEEP].notna().sum(1)
    people = (
        df.sort_values(["email", "non_null"], ascending=[True, False]).groupby("email", as_index=False).first()[ID_KEEP]
    )

    # Liste concaténée des certificats par email
    cert_list = (
        df[["email", "Type de certificat"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["email", "Type de certificat"])
        .groupby("email")["Type de certificat"]
        .apply(lambda s: ", ".join(s))
        .reset_index(name="Certificats")
    )

    # Dates début / fin des certificats DPE et Audit
    def _norm(s: str) -> str:
        """enlève accents + passe en minuscules pour comparer."""
        return "".join(c for c in _ud.normalize("NFKD", s) if not _ud.combining(c)).lower()

    DATE_DEBUT_COL = next(
        (c for c in df.columns if "date" in _norm(c) and "debut" in _norm(c) and "validite" in _norm(c)),
        None,
    )
    DATE_FIN_COL = next(
        (c for c in df.columns if "date" in _norm(c) and "fin" in _norm(c) and "validite" in _norm(c)),
        None,
    )

    if DATE_DEBUT_COL and DATE_FIN_COL:
        # Traitement des dates pour les certificats DPE
        dpe_dates = (
            df[df["Type de certificat"].str.contains("DPE", case=False, na=False)][
                ["email", DATE_DEBUT_COL, DATE_FIN_COL]
            ]
            .dropna(subset=["email"])
            .assign(
                **{
                    DATE_DEBUT_COL: lambda x: pd.to_datetime(x[DATE_DEBUT_COL], errors="coerce", dayfirst=True),
                    DATE_FIN_COL: lambda x: pd.to_datetime(x[DATE_FIN_COL], errors="coerce", dayfirst=True),
                }
            )
            .sort_values(["email", DATE_FIN_COL], ascending=[True, False])
            .groupby("email", as_index=False)
            .first()
            .rename(columns={DATE_DEBUT_COL: "DPE_debut", DATE_FIN_COL: "DPE_fin"})
        )
        
        # Traitement des dates pour les certificats Audit énergétique
        audit_dates = (
            df[df["Type de certificat"].str.contains("Audit énergétique", case=False, na=False)][
                ["email", DATE_DEBUT_COL, DATE_FIN_COL]
            ]
            .dropna(subset=["email"])
            .assign(
                **{
                    DATE_DEBUT_COL: lambda x: pd.to_datetime(x[DATE_DEBUT_COL], errors="coerce", dayfirst=False),
                    DATE_FIN_COL: lambda x: pd.to_datetime(x[DATE_FIN_COL], errors="coerce", dayfirst=False),
                }
            )
            # Filtrer les lignes où les deux dates sont valides
            .dropna(subset=[DATE_DEBUT_COL, DATE_FIN_COL])
            .sort_values(["email", DATE_FIN_COL], ascending=[True, False])
            .groupby("email", as_index=False)
            .first()
            .rename(columns={DATE_DEBUT_COL: "Audit_debut", DATE_FIN_COL: "Audit_fin"})
        )
    else:
        dpe_dates = pd.DataFrame(columns=["email", "DPE_debut", "DPE_fin"])
        audit_dates = pd.DataFrame(columns=["email", "Audit_debut", "Audit_fin"])

    # Fusion finale
    result = people.merge(cert_list, on="email", how="left").merge(dpe_dates, on="email", how="left")
    result = result.merge(audit_dates, on="email", how="left")
    
    return result


# ─────────────────────────────────────────────
# 2)  Option --latest : télécharge le CSV récent
# ─────────────────────────────────────────────
def fetch_latest_csv() -> pd.DataFrame:
    """Télécharge le dernier CSV disponible sur data.gouv.fr et le renvoie en DataFrame."""
    import asyncio
    try:
        from scripts.diag_list import DiagListRessource  # import tardif pour garder la dépendance optionnelle
    except ModuleNotFoundError:
        # Fallback pour l'exécution directe du script
        from diag_list import DiagListRessource

    resources = DiagListRessource.fetch_all_sync()
    print_diag_list_ressources(resources)
    if not resources:
        raise RuntimeError("Impossible de récupérer la liste des ressources (data.gouv.fr).")

    latest = resources[0]
    print(f"⬇️  Téléchargement {latest.label} …")
    return asyncio.run(latest.fetch_csv_dataframe())


# ─────────────────────────────────────────────
# 3)  Interface CLI minimale
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", type=Path, help="CSV brut à nettoyer")
    group.add_argument("--latest", action="store_true", help="Télécharge la dernière version du CSV")

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("tmp/diagnostiqueurs_cleaned.csv"),
        help="CSV nettoyé (défaut : tmp/diagnostiqueurs_cleaned.csv)",
    )
    args = parser.parse_args()

    # Charge les données
    if args.latest:
        df_raw = fetch_latest_csv()
    else:
        df_raw = pd.read_csv(args.input, sep=";")

    # Nettoie et sauvegarde
    df_clean = clean_registry(df_raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(args.output, index=False)

    print(f"✅ Nettoyage terminé → {args.output} ({len(df_clean):,} enregistrements uniques)")


if __name__ == "__main__":
    main()
