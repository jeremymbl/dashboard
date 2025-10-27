"""
Analyze quality samples comparing Mistral vs GPT-4o transcriptions.

This script performs a detailed comparison of Mistral and GPT-4o transcription outputs,
identifying cases with high similarity (good quality) and cases with low similarity
(potential issues like hallucinations, difficult audio, etc.).

Output: tmp/quality_analysis/mistral_vs_gpt4o_quality_samples.txt
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from difflib import SequenceMatcher
from src.data_sources import get_supabase


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate fuzzy similarity between two texts using SequenceMatcher."""
    return SequenceMatcher(None, text1, text2).ratio()


def main():
    sb = get_supabase()

    # Filter to new racing strategy (when Mistral was introduced)
    start_date = "2025-10-20T15:57:26+00:00"

    print(f"Fetching data from {start_date} onwards...")

    # Load jobs
    jobs_data = (
        sb.schema("auditoo")
        .table("transcription_jobs")
        .select("*")
        .gte("requested_at", start_date)
        .execute()
        .data
    )
    jobs_df = pd.DataFrame(jobs_data)

    print(f"Loaded {len(jobs_df)} jobs")
    print("Comparing Mistral vs GPT-4o transcriptions...")

    # Compare Mistral vs GPT-4o on same requests
    comparisons = []
    for req_id in jobs_df["transcription_request_id"].unique():
        req_jobs = jobs_df[jobs_df["transcription_request_id"] == req_id]

        mistral_job = req_jobs[req_jobs["model"] == "mistral:voxtral-mini-latest"]
        gpt4o_job = req_jobs[req_jobs["model"] == "openai:gpt-4o-transcribe"]

        if not mistral_job.empty and not gpt4o_job.empty:
            mistral_text = mistral_job.iloc[0]["text"] or ""
            gpt4o_text = gpt4o_job.iloc[0]["text"] or ""

            similarity = calculate_similarity(mistral_text, gpt4o_text)

            comparisons.append(
                {
                    "request_id": req_id,
                    "similarity": similarity,
                    "mistral_text": mistral_text,
                    "gpt4o_text": gpt4o_text,
                    "mistral_duration": mistral_job.iloc[0]["duration"],
                    "gpt4o_duration": gpt4o_job.iloc[0]["duration"],
                }
            )

    comp_df = pd.DataFrame(comparisons)

    print(f"Analyzed {len(comp_df)} comparisons")

    # Save to file
    output_dir = Path(__file__).parent.parent / "tmp" / "quality_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "mistral_vs_gpt4o_quality_samples.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("ANALYSE QUALITÉ: Mistral vs GPT-4o - Échantillons représentatifs\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"📊 STATISTIQUES GLOBALES\n")
        f.write(f"Total de comparaisons: {len(comp_df)}\n")
        f.write(f"Similarité moyenne: {comp_df['similarity'].mean():.1%}\n")
        f.write(f"Similarité médiane: {comp_df['similarity'].median():.1%}\n")
        f.write(
            f"Gain de temps moyen: {(comp_df['gpt4o_duration'] - comp_df['mistral_duration']).mean():.3f}s\n\n"
        )

        # Distribution
        f.write(f"📈 DISTRIBUTION:\n")
        f.write(
            f"  ≥95% similaire: {(comp_df['similarity'] >= 0.95).sum()} ({100*(comp_df['similarity'] >= 0.95).sum()/len(comp_df):.1f}%)\n"
        )
        f.write(
            f"  90-95% similaire: {((comp_df['similarity'] >= 0.90) & (comp_df['similarity'] < 0.95)).sum()} ({100*((comp_df['similarity'] >= 0.90) & (comp_df['similarity'] < 0.95)).sum()/len(comp_df):.1f}%)\n"
        )
        f.write(
            f"  80-90% similaire: {((comp_df['similarity'] >= 0.80) & (comp_df['similarity'] < 0.90)).sum()} ({100*((comp_df['similarity'] >= 0.80) & (comp_df['similarity'] < 0.90)).sum()/len(comp_df):.1f}%)\n"
        )
        f.write(
            f"  <80% différent: {(comp_df['similarity'] < 0.80).sum()} ({100*(comp_df['similarity'] < 0.80).sum()/len(comp_df):.1f}%) ⚠️\n\n"
        )

        f.write("=" * 100 + "\n\n")

        # TOP 10 pires cas
        f.write("❌ TOP 10 PIRES CAS (Similarité la plus faible)\n")
        f.write("=" * 100 + "\n\n")

        for idx, (_, row) in enumerate(
            comp_df.nsmallest(10, "similarity").iterrows(), 1
        ):
            f.write(f"CAS #{idx} - Similarité: {row['similarity']:.1%}\n")
            f.write(f"Request ID: {row['request_id']}\n")
            f.write(
                f"Durées: Mistral={row['mistral_duration']:.3f}s | GPT-4o={row['gpt4o_duration']:.3f}s\n\n"
            )
            f.write(
                f"MISTRAL ({len(row['mistral_text'])} caractères):\n{row['mistral_text']}\n\n"
            )
            f.write(
                f"GPT-4o ({len(row['gpt4o_text'])} caractères):\n{row['gpt4o_text']}\n\n"
            )

            if row["similarity"] < 0.5:
                f.write("⚠️  VERDICT: Très différent - Hallucination ou audio difficile\n")
            elif row["similarity"] < 0.8:
                f.write("⚠️  VERDICT: Différences notables\n")
            else:
                f.write("✅ VERDICT: Différences mineures\n")
            f.write("\n" + "-" * 100 + "\n\n")

        f.write("\n" + "=" * 100 + "\n\n")

        # TOP 10 meilleurs cas
        f.write("✅ TOP 10 MEILLEURS CAS (Similarité la plus élevée)\n")
        f.write("=" * 100 + "\n\n")

        for idx, (_, row) in enumerate(
            comp_df.nlargest(10, "similarity").iterrows(), 1
        ):
            f.write(f"CAS #{idx} - Similarité: {row['similarity']:.1%}\n")
            f.write(f"Request ID: {row['request_id']}\n")
            f.write(
                f"Durées: Mistral={row['mistral_duration']:.3f}s | GPT-4o={row['gpt4o_duration']:.3f}s\n"
            )
            f.write(
                f"Gain de temps: {row['gpt4o_duration'] - row['mistral_duration']:.3f}s\n\n"
            )
            f.write(
                f"MISTRAL ({len(row['mistral_text'])} caractères):\n{row['mistral_text'][:300]}{'...' if len(row['mistral_text']) > 300 else ''}\n\n"
            )
            f.write(
                f"GPT-4o ({len(row['gpt4o_text'])} caractères):\n{row['gpt4o_text'][:300]}{'...' if len(row['gpt4o_text']) > 300 else ''}\n\n"
            )
            f.write("✅ VERDICT: Qualité excellente!\n")
            f.write("\n" + "-" * 100 + "\n\n")

        f.write("\n" + "=" * 100 + "\n\n")

        # Cas moyens
        f.write("📊 ÉCHANTILLON DE CAS MOYENS (Similarité ~90%)\n")
        f.write("=" * 100 + "\n\n")

        medium_cases = comp_df[
            (comp_df["similarity"] >= 0.88) & (comp_df["similarity"] <= 0.92)
        ]
        for idx, (_, row) in enumerate(medium_cases.head(5).iterrows(), 1):
            f.write(f"CAS #{idx} - Similarité: {row['similarity']:.1%}\n")
            f.write(f"Request ID: {row['request_id']}\n")
            f.write(
                f"Durées: Mistral={row['mistral_duration']:.3f}s | GPT-4o={row['gpt4o_duration']:.3f}s\n\n"
            )
            f.write(f"MISTRAL:\n{row['mistral_text']}\n\n")
            f.write(f"GPT-4o:\n{row['gpt4o_text']}\n\n")
            f.write("👍 VERDICT: Qualité très acceptable\n")
            f.write("\n" + "-" * 100 + "\n\n")

        # Summary
        f.write("\n" + "=" * 100 + "\n")
        f.write("💡 SYNTHÈSE POUR LE CTO\n")
        f.write("=" * 100 + "\n\n")

        excellent = (comp_df["similarity"] >= 0.95).sum()
        good = (
            (comp_df["similarity"] >= 0.85) & (comp_df["similarity"] < 0.95)
        ).sum()
        acceptable = (
            (comp_df["similarity"] >= 0.70) & (comp_df["similarity"] < 0.85)
        ).sum()
        poor = (comp_df["similarity"] < 0.70).sum()

        f.write(
            f"✅ {excellent} cas ({100*excellent/len(comp_df):.1f}%) - EXCELLENT (≥95% similarité)\n"
        )
        f.write(
            f"👍 {good} cas ({100*good/len(comp_df):.1f}%) - BON (85-95% similarité)\n"
        )
        f.write(
            f"⚠️  {acceptable} cas ({100*acceptable/len(comp_df):.1f}%) - ACCEPTABLE (70-85% similarité)\n"
        )
        f.write(
            f"❌ {poor} cas ({100*poor/len(comp_df):.1f}%) - PROBLÉMATIQUE (<70% similarité)\n\n"
        )

        f.write(f"RECOMMANDATION:\n")
        if poor / len(comp_df) < 0.10:
            f.write(f"✅ Mistral est fiable! Moins de 10% de cas problématiques.\n")
            f.write(f"   La stratégie de racing actuelle est optimale.\n")
        elif poor / len(comp_df) < 0.20:
            f.write(f"⚠️  Environ {100*poor/len(comp_df):.1f}% de cas problématiques.\n")
            f.write(f"   Garder la stratégie de racing mais surveiller.\n")
        else:
            f.write(f"❌ Trop de cas problématiques ({100*poor/len(comp_df):.1f}%).\n")
            f.write(f"   Considérer d'attendre GPT-4o plus souvent.\n")

        f.write(
            f"\n⏱️  GAIN DE TEMPS: {(comp_df['gpt4o_duration'] - comp_df['mistral_duration']).sum():.1f}s économisés au total\n"
        )
        f.write(
            f"   soit {(comp_df['gpt4o_duration'] - comp_df['mistral_duration']).mean():.3f}s par requête en moyenne\n"
        )

    print(f"\n✅ Rapport détaillé sauvegardé: {output_file}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 RÉSUMÉ RAPIDE:")
    print(f"{'='*80}")
    print(f"  • {len(comp_df)} comparaisons")
    print(f"  • Similarité moyenne: {comp_df['similarity'].mean():.1%}")
    print(f"  • Cas excellents (≥95%): {(comp_df['similarity'] >= 0.95).sum()} ({100*(comp_df['similarity'] >= 0.95).sum()/len(comp_df):.1f}%)")
    print(f"  • Cas problématiques (<70%): {(comp_df['similarity'] < 0.70).sum()} ({100*(comp_df['similarity'] < 0.70).sum()/len(comp_df):.1f}%)")

    # Print hallucination examples
    print(f"\n{'='*80}")
    print(f"🔍 EXEMPLES D'HALLUCINATIONS DE MISTRAL:")
    print(f"{'='*80}")

    worst_cases = comp_df.nsmallest(3, "similarity")
    for idx, (_, row) in enumerate(worst_cases.iterrows(), 1):
        if row['similarity'] < 0.3:
            print(f"\nEXEMPLE #{idx} - Similarité: {row['similarity']:.1%}")
            print(f"Request ID: {row['request_id']}")
            print(f"\n  MISTRAL a transcrit ({len(row['mistral_text'])} chars):")
            print(f"    \"{row['mistral_text'][:150]}{'...' if len(row['mistral_text']) > 150 else ''}\"")
            print(f"\n  GPT-4o a transcrit ({len(row['gpt4o_text'])} chars):")
            print(f"    \"{row['gpt4o_text'][:150]}{'...' if len(row['gpt4o_text']) > 150 else ''}\"")
            print(f"\n  💡 Observation: ", end="")

            mistral_lower = row['mistral_text'].lower()
            gpt4o_lower = row['gpt4o_text'].lower()

            # Detect language mismatch
            spanish_words = ['el', 'de', 'la', 'que', 'es', 'una', 'por']
            french_words = ['le', 'de', 'la', 'est', 'une', 'pour', 'dans']

            mistral_has_spanish = any(word in mistral_lower.split() for word in spanish_words)
            gpt4o_has_french = any(word in gpt4o_lower.split() for word in french_words)

            if mistral_has_spanish and gpt4o_has_french:
                print("Mistral a halluciné en espagnol alors que l'audio est en français!")
            elif len(row['mistral_text']) > len(row['gpt4o_text']) * 3:
                print("Mistral a généré beaucoup trop de texte (hallucination longue)")
            elif len(row['gpt4o_text']) > len(row['mistral_text']) * 3:
                print("Mistral a manqué une grande partie du contenu")
            else:
                print("Transcriptions complètement différentes (audio difficile ou erreur)")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
