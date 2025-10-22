"""
Generate box plot comparing model performance for transcription.

This script creates a box plot visualization showing min, max, mean, median, Q1, Q3
for each transcription model (Mistral, GPT-4o, GPT-4o-mini).

Output: tmp/model_performance_boxplot.png
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_sources import get_supabase


def main():
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 11

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

    # Prepare data
    jobs_df["model_short"] = jobs_df["model"].map(
        {
            "mistral:voxtral-mini-latest": "Mistral",
            "openai:gpt-4o-transcribe": "GPT-4o",
            "openai:gpt-4o-mini-transcribe": "GPT-4o-mini",
        }
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create box plot
    bp = ax.boxplot(
        [
            jobs_df[jobs_df["model_short"] == model]["duration"].values
            for model in ["Mistral", "GPT-4o", "GPT-4o-mini"]
        ],
        labels=["Mistral", "GPT-4o", "GPT-4o-mini"],
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops=dict(
            marker="D",
            markerfacecolor="red",
            markeredgecolor="red",
            markersize=8,
            label="Moyenne",
        ),
    )

    # Color the boxes
    colors = ["#10b981", "#3b82f6", "#8b5cf6"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add statistics text
    y_max = jobs_df["duration"].max() * 1.15
    for i, model in enumerate(["Mistral", "GPT-4o", "GPT-4o-mini"]):
        model_data = jobs_df[jobs_df["model_short"] == model]["duration"]

        stats_text = (
            f"n = {len(model_data)}\n"
            f"Min: {model_data.min():.3f}s\n"
            f"Q1: {model_data.quantile(0.25):.3f}s\n"
            f"Médiane: {model_data.median():.3f}s\n"
            f"Moyenne: {model_data.mean():.3f}s\n"
            f"Q3: {model_data.quantile(0.75):.3f}s\n"
            f"Max: {model_data.max():.3f}s"
        )

        ax.text(
            i + 1,
            y_max * 0.95,
            stats_text,
            ha="center",
            va="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8,
                edgecolor=colors[i],
                linewidth=2,
            ),
            fontsize=10,
            family="monospace",
        )

    # Add title and labels
    ax.set_title(
        "Performance de Transcription par Modèle\n(Période: 20 Oct 2025 - 22 Oct 2025)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylabel("Durée (secondes)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Modèle", fontsize=13, fontweight="bold")

    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add legend for mean marker
    ax.legend([bp["means"][0]], ["Moyenne"], loc="upper right", fontsize=11)

    plt.tight_layout()

    # Save
    output_path = Path(__file__).parent.parent / "tmp" / "model_performance_boxplot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Chart sauvegardé: {output_path}")


if __name__ == "__main__":
    main()
