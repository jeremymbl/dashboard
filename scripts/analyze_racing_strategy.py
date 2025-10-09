#!/usr/bin/env python3
"""
Racing Strategy Analysis Script
Outputs pure numerical data about transcription racing strategy effectiveness.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from src.data_sources import get_supabase


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate fuzzy similarity between two texts."""
    return SequenceMatcher(None, text1, text2).ratio()


def fetch_all_rows(sb, schema: str, table_name: str, select: str = "*"):
    """Fetch all rows from a table with pagination."""
    all_data = []
    offset = 0
    batch_size = 1000
    while True:
        batch = (
            sb.schema(schema)
            .table(table_name)
            .select(select)
            .range(offset, offset + batch_size - 1)
            .execute()
            .data
        )
        if not batch:
            break
        all_data.extend(batch)
        if len(batch) < batch_size:
            break
        offset += batch_size
    return all_data


def main():
    sb = get_supabase()

    # Load data
    requests_df = pd.DataFrame(fetch_all_rows(sb, "auditoo", "transcription_requests"))
    jobs_df = pd.DataFrame(fetch_all_rows(sb, "auditoo", "transcription_jobs"))
    responses_df = pd.DataFrame(
        fetch_all_rows(sb, "auditoo", "transcription_responses")
    )

    if requests_df.empty or jobs_df.empty or responses_df.empty:
        print("ERROR: No data")
        return

    # Parse timestamps
    jobs_df["requested_at"] = pd.to_datetime(jobs_df["requested_at"], errors="coerce")
    jobs_df["responded_at"] = pd.to_datetime(jobs_df["responded_at"], errors="coerce")
    jobs_df["duration"] = (
        jobs_df["responded_at"] - jobs_df["requested_at"]
    ).dt.total_seconds()

    # Get winner mapping
    winner_map = responses_df.set_index("transcription_request_id")[
        "transcription_result_id"
    ].to_dict()

    print("=" * 80)
    print("RACING STRATEGY ANALYSIS - NUMERICAL DATA")
    print("=" * 80)

    # === SECTION 1: OVERALL STATISTICS ===
    print("\n[OVERALL_STATS]")
    print(f"total_requests,{len(requests_df)}")
    print(f"total_jobs,{len(jobs_df)}")
    print(f"total_responses,{len(responses_df)}")
    print(f"jobs_per_request,{len(jobs_df) / len(requests_df):.2f}")

    # === SECTION 2: MODEL PERFORMANCE ===
    print("\n[MODEL_PERFORMANCE]")
    print("model,total_jobs,avg_duration_s,median_duration_s,p90_duration_s,success_rate")
    for model in sorted(jobs_df["model"].unique()):
        model_jobs = jobs_df[jobs_df["model"] == model]
        success = model_jobs[model_jobs["status"] == "success"]
        print(
            f"{model},{len(model_jobs)},{success['duration'].mean():.3f},"
            f"{success['duration'].median():.3f},"
            f"{success['duration'].quantile(0.9):.3f},"
            f"{len(success) / len(model_jobs):.4f}"
        )

    # === SECTION 3: RACING ANALYSIS ===
    print("\n[RACING_ANALYSIS]")

    racing_data = []
    for req_id in jobs_df["transcription_request_id"].unique():
        req_jobs = jobs_df[jobs_df["transcription_request_id"] == req_id]
        if len(req_jobs) < 2:
            continue

        # Find fastest
        fastest_idx = req_jobs["duration"].idxmin()
        fastest = req_jobs.loc[fastest_idx]

        # Find winner
        winner_id = winner_map.get(req_id)
        if not winner_id:
            continue

        winner = req_jobs[req_jobs["id"] == winner_id]
        if winner.empty:
            continue
        winner = winner.iloc[0]

        # Calculate fuzzy similarity if texts differ
        fastest_text = fastest["text"] or ""
        winner_text = winner["text"] or ""
        similarity = calculate_similarity(fastest_text, winner_text)

        racing_data.append(
            {
                "request_id": req_id,
                "fastest_model": fastest["model"],
                "fastest_duration": fastest["duration"],
                "winner_model": winner["model"],
                "winner_duration": winner["duration"],
                "time_wasted": winner["duration"] - fastest["duration"],
                "strategy_used": fastest["model"] != winner["model"],
                "similarity": similarity,
                "char_diff": abs(len(fastest_text) - len(winner_text)),
            }
        )

    race_df = pd.DataFrame(racing_data)

    # Summary stats
    print(f"analyzable_requests,{len(race_df)}")
    print(f"strategy_used_count,{race_df['strategy_used'].sum()}")
    print(f"strategy_used_pct,{100 * race_df['strategy_used'].mean():.2f}")
    print(f"avg_time_wasted_s,{race_df[race_df['strategy_used']]['time_wasted'].mean():.3f}")
    print(f"median_time_wasted_s,{race_df[race_df['strategy_used']]['time_wasted'].median():.3f}")
    print(f"p90_time_wasted_s,{race_df[race_df['strategy_used']]['time_wasted'].quantile(0.9):.3f}")
    print(f"p95_time_wasted_s,{race_df[race_df['strategy_used']]['time_wasted'].quantile(0.95):.3f}")
    print(f"total_time_wasted_s,{race_df[race_df['strategy_used']]['time_wasted'].sum():.1f}")

    # === SECTION 4: FASTEST MODEL BREAKDOWN ===
    print("\n[FASTEST_MODEL_BREAKDOWN]")
    print("model,count,pct")
    fastest_counts = race_df["fastest_model"].value_counts()
    for model, count in fastest_counts.items():
        print(f"{model},{count},{100 * count / len(race_df):.2f}")

    # === SECTION 5: WINNER MODEL BREAKDOWN ===
    print("\n[WINNER_MODEL_BREAKDOWN]")
    print("model,count,pct")
    winner_counts = race_df["winner_model"].value_counts()
    for model, count in winner_counts.items():
        print(f"{model},{count},{100 * count / len(race_df):.2f}")

    # === SECTION 6: SIMILARITY ANALYSIS (FUZZY MATCHING) ===
    print("\n[SIMILARITY_ANALYSIS]")
    strategy_used = race_df[race_df["strategy_used"]]
    print(f"avg_similarity,{strategy_used['similarity'].mean():.4f}")
    print(f"median_similarity,{strategy_used['similarity'].median():.4f}")
    print(f"p10_similarity,{strategy_used['similarity'].quantile(0.1):.4f}")
    print(f"p90_similarity,{strategy_used['similarity'].quantile(0.9):.4f}")
    print(f"highly_similar_95pct,{(strategy_used['similarity'] >= 0.95).sum()}")
    print(f"highly_similar_95pct_pct,{100 * (strategy_used['similarity'] >= 0.95).mean():.2f}")
    print(f"similar_90pct,{(strategy_used['similarity'] >= 0.90).sum()}")
    print(f"similar_90pct_pct,{100 * (strategy_used['similarity'] >= 0.90).mean():.2f}")
    print(f"similar_85pct,{(strategy_used['similarity'] >= 0.85).sum()}")
    print(f"similar_85pct_pct,{100 * (strategy_used['similarity'] >= 0.85).mean():.2f}")
    print(f"avg_char_diff,{strategy_used['char_diff'].mean():.1f}")
    print(f"median_char_diff,{strategy_used['char_diff'].median():.0f}")

    # === SECTION 7: CROSS-TABULATION (FASTEST vs WINNER) ===
    print("\n[CROSS_TAB_FASTEST_VS_WINNER]")
    crosstab = pd.crosstab(race_df["fastest_model"], race_df["winner_model"])
    print("fastest_model," + ",".join(crosstab.columns))
    for idx in crosstab.index:
        row_data = ",".join(str(crosstab.loc[idx, col]) for col in crosstab.columns)
        print(f"{idx},{row_data}")

    # === SECTION 8: GPT-4o vs GPT-4o-mini COMPARISON ===
    print("\n[GPT4O_VS_MINI]")
    gpt4o_mini_comparison = []
    for req_id in jobs_df["transcription_request_id"].unique():
        req_jobs = jobs_df[jobs_df["transcription_request_id"] == req_id]

        gpt4o = req_jobs[req_jobs["model"] == "openai:gpt-4o-transcribe"]
        mini = req_jobs[req_jobs["model"] == "openai:gpt-4o-mini-transcribe"]

        if not gpt4o.empty and not mini.empty:
            gpt4o_row = gpt4o.iloc[0]
            mini_row = mini.iloc[0]

            gpt4o_text = gpt4o_row["text"] or ""
            mini_text = mini_row["text"] or ""
            similarity = calculate_similarity(gpt4o_text, mini_text)

            gpt4o_mini_comparison.append(
                {
                    "gpt4o_duration": gpt4o_row["duration"],
                    "mini_duration": mini_row["duration"],
                    "latency_diff": gpt4o_row["duration"] - mini_row["duration"],
                    "similarity": similarity,
                    "highly_similar": similarity >= 0.95,
                    "mini_faster": mini_row["duration"] < gpt4o_row["duration"],
                }
            )

    if gpt4o_mini_comparison:
        comp_df = pd.DataFrame(gpt4o_mini_comparison)
        print(f"total_comparisons,{len(comp_df)}")
        print(f"avg_similarity,{comp_df['similarity'].mean():.4f}")
        print(f"median_similarity,{comp_df['similarity'].median():.4f}")
        print(f"highly_similar_95pct_count,{comp_df['highly_similar'].sum()}")
        print(f"highly_similar_95pct_pct,{100 * comp_df['highly_similar'].mean():.2f}")
        print(f"similar_90pct_count,{(comp_df['similarity'] >= 0.9).sum()}")
        print(f"similar_90pct_pct,{100 * (comp_df['similarity'] >= 0.9).mean():.2f}")
        print(f"mini_faster_count,{comp_df['mini_faster'].sum()}")
        print(f"mini_faster_pct,{100 * comp_df['mini_faster'].mean():.2f}")
        print(f"avg_latency_diff_s,{comp_df['latency_diff'].mean():.3f}")
        print(f"median_latency_diff_s,{comp_df['latency_diff'].median():.3f}")
        print(f"avg_gpt4o_duration_s,{comp_df['gpt4o_duration'].mean():.3f}")
        print(f"avg_mini_duration_s,{comp_df['mini_duration'].mean():.3f}")
        print(f"speedup_pct,{100 * comp_df['latency_diff'].mean() / comp_df['gpt4o_duration'].mean():.2f}")

    # === SECTION 9: PER-REQUEST DETAILS (CSV FORMAT) ===
    print("\n[PER_REQUEST_DETAILS]")
    print(
        "request_id,fastest_model,fastest_duration_s,winner_model,winner_duration_s,"
        "time_wasted_s,strategy_used,similarity,char_diff"
    )
    for _, row in race_df.iterrows():
        print(
            f"{row['request_id']},{row['fastest_model']},{row['fastest_duration']:.3f},"
            f"{row['winner_model']},{row['winner_duration']:.3f},"
            f"{row['time_wasted']:.3f},{int(row['strategy_used'])},"
            f"{row['similarity']:.4f},{row['char_diff']}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
