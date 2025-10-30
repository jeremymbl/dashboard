#!/usr/bin/env python3
"""
Diagnostic script to investigate missing agent run data in Project Inspector.
"""

import streamlit as st
from logfire.query_client import LogfireQueryClient

# Get secrets
_LF_READ_TOKEN = st.secrets["LOGFIRE_TOKEN"]
_PROJECT_GUID = st.secrets["_PROJECT_GUID"]

user_project_id = "0e8e9b44-c318-45a3-acbd-aa9eb0d05656"
lookback_days = 30

print(f"\n🔍 Investigating project: {user_project_id}\n")

# Step 1: Find initial prompts
query1 = f"""
SELECT
    r.trace_id,
    r.span_name,
    r.start_timestamp AT TIME ZONE 'Europe/Paris' AS "timestamp"
FROM records r
WHERE r.project_id = '{_PROJECT_GUID}'
    AND r.span_name LIKE 'POST /projects/%'
    AND r.start_timestamp >= NOW() - INTERVAL '{lookback_days} days'
    AND (
        r.attributes -> 'fastapi.arguments.values' ->> 'project_id' = '{user_project_id}'
        OR r.attributes -> 'fastapi.arguments.values' LIKE '%"project_id":"{user_project_id}"%'
    )
ORDER BY r.start_timestamp DESC
LIMIT 5
"""

print("📋 Step 1: Finding initial prompt traces...")
with LogfireQueryClient(read_token=_LF_READ_TOKEN) as client:
    result1 = client.query_json_rows(sql=query1)
    prompts = result1.get("rows", result1)

if not prompts:
    print("❌ No prompts found for this project!")
    exit(1)

print(f"✅ Found {len(prompts)} recent prompts\n")

for idx, prompt in enumerate(prompts, 1):
    trace_id = prompt['trace_id']
    print(f"--- Prompt #{idx} ---")
    print(f"Trace ID: {trace_id}")
    print(f"Span Name: {prompt['span_name']}")
    print(f"Timestamp: {prompt['timestamp']}")

    # Step 2: Find ALL spans for this trace
    query2 = f"""
    SELECT
        span_name,
        COUNT(*) as count
    FROM records
    WHERE trace_id = '{trace_id}'
    GROUP BY span_name
    ORDER BY span_name
    """

    print(f"\n🔎 Analyzing all span names for trace {trace_id}:")
    with LogfireQueryClient(read_token=_LF_READ_TOKEN) as client:
        result2 = client.query_json_rows(sql=query2)
        spans = result2.get("rows", result2)

    for span in spans:
        print(f"  - {span['span_name']}: {span['count']} occurrences")

    # Step 3: Check for agent run span specifically
    query3 = f"""
    SELECT
        span_name,
        attributes
    FROM records
    WHERE trace_id = '{trace_id}'
        AND span_name = 'agent run'
    LIMIT 1
    """

    print(f"\n🤖 Checking for 'agent run' span:")
    with LogfireQueryClient(read_token=_LF_READ_TOKEN) as client:
        result3 = client.query_json_rows(sql=query3)
        agent_spans = result3.get("rows", result3)

    if agent_spans:
        agent_span = agent_spans[0]
        attrs = agent_span.get('attributes', {})
        has_messages = 'pydantic_ai.all_messages' in attrs
        has_final_result = 'final_result' in attrs
        print(f"  ✅ Found 'agent run' span!")
        print(f"  - Has pydantic_ai.all_messages: {has_messages}")
        print(f"  - Has final_result: {has_final_result}")

        if has_messages:
            messages = attrs.get('pydantic_ai.all_messages', [])
            print(f"  - Number of messages: {len(messages)}")
    else:
        print(f"  ❌ No 'agent run' span found!")

    # Step 4: Check what our V3 query would return
    query4 = f"""
    SELECT
        span_name,
        COUNT(*) as count
    FROM records
    WHERE trace_id = '{trace_id}'
        AND (span_name LIKE 'POST /projects/%' OR span_name = 'agent run')
    GROUP BY span_name
    """

    print(f"\n📊 What V3 query returns (with filter):")
    with LogfireQueryClient(read_token=_LF_READ_TOKEN) as client:
        result4 = client.query_json_rows(sql=query4)
        filtered_spans = result4.get("rows", result4)

    for span in filtered_spans:
        print(f"  - {span['span_name']}: {span['count']} occurrences")

    print("\n" + "="*80 + "\n")

print("\n✅ Investigation complete!")
