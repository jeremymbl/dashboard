"""
Project Inspector Page

Allows debugging user interactions by inspecting:
- User prompts
- Run messages sequence
- Agent responses
- Logfire spans

Built on existing data_sources.py infrastructure.
"""

import streamlit as st
import json
from src.data_sources import (
    fetch_project_interactions,
    construct_logfire_url,
)

# Page config
st.set_page_config(
    page_title="Project Inspector",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Project Inspector")
st.markdown("Inspect user interactions by project ID to debug issues")
st.info("💡 **Note:** Only shows interactions where the agent responded (excludes audio transcriptions without agent processing)")

# Sidebar inputs
with st.sidebar:
    st.header("Filters")

    user_project_id = st.text_input(
        "Project ID",
        placeholder="cd79350d-fde4-4849-8e32-b05c2ea506de",
        help="Enter the user's project ID (not logfire project_id)"
    )

    lookback_days = st.slider(
        "Lookback Period (days)",
        min_value=1,
        max_value=90,
        value=7,
        help="Number of days to look back"
    )

    limit = st.slider(
        "Max Interactions",
        min_value=10,
        max_value=200,
        value=50,
        help="Maximum number of interactions to fetch"
    )

    fetch_button = st.button("🔍 Fetch Interactions", type="primary")

# Main content
if not user_project_id:
    st.info("👈 Enter a project ID in the sidebar to begin")
    st.markdown("""
    ### How to use this tool:

    1. **Find the project ID** from the URL or database
    2. **Enter it in the sidebar** and click "Fetch Interactions"
    3. **Explore the timeline** of all user interactions
    4. **Click on each interaction** to see:
       - User prompt
       - Run messages (internal processing)
       - Agent response
       - Link to logfire for detailed debugging

    ### Example project IDs:
    - `cd79350d-fde4-4849-8e32-b05c2ea506de`
    - `5a62baf1-8fed-43df-8625-248f1b2430e4`
    """)
    st.stop()

if fetch_button or 'interactions' in st.session_state:
    # Fetch data
    if fetch_button:
        with st.spinner("Fetching interactions from logfire..."):
            interactions = fetch_project_interactions(
                user_project_id,
                lookback_days=lookback_days,
                limit=limit
            )
            st.session_state['interactions'] = interactions
            st.session_state['project_id'] = user_project_id
    else:
        interactions = st.session_state.get('interactions', [])

    # Display summary
    if not interactions:
        st.warning(f"No agent interactions found for project {user_project_id}")
        st.info("""
        **Possible reasons:**
        - No agent responses in the selected time period (only audio transcriptions)
        - Incorrect project ID
        - Project has no chat prompts yet

        **Note:** This tool only shows interactions where the agent processed and responded.
        Audio transcriptions without agent processing are excluded.

        Try increasing the lookback period or verify the project ID.
        """)
        st.stop()

    st.success(f"Found {len(interactions)} agent interactions for project `{user_project_id}` (audio transcriptions excluded)")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Interactions", len(interactions))

    with col2:
        success_count = sum(1 for i in interactions if i['status'] == 'success')
        st.metric("Successful", success_count, delta=None if success_count == len(interactions) else "")

    with col3:
        failure_count = len(interactions) - success_count
        st.metric("Failed", failure_count, delta=None if failure_count == 0 else "")

    with col4:
        avg_duration = sum(i.get('duration', 0) for i in interactions) / len(interactions)
        st.metric("Avg Duration", f"{avg_duration:.2f}s")

    st.divider()

    # Display interactions
    st.subheader("Interaction Timeline (Most Recent First)")

    for idx, interaction in enumerate(interactions):
        email_display = interaction['email'] or "Unknown User"

        with st.expander(
            f"#{idx+1} • {interaction['timestamp']} • {email_display}",
            expanded=(idx == 0)  # Expand first one by default
        ):
            # Metadata section
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("**📋 Metadata**")
                st.text(f"Trace ID: {interaction['trace_id']}")
                st.text(f"Email: {interaction['email'] or 'N/A'}")
                st.text(f"Scope: {interaction['scope'] or 'N/A'}")
                st.text(f"Status: {interaction['status']}")
                st.text(f"Duration: {interaction.get('duration', 0):.3f}s")

            with col2:
                st.markdown("**🔗 Logfire**")
                st.markdown(f"[🔍 View in Logfire]({interaction['trace_url']})")
                st.caption("Opens full trace with all spans")

            # User prompt section
            st.markdown("---")
            st.markdown("### 👤 User Prompt")
            if interaction['prompt_text']:
                st.text_area(
                    "User's input",
                    value=interaction['prompt_text'],
                    height=100,
                    key=f"prompt_{idx}",
                    disabled=True,
                    label_visibility="collapsed"
                )
            else:
                st.info("No prompt text available")

            # Run messages section
            st.markdown("---")
            st.markdown("### ⚙️ Run Messages Sequence")

            run_messages = interaction.get('run_messages', [])
            if run_messages:
                st.caption(f"Total messages in conversation: {len(run_messages)}")

                # Show messages in tabs for better organization
                msg_tabs = st.tabs([f"Message {i+1}" for i in range(min(len(run_messages), 10))])

                for tab_idx, (tab, msg) in enumerate(zip(msg_tabs, run_messages[:10])):
                    with tab:
                        role = msg.get('role', 'unknown')
                        role_emoji = "👤" if role == "user" else "🤖"

                        st.markdown(f"**{role_emoji} Role:** `{role}`")

                        parts = msg.get('parts', [])

                        for part_idx, part in enumerate(parts):
                            part_type = part.get('type', 'unknown')

                            if part_type == 'text':
                                st.markdown("**💬 Text Content:**")
                                content = part.get('content', 'N/A')
                                st.code(content, language=None)

                            elif part_type == 'tool_call':
                                st.markdown("**🔧 Tool Call:**")
                                tool_data = {
                                    'tool_name': part.get('name'),
                                    'tool_id': part.get('id'),
                                    'arguments': part.get('arguments')
                                }
                                st.json(tool_data)

                            elif part_type == 'tool_call_response':
                                st.markdown("**📥 Tool Response:**")
                                response_data = {
                                    'tool_name': part.get('name'),
                                    'tool_id': part.get('id'),
                                    'result': part.get('result')
                                }
                                st.json(response_data)

                        # Show finish reason if present
                        finish_reason = msg.get('finish_reason')
                        if finish_reason:
                            st.caption(f"✓ Finish reason: `{finish_reason}`")

                if len(run_messages) > 10:
                    st.info(f"Showing first 10 of {len(run_messages)} messages. View full trace in logfire for complete history.")
            else:
                st.warning("⚠️ No run messages found (this should not happen - please report this issue)")

            # Agent response section
            st.markdown("---")
            st.markdown("### 🤖 Agent Response")

            agent_response = interaction.get('agent_response', {})
            if agent_response and agent_response.get('final_result'):
                final_result = agent_response.get('final_result')
                st.text_area(
                    "Final agent response",
                    value=final_result,
                    height=150,
                    key=f"response_{idx}",
                    disabled=True,
                    label_visibility="collapsed"
                )

                # Model info
                col1, col2, col3 = st.columns(3)
                with col1:
                    model = agent_response.get('model_name', 'N/A')
                    st.metric("Model", model)
                with col2:
                    input_tokens = agent_response.get('input_tokens', 0)
                    st.metric("Input Tokens", f"{input_tokens:,}")
                with col3:
                    output_tokens = agent_response.get('output_tokens', 0)
                    st.metric("Output Tokens", f"{output_tokens:,}")
            else:
                st.warning("⚠️ No agent response found (this should not happen - please report this issue)")

    # Export section
    st.divider()
    st.markdown("### 📥 Export Data")
    st.caption("Download all interactions as JSON for further analysis")

    export_json = st.download_button(
        label="💾 Download as JSON",
        data=json.dumps(interactions, indent=2, default=str),
        file_name=f"project_{user_project_id}_interactions_{len(interactions)}.json",
        mime="application/json"
    )
