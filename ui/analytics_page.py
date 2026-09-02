"""
Analytics Page — Vault statistics and visualizations
"""

import streamlit as st
from collections import Counter
from typing import List, Dict, Any

from src.vector_store import get_collection_stats


def render_analytics_page():
    """Render the vault analytics dashboard."""
    st.markdown("## 📊 Vault Analytics")
    st.caption("Insights and statistics about your knowledge base.")

    if not st.session_state.get("indexed"):
        st.info("📭 No vault indexed yet. Upload files in the sidebar to see analytics.")
        return

    vault_stats = st.session_state.get("vault_stats", {})
    chunk_stats = st.session_state.get("chunk_stats", {})
    documents = st.session_state.get("documents", [])

    # --- Overview Metrics ---
    st.markdown("### 📈 Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📝 Notes",
            vault_stats.get("total_files", 0),
            help="Total number of unique notes",
        )
    with col2:
        st.metric(
            "🧩 Chunks",
            chunk_stats.get("total_chunks", 0),
            help="Total text chunks in vector store",
        )
    with col3:
        st.metric(
            "🏷️ Tags",
            vault_stats.get("unique_tags", 0),
            help="Unique tags across all notes",
        )
    with col4:
        st.metric(
            "📖 Words",
            f"{vault_stats.get('total_words', 0):,}",
            help="Total words in the vault",
        )

    # --- Tag Analysis ---
    st.markdown("### 🏷️ Tag Analysis")
    all_tags = []
    for doc in documents:
        tags = doc.metadata.get("tags", "")
        if tags:
            all_tags.extend([t.strip() for t in tags.split(",") if t.strip()])

    if all_tags:
        tag_counts = Counter(all_tags)
        sorted_tags = tag_counts.most_common(20)

        # Tag frequency chart
        import pandas as pd
        tag_df = pd.DataFrame(sorted_tags, columns=["Tag", "Count"])

        col_chart, col_table = st.columns([2, 1])

        with col_chart:
            st.bar_chart(tag_df.set_index("Tag"))

        with col_table:
            st.markdown("**Top Tags:**")
            for tag, count in sorted_tags:
                st.markdown(f"- `{tag}` — {count} notes")
    else:
        st.info("No tags found in the vault.")

    # --- Source Distribution ---
    st.markdown("### 📁 Notes by Source")
    source_counts = Counter()
    source_words = {}
    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        source_counts[source] += 1
        source_words[source] = doc.metadata.get("word_count", 0)

    if source_counts:
        import pandas as pd
        source_data = pd.DataFrame([
            {"Note": s, "Chunks": c, "Words": source_words.get(s, 0)}
            for s, c in source_counts.most_common()
        ])

        col_dist, col_detail = st.columns([2, 1])

        with col_dist:
            st.bar_chart(source_data.set_index("Note")["Chunks"])

        with col_detail:
            st.dataframe(
                source_data,
                use_container_width=True,
                hide_index=True,
            )

    # --- Chunk Size Distribution ---
    st.markdown("### 📐 Chunk Statistics")
    col_avg, col_min, col_max = st.columns(3)

    with col_avg:
        avg = chunk_stats.get("avg_length", 0)
        st.metric("Avg Chunk Size", f"{avg:.0f} chars")
    with col_min:
        min_len = chunk_stats.get("min_length", 0)
        st.metric("Smallest Chunk", f"{min_len} chars")
    with col_max:
        max_len = chunk_stats.get("max_length", 0)
        st.metric("Largest Chunk", f"{max_len} chars")

    # --- Connection Graph (text-based) ---
    st.markdown("### 🔗 Note Connections")
    connections = {}
    for doc in documents:
        source = doc.metadata.get("source", "")
        backlinks = doc.metadata.get("backlinks", "")
        if backlinks and source:
            links = [b.strip() for b in backlinks.split(",") if b.strip()]
            if links:
                connections[source] = links

    if connections:
        for note, links in connections.items():
            links_str = ", ".join(f"`{l}`" for l in links[:5])
            st.markdown(f"**{note}** → {links_str}")
    else:
        st.info("No backlink connections detected in the vault.")
        st.caption("Tip: Use `[[Wiki Links]]` in your Obsidian notes to create connections!")

    # --- Database Info ---
    st.markdown("### 🗄️ Vector Database")
    collection_stats = get_collection_stats()
    st.json(collection_stats)
