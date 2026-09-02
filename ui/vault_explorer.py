"""
Vault Explorer — Browse, search, and filter indexed notes
"""

import streamlit as st
from typing import List, Dict, Any

from src.document_processor import parse_markdown


def render_vault_explorer():
    """Render the vault explorer page."""
    st.markdown("## 📚 Vault Explorer")
    st.caption("Browse, search, and explore your indexed notes.")

    if not st.session_state.get("indexed"):
        st.info("📭 No vault indexed yet. Upload files in the sidebar to explore your notes.")
        return

    documents = st.session_state.get("documents", [])
    if not documents:
        st.info("📭 No documents found.")
        return

    # Collect all unique sources
    sources_map = {}
    all_tags = set()

    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        if source not in sources_map:
            sources_map[source] = {
                "source": source,
                "title": doc.metadata.get("title", source),
                "tags": doc.metadata.get("tags", ""),
                "word_count": doc.metadata.get("word_count", 0),
                "chunk_count": 0,
                "content_preview": doc.page_content[:300] if doc.page_content else "",
            }
        sources_map[source]["chunk_count"] += 1

        tags = doc.metadata.get("tags", "")
        if tags:
            for t in tags.split(","):
                t = t.strip()
                if t:
                    all_tags.add(t)

    notes = list(sources_map.values())
    notes.sort(key=lambda x: x["source"])

    # Search & Filter
    col_search, col_filter = st.columns([3, 1])

    with col_search:
        search_query = st.text_input(
            "🔍 Search notes",
            placeholder="Search by filename, title, or content...",
            key="explorer_search",
        )

    with col_filter:
        tag_list = sorted(list(all_tags))
        selected_tag = st.selectbox(
            "🏷️ Filter by tag",
            options=["All"] + tag_list,
            key="explorer_tag_filter",
        )

    # Apply filters
    filtered_notes = notes

    if search_query:
        sq = search_query.lower()
        filtered_notes = [
            n for n in filtered_notes
            if sq in n["source"].lower()
            or sq in n["title"].lower()
            or sq in n["content_preview"].lower()
        ]

    if selected_tag and selected_tag != "All":
        filtered_notes = [
            n for n in filtered_notes
            if selected_tag in n.get("tags", "")
        ]

    st.caption(f"Showing {len(filtered_notes)} of {len(notes)} notes")

    # Note cards
    for note in filtered_notes:
        with st.expander(
            f"📝 {note['title']} (`{note['source']}`) — {note['word_count']} words, {note['chunk_count']} chunks",
            expanded=False,
        ):
            # Tags
            tags = note.get("tags", "")
            if tags:
                tag_badges = " ".join(
                    [f"`{t.strip()}`" for t in tags.split(",") if t.strip()]
                )
                st.markdown(f"**Tags:** {tag_badges}")

            # Content preview
            st.markdown("**Preview:**")
            st.text(note["content_preview"] + ("..." if len(note["content_preview"]) == 300 else ""))

            # Word count
            st.caption(f"📊 {note['word_count']} words | {note['chunk_count']} chunks")


def get_note_connections(documents):
    """Find backlink connections between notes."""
    connections = {}
    for doc in documents:
        source = doc.metadata.get("source", "")
        backlinks = doc.metadata.get("backlinks", "")
        if backlinks and source:
            links = [b.strip() for b in backlinks.split(",") if b.strip()]
            if links:
                connections[source] = links
    return connections
