"""
Sidebar — Upload, settings, vault info (no API key input — token from .env)
"""

import os
import streamlit as st

from config.settings import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    SUPPORTED_EXTENSIONS,
)
from src.document_processor import process_vault, get_vault_stats
from src.chunker import chunk_documents, get_chunk_stats
from src.vector_store import (
    get_vector_store,
    create_vector_store,
    get_collection_stats,
    clear_all_data,
    close_vector_store,
)
from src.embeddings import get_embedding_function
from src.utils import extract_zip_to_files, filter_markdown_files


def render_sidebar():
    """Render the complete sidebar UI."""
    with st.sidebar:
        st.image("https://img.icons8.com/3d-fluency/94/brain.png", width=64)
        st.title("🧠 Obsidian RAG")
        st.caption("AI-powered knowledge assistant")

        st.divider()

        st.info("🔑 API key is configured via `.env` file — no manual entry needed.")

        st.divider()

        # --- File Upload ---
        uploaded_files = render_upload_section()

        st.divider()

        # --- Vault Stats ---
        render_vault_stats_section()

        st.divider()

        # --- Actions ---
        render_actions_section()

        return uploaded_files


def render_upload_section():
    """Render file upload section."""
    st.subheader("📁 Upload Vault")

    upload_type = st.radio(
        "Upload method",
        ["Multiple .md files", "ZIP archive"],
        horizontal=True,
        key="upload_type",
    )

    uploaded_files = []

    if upload_type == "Multiple .md files":
        uploaded_files = st.file_uploader(
            "Upload markdown files",
            type=["md", "markdown"],
            accept_multiple_files=True,
            help=f"Max {SUPPORTED_EXTENSIONS} files",
        )
    else:
        zip_file = st.file_uploader(
            "Upload vault as ZIP",
            type=["zip"],
            help="Upload your entire Obsidian vault as a .zip file",
        )
        if zip_file:
            uploaded_files = extract_zip_to_files(zip_file)

    if uploaded_files:
        st.success(f"📦 {len(uploaded_files)} file(s) ready")

        if st.button("🚀 Process & Index Vault", type="primary", use_container_width=True):
            process_and_index(uploaded_files)

    return uploaded_files


def process_and_index(uploaded_files):
    """Process uploaded files and create vector store."""
    # Check Google API key is configured
    google_key = os.environ.get("GOOGLE_API_KEY")
    if not google_key:
        st.error("❌ GOOGLE_API_KEY not found. Please add it to your .env file.")
        st.info("Get a free key at: https://aistudio.google.com/apikey")
        return

    chunk_size = DEFAULT_CHUNK_SIZE
    chunk_overlap = DEFAULT_CHUNK_OVERLAP

    progress = st.progress(0, text="Starting processing...")

    try:
        # Step 1: Process documents
        progress.progress(10, text="📄 Parsing markdown files...")
        documents = process_vault(uploaded_files)

        if not documents:
            st.error("❌ No valid markdown content found in uploaded files.")
            progress.empty()
            return

        # Step 2: Chunk documents
        progress.progress(40, text=f"✂️ Splitting into chunks (size={chunk_size})...")
        chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Step 3: Create embeddings & vector store (embeddings run locally)
        progress.progress(60, text="🧮 Generating embeddings (first run downloads model)...")
        embedding_fn = get_embedding_function()

        progress.progress(75, text="💾 Building vector database...")
        vector_store = create_vector_store(chunks, embedding_fn)
        # Data is persisted on disk; release the DB handles so repeated
        # indexing doesn't leak open ChromaDB clients in the server process.
        close_vector_store(vector_store)

        # Step 4: Store stats in session
        vault_stats = get_vault_stats(documents)
        chunk_stats = get_chunk_stats(chunks)

        st.session_state["vault_stats"] = vault_stats
        st.session_state["chunk_stats"] = chunk_stats
        st.session_state["documents"] = documents
        st.session_state["chunks"] = chunks
        st.session_state["indexed"] = True

        progress.progress(100, text="✅ Done!")

        st.success(
            f"✅ Indexed **{vault_stats['total_files']}** notes → "
            f"**{chunk_stats['total_chunks']}** chunks"
        )

    except Exception as e:
        st.error(f"❌ Error processing vault: {str(e)}")
    finally:
        progress.empty()


def render_vault_stats_section():
    """Render vault statistics in the sidebar."""
    st.subheader("📊 Vault Info")

    if st.session_state.get("indexed"):
        vault_stats = st.session_state.get("vault_stats", {})
        chunk_stats = st.session_state.get("chunk_stats", {})

        col1, col2 = st.columns(2)
        with col1:
            st.metric("📝 Notes", vault_stats.get("total_files", 0))
        with col2:
            st.metric("🧩 Chunks", chunk_stats.get("total_chunks", 0))

        col3, col4 = st.columns(2)
        with col3:
            st.metric("🏷️ Tags", vault_stats.get("unique_tags", 0))
        with col4:
            st.metric("📖 Words", f"{vault_stats.get('total_words', 0):,}")

        tags = vault_stats.get("tags", [])
        if tags:
            st.caption("Tags: " + ", ".join(tags[:10]))
            if len(tags) > 10:
                st.caption(f"  ... and {len(tags) - 10} more")

        collection_stats = get_collection_stats()
        st.caption(f"🗄️ DB: {collection_stats.get('total_chunks', 0)} vectors stored")
    else:
        st.info("📭 No vault indexed yet. Upload files above to get started!")


def render_actions_section():
    """Render action buttons."""
    st.subheader("🔧 Actions")

    if st.button("🗑️ Clear Vault & Reset", use_container_width=True):
        clear_all_data()
        for key in ["indexed", "vault_stats", "chunk_stats", "documents", "chunks"]:
            st.session_state.pop(key, None)
        st.session_state.pop("chat_history", None)
        st.success("✅ Vault cleared!")
        st.rerun()

    if st.button("🔄 Clear Chat History", use_container_width=True):
        st.session_state.pop("chat_history", None)
        st.success("✅ Chat cleared!")
        st.rerun()

    st.markdown("---")
    st.caption("📥 **Demo:** Try the sample vault included in the repo!")
