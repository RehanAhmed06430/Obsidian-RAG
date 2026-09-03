"""
Chat Page — Main conversational interface
"""

import streamlit as st
from typing import List, Dict, Any

from config.settings import DEFAULT_TOP_K
from src.chat_manager import ChatManager
from src.vector_store import get_vector_store, similarity_search_with_scores
from src.embeddings import get_embedding_function
from src.rag_chain import query_rag_with_sources


def render_chat_page():
    """Render the main chat interface."""
    # Header
    st.markdown("## 💬 Chat with Your Vault")
    st.caption("Ask questions about your notes — answers are grounded in your personal knowledge base.")

    if not st.session_state.get("indexed"):
        render_empty_state()
        return

    # Display existing chat history
    render_chat_history()

    # Chat input
    if question := st.chat_input("Ask a question about your notes..."):
        handle_question(question)


def render_empty_state():
    """Show empty state when no vault is indexed."""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; padding: 4rem 2rem;'>
            <h2 style='color: #666;'>🧠 Welcome to Obsidian RAG</h2>
            <p style='color: #888; font-size: 1.1rem;'>
                Upload your Obsidian vault in the sidebar to start chatting with your notes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Starter questions
    st.markdown("### 💡 Things you'll be able to ask:")
    starter_qs = [
        "What are the main topics in my notes?",
        "Summarize my notes about machine learning.",
        "What did I write about neural networks?",
        "Find connections between my notes on Python and ML.",
    ]
    for q in starter_qs:
        st.markdown(f"- *\"{q}\"*")


def render_chat_history():
    """Render all previous messages in the conversation."""
    history = ChatManager.get_history()

    for msg in history:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        with st.chat_message(role):
            st.markdown(content)
            if sources:
                with st.expander("📚 Sources", expanded=False):
                    st.markdown(ChatManager.format_sources_for_display(sources))


def handle_question(question: str):
    """Process a user question and display the response."""
    top_k = DEFAULT_TOP_K

    # Add user message
    ChatManager.add_message("user", question)

    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching your notes and generating answer..."):
            try:
                embedding_fn = get_embedding_function()
                vector_store = get_vector_store(embedding_fn)

                # Check if vector store has data
                collection_stats = vector_store._collection.count() if hasattr(vector_store, '_collection') else 0
                if collection_stats == 0:
                    response = "It looks like the vault hasn't been indexed yet. Please process your files using the sidebar."
                    sources = []
                else:
                    chat_history = ChatManager.get_history_for_llm()
                    response, sources = query_rag_with_sources(
                        vector_store=vector_store,
                        question=question,
                        chat_history=chat_history[:-1],  # exclude current question
                        top_k=top_k,
                    )

                st.markdown(response)

                # Display sources
                if sources:
                    with st.expander("📚 Sources", expanded=False):
                        st.markdown(ChatManager.format_sources_for_display(sources))

                # Save to history
                source_dicts = []
                for doc in sources:
                    if hasattr(doc, "metadata"):
                        source_dicts.append(doc.metadata)
                ChatManager.add_message("assistant", response, source_dicts)

            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                ChatManager.add_message("assistant", error_msg)
