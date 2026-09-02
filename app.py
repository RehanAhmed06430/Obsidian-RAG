"""
🧠 Obsidian Vault RAG Knowledge Assistant
Main Streamlit application entry point.
"""

import streamlit as st
import os

from config.settings import APP_TITLE, APP_ICON, APP_LAYOUT, PAGE_CHAT, PAGE_EXPLORER, PAGE_ANALYTICS
from ui.sidebar import render_sidebar
from ui.chat_page import render_chat_page
from ui.vault_explorer import render_vault_explorer
from ui.analytics_page import render_analytics_page


def _get_api_key() -> str:
    """Get Gemini API key from environment, Streamlit secrets, or session state."""
    # 1. Check environment variable
    key = os.environ.get("GOOGLE_API_KEY", "")
    if key:
        return key
    # 2. Check Streamlit secrets (for Streamlit Cloud deployment)
    try:
        key = st.secrets.get("GOOGLE_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    # 3. Return empty (user will enter in sidebar)
    return ""


def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "indexed": False,
        "vault_stats": {},
        "chunk_stats": {},
        "documents": [],
        "chunks": [],
        "chat_history": [],
        "google_api_key": _get_api_key(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_custom_css():
    """Inject custom CSS for styling."""
    css_path = "assets/style.css"
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Additional inline styles
    st.markdown("""
    <style>
    /* Main container */
    .stApp {
        max-width: 100%;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
        padding: 10px;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
    }

    /* Make tabs look better */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }

    /* Success/info messages */
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    """Main application function."""
    st.set_page_config(
        page_title="Obsidian Vault RAG",
        page_icon=APP_ICON,
        layout=APP_LAYOUT,
        initial_sidebar_state="expanded",
    )

    init_session_state()
    load_custom_css()

    # Render sidebar and get user inputs
    api_key, uploaded_files, chunk_size, chunk_overlap, top_k = render_sidebar()

    # Tab navigation
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs([PAGE_CHAT, PAGE_EXPLORER, PAGE_ANALYTICS])

    with tab1:
        render_chat_page()

    with tab2:
        render_vault_explorer()

    with tab3:
        render_analytics_page()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 10px;'>"
        "🧠 Obsidian Vault RAG Knowledge Assistant — Powered by Gemini + ChromaDB + LangChain"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
