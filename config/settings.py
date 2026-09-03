"""
Application Settings & Constants
Obsidian Vault RAG Knowledge Assistant
"""

import os
from pathlib import Path

# Project root (directory containing this project's settings/config)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Embedding Model (Google Gemini API) ---
EMBEDDING_MODEL = "gemini-embedding-001" 
EMBEDDING_DIMENSIONS = 768

# --- LLM Model (Google Gemini) ---
LLM_MODEL = "gemini-3.5-flash"  # this is the best llm model from google now (in 2026)

# --- Chunking ---
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# --- Retrieval ---
DEFAULT_TOP_K = 5
MAX_TOP_K = 10

# --- ChromaDB ---
CHROMA_COLLECTION_NAME = "obsidian_vault"
# Absolute path (anchored to project root) so the DB resolves correctly
# no matter which working directory the app is launched from.
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")

# --- UI ---
APP_TITLE = "🧠 Obsidian Vault RAG Assistant"
APP_ICON = "🧠"
APP_LAYOUT = "wide"
PAGE_CHAT = "💬 Chat"
PAGE_EXPLORER = "📚 Vault Explorer"
PAGE_ANALYTICS = "📊 Analytics"

# --- Conversation ---
MAX_HISTORY_LENGTH = 10

# --- File Upload ---
SUPPORTED_EXTENSIONS = {".md", ".markdown"}
MAX_FILE_SIZE_MB = 10
MAX_FILES = 500
