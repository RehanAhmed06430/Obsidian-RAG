"""
Application Settings & Constants
Obsidian Vault RAG Knowledge Assistant
"""

import os

# --- Embedding Model ---
# gemini-embedding-2-preview is the recommended model for the new google-genai SDK
EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIMENSIONS = 768

# --- LLM ---
# gemini-2.0-flash has higher free-tier limits (1500 req/day) vs 3.5-flash (20 req/day)
LLM_MODEL = "gemini-2.0-flash"

# --- Chunking ---
DEFAULT_CHUNK_SIZE = 500       # tokens
DEFAULT_CHUNK_OVERLAP = 50     # tokens

# --- Retrieval ---
DEFAULT_TOP_K = 5
MAX_TOP_K = 10

# --- ChromaDB ---
CHROMA_COLLECTION_NAME = "obsidian_vault"
CHROMA_PERSIST_DIR = "chroma_db"

# --- UI ---
APP_TITLE = "🧠 Obsidian Vault RAG Assistant"
APP_ICON = "🧠"
APP_LAYOUT = "wide"
PAGE_CHAT = "💬 Chat"
PAGE_EXPLORER = "📚 Vault Explorer"
PAGE_ANALYTICS = "📊 Analytics"

# --- Conversation ---
MAX_HISTORY_LENGTH = 10  # max exchanges to keep in context

# --- File Upload ---
SUPPORTED_EXTENSIONS = {".md", ".markdown"}
MAX_FILE_SIZE_MB = 10
MAX_FILES = 500
