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
# Use gemini-2.0-flash which is available in the new google-genai SDK
LLM_MODEL = "gemini-3.5-flash"

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
