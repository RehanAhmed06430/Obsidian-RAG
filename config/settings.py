"""
Application Settings & Constants
Obsidian Vault RAG Knowledge Assistant
"""

import os

# --- Embedding Model (runs locally, no API key needed) ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# --- LLM Model (Groq free tier) ---
# Free models on Groq: llama-3.3-70b-versatile, llama3-70b-8192, mixtral-8x7b-32768
LLM_MODEL = "llama-3.3-70b-versatile"

# --- Chunking ---
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

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
MAX_HISTORY_LENGTH = 10

# --- File Upload ---
SUPPORTED_EXTENSIONS = {".md", ".markdown"}
MAX_FILE_SIZE_MB = 10
MAX_FILES = 500
