"""
Embeddings — Hugging Face sentence-transformers (runs locally, no API key)
"""

import time
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config.settings import EMBEDDING_MODEL


def get_embedding_function(
    model_name: str = EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """
    Create and return a HuggingFace embedding function.

    Uses sentence-transformers locally — no API key required.
    Default model: all-MiniLM-L6-v2 (384 dimensions, fast, high quality).
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_documents_with_retry(
    embedding_fn: HuggingFaceEmbeddings,
    documents: List[Document],
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> List[Document]:
    """
    Embed documents with retry logic.
    Returns the same documents (embeddings are handled by ChromaDB internally).
    """
    for attempt in range(max_retries):
        try:
            # Verify the embedding function works
            embedding_fn.embed_query("test")
            return documents
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                time.sleep(wait_time)
            else:
                raise RuntimeError(
                    f"Failed to initialize embedding after {max_retries} attempts: {e}"
                ) from e

    return documents


def get_embedding_dimensions() -> int:
    """Return the embedding dimension for the configured model."""
    return 384  # all-MiniLM-L6-v2
