"""
Embeddings — Google Gemini embedding model wrapper
"""

import time
from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from config.settings import EMBEDDING_MODEL


def get_embedding_function(
    api_key: str,
    model_name: str = EMBEDDING_MODEL,
) -> GoogleGenerativeAIEmbeddings:
    """
    Create and return a Gemini embedding function.
    Wraps GoogleGenerativeAIEmbeddings from LangChain.
    """
    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=api_key,
    )


def embed_documents_with_retry(
    embedding_fn: GoogleGenerativeAIEmbeddings,
    documents: List[Document],
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> List[Document]:
    """
    Embed documents with retry logic for API rate limits.
    Returns the same documents (embeddings are handled by ChromaDB internally).
    """
    for attempt in range(max_retries):
        try:
            # ChromaDB handles embedding via the embedding function,
            # so we just verify the function works by embedding a test string.
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
    # embedding-004 uses 768 dimensions
    return 768
