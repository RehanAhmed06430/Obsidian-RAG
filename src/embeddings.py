"""
Embeddings — Google Gemini API (embedding-001)
"""

import os
import time
from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from config.settings import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS


def get_embedding_function(
    model_name: str = EMBEDDING_MODEL,
) -> GoogleGenerativeAIEmbeddings:
    """
    Create and return a Google Generative AI embedding function.

    Uses Google's embedding-001 model via the Gemini API.
    Requires GOOGLE_API_KEY environment variable.
    Default model: models/embedding-001 (768 dimensions).
    """
    google_key = os.environ.get("GOOGLE_API_KEY")
    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=google_key,
    )


def embed_documents_with_retry(
    embedding_fn: GoogleGenerativeAIEmbeddings,
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
    return EMBEDDING_DIMENSIONS  # embedding-001 = 768
