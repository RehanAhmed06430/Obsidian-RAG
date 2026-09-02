"""
Vector Store — ChromaDB operations for vector storage & retrieval
"""

import shutil
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    DEFAULT_TOP_K,
)


def _get_chroma_settings() -> ChromaSettings:
    """Return consistent ChromaDB settings used across all operations."""
    return ChromaSettings(
        anonymized_telemetry=False,
        is_persistent=True,
    )


def get_vector_store(
    embedding_fn: GoogleGenerativeAIEmbeddings,
    collection_name: str = CHROMA_COLLECTION_NAME,
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """
    Get or create a ChromaDB vector store.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=persist_directory,
        client_settings=_get_chroma_settings(),
    )


def create_vector_store(
    chunks: List[Document],
    embedding_fn: GoogleGenerativeAIEmbeddings,
    collection_name: str = CHROMA_COLLECTION_NAME,
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """
    Create a new ChromaDB vector store from document chunks.
    If the collection already exists, it is recreated.
    """
    # Clear existing data to start fresh
    clear_collection(persist_directory, collection_name)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        collection_name=collection_name,
        persist_directory=persist_directory,
        client_settings=_get_chroma_settings(),
    )

    return vector_store


def similarity_search(
    vector_store: Chroma,
    query: str,
    k: int = DEFAULT_TOP_K,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Perform similarity search on the vector store.
    Returns the top-k most relevant document chunks.
    """
    search_kwargs = {"k": k}
    if filter_metadata:
        search_kwargs["filter"] = filter_metadata

    results = vector_store.similarity_search(query, **search_kwargs)
    return results


def similarity_search_with_scores(
    vector_store: Chroma,
    query: str,
    k: int = DEFAULT_TOP_K,
) -> List[tuple]:
    """
    Similarity search that also returns relevance scores.
    Returns list of (Document, score) tuples.
    """
    results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    return results


def get_collection_stats(
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Dict[str, Any]:
    """
    Get statistics about the current ChromaDB collection.
    """
    try:
        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=_get_chroma_settings(),
        )
        collection = client.get_collection(collection_name)
        count = collection.count()

        return {
            "collection_name": collection_name,
            "total_chunks": count,
            "persist_directory": persist_directory,
        }
    except Exception:
        return {
            "collection_name": collection_name,
            "total_chunks": 0,
            "persist_directory": persist_directory,
        }


def clear_collection(
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> bool:
    """
    Clear/delete the ChromaDB collection.
    Returns True if successful.
    """
    try:
        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=_get_chroma_settings(),
        )
        client.delete_collection(collection_name)
        return True
    except Exception:
        return False


def clear_all_data(persist_directory: str = CHROMA_PERSIST_DIR) -> bool:
    """
    Delete the entire ChromaDB persist directory.
    Used when resetting the entire vector store.
    """
    try:
        # Try to delete collection first
        try:
            client = chromadb.PersistentClient(
                path=persist_directory,
                settings=_get_chroma_settings(),
            )
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass

        # Then remove the directory
        shutil.rmtree(persist_directory, ignore_errors=True)
        return True
    except Exception:
        return False
