"""
Chunker — Text splitting with markdown-aware strategies
Uses LangChain's RecursiveCharacterTextSplitter with markdown separators.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def create_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Create a markdown-aware text splitter.
    Tries to split at natural markdown boundaries first.
    """
    # Markdown-aware separators (in priority order)
    separators = [
        "\n\n\n",      # triple newline (section breaks)
        "\n## ",       # h2 heading
        "\n### ",      # h3 heading
        "\n#### ",     # h4 heading
        "\n\n",        # paragraph break
        "\n",          # line break
        ". ",          # sentence end
        "! ",          # sentence end (exclamation)
        "? ",          # sentence end (question)
        " ",           # word boundary
        "",            # character
    ]

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
        keep_separator=True,
    )


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into overlapping chunks while preserving metadata.

    Each chunk inherits the metadata from its parent document
    and gains a chunk_index field.
    """
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunked_docs = []

    for doc in documents:
        chunks = splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            chunk_metadata = doc.metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)

            chunked_doc = Document(
                page_content=chunk,
                metadata=chunk_metadata,
            )
            chunked_docs.append(chunked_doc)

    return chunked_docs


def get_chunk_stats(chunks: List[Document]) -> dict:
    """Return statistics about the generated chunks."""
    if not chunks:
        return {"total_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}

    lengths = [len(c.page_content) for c in chunks]
    sources = set(c.metadata.get("source", "unknown") for c in chunks)

    return {
        "total_chunks": len(chunks),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "unique_sources": len(sources),
    }
