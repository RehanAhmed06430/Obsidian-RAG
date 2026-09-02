"""
Tests for the Chunker module
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from src.chunker import chunk_documents, get_chunk_stats, create_text_splitter
from config.settings import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def test_chunk_documents_basic():
    """Test basic document chunking."""
    doc = Document(
        page_content="This is a test document. " * 50,
        metadata={"source": "test.md", "title": "Test"},
    )
    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1, "Should split into multiple chunks"
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.md"
        assert "chunk_index" in chunk.metadata


def test_chunk_documents_preserves_metadata():
    """Test that metadata is preserved in chunks."""
    doc = Document(
        page_content="Word " * 200,
        metadata={"source": "notes.md", "tags": "python,ml"},
    )
    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)
    for chunk in chunks:
        assert chunk.metadata["source"] == "notes.md"
        assert chunk.metadata["tags"] == "python,ml"
        assert "total_chunks" in chunk.metadata


def test_chunk_documents_empty():
    """Test chunking empty content."""
    doc = Document(page_content="", metadata={"source": "empty.md"})
    chunks = chunk_documents([doc])
    assert isinstance(chunks, list)


def test_chunk_stats():
    """Test chunk statistics calculation."""
    docs = [
        Document(page_content="Short", metadata={"source": "a.md"}),
        Document(page_content="A bit longer content here", metadata={"source": "b.md"}),
    ]
    chunks = chunk_documents(docs, chunk_size=10, chunk_overlap=2)
    stats = get_chunk_stats(chunks)
    assert stats["total_chunks"] > 0
    assert stats["min_length"] > 0
    assert stats["max_length"] >= stats["min_length"]


def test_create_text_splitter():
    """Test that text splitter is created correctly."""
    splitter = create_text_splitter(chunk_size=300, chunk_overlap=30)
    assert splitter is not None
    text = "This is a test. " * 100
    chunks = splitter.split_text(text)
    assert len(chunks) > 1


if __name__ == "__main__":
    test_chunk_documents_basic()
    test_chunk_documents_preserves_metadata()
    test_chunk_documents_empty()
    test_chunk_stats()
    test_create_text_splitter()
    print("ALL CHUNKER TESTS PASSED")
