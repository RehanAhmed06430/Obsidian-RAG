"""
Tests for RAG pipeline components
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processor import (
    parse_frontmatter,
    extract_tags,
    extract_wikilinks,
    extract_embeds,
    clean_obsidian_syntax,
    parse_markdown,
)
from langchain_core.documents import Document


def test_parse_frontmatter():
    """Test YAML frontmatter parsing."""
    content = """---
title: Test Note
tags: [python, ml]
created: 2026-08-01
---

# Content here"""

    fm, remaining = parse_frontmatter(content)
    assert fm["title"] == "Test Note"
    assert "python" in fm["tags"]
    assert "Content here" in remaining


def test_parse_frontmatter_empty():
    """Test with no frontmatter."""
    content = "# Just a heading\n\nSome content."
    fm, remaining = parse_frontmatter(content)
    assert fm == {}
    assert remaining == content


def test_extract_tags():
    """Test tag extraction."""
    content = "# Heading\n\nThis uses #machine-learning and #python tags."
    tags = extract_tags(content)
    assert "machine-learning" in tags
    assert "python" in tags


def test_extract_wikilinks():
    """Test wikilink extraction."""
    content = "See [[Neural Networks]] and [[Machine Learning|ML notes]]."
    links = extract_wikilinks(content)
    assert "Neural Networks" in links
    assert "Machine Learning" in links


def test_extract_embeds():
    """Test embed extraction."""
    content = "Here is ![[image.png]] and ![[diagram.svg|Diagram]]."
    embeds = extract_embeds(content)
    assert "image.png" in embeds
    assert "diagram.svg" in embeds


def test_clean_obsidian_syntax():
    """Test Obsidian syntax cleaning."""
    content = "# Title\n\n**Bold** and [[link]] and ![[embed]]\n\n> [!note] Callout"
    cleaned = clean_obsidian_syntax(content)
    assert "[[link]]" not in cleaned
    assert "![[embed]]" not in cleaned
    assert "Bold" in cleaned
    assert "link" in cleaned


def test_parse_markdown():
    """Test full markdown parsing."""
    content = """---
title: ML Notes
tags: [AI]
---

# Machine Learning

This is about [[Neural Networks]] and uses #deep-learning.

Some more content here.
"""
    parsed = parse_markdown(content, "ml_notes.md")
    assert parsed.title == "ML Notes"
    assert "deep-learning" in parsed.tags
    assert "Neural Networks" in parsed.backlinks
    assert parsed.word_count > 0


def test_document_creation():
    """Test that parsed notes can become LangChain Documents."""
    content = """---
title: Test
tags: [test]
---

# Test Note

Content here with #tag and [[link]].
"""
    parsed = parse_markdown(content, "test.md")
    doc = Document(
        page_content=clean_obsidian_syntax(parsed.content),
        metadata={
            "source": parsed.filename,
            "title": parsed.title,
            "tags": ", ".join(parsed.tags),
        },
    )
    assert doc.page_content
    assert doc.metadata["source"] == "test.md"


if __name__ == "__main__":
    test_parse_frontmatter()
    test_parse_frontmatter_empty()
    test_extract_tags()
    test_extract_wikilinks()
    test_extract_embeds()
    test_clean_obsidian_syntax()
    test_parse_markdown()
    test_document_creation()
    print("ALL RAG TESTS PASSED")
