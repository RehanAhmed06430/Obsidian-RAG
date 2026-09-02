"""
Document Processor — Markdown parsing & Obsidian-specific features
Handles: frontmatter, tags, wikilinks, embeds, code blocks, callouts
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import streamlit as st
from langchain_core.documents import Document


@dataclass
class ParsedNote:
    """Represents a parsed Obsidian markdown note."""
    filename: str
    content: str
    title: str
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    backlinks: List[str] = field(default_factory=list)
    embeds: List[str] = field(default_factory=list)
    word_count: int = 0
    heading_count: int = 0


def parse_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    """
    Extract YAML frontmatter from markdown content.
    Returns (metadata_dict, remaining_content).
    """
    frontmatter = {}
    stripped = content.strip()

    if stripped.startswith("---"):
        # Find closing ---
        end_match = re.search(r"\n---\s*\n", stripped[3:])
        if end_match:
            fm_block = stripped[3 : 3 + end_match.start()]
            remaining = stripped[3 + end_match.end() :]

            # Simple YAML parser (avoids yaml dependency)
            for line in fm_block.strip().split("\n"):
                line = line.strip()
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip()
                    # Handle lists: tags: [a, b, c]
                    if value.startswith("[") and value.endswith("]"):
                        value = [
                            v.strip().strip('"').strip("'")
                            for v in value[1:-1].split(",")
                        ]
                    elif value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    elif value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    else:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    frontmatter[key] = value
            return frontmatter, remaining

    return frontmatter, content


def extract_tags(content: str) -> List[str]:
    """
    Extract all tags from markdown content.
    Matches #tag (but not headings like ## Title).
    """
    # Match hashtags that are word-bounded (not headings)
    tags = re.findall(r"(?<!#)#([a-zA-Z][a-zA-Z0-9_/-]*)", content)
    return list(set(tags))


def extract_wikilinks(content: str) -> List[str]:
    """
    Extract Obsidian [[wikilinks]] from content.
    Handles [[Page Name]] and [[Page Name|display text]].
    """
    links = re.findall(r"\[\[([^\]|]+?)(?:\|[^\]]+)?\]\]", content)
    return list(set(links))


def extract_embeds(content: str) -> List[str]:
    """
    Extract Obsidian embeds ![[filename]] from content.
    """
    embeds = re.findall(r"!\[\[([^\]|]+?)(?:\|[^\]]+)?\]\]", content)
    return list(set(embeds))


def extract_headings(content: str) -> List[str]:
    """Extract all markdown headings from content."""
    return re.findall(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE)


def clean_obsidian_syntax(content: str) -> str:
    """
    Clean Obsidian-specific syntax to produce plain-ish text
    suitable for embedding.
    """
    # Remove frontmatter
    content = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)
    # Remove wiki-links but keep display text: [[Page|Text]] -> Text, [[Page]] -> Page
    content = re.sub(r"\[\[([^\]|]+?)(?:\|([^\]]+))?\]\]", lambda m: m.group(2) or m.group(1), content)
    # Remove embeds: ![[file]] -> [file]
    content = re.sub(r"!\[\[([^\]]+)\]\]", r"[\1]", content)
    # Remove bold/italic markers
    content = re.sub(r"(\*{1,2}|_{1,2})(.*?)\1", r"\2", content)
    # Remove inline code backticks
    content = re.sub(r"`([^`]+)`", r"\1", content)
    # Remove callout syntax: > [!note] Title
    content = re.sub(r">\s*\[![a-zA-Z]+\]\s*", "", content)
    # Remove image syntax ![alt](url)
    content = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", content)
    # Remove link syntax but keep text: [text](url) -> text
    content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
    # Remove HTML tags
    content = re.sub(r"<[^>]+>", "", content)
    # Collapse multiple newlines
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def parse_markdown(file_content: str, filename: str) -> ParsedNote:
    """
    Parse a single markdown file into a ParsedNote object.
    """
    frontmatter, body = parse_frontmatter(content=file_content)

    # Extract metadata
    title = frontmatter.get("title", "") or filename.replace(".md", "").replace("-", " ")
    tags = list(set(
        extract_tags(body)
        + (frontmatter.get("tags", []) if isinstance(frontmatter.get("tags"), list) else [])
    ))
    backlinks = extract_wikilinks(body)
    embeds = extract_embeds(body)
    headings = extract_headings(body)
    word_count = len(body.split())
    heading_count = len(headings)

    return ParsedNote(
        filename=filename,
        content=body,
        title=title,
        frontmatter=frontmatter,
        tags=tags,
        backlinks=backlinks,
        embeds=embeds,
        word_count=word_count,
        heading_count=heading_count,
    )


def process_vault(uploaded_files) -> List[Document]:
    """
    Process all uploaded markdown files into LangChain Document objects.
    Returns a list of Document with metadata.
    """
    documents = []

    for uploaded_file in uploaded_files:
        try:
            content = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            st.warning(f"⚠️ Could not decode {uploaded_file.name}, skipping.")
            continue

        filename = uploaded_file.name
        parsed = parse_markdown(content, filename)

        # Create clean text for embedding
        clean_text = clean_obsidian_syntax(parsed.content)

        if not clean_text.strip():
            continue

        metadata = {
            "source": filename,
            "title": parsed.title,
            "tags": ", ".join(parsed.tags) if parsed.tags else "",
            "backlinks": ", ".join(parsed.backlinks) if parsed.backlinks else "",
            "word_count": parsed.word_count,
            "heading_count": parsed.heading_count,
        }

        # Add frontmatter fields to metadata
        for key, value in parsed.frontmatter.items():
            if key not in metadata and isinstance(value, (str, int, float, bool)):
                metadata[f"fm_{key}"] = str(value)

        doc = Document(
            page_content=clean_text,
            metadata=metadata,
        )
        documents.append(doc)

    return documents


def get_vault_stats(documents: List[Document]) -> Dict[str, Any]:
    """Compute statistics for a processed vault."""
    all_tags = set()
    total_words = 0
    files = set()

    for doc in documents:
        files.add(doc.metadata.get("source", "unknown"))
        total_words += doc.metadata.get("word_count", 0)
        tags_str = doc.metadata.get("tags", "")
        if tags_str:
            all_tags.update(t.strip() for t in tags_str.split(",") if t.strip())

    return {
        "total_documents": len(documents),
        "total_files": len(files),
        "total_words": total_words,
        "unique_tags": len(all_tags),
        "tags": list(all_tags),
        "files": list(files),
    }
