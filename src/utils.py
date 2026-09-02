"""
Utility Functions — Helper utilities for the project
"""

import os
import io
import zipfile
import tempfile
from typing import List, Tuple

from config.settings import SUPPORTED_EXTENSIONS


def is_valid_markdown(filename: str) -> bool:
    """Check if a file has a supported markdown extension."""
    _, ext = os.path.splitext(filename)
    return ext.lower() in SUPPORTED_EXTENSIONS


def filter_markdown_files(files: list) -> list:
    """Filter a list of files to only include markdown files."""
    return [f for f in files if is_valid_markdown(f.name)]


class ZipMarkdownFile:
    """File-like wrapper for markdown files extracted from a ZIP archive.
    Mimics Streamlit's UploadedFile so process_vault() can call .read() and .name.
    """
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def read(self) -> bytes:
        return self._content


def extract_zip_to_files(uploaded_zip) -> List[ZipMarkdownFile]:
    """
    Extract a zip file and return a list of ZipMarkdownFile objects
    for all markdown files inside.
    """
    md_files = []

    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_zip.read()), "r") as z:
            for file_info in z.infolist():
                if file_info.is_dir():
                    continue

                filename = os.path.basename(file_info.filename)

                # Get the relative path within the vault
                dir_path = os.path.dirname(file_info.filename)
                if dir_path:
                    # Include subdirectory in the filename
                    filename = f"{dir_path}/{filename}"

                if is_valid_markdown(filename):
                    content = z.read(file_info.filename)
                    display_name = os.path.basename(file_info.filename)
                    md_files.append(ZipMarkdownFile(display_name, content))
    except zipfile.BadZipFile:
        return []

    return md_files


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length, adding ellipsis if needed."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + "..."


def format_number(n: int) -> str:
    """Format a number with K/M suffix for large numbers."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def extract_response_text(response) -> str:
    """
    Safely extract clean text from a LangChain chat response object.

    WHY THIS IS NEEDED:
    Gemini 1.5 / 2.x models return `response.content` as a plain string.
    Gemini 3.x models return `response.content` as a list of dicts
    (e.g. [{'type': 'text', 'text': '...', 'extras': {'signature': '...'}}])
    because of "thought signatures" used for multi-turn reasoning.
    This function handles both shapes so the rest of the codebase
    never needs to care which model is configured.

    Args:
        response: A LangChain AIMessage or similar object returned by
                  ChatGoogleGenerativeAI.invoke().

    Returns:
        A plain string containing the assistant's text response,
        suitable for rendering in the UI.
    """
    import logging
    logger = logging.getLogger(__name__)

    # 1. Try LangChain's built-in .text accessor (works for most models)
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # 2. Inspect .content directly
    content = getattr(response, "content", None)

    # 2a. Plain string — Gemini 1.5 / 2.x style
    if isinstance(content, str):
        return content.strip()

    # 2b. List of dicts — Gemini 3.x style with thought signatures
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
        # Return joined parts even if empty (the model chose to say nothing)
        return "\n".join(parts)

    # 3. Fallback — never crash, just return whatever we can
    logger.warning(
        "extract_response_text: unexpected response shape (type=%s). "
        "Falling back to str().", type(content)
    )
    return str(content) if content is not None else ""
