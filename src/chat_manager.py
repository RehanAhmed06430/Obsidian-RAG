"""
Chat Manager — Conversation history management
Handles multi-turn conversation with source citations.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import streamlit as st

from config.settings import MAX_HISTORY_LENGTH


@dataclass
class ChatMessage:
    """Represents a single message in the conversation."""
    role: str           # "user" or "assistant"
    content: str
    sources: List[Dict[str, str]] = field(default_factory=list)
    timestamp: Optional[str] = None


class ChatManager:
    """
    Manages conversation history in Streamlit session state.
    Provides methods to add, retrieve, and clear messages.
    """

    SESSION_KEY = "chat_history"

    @classmethod
    def _get_history(cls) -> List[Dict[str, Any]]:
        """Get the conversation history from session state."""
        if cls.SESSION_KEY not in st.session_state:
            st.session_state[cls.SESSION_KEY] = []
        return st.session_state[cls.SESSION_KEY]

    @classmethod
    def add_message(
        cls,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Add a message to the conversation history."""
        history = cls._get_history()

        message = {
            "role": role,
            "content": content,
            "sources": sources or [],
        }

        history.append(message)

        # Trim history to max length (each "exchange" = user + assistant = 2 messages)
        max_messages = MAX_HISTORY_LENGTH * 2
        if len(history) > max_messages:
            st.session_state[cls.SESSION_KEY] = history[-max_messages:]

    @classmethod
    def get_history(cls) -> List[Dict[str, Any]]:
        """Get the full conversation history."""
        return cls._get_history()

    @classmethod
    def get_history_for_llm(cls) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for the LLM.
        Returns list of {role, content} dicts.
        """
        history = cls._get_history()
        return [{"role": msg["role"], "content": msg["content"]} for msg in history]

    @classmethod
    def clear_history(cls) -> None:
        """Clear the entire conversation history."""
        st.session_state[cls.SESSION_KEY] = []

    @classmethod
    def get_message_count(cls) -> int:
        """Return the total number of messages in history."""
        return len(cls._get_history())

    @classmethod
    def get_exchange_count(cls) -> int:
        """Return the number of complete exchanges (user+assistant pairs)."""
        return cls.get_message_count() // 2

    @classmethod
    def format_sources_for_display(cls, sources: List) -> str:
        """Format source documents for display in the UI."""
        if not sources:
            return ""

        lines = ["**📚 Sources:**\n"]
        seen_sources = set()

        for i, doc in enumerate(sources, 1):
            if hasattr(doc, "metadata"):
                source = doc.metadata.get("source", "Unknown")
                title = doc.metadata.get("title", source)
                tags = doc.metadata.get("tags", "")
            else:
                source = doc.get("source", "Unknown")
                title = doc.get("title", source)
                tags = doc.get("tags", "")

            source_key = f"{source}:{title}"
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)

            line = f"**[{i}] {title}** (`{source}`)"
            if tags:
                tag_list = [t.strip() for t in tags.split(",") if t.strip()]
                if tag_list:
                    line += f" — Tags: {', '.join(tag_list[:5])}"
            lines.append(line)

        return "\n".join(lines)
