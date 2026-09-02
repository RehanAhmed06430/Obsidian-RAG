"""
Tests for src.utils — especially extract_response_text()
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import extract_response_text


# ---------------------------------------------------------------------------
# Mock response objects that mimic LangChain AIMessage shapes
# ---------------------------------------------------------------------------

class MockResponse:
    """Minimal mock that mimics an AIMessage."""
    def __init__(self, content=None, text=None):
        self.content = content
        if text is not None:
            self.text = text


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_plain_string_content():
    """Gemini 1.5 / 2.x style: .content is a plain string."""
    resp = MockResponse(content="Hello, world!")
    assert extract_response_text(resp) == "Hello, world!"


def test_list_of_dicts_content():
    """Gemini 3.x style: .content is a list of dicts with 'type' == 'text'."""
    resp = MockResponse(content=[
        {"type": "text", "text": "Part one."},
        {"type": "text", "text": "Part two."},
    ])
    result = extract_response_text(resp)
    assert "Part one." in result
    assert "Part two." in result


def test_list_with_extras_and_signature():
    """Gemini 3.x style: list includes extras/signature blocks that must be skipped."""
    resp = MockResponse(content=[
        {"type": "text", "text": "Real answer here."},
        {"type": "text", "text": "", "extras": {"signature": "abc123"}},
        {"type": "function_call", "name": "some_tool", "args": {}},
    ])
    result = extract_response_text(resp)
    assert result == "Real answer here."
    assert "abc123" not in result
    assert "function_call" not in result


def test_list_with_thinking_blocks():
    """List may contain thinking blocks that should be skipped."""
    resp = MockResponse(content=[
        {"type": "thinking", "text": "Internal reasoning..."},
        {"type": "text", "text": "The actual response."},
    ])
    result = extract_response_text(resp)
    assert result == "The actual response."
    assert "Internal reasoning" not in result


def test_text_attribute_preferred():
    """If .text exists and is non-empty, prefer it over .content."""
    resp = MockResponse(content="stale content", text="fresh text")
    assert extract_response_text(resp) == "fresh text"


def test_empty_content_returns_empty():
    """Empty string content returns empty string."""
    resp = MockResponse(content="")
    assert extract_response_text(resp) == ""


def test_none_content_returns_empty():
    """None content returns empty string, not None."""
    resp = MockResponse(content=None)
    assert extract_response_text(resp) == ""


def test_list_with_all_empty_texts():
    """List where every text block is empty/whitespace returns empty."""
    resp = MockResponse(content=[
        {"type": "text", "text": ""},
        {"type": "text", "text": "   "},
    ])
    assert extract_response_text(resp) == ""


def test_integer_content_fallback():
    """Unexpected content type falls back to str() without crashing."""
    resp = MockResponse(content=42)
    result = extract_response_text(resp)
    assert result == "42"


def test_whitespace_stripped():
    """Leading/trailing whitespace is stripped."""
    resp = MockResponse(content="  Hello  ")
    assert extract_response_text(resp) == "Hello"


def test_realistic_gemini3_response():
    """Full realistic Gemini 3.x response with signature."""
    resp = MockResponse(content=[
        {
            "type": "text",
            "text": (
                "Supervised learning is a type of machine learning that "
                "relies on labeled training data to help a system learn."
            ),
        },
        {
            "type": "text",
            "text": "",
            "extras": {
                "signature": "EpMNCpANARFNMg/xHSerXj24QcTE..."
            },
        },
    ])
    result = extract_response_text(resp)
    assert "Supervised learning" in result
    assert "EpMNCpANARFNMg" not in result
    assert len(result) > 20


if __name__ == "__main__":
    test_plain_string_content()
    test_list_of_dicts_content()
    test_list_with_extras_and_signature()
    test_list_with_thinking_blocks()
    test_text_attribute_preferred()
    test_empty_content_returns_empty()
    test_none_content_returns_empty()
    test_list_with_all_empty_texts()
    test_integer_content_fallback()
    test_whitespace_stripped()
    test_realistic_gemini3_response()
    print("ALL UTILS TESTS PASSED")
