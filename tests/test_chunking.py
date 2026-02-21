"""Tests for text chunking module."""

from __future__ import annotations

import pytest


def test_chunk_short_text():
    from qa_generator.chunking import chunk_text

    text = "Short text."
    chunks = chunk_text(text, chunk_size=1000, use_langchain=False)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."


def test_chunk_empty_text():
    from qa_generator.chunking import chunk_text

    chunks = chunk_text("", chunk_size=1000, use_langchain=False)
    assert chunks == []


def test_chunk_whitespace_only():
    from qa_generator.chunking import chunk_text

    chunks = chunk_text("   \n\n  ", chunk_size=1000, use_langchain=False)
    assert chunks == []


def test_chunk_long_text_produces_multiple_chunks():
    from qa_generator.chunking import chunk_text

    text = ("This is a sentence. " * 200)  # ~4000 chars
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50, use_langchain=False)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 600  # allow some leeway for separator joins


def test_chunk_no_empty_chunks():
    from qa_generator.chunking import chunk_text

    text = "\n".join(f"Paragraph {i}: " + "word " * 50 for i in range(20))
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=50, use_langchain=False)
    for chunk in chunks:
        assert chunk.strip() != ""
