"""Intelligent text chunking with overlapping windows."""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 200
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_text(
    text: str,
    separators: List[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Recursive character text splitter (pure-Python fallback)."""
    if not text:
        return []

    separator = separators[0]
    next_separators = separators[1:]

    splits = re.split(re.escape(separator), text) if separator else list(text)

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for split in splits:
        split_len = len(split)

        if current_len + split_len + (len(separator) if current else 0) > chunk_size:
            if current:
                chunk = separator.join(current)
                if len(chunk) > 0:
                    # Try to further split if still too big and we have more seps
                    if len(chunk) > chunk_size and next_separators:
                        chunks.extend(
                            _split_text(chunk, next_separators, chunk_size, chunk_overlap)
                        )
                    else:
                        chunks.append(chunk)
                # Keep overlap
                while current and sum(len(s) for s in current) + len(separator) * (len(current) - 1) > chunk_overlap:
                    current.pop(0)
                current_len = sum(len(s) for s in current) + len(separator) * max(0, len(current) - 1)

        current.append(split)
        current_len += split_len + (len(separator) if len(current) > 1 else 0)

    if current:
        chunk = separator.join(current)
        if chunk:
            if len(chunk) > chunk_size and next_separators:
                chunks.extend(
                    _split_text(chunk, next_separators, chunk_size, chunk_overlap)
                )
            else:
                chunks.append(chunk)

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    use_langchain: bool = True,
) -> List[str]:
    """Split *text* into overlapping chunks of at most *chunk_size* characters.

    Args:
        text: Source text to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
        use_langchain: Prefer LangChain's ``RecursiveCharacterTextSplitter`` when
            available.  Falls back to the built-in splitter otherwise.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    if use_langchain:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = splitter.split_text(text)
            logger.info(
                "Chunked text into %d chunks (LangChain, size=%d, overlap=%d)",
                len(chunks),
                chunk_size,
                chunk_overlap,
            )
            return chunks
        except ImportError:
            pass  # fall through to built-in splitter

    chunks = _split_text(text, _DEFAULT_SEPARATORS, chunk_size, chunk_overlap)
    # Filter empty / whitespace-only chunks
    chunks = [c for c in chunks if c.strip()]
    logger.info(
        "Chunked text into %d chunks (built-in, size=%d, overlap=%d)",
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
