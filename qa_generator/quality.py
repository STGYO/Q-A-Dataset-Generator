"""Quality scoring and filtering for generated Q&A pairs."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_QUESTION_LEN = 10
_MIN_ANSWER_LEN = 5
_MAX_ANSWER_LEN = 2000


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _is_non_empty(pair: Dict[str, str], q_key: str, a_key: str) -> bool:
    """Both question and answer must have content."""
    return bool(pair.get(q_key, "").strip()) and bool(pair.get(a_key, "").strip())


def _has_min_length(pair: Dict[str, str], q_key: str, a_key: str) -> bool:
    return (
        len(pair.get(q_key, "")) >= _MIN_QUESTION_LEN
        and len(pair.get(a_key, "")) >= _MIN_ANSWER_LEN
    )


def _answer_not_too_long(pair: Dict[str, str], a_key: str) -> bool:
    return len(pair.get(a_key, "")) <= _MAX_ANSWER_LEN


def _question_ends_with_mark(pair: Dict[str, str], q_key: str) -> bool:
    """Heuristic: a question should end with '?' or at least contain interrogative."""
    q = pair.get(q_key, "").strip()
    interrogatives = re.compile(
        r"\b(what|who|where|when|why|how|which|is|are|does|do|can|could|should|would)\b",
        re.IGNORECASE,
    )
    return q.endswith("?") or bool(interrogatives.search(q))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_pair(
    pair: Dict[str, str],
    q_key: str = "question",
    a_key: str = "answer",
) -> float:
    """Compute a quality score in [0, 1] for a single Q&A pair.

    Returns 0.0 immediately when question or answer is empty.
    Otherwise scores on three additional criteria (each contributes equally):
    1. Minimum length thresholds.
    2. Answer within maximum length.
    3. Question looks like a real question (interrogative word or "?").
    """
    if not _is_non_empty(pair, q_key, a_key):
        return 0.0

    checks = [
        _has_min_length(pair, q_key, a_key),
        _answer_not_too_long(pair, a_key),
        _question_ends_with_mark(pair, q_key),
    ]
    return sum(checks) / len(checks)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def deduplicate(
    pairs: List[Dict[str, str]],
    q_key: str = "question",
    a_key: str = "answer",
) -> List[Dict[str, str]]:
    """Remove duplicate Q&A pairs (case-insensitive, whitespace-normalized)."""
    seen: set = set()
    unique: List[Dict[str, str]] = []
    for pair in pairs:
        key = (
            _normalize(pair.get(q_key, "")),
            _normalize(pair.get(a_key, "")),
        )
        if key not in seen:
            seen.add(key)
            unique.append(pair)

    removed = len(pairs) - len(unique)
    if removed:
        logger.info("Deduplication removed %d duplicate pairs", removed)
    return unique


# ---------------------------------------------------------------------------
# Filter pipeline
# ---------------------------------------------------------------------------


def filter_pairs(
    pairs: List[Dict[str, str]],
    q_key: str = "question",
    a_key: str = "answer",
    min_score: float = 0.5,
    deduplicate_pairs: bool = True,
) -> List[Dict[str, str]]:
    """Filter and deduplicate Q&A pairs.

    Args:
        pairs: Raw list of ``{question, answer}`` dicts (or custom-keyed).
        q_key: Key used for the question field.
        a_key: Key used for the answer field.
        min_score: Minimum quality score (0–1) to keep a pair.
        deduplicate_pairs: Whether to deduplicate after scoring.

    Returns:
        Filtered list with an added ``"_score"`` field.
    """
    scored: List[Dict[str, str]] = []
    for pair in pairs:
        s = score_pair(pair, q_key=q_key, a_key=a_key)
        if s >= min_score:
            pair = dict(pair)  # shallow copy
            pair["_score"] = round(s, 4)
            scored.append(pair)

    if deduplicate_pairs:
        scored = deduplicate(scored, q_key=q_key, a_key=a_key)

    logger.info(
        "Quality filter: kept %d / %d pairs (min_score=%.2f)",
        len(scored),
        len(pairs),
        min_score,
    )
    return scored
