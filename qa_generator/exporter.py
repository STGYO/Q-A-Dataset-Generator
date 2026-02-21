"""Export Q&A datasets to JSON and JSONL with customizable schemas."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema renaming
# ---------------------------------------------------------------------------


def apply_schema(
    pairs: List[Dict],
    question_key: str = "question",
    answer_key: str = "answer",
    include_score: bool = True,
    src_question_key: str = "question",
    src_answer_key: str = "answer",
) -> List[Dict]:
    """Rename question/answer fields to the user-supplied output key names.

    *question_key* and *answer_key* define the OUTPUT field names.
    *src_question_key* and *src_answer_key* define the SOURCE (internal) field
    names to read from — they default to ``"question"`` and ``"answer"``.

    Any extra metadata keys (e.g. ``chunk_index``, ``_score``) are preserved
    unless *include_score* is ``False``.
    """
    output = []
    for pair in pairs:
        record: Dict = {}
        # Map source keys to output keys
        if src_question_key in pair:
            record[question_key] = pair[src_question_key]
        if src_answer_key in pair:
            record[answer_key] = pair[src_answer_key]
        # Copy remaining metadata
        for k, v in pair.items():
            if k in (src_question_key, src_answer_key):
                continue
            if k == "_score" and not include_score:
                continue
            record[k] = v
        output.append(record)
    return output


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def export_json(
    pairs: List[Dict],
    output_path: str | Path,
    question_key: str = "question",
    answer_key: str = "answer",
    include_score: bool = False,
    indent: int = 2,
) -> Path:
    """Write the dataset as a JSON array to *output_path*.

    Returns:
        The resolved :class:`Path` of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = apply_schema(
        pairs,
        question_key=question_key,
        answer_key=answer_key,
        include_score=include_score,
    )
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)
    logger.info("Exported %d pairs as JSON → %s", len(data), output_path)
    return output_path


def export_jsonl(
    pairs: List[Dict],
    output_path: str | Path,
    question_key: str = "question",
    answer_key: str = "answer",
    include_score: bool = False,
) -> Path:
    """Write the dataset as JSONL (one JSON object per line) to *output_path*.

    Returns:
        The resolved :class:`Path` of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = apply_schema(
        pairs,
        question_key=question_key,
        answer_key=answer_key,
        include_score=include_score,
    )
    with output_path.open("w", encoding="utf-8") as fh:
        for record in data:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Exported %d pairs as JSONL → %s", len(data), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------


def export(
    pairs: List[Dict],
    output_path: str | Path,
    fmt: str = "jsonl",
    question_key: str = "question",
    answer_key: str = "answer",
    include_score: bool = False,
) -> Path:
    """Dispatch to :func:`export_json` or :func:`export_jsonl` based on *fmt*.

    Args:
        pairs: List of Q&A pair dicts (with ``question``/``answer`` keys).
        output_path: Destination file path.
        fmt: ``"json"`` or ``"jsonl"``.
        question_key: Output key for the question field.
        answer_key: Output key for the answer field.
        include_score: Whether to include the ``_score`` metadata field.

    Returns:
        Resolved path of the written file.
    """
    fmt = fmt.lower().strip(".")
    if fmt == "json":
        return export_json(
            pairs,
            output_path,
            question_key=question_key,
            answer_key=answer_key,
            include_score=include_score,
        )
    if fmt == "jsonl":
        return export_jsonl(
            pairs,
            output_path,
            question_key=question_key,
            answer_key=answer_key,
            include_score=include_score,
        )
    raise ValueError(f"Unsupported export format: '{fmt}'. Choose 'json' or 'jsonl'.")
